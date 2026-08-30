"""C0/C1: CMA-ES over frozen VNC coupling and E4 feedback gains."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import cma
import mujoco
import numpy as np
import torch


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE.parent))
from fetch_manc import load_annotations  # noqa: E402
from hexapod_gait import XML, home_stance, pd_torque  # noqa: E402
from probe_structured_descending import DESC_KEYS, brain_type_indices, phase_metrics  # noqa: E402
from run_hexapod_chase import retina  # noqa: E402
from run_sensory_feedback import ANGLE_LIMIT_RAD, LEG_NAMES, PROPRIO_SUBCLASSES, foot_loads  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


# Frozen protocol. None of these values is a CMA-ES gene.
VNC_TAU = 0.02
VNC_DT = 0.005
VNC_SUBSTEPS = 4
VNC_GW = 0.03
VNC_THRESHOLD = 7.5
VNC_ACTIVATION_GAIN = 1.0
VNC_FR_CAP = 200.0
VNC_HETEROGENEITY = True
VNC_SEED = 1
CTRL_HZ = 50.0
SETTLE_S = 0.7
KP = 3.0
KD = 0.05
MAP_GAINS = (-0.55, 0.65, 0.65)
STANCE = home_stance()
ANGLE_NORMALIZATION_RAD = ANGLE_LIMIT_RAD
ROLLOUT_SECS = 4.0
EVALUATION_SECS = 8.0
FALL_CONST = 0.5
H_MIN = 0.02
EVALUATION_SEEDS = (20260829, 20260830, 20260831, 20260832)
OPTIMIZER_SEED = 20260829
POPSIZE = 14
MAX_GENERATIONS = 200
MAX_HOURS = 2.0


def _gene(index: int, name: str, default: float, lower: float, upper: float, injection: str) -> dict:
    return {
        "index": index,
        "name": name,
        "default": default,
        "bounds": [lower, upper],
        "injection_point": injection,
    }


def gene_spec() -> list[dict]:
    genes = [
        _gene(0, "dng100_level", 100.0, 0.0, 300.0, "DNg100 left/right descending current"),
        _gene(1, "dnb08_level", 100.0, 0.0, 300.0, "DNb08 left/right descending current"),
        _gene(2, "left_right_balance", 0.0, -1.0, 1.0, "multiplicative left/right descending balance"),
        _gene(3, "tonic_offset", 0.0, -50.0, 50.0, "uniform current added to every VNC neuron"),
    ]
    pathways = (
        ("T1", "T2"), ("T1", "T3"), ("T2", "T1"),
        ("T2", "T3"), ("T3", "T1"), ("T3", "T2"),
    )
    for row, source in pathways:
        index = len(genes)
        genes.append(_gene(index, f"{row}_from_{source}", 1.0, 0.0, 3.0, f"all CSR edges source {source} -> target {row}"))
    index = len(genes)
    for subclass in PROPRIO_SUBCLASSES:
        label = subclass.replace(" ", "_")
        genes.append(_gene(index, f"{label}_gain", 100.0, 0.0, 300.0, f"all E4 {subclass} sensory currents"))
        index += 1
        genes.append(_gene(index, f"{label}_sign", 1.0, -1.0, 1.0, f"sign of all E4 {subclass} sensory currents"))
        index += 1
    assert len(genes) == 18
    return genes


GENES = gene_spec()
DEFAULT_GENE = np.asarray([item["default"] for item in GENES], dtype=np.float64)
LOWER_BOUNDS = np.asarray([item["bounds"][0] for item in GENES], dtype=np.float64)
UPPER_BOUNDS = np.asarray([item["bounds"][1] for item in GENES], dtype=np.float64)
INITIAL_STDS = 0.3 * (UPPER_BOUNDS - LOWER_BOUNDS)


def csr_hash(weight: torch.Tensor, values: bool = True) -> str:
    weight = weight.to_sparse_csr().cpu()
    digest = hashlib.sha256()
    for array in (weight.crow_indices().numpy(), weight.col_indices().numpy()):
        digest.update(array.tobytes())
    if values:
        digest.update(weight.values().numpy().tobytes())
    return digest.hexdigest()


def _copy_csr(weight: torch.Tensor, values: torch.Tensor | None = None) -> torch.Tensor:
    weight = weight.to_sparse_csr().cpu()
    return torch.sparse_csr_tensor(
        weight.crow_indices().clone(),
        weight.col_indices().clone(),
        (weight.values() if values is None else values).clone(),
        size=weight.shape,
    )


def _condition_weight(data: Path, weights_name: str, index_name: str, condition: str) -> tuple[torch.Tensor, str]:
    vnc = VNCStepper(
        data,
        substeps=VNC_SUBSTEPS,
        tau=VNC_TAU,
        dt=VNC_DT,
        gW=VNC_GW,
        weights_name=weights_name,
        index_name=index_name,
        model="paper",
        threshold=VNC_THRESHOLD,
        activation_gain=VNC_ACTIVATION_GAIN,
        fr_cap=VNC_FR_CAP,
        heterogeneity=VNC_HETEROGENEITY,
        seed=VNC_SEED,
    )
    if condition == "connectome":
        return _copy_csr(vnc.W), csr_hash(vnc.W)
    from reservoir_3way import degree_preserving_rewire, randomize_weights
    if condition == "random":
        randomized = randomize_weights(vnc.W)
        return _copy_csr(randomized), csr_hash(randomized)
    if condition == "rewired":
        rewired = degree_preserving_rewire(vnc.W)
        return _copy_csr(rewired), csr_hash(rewired)
    raise ValueError(f"unknown condition {condition!r}")


class RolloutContext:
    """Worker-local frozen graphs, brain, body model, and injection masks."""

    def __init__(
        self,
        vnc_data: Path = RESULTS,
        brain_data: Path = DATA,
        condition: str = "connectome",
        weights_name: str = "e4_sensory_weight_csr.pkl",
        index_name: str = "e4_sensory_io_index.npz",
    ):
        self.vnc_data = Path(vnc_data)
        self.brain_data = Path(brain_data)
        self.condition = condition
        self.weights_name = weights_name
        self.index_name = index_name
        self.base_weight, self.base_weight_hash = _condition_weight(self.vnc_data, weights_name, index_name, condition)
        self.base_pattern_hash = csr_hash(self.base_weight, values=False)
        self.vnc = VNCStepper(
            self.vnc_data,
            substeps=VNC_SUBSTEPS,
            tau=VNC_TAU,
            dt=VNC_DT,
            gW=VNC_GW,
            weights_name=weights_name,
            index_name=index_name,
            model="paper",
            threshold=VNC_THRESHOLD,
            activation_gain=VNC_ACTIVATION_GAIN,
            fr_cap=VNC_FR_CAP,
            heterogeneity=VNC_HETEROGENEITY,
            seed=VNC_SEED,
        )
        self.vnc.W = _copy_csr(self.base_weight).to(self.vnc.device)
        self.io = self.vnc.io
        self.n = self.vnc.n
        self.edge_rows = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.base_weight.crow_indices().numpy()))
        self.node_neuromeres = self._node_neuromeres()
        self.path_masks = self._path_masks()
        self.sensory_rows = self.io["sensory_feedback"].astype(np.int64)
        self.sensory_classes = self.io["sensory_feedback_subclasses"].astype("U32")
        self.sensory_class_masks = {name: self.sensory_classes == name for name in PROPRIO_SUBCLASSES}

        with (self.brain_data / "weight_csr.pkl").open("rb") as stream:
            self.brain_weight = pickle.load(stream)
        if self.brain_weight.layout != torch.sparse_csr:
            self.brain_weight = self.brain_weight.to_sparse_csr()
        self.brain_weight = self.brain_weight.float()
        brain_io = np.load(self.brain_data / "io_index.npz")
        self.visual_left = torch.from_numpy(brain_io["vis_left"]).long()
        self.visual_right = torch.from_numpy(brain_io["vis_right"]).long()
        self.descending_indices = brain_type_indices(self.brain_data)

        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in range(6)]
        self.mass_per_leg = float(self.model.body_mass.sum() * 9.81 / 6.0)
        self.physics_steps = max(1, int(round((1.0 / CTRL_HZ) / self.model.opt.timestep)))

    def _node_neuromeres(self) -> np.ndarray:
        annotation_path = self.brain_data / "manc" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        annotations = load_annotations(annotation_path).set_index("bodyId")
        body_ids = self.io["node_body_ids"].astype(np.int64)
        return annotations.reindex(body_ids)["somaNeuromere"].fillna("NA").astype(str).to_numpy()

    def _path_masks(self) -> dict[tuple[str, str], np.ndarray]:
        rows = self.node_neuromeres[self.edge_rows]
        cols = self.node_neuromeres[self.base_weight.col_indices().numpy()]
        pathways = {}
        for row, source in (("T1", "T2"), ("T1", "T3"), ("T2", "T1"), ("T2", "T3"), ("T3", "T1"), ("T3", "T2")):
            pathways[(row, source)] = (rows == row) & (cols == source)
        return pathways

    def apply_gene(self, gene: np.ndarray) -> None:
        gene = np.asarray(gene, dtype=np.float64)
        if gene.shape != (len(GENES),):
            raise ValueError(f"gene has shape {gene.shape}; expected {(len(GENES),)}")
        values = self.base_weight.values().clone()
        for offset, pathway in enumerate(self.path_masks, start=4):
            mask = torch.from_numpy(self.path_masks[pathway])
            values[mask] *= float(gene[offset])
        self.vnc.W = torch.sparse_csr_tensor(
            self.base_weight.crow_indices().clone(),
            self.base_weight.col_indices().clone(),
            values,
            size=self.base_weight.shape,
        ).to(self.vnc.device)

    def current(self, gene: np.ndarray, outputs: dict[str, float], loads: np.ndarray, angles: np.ndarray) -> np.ndarray:
        current = np.zeros(self.n, dtype=np.float32)
        balance = float(np.clip(gene[2], -1.0, 1.0))
        for key in DESC_KEYS:
            level = float(gene[0] if key.startswith("dng100") else gene[1])
            side_factor = 1.0 - balance if key.endswith("left") else 1.0 + balance
            current[self.io[key]] = level * side_factor * outputs[key]
        class_offset = 10
        for class_index, subclass in enumerate(PROPRIO_SUBCLASSES):
            gain = float(gene[class_offset + class_index * 2])
            sign = float(gene[class_offset + class_index * 2 + 1])
            current[self.sensory_rows[self.sensory_class_masks[subclass]]] = sign * gain * (
                loads[self._sensory_leg_indices(subclass)] if subclass == "campaniform sensilla" else angles[self._sensory_leg_indices(subclass)]
            )
        current += float(gene[3])
        return current

    def _sensory_leg_indices(self, subclass: str) -> np.ndarray:
        names = self.io["sensory_feedback_legs"].astype("U3") + self.io["sensory_feedback_sides"].astype("U1")
        leg_index = {name: index for index, name in enumerate(LEG_NAMES)}
        return np.asarray([leg_index[name] for name in names[self.sensory_class_masks[subclass]]], dtype=np.int64)


def _brain_step(context: RolloutContext, brain_state: torch.Tensor, target_xy: tuple[float, float], sim) -> tuple[dict[str, float], float, float]:
    trunk = (float(sim.qpos[0]), float(sim.qpos[1]))
    yaw = 2.0 * np.arctan2(sim.qpos[6], sim.qpos[3])
    present, azimuth, _, size = retina(trunk, yaw, target_xy)
    magnitude = 1.2 * (0.5 + min(1.0, size))
    visual_current = torch.zeros(len(brain_state), dtype=torch.float32)
    if present:
        visual_current[context.visual_left] = magnitude * (max(0.0, azimuth) + 0.15)
        visual_current[context.visual_right] = magnitude * (max(0.0, -azimuth) + 0.15)
    with torch.no_grad():
        for _ in range(12):
            rates = torch.relu(torch.tanh(brain_state))
            brain_state.mul_(0.75).add_(0.25 * (0.006 * (context.brain_weight @ rates) + visual_current))
            brain_state.clamp_(-5.0, 5.0)
        rates = torch.relu(torch.tanh(brain_state))
    outputs = {key: float(rates[indices].mean()) for key, indices in context.descending_indices.items()}
    return outputs, float(azimuth), float(size)


def rollout(
    gene: np.ndarray,
    seed: int,
    context: RolloutContext | None = None,
    secs: float = ROLLOUT_SECS,
    target_xy: tuple[float, float] | None = None,
    initial_yaw: float | None = None,
    video: Path | None = None,
) -> dict:
    """Run one fixed-scene episode and return locomotion plus unscored phase."""
    context = context or RolloutContext()
    context.apply_gene(np.asarray(gene, dtype=np.float64))
    rng = np.random.default_rng(seed)
    if target_xy is None:
        distance = float(rng.uniform(1.0, 1.8))
        angle = float(rng.uniform(-np.pi, np.pi))
        target_offset = (distance * np.cos(angle), distance * np.sin(angle))
    else:
        target_offset = target_xy
    yaw0 = float(rng.uniform(-np.pi, np.pi) if initial_yaw is None else initial_yaw)
    sim = mujoco.MjData(context.model)
    sim.qpos[2] = 0.15
    sim.qpos[3:7] = (np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2))
    brain_state = torch.zeros(context.brain_weight.shape[0], dtype=torch.float32)
    stance = STANCE.reshape(18).astype(np.float32)
    for _ in range(int(SETTLE_S / context.model.opt.timestep)):
        sim.ctrl[:] = pd_torque(context.model, sim, stance, KP, KD)
        mujoco.mj_step(context.model, sim)
    x0, y0 = float(sim.qpos[0]), float(sim.qpos[1])
    body_heading = yaw0
    target_world = (x0 + target_offset[0], y0 + target_offset[1])
    last = (x0, y0)
    targets, heights = [], []
    path = 0.0
    brain_turns, brain_forwards = [], []
    renderer = mujoco.Renderer(context.model, 480, 640) if video else None
    cameras = []
    if renderer:
        for azimuth, elevation, distance in ((135.0, 25.0, 1.4), (0.0, 80.0, 1.8), (90.0, 25.0, 1.4), (180.0, 25.0, 1.4)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(context.model, camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
            cameras.append(camera)
    frames = []
    for step in range(int(round(secs * CTRL_HZ))):
        outputs, azimuth, size = _brain_step(context, brain_state, target_world, sim)
        present = bool(np.hypot(target_world[0] - sim.qpos[0], target_world[1] - sim.qpos[1]) < 12.0 and abs(azimuth) < np.deg2rad(150))
        gate = max(0.1, float(np.cos(azimuth))) if present else 0.0
        forward = float(np.clip(12.0 * (outputs["dng100_left"] + outputs["dng100_right"] + outputs["dnb08_left"] + outputs["dnb08_right"]), 0.0, 1.0)) * gate
        turn = -6.0 * ((outputs["dng100_right"] + outputs["dnb08_right"]) - (outputs["dng100_left"] + outputs["dnb08_left"]))
        qpos = sim.qpos[7:25].copy()
        loads = foot_loads(context.model, sim, context.floor_id, context.foot_ids, context.mass_per_leg)
        angles = np.clip(np.mean(np.abs(qpos.reshape(6, 3) - STANCE.reshape(1, 3)), axis=1) / ANGLE_NORMALIZATION_RAD, 0.0, 1.0)
        current = context.current(np.asarray(gene), outputs, loads, angles)
        motor_current = context.vnc.step(current).cpu().numpy()
        # Map the live motor outputs only for diagnostics; learned genes act in the E4 graph.
        from mn_to_joint import translate
        target = translate(motor_current, context.io, gains=MAP_GAINS)
        targets.append(target)
        brain_forwards.append(forward)
        brain_turns.append(turn)
        for _ in range(context.physics_steps):
            sim.ctrl[:] = pd_torque(context.model, sim, target, KP, KD)
            mujoco.mj_step(context.model, sim)
        now = (float(sim.qpos[0]), float(sim.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        heights.append(float(sim.qpos[2]))
        if renderer and step % 2 == 0:
            views = []
            for camera in cameras:
                camera.lookat[:] = (float(sim.qpos[0]), float(sim.qpos[1]), 0.08)
                renderer.update_scene(sim, camera=camera)
                views.append(renderer.render())
            frames.append(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))
    targets = np.asarray(targets)
    heights = np.asarray(heights)
    dx, dy = float(sim.qpos[0]) - x0, float(sim.qpos[1]) - y0
    phase = phase_metrics(targets.reshape(-1, 6, 3)[..., 0], 1.0 / CTRL_HZ)
    result = {
        "seed": int(seed),
        "forward_distance_m": float(dx * np.cos(body_heading) + dy * np.sin(body_heading)),
        "displacement_m": float(np.hypot(dx, dy)),
        "path_m": path,
        "min_height_m": float(heights.min()),
        "final_height_m": float(heights[-1]),
        "fell": bool(heights.min() < H_MIN),
        "phase": phase,
        "target_temporal_std_mean_rad": float(targets.std(axis=0).mean()),
        "brain_forward_mean": float(np.mean(brain_forwards)),
        "brain_turn_mean": float(np.mean(brain_turns)),
        "condition": context.condition,
        "base_weight_hash": context.base_weight_hash,
        "base_pattern_hash": context.base_pattern_hash,
    }
    if renderer:
        import imageio.v2 as imageio
        imageio.mimwrite(video, frames, fps=25)
    return result


def locomotion_fitness(metrics: list[dict]) -> float:
    """Predeclared objective: forward displacement minus fixed fall penalty."""
    return float(np.mean([row["forward_distance_m"] - FALL_CONST * float(row["fell"]) for row in metrics]))


def evaluate_gene(gene: np.ndarray, seeds: tuple[int, ...] = EVALUATION_SEEDS, context: RolloutContext | None = None, secs: float = ROLLOUT_SECS) -> tuple[float, list[dict]]:
    context = context or RolloutContext()
    metrics = [rollout(gene, seed, context=context, secs=secs) for seed in seeds]
    return locomotion_fitness(metrics), metrics


_WORKER_CONTEXT: RolloutContext | None = None
_WORKER_SEEDS: tuple[int, ...] = EVALUATION_SEEDS
_WORKER_SECS: float = ROLLOUT_SECS


def _init_worker(config: dict) -> None:
    global _WORKER_CONTEXT, _WORKER_SEEDS, _WORKER_SECS
    context_config = {key: config[key] for key in ("vnc_data", "brain_data", "condition", "weights_name", "index_name")}
    _WORKER_CONTEXT = RolloutContext(**context_config)
    _WORKER_SEEDS = tuple(config.get("seeds", EVALUATION_SEEDS))
    _WORKER_SECS = float(config.get("secs", ROLLOUT_SECS))


def _evaluate_worker(gene: list[float]) -> float:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("CMA-ES worker was not initialized")
    score, _ = evaluate_gene(np.asarray(gene), _WORKER_SEEDS, _WORKER_CONTEXT, _WORKER_SECS)
    return score


def _config(vnc_data: Path, brain_data: Path, condition: str, weights_name: str, index_name: str, seeds: tuple[int, ...], secs: float) -> dict:
    return {
        "vnc_data": str(vnc_data),
        "brain_data": str(brain_data),
        "condition": condition,
        "weights_name": weights_name,
        "index_name": index_name,
        "seeds": list(seeds),
        "secs": secs,
    }


def write_gene_spec(output: Path = RESULTS / "cmaes_gene_spec.json") -> None:
    payload = {
        "genes": GENES,
        "dimension": len(GENES),
        "default_gene": DEFAULT_GENE.tolist(),
        "initial_stds": INITIAL_STDS.tolist(),
        "frozen_constants_excluded": {
            "vnc_gW": VNC_GW,
            "vnc_tau": VNC_TAU,
            "vnc_dt": VNC_DT,
            "vnc_substeps": VNC_SUBSTEPS,
            "vnc_threshold": VNC_THRESHOLD,
            "vnc_activation_gain": VNC_ACTIVATION_GAIN,
            "vnc_fr_cap": VNC_FR_CAP,
            "pd_kp": KP,
            "pd_kd": KD,
            "stance": STANCE.tolist(),
        },
        "optional_feedback_delay": False,
        "weights_are_modified": False,
        "sensory_encoding": "fixed E4 contact/joint-angle encoding; class gain and sign only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def regression_check(context: RolloutContext, output: Path = RESULTS / "cmaes_regression_check.json") -> dict:
    expected = json.loads((RESULTS / "e4_sensory_feedback.json").read_text(encoding="utf-8"))
    row = rollout(
        DEFAULT_GENE,
        seed=0,
        context=context,
        secs=EVALUATION_SECS,
        target_xy=(1.2, 0.8),
        initial_yaw=0.0,
    )
    expected_loco = expected["locomotion"]
    checks = {
        "path_m": (row["path_m"], expected_loco["path_m"]),
        "forward_distance_m": (row["forward_distance_m"], expected_loco["forward_displacement_m"]),
        "displacement_m": (row["displacement_m"], expected_loco["displacement_m"]),
        "min_height_m": (row["min_height_m"], expected_loco["min_height_m"]),
        "final_height_m": (row["final_height_m"], expected_loco["final_height_m"]),
        "fell": (row["fell"], expected_loco["fell"]),
    }
    errors = {
        key: abs(float(actual) - float(reference))
        for key, (actual, reference) in checks.items()
        if isinstance(actual, (int, float, bool)) and not isinstance(actual, bool)
    }
    passed = all(error <= 1e-6 for error in errors.values()) and checks["fell"][0] == checks["fell"][1]
    result = {
        "gene": DEFAULT_GENE.tolist(),
        "expected_source": str(RESULTS / "e4_sensory_feedback.json"),
        "observed": row,
        "expected_locomotion": expected_loco,
        "absolute_errors": errors,
        "tolerance": 1e-6,
        "pass": passed,
        "base_pattern_hash_before": context.base_pattern_hash,
        "base_pattern_hash_after": csr_hash(context.vnc.W, values=False),
        "frozen_constants_unchanged": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def _load_best(path: Path) -> tuple[float, np.ndarray, list[dict]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["best_fitness"]), np.asarray(data["best_gene"], dtype=np.float64), data.get("history", [])


def run_search(
    condition: str,
    vnc_data: Path = RESULTS,
    brain_data: Path = DATA,
    weights_name: str = "e4_sensory_weight_csr.pkl",
    index_name: str = "e4_sensory_io_index.npz",
    workers: int = 8,
    max_generations: int = MAX_GENERATIONS,
    max_hours: float = MAX_HOURS,
    popsize: int = POPSIZE,
    fresh: bool = False,
) -> dict:
    checkpoint = RESULTS / f"cmaes_checkpoint_{condition}.pkl"
    running = RESULTS / f"cmaes_best_gene_{condition}.json"
    csv_path = RESULTS / f"cmaes_fitness_{condition}.csv"
    config = _config(vnc_data, brain_data, condition, weights_name, index_name, EVALUATION_SEEDS, ROLLOUT_SECS)
    config["seeds"] = list(EVALUATION_SEEDS)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not fresh:
        with checkpoint.open("rb") as stream:
            es = pickle.load(stream)
        loaded = _load_best(running)
        if loaded:
            best_fitness, best_gene, history = loaded
        else:
            best_fitness, best_gene, history = -np.inf, DEFAULT_GENE.copy(), []
    else:
        options = {
            "bounds": [LOWER_BOUNDS.tolist(), UPPER_BOUNDS.tolist()],
            "popsize": popsize,
            "CMA_stds": INITIAL_STDS.tolist(),
            "seed": OPTIMIZER_SEED,
            "verb_disp": 0,
            "verbose": -9,
        }
        es = cma.CMAEvolutionStrategy(DEFAULT_GENE.tolist(), 1.0, options)
        best_fitness, best_gene, history = -np.inf, DEFAULT_GENE.copy(), []
        if csv_path.exists():
            csv_path.unlink()
    fieldnames = ["generation", "best_fitness", "mean_fitness", "wall_clock_s", "best_gene"]
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fieldnames).writeheader()
    start = time.monotonic()
    context = RolloutContext(vnc_data, brain_data, condition, weights_name, index_name)
    global _WORKER_CONTEXT, _WORKER_SEEDS, _WORKER_SECS
    _WORKER_CONTEXT = context
    _WORKER_SEEDS = EVALUATION_SEEDS
    _WORKER_SECS = ROLLOUT_SECS
    pool = None
    try:
        if workers > 1:
            pool = mp.get_context("fork").Pool(processes=workers, initializer=_init_worker, initargs=(config,))
        while es.countiter < max_generations and time.monotonic() - start < max_hours * 3600:
            population = es.ask()
            if pool:
                scores = np.asarray(pool.map(_evaluate_worker, population), dtype=np.float64)
            else:
                scores = np.asarray([_evaluate_worker(gene) for gene in population], dtype=np.float64)
            es.tell(population, (-scores).tolist())
            generation_best = int(np.argmax(scores))
            if float(scores[generation_best]) > best_fitness:
                best_fitness = float(scores[generation_best])
                best_gene = np.asarray(population[generation_best], dtype=np.float64)
            row = {
                "generation": int(es.countiter),
                "best_fitness": best_fitness,
                "mean_fitness": float(scores.mean()),
                "wall_clock_s": time.monotonic() - start,
                "best_gene": json.dumps(best_gene.tolist(), separators=(",", ":")),
            }
            history.append(row)
            with csv_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fieldnames).writerow(row)
            running.write_text(json.dumps({
                "condition": condition,
                "best_fitness": best_fitness,
                "best_gene": best_gene.tolist(),
                "generation": row["generation"],
                "history": history,
                "optimizer_seed": OPTIMIZER_SEED,
                "checkpoint": str(checkpoint),
            }, indent=2) + "\n", encoding="utf-8")
            with checkpoint.open("wb") as stream:
                pickle.dump(es, stream, protocol=pickle.HIGHEST_PROTOCOL)
            print(json.dumps({key: row[key] for key in ("generation", "best_fitness", "mean_fitness", "wall_clock_s")}, sort_keys=True), flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    result = {
        "condition": condition,
        "best_fitness": best_fitness,
        "best_gene": best_gene.tolist(),
        "generation": int(es.countiter),
        "stopped_by": "generation_cap" if es.countiter >= max_generations else "wall_clock_cap",
        "fitness_csv": str(csv_path),
        "checkpoint": str(checkpoint),
        "base_pattern_hash": context.base_pattern_hash,
    }
    canonical = RESULTS / "cmaes_best_gene.json" if condition == "connectome" else RESULTS / f"cmaes_best_gene_{condition}.json"
    canonical.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate_best(condition: str, gene: np.ndarray, context: RolloutContext, output: Path, video: Path) -> dict:
    metrics = [rollout(gene, seed, context=context, secs=EVALUATION_SECS, video=video if index == 0 else None) for index, seed in enumerate(EVALUATION_SEEDS)]
    result = {
        "condition": condition,
        "gene": np.asarray(gene).tolist(),
        "fitness": locomotion_fitness(metrics),
        "evaluation_secs": EVALUATION_SECS,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "metrics": metrics,
        "measured_phase_not_in_fitness": True,
        "fitness_definition": "mean forward displacement minus fixed fall penalty",
        "fall_penalty": FALL_CONST,
        "height_threshold": H_MIN,
        "weights_frozen": True,
        "video": str(video),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=("connectome", "random", "rewired"), default="connectome")
    ap.add_argument("--vnc_data", type=Path, default=RESULTS)
    ap.add_argument("--brain_data", type=Path, default=DATA)
    ap.add_argument("--weights_name", default="e4_sensory_weight_csr.pkl")
    ap.add_argument("--index_name", default="e4_sensory_io_index.npz")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_generations", type=int, default=MAX_GENERATIONS)
    ap.add_argument("--max_hours", type=float, default=MAX_HOURS)
    ap.add_argument("--popsize", type=int, default=POPSIZE)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--write_gene_spec", action="store_true")
    ap.add_argument("--regression_check", action="store_true")
    ap.add_argument("--evaluate_best", action="store_true")
    ap.add_argument("--gene_file", type=Path, default=None)
    a = ap.parse_args()
    write_gene_spec()
    if a.write_gene_spec:
        print(f"gene spec -> {RESULTS / 'cmaes_gene_spec.json'}")
    context = RolloutContext(a.vnc_data, a.brain_data, a.condition, a.weights_name, a.index_name)
    if a.regression_check:
        if a.condition != "connectome":
            raise ValueError("regression check is defined for the real E4 connectome")
        print(json.dumps(regression_check(context), sort_keys=True))
        return
    if a.evaluate_best:
        path = a.gene_file or (RESULTS / ("cmaes_best_gene.json" if a.condition == "connectome" else f"cmaes_best_gene_{a.condition}.json"))
        best = json.loads(path.read_text(encoding="utf-8"))
        result = evaluate_best(a.condition, np.asarray(best.get("best_gene", best.get("gene"))), context, RESULTS / "cmaes_walk_metrics.json", RESULTS / "cmaes_walk_multiview.mp4")
        print(json.dumps(result, sort_keys=True))
        return
    result = run_search(a.condition, a.vnc_data, a.brain_data, a.weights_name, a.index_name, a.workers, a.max_generations, a.max_hours, a.popsize, a.fresh)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
