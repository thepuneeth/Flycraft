"""Outcome D: fixed-topology VNC weight learning with matched controls.

The CPU fallback is deliberately small: a closed-form decoder is fit for each
candidate low-rank edge update, then CMA-ES searches that declared weight
subspace. Locomotion fine-tuning searches the same weight subspace with the
same body-only objective. Tripod metrics are evaluation-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, deque
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
from hexapod_gait import H_MIN, HexapodGait, XML, home_stance, pd_torque  # noqa: E402
from probe_vnc_phase import phase_metrics  # noqa: E402
from reservoir_3way import degree_preserving_rewire  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


# Frozen Path-2 dynamics/body values. D changes only W values and decoder.
VNC_TAU = 0.02
VNC_DT = 0.005
VNC_SUBSTEPS = 4
VNC_GW = 0.03
VNC_THRESHOLD = 7.5
VNC_ACTIVATION_GAIN = 1.0
VNC_FR_CAP = 200.0
VNC_HETEROGENEITY = True
VNC_SEED = 1
VNC_DRIVE = 100.0
CTRL_HZ = 50.0
KP = 3.0
KD = 0.05
STANCE = home_stance().reshape(18)
FALL_PENALTY = 0.5
UPRIGHT_COS_MIN = 0.5
EVALUATION_SEEDS = (20260829, 20260830, 20260831, 20260832)

# Declared D training protocol. These are not tuned after the first run.
TARGET_DT = 1.0 / CTRL_HZ
TARGET_STEPS = 400
TARGET_FORWARD = 0.8
TARGET_TURN = 0.0
TRAIN_FRACTION = 0.7
LOW_RANK = 8
FEATURE_SEED = 20260829
WEIGHT_DELTA_SCALE = 3.0
DECODER_RIDGE = 1e-3
ACTION_LIMITS = (-2.5, 2.5)
D1_POPSIZE = 8
D1_GENERATIONS = 12
D1_SIGMA = 0.8
D2_POPSIZE = 6
D2_GENERATIONS = 12
D2_SIGMA = 0.15
D2_UPRIGHT_BONUS = 0.05
D2_TRAIN_SECS = 4.0
BOOTSTRAP_REPS = 10_000
OPTIMIZER_SEED = 20260829
RANDOM_TOPOLOGY_SEED = 20260829
RANDOM_VALUE_SEED = 20260830
REWIRE_SEED = 20260829

GRAPH_WEIGHTS = "all_legs_weight_csr.pkl"
GRAPH_INDEX = "all_legs_io_index.npz"
CONDITIONS = ("connectome", "random", "rewired")


def digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def r2_columns(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    residual = np.sum((truth - prediction) ** 2, axis=0)
    total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    return 1.0 - residual / np.maximum(total, np.finfo(float).eps)


def bootstrap_ci(values: np.ndarray, seed: int = OPTIMIZER_SEED) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def make_target() -> np.ndarray:
    gait = HexapodGait()
    return np.stack([gait.targets(TARGET_FORWARD, TARGET_TURN, TARGET_DT) for _ in range(TARGET_STEPS)]).astype(np.float32)


def ensure_target() -> tuple[Path, np.ndarray]:
    path = RESULTS / "outcomeD_imitation_target.npz"
    q = make_target()
    if path.exists():
        with np.load(path) as saved:
            if not np.array_equal(saved["q"], q):
                raise AssertionError(f"declared target changed: {path}")
        return path, q
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        q=q,
        sample_dt_s=np.asarray(TARGET_DT),
        forward=np.asarray(TARGET_FORWARD),
        turn=np.asarray(TARGET_TURN),
        stance=STANCE.astype(np.float32),
        source=np.asarray("Path-1 HexapodGait defaults"),
        performance_tuned=np.asarray(False),
    )
    return path, q


def replay_target(q: np.ndarray) -> dict:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    settle = STANCE
    for _ in range(int(0.7 / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, settle, KP, KD)
        mujoco.mj_step(model, data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    yaw0 = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
    heights = []
    path = 0.0
    last = (x0, y0)
    physics_steps = max(1, int(round(TARGET_DT / model.opt.timestep)))
    for row in q:
        for _ in range(physics_steps):
            data.ctrl[:] = pd_torque(model, data, row, KP, KD)
            mujoco.mj_step(model, data)
        now = (float(data.qpos[0]), float(data.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        heights.append(float(data.qpos[2]))
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    phases = phase_metrics(q.reshape(-1, 6, 3)[..., 0], TARGET_DT, "E3-PRESENT", "E3-ABSENT")
    return {
        "steps": int(len(q)),
        "secs": float(len(q) * TARGET_DT),
        "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "path_m": path,
        "min_height_m": float(np.min(heights)),
        "fell": bool(np.min(heights) < H_MIN),
        "tripod_coherence_scalar": float(phases["tripod_coherence_scalar"]),
        "phase": phases,
    }


def random_support(weight: torch.Tensor, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Configuration-model support with the real in/out degree sequences."""
    weight = weight.to_sparse_csr().cpu()
    crow = weight.crow_indices().numpy().copy()
    degrees = np.diff(crow)
    rng = np.random.default_rng(seed)
    n = int(weight.shape[1])
    rows = np.repeat(np.arange(n, dtype=np.int64), degrees)
    real_columns = weight.col_indices().numpy()
    column_stubs = np.repeat(np.arange(n, dtype=np.int64), np.bincount(real_columns, minlength=n))
    rng.shuffle(column_stubs)
    keys = rows * n + column_stubs
    counts = Counter(keys.tolist())
    queue = deque(i for i, key in enumerate(keys) if counts[key] > 1)
    for _ in range(len(queue) + len(keys)):
        if not queue:
            break
        i = queue.popleft()
        old1 = int(keys[i])
        if counts[old1] <= 1:
            continue
        row1, column1 = int(rows[i]), int(column_stubs[i])
        for _ in range(10_000):
            j = int(rng.integers(len(keys)))
            row2, column2 = int(rows[j]), int(column_stubs[j])
            old2 = int(keys[j])
            if row1 == row2 or column1 == column2:
                continue
            new1, new2 = row1 * n + column2, row2 * n + column1
            if counts.get(new1, 0) or counts.get(new2, 0):
                continue
            counts[old1] -= 1
            counts[old2] -= 1
            counts[new1] += 1
            counts[new2] += 1
            column_stubs[i], column_stubs[j] = column_stubs[j], column_stubs[i]
            keys[i], keys[j] = new1, new2
            queue.extend((i, j))
            break
        else:
            raise RuntimeError("configuration-model duplicate repair did not converge")
    if len(np.unique(keys)) != len(keys):
        raise RuntimeError("configuration-model support still contains duplicate edges")
    if not np.array_equal(np.bincount(column_stubs, minlength=n), np.bincount(real_columns, minlength=n)):
        raise AssertionError("configuration-model column degree sequence changed")
    for start, end in zip(crow[:-1], crow[1:]):
        column_stubs[start:end].sort()
    columns = column_stubs
    assert len(columns) == int(weight._nnz())
    return crow, columns


class Graph:
    def __init__(self, condition: str, data: Path = RESULTS):
        self.condition = condition
        with (data / GRAPH_WEIGHTS).open("rb") as stream:
            real = pickle.load(stream).to_sparse_csr().cpu()
        self.n = int(real.shape[0])
        self.real_values = real.values().numpy().astype(np.float32, copy=True)
        real_crow = real.crow_indices().numpy().astype(np.int64, copy=True)
        real_col = real.col_indices().numpy().astype(np.int64, copy=True)
        if condition == "connectome":
            self.crow, self.col = real_crow, real_col
            self.initial_values = self.real_values.copy()
            self.initialization = "real signed MANC values"
        else:
            value_rng = np.random.default_rng(RANDOM_VALUE_SEED)
            self.initial_values = self.real_values.copy()
            value_rng.shuffle(self.initial_values)
            if condition == "random":
                self.crow, self.col = random_support(real, RANDOM_TOPOLOGY_SEED)
                self.initialization = "seeded permutation of real signed values on configuration-model matched in/out-degree support"
            elif condition == "rewired":
                rewired = degree_preserving_rewire(real, REWIRE_SEED)
                self.crow = rewired.crow_indices().numpy().astype(np.int64, copy=True)
                self.col = rewired.col_indices().numpy().astype(np.int64, copy=True)
                self.initialization = "seeded permutation of real signed values on degree-preserving rewired support"
            else:
                raise ValueError(f"unknown condition {condition!r}")
        if len(self.initial_values) != len(self.col):
            raise AssertionError("graph support/value mismatch")
        self.rows = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.crow))
        feature_rng = np.random.default_rng(FEATURE_SEED)
        self.row_features = (feature_rng.normal(size=(self.n, LOW_RANK)) / np.sqrt(LOW_RANK)).astype(np.float32)
        self.col_features = (feature_rng.normal(size=(self.n, LOW_RANK)) / np.sqrt(LOW_RANK)).astype(np.float32)
        self.io = np.load(data / GRAPH_INDEX)
        self.motor_out = self.io["motor_out"].astype(np.int64)
        self.pattern_hash = digest_arrays(self.crow, self.col)
        self.initial_hash = digest_arrays(self.crow, self.col, self.initial_values)

    def values(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        if theta.shape != (LOW_RANK,):
            raise ValueError(f"theta has shape {theta.shape}; expected {(LOW_RANK,)}")
        basis = np.sum(self.row_features[self.rows] * self.col_features[self.col] * theta[None, :], axis=1)
        return self.initial_values * (1.0 + WEIGHT_DELTA_SCALE * basis)

    def sparse(self, theta: np.ndarray) -> torch.Tensor:
        values = torch.from_numpy(self.values(theta))
        return torch.sparse_csr_tensor(
            torch.from_numpy(self.crow),
            torch.from_numpy(self.col),
            values,
            size=(self.n, self.n),
        )

    def summary(self) -> dict:
        io = self.io
        descending = np.unique(io["descending_in"])
        motor = np.unique(io["motor_out"])
        return {
            "condition": self.condition,
            "nodes": self.n,
            "signed_edges": int(len(self.col)),
            "motor_outputs": int(len(io["motor_out"])),
            "descending_inputs": int(len(io["descending_in"])),
            "intrinsic_nodes": int(self.n - len(np.union1d(descending, motor))),
            "published_coordination_candidates": int(len(io["published_coordination"])),
            "afferent_partition": "descending_in (DNg100/DNb08 and fixed descending inputs)",
            "intrinsic_partition": "all remaining non-efferent graph nodes",
            "efferent_partition": "motor_out",
            "pattern_hash": self.pattern_hash,
            "initial_weight_hash": self.initial_hash,
            "initialization": self.initialization,
        }


class Trainer:
    def __init__(self, graph: Graph, data: Path = RESULTS):
        self.graph = graph
        self.vnc = VNCStepper(
            data,
            substeps=VNC_SUBSTEPS,
            tau=VNC_TAU,
            dt=VNC_DT,
            gW=VNC_GW,
            weights_name=GRAPH_WEIGHTS,
            index_name=GRAPH_INDEX,
            model="paper",
            threshold=VNC_THRESHOLD,
            activation_gain=VNC_ACTIVATION_GAIN,
            fr_cap=VNC_FR_CAP,
            heterogeneity=VNC_HETEROGENEITY,
            seed=VNC_SEED,
        )
        self.vnc.W = graph.sparse(np.zeros(LOW_RANK, dtype=np.float32))
        self.current = np.zeros(graph.n, dtype=np.float32)
        self.set_drive(VNC_DRIVE)

    def set_drive(self, drive: float) -> None:
        if not np.isfinite(drive):
            raise ValueError(f"drive must be finite, got {drive!r}")
        self.current.fill(0.0)
        for key in ("dng100_left", "dng100_right", "dnb08_left", "dnb08_right"):
            self.current[self.vnc.io[key]] = float(drive)

    def set_theta(self, theta: np.ndarray) -> None:
        self.vnc.W = self.graph.sparse(theta)

    def states(self, theta: np.ndarray, steps: int) -> np.ndarray:
        self.set_theta(theta)
        self.vnc.reset()
        return np.stack([self.vnc.step(self.current).cpu().numpy() for _ in range(steps)]).astype(np.float32)


def fit_decoder(states: np.ndarray, q: np.ndarray) -> dict:
    split = int(len(states) * TRAIN_FRACTION)
    mean = states[:split].mean(axis=0)
    scale = states[:split].std(axis=0)
    scale[scale < 1e-7] = 1.0
    design = (states - mean) / scale
    x = design[:split].astype(np.float64)
    y = (q[:split] - STANCE).astype(np.float64)
    gram = x.T @ x + DECODER_RIDGE * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, x.T @ y).astype(np.float32)
    prediction = design @ weights + STANCE
    train_r2 = r2_columns(q[:split], prediction[:split])
    heldout_r2 = r2_columns(q[split:], prediction[split:])
    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "weights": weights,
        "stance": STANCE.astype(np.float32),
        "split": split,
        "train_r2": train_r2,
        "heldout_r2": heldout_r2,
        "train_mse": float(np.mean((prediction[:split] - q[:split]) ** 2)),
        "heldout_mse": float(np.mean((prediction[split:] - q[split:]) ** 2)),
        "prediction": prediction,
    }


def decode(motor: np.ndarray, decoder: dict, clip: bool = False) -> np.ndarray:
    action = (np.asarray(motor) - decoder["mean"]) / decoder["scale"] @ decoder["weights"] + decoder["stance"]
    if clip:
        action = np.clip(action, ACTION_LIMITS[0], ACTION_LIMITS[1])
    return action


def imitation_candidate(trainer: Trainer, q: np.ndarray, theta: np.ndarray) -> tuple[float, dict]:
    states = trainer.states(theta, len(q))
    decoder = fit_decoder(states, q)
    score = -decoder["heldout_mse"]
    return score, decoder


def checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def train_d1(graph: Graph, q: np.ndarray, fresh: bool = False) -> dict:
    trainer = Trainer(graph)
    path = RESULTS / f"outcomeD_ckpt_{graph.condition}_d1.pkl"
    loaded = None if fresh or not path.exists() else pickle.load(path.open("rb"))
    if loaded and loaded.get("stage") == "d1":
        es = loaded["es"]
        best_theta = np.asarray(loaded["best_theta"], dtype=np.float64)
        best_score = float(loaded["best_score"])
        history = loaded["history"]
    else:
        es = cma.CMAEvolutionStrategy(
            np.zeros(LOW_RANK),
            D1_SIGMA,
            {"popsize": D1_POPSIZE, "seed": OPTIMIZER_SEED, "verb_disp": 0, "verbose": -9},
        )
        best_theta = np.zeros(LOW_RANK, dtype=np.float64)
        best_score = -np.inf
        history = []
    while es.countiter < D1_GENERATIONS:
        population = es.ask()
        scores = []
        for theta in population:
            score, _ = imitation_candidate(trainer, q, np.asarray(theta))
            scores.append(score)
        es.tell(population, (-np.asarray(scores)).tolist())
        winner = int(np.argmax(scores))
        if float(scores[winner]) > best_score:
            best_score = float(scores[winner])
            best_theta = np.asarray(population[winner], dtype=np.float64)
        row = {
            "generation": int(es.countiter),
            "best_heldout_score": best_score,
            "mean_heldout_score": float(np.mean(scores)),
        }
        history.append(row)
        checkpoint(path, {
            "stage": "d1",
            "condition": graph.condition,
            "es": es,
            "best_theta": best_theta,
            "best_score": best_score,
            "history": history,
            "graph_pattern_hash": graph.pattern_hash,
            "target_steps": len(q),
        })
        print(json.dumps({"condition": graph.condition, "stage": "d1", **row}, sort_keys=True), flush=True)
    states = trainer.states(best_theta, len(q))
    decoder = fit_decoder(states, q)
    return {
        "theta": best_theta,
        "decoder": decoder,
        "best_score": best_score,
        "generations": int(es.countiter),
        "history": history,
        "states": states,
    }


def _camera_rig(model):
    cameras = []
    for azimuth, elevation, distance in ((135.0, 25.0, 1.4), (0.0, 80.0, 1.8), (90.0, 25.0, 1.4), (180.0, 25.0, 1.4)):
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        cameras.append(camera)
    return cameras


def root_up_z(data: mujoco.MjData) -> float:
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    return float(rotation[8])


def locomotion_episode(
    trainer: Trainer,
    theta: np.ndarray,
    decoder: dict,
    seed: int,
    secs: float,
    video: Path | None = None,
) -> dict:
    trainer.set_theta(theta)
    trainer.vnc.reset()
    rng = np.random.default_rng(seed)
    model = trainer_model(trainer)
    data = mujoco.MjData(model)
    yaw0 = float(rng.uniform(-np.pi, np.pi))
    data.qpos[2] = 0.15
    data.qpos[3:7] = (np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2))
    for _ in range(int(0.7 / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, STANCE, KP, KD)
        mujoco.mj_step(model, data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    last = (x0, y0)
    path = 0.0
    heights = []
    up_z_values = []
    actions = []
    physics_steps = max(1, int(round(TARGET_DT / model.opt.timestep)))
    renderer = mujoco.Renderer(model, 480, 640) if video else None
    cameras = _camera_rig(model) if renderer else []
    frames = []
    for step in range(int(round(secs * CTRL_HZ))):
        motor = trainer.vnc.step(trainer.current).cpu().numpy()
        action = decode(motor, decoder, clip=True)
        actions.append(action)
        for _ in range(physics_steps):
            data.ctrl[:] = pd_torque(model, data, action, KP, KD)
            mujoco.mj_step(model, data)
        now = (float(data.qpos[0]), float(data.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        heights.append(float(data.qpos[2]))
        up_z_values.append(root_up_z(data))
        if renderer and step % 2 == 0:
            views = []
            for camera in cameras:
                camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.08)
                renderer.update_scene(data, camera=camera)
                views.append(renderer.render())
            frames.append(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))
    actions = np.asarray(actions)
    heights = np.asarray(heights)
    up_z_values = np.asarray(up_z_values)
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    fell = bool(heights.min() < H_MIN or up_z_values.min() < UPRIGHT_COS_MIN)
    phase = phase_metrics(actions.reshape(-1, 6, 3)[..., 0], TARGET_DT, "E3-PRESENT", "E3-ABSENT")
    upright = float(not fell)
    locomotion_return = float(dx * np.cos(yaw0) + dy * np.sin(yaw0) + D2_UPRIGHT_BONUS * upright - FALL_PENALTY * float(fell))
    if renderer:
        import imageio.v2 as imageio
        imageio.mimwrite(video, frames, fps=25)
    return {
        "seed": int(seed),
        "secs": float(secs),
        "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "path_m": float(path),
        "min_height_m": float(heights.min()),
        "min_root_up_z": float(up_z_values.min()),
        "max_tilt_deg": float(np.degrees(np.arccos(np.clip(up_z_values.min(), -1.0, 1.0)))),
        "final_height_m": float(heights[-1]),
        "fell": fell,
        "upright": bool(not fell),
        "tripod_coherence_scalar": float(phase["tripod_coherence_scalar"]),
        "phase": phase,
        "action_temporal_std_mean_rad": float(actions.std(axis=0).mean()),
        "locomotion_return": locomotion_return,
    }


def trainer_model(trainer: Trainer):
    if not hasattr(trainer, "model"):
        trainer.model = mujoco.MjModel.from_xml_path(str(XML))
    return trainer.model


def locomotion_score(trainer: Trainer, theta: np.ndarray, decoder: dict) -> tuple[float, list[dict]]:
    rows = [locomotion_episode(trainer, theta, decoder, seed, D2_TRAIN_SECS) for seed in EVALUATION_SEEDS]
    return float(np.mean([row["locomotion_return"] for row in rows])), rows


def train_d2(graph: Graph, d1: dict, fresh: bool = False) -> dict:
    trainer = Trainer(graph)
    path = RESULTS / f"outcomeD_ckpt_{graph.condition}_d2.pkl"
    loaded = None if fresh or not path.exists() else pickle.load(path.open("rb"))
    if loaded and loaded.get("stage") == "d2":
        es = loaded["es"]
        best_theta = np.asarray(loaded["best_theta"], dtype=np.float64)
        best_score = float(loaded["best_score"])
        history = loaded["history"]
    else:
        es = cma.CMAEvolutionStrategy(
            np.asarray(d1["theta"], dtype=np.float64),
            D2_SIGMA,
            {"popsize": D2_POPSIZE, "seed": OPTIMIZER_SEED, "verb_disp": 0, "verbose": -9},
        )
        best_theta = np.asarray(d1["theta"], dtype=np.float64)
        best_score, _ = locomotion_score(trainer, best_theta, d1["decoder"])
        history = [{"generation": 0, "best_locomotion_return": best_score, "mean_locomotion_return": best_score}]
    while es.countiter < D2_GENERATIONS:
        population = es.ask()
        scores = []
        for theta in population:
            score, _ = locomotion_score(trainer, np.asarray(theta), d1["decoder"])
            scores.append(score)
        es.tell(population, (-np.asarray(scores)).tolist())
        winner = int(np.argmax(scores))
        if float(scores[winner]) > best_score:
            best_score = float(scores[winner])
            best_theta = np.asarray(population[winner], dtype=np.float64)
        row = {
            "generation": int(es.countiter),
            "best_locomotion_return": best_score,
            "mean_locomotion_return": float(np.mean(scores)),
        }
        history.append(row)
        checkpoint(path, {
            "stage": "d2",
            "condition": graph.condition,
            "es": es,
            "best_theta": best_theta,
            "best_score": best_score,
            "history": history,
            "d1_theta": d1["theta"],
            "graph_pattern_hash": graph.pattern_hash,
        })
        print(json.dumps({"condition": graph.condition, "stage": "d2", **row}, sort_keys=True), flush=True)
    return {"theta": best_theta, "best_score": best_score, "generations": int(es.countiter), "history": history}


def evaluate_model(trainer: Trainer, theta: np.ndarray, decoder: dict, condition: str, stage: str) -> list[dict]:
    video = RESULTS / "outcomeD_walk_multiview.mp4" if condition == "connectome" and stage == "d2" else RESULTS / f"outcomeD_walk_multiview_{condition}.mp4"
    rows = []
    for index, seed in enumerate(EVALUATION_SEEDS):
        rows.append(locomotion_episode(trainer, theta, decoder, seed, 8.0, video if index == 0 else None))
    return rows


def serial_decoder(decoder: dict) -> dict:
    return {
        "split": int(decoder["split"]),
        "train_r2": decoder["train_r2"].tolist(),
        "heldout_r2": decoder["heldout_r2"].tolist(),
        "train_mse": decoder["train_mse"],
        "heldout_mse": decoder["heldout_mse"],
        "weight_hash": digest_arrays(decoder["weights"], decoder["mean"], decoder["scale"]),
    }


def condition_report(graph: Graph, d1: dict, d2: dict, d1_eval: list[dict], d2_eval: list[dict]) -> dict:
    dynamic = np.tile((True, True, False), 6)
    heldout = np.asarray(d1["decoder"]["heldout_r2"])
    return {
        "condition": graph.condition,
        "graph": graph.summary(),
        "d1": {
            "method": "CMA-ES over 8-dimensional fixed low-rank edge-value update; closed-form ridge decoder per candidate",
            "generations": d1["generations"],
            "best_theta": np.asarray(d1["theta"]).tolist(),
            "best_heldout_score": float(d1["best_score"]),
            "decoder": serial_decoder(d1["decoder"]),
            "heldout_r2_mean_all": float(heldout.mean()),
            "heldout_r2_mean_dynamic": float(heldout[dynamic].mean()),
            "closed_loop_8s": d1_eval,
        },
        "d2": {
            "method": "CMA-ES weight fine-tune; return is forward distance + fixed upright bonus - fixed fall penalty",
            "generations": d2["generations"],
            "best_theta": np.asarray(d2["theta"]).tolist(),
            "best_training_return": float(d2["best_score"]),
            "closed_loop_8s": d2_eval,
        },
        "objective": {
            "d1": "held-out joint-trajectory reconstruction MSE to declared Path-1 q*(t)",
            "d2": "forward distance + 0.05 if root up_z >= 0.5 and height >= 0.02 m - 0.5 if either upright check fails",
            "tripod_in_objective": False,
            "catch_in_objective": False,
            "prey_in_objective": False,
            "target_reaching_in_objective": False,
        },
        "frozen_dynamics": {
            "tau": VNC_TAU,
            "dt": VNC_DT,
            "substeps": VNC_SUBSTEPS,
            "gW": VNC_GW,
            "threshold": VNC_THRESHOLD,
            "activation_gain": VNC_ACTIVATION_GAIN,
            "fr_cap": VNC_FR_CAP,
            "heterogeneity": VNC_HETEROGENEITY,
            "heterogeneity_seed": VNC_SEED,
            "upright_cos_min": UPRIGHT_COS_MIN,
            "control_hz": CTRL_HZ,
            "pd_kp": KP,
            "pd_kd": KD,
            "motor_outputs": 260,
        },
        "honesty": {
            "weight_values_trained": True,
            "topology_trained": False,
            "decoder_trained": True,
            "imitation_target_is_kinematic_only": True,
            "tripod_measured_post_hoc": True,
        },
    }


def write_three_way(reports: list[dict]) -> None:
    rows = []
    for report in reports:
        d1 = report["d1"]["closed_loop_8s"]
        d2 = report["d2"]["closed_loop_8s"]
        d1_forward = np.asarray([row["forward_distance_m"] for row in d1])
        d2_forward = np.asarray([row["forward_distance_m"] for row in d2])
        d1_coherence = np.asarray([row["tripod_coherence_scalar"] for row in d1])
        d2_coherence = np.asarray([row["tripod_coherence_scalar"] for row in d2])
        d2_return = np.asarray([row["locomotion_return"] for row in d2])
        rows.append({
            "condition": report["condition"],
            "d1_heldout_r2_all": report["d1"]["heldout_r2_mean_all"],
            "d1_heldout_r2_dynamic": report["d1"]["heldout_r2_mean_dynamic"],
            "d1_forward_m": float(d1_forward.mean()),
            "d1_forward_ci_low": bootstrap_ci(d1_forward)[0],
            "d1_forward_ci_high": bootstrap_ci(d1_forward)[1],
            "d1_tripod_coherence": float(d1_coherence.mean()),
            "d1_tripod_ci_low": bootstrap_ci(d1_coherence)[0],
            "d1_tripod_ci_high": bootstrap_ci(d1_coherence)[1],
            "d2_forward_m": float(d2_forward.mean()),
            "d2_forward_ci_low": bootstrap_ci(d2_forward)[0],
            "d2_forward_ci_high": bootstrap_ci(d2_forward)[1],
            "d2_return": float(d2_return.mean()),
            "d2_return_ci_low": bootstrap_ci(d2_return)[0],
            "d2_return_ci_high": bootstrap_ci(d2_return)[1],
            "d2_tripod_coherence": float(d2_coherence.mean()),
            "d2_tripod_ci_low": bootstrap_ci(d2_coherence)[0],
            "d2_tripod_ci_high": bootstrap_ci(d2_coherence)[1],
            "d2_falls": int(sum(row["fell"] for row in d2)),
        })
    out = RESULTS / "outcomeD_3way.csv"
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [row["condition"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    axes[0].bar(labels, [row["d1_heldout_r2_dynamic"] for row in rows], color="#5b8db8")
    axes[0].set_title("D1 held-out dynamic R²")
    axes[1].bar(labels, [row["d2_return"] for row in rows], color="#6a9f58")
    axes[1].set_title("D2 upright-gated return")
    axes[1].set_ylabel("forward m + fixed upright/fall terms")
    axes[2].bar(labels, [row["d2_tripod_coherence"] for row in rows], color="#999999")
    axes[2].set_title("D2 command tripod scalar\n(post hoc; not body stability)")
    figure.savefig(RESULTS / "outcomeD_3way.png", dpi=200, facecolor="white")
    plt.close(figure)


def train_condition(condition: str, q: np.ndarray, fresh: bool) -> dict:
    graph = Graph(condition)
    d1 = train_d1(graph, q, fresh=fresh)
    d1_eval = evaluate_model(Trainer(graph), d1["theta"], d1["decoder"], condition, "d1")
    d2 = train_d2(graph, d1, fresh=fresh)
    d2_eval = evaluate_model(Trainer(graph), d2["theta"], d1["decoder"], condition, "d2")
    report = condition_report(graph, d1, d2, d1_eval, d2_eval)
    output = RESULTS / f"outcomeD_metrics_{condition}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "condition": condition,
        "d1_heldout_r2_dynamic": report["d1"]["heldout_r2_mean_dynamic"],
        "d2_forward_mean_m": float(np.mean([row["forward_distance_m"] for row in d2_eval])),
        "d2_coherence_mean": float(np.mean([row["tripod_coherence_scalar"] for row in d2_eval])),
    }, sort_keys=True), flush=True)
    return report


def completed_reports() -> list[dict]:
    return [
        json.loads((RESULTS / f"outcomeD_metrics_{condition}.json").read_text(encoding="utf-8"))
        for condition in CONDITIONS
        if (RESULTS / f"outcomeD_metrics_{condition}.json").exists()
    ]


def heldout_episode_summary(rows: list[dict]) -> dict:
    forward = np.asarray([row["forward_distance_m"] for row in rows], dtype=np.float64)
    returns = np.asarray([row["locomotion_return"] for row in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "forward_distance_m": {
            "mean": float(forward.mean()),
            "ci_low": bootstrap_ci(forward)[0],
            "ci_high": bootstrap_ci(forward)[1],
        },
        "locomotion_return": {
            "mean": float(returns.mean()),
            "ci_low": bootstrap_ci(returns)[0],
            "ci_high": bootstrap_ci(returns)[1],
        },
        "falls": int(sum(row["fell"] for row in rows)),
        "phase_verdicts": sorted(set(row["phase"]["verdict"] for row in rows)),
        "per_seed": [
            {
                "seed": row["seed"],
                "forward_distance_m": row["forward_distance_m"],
                "locomotion_return": row["locomotion_return"],
                "fell": row["fell"],
                "min_root_up_z": row["min_root_up_z"],
                "max_tilt_deg": row["max_tilt_deg"],
                "tripod_coherence_scalar": row["tripod_coherence_scalar"],
                "phase_verdict": row["phase"]["verdict"],
            }
            for row in rows
        ],
    }


def evaluate_heldout_drive(q: np.ndarray, drive: float) -> dict:
    """Evaluate saved D1/D2 models at one untrained descending-drive level."""
    reports = {}
    for condition in CONDITIONS:
        graph = Graph(condition)
        d1 = train_d1(graph, q, fresh=False)
        d2_path = RESULTS / f"outcomeD_ckpt_{condition}_d2.pkl"
        if not d2_path.exists():
            raise FileNotFoundError(f"missing completed D2 checkpoint: {d2_path}")
        d2 = pickle.load(d2_path.open("rb"))
        trainer = Trainer(graph)
        trainer.set_drive(drive)
        d1_rows = [
            locomotion_episode(trainer, d1["theta"], d1["decoder"], seed, 8.0)
            for seed in EVALUATION_SEEDS
        ]
        d2_rows = [
            locomotion_episode(trainer, np.asarray(d2["best_theta"]), d1["decoder"], seed, 8.0)
            for seed in EVALUATION_SEEDS
        ]
        reports[condition] = {
            "d1": heldout_episode_summary(d1_rows),
            "d2": heldout_episode_summary(d2_rows),
        }
    return {
        "gate": "Outcome D held-out drive audit",
        "drive": float(drive),
        "training_drive": VNC_DRIVE,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "retrained": False,
        "weights_changed": False,
        "conditions": reports,
        "prey_generalization": "not applicable: Outcome D has no prey state or prey input",
        "sample_efficiency": "not estimable: no fixed distance threshold or wall-time logger was declared before training",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, default=None)
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--heldout-drive", type=float, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    target_path, q = ensure_target()
    replay_path = RESULTS / "outcomeD_target_replay.json"
    if not replay_path.exists():
        replay_path.write_text(json.dumps({"target_artifact": str(target_path), "replay": replay_target(q)}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target_artifact": str(target_path), "target_replay": str(replay_path)}, sort_keys=True), flush=True)
    if args.target_only:
        return
    if args.heldout_drive is not None:
        output = RESULTS / f"outcomeD_heldout_drive_{args.heldout_drive:g}.json"
        output.write_text(json.dumps(evaluate_heldout_drive(q, args.heldout_drive), indent=2) + "\n", encoding="utf-8")
        print(f"held-out drive report -> {output}")
        return
    conditions = (args.condition,) if args.condition else CONDITIONS
    reports = []
    for condition in conditions:
        reports.append(train_condition(condition, q, fresh=args.fresh))
    reports = completed_reports()
    combined = RESULTS / "outcomeD_3way.json"
    combined.write_text(json.dumps({
        "gate": "Outcome D",
        "conditions_completed": [report["condition"] for report in reports],
        "reports": reports,
        "target_artifact": str(target_path),
        "stop_before_final_claim": True,
    }, indent=2) + "\n", encoding="utf-8")
    write_three_way(reports)
    print("RESULT_JSON " + json.dumps({"conditions": [report["condition"] for report in reports], "stop_before_final_claim": True}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
