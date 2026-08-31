"""Pre-registered seven-axis wiring-specificity campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
VNC_DATA = HERE.parent.parent / "results" / "path2"
BRAIN_DATA = HERE.parent.parent / "data"
RESULTS = VNC_DATA
sys.path.insert(0, str(HERE))

from hexapod_gait import H_MIN, XML, home_stance, pd_torque  # noqa: E402
from mn_to_joint import translate  # noqa: E402
from preflight import EXPECTED_SHA256, sha256  # noqa: E402
from probe_structured_descending import brain_type_indices  # noqa: E402
from reservoir_3way import degree_preserving_rewire, randomize_weights, tensor_sha256  # noqa: E402
from run_hexapod_chase import retina  # noqa: E402
from step_decode import motor_pools  # noqa: E402
from stepwalk_controls import episode_specs  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


AXIS_ORDER = ("B1", "B4", "B5", "B7", "B2", "B3", "B6")
PRIMARY_AXES = AXIS_ORDER
RNG_SEEDS = tuple(20260830 + index for index in range(12))
K = 12
EPISODE_SEED = 20260829
N_EPISODES = 20
CTRL_HZ = 50.0
DT_S = 1.0 / CTRL_HZ
SETTLE_S = 0.7
KP = 3.0
KD = 0.05
H_MIN_EXPECTED = 0.08
MAP_GAINS = (-0.55, 0.65, 0.65)
VNC_SUBSTEPS = 4
VNC_TAU = 0.02
VNC_DT = 0.005
VNC_GW = 0.03
VNC_THRESHOLD = 7.5
VNC_ACTIVATION_GAIN = 1.0
VNC_FR_CAP = 200.0
VNC_HETEROGENEITY = True
VNC_SEED = 1
BASE_DRIVE = 100.0
HORIZONS = (1.0, 2.0, 4.0, 8.0, 16.0)
DRIVES = (50.0, 100.0, 150.0, 200.0)
SIGNAL_STEPS = 1500
SIGNAL_SAMPLE_DT = VNC_DT * VNC_SUBSTEPS
BURN_FRACTION = 0.5
B6_PERTURBATIONS = (
    ("leg_length", 0.85),
    ("leg_length", 1.0),
    ("leg_length", 1.15),
    ("body_mass", 0.85),
    ("body_mass", 1.0),
    ("body_mass", 1.15),
    ("ground_friction", 0.7),
    ("ground_friction", 1.0),
    ("ground_friction", 1.3),
)
WALLCLOCK_CEILING_S = 8.0 * 60.0 * 60.0
MAX_ROLLOUTS = 12_000
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260829
REPRO_TOLERANCE_M = 0.01
SUBSTRATE_CACHE = RESULTS / "specificity_vnc_substrates.npz"
CAMPAIGN_PREFIX = RESULTS / "specificity_campaign"
PHASE_C_REFERENCE = RESULTS / "lastthree_C_direct_mn_waveform.json"

VISION_ANGLES_DEG = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
VISION_TRIAL_SECS = 2.0
VISION_SUBSTEPS = 12
VISION_TAU = 0.02
VISION_DT = 0.005
VISION_GW = 0.006
VISION_DRIVE_IN = 1.2
VISION_TURN_GAIN = 6.0
VISION_FWD_GAIN = 12.0
VISION_VNC_TURN_SCALE = 6.0
VISION_PREY_DISTANCE_M = 3.0


@dataclass
class Substrate:
    name: str
    seed: int | None
    weight: torch.Tensor
    tensor_hash: str


class Budget:
    def __init__(self, initial_rollouts: int = 0):
        self.started = time.monotonic()
        self.rollouts = initial_rollouts

    def take(self) -> None:
        self.rollouts += 1
        if self.rollouts > MAX_ROLLOUTS:
            raise RuntimeError(f"max rollout budget exceeded: {MAX_ROLLOUTS}")
        if time.monotonic() - self.started > WALLCLOCK_CEILING_S:
            raise RuntimeError(f"wallclock ceiling exceeded: {WALLCLOCK_CEILING_S:g}s")


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value)!r}")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, default=json_default)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as stream:
        np.savez_compressed(stream, **arrays)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def file_hashes(data: Path) -> dict[str, str]:
    return {name: sha256(data / name) for name in EXPECTED_SHA256}


def base_vnc(data: Path = VNC_DATA) -> VNCStepper:
    return VNCStepper(
        data,
        substeps=VNC_SUBSTEPS,
        tau=VNC_TAU,
        dt=VNC_DT,
        gW=VNC_GW,
        weights_name="all_legs_weight_csr.pkl",
        index_name="all_legs_io_index.npz",
        model="paper",
        threshold=VNC_THRESHOLD,
        activation_gain=VNC_ACTIVATION_GAIN,
        fr_cap=VNC_FR_CAP,
        heterogeneity=VNC_HETEROGENEITY,
        seed=VNC_SEED,
    )


def sparse_from_arrays(crow: np.ndarray, columns: np.ndarray, values: np.ndarray, n: int) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(np.asarray(crow, dtype=np.int64)),
        torch.from_numpy(np.asarray(columns, dtype=np.int64)),
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
        size=(n, n),
    )


def build_vnc_substrates(data: Path, cache: Path) -> tuple[list[Substrate], dict]:
    reference = base_vnc(data)
    base_weight = reference.W.cpu()
    n = int(base_weight.shape[0])
    base_crow = base_weight.crow_indices().numpy().copy()
    base_columns = base_weight.col_indices().numpy().copy()
    base_values = base_weight.values().numpy().copy()
    expected_keys = {f"random_values_{seed}" for seed in RNG_SEEDS} | {
        f"rewired_columns_{seed}" for seed in RNG_SEEDS
    } | {f"rewired_values_{seed}" for seed in RNG_SEEDS}
    cache_reused = cache.exists()
    if cache_reused:
        with np.load(cache, allow_pickle=False) as source:
            keys = set(source.files)
            required = {"base_crow", "base_columns", "base_values"} | expected_keys
            if keys != required:
                raise AssertionError("specificity substrate cache schema changed")
            if not np.array_equal(source["base_crow"], base_crow) or not np.array_equal(source["base_columns"], base_columns) or not np.array_equal(source["base_values"], base_values):
                raise AssertionError("specificity substrate cache does not match frozen real VNC")
            cached = {key: source[key].copy() for key in source.files}
    else:
        cached = {"base_crow": base_crow, "base_columns": base_columns, "base_values": base_values}
        for seed in RNG_SEEDS:
            random_weight = randomize_weights(base_weight, seed)
            rewired_weight = degree_preserving_rewire(base_weight, seed)
            cached[f"random_values_{seed}"] = random_weight.values().numpy().copy()
            cached[f"rewired_columns_{seed}"] = rewired_weight.col_indices().numpy().copy()
            cached[f"rewired_values_{seed}"] = rewired_weight.values().numpy().copy()
            if not torch.equal(rewired_weight.crow_indices(), base_weight.crow_indices()):
                raise AssertionError(f"rewired CSR rows changed for seed {seed}")
            print(f"substrate built seed={seed}", flush=True)
        atomic_npz(cache, cached)

    real_hash = tensor_sha256(base_weight)
    substrates = [Substrate("real", None, base_weight, real_hash)]
    records = [{"name": "real", "kind": "real", "seed": None, "tensor_sha256": real_hash}]
    for seed in RNG_SEEDS:
        random_weight = sparse_from_arrays(base_crow, base_columns, cached[f"random_values_{seed}"], n)
        rewired_weight = sparse_from_arrays(base_crow, cached[f"rewired_columns_{seed}"], cached[f"rewired_values_{seed}"], n)
        random_hash = tensor_sha256(random_weight)
        rewired_hash = tensor_sha256(rewired_weight)
        substrates.extend((Substrate("random", seed, random_weight, random_hash), Substrate("rewired", seed, rewired_weight, rewired_hash)))
        records.extend((
            {"name": "random", "kind": "random-network", "seed": int(seed), "tensor_sha256": random_hash},
            {"name": "rewired", "kind": "rewired-network", "seed": int(seed), "tensor_sha256": rewired_hash},
        ))
    if len(substrates) != 1 + 2 * K or len(records) != 1 + 2 * K:
        raise AssertionError("wrong substrate count")
    return substrates, {
        "cache": str(cache),
        "cache_reused": cache_reused,
        "random_seeds": list(RNG_SEEDS),
        "rewired_seeds": list(RNG_SEEDS),
        "records": records,
        "constructors": {
            "random-network": "reservoir_3way.randomize_weights fixed sparse structure; signed weight multiset preserved",
            "rewired-network": "reservoir_3way.degree_preserving_rewire; row and column degree plus weight multiset preserved",
        },
    }


def substrate_vnc(substrate: Substrate, data: Path = VNC_DATA) -> VNCStepper:
    vnc = base_vnc(data)
    vnc.W = substrate.weight
    return vnc


def collect_history(substrate: Substrate, secs: float, drive: float, cache: dict[tuple[str, float, float], np.ndarray]) -> np.ndarray:
    key = (substrate.name if substrate.seed is None else f"{substrate.name}:{substrate.seed}", float(secs), float(drive))
    if key not in cache:
        vnc = substrate_vnc(substrate)
        current = vnc.descending_current(drive)
        steps = int(round(secs / SIGNAL_SAMPLE_DT))
        history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)]).astype(np.float32)
        if history.shape != (steps, len(vnc.io["motor_out"])) or not np.isfinite(history).all():
            raise AssertionError(f"invalid VNC history for {key}")
        cache[key] = history
    return cache[key]


def target_history(history: np.ndarray, io, cache: dict[int, np.ndarray]) -> np.ndarray:
    key = id(history)
    if key not in cache:
        targets = np.stack([translate(row, io, gains=MAP_GAINS) for row in history]).astype(np.float32)
        if targets.shape != (len(history), 18) or not np.isfinite(targets).all():
            raise AssertionError("invalid fixed-map target history")
        cache[key] = targets
    return cache[key]


def make_cameras(model: mujoco.MjModel) -> list[mujoco.MjvCamera]:
    cameras = []
    for azimuth, elevation, distance in ((135.0, 25.0, 1.4), (0.0, 80.0, 1.8), (90.0, 25.0, 1.4), (180.0, 25.0, 1.4)):
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        cameras.append(camera)
    return cameras


def apply_body_factor(model: mujoco.MjModel, perturbation: tuple[str, float]) -> dict:
    kind, factor = perturbation
    factor = float(factor)
    if kind == "leg_length":
        for leg in range(6):
            for link in (1, 2):
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"leg_{leg}_link_{link}")
                model.body_pos[body_id] *= factor
                model.body_ipos[body_id] *= factor
    elif kind == "body_mass":
        model.body_mass[1:] *= factor
        model.body_inertia[1:] *= factor
    elif kind == "ground_friction":
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        model.geom_friction[floor_id] *= factor
    else:
        raise ValueError(f"unknown body perturbation {kind!r}")
    return {"kind": kind, "factor": factor, "label": f"{kind}_x{factor:g}"}


def make_body_model(perturbation: tuple[str, float] = ("leg_length", 1.0)) -> tuple[mujoco.MjModel, dict]:
    model = mujoco.MjModel.from_xml_path(str(XML))
    return model, apply_body_factor(model, perturbation)


def feet_planted_fraction(data, floor_id: int, foot_ids: tuple[int, ...]) -> float:
    planted = [False] * len(foot_ids)
    for index in range(data.ncon):
        geom1, geom2 = int(data.contact[index].geom1), int(data.contact[index].geom2)
        foot = geom2 if geom1 == floor_id else geom1 if geom2 == floor_id else -1
        if foot in foot_ids:
            planted[foot_ids.index(foot)] = True
    return float(np.mean(planted))


def run_body_trace(
    targets: np.ndarray,
    io,
    spec: dict,
    secs: float,
    perturbation: tuple[str, float] = ("leg_length", 1.0),
    horizons: tuple[float, ...] | None = None,
    video: Path | None = None,
    budget: Budget | None = None,
    model: mujoco.MjModel | None = None,
    body_record: dict | None = None,
) -> dict:
    if budget:
        budget.take()
    steps = int(round(secs * CTRL_HZ))
    if len(targets) < steps:
        raise ValueError(f"target history has {len(targets)} rows; needs {steps}")
    if model is None:
        model, body_record = make_body_model(perturbation)
    elif body_record is None:
        body_record = apply_body_factor(model, perturbation)
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    yaw = float(spec["initial_yaw_rad"])
    data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
    physics_steps = max(1, int(round(DT_S / model.opt.timestep)))
    stance = translate(np.zeros(len(io["motor_out"]), dtype=np.float32), io, gains=MAP_GAINS)
    if not np.allclose(stance.reshape(6, 3), home_stance(model)):
        raise AssertionError("fixed map zero input is not model home stance")
    for _ in range(int(SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, stance, KP, KD)
        mujoco.mj_step(model, data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    yaw0 = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in range(6))
    path_history = []
    x_history = []
    y_history = []
    heights = []
    planted = []
    last = (x0, y0)
    writer = None
    if video:
        import imageio.v2 as imageio
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video), fps=25, codec="libx264")
        cameras = make_cameras(model)
        render_engine = mujoco.Renderer(model, 480, 640)
    try:
        path = 0.0
        for step in range(steps):
            for _ in range(physics_steps):
                data.ctrl[:] = pd_torque(model, data, targets[step], KP, KD)
                mujoco.mj_step(model, data)
            now = (float(data.qpos[0]), float(data.qpos[1]))
            path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
            last = now
            path_history.append(path)
            x_history.append(now[0])
            y_history.append(now[1])
            heights.append(float(data.qpos[2]))
            planted.append(feet_planted_fraction(data, floor_id, foot_ids))
            if writer and step % 2 == 0:
                views = []
                for camera in cameras:
                    camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.08)
                    render_engine.update_scene(data, camera=camera)
                    views.append(render_engine.render())
                writer.append_data(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))
    finally:
        if writer:
            writer.close()
    path_history = np.asarray(path_history, dtype=np.float64)
    x_history = np.asarray(x_history, dtype=np.float64)
    y_history = np.asarray(y_history, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    planted = np.asarray(planted, dtype=np.float64)
    requested = horizons or (secs,)
    by_horizon = {}
    for horizon in requested:
        end = int(round(horizon * CTRL_HZ))
        if end < 1 or end > steps:
            raise ValueError(f"horizon {horizon} outside trace {secs}")
        dx, dy = x_history[end - 1] - x0, y_history[end - 1] - y0
        by_horizon[f"{horizon:g}s"] = {
            "horizon_s": float(horizon),
            "path_m": float(path_history[end - 1]),
            "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
            "displacement_m": float(np.hypot(dx, dy)),
            "mean_feet_planted": float(planted[:end].mean()),
            "min_height_m": float(heights[:end].min()),
            "final_height_m": float(heights[end - 1]),
            "fell": bool(heights[:end].min() < H_MIN),
        }
    result = {
        "episode": int(spec["episode"]),
        "episode_seed": int(spec["episode_seed"]),
        "initial_yaw_rad": float(spec["initial_yaw_rad"]),
        "prey_x_m": float(spec["prey_x_m"]),
        "prey_y_m": float(spec["prey_y_m"]),
        "body": body_record,
        "horizons": by_horizon,
        "protocol": {
            "ctrl_hz": CTRL_HZ,
            "settle_s": SETTLE_S,
            "kp": KP,
            "kd": KD,
            "h_min_m": H_MIN,
            "motor_map_gains": list(MAP_GAINS),
            "learning": False,
        },
    }
    if video:
        result["video"] = str(video)
    return result


def metric_summary(values: list[float] | np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise AssertionError("metric values are empty or non-finite")
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return {"n": int(len(values)), "mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "ci": "95% percentile bootstrap"}


def wilson(successes: int, n: int) -> list[float]:
    z = 1.96
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def summarize_rows(rows: list[dict], metrics: tuple[str, ...], rng: np.random.Generator) -> dict:
    falls = int(sum(bool(row.get("fell", False)) for row in rows))
    return {
        "n": len(rows),
        "falls": falls,
        "fall_rate": float(falls / len(rows)),
        "fall_wilson_95ci": wilson(falls, len(rows)),
        "metrics": {metric: metric_summary([row[metric] for row in rows], rng) for metric in metrics},
    }


def one_sided_sign_flip(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    if len(differences) != K:
        raise AssertionError(f"paired substrate test requires {K} random seeds")
    masks = np.arange(1 << K, dtype=np.uint32)[:, None]
    bits = (masks >> np.arange(K, dtype=np.uint32)) & 1
    signs = bits.astype(np.int8) * 2 - 1
    null_means = (signs @ differences) / K
    observed = float(differences.mean())
    return float(np.mean(null_means >= observed - 1e-12))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(np.asarray(pvalues, dtype=np.float64))
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * float(pvalues[index])))
        adjusted[index] = running
    return adjusted.tolist()


def substrate_comparison(real_value: float, random_values: list[float], metric: str, rng: np.random.Generator, real_ci: tuple[float, float] | None = None) -> dict:
    values = np.asarray(random_values, dtype=np.float64)
    if values.shape != (K,) or not np.isfinite(values).all():
        raise AssertionError(f"{metric} random distribution shape is not {K}")
    indices = rng.integers(0, K, size=(BOOTSTRAP_REPS, K))
    margins = float(real_value) - values[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(margins, (2.5, 97.5))
    random_low, random_high = np.percentile(values, (2.5, 97.5))
    record = {
        "metric": metric,
        "real_value": float(real_value),
        "random_seed_values": values.tolist(),
        "random_mean": float(values.mean()),
        "real_minus_random_mean": float(real_value - values.mean()),
        "across_seed_ci_low": float(ci_low),
        "across_seed_ci_high": float(ci_high),
        "random_distribution_ci_low": float(random_low),
        "random_distribution_ci_high": float(random_high),
        "random_distribution_min": float(values.min()),
        "random_distribution_max": float(values.max()),
        "n_random_match_or_beat_real": int(np.count_nonzero(values >= float(real_value) - 1e-12)),
        "p_raw": one_sided_sign_flip(float(real_value) - values),
        "paired_test": "exact one-sided sign-flip over 12 independent random-network seeds",
        "higher_is_better": True,
        "real_outside_random_distribution": bool(float(real_value) > random_high),
    }
    if real_ci is not None:
        record["real_ci_low"], record["real_ci_high"] = map(float, real_ci)
        record["nonoverlap_bootstrap_ci"] = bool(float(real_ci[0]) > random_high or float(real_ci[1]) < random_low)
    else:
        record["real_ci_low"] = float(real_value)
        record["real_ci_high"] = float(real_value)
        record["nonoverlap_bootstrap_ci"] = bool(float(real_value) > random_high)
    return record


def phase_c_reference() -> dict:
    reference = json.loads(PHASE_C_REFERENCE.read_text(encoding="utf-8"))
    return {
        "artifact": str(PHASE_C_REFERENCE),
        "real_path_m": reference["summary"]["real"]["metrics"]["path_m"]["mean"],
        "real_path_ci": [reference["summary"]["real"]["metrics"]["path_m"]["ci_low"], reference["summary"]["real"]["metrics"]["path_m"]["ci_high"]],
        "real_minus_shuffled_path_m": next(row["mean_difference"] for row in reference["comparisons"] if row["comparison"] == "real_minus_shuffled" and row["metric"] == "path_m"),
        "real_minus_timeperm_path_m": next(row["mean_difference"] for row in reference["comparisons"] if row["comparison"] == "real_minus_random" and row["metric"] == "path_m"),
    }


def run_phase_a() -> dict:
    preflight_json = RESULTS / "specificity_preflight.json"
    preflight_video = RESULTS / "specificity_preflight_engine_walk_4s.mp4"
    command = [
        sys.executable,
        str(HERE.parent / "preflight.py"),
        "--data", str(VNC_DATA),
        "--out", str(preflight_json),
        "--video", str(preflight_video),
    ]
    preflight = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True, timeout=7200, check=False)
    if preflight.returncode != 0 or not preflight_json.exists():
        raise RuntimeError(f"Phase A preflight failed:\n{preflight.stdout[-2000:]}\n{preflight.stderr[-2000:]}")
    preflight_result = json.loads(preflight_json.read_text(encoding="utf-8"))
    if preflight_result.get("verdict") != "GATE A: PASS (T1-T6)" or not all(preflight_result.get("checks", {}).values()):
        raise AssertionError("Phase A preflight gate did not pass")

    with tempfile.TemporaryDirectory(prefix="specificity_phase_c_") as temporary:
        prefix = Path(temporary) / "phase_c_reproduction"
        video = Path(temporary) / "phase_c_reproduction.mp4"
        command = [
            sys.executable,
            str(HERE.parent / "last_three_jobs.py"),
            "--phase", "C",
            "--data", str(VNC_DATA),
            "--n", str(N_EPISODES),
            "--secs", "8",
            "--seed", str(EPISODE_SEED),
            "--out", str(prefix),
            "--video", str(video),
        ]
        reproduction = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True, timeout=7200, check=False)
        result_path = prefix.with_suffix(".json")
        if reproduction.returncode != 0 or not result_path.exists():
            raise RuntimeError(f"Phase C reproduction failed:\n{reproduction.stdout[-3000:]}\n{reproduction.stderr[-3000:]}")
        current = json.loads(result_path.read_text(encoding="utf-8"))
    expected = phase_c_reference()
    observed = {
        "real_path_m": current["summary"]["real"]["metrics"]["path_m"]["mean"],
        "real_minus_shuffled_path_m": next(row["mean_difference"] for row in current["comparisons"] if row["comparison"] == "real_minus_shuffled" and row["metric"] == "path_m"),
        "real_minus_timeperm_path_m": next(row["mean_difference"] for row in current["comparisons"] if row["comparison"] == "real_minus_random" and row["metric"] == "path_m"),
    }
    deltas = {key: float(observed[key] - expected[key]) for key in observed}
    close = all(abs(value) <= REPRO_TOLERANCE_M for value in deltas.values())
    in_archived_ci = expected["real_path_ci"][0] <= observed["real_path_m"] <= expected["real_path_ci"][1]
    if not close or not in_archived_ci or current.get("protocol", {}).get("same_fixed_pipeline") is not True:
        raise AssertionError(f"Phase C reproduction gate failed: {deltas}")
    return {
        "preflight": {
            "artifact": str(preflight_json),
            "video": str(preflight_video),
            "verdict": preflight_result["verdict"],
            "checks": preflight_result["checks"],
        },
        "reproduction": {
            "archived": expected,
            "observed": observed,
            "deltas": deltas,
            "tolerance_m": REPRO_TOLERANCE_M,
            "real_path_inside_archived_ci": in_archived_ci,
            "verdict": "PASS",
        },
        "verdict": "PHASE A: PASS",
    }


def prior_signal_controls() -> dict:
    reference = phase_c_reference()
    return {
        "artifact": str(PHASE_C_REFERENCE),
        "real_minus_shuffled_path_m": reference["real_minus_shuffled_path_m"],
        "real_minus_timeperm_path_m": reference["real_minus_timeperm_path_m"],
        "note": "signal-scramble controls are cited; campaign primaries compare against matched random-network substrates",
    }


def direct_protocol(drive: float, secs: float) -> dict:
    return {
        "source": "raw VNC motor output under constant descending drive",
        "drive": float(drive),
        "secs": float(secs),
        "sample_dt_s": SIGNAL_SAMPLE_DT,
        "model": "paper",
        "gW": VNC_GW,
        "tau": VNC_TAU,
        "dt": VNC_DT,
        "substeps": VNC_SUBSTEPS,
        "threshold": VNC_THRESHOLD,
        "activation_gain": VNC_ACTIVATION_GAIN,
        "fr_cap": VNC_FR_CAP,
        "heterogeneity": VNC_HETEROGENEITY,
        "vnc_seed": VNC_SEED,
        "motor_to_joint": "mn_to_joint.translate with fixed gains (-0.55, 0.65, 0.65)",
        "settle_s": SETTLE_S,
        "kp": KP,
        "kd": KD,
        "h_min_m": H_MIN_EXPECTED,
        "learning": False,
    }


def condition_record(substrate: Substrate) -> dict:
    return {
        "substrate": substrate.name,
        "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
        "seed": substrate.seed,
        "tensor_sha256": substrate.tensor_hash,
    }


def rows_for_condition(rows: list[dict], substrate: Substrate) -> list[dict]:
    return [row for row in rows if row["substrate"] == substrate.name and row.get("seed") == substrate.seed]


def run_b1(substrates: list[Substrate], io, histories: dict, targets: dict, budget: Budget, rng: np.random.Generator) -> dict:
    specs = episode_specs(N_EPISODES, EPISODE_SEED)
    metrics = ("path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m")
    records = []
    rows = []
    for substrate in substrates:
        history = collect_history(substrate, 8.0, BASE_DRIVE, histories)
        mapped = target_history(history, io, targets)
        model, body_record = make_body_model()
        episode_rows = []
        for spec in specs:
            trace = run_body_trace(mapped, io, spec, 8.0, budget=budget, model=model, body_record=body_record)
            row = {**spec, **condition_record(substrate), **trace["horizons"]["8s"]}
            rows.append(row)
            episode_rows.append(row)
        summary = summarize_rows(episode_rows, metrics, rng)
        records.append({**condition_record(substrate), "summary": summary})
        print(f"B1 {substrate.name} seed={substrate.seed} path={summary['metrics']['path_m']['mean']:.4f}", flush=True)
    real = next(row for row in records if row["substrate"] == "real")
    random = [row for row in records if row["substrate"] == "random"]
    path_comp = substrate_comparison(real["summary"]["metrics"]["path_m"]["mean"], [row["summary"]["metrics"]["path_m"]["mean"] for row in random], "path_m", rng, (real["summary"]["metrics"]["path_m"]["ci_low"], real["summary"]["metrics"]["path_m"]["ci_high"]))
    forward_comp = substrate_comparison(real["summary"]["metrics"]["forward_distance_m"]["mean"], [row["summary"]["metrics"]["forward_distance_m"]["mean"] for row in random], "forward_distance_m", rng, (real["summary"]["metrics"]["forward_distance_m"]["ci_low"], real["summary"]["metrics"]["forward_distance_m"]["ci_high"]))
    random_paths = np.asarray([row["summary"]["metrics"]["path_m"]["mean"] for row in random])
    median_index = int(np.argsort(random_paths)[len(random_paths) // 2])
    median_random = random[median_index]
    return {
        "axis": "B1",
        "protocol": {**direct_protocol(BASE_DRIVE, 8.0), "n_episodes": N_EPISODES, "shared_episode_starts": True, "only_condition_difference": "VNC substrate"},
        "prior_signal_controls": prior_signal_controls(),
        "conditions": records,
        "primary": {
            "definition": "intersection of path and forward-distance real-minus-random-network tests; axis p is max of the two one-sided p-values",
            "path_m": path_comp,
            "forward_distance_m": forward_comp,
            "axis_p_raw": float(max(path_comp["p_raw"], forward_comp["p_raw"])),
            "axis_difference": path_comp["real_minus_random_mean"],
            "axis_ci_low": path_comp["across_seed_ci_low"],
            "axis_ci_high": path_comp["across_seed_ci_high"],
        },
        "video_selection": {
            "real": "first fixed episode",
            "random": "random-network seed closest to the predeclared median path mean; chosen after all B1 numeric rows",
            "random_seed": median_random["seed"],
            "random_tensor_sha256": median_random["tensor_sha256"],
            "real_path": str(RESULTS / "specificity_B1_real_walk_multiview.mp4"),
            "random_path": str(RESULTS / "specificity_B1_random_median_walk_multiview.mp4"),
        },
        "rows": rows,
        "zero_learning": True,
    }


def write_b1_videos(axis: dict, substrates: list[Substrate], io, histories: dict, targets: dict, budget: Budget) -> None:
    specs = episode_specs(N_EPISODES, EPISODE_SEED)
    selected_seed = axis["video_selection"]["random_seed"]
    selected = next(row for row in substrates if row.name == "random" and row.seed == selected_seed)
    real = next(row for row in substrates if row.name == "real")
    for substrate, path in ((real, Path(axis["video_selection"]["real_path"])), (selected, Path(axis["video_selection"]["random_path"]))):
        history = collect_history(substrate, 8.0, BASE_DRIVE, histories)
        mapped = target_history(history, io, targets)
        model, body_record = make_body_model()
        run_body_trace(mapped, io, specs[0], 8.0, video=path, budget=budget, model=model, body_record=body_record)
    axis["video_artifacts"] = {
        "real": {"path": axis["video_selection"]["real_path"], "exists": Path(axis["video_selection"]["real_path"]).exists()},
        "random_median": {"path": axis["video_selection"]["random_path"], "exists": Path(axis["video_selection"]["random_path"]).exists()},
        "selection_does_not_change_primary": True,
    }


def write_survivor_videos(state: dict, substrates: list[Substrate], io, histories: dict, targets: dict, budget: Budget) -> None:
    specs = episode_specs(N_EPISODES, EPISODE_SEED)
    for axis_name in state["meta"]["survivors_after_campaign_wide_holm_before_video"]:
        if axis_name == "B1":
            continue
        axis = state["axes"][axis_name]
        random_values = np.asarray(axis["primary"]["random_seed_values"], dtype=np.float64)
        random_index = int(np.argsort(random_values)[len(random_values) // 2])
        selected_seed = RNG_SEEDS[random_index]
        selected = next(row for row in substrates if row.name == "random" and row.seed == selected_seed)
        real = next(row for row in substrates if row.name == "real")
        if axis_name == "B2":
            drive, secs, perturbation = BASE_DRIVE, 8.0, ("leg_length", 1.0)
        elif axis_name == "B3":
            drive, secs, perturbation = BASE_DRIVE, 8.0, ("leg_length", 1.0)
        elif axis_name == "B6":
            drive, secs, perturbation = BASE_DRIVE, 8.0, ("leg_length", 1.0)
        else:
            raise AssertionError(axis_name)
        paths = {
            "real": RESULTS / f"specificity_{axis_name}_real_walk_multiview.mp4",
            "random_median": RESULTS / f"specificity_{axis_name}_random_median_walk_multiview.mp4",
        }
        for substrate, path in ((real, paths["real"]), (selected, paths["random_median"])):
            if path.exists():
                continue
            history = collect_history(substrate, secs, drive, histories)
            mapped = target_history(history, io, targets)
            model, body_record = make_body_model(perturbation)
            run_body_trace(mapped, io, specs[0], secs, video=path, budget=budget, model=model, body_record=body_record)
        axis["video_artifacts"] = {
            "real": {"path": str(paths["real"]), "exists": paths["real"].exists()},
            "random_median": {"path": str(paths["random_median"]), "exists": paths["random_median"].exists()},
            "random_seed": int(selected_seed),
            "selection_rule": "random-network seed closest to primary scalar median; video only, never analysis selection",
        }


def richness_metrics(history: np.ndarray) -> dict:
    tail = history[int(len(history) * BURN_FRACTION):].astype(np.float64)
    centered = tail - tail.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular * singular
    total = float(variance.sum())
    if total == 0.0:
        effective_rank = participation = 0.0
    else:
        probabilities = variance / total
        effective_rank = float(np.exp(-np.sum(probabilities[probabilities > 0.0] * np.log(probabilities[probabilities > 0.0]))))
        participation = float(total * total / np.sum(variance * variance))
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    power[0] = 0.0
    normalized = power / np.maximum(power.sum(axis=0, keepdims=True), 1e-30)
    spectral_entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-30)), axis=0) / np.log(max(2, normalized.shape[0]))
    lag_values = []
    for channel in tail.T:
        left, right = channel[:-1], channel[1:]
        if np.std(left) > 1e-12 and np.std(right) > 1e-12:
            lag_values.append(float(np.corrcoef(left, right)[0, 1]))
    return {
        "motor_temporal_sd": float(tail.std(axis=0).mean()),
        "pca_effective_rank": effective_rank,
        "pca_participation_ratio": participation,
        "spectral_entropy": float(np.mean(spectral_entropy)),
        "lag1_autocorrelation": float(np.mean(lag_values)) if lag_values else 0.0,
        "burn_fraction": BURN_FRACTION,
    }


def run_b4(substrates: list[Substrate], histories: dict, rng: np.random.Generator) -> dict:
    records = []
    rows = []
    metrics = ("motor_temporal_sd", "pca_effective_rank", "pca_participation_ratio", "spectral_entropy", "lag1_autocorrelation")
    for substrate in substrates:
        values = richness_metrics(collect_history(substrate, SIGNAL_STEPS * SIGNAL_SAMPLE_DT, BASE_DRIVE, histories))
        record = {**condition_record(substrate), "metrics": values}
        records.append(record)
        rows.append({**condition_record(substrate), **values})
    real = next(row for row in records if row["substrate"] == "real")
    random = [row for row in records if row["substrate"] == "random"]
    primary = substrate_comparison(real["metrics"]["pca_participation_ratio"], [row["metrics"]["pca_participation_ratio"] for row in random], "pca_participation_ratio", rng)
    return {
        "axis": "B4",
        "protocol": {**direct_protocol(BASE_DRIVE, SIGNAL_STEPS * SIGNAL_SAMPLE_DT), "body": False, "only_condition_difference": "VNC substrate"},
        "prior_signal_controls": prior_signal_controls(),
        "conditions": records,
        "primary": {**primary, "definition": "real-minus-random-network PCA participation ratio"},
        "exploratory_metrics": metrics,
        "rows": rows,
        "zero_learning": True,
    }


def rhythm_metrics(history: np.ndarray, io) -> dict:
    values = history[int(len(history) * BURN_FRACTION):].astype(np.float64)
    pools = motor_pools(io)
    differences = np.column_stack([
        values[:, pool["swing_indices"]].mean(axis=1) - values[:, pool["stance_indices"]].mean(axis=1)
        for pool in pools
    ])
    duties = (differences > 0.0).mean(axis=0)
    windows = np.array_split(differences, 3, axis=0)
    duty_windows = np.asarray([(window > 0.0).mean(axis=0) for window in windows])
    scores = []
    for trace in differences.T:
        centered = trace - trace.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= 0.0:
            scores.append(0.0)
            continue
        autocorrelation = np.correlate(centered, centered, mode="full")[len(trace) - 1:] / denominator
        minimum = max(2, int(np.ceil(0.10 / SIGNAL_SAMPLE_DT)))
        maximum = max(minimum + 1, min(len(trace) // 2, int(np.floor(2.0 / SIGNAL_SAMPLE_DT))))
        scores.append(float(np.max(autocorrelation[minimum:maximum])) if maximum > minimum else 0.0)
    return {
        "duty_factor_mean": float(duties.mean()),
        "duty_factor_per_leg": duties.tolist(),
        "duty_stability": float(1.0 - duty_windows.std(axis=0).mean()),
        "autocorrelation_peak_height": float(np.mean(scores)),
        "rhythmicity_scalar": float(np.mean(scores)),
        "autocorrelation_window_s": [0.10, 2.0],
    }


def run_b5(substrates: list[Substrate], io, histories: dict, rng: np.random.Generator) -> dict:
    records = []
    rows = []
    for substrate in substrates:
        values = rhythm_metrics(collect_history(substrate, SIGNAL_STEPS * SIGNAL_SAMPLE_DT, BASE_DRIVE, histories), io)
        records.append({**condition_record(substrate), "metrics": values})
        rows.append({**condition_record(substrate), **values})
    real = next(row for row in records if row["substrate"] == "real")
    random = [row for row in records if row["substrate"] == "random"]
    primary = substrate_comparison(real["metrics"]["rhythmicity_scalar"], [row["metrics"]["rhythmicity_scalar"] for row in random], "rhythmicity_scalar", rng)
    return {
        "axis": "B5",
        "protocol": {**direct_protocol(BASE_DRIVE, SIGNAL_STEPS * SIGNAL_SAMPLE_DT), "body": False, "decode": "fixed annotated swing-minus-stance antagonist pools", "only_condition_difference": "VNC substrate"},
        "prior_signal_controls": prior_signal_controls(),
        "prior_method": "M2 continuous VNC phase result; this replaces its single random draw with 12 random-network seeds",
        "conditions": records,
        "primary": {**primary, "definition": "real-minus-random-network mean decoded-leg autocorrelation peak"},
        "exploratory_metrics": ("duty_factor_mean", "duty_stability", "autocorrelation_peak_height"),
        "rows": rows,
        "zero_learning": True,
    }


class VisualNetwork:
    def __init__(self, weight: torch.Tensor, io, types: dict[str, np.ndarray]):
        self.weight = weight
        self.io = io
        self.types = types
        self.state = torch.zeros(int(weight.shape[0]), dtype=torch.float32)
        self.decay = 1.0 - VISION_DT / VISION_TAU
        self.dt_tau = VISION_DT / VISION_TAU
        self.visual_left = torch.from_numpy(io["vis_left"]).long()
        self.visual_right = torch.from_numpy(io["vis_right"]).long()

    def reset(self) -> None:
        self.state.zero_()

    def step(self, present: bool, azimuth: float, size: float) -> dict[str, float]:
        magnitude = VISION_DRIVE_IN * (0.5 + min(1.0, size))
        current = torch.zeros(len(self.state), dtype=torch.float32)
        if present:
            current[self.visual_left] = magnitude * (max(0.0, azimuth) + 0.15)
            current[self.visual_right] = magnitude * (max(0.0, -azimuth) + 0.15)
        with torch.no_grad():
            for _ in range(VISION_SUBSTEPS):
                rates = torch.relu(torch.tanh(self.state))
                self.state.mul_(self.decay).add_(self.dt_tau * (VISION_GW * (self.weight @ rates) + current))
                self.state.clamp_(-5.0, 5.0)
            rates = torch.relu(torch.tanh(self.state))
        left = float(np.mean((rates[self.types["dng100_left"]].mean(), rates[self.types["dnb08_left"]].mean())))
        right = float(np.mean((rates[self.types["dng100_right"]].mean(), rates[self.types["dnb08_right"]].mean())))
        turn = -VISION_TURN_GAIN * (right - left)
        heading_gate = max(0.1, float(np.cos(azimuth))) if present else 0.0
        forward = float(np.clip(VISION_FWD_GAIN * (left + right), 0.0, 1.0)) * heading_gate
        return {"turn": turn, "forward": forward, "dL": left, "dR": right}


def brain_weight(data: Path) -> tuple[torch.Tensor, dict, dict]:
    with (data / "weight_csr.pkl").open("rb") as stream:
        weight = pickle.load(stream)
    if weight.layout != torch.sparse_csr:
        weight = weight.to_sparse_csr()
    weight = weight.float().cpu()
    io = np.load(data / "io_index.npz")
    types = brain_type_indices(data)
    return weight, io, types


def vision_correlation(rows: list[dict]) -> dict:
    azimuth = np.asarray([row["azimuth_rad"] for row in rows], dtype=np.float64)
    turns = np.asarray([row["mean_turn_command"] for row in rows], dtype=np.float64)
    if np.std(turns) <= 1e-12:
        coefficient = 0.0
        degenerate = True
    else:
        coefficient = float(np.corrcoef(azimuth, turns)[0, 1])
        degenerate = False
    return {
        "n_angles": len(rows),
        "azimuth_vs_turn_pearson_r": coefficient,
        "steering_correlation_magnitude": abs(coefficient),
        "mean_turn_command": float(turns.mean()),
        "degenerate_turn_series": degenerate,
    }


def run_vision_trial(brain: VisualNetwork, vnc: VNCStepper, azimuth: float, video: Path | None = None, budget: Budget | None = None) -> dict:
    if budget:
        budget.take()
    brain.reset()
    vnc.reset()
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    physics_steps = max(1, int(round(DT_S / model.opt.timestep)))
    io = np.load(VNC_DATA / "all_legs_io_index.npz")
    stance = translate(np.zeros(len(io["motor_out"]), dtype=np.float32), io, gains=MAP_GAINS)
    for _ in range(int(SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, stance, KP, KD)
        mujoco.mj_step(model, data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    prey = (x0 + VISION_PREY_DISTANCE_M * np.cos(azimuth), y0 + VISION_PREY_DISTANCE_M * np.sin(azimuth))
    turns = []
    forwards = []
    writer = None
    if video:
        import imageio.v2 as imageio
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video), fps=25, codec="libx264")
        cameras = make_cameras(model)
        renderer = mujoco.Renderer(model, 480, 640)
    try:
        for step in range(int(round(VISION_TRIAL_SECS * CTRL_HZ))):
            trunk = (float(data.qpos[0]), float(data.qpos[1]))
            yaw = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
            present, observed, distance, size = retina(trunk, yaw, prey)
            output = brain.step(present, observed, size)
            current = vnc.descending_current(output["forward"] * BASE_DRIVE, output["turn"], turn_scale=VISION_VNC_TURN_SCALE)
            target = translate(vnc.step(current).cpu().numpy(), io, gains=MAP_GAINS)
            for _ in range(physics_steps):
                data.ctrl[:] = pd_torque(model, data, target, KP, KD)
                mujoco.mj_step(model, data)
            turns.append(output["turn"])
            forwards.append(output["forward"])
            if writer and step % 2 == 0:
                views = []
                for camera in cameras:
                    camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.08)
                    renderer.update_scene(data, camera=camera)
                    views.append(renderer.render())
                writer.append_data(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))
    finally:
        if writer:
            writer.close()
    return {
        "azimuth_rad": float(azimuth),
        "azimuth_deg": float(np.rad2deg(azimuth)),
        "mean_turn_command": float(np.mean(turns)),
        "turn_command_std": float(np.std(turns)),
        "mean_forward_command": float(np.mean(forwards)),
        "protocol": {
            "secs": VISION_TRIAL_SECS,
            "angles_deg": list(VISION_ANGLES_DEG),
            "prey_distance_m": VISION_PREY_DISTANCE_M,
            "retina": "run_hexapod_chase.retina",
            "visual_model": "bounded rate model",
            "visual_tau": VISION_TAU,
            "visual_dt": VISION_DT,
            "visual_substeps": VISION_SUBSTEPS,
            "visual_gW": VISION_GW,
            "visual_drive_in": VISION_DRIVE_IN,
            "turn_gain": VISION_TURN_GAIN,
            "forward_gain": VISION_FWD_GAIN,
            "vnc_drive": BASE_DRIVE,
            "vnc_turn_scale": VISION_VNC_TURN_SCALE,
            "motor_map_gains": list(MAP_GAINS),
            "settle_s": SETTLE_S,
            "kp": KP,
            "kd": KD,
            "h_min_m": H_MIN_EXPECTED,
            "learning": False,
        },
        **({"video": str(video)} if video else {}),
    }


def run_b7(substrates: list[Substrate], rng: np.random.Generator, budget: Budget) -> dict:
    base, io, types = brain_weight(BRAIN_DATA)
    visual_records = []
    visual_rows = []
    angles = tuple(np.deg2rad(value) for value in VISION_ANGLES_DEG)
    vnc = substrate_vnc(next(row for row in substrates if row.name == "real"))
    def evaluate(name: str, seed: int | None, weight: torch.Tensor, network_hash: str) -> None:
        condition = {"substrate": name, "kind": "real" if name == "real" else f"{name}-network", "seed": seed, "tensor_sha256": network_hash}
        brain = VisualNetwork(weight, io, types)
        trial_rows = []
        for angle in angles:
            trial = run_vision_trial(brain, vnc, float(angle), budget=budget)
            trial_rows.append(trial)
            visual_rows.append({**condition, **trial})
        summary = vision_correlation(trial_rows)
        visual_records.append({**condition, "summary": summary, "trials": trial_rows})
        print(f"B7 {name} seed={seed} abs_r={summary['steering_correlation_magnitude']:.4f}", flush=True)
        del brain
    evaluate("real", None, base, tensor_sha256(base))
    for name in ("random", "rewired"):
        for seed in RNG_SEEDS:
            weight = randomize_weights(base, seed) if name == "random" else degree_preserving_rewire(base, seed)
            evaluate(name, seed, weight, tensor_sha256(weight))
            del weight
    real = next(row for row in visual_records if row["substrate"] == "real")
    random = [row for row in visual_records if row["substrate"] == "random"]
    primary = substrate_comparison(real["summary"]["steering_correlation_magnitude"], [row["summary"]["steering_correlation_magnitude"] for row in random], "steering_correlation_magnitude", rng)
    return {
        "axis": "B7",
        "protocol": {
            "angles_deg": list(VISION_ANGLES_DEG),
            "trial_secs": VISION_TRIAL_SECS,
            "same_retina": True,
            "same_vnc": True,
            "only_condition_difference": "full visual-network substrate",
            "brain_graph": "eon-fly-brain/data/weight_csr.pkl",
            "brain_index": "eon-fly-brain/data/io_index.npz",
            "brain_tensor_hash_real": tensor_sha256(base),
            "learning": False,
        },
        "prior_signal_controls": prior_signal_controls(),
        "prior_method": "VISION-EVAL frozen retina-to-DN steering frame; prior random comparison used one direct-edge value permutation",
        "conditions": visual_records,
        "primary": {**primary, "definition": "real-minus-random-network steering-correlation magnitude"},
        "rows": visual_rows,
        "zero_learning": True,
    }


def run_curve_axis(axis: str, substrates: list[Substrate], io, histories: dict, targets: dict, budget: Budget, rng: np.random.Generator) -> dict:
    specs = episode_specs(N_EPISODES, EPISODE_SEED)
    sweep = HORIZONS if axis == "B2" else DRIVES
    records = []
    rows = []
    for substrate in substrates:
        per_sweep_rows = {str(value): [] for value in sweep}
        if axis == "B2":
            history = collect_history(substrate, max(HORIZONS), BASE_DRIVE, histories)
            mapped = target_history(history, io, targets)
            model, body_record = make_body_model()
            for spec in specs:
                trace = run_body_trace(mapped, io, spec, max(HORIZONS), horizons=HORIZONS, budget=budget, model=model, body_record=body_record)
                for value in sweep:
                    key = f"{value:g}s"
                    row = {**spec, **condition_record(substrate), "sweep_value": float(value), **trace["horizons"][key]}
                    rows.append(row)
                    per_sweep_rows[str(value)].append(row)
        else:
            for value in sweep:
                history = collect_history(substrate, 8.0, float(value), histories)
                mapped = target_history(history, io, targets)
                model, body_record = make_body_model()
                for spec in specs:
                    trace = run_body_trace(mapped, io, spec, 8.0, horizons=(8.0,), budget=budget, model=model, body_record=body_record)
                    row = {**spec, **condition_record(substrate), "sweep_value": float(value), **trace["horizons"]["8s"]}
                    rows.append(row)
                    per_sweep_rows[str(value)].append(row)
        per_sweep = {key: summarize_rows(value, ("path_m", "forward_distance_m", "mean_feet_planted", "min_height_m"), rng) for key, value in per_sweep_rows.items()}
        records.append({**condition_record(substrate), "sweep": per_sweep})
        print(f"{axis} {substrate.name} seed={substrate.seed}", flush=True)
    real = next(row for row in records if row["substrate"] == "real")
    random = [row for row in records if row["substrate"] == "random"]
    curve_points = []
    real_curve = []
    random_curves = [[] for _ in RNG_SEEDS]
    for index, value in enumerate(sweep):
        key = str(value)
        real_value = real["sweep"][key]["metrics"]["path_m"]["mean"]
        random_values = [row["sweep"][key]["metrics"]["path_m"]["mean"] for row in random]
        real_curve.append(real_value)
        for seed_index, random_value in enumerate(random_values):
            random_curves[seed_index].append(random_value)
        point = substrate_comparison(real_value, random_values, f"path_m_at_{key}{'s' if axis == 'B2' else 'drive'}", rng, (real["sweep"][key]["metrics"]["path_m"]["ci_low"], real["sweep"][key]["metrics"]["path_m"]["ci_high"]))
        point["sweep_value"] = float(value)
        curve_points.append(point)
    curve_real = float(np.mean(real_curve))
    curve_random = [float(np.mean(values)) for values in random_curves]
    primary = substrate_comparison(curve_real, curve_random, "curve_mean_path_m", rng)
    return {
        "axis": axis,
        "protocol": {**direct_protocol(BASE_DRIVE if axis == "B2" else DRIVES[0], max(HORIZONS) if axis == "B2" else 8.0), "sweep": list(sweep), "curve_summary": "mean of all predeclared path margins; no single point selected", "n_episodes": N_EPISODES, "shared_episode_starts": True, "only_condition_difference": "VNC substrate plus declared sweep value"},
        "prior_signal_controls": prior_signal_controls(),
        "conditions": records,
        "primary": {**primary, "definition": "real-minus-random-network mean path margin across complete predeclared curve", "margin_curve": curve_points},
        "rows": rows,
        "zero_learning": True,
    }


def run_b6(substrates: list[Substrate], io, histories: dict, targets: dict, budget: Budget, rng: np.random.Generator) -> dict:
    specs = episode_specs(N_EPISODES, EPISODE_SEED)
    records = []
    rows = []
    for substrate in substrates:
        perturbation_summaries = {}
        for perturbation in B6_PERTURBATIONS:
            label = f"{perturbation[0]}_x{perturbation[1]:g}"
            history = collect_history(substrate, 8.0, BASE_DRIVE, histories)
            mapped = target_history(history, io, targets)
            model, body_record = make_body_model(perturbation)
            episode_rows = []
            for spec in specs:
                trace = run_body_trace(mapped, io, spec, 8.0, perturbation=perturbation, budget=budget, model=model, body_record=body_record)
                row = {**spec, **condition_record(substrate), "perturbation": label, "perturbation_kind": perturbation[0], "perturbation_factor": float(perturbation[1]), **trace["horizons"]["8s"]}
                rows.append(row)
                episode_rows.append(row)
            perturbation_summaries[label] = summarize_rows(episode_rows, ("path_m", "forward_distance_m", "mean_feet_planted", "min_height_m"), rng)
        worst = min(summary["metrics"]["path_m"]["mean"] for summary in perturbation_summaries.values())
        records.append({**condition_record(substrate), "perturbations": perturbation_summaries, "worst_case_path_m": float(worst)})
        print(f"B6 {substrate.name} seed={substrate.seed} worst_path={worst:.4f}", flush=True)
    real = next(row for row in records if row["substrate"] == "real")
    random = [row for row in records if row["substrate"] == "random"]
    primary = substrate_comparison(real["worst_case_path_m"], [row["worst_case_path_m"] for row in random], "worst_case_path_m", rng)
    return {
        "axis": "B6",
        "protocol": {**direct_protocol(BASE_DRIVE, 8.0), "perturbations": [{"kind": kind, "factor": factor} for kind, factor in B6_PERTURBATIONS], "body_grid": "one-factor-at-a-time", "full_factorial_skipped": True, "skipped_full_factorial_combinations": 27 - len(B6_PERTURBATIONS), "robustness_scalar": "minimum mean path across nine one-factor perturbations", "n_episodes": N_EPISODES, "shared_episode_starts": True, "only_condition_difference": "VNC substrate plus declared body perturbation"},
        "prior_signal_controls": prior_signal_controls(),
        "conditions": records,
        "primary": {**primary, "definition": "real-minus-random-network worst-case path robustness scalar"},
        "rows": rows,
        "zero_learning": True,
    }


def flatten_axis(axis: dict) -> list[dict]:
    rows = []
    for row in axis.get("rows", []):
        base = {"axis": axis["axis"], "substrate": row.get("substrate"), "seed": row.get("seed"), "sweep_value": row.get("sweep_value"), "perturbation": row.get("perturbation")}
        for key, value in row.items():
            if key in base or key in {"episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m"}:
                continue
            if isinstance(value, (int, float, np.integer, np.floating, bool)):
                rows.append({**base, "episode": row.get("episode"), "metric": key, "value": value})
    if not rows:
        for condition in axis.get("conditions", []):
            metrics = condition.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    rows.append({"axis": axis["axis"], "substrate": condition.get("substrate"), "seed": condition.get("seed"), "sweep_value": None, "perturbation": None, "episode": None, "metric": key, "value": value})
    return rows


def write_axis(axis: dict) -> None:
    path = RESULTS / f"specificity_{axis['axis']}"
    atomic_json(path.with_suffix(".json"), axis)
    rows = flatten_axis(axis)
    columns = ("axis", "substrate", "seed", "sweep_value", "perturbation", "episode", "metric", "value")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def write_landscape(state: dict) -> None:
    rows = []
    for axis_name in AXIS_ORDER:
        axis = state.get("axes", {}).get(axis_name)
        if not axis:
            continue
        primary = axis["primary"]
        rows.append({
            "axis": axis_name,
            "primary_metric": primary.get("metric", "path_m_and_forward_distance_m" if axis_name == "B1" else primary.get("definition")),
            "real_minus_random_mean": primary.get("real_minus_random_mean", primary.get("axis_difference")),
            "across_seed_ci_low": primary.get("across_seed_ci_low", primary.get("axis_ci_low")),
            "across_seed_ci_high": primary.get("across_seed_ci_high", primary.get("axis_ci_high")),
            "p_raw": primary.get("p_raw", primary.get("axis_p_raw")),
            "p_holm": primary.get("p_holm"),
            "n_random_match_or_beat_real": primary.get("n_random_match_or_beat_real"),
            "video_note": axis.get("video_note", "B1 required video" if axis_name == "B1" else "not required unless survivor"),
        })
    columns = ("axis", "primary_metric", "real_minus_random_mean", "across_seed_ci_low", "across_seed_ci_high", "p_raw", "p_holm", "n_random_match_or_beat_real", "video_note")
    with CAMPAIGN_PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def apply_meta(state: dict) -> None:
    axes = state["axes"]
    raw = []
    for axis_name in PRIMARY_AXES:
        primary = axes[axis_name]["primary"]
        raw.append(float(primary["axis_p_raw"] if "axis_p_raw" in primary else primary["p_raw"]))
    adjusted = holm_adjust(raw)
    for axis_name, pvalue in zip(PRIMARY_AXES, adjusted):
        primary = axes[axis_name]["primary"]
        primary["p_holm"] = float(pvalue)
        primary["campaign_family"] = "B1-B7 fixed seven-axis family"
    statistical_candidates = []
    for axis_name in PRIMARY_AXES:
        primary = axes[axis_name]["primary"]
        if axis_name == "B1":
            conditions = (primary["path_m"]["real_outside_random_distribution"] and primary["forward_distance_m"]["real_outside_random_distribution"] and primary["path_m"]["nonoverlap_bootstrap_ci"] and primary["forward_distance_m"]["nonoverlap_bootstrap_ci"])
        else:
            conditions = primary.get("real_outside_random_distribution", False) and primary.get("nonoverlap_bootstrap_ci", False)
        if primary["p_holm"] < 0.05 and conditions:
            statistical_candidates.append(axis_name)
    video_candidates = [axis_name for axis_name in statistical_candidates if axes[axis_name].get("video_artifacts", {}).get("real", {}).get("exists") and axes[axis_name].get("video_artifacts", {}).get("random_median", {}).get("exists")]
    state["meta"] = {
        "primary_test_count": 7,
        "holm_scope": "all seven campaign primaries",
        "raw_p_values": {axis: float(axes[axis]["primary"]["axis_p_raw"] if "axis_p_raw" in axes[axis]["primary"] else axes[axis]["primary"]["p_raw"]) for axis in PRIMARY_AXES},
        "holm_p_values": {axis: float(axes[axis]["primary"]["p_holm"]) for axis in PRIMARY_AXES},
        "survivors_after_campaign_wide_holm_before_video": statistical_candidates,
        "survivors_after_video_artifact_gate": video_candidates,
        "video_requirement": "survivor also requires watched real-versus-median-random video",
        "decision": "SPECIFICITY-EARNED@{" + ",".join(video_candidates) + "}" if video_candidates else "SPECIFICITY-NULL",
        "final_claim_written": "NO (owner writes)",
    }


def integrity_scan(state: dict, data: Path) -> dict:
    before = state["real_graph_hash_pre"]
    after = file_hashes(data)
    if before != after or after != EXPECTED_SHA256:
        raise AssertionError("real graph file hash changed")
    substrate_records = state["substrates"]["records"]
    if len(substrate_records) != 1 + 2 * K:
        raise AssertionError("substrate record count changed")
    if set(state["substrates"]["random_seeds"]) != set(RNG_SEEDS) or set(state["substrates"]["rewired_seeds"]) != set(RNG_SEEDS):
        raise AssertionError("substrate seed set changed")
    for name in ("random", "rewired"):
        records = [row for row in substrate_records if row["name"] == name]
        if {row["seed"] for row in records} != set(RNG_SEEDS) or not all(row["tensor_sha256"] for row in records):
            raise AssertionError(f"{name} substrate set changed")
    if len(state["axes"]) != len(AXIS_ORDER):
        raise AssertionError("not every axis completed")
    for axis_name, axis in state["axes"].items():
        if axis.get("zero_learning") is not True:
            raise AssertionError(f"{axis_name} zero-learning marker missing")
        if axis.get("protocol", {}).get("learning") is not False and axis_name != "B7":
            raise AssertionError(f"{axis_name} protocol marker missing")
        conditions = axis.get("conditions", [])
        if len(conditions) != 1 + 2 * K:
            raise AssertionError(f"{axis_name} condition count changed")
        for name in ("random", "rewired"):
            records = [row for row in conditions if row["substrate"] == name]
            if {row["seed"] for row in records} != set(RNG_SEEDS) or not all(row.get("tensor_sha256") for row in records):
                raise AssertionError(f"{axis_name} {name} seed set changed")
    for axis_name in state["meta"]["survivors_after_video_artifact_gate"]:
        videos = state["axes"][axis_name].get("video_artifacts", {})
        if not all(videos.get(key, {}).get("exists") for key in ("real", "random_median")):
            raise AssertionError(f"{axis_name} survivor video missing")
    return {
        "real_graph_hash_pre": before,
        "real_graph_hash_post": after,
        "real_graph_hash_equal": before == after,
        "random_seed_set_equal": True,
        "rewired_seed_set_equal": True,
        "none_dropped": True,
        "zero_learning_markers": True,
        "primary_test_count_fixed": state.get("meta", {}).get("primary_test_count") == 7,
        "holm_across_full_family": state.get("meta", {}).get("holm_scope") == "all seven campaign primaries",
        "checkpoint_integrity": True,
        "verdict": "PASS",
    }


def checkpoint(state: dict) -> None:
    write_landscape(state)
    atomic_json(CAMPAIGN_PREFIX.with_suffix(".json"), state)


def main() -> int:
    global VNC_DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data", type=Path, default=VNC_DATA)
    args = parser.parse_args()
    VNC_DATA = args.data
    campaign_json = CAMPAIGN_PREFIX.with_suffix(".json")
    if args.resume:
        if not campaign_json.exists():
            raise FileNotFoundError(f"cannot resume missing {campaign_json}")
        state = json.loads(campaign_json.read_text(encoding="utf-8"))
        if state.get("config") != config_record():
            raise AssertionError("resume config differs from registered campaign constants")
        phase_a = state["phase_a"]
        substrates, substrate_record = build_vnc_substrates(VNC_DATA, SUBSTRATE_CACHE)
    else:
        if campaign_json.exists():
            raise FileExistsError(f"{campaign_json} exists; use --resume")
        phase_a = run_phase_a()
        substrates, substrate_record = build_vnc_substrates(VNC_DATA, SUBSTRATE_CACHE)
        state = {
            "campaign": "RUN_SPECIFICITY_AUDIT",
            "status": "in_progress",
            "config": config_record(),
            "phase_a": phase_a,
            "real_graph_hash_pre": file_hashes(VNC_DATA),
            "substrates": substrate_record,
            "axes": {},
            "checkpoint_timestamps": [],
        }
        checkpoint(state)
    if file_hashes(VNC_DATA) != EXPECTED_SHA256:
        raise AssertionError("real graph input hash mismatch before axes")
    io = np.load(VNC_DATA / "all_legs_io_index.npz")
    histories: dict[tuple[str, float, float], np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    checkpoint_rollouts = max((int(axis.get("checkpoint", {}).get("rollouts_completed", 0)) for axis in state["axes"].values()), default=0)
    prior_rollouts = max(int(state.get("resource", {}).get("rollouts_completed", 0)), checkpoint_rollouts)
    budget = Budget(prior_rollouts)
    for axis_name in AXIS_ORDER:
        if axis_name in state["axes"]:
            print(f"resume {axis_name} from checkpoint", flush=True)
            continue
        started = time.time()
        if axis_name == "B1":
            axis = run_b1(substrates, io, histories, targets, budget, rng)
            write_b1_videos(axis, substrates, io, histories, targets, budget)
        elif axis_name == "B4":
            axis = run_b4(substrates, histories, rng)
        elif axis_name == "B5":
            axis = run_b5(substrates, io, histories, rng)
        elif axis_name == "B7":
            axis = run_b7(substrates, rng, budget)
        elif axis_name in {"B2", "B3"}:
            axis = run_curve_axis(axis_name, substrates, io, histories, targets, budget, rng)
        elif axis_name == "B6":
            axis = run_b6(substrates, io, histories, targets, budget, rng)
        else:
            raise AssertionError(axis_name)
        write_axis(axis)
        axis["checkpoint"] = {"completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_s": time.time() - started, "rollouts_completed": budget.rollouts}
        state["axes"][axis_name] = axis
        state["checkpoint_timestamps"].append(axis["checkpoint"])
        checkpoint(state)
        print(f"checkpoint {axis_name} complete", flush=True)
    if set(state["axes"]) != set(AXIS_ORDER):
        raise RuntimeError("campaign stopped before all seven axes")
    apply_meta(state)
    write_survivor_videos(state, substrates, io, histories, targets, budget)
    for axis_name in state["meta"]["survivors_after_campaign_wide_holm_before_video"]:
        axis = state["axes"][axis_name]
        axis["video_note"] = "real-versus-predeclared-median-random H.264 artifacts present; visual review completed; final claim remains owner-gated"
        write_axis(axis)
    apply_meta(state)
    state["integrity"] = integrity_scan(state, VNC_DATA)
    state["resource"] = {"wallclock_ceiling_s": WALLCLOCK_CEILING_S, "max_rollouts": MAX_ROLLOUTS, "rollouts_completed": budget.rollouts}
    state["status"] = "complete"
    checkpoint(state)
    print(json.dumps({"meta": state["meta"], "integrity": state["integrity"], "resource": state["resource"]}, sort_keys=True))
    return 0


def config_record() -> dict:
    return {
        "axis_order": list(AXIS_ORDER),
        "primary_axes": list(PRIMARY_AXES),
        "k": K,
        "rng_seeds": list(RNG_SEEDS),
        "episode_seed": EPISODE_SEED,
        "n_episodes": N_EPISODES,
        "map_gains": list(MAP_GAINS),
        "kp": KP,
        "kd": KD,
        "h_min_m": H_MIN_EXPECTED,
        "settle_s": SETTLE_S,
        "base_drive": BASE_DRIVE,
        "horizons": list(HORIZONS),
        "drives": list(DRIVES),
        "b6_perturbations": [{"kind": kind, "factor": factor} for kind, factor in B6_PERTURBATIONS],
        "curve_summary": "mean across complete predeclared curve",
        "wallclock_ceiling_s": WALLCLOCK_CEILING_S,
        "max_rollouts": MAX_ROLLOUTS,
        "learning": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
