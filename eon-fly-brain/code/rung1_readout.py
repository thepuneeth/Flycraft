"""Rung 1: minimal-training, memoryless readout moat test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
DATA = HERE.parent.parent / "data"
sys.path.insert(0, str(HERE))
from analyze_control_study import (  # noqa: E402
    bootstrap_ci,
    exact_sign_flip,
    holm_adjust,
    paired_bootstrap_ci,
    wilson_interval,
)
from collect_reservoir_states import collect, make_stepper, population_rates  # noqa: E402
from hexapod_gait import H_MIN, XML, HexapodGait, home_stance, pd_torque  # noqa: E402
from mn_to_joint import translate  # noqa: E402
from preflight import EXPECTED_SHA256, sha256  # noqa: E402
from reservoir_3way import degree_preserving_rewire, randomize_weights, tensor_sha256  # noqa: E402
from run_hexapod_chase import retina  # noqa: E402
from run_reservoir_walk import FrozenBrain  # noqa: E402
from run_stepwalk import feet_planted_fraction  # noqa: E402
from gait_engine import coxa_phase_signal  # noqa: E402
from probe_vnc_phase import phase_metrics  # noqa: E402
from stepwalk_controls import episode_specs  # noqa: E402


CONDITIONS = ("connectome", "random", "rewired", "static-input")
DYNAMIC_CONDITIONS = CONDITIONS[:3]
WEIGHTS_NAME = "all_legs_weight_csr.pkl"
INDEX_NAME = "all_legs_io_index.npz"
CTRL_HZ = 50.0
DT_S = 1.0 / CTRL_HZ
VNC_DRIVE = 100.0
VNC_TURN_SCALE = 6.0
BURN_STEPS = 3000
RECORD_STEPS = 3000
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
HIDDEN_UNITS = 64
EPOCHS = 80
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
TRAIN_SEED = 20260829
RANDOM_WEIGHT_SEED = 20260829
REWIRE_SEED = 20260829
N_EPISODES = 20
EPISODE_SEED = 20260829
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260829
KP = 3.0
KD = 0.05
SETTLE_S = 0.7
STANCE = home_stance().reshape(18).astype(np.float32)

PRE_REGISTERED_NULL = (
    "connectome does not beat random on the body walking metrics under identical training "
    "-> RUNG1-MOAT-NULL: the trained readout learns the gait regardless of substrate; "
    "the connectome is not a privileged locomotor substrate even under minimal training"
)
PRE_REGISTERED_POSITIVE = (
    "connectome walks a stable tripod, beats random and rewired on coherence and feet-planted "
    "with Holm p<0.05 and non-overlapping bootstrap CIs, static-input does not walk, and the "
    "watched connectome video shows a genuine tripod -> RUNG1-MOAT-EARNED"
)


def prefix_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def condition_stepper(data: Path, condition: str):
    if condition not in DYNAMIC_CONDITIONS:
        raise ValueError(f"no graph stepper for {condition!r}")
    vnc = make_stepper(data, WEIGHTS_NAME, INDEX_NAME)
    if condition == "random":
        vnc.W = randomize_weights(vnc.W, RANDOM_WEIGHT_SEED).to(vnc.device)
    elif condition == "rewired":
        vnc.W = degree_preserving_rewire(vnc.W, REWIRE_SEED).to(vnc.device)
    return vnc


def graph_file_record(data: Path) -> dict:
    hashes = {name: sha256(data / name) for name in EXPECTED_SHA256}
    return {"sha256": hashes, "hash_equal": hashes == EXPECTED_SHA256}


def graph_tensor_record(data: Path) -> dict[str, str]:
    result = {}
    for condition in DYNAMIC_CONDITIONS:
        vnc = condition_stepper(data, condition)
        result[condition] = tensor_sha256(vnc.W)
    return result


def clean_targets(steps: int) -> np.ndarray:
    gait = HexapodGait(stance=None)
    target = np.stack([gait.targets(0.8, 0.0, DT_S) for _ in range(steps)]).astype(np.float32)
    assert target.shape == (steps, 18)
    assert np.isfinite(target).all()
    return target


def split_bounds(n: int) -> dict[str, tuple[int, int]]:
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VAL_FRACTION))
    if not 0 < train_end < val_end < n:
        raise AssertionError(f"invalid split for {n} rows")
    return {"train": (0, train_end), "validation": (train_end, val_end), "test": (val_end, n)}


def standardize(raw: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = raw[:train_end].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = raw[:train_end].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-7] = 1.0
    normalized = ((raw - mean) / scale).astype(np.float32)
    assert np.isfinite(normalized).all()
    return normalized, mean, scale


def run_t1(data: Path, prefix: Path) -> None:
    target = clean_targets(RECORD_STEPS)
    bounds = split_bounds(RECORD_STEPS)
    train_end = bounds["train"][1]
    real_vnc = condition_stepper(data, "connectome")
    real_before = tensor_sha256(real_vnc.W)
    real_full = collect(real_vnc, VNC_DRIVE, BURN_STEPS, RECORD_STEPS)
    active = np.flatnonzero(real_full[:train_end].std(axis=0) > 1e-7).astype(np.int64)
    if len(active) == 0:
        raise AssertionError("connectome produced no varying state columns")

    raw: dict[str, np.ndarray] = {"connectome": real_full[:, active]}
    graph_before = {"connectome": real_before}
    for condition in ("random", "rewired"):
        vnc = condition_stepper(data, condition)
        graph_before[condition] = tensor_sha256(vnc.W)
        raw[condition] = collect(vnc, VNC_DRIVE, BURN_STEPS, RECORD_STEPS)[:, active]
    static_state = real_full[:train_end].mean(axis=0)[active].astype(np.float32)
    raw["static-input"] = np.repeat(static_state[None, :], RECORD_STEPS, axis=0)

    normalized, means, scales = {}, {}, {}
    for condition in CONDITIONS:
        normalized[condition], means[condition], scales[condition] = standardize(raw[condition], train_end)
        assert normalized[condition].shape == (RECORD_STEPS, len(active))

    train_indices = np.arange(*bounds["train"])
    validation_indices = np.arange(*bounds["validation"])
    test_indices = np.arange(*bounds["test"])
    assert not np.intersect1d(train_indices, validation_indices).size
    assert not np.intersect1d(train_indices, test_indices).size
    assert not np.intersect1d(validation_indices, test_indices).size
    assert np.array_equal(target.shape, (RECORD_STEPS, 18))
    assert all(value.shape == normalized["connectome"].shape for value in normalized.values())

    dataset_path = prefix_path(prefix, "_bc_dataset.npz")
    arrays = {
        "target": target,
        "active_indices": active,
        "train_end": np.asarray(train_end),
        "validation_end": np.asarray(bounds["validation"][1]),
    }
    for condition in CONDITIONS:
        arrays[f"x_{condition}"] = normalized[condition]
        arrays[f"train_mean_{condition}"] = means[condition]
        arrays[f"train_scale_{condition}"] = scales[condition]
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **arrays)

    rows = []
    stats = {}
    for condition in CONDITIONS:
        row = {
            "condition": condition,
            "rows": RECORD_STEPS,
            "input_dim": len(active),
            "train_mean_abs": float(np.abs(means[condition]).mean()),
            "train_scale_mean": float(scales[condition].mean()),
            "train_scale_nonunit": int(np.count_nonzero(scales[condition] != 1.0)),
            "train_x_mean_abs": float(np.abs(normalized[condition][:train_end].mean(axis=0)).mean()),
            "train_x_std_mean": float(normalized[condition][:train_end].std(axis=0).mean()),
        }
        rows.append(row)
        stats[condition] = row
    setup = {
        "target_source": "HexapodGait clean-clock target at forward=.8, turn=0, 50 Hz",
        "target_shape": list(target.shape),
        "target_sha256": array_hash(target),
        "input_source": "collect_reservoir_states.collect population rates after fixed tonic drive",
        "vnc_protocol": {
            "drive": VNC_DRIVE,
            "burn_steps": BURN_STEPS,
            "record_steps": RECORD_STEPS,
            "sample_dt_s": DT_S,
            "model": "paper",
            "tau": 0.02,
            "dt": 0.005,
            "substeps": 4,
            "gW": 0.03,
            "threshold": 7.5,
            "activation_gain": 1.0,
            "fr_cap": 200.0,
            "heterogeneity": True,
            "seed": 1,
        },
        "active_column_rule": "connectome training-window state std > 1e-7; same indices for every substrate",
        "active_column_count": len(active),
        "active_indices_sha256": array_hash(active),
        "standardization": "per-condition mean and scale from that condition's training window only; scales below 1e-7 replaced by 1",
        "split": {key: list(value) for key, value in bounds.items()},
        "deployment_holdout": "closed-loop body drives are separate from this recorded trajectory",
        "no_body_outcome_used": True,
        "conditions": list(CONDITIONS),
        "random_rule": "R3 seeded signed-edge weight permutation on fixed sparse graph",
        "rewired_rule": "R3 seeded degree-preserving directed endpoint swaps",
        "static_rule": "connectome training-window mean state repeated for every row",
        "graph_tensor_sha256_before": graph_before,
        "graph_files": graph_file_record(data),
        "stats": stats,
    }
    t1 = {
        "gate": "GATE T-1: PASS",
        "dataset": str(dataset_path),
        "setup": setup,
        "self_checks": {
            "xy_shapes": True,
            "step_alignment": True,
            "train_window_only_standardization": True,
            "split_rows_disjoint": True,
            "identical_input_shape": True,
            "finite": True,
        },
    }
    write_json(prefix_path(prefix, "_t1.json"), t1)
    write_rows(prefix_path(prefix, "_t1.csv"), rows)
    print(f"GATE T-1: PASS (X={RECORD_STEPS}x{len(active)} Y={RECORD_STEPS}x18 active_columns={len(active)})")
    print(json.dumps({"dataset": str(dataset_path), "split": setup["split"], "stats": stats}, sort_keys=True))


class ReadoutMLP(torch.nn.Module):
    """One-hidden-layer, feedforward, memoryless joint-target map."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.hidden = torch.nn.Linear(input_dim, HIDDEN_UNITS)
        self.output = torch.nn.Linear(HIDDEN_UNITS, 18)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(values)))


def train_readout(x: np.ndarray, y: np.ndarray, bounds: dict[str, tuple[int, int]]) -> tuple[ReadoutMLP, dict]:
    """Fit only joint-target MSE; validation MSE selects saved epoch."""
    torch.manual_seed(TRAIN_SEED)
    model = ReadoutMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    train_start, train_end = bounds["train"]
    val_start, val_end = bounds["validation"]
    train_order_rng = np.random.default_rng(TRAIN_SEED)
    train_order = np.arange(train_start, train_end)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = train_order_rng.permutation(train_order)
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            batch = torch.from_numpy(order[start:start + BATCH_SIZE])
            prediction = model(x_tensor[batch])
            loss = torch.mean((prediction - y_tensor[batch]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_prediction = model(x_tensor[train_start:train_end])
            validation_prediction = model(x_tensor[val_start:val_end])
            train_mse = float(torch.mean((train_prediction - y_tensor[train_start:train_end]) ** 2))
            validation_mse = float(torch.mean((validation_prediction - y_tensor[val_start:val_end]) ** 2))
        history.append({"epoch": epoch, "train_mse_rad2": train_mse, "validation_mse_rad2": validation_mse})
        if validation_mse < best_val:
            best_val = validation_mse
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise AssertionError("training produced no selected model")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor).numpy()
    errors = {
        "train_mse_rad2": float(np.mean((predictions[train_start:train_end] - y[train_start:train_end]) ** 2)),
        "validation_mse_rad2": float(np.mean((predictions[val_start:val_end] - y[val_start:val_end]) ** 2)),
        "test_mse_rad2": float(np.mean((predictions[bounds["test"][0]:bounds["test"][1]] - y[bounds["test"][0]:bounds["test"][1]]) ** 2)),
    }
    return model, {"best_epoch": best_epoch, "best_validation_mse_rad2": best_val, "errors": errors, "history": history}


def save_readout(path: Path, model: ReadoutMLP, active: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> None:
    state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items()}
    np.savez_compressed(
        path,
        hidden_weight=state["hidden.weight"],
        hidden_bias=state["hidden.bias"],
        output_weight=state["output.weight"],
        output_bias=state["output.bias"],
        active_indices=active,
        train_mean=mean,
        train_scale=scale,
        stance=STANCE,
        input_dim=np.asarray(len(active)),
        hidden_units=np.asarray(HIDDEN_UNITS),
        output_dim=np.asarray(18),
        readout_kind=np.asarray("memoryless_mlp"),
    )


def load_readout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key].copy() for key in source.files}
    if tuple(result["hidden_weight"].shape) != (HIDDEN_UNITS, len(result["active_indices"])):
        raise AssertionError(f"bad hidden weight shape in {path}")
    if tuple(result["output_weight"].shape) != (18, HIDDEN_UNITS):
        raise AssertionError(f"bad output weight shape in {path}")
    return result


def predict_state(state: np.ndarray, readout: dict[str, np.ndarray]) -> np.ndarray:
    x = (np.asarray(state, dtype=np.float32)[readout["active_indices"]] - readout["train_mean"]) / readout["train_scale"]
    hidden = np.tanh(readout["hidden_weight"] @ x + readout["hidden_bias"])
    return (readout["output_weight"] @ hidden + readout["output_bias"]).astype(np.float32)


def predict_standardized(x: np.ndarray, readout: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    hidden = np.tanh(readout["hidden_weight"] @ values + readout["hidden_bias"])
    return (readout["output_weight"] @ hidden + readout["output_bias"]).astype(np.float32)


def run_t2(data: Path, prefix: Path) -> None:
    t1_path = prefix_path(prefix, "_t1.json")
    dataset_path = prefix_path(prefix, "_bc_dataset.npz")
    t1 = json.loads(t1_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as source:
        target = source["target"].copy()
        active = source["active_indices"].copy()
        train_end = int(source["train_end"])
        validation_end = int(source["validation_end"])
        x = {condition: source[f"x_{condition}"].copy() for condition in CONDITIONS}
        means = {condition: source[f"train_mean_{condition}"].copy() for condition in CONDITIONS}
        scales = {condition: source[f"train_scale_{condition}"].copy() for condition in CONDITIONS}
    bounds = {"train": (0, train_end), "validation": (train_end, validation_end), "test": (validation_end, len(target))}
    setup = {
        "architecture": {
            "class": "Linear-Tanh-Linear",
            "input_dim": len(active),
            "hidden_units": HIDDEN_UNITS,
            "activation": "tanh",
            "output_dim": 18,
            "memoryless": True,
            "input": "current standardized VNC population state only",
        },
        "loss": "mean squared error on 18 clean-clock joint targets",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epoch_cap": EPOCHS,
        "seed": TRAIN_SEED,
        "target_sha256": array_hash(target),
        "active_indices_sha256": array_hash(active),
        "split": {key: list(value) for key, value in bounds.items()},
        "selection": "lowest held-out validation joint-target MSE only",
        "body_outcome_used": False,
        "vnc_weights_trained": False,
    }
    setup_hash = json_hash(setup)
    rows, reports = [], {}
    for condition in CONDITIONS:
        model, report = train_readout(x[condition], target, bounds)
        artifact_path = prefix_path(prefix, f"_readout_{condition}.npz")
        save_readout(artifact_path, model, active, means[condition], scales[condition])
        row = {
            "condition": condition,
            "train_mse_rad2": report["errors"]["train_mse_rad2"],
            "validation_mse_rad2": report["errors"]["validation_mse_rad2"],
            "test_mse_rad2": report["errors"]["test_mse_rad2"],
            "best_epoch": report["best_epoch"],
            "artifact": str(artifact_path),
            "setup_hash": setup_hash,
        }
        rows.append(row)
        reports[condition] = {**row, "history": report["history"]}
        print(json.dumps(row, sort_keys=True), flush=True)
    if len({row["setup_hash"] for row in rows}) != 1:
        raise AssertionError("identical training setup hashes differ")
    if any(load_readout(Path(row["artifact"]))["active_indices"].shape != active.shape for row in rows):
        raise AssertionError("readout input shapes differ")

    graph_before = t1["setup"]["graph_tensor_sha256_before"]
    graph_after = graph_tensor_record(data)
    if graph_before != graph_after:
        raise AssertionError("graph tensor hash changed across training")
    files_before = graph_file_record(data)
    files_after = graph_file_record(data)
    if files_before != files_after or not files_after["hash_equal"]:
        raise AssertionError("committed graph files changed across training")
    result = {
        "gate": "GATE T-2: PASS",
        "setup": setup,
        "setup_hash": setup_hash,
        "conditions": reports,
        "identical_setup_assertion": True,
        "graph_hash_before_training": graph_before,
        "graph_hash_after_training": graph_after,
        "graph_files_before_training": files_before,
        "graph_files_after_training": files_after,
        "frozen_graph_hash_equal": True,
        "only_learned_object": "four separately saved readout MLPs",
        "body_outcome_used_for_loss_or_selection": False,
    }
    write_json(prefix_path(prefix, "_training.json"), result)
    write_rows(prefix_path(prefix, "_training.csv"), rows)
    print("GATE T-2: PASS (identical setup; graph hash equal before/after; validation-MSE selection only)")


def load_t3_inputs(prefix: Path) -> tuple[dict, dict[str, np.ndarray], np.ndarray]:
    training = json.loads(prefix_path(prefix, "_training.json").read_text(encoding="utf-8"))
    with np.load(prefix_path(prefix, "_bc_dataset.npz"), allow_pickle=False) as source:
        target = source["target"].copy()
    readouts = {condition: load_readout(prefix_path(prefix, f"_readout_{condition}.npz")) for condition in CONDITIONS}
    return training, readouts, target


def body_episode(vnc, readout: dict[str, np.ndarray] | None, spec: dict, secs: float, video: Path | None = None, static_target: np.ndarray | None = None) -> dict:
    if vnc is None and static_target is None:
        raise ValueError("static episode requires static target")
    if vnc is not None:
        vnc.reset()
        current = vnc.descending_current(VNC_DRIVE)
        for _ in range(BURN_STEPS):
            vnc.step(current)
    brain = FrozenBrain(DATA)
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    initial_yaw = float(spec["initial_yaw_rad"])
    data.qpos[3:7] = (np.cos(initial_yaw / 2.0), 0.0, 0.0, np.sin(initial_yaw / 2.0))
    physics_steps = max(1, int(round(DT_S / model.opt.timestep)))
    for _ in range(int(SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, STANCE, KP, KD)
        mujoco.mj_step(model, data)

    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    yaw0 = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
    prey = (x0 + float(spec["prey_x_m"]), y0 + float(spec["prey_y_m"]))
    last = (x0, y0)
    path = 0.0
    min_distance = float(np.hypot(spec["prey_x_m"], spec["prey_y_m"]))
    caught = False
    targets, heights, planted = [], [], []
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in range(6))
    renderer = mujoco.Renderer(model, 480, 640) if video else None
    cameras = []
    if renderer:
        for azimuth, elevation, distance in ((135.0, 25.0, 1.4), (0.0, 80.0, 1.8), (90.0, 25.0, 1.4), (180.0, 25.0, 1.4)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
            cameras.append(camera)
    frames = []
    for step in range(int(round(secs * CTRL_HZ))):
        trunk = (float(data.qpos[0]), float(data.qpos[1]))
        yaw = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
        present, azimuth, distance, size = retina(trunk, yaw, prey)
        forward, turn = brain.step(present, azimuth, size)
        if vnc is not None:
            current = vnc.descending_current(forward * VNC_DRIVE, turn, turn_scale=VNC_TURN_SCALE)
            vnc.step(current)
            target = predict_state(population_rates(vnc), readout)
        else:
            target = static_target.copy()
        targets.append(target)
        for _ in range(physics_steps):
            data.ctrl[:] = pd_torque(model, data, target, KP, KD)
            mujoco.mj_step(model, data)
        now = (float(data.qpos[0]), float(data.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        min_distance = min(min_distance, distance)
        caught |= distance < 0.25
        heights.append(float(data.qpos[2]))
        planted.append(feet_planted_fraction(data, floor_id, foot_ids))
        if renderer and step % 2 == 0:
            views = []
            for camera in cameras:
                camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.08)
                renderer.update_scene(data, camera=camera)
                views.append(renderer.render())
            frames.append(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))
    targets = np.asarray(targets, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    phase = phase_metrics(coxa_phase_signal(targets), DT_S, "RUNG1-PHASE-PRESENT", "RUNG1-PHASE-ABSENT")
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    result = {
        **{key: spec[key] for key in ("episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m")},
        "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "path_m": float(path),
        "min_distance_m": float(min_distance),
        "min_height_m": float(heights.min()),
        "mean_ride_height_m": float(heights.mean()),
        "mean_feet_planted": float(np.mean(planted)),
        "min_feet_planted": float(np.min(planted)),
        "fell": bool(heights.min() < H_MIN),
        "upright": bool(heights.min() >= H_MIN),
        "caught": bool(caught),
        "tripod_coherence_scalar": float(phase["tripod_coherence_scalar"]),
        "phase_verdict": phase["verdict"],
        "within_pairs_in_phase": int(phase["within_pairs_in_phase"]),
        "cross_pairs_antiphase": int(phase["cross_pairs_antiphase"]),
        "target_temporal_std_mean_rad": float(targets.std(axis=0).mean()),
        "target_temporal_std_max_rad": float(targets.std(axis=0).max()),
        "protocol": {
            "secs": secs,
            "ctrl_hz": CTRL_HZ,
            "vnc_drive": VNC_DRIVE,
            "vnc_turn_scale": VNC_TURN_SCALE,
            "vnc_burn_steps": BURN_STEPS,
            "settle_s": SETTLE_S,
            "kp": KP,
            "kd": KD,
            "h_min_m": H_MIN,
            "steering": "FrozenBrain exact run_vnc_chase forward/turn pass",
            "readout_input": "current VNC population state only",
            "learning": False,
        },
    }
    if renderer:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, frames, fps=25)
        result["video"] = str(video)
    return result


def summarize(rows: list[dict], rng: np.random.Generator) -> dict:
    result = {
        "n": len(rows),
        "falls": int(sum(row["fell"] for row in rows)),
        "fall_rate": float(np.mean([row["fell"] for row in rows])),
        "fall_wilson_95ci": wilson_interval(int(sum(row["fell"] for row in rows)), len(rows)),
        "phase_present": int(sum(row["phase_verdict"] == "RUNG1-PHASE-PRESENT" for row in rows)),
        "metrics": {},
    }
    for metric in ("path_m", "forward_distance_m", "tripod_coherence_scalar", "mean_feet_planted", "min_height_m", "target_temporal_std_mean_rad"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise AssertionError(f"non-finite metric {metric}")
        low, high = bootstrap_ci(values, rng, BOOTSTRAP_REPS)
        result["metrics"][metric] = {
            "mean": float(values.mean()),
            "ci_low": low,
            "ci_high": high,
            "ci": "95% percentile bootstrap",
        }
    return result


def compare(groups: dict[str, list[dict]], right: str, metric: str, rng: np.random.Generator) -> dict:
    diff = np.asarray([left[metric] - other[metric] for left, other in zip(groups["connectome"], groups[right])], dtype=np.float64)
    low, high = paired_bootstrap_ci(diff, rng, BOOTSTRAP_REPS)
    return {
        "comparison": f"connectome_minus_{right}",
        "metric": metric,
        "n_pairs": len(diff),
        "mean_difference": float(diff.mean()),
        "ci_low": low,
        "ci_high": high,
        "p_exact_two_sided_paired_sign_flip": exact_sign_flip(diff),
    }


def non_overlapping(summary: dict, right: str, metric: str) -> bool:
    left_ci = summary["connectome"]["metrics"][metric]
    right_ci = summary[right]["metrics"][metric]
    return bool(left_ci["ci_low"] > right_ci["ci_high"] or right_ci["ci_low"] > left_ci["ci_high"])


def evaluate_moat(report: dict, watched_verdict: str | None = None) -> dict:
    summary = report["summary"]
    comparisons = report["comparisons"]
    by_key = {(row["comparison"], row["metric"]): row for row in comparisons}
    clean = report["clean_clock_control"]
    connectome = summary["connectome"]
    stable = {
        "phase_present_all": connectome["phase_present"] == N_EPISODES,
        "fall_count_le_1": connectome["falls"] <= 1,
        "positive_forward": connectome["metrics"]["forward_distance_m"]["mean"] > 0.0,
        "feet_within_0.05_of_clean_control": connectome["metrics"]["mean_feet_planted"]["mean"] >= clean["mean_feet_planted"] - 0.05,
    }
    superiority = {}
    for right in ("random", "rewired"):
        for metric in ("tripod_coherence_scalar", "mean_feet_planted"):
            row = by_key[(f"connectome_minus_{right}", metric)]
            superiority[f"{right}_{metric}"] = {
                "positive_difference": row["mean_difference"] > 0.0,
                "holm_p_lt_0.05": row["p_holm"] < 0.05,
                "non_overlapping_cis": non_overlapping(summary, right, metric),
            }
    static = report["static_input_leak_check"]
    video_ok = watched_verdict is not None and watched_verdict.lower().startswith("genuine tripod")
    positive = bool(
        all(stable.values())
        and all(all(value.values()) for value in superiority.values())
        and static["pass"]
        and video_ok
    )
    return {
        "stable_connectome": stable,
        "superiority": superiority,
        "static_input_clean": static["pass"],
        "watched_video_genuine_tripod": video_ok,
        "positive_conjunction": positive,
        "verdict": "RUNG1-MOAT-EARNED" if positive else "RUNG1-MOAT-NULL",
    }


def run_t3(data: Path, prefix: Path) -> None:
    training, readouts, target = load_t3_inputs(prefix)
    prereg = {
        "status": "preregistered_before_body_deployment",
        "gate": "T-3",
        "pre_registered_null": PRE_REGISTERED_NULL,
        "pre_registered_positive": PRE_REGISTERED_POSITIVE,
        "protocol": {
            "conditions": list(CONDITIONS),
            "n_per_condition": N_EPISODES,
            "episode_seed": EPISODE_SEED,
            "secs": 8.0,
            "ctrl_hz": CTRL_HZ,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "paired_tests": "connectome minus random and rewired on coherence and feet-planted; exact sign-flip plus Holm across two controls per metric",
            "static_walk_threshold": {"path_m_lt": 1.0, "forward_m_lt": 0.5},
            "positive_stability": {"falls_at_most": 1, "feet_within_m": 0.05, "phase_present_all": True},
        },
        "frozen_graph_files": graph_file_record(data),
        "training_setup_hash": training["setup_hash"],
        "body_outcome_not_used_for_training_or_selection": True,
    }
    write_json(prefix_path(prefix, "_preregistered.json"), prereg)
    write_json(prefix_path(prefix, "_t3.json"), prereg)
    print("T-3 preregistration written before body deployment", flush=True)

    specs = episode_specs(N_EPISODES, EPISODE_SEED)
    groups: dict[str, list[dict]] = {condition: [] for condition in CONDITIONS}
    static_target = predict_standardized(np.zeros(len(readouts["static-input"]["active_indices"]), dtype=np.float32), readouts["static-input"])
    static_target_std = 0.0
    static_vnc = None
    static_rows = []
    for index, spec in enumerate(specs):
        row = body_episode(static_vnc, readouts["static-input"], spec, 8.0, static_target=static_target)
        row["condition"] = "static-input"
        static_rows.append(row)
        print(f"static-input {index + 1}/{N_EPISODES} path={row['path_m']:.4f} forward={row['forward_distance_m']:+.4f} fell={row['fell']}", flush=True)
    static_path = float(np.mean([row["path_m"] for row in static_rows]))
    static_forward = float(np.mean([row["forward_distance_m"] for row in static_rows]))
    static_pass = bool(
        static_target_std <= 1e-7
        and static_path < prereg["protocol"]["static_walk_threshold"]["path_m_lt"]
        and static_forward < prereg["protocol"]["static_walk_threshold"]["forward_m_lt"]
    )
    if not static_pass:
        raise AssertionError("GATE T-3 FAIL: static-input leak check failed; downstream results not trusted")
    groups["static-input"] = static_rows
    static_check = {
        "pass": static_pass,
        "constant_target_temporal_std_rad": static_target_std,
        "mean_path_m": static_path,
        "mean_forward_m": static_forward,
        "threshold": prereg["protocol"]["static_walk_threshold"],
    }

    for condition in DYNAMIC_CONDITIONS:
        vnc = condition_stepper(data, condition)
        for index, spec in enumerate(specs):
            video = prefix_path(prefix, "_connectome_walk_multiview.mp4") if condition == "connectome" and index == 0 else None
            row = body_episode(vnc, readouts[condition], spec, 8.0, video=video)
            row["condition"] = condition
            groups[condition].append(row)
            print(f"{condition} {index + 1}/{N_EPISODES} path={row['path_m']:.4f} forward={row['forward_distance_m']:+.4f} coherence={row['tripod_coherence_scalar']:+.4f} feet={row['mean_feet_planted']:.3f} fell={row['fell']}", flush=True)

    if any(len(rows) != N_EPISODES for rows in groups.values()):
        raise AssertionError("T-3 condition row count mismatch")
    keys = ("episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m")
    reference = [[row[key] for key in keys] for row in groups["connectome"]]
    for condition in CONDITIONS[1:]:
        actual = [[row[key] for key in keys] for row in groups[condition]]
        if not np.allclose(np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)):
            raise AssertionError(f"T-3 starts not paired for {condition}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {condition: summarize(groups[condition], rng) for condition in CONDITIONS}
    comparisons = []
    for metric in ("tripod_coherence_scalar", "mean_feet_planted"):
        metric_rows = [compare(groups, right, metric, rng) for right in ("random", "rewired")]
        for row, adjusted in zip(metric_rows, holm_adjust([item["p_exact_two_sided_paired_sign_flip"] for item in metric_rows])):
            row["p_holm"] = adjusted
            comparisons.append(row)
    clean_json = RESULTS / "rung1_preflight.json"
    clean_data = json.loads(clean_json.read_text(encoding="utf-8"))
    clean = clean_data["clean_engine"]
    report = {
        **prereg,
        "gate": "GATE T-3: PASS",
        "status": "awaiting_watched_video_verdict",
        "clean_clock_control": {
            "forward_distance_m": clean["forward_distance_m"],
            "path_m": clean["path_m"],
            "mean_feet_planted": clean["mean_feet_planted"],
            "fell": clean["fell"],
        },
        "static_input_leak_check": static_check,
        "summary": summary,
        "comparisons": comparisons,
        "rows": [row for condition in CONDITIONS for row in groups[condition]],
        "watched_video": {"path": str(prefix_path(prefix, "_connectome_walk_multiview.mp4")), "verdict": "PENDING"},
        "graph_hash_after_body": graph_file_record(data),
        "frozen_graph_hash_equal": graph_file_record(data)["hash_equal"],
        "readout_artifacts": {condition: str(prefix_path(prefix, f"_readout_{condition}.npz")) for condition in CONDITIONS},
        "target_sha256": array_hash(target),
    }
    write_json(prefix_path(prefix, "_t3.json"), report)
    fields = (
        "condition", "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "path_m", "forward_distance_m", "displacement_m", "tripod_coherence_scalar",
        "mean_feet_planted", "min_height_m", "fell", "phase_verdict", "within_pairs_in_phase",
        "cross_pairs_antiphase", "target_temporal_std_mean_rad",
    )
    write_rows(prefix_path(prefix, "_t3.csv"), [
        {key: row[key] for key in fields} for condition in CONDITIONS for row in groups[condition]
    ], fields)
    print("GATE T-3: PASS (static-input leak check PASS; four conditions; paired stats written)")
    print("READY FOR VIDEO REVIEW (run finalize after watching connectome video)")


def finalize(prefix: Path, watched_verdict: str) -> None:
    path = prefix_path(prefix, "_t3.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["watched_video"] = {"path": report["watched_video"]["path"], "verdict": watched_verdict}
    decision = evaluate_moat(report, watched_verdict)
    report["decision"] = decision
    report["status"] = "complete_no_final_claim"
    write_json(path, report)
    print("GATE T-3: PASS (watched real video: " + watched_verdict + ")")
    print(decision["verdict"])
    print("READY FOR CLAIM REVIEW")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("t1", "t2", "t3", "finalize"), required=True)
    parser.add_argument("--data", type=Path, default=RESULTS)
    parser.add_argument("--prefix", type=Path, default=RESULTS / "rung1")
    parser.add_argument("--watched-verdict", default=None)
    args = parser.parse_args()
    if args.phase == "t1":
        run_t1(args.data, args.prefix)
    elif args.phase == "t2":
        run_t2(args.data, args.prefix)
    elif args.phase == "t3":
        run_t3(args.data, args.prefix)
    else:
        if not args.watched_verdict:
            raise ValueError("--watched-verdict is required for finalize")
        finalize(args.prefix, args.watched_verdict)


if __name__ == "__main__":
    main()
