"""Frozen last-three B/C/D zero-learning transfer controls."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
ANATOMY_DATA = HERE.parent.parent / "data"
sys.path.insert(0, str(HERE))
from analyze_control_study import (  # noqa: E402
    bootstrap_ci,
    exact_sign_flip,
    holm_adjust,
    paired_bootstrap_ci,
    wilson_interval,
)
from audit_intersegmental import cross_edges, load_annotations  # noqa: E402
from continuous_phase import (  # noqa: E402
    BODY_STEPS,
    CTRL_HZ,
    continuous_phase_metrics,
    feet_planted_fraction,
    file_sha256,
    random_phase_source,
    run_engine,
    vnc_phase_source,
)
from fetch_manc import NEUROTRANSMITTERS, nt_signs  # noqa: E402
from gait_engine import coxa_phase_signal  # noqa: E402
from hexapod_gait import H_MIN, XML, home_stance, pd_torque  # noqa: E402
from mn_to_joint import translate  # noqa: E402
from preflight import EXPECTED_SHA256, sha256  # noqa: E402
from probe_vnc_phase import phase_metrics  # noqa: E402
from stepwalk_controls import episode_specs  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


CONDITIONS = ("real", "shuffled", "random")
LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
LEG_INDEX = {name: index for index, name in enumerate(LEG_NAMES)}
TRIPOD_A = (0, 3, 4)
TRIPOD_B = (1, 2, 5)
PLANNED_N = 20
PLANNED_SECS = 8.0
COUPLING_SEED = 20260830
RANDOM_PHASE_SEED_OFFSET = 1
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260829
GAIN_CANDIDATES = tuple(index / 100.0 for index in range(21))
CLEAN_MIN_CONCENTRATION = 0.99
CLEAN_MAX_PHASE_STEP_RAD = 1.0
CLEAN_MAX_STEP_CHANGE_RAD = 0.5
SUMMARY_METRICS = (
    "forward_distance_m",
    "path_m",
    "displacement_m",
    "coherence_scalar",
    "mean_feet_planted",
    "mean_swing_duty",
    "min_height_m",
)
DIRECT_SUMMARY_METRICS = (
    "forward_distance_m",
    "path_m",
    "displacement_m",
    "tripod_coherence_scalar",
    "mean_feet_planted",
    "min_height_m",
    "motor_temporal_std_mean",
    "target_temporal_std_mean_rad",
)
MAP_GAINS = (-0.55, 0.65, 0.65)
DIRECT_VNC_DRIVE = 100.0
DIRECT_RANDOM_SEED = 20260831
DIRECT_SETTLE_S = 0.7
TONIC_VNC_DRIVE = 10.0
TONIC_VNC_STEPS = 1500
TONIC_RANDOM_SEED = 20260832
TONIC_HOLD_S = 1.5
KP = 3.0
KD = 0.05


def endpoint_legs(annotations, body_ids: np.ndarray) -> np.ndarray:
    rows = annotations.reindex(body_ids)
    names = rows["somaNeuromere"].astype("string") + rows["somaSide"].astype("string")
    return names.map(LEG_INDEX).fillna(-1).to_numpy(dtype=np.int8)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    scale = np.abs(matrix).sum(axis=1)
    if np.any(scale <= 0.0):
        raise AssertionError("every post-leg row needs nonzero signed edge mass")
    return matrix / scale[:, None]


def build_coupling_matrices(data: Path, seed: int = COUPLING_SEED) -> dict:
    annotations = load_annotations(data)
    pre, post, weight, _, _ = cross_edges(data, annotations)
    signs = nt_signs(ANATOMY_DATA / "manc" / NEUROTRANSMITTERS, np.unique(pre))
    pre_leg = endpoint_legs(annotations, pre)
    post_leg = endpoint_legs(annotations, post)
    pre_sign = np.asarray([signs.get(int(body), 0.0) for body in pre], dtype=np.float64)
    valid = (pre_leg >= 0) & (post_leg >= 0) & (pre_leg != post_leg) & (pre_sign != 0.0)
    raw = np.zeros((6, 6), dtype=np.float64)
    np.add.at(raw, (post_leg[valid], pre_leg[valid]), weight[valid] * pre_sign[valid])
    mask = raw != 0.0
    if not (0 < int(mask.sum()) < 30):
        raise AssertionError(f"unexpected K sparsity: {int(mask.sum())}")

    values = raw[mask].copy()
    shuffled = np.zeros_like(raw)
    shuffled[mask] = np.random.default_rng(seed).permutation(values)
    random = np.zeros_like(raw)
    random_rng = np.random.default_rng(seed + 1)
    random[mask] = random_rng.permutation(np.abs(values)) * random_rng.permutation(np.sign(values))
    matrices = {"real": raw, "shuffled": shuffled, "random": random}
    normalized = {condition: normalize_rows(matrix) for condition, matrix in matrices.items()}

    io = np.load(data / "all_legs_io_index.npz")
    published_count = int(len(io["published_coordination_body_ids"]))
    if published_count != 212:
        raise AssertionError(f"published coordination count changed: {published_count}")
    if int(len(set(io["published_coordination_body_ids"].tolist()))) != 212:
        raise AssertionError("published coordination body IDs are not unique")
    return {
        "matrices": normalized,
        "record": {
            "leg_order": list(LEG_NAMES),
            "matrix_orientation": "K[post_leg, pre_leg]",
            "edge_source": "audit_intersegmental.cross_edges official T1/T2/T3 edges at absolute synapse floor 5",
            "sign_source": "fetch_manc.nt_signs predicted_nt: acetylcholine +1; GABA/glutamate -1; unknown omitted",
            "raw_cross_edge_rows": int(len(pre)),
            "signed_leg_edge_rows": int(valid.sum()),
            "dropped_cross_edge_rows": int((~valid).sum()),
            "known_presynaptic_sign_rows": int((pre_sign != 0.0).sum()),
            "raw_nonzero_cells": int(mask.sum()),
            "raw_zero_cells": int((~mask).sum()),
            "raw_signed_matrices": {condition: matrix.tolist() for condition, matrix in matrices.items()},
            "row_normalization": "each post-leg row divided by its own absolute signed raw K row sum; fixed anatomy-only input scaling",
            "normalized_matrices": {condition: matrix.tolist() for condition, matrix in normalized.items()},
            "shuffle_seed": int(seed),
            "random_seed": int(seed + 1),
            "shuffled_rule": "permute signed nonzero raw K entries across the fixed real nonzero mask",
            "random_rule": "independently permute raw magnitudes and signs across the fixed real nonzero mask",
            "published_coordination_cells": published_count,
            "published_coordination_cells_in_graph": int(len(io["published_coordination"])),
        },
    }


def coupled_phases(source: np.ndarray, coupling: np.ndarray, gain: float) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 6 or len(source) < 2:
        raise ValueError(f"expected phase history with shape (time, 6), got {source.shape}")
    result = np.empty_like(source)
    result[0] = source[0]
    for step in range(1, len(source)):
        previous = result[step - 1]
        phase_difference = previous[None, :] - previous[:, None]
        coupling_term = np.sum(coupling * np.sin(phase_difference), axis=1)
        result[step] = previous + source[step] - source[step - 1] + gain * coupling_term
    if not np.isfinite(result).all():
        raise AssertionError("coupled phase contains non-finite values")
    return result


def clean_clock() -> np.ndarray:
    frequency = 0.8 + 0.5 * 0.8
    phase = np.arange(BODY_STEPS, dtype=np.float64)[:, None] * (2.0 * np.pi * frequency / CTRL_HZ)
    offsets = np.asarray([np.pi if index in TRIPOD_A else 0.0 for index in range(6)])
    return phase + offsets[None, :]


def clean_stability(phase: np.ndarray, coupling: np.ndarray, gain: float) -> dict:
    result = coupled_phases(phase, coupling, gain)
    metric = continuous_phase_metrics(result)
    concentration = np.asarray(metric["phase_concentration_matrix"], dtype=np.float64)
    upper = concentration[np.triu_indices(6, 1)]
    steps = np.diff(result, axis=0)
    step_changes = np.diff(steps, axis=0)
    return {
        "gain": float(gain),
        "min_pair_concentration": float(upper.min()),
        "max_phase_step_rad": float(np.abs(steps).max()),
        "max_step_change_rad": float(np.abs(step_changes).max()),
        "coherence_scalar": float(metric["coherence_scalar"]),
        "stable": bool(
            np.isfinite(result).all()
            and upper.min() >= CLEAN_MIN_CONCENTRATION
            and np.abs(steps).max() <= CLEAN_MAX_PHASE_STEP_RAD
            and np.abs(step_changes).max() <= CLEAN_MAX_STEP_CHANGE_RAD
        ),
    }


def calibrate_gain(matrices: dict[str, np.ndarray]) -> dict:
    clock = clean_clock()
    rows = []
    selected = None
    for gain in GAIN_CANDIDATES:
        per_condition = {condition: clean_stability(clock, matrices[condition], gain) for condition in CONDITIONS}
        stable_all = all(row["stable"] for row in per_condition.values())
        rows.append({"gain": gain, "stable_all_conditions": stable_all, "conditions": per_condition})
        if stable_all:
            selected = gain
    if selected is None:
        raise AssertionError("no clean-clock-stable coupling gain")
    return {
        "candidate_gains": list(GAIN_CANDIDATES),
        "selected_gain": float(selected),
        "criterion": {
            "minimum_pair_concentration": CLEAN_MIN_CONCENTRATION,
            "maximum_phase_step_rad": CLEAN_MAX_PHASE_STEP_RAD,
            "maximum_step_change_rad": CLEAN_MAX_STEP_CHANGE_RAD,
            "basis": "clean ideal alternating-tripod clock only; no body outcome or condition performance used",
        },
        "conditions": list(CONDITIONS),
        "table": rows,
    }


def summarize(group: list[dict], rng: np.random.Generator, metric_names: tuple[str, ...] = SUMMARY_METRICS) -> dict:
    fell = int(sum(bool(row["fell"]) for row in group))
    result = {
        "n": len(group),
        "falls": fell,
        "fall_rate": float(fell / len(group)),
        "fall_wilson_95ci": wilson_interval(fell, len(group)),
        "metrics": {},
    }
    for name in metric_names:
        values = np.asarray([row[name] for row in group], dtype=np.float64)
        if not np.isfinite(values).all():
            raise AssertionError(f"non-finite B metric: {name}")
        low, high = bootstrap_ci(values, rng, BOOTSTRAP_REPS)
        result["metrics"][name] = {
            "mean": float(values.mean()),
            "ci_low": low,
            "ci_high": high,
            "ci": "95% percentile bootstrap",
        }
    return result


def comparisons(
    groups: dict[str, list[dict]],
    rng: np.random.Generator,
    metric_names: tuple[str, ...] = SUMMARY_METRICS,
) -> list[dict]:
    rows = []
    for metric in metric_names:
        metric_rows = []
        for right in ("shuffled", "random"):
            diff = np.asarray([left[metric] - other[metric] for left, other in zip(groups["real"], groups[right])], dtype=np.float64)
            low, high = paired_bootstrap_ci(diff, rng, BOOTSTRAP_REPS)
            row = {
                "comparison": f"real_minus_{right}",
                "metric": metric,
                "n_pairs": len(diff),
                "mean_difference": float(diff.mean()),
                "ci_low": low,
                "ci_high": high,
                "p_exact_two_sided_paired_sign_flip": exact_sign_flip(diff),
            }
            metric_rows.append(row)
        adjusted = holm_adjust([row["p_exact_two_sided_paired_sign_flip"] for row in metric_rows])
        for row, pvalue in zip(metric_rows, adjusted):
            row["p_holm_across_two_controls_for_metric"] = pvalue
            rows.append(row)
    return rows


def graph_record(data: Path) -> dict:
    hashes = {name: sha256(data / name) for name in EXPECTED_SHA256}
    return {"sha256": hashes, "hash_equal": hashes == EXPECTED_SHA256}


def make_cameras(model: mujoco.MjModel) -> list[mujoco.MjvCamera]:
    cameras = []
    for azimuth, elevation, distance in ((135.0, 25.0, 1.4), (0.0, 80.0, 1.8), (90.0, 25.0, 1.4), (180.0, 25.0, 1.4)):
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        cameras.append(camera)
    return cameras


def motor_history(
    data: Path,
    weights_name: str,
    index_name: str,
    steps: int,
    shuffle: bool,
    drive: float = DIRECT_VNC_DRIVE,
) -> tuple[np.ndarray, np.lib.npyio.NpzFile, dict]:
    vnc = VNCStepper(
        data,
        substeps=4,
        tau=0.02,
        dt=0.005,
        gW=0.03,
        weights_name=weights_name,
        index_name=index_name,
        model="paper",
        threshold=7.5,
        activation_gain=1.0,
        fr_cap=200.0,
        heterogeneity=True,
        seed=1,
        shuffle=shuffle,
    )
    current = vnc.descending_current(drive)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)])
    return history, vnc.io, {
        "source": "shuffled_vnc_motor" if shuffle else "real_vnc_motor",
        "weights_name": weights_name,
        "index_name": index_name,
        "weights_sha256": file_sha256(data / weights_name),
        "protocol": {
            "model": "paper",
            "drive": float(drive),
            "gW": 0.03,
            "tau": 0.02,
            "dt": 0.005,
            "substeps": 4,
            "threshold": 7.5,
            "activation_gain": 1.0,
            "fr_cap": 200.0,
            "heterogeneity": True,
            "seed": 1,
            "steps": steps,
            "sample_dt_s": 0.02,
        },
        "motor_output_count": int(history.shape[1]),
        "shuffle": bool(shuffle),
    }


def direct_sources(data: Path, weights_name: str, index_name: str, steps: int) -> tuple[dict[str, np.ndarray], np.lib.npyio.NpzFile, dict]:
    real, io, real_record = motor_history(data, weights_name, index_name, steps, False)
    shuffled, shuffled_io, shuffled_record = motor_history(data, weights_name, index_name, steps, True)
    if not np.array_equal(io["motor_body_ids"], shuffled_io["motor_body_ids"]):
        raise AssertionError("shuffled motor output order changed")
    random = real.copy()
    random_rng = np.random.default_rng(DIRECT_RANDOM_SEED)
    for channel in range(random.shape[1]):
        random_rng.shuffle(random[:, channel])
    return {"real": real, "shuffled": shuffled, "random": random}, io, {
        "real": real_record,
        "shuffled": shuffled_record,
        "random": {
            "source": "per-channel seeded time permutation of real VNC motor history",
            "seed": DIRECT_RANDOM_SEED,
            "matched_to": "real motor output marginal at every channel",
            "protocol": {"steps": steps, "sample_dt_s": 0.02},
            "motor_output_count": int(random.shape[1]),
        },
    }


def run_direct_episode(motors: np.ndarray, io, episode: dict, video: Path | None = None, secs: float = PLANNED_SECS) -> dict:
    steps = int(round(secs * CTRL_HZ))
    if motors.shape != (steps, len(io["motor_out"])):
        raise ValueError(f"motor history shape {motors.shape}; expected {(steps, len(io['motor_out']))}")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    yaw = float(episode["initial_yaw_rad"])
    data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
    physics_steps = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    stance = translate(np.zeros(len(io["motor_out"]), dtype=np.float32), io, gains=MAP_GAINS)
    assert np.allclose(stance.reshape(6, 3), home_stance(model))
    for _ in range(int(DIRECT_SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, stance, KP, KD)
        mujoco.mj_step(model, data)

    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    yaw0 = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
    last = (x0, y0)
    path = 0.0
    heights = []
    planted = []
    targets = []
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in range(6))
    renderer = mujoco.Renderer(model, 480, 640) if video else None
    cameras = make_cameras(model) if renderer else []
    frames = []
    for step, motor in enumerate(motors):
        target = translate(motor, io, gains=MAP_GAINS)
        targets.append(target)
        for _ in range(physics_steps):
            data.ctrl[:] = pd_torque(model, data, target, KP, KD)
            mujoco.mj_step(model, data)
        now = (float(data.qpos[0]), float(data.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        heights.append(float(data.qpos[2]))
        planted.append(feet_planted_fraction(data, floor_id, foot_ids))
        if renderer and step % 2 == 0:
            views = []
            for camera in cameras:
                camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.08)
                renderer.update_scene(data, camera=camera)
                views.append(renderer.render())
            frames.append(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))

    motors = np.asarray(motors, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    phase = phase_metrics(coxa_phase_signal(targets), 1.0 / CTRL_HZ, "C1-PRESENT", "C1-ABSENT")
    result = {
        "episode": int(episode["episode"]),
        "episode_seed": int(episode["episode_seed"]),
        "initial_yaw_rad": yaw,
        "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "path_m": float(path),
        "tripod_coherence_scalar": float(phase["tripod_coherence_scalar"]),
        "phase": phase,
        "mean_feet_planted": float(np.mean(planted)),
        "min_height_m": float(heights.min()),
        "final_height_m": float(heights[-1]),
        "fell": bool(heights.min() < H_MIN),
        "motor_temporal_std_mean": float(motors.std(axis=0).mean()),
        "target_temporal_std_mean_rad": float(targets.std(axis=0).mean()),
        "active_target_channels": int(np.count_nonzero(targets.std(axis=0) > 1e-9)),
        "protocol": {
            "secs": secs,
            "ctrl_hz": CTRL_HZ,
            "settle_s": DIRECT_SETTLE_S,
            "kp": KP,
            "kd": KD,
            "motor_to_joint": "mn_to_joint.translate(motor_activity, io, gains=(-0.55, 0.65, 0.65))",
            "engine": None,
            "decode": None,
            "phase_machine": None,
            "h_min_m": H_MIN,
            "learning": False,
        },
    }
    if renderer:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, frames, fps=25)
        result["video"] = str(video)
    assert targets.shape == (steps, 18)
    return result


def run_c(args) -> int:
    steps = int(round(args.secs * CTRL_HZ))
    sources, io, source_records = direct_sources(args.data, args.weights_name, args.index_name, steps)
    specs = episode_specs(args.n, args.seed)
    rows = []
    for condition in CONDITIONS:
        for index, spec in enumerate(specs):
            video = args.video if condition == "real" and index == 0 else None
            result = run_direct_episode(sources[condition][0:steps], io, spec, video=video, secs=args.secs)
            rows.append({**spec, **result, "condition": condition})
            print(
                f"{condition} {index + 1}/{args.n} path={result['path_m']:.4f} "
                f"coherence={result['tripod_coherence_scalar']:+.4f} "
                f"motor_std={result['motor_temporal_std_mean']:.5f} fell={result['fell']}",
                flush=True,
            )
    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    reference = [(row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"]) for row in groups["real"]]
    shared_starts = all(
        [(row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"]) for row in groups[condition]] == reference
        for condition in CONDITIONS
    )
    protocols = [row["protocol"] for row in rows]
    same_pipeline = all(protocol == protocols[0] for protocol in protocols)
    finite_metrics = all(np.isfinite(row[name]) for row in rows for name in DIRECT_SUMMARY_METRICS)
    graph = graph_record(args.data)
    if not shared_starts or not same_pipeline or not finite_metrics or not graph["hash_equal"] or any(len(group) != args.n for group in groups.values()):
        raise AssertionError("Phase C integrity failed")
    e4_path = RESULTS / "e4_sensory_feedback.json"
    e4 = json.loads(e4_path.read_text(encoding="utf-8"))
    e4_record_path = str(e4_path.relative_to(HERE.parents[2]))
    if e4.get("verdict") != "E4-ABSENT" or e4.get("zero_learning") is not True:
        raise AssertionError("corrected E-series reference is not E4-ABSENT/zero-learning")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {condition: summarize(groups[condition], rng, DIRECT_SUMMARY_METRICS) for condition in CONDITIONS}
    comparison_rows = comparisons(groups, rng, DIRECT_SUMMARY_METRICS)
    result = {
        "method": "C direct MN waveform",
        "protocol": {
            "planned_n_per_condition": PLANNED_N,
            "planned_secs": PLANNED_SECS,
            "n_per_condition": args.n,
            "secs": args.secs,
            "episode_seed": args.seed,
            "shared_episode_starts": shared_starts,
            "same_fixed_pipeline": same_pipeline,
            "only_condition_difference": "motor source",
            "vnc_to_motor": "VNCStepper paper output at fixed drive 100; real/shuffled sources",
            "random_motor_source": "per-channel seeded time permutation of real motor history",
            "motor_map_gains": list(MAP_GAINS),
            "settle_s": DIRECT_SETTLE_S,
            "kp": KP,
            "kd": KD,
            "h_min_m": H_MIN,
            "engine": None,
            "decode": None,
            "phase_machine": None,
            "learning": False,
        },
        "graph": graph,
        "sources": source_records,
        "e_series_reference": {"artifact": e4_record_path, "verdict": e4["verdict"], "zero_learning": e4["zero_learning"], "body_corrected": True},
        "summary": summary,
        "comparisons": comparison_rows,
        "pre_registered_null": "raw MN waveform will not produce a walk; corrected E-series ABSENT is reproduced on the standing body",
        "gate_c1": {
            "shared_episode_starts": shared_starts,
            "same_fixed_pipeline": same_pipeline,
            "finite_metrics": finite_metrics,
            "graph_hash_equal": graph["hash_equal"],
            "e_series_absent_reproduced": e4["verdict"] == "E4-ABSENT",
            "rows": len(rows),
            "verdict": "GATE C1: PASS",
        },
        "zero_learning": True,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = (
        "condition", "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "forward_distance_m", "path_m", "displacement_m", "tripod_coherence_scalar", "mean_feet_planted",
        "min_height_m", "motor_temporal_std_mean", "target_temporal_std_mean_rad", "active_target_channels", "fell",
    )
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    print(json.dumps({"summary": summary, "comparisons": comparison_rows}, sort_keys=True))
    print("GATE C1: PASS (shared starts; same fixed map; finite metrics; E4-ABSENT reference)")
    print(f"json -> {args.out.with_suffix('.json')}")
    print(f"csv -> {args.out.with_suffix('.csv')}")
    print(f"video -> {args.video}")
    return 0


TONIC_SUMMARY_METRICS = (
    "mean_ride_height_m",
    "final_height_m",
    "min_height_m",
    "mean_feet_planted",
    "path_m",
    "displacement_m",
    "target_offset_norm_rad",
)


def tonic_sources(data: Path, weights_name: str, index_name: str) -> tuple[dict[str, np.ndarray], np.lib.npyio.NpzFile, dict]:
    real, io, real_record = motor_history(data, weights_name, index_name, TONIC_VNC_STEPS, False, TONIC_VNC_DRIVE)
    shuffled, shuffled_io, shuffled_record = motor_history(data, weights_name, index_name, TONIC_VNC_STEPS, True, TONIC_VNC_DRIVE)
    if not np.array_equal(io["motor_body_ids"], shuffled_io["motor_body_ids"]):
        raise AssertionError("shuffled tonic motor output order changed")
    random = real.copy()
    random_rng = np.random.default_rng(TONIC_RANDOM_SEED)
    for channel in range(random.shape[1]):
        random_rng.shuffle(random[:, channel])
    return {"real": real, "shuffled": shuffled, "random": random}, io, {
        "real": real_record,
        "shuffled": shuffled_record,
        "random": {
            "source": "per-channel seeded time permutation of real low-drive VNC motor history",
            "seed": TONIC_RANDOM_SEED,
            "matched_to": "real low-drive motor output marginal at every channel",
            "protocol": {"steps": TONIC_VNC_STEPS, "sample_dt_s": 0.02, "drive": TONIC_VNC_DRIVE},
            "motor_output_count": int(random.shape[1]),
        },
    }


def run_tonic_episode(motors: np.ndarray, io, episode: dict, video: Path | None = None) -> dict:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    yaw = float(episode["initial_yaw_rad"])
    data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
    physics_steps = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    home = translate(np.zeros(len(io["motor_out"]), dtype=np.float32), io, gains=MAP_GAINS)
    assert np.allclose(home.reshape(6, 3), home_stance(model))
    for _ in range(int(DIRECT_SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, home, KP, KD)
        mujoco.mj_step(model, data)
    mean_motor = np.asarray(motors[len(motors) // 2:].mean(axis=0), dtype=np.float64)
    target = translate(mean_motor, io, gains=MAP_GAINS)
    hold_steps = int(round(TONIC_HOLD_S * CTRL_HZ))
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    last = (x0, y0)
    path = 0.0
    heights = []
    planted = []
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in range(6))
    renderer = mujoco.Renderer(model, 480, 640) if video else None
    cameras = make_cameras(model) if renderer else []
    frames = []
    for step in range(hold_steps):
        for _ in range(physics_steps):
            data.ctrl[:] = pd_torque(model, data, target, KP, KD)
            mujoco.mj_step(model, data)
        now = (float(data.qpos[0]), float(data.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        heights.append(float(data.qpos[2]))
        planted.append(feet_planted_fraction(data, floor_id, foot_ids))
        if renderer and step % 2 == 0:
            views = []
            for camera in cameras:
                camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.08)
                renderer.update_scene(data, camera=camera)
                views.append(renderer.render())
            frames.append(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))
    heights = np.asarray(heights, dtype=np.float64)
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    result = {
        "episode": int(episode["episode"]),
        "episode_seed": int(episode["episode_seed"]),
        "initial_yaw_rad": yaw,
        "mean_ride_height_m": float(heights.mean()),
        "final_height_m": float(heights[-1]),
        "min_height_m": float(heights.min()),
        "mean_feet_planted": float(np.mean(planted)),
        "path_m": float(path),
        "displacement_m": float(np.hypot(dx, dy)),
        "target_offset_norm_rad": float(np.linalg.norm(target - home)),
        "target_mean_joint_rad": target.tolist(),
        "motor_mean": mean_motor.tolist(),
        "fell": bool(heights.min() < H_MIN),
        "pose_verdict": "D-POSE-PRESENT" if heights.mean() >= 0.12 and heights.min() >= H_MIN else "D-POSE-ABSENT",
        "protocol": {
            "hold_s": TONIC_HOLD_S,
            "ctrl_hz": CTRL_HZ,
            "pre_hold_settle_s": DIRECT_SETTLE_S,
            "kp": KP,
            "kd": KD,
            "motor_to_joint": "mn_to_joint.translate(mean_motor_tail, io, gains=(-0.55, 0.65, 0.65))",
            "engine": None,
            "decode": None,
            "phase_machine": None,
            "h_min_m": H_MIN,
            "pose_height_threshold_m": 0.12,
            "learning": False,
        },
    }
    if renderer:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, frames, fps=25)
        result["video"] = str(video)
    return result


def run_d(args) -> int:
    sources, io, source_records = tonic_sources(args.data, args.weights_name, args.index_name)
    specs = episode_specs(args.n, args.seed)
    rows = []
    for condition in CONDITIONS:
        mean_motor = sources[condition][len(sources[condition]) // 2:].mean(axis=0)
        print(f"{condition} tonic mean_motor_norm={np.linalg.norm(mean_motor):.6f}", flush=True)
        for index, spec in enumerate(specs):
            video = args.video if condition == "real" and index == 0 else None
            result = run_tonic_episode(sources[condition], io, spec, video=video)
            rows.append({**spec, **result, "condition": condition})
            print(
                f"{condition} {index + 1}/{args.n} height={result['mean_ride_height_m']:.4f} "
                f"path={result['path_m']:.4f} feet={result['mean_feet_planted']:.3f} "
                f"fell={result['fell']} {result['pose_verdict']}",
                flush=True,
            )
    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    reference = [(row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"]) for row in groups["real"]]
    shared_starts = all(
        [(row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"]) for row in groups[condition]] == reference
        for condition in CONDITIONS
    )
    protocols = [row["protocol"] for row in rows]
    same_pipeline = all(protocol == protocols[0] for protocol in protocols)
    finite_metrics = all(np.isfinite(row[name]) for row in rows for name in TONIC_SUMMARY_METRICS)
    graph = graph_record(args.data)
    if not shared_starts or not same_pipeline or not finite_metrics or not graph["hash_equal"] or any(len(group) != args.n for group in groups.values()):
        raise AssertionError("Phase D integrity failed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {condition: summarize(groups[condition], rng, TONIC_SUMMARY_METRICS) for condition in CONDITIONS}
    for condition in CONDITIONS:
        summary[condition]["pose_verdict"] = "D-POSE-PRESENT" if summary[condition]["metrics"]["mean_ride_height_m"]["mean"] >= 0.12 and summary[condition]["falls"] == 0 else "D-POSE-ABSENT"
    comparison_rows = comparisons(groups, rng, TONIC_SUMMARY_METRICS)
    result = {
        "method": "D connectome tonic pose",
        "protocol": {
            "planned_n_per_condition": PLANNED_N,
            "n_per_condition": args.n,
            "episode_seed": args.seed,
            "shared_episode_starts": shared_starts,
            "same_fixed_pipeline": same_pipeline,
            "tonic_drive": TONIC_VNC_DRIVE,
            "tonic_vnc_steps": TONIC_VNC_STEPS,
            "steady_state_window": "final 50% of constant-drive VNC history",
            "hold_s": TONIC_HOLD_S,
            "pose_height_threshold_m": 0.12,
            "motor_map_gains": list(MAP_GAINS),
            "h_min_m": H_MIN,
            "engine": None,
            "decode": None,
            "phase_machine": None,
            "learning": False,
        },
        "graph": graph,
        "sources": source_records,
        "summary": summary,
        "comparisons": comparison_rows,
        "pre_registered_rule": "constant low-drive VNC tail mean is translated once and held; mean ride height >= 0.12 m earns pose, otherwise no pose; no search or fallback",
        "gate_d": {
            "shared_episode_starts": shared_starts,
            "same_fixed_pipeline": same_pipeline,
            "finite_metrics": finite_metrics,
            "graph_hash_equal": graph["hash_equal"],
            "rows": len(rows),
            "verdict": "GATE D: PASS",
        },
        "zero_learning": True,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = (
        "condition", "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "mean_ride_height_m", "final_height_m", "min_height_m", "mean_feet_planted", "path_m",
        "displacement_m", "target_offset_norm_rad", "pose_verdict", "fell",
    )
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    print(json.dumps({"summary": summary, "comparisons": comparison_rows}, sort_keys=True))
    print("GATE D: PASS (shared starts; same fixed map; finite metrics; height rule recorded)")
    print(f"json -> {args.out.with_suffix('.json')}")
    print(f"csv -> {args.out.with_suffix('.csv')}")
    print(f"video -> {args.video}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("B", "C", "D"), default="B")
    parser.add_argument("--data", type=Path, default=RESULTS)
    parser.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    parser.add_argument("--index_name", default="all_legs_io_index.npz")
    parser.add_argument("--n", type=int, default=PLANNED_N)
    parser.add_argument("--secs", type=float, default=PLANNED_SECS)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = RESULTS / {
            "B": "lastthree_B_dynamic_coupling",
            "C": "lastthree_C_direct_mn_waveform",
            "D": "lastthree_D_tonic_pose",
        }[args.phase]
    if args.video is None:
        args.video = RESULTS / {
            "B": "lastthree_B_real_multiview.mp4",
            "C": "lastthree_C_real_multiview.mp4",
            "D": "lastthree_D_real_multiview.mp4",
        }[args.phase]
    if args.phase == "C":
        if args.n < 1 or args.secs <= 0.0 or int(round(args.secs * CTRL_HZ)) > BODY_STEPS:
            raise ValueError(f"run must use 1<=n and 0<secs<={PLANNED_SECS:g}")
        return run_c(args)
    if args.phase == "D":
        if args.n < 1:
            raise ValueError("run must use n>=1")
        return run_d(args)
    if args.n < 1 or args.secs <= 0.0 or int(round(args.secs * CTRL_HZ)) > BODY_STEPS:
        raise ValueError(f"run must use 1<=n and 0<secs<={PLANNED_SECS:g}")

    matrices = build_coupling_matrices(args.data)
    gain = calibrate_gain(matrices["matrices"])
    real_source = vnc_phase_source(args.data, args.weights_name, args.index_name, False)
    shuffled_source = vnc_phase_source(args.data, args.weights_name, args.index_name, True)
    random_source = random_phase_source(real_source, args.seed + RANDOM_PHASE_SEED_OFFSET)
    sources = {"real": real_source, "shuffled": shuffled_source, "random": random_source}
    phases = {
        condition: coupled_phases(source["phase"], matrices["matrices"][condition], gain["selected_gain"])
        for condition, source in sources.items()
    }
    steps = int(round(args.secs * CTRL_HZ))
    specs = episode_specs(args.n, args.seed)
    rows = []
    for condition in CONDITIONS:
        for index, spec in enumerate(specs):
            video = args.video if condition == "real" and index == 0 else None
            result = run_engine(phases[condition][:steps], spec, video=video, secs=args.secs)
            rows.append({**spec, **result, "condition": condition})
            print(
                f"{condition} {index + 1}/{args.n} path={result['path_m']:.4f} "
                f"coherence={result['coherence_scalar']:+.4f} "
                f"feet={result['mean_feet_planted']:.3f} fell={result['fell']}",
                flush=True,
            )

    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    reference = [(row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"]) for row in groups["real"]]
    shared_starts = all(
        [(row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"]) for row in groups[condition]] == reference
        for condition in CONDITIONS
    )
    protocols = [row["protocol"] for row in rows]
    same_engine_parameters = all(protocol == protocols[0] for protocol in protocols)
    finite_metrics = all(np.isfinite(row[name]) for row in rows for name in SUMMARY_METRICS)
    graph = graph_record(args.data)
    if not shared_starts or not same_engine_parameters or not finite_metrics or not graph["hash_equal"] or any(len(group) != args.n for group in groups.values()):
        raise AssertionError("Phase B integrity failed")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {condition: summarize(groups[condition], rng) for condition in CONDITIONS}
    comparison_rows = comparisons(groups, rng)
    result = {
        "method": "B dynamic inter-leg coupling",
        "protocol": {
            "planned_n_per_condition": PLANNED_N,
            "planned_secs": PLANNED_SECS,
            "n_per_condition": args.n,
            "secs": args.secs,
            "episode_seed": args.seed,
            "shared_episode_starts": shared_starts,
            "same_engine_parameters": same_engine_parameters,
            "only_condition_differences": "edge-derived K source plus matched phase source control",
            "coupling_equation": "theta_i[t] = theta_i[t-1] + source_delta_i[t] + g * sum_j K[i,j] * sin(theta_j[t-1] - theta_i[t-1])",
            "coupling_gain": gain["selected_gain"],
            "normalization": matrices["record"]["row_normalization"],
            "engine": "continuous_phase.run_engine / StepGaitEngine.continuous_targets; fixed Path-1 waveform",
            "forward": 0.8,
            "turn": 0.0,
            "settle_s": 0.7,
            "kp": 3.0,
            "kd": 0.05,
            "h_min_m": H_MIN,
            "learning": False,
        },
        "graph": graph,
        "coupling": matrices["record"],
        "gain_calibration": gain,
        "sources": {condition: source["record"] for condition, source in sources.items()},
        "summary": summary,
        "comparisons": comparison_rows,
        "pre_registered_null": "real-edge coupling will not outperform shuffled or random K on tripod coherence, feet planted, or path; coordination will not transfer",
        "gate_b": {
            "shared_episode_starts": shared_starts,
            "same_engine_parameters": same_engine_parameters,
            "finite_metrics": finite_metrics,
            "graph_hash_equal": graph["hash_equal"],
            "rows": len(rows),
            "verdict": "GATE B: PASS",
        },
        "zero_learning": True,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = (
        "condition", "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "forward_distance_m", "path_m", "displacement_m", "coherence_scalar", "mean_swing_duty",
        "mean_feet_planted", "min_height_m", "fell",
    )
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    print(json.dumps({"summary": summary, "comparisons": comparison_rows, "selected_gain": gain["selected_gain"]}, sort_keys=True))
    print("GATE B: PASS (shared starts; same engine; finite metrics; graph hashes)")
    print(f"json -> {args.out.with_suffix('.json')}")
    print(f"csv -> {args.out.with_suffix('.csv')}")
    print(f"video -> {args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
