"""Method 1: replace the hand-coded tripod grouping with frozen VNC phase groups."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from analyze_control_study import bootstrap_ci, exact_sign_flip, holm_adjust, paired_bootstrap_ci  # noqa: E402
from audit_vnc_legs import leg_scores  # noqa: E402
from gait_engine import coxa_phase_signal  # noqa: E402
from hexapod_gait import H_MIN, HexapodGait, XML, pd_torque  # noqa: E402
from probe_vnc_phase import PHASE_TOLERANCE_CYCLES, phase_metrics, wrap_cycles  # noqa: E402
from stepwalk_controls import episode_specs  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
VNC_STEPS = 1500
VNC_SAMPLE_DT = 0.005 * 4
VNC_SEED = 1
CTRL_HZ = 50.0
DT_S = 1.0 / CTRL_HZ
SETTLE_S = 0.7
FORWARD = 0.8
TURN = 0.0
GROUP_TOLERANCE_CYCLES = 0.25
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260829
RANDOM_GROUP_SEED_OFFSET = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_phase(data: Path, weights_name: str, index_name: str, shuffle: bool) -> dict:
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
        seed=VNC_SEED,
        shuffle=shuffle,
    )
    current = vnc.descending_current(100.0)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(VNC_STEPS)])
    scores = leg_scores(history, vnc.io)[..., 0]
    phase = phase_metrics(scores, VNC_SAMPLE_DT, "M1-PRESENT", "M1-ABSENT")
    return {
        "source": "shuffled_vnc" if shuffle else "real_vnc",
        "weights_name": weights_name,
        "index_name": index_name,
        "weights_sha256": file_sha256(data / weights_name),
        "shuffle": shuffle,
        "protocol": {
            "model": "paper",
            "drive": 100.0,
            "gW": 0.03,
            "tau": 0.02,
            "dt": 0.005,
            "substeps": 4,
            "heterogeneity": True,
            "seed": VNC_SEED,
            "steps": VNC_STEPS,
            "sample_dt_s": VNC_SAMPLE_DT,
            "burn_fraction": 0.5,
            "bandpass_hz": [1.5, 4.0],
        },
        "phase": phase,
    }


def derive_grouping(phase_cycles: np.ndarray) -> dict:
    matrix = np.asarray(phase_cycles, dtype=np.float64)
    if matrix.shape != (6, 6):
        raise ValueError(f"phase matrix shape {matrix.shape}; expected (6, 6)")
    relative = np.asarray(wrap_cycles(matrix[:, 0]), dtype=np.float64)
    group_a = [0]
    group_b = []
    relations = [{"leg": LEG_NAMES[0], "relative_phase_cycles": 0.0, "relation": "anchor"}]
    for leg in range(1, 6):
        delta = float(relative[leg])
        same = abs(delta) < GROUP_TOLERANCE_CYCLES
        anti = abs(abs(delta) - 0.5) <= GROUP_TOLERANCE_CYCLES
        if same:
            group_a.append(leg)
            relation = "same"
        elif anti:
            group_b.append(leg)
            relation = "anti"
        else:
            in_a = float(np.cos(2.0 * np.pi * delta)) >= 0.0
            (group_a if in_a else group_b).append(leg)
            relation = "cosine_fallback_same" if in_a else "cosine_fallback_anti"
        relations.append({"leg": LEG_NAMES[leg], "relative_phase_cycles": delta, "relation": relation})
    return {
        "rule": "anchor leg T1L; same if |wrapped phase| < 0.25 cycles, anti if within 0.25 cycles of 0.5; cosine sign only at boundary",
        "anchor_leg": LEG_NAMES[0],
        "group_a": [LEG_NAMES[leg] for leg in group_a],
        "group_b": [LEG_NAMES[leg] for leg in group_b],
        "group_a_indices": group_a,
        "group_b_indices": group_b,
        "group_sizes": [len(group_a), len(group_b)],
        "relative_phase_cycles": relative.tolist(),
        "classification": relations,
    }


def random_grouping(real_grouping: dict, seed: int) -> dict:
    size_b = len(real_grouping["group_b_indices"])
    candidates = np.arange(1, 6, dtype=np.int64)
    chosen = sorted(np.random.default_rng(seed).choice(candidates, size=size_b, replace=False).tolist())
    group_a = [leg for leg in range(6) if leg not in chosen]
    return {
        "rule": "seeded uniform random group B matched to real group-B size; anchor leg T1L fixed in group A",
        "anchor_leg": LEG_NAMES[0],
        "group_a": [LEG_NAMES[leg] for leg in group_a],
        "group_b": [LEG_NAMES[leg] for leg in chosen],
        "group_a_indices": group_a,
        "group_b_indices": chosen,
        "group_sizes": [len(group_a), len(chosen)],
        "seed": seed,
    }


def group_pairs(grouping: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    group_a = grouping["group_a_indices"]
    group_b = grouping["group_b_indices"]
    within = list(combinations(group_a, 2)) + list(combinations(group_b, 2))
    cross = [(left, right) for left in group_a for right in group_b]
    return within, cross


def grouping_signal_metrics(signal: np.ndarray, grouping: dict) -> dict:
    values = np.asarray(signal, dtype=np.float64).reshape(-1, 6)
    matrix = np.corrcoef(values[len(values) // 2:], rowvar=False)
    within, cross = group_pairs(grouping)
    within_value = float(np.mean([matrix[a, b] for a, b in within])) if within else None
    cross_value = float(np.mean([matrix[a, b] for a, b in cross])) if cross else None
    return {
        "definition": "Pearson correlation of centered world-forward coxa target traces; raw clean-clock rate, no VNC band-pass",
        "within_coherence": within_value,
        "cross_coherence": cross_value,
        "coherence_scalar": None if within_value is None or cross_value is None else within_value - cross_value,
        "within_pairs_same": int(sum(matrix[a, b] > 0.0 for a, b in within)),
        "within_pair_count": len(within),
        "cross_pairs_antiphase": int(sum(matrix[a, b] < 0.0 for a, b in cross)),
        "cross_pair_count": len(cross),
        "grouping": grouping,
        "correlation_matrix": matrix.tolist(),
    }


def feet_planted_fraction(data, floor_id: int, foot_ids: tuple[int, ...]) -> float:
    planted = [False] * len(foot_ids)
    for index in range(data.ncon):
        geom1, geom2 = int(data.contact[index].geom1), int(data.contact[index].geom2)
        foot = geom2 if geom1 == floor_id else geom1 if geom2 == floor_id else -1
        if foot in foot_ids:
            planted[foot_ids.index(foot)] = True
    return float(np.mean(planted))


def run_engine(grouping: dict, episode: dict, video: Path | None = None, secs: float = 8.0) -> dict:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    yaw = float(episode["initial_yaw_rad"])
    data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
    gait = HexapodGait(stance=None, phase_group=grouping["group_b_indices"])
    target = gait.targets(0.0, 0.0, 0.0).astype(np.float32)
    physics_steps = max(1, int(round(DT_S / model.opt.timestep)))
    for _ in range(int(SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, target, 3.0, 0.05)
        mujoco.mj_step(model, data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    yaw0 = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
    last = (x0, y0)
    path = 0.0
    heights, planted, targets = [], [], []
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
        target = gait.targets(FORWARD, TURN, DT_S)
        targets.append(target)
        for _ in range(physics_steps):
            data.ctrl[:] = pd_torque(model, data, target, 3.0, 0.05)
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
    targets = np.asarray(targets, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    phase = phase_metrics(coxa_phase_signal(targets), DT_S, "M1-PRESENT", "M1-ABSENT")
    group_phase = grouping_signal_metrics(coxa_phase_signal(targets), grouping)
    result = {
        "episode": int(episode["episode"]),
        "episode_seed": int(episode["episode_seed"]),
        "initial_yaw_rad": yaw,
        "grouping": grouping,
        "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "path_m": float(path),
        "min_height_m": float(heights.min()),
        "mean_feet_planted": float(np.mean(planted)),
        "fell": bool(heights.min() < H_MIN),
        "coherence_scalar": group_phase["coherence_scalar"],
        "group_phase": group_phase,
        "protocol": {
            "secs": secs,
            "ctrl_hz": CTRL_HZ,
            "forward": FORWARD,
            "turn": TURN,
            "settle_s": SETTLE_S,
            "kp": 3.0,
            "kd": 0.05,
            "h_min_m": H_MIN,
            "engine": "HexapodGait with only phase_group replaced",
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
    out = {"n": len(rows), "falls": int(sum(row["fell"] for row in rows)), "metrics": {}}
    for name in ("path_m", "coherence_scalar", "mean_feet_planted"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        low, high = bootstrap_ci(values, rng, BOOTSTRAP_REPS)
        out["metrics"][name] = {"mean": float(values.mean()), "ci_low": low, "ci_high": high, "ci": "95% percentile bootstrap"}
    return out


def compare(left: list[dict], right: list[dict], name: str, rng: np.random.Generator) -> dict:
    diff = np.asarray([a["path_m"] - b["path_m"] for a, b in zip(left, right)], dtype=np.float64)
    low, high = paired_bootstrap_ci(diff, rng, BOOTSTRAP_REPS)
    return {
        "comparison": name,
        "n_pairs": len(diff),
        "mean_path_difference_m": float(diff.mean()),
        "ci_low": low,
        "ci_high": high,
        "p_exact_two_sided_paired_sign_flip": exact_sign_flip(diff),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RESULTS)
    parser.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    parser.add_argument("--index_name", default="all_legs_io_index.npz")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--secs", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--out", type=Path, default=RESULTS / "connectome_coordination_method1")
    parser.add_argument("--video", type=Path, default=RESULTS / "connectome_coordination_method1_real.mp4")
    args = parser.parse_args()
    if args.n < 20:
        raise ValueError("--n must be >=20")

    real_source = source_phase(args.data, args.weights_name, args.index_name, False)
    shuffled_source = source_phase(args.data, args.weights_name, args.index_name, True)
    real_group = derive_grouping(real_source["phase"]["phase_difference_matrix_cycles"])
    shuffled_group = derive_grouping(shuffled_source["phase"]["phase_difference_matrix_cycles"])
    random_group = random_grouping(real_group, args.seed + RANDOM_GROUP_SEED_OFFSET)
    groupings = {"real": real_group, "shuffled": shuffled_group, "random": random_group}
    specs = episode_specs(args.n, args.seed)
    rows = []
    for condition, grouping in groupings.items():
        for index, spec in enumerate(specs):
            row = run_engine(grouping, spec, args.video if condition == "real" and index == 0 else None, args.secs)
            row["condition"] = condition
            rows.append(row)
            print(f"{condition} {index + 1}/{args.n} path={row['path_m']:.4f} coherence={row['coherence_scalar']!s} feet={row['mean_feet_planted']:.3f} fell={row['fell']}", flush=True)

    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in groupings}
    shared_starts = all(
        [(row["episode"], row["episode_seed"], row["initial_yaw_rad"]) for row in groups[condition]]
        == [(row["episode"], row["episode_seed"], row["initial_yaw_rad"]) for row in groups["real"]]
        for condition in groupings
    )
    finite_metrics = all(np.isfinite(row[name]) for row in rows for name in ("path_m", "coherence_scalar", "mean_feet_planted"))
    engine_protocols = [row["protocol"] for row in rows]
    same_engine_parameters = all(protocol == engine_protocols[0] for protocol in engine_protocols)
    if not shared_starts or not same_engine_parameters or not finite_metrics or any(len(group) != args.n for group in groups.values()):
        raise AssertionError("Method 1 control integrity failed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {condition: summarize(groups[condition], rng) for condition in groupings}
    comparisons = [compare(groups["real"], groups["shuffled"], "real_minus_shuffled", rng), compare(groups["real"], groups["random"], "real_minus_random", rng)]
    adjusted = holm_adjust([row["p_exact_two_sided_paired_sign_flip"] for row in comparisons])
    for row, pvalue in zip(comparisons, adjusted):
        row["p_holm_across_two_method1_tests"] = pvalue
    result = {
        "method": "M1 connectome-set inter-leg coordination",
        "protocol": {
            "n_per_condition": args.n,
            "secs": args.secs,
            "episode_seed": args.seed,
            "shared_episode_starts": True,
            "engine_parameters_shared": True,
            "only_condition_difference": "phase_group source",
            "group_tolerance_cycles": GROUP_TOLERANCE_CYCLES,
            "phase_metric_tolerance_cycles": PHASE_TOLERANCE_CYCLES,
            "random_group_seed": args.seed + RANDOM_GROUP_SEED_OFFSET,
            "zero_learning": True,
        },
        "sources": {"real": real_source, "shuffled": shuffled_source},
        "groupings": groupings,
        "summary": summary,
        "comparisons": comparisons,
        "gate_b": {
            "shared_episode_starts": shared_starts,
            "same_engine_parameters": same_engine_parameters,
            "finite_metrics": finite_metrics,
            "rows": len(rows),
            "verdict": "GATE B: PASS",
        },
        "verdict": "M1-APPLIED",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = ("condition", "episode", "episode_seed", "initial_yaw_rad", "path_m", "forward_distance_m", "displacement_m", "coherence_scalar", "mean_feet_planted", "min_height_m", "fell")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    print(json.dumps({"groupings": groupings, "summary": summary, "comparisons": comparisons}, sort_keys=True))
    print(f"GATE B: PASS (n={args.n} per condition; shared starts={shared_starts}; same engine=True; finite metrics={finite_metrics})")
    print(f"method 1 -> {args.out.with_suffix('.json')}")
    print(f"video -> {args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
