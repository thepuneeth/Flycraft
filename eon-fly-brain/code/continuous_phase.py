"""Method 2: drive the fixed gait shape with continuous VNC antagonist phase."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from analyze_control_study import bootstrap_ci, exact_sign_flip, holm_adjust, paired_bootstrap_ci  # noqa: E402
from gait_engine import StepGaitEngine  # noqa: E402
from hexapod_gait import H_MIN, XML, pd_torque  # noqa: E402
from probe_vnc_phase import PHASE_TOLERANCE_CYCLES, analytic_signal, fft_bandpass, wrap_cycles  # noqa: E402
from step_decode import motor_pools  # noqa: E402
from stepwalk_controls import episode_specs  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
TRIPOD_A = (0, 3, 4)
TRIPOD_B = (1, 2, 5)
VNC_STEPS = 1500
SAMPLE_DT_S = 0.005 * 4
BODY_STEPS = 8 * 50
BANDPASS_HZ = (1.5, 4.0)
PHASE_TO_ENGINE_OFFSET_RAD = np.pi / 2.0
CTRL_HZ = 50.0
DT_S = 1.0 / CTRL_HZ
SETTLE_S = 0.7
FORWARD = 0.8
TURN = 0.0
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260829
RANDOM_PHASE_SEED_OFFSET = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_difference(activity: np.ndarray, io: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[dict[str, object]]]:
    values = np.asarray(activity, dtype=np.float64)
    pools = motor_pools(io)
    difference = np.column_stack([
        values[:, pool["swing_indices"]].mean(axis=1) - values[:, pool["stance_indices"]].mean(axis=1)
        for pool in pools
    ])
    return difference, pools


def vnc_phase_source(data: Path, weights_name: str, index_name: str, shuffle: bool) -> dict:
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
    current = vnc.descending_current(100.0)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(VNC_STEPS)])
    difference, pools = pool_difference(history, vnc.io)
    filtered = fft_bandpass(difference, SAMPLE_DT_S, *BANDPASS_HZ)
    phase = np.unwrap(np.angle(analytic_signal(filtered)), axis=0) + PHASE_TO_ENGINE_OFFSET_RAD
    tail = phase[VNC_STEPS // 2:]
    body_phase = phase[-BODY_STEPS:]
    frequency_hz = np.polyfit(np.arange(len(tail)) * SAMPLE_DT_S, tail, 1)[0] / (2.0 * np.pi)
    return {
        "phase": body_phase,
        "record": {
            "source": "shuffled_vnc" if shuffle else "real_vnc",
            "shuffle": shuffle,
            "weights_name": weights_name,
            "index_name": index_name,
            "weights_sha256": file_sha256(data / weights_name),
            "protocol": {
                "model": "paper",
                "drive": 100.0,
                "gW": 0.03,
                "tau": 0.02,
                "dt": 0.005,
                "substeps": 4,
                "heterogeneity": True,
                "seed": 1,
                "steps": VNC_STEPS,
                "sample_dt_s": SAMPLE_DT_S,
                "burn_fraction": 0.5,
                "bandpass_hz": list(BANDPASS_HZ),
                "phase_to_engine_offset_rad": PHASE_TO_ENGINE_OFFSET_RAD,
                "phase_definition": "analytic phase of band-passed mean flexor-minus-extensor antagonist activity",
            },
            "motor_pool_counts": [{
                "leg": pool["leg"],
                "swing_count": len(pool["swing_indices"]),
                "stance_count": len(pool["stance_indices"]),
            } for pool in pools],
            "body_phase_frequency_hz": frequency_hz.tolist(),
            "body_swing_duty_per_leg": (np.sin(body_phase) > 0.0).mean(axis=0).tolist(),
            "body_swing_duty_mean": float((np.sin(body_phase) > 0.0).mean()),
            "body_phase_amplitude_mean": np.abs(analytic_signal(filtered)[-BODY_STEPS:]).mean(axis=0).tolist(),
        },
    }


def random_phase_source(real: dict, seed: int) -> dict:
    frequency_hz = np.asarray(real["record"]["body_phase_frequency_hz"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(0.0, 2.0 * np.pi, 6)
    phase = np.arange(BODY_STEPS, dtype=np.float64)[:, None] * (2.0 * np.pi * frequency_hz[None, :]) * DT_S + offsets
    return {
        "phase": phase,
        "record": {
            "source": "random_continuous_phase",
            "shuffle": False,
            "seed": seed,
            "protocol": {
                "phase_definition": "independent linear phases with real-source per-leg mean frequencies and seeded uniform initial phases",
                "frequency_match_source": "real_vnc body phase frequency",
                "phase_to_engine_offset_rad": PHASE_TO_ENGINE_OFFSET_RAD,
            },
            "body_phase_frequency_hz": frequency_hz.tolist(),
            "initial_phase_rad": offsets.tolist(),
            "body_swing_duty_per_leg": (np.sin(phase) > 0.0).mean(axis=0).tolist(),
            "body_swing_duty_mean": float((np.sin(phase) > 0.0).mean()),
        },
    }


def continuous_phase_metrics(phases: np.ndarray) -> dict:
    values = np.asarray(phases, dtype=np.float64).reshape(-1, 6)
    difference = np.zeros((6, 6), dtype=np.float64)
    concentration = np.zeros((6, 6), dtype=np.float64)
    for left in range(6):
        for right in range(6):
            vector = np.exp(1j * (values[:, left] - values[:, right]))
            mean_vector = np.mean(vector)
            difference[left, right] = np.angle(mean_vector)
            concentration[left, right] = abs(mean_vector)
    within = [(a, b) for group in (TRIPOD_A, TRIPOD_B) for i, a in enumerate(group) for b in group[i + 1:]]
    cross = [(a, b) for a in TRIPOD_A for b in TRIPOD_B]
    difference_cycles = difference / (2.0 * np.pi)
    within_cycles = np.asarray([difference_cycles[a, b] for a, b in within])
    cross_cycles = np.asarray([difference_cycles[a, b] for a, b in cross])
    within_ok = np.abs(wrap_cycles(within_cycles)) <= PHASE_TOLERANCE_CYCLES
    cross_ok = (0.5 - np.abs(wrap_cycles(cross_cycles))) <= PHASE_TOLERANCE_CYCLES
    within_cos = float(np.mean([np.cos(difference[a, b]) for a, b in within]))
    cross_cos = float(np.mean([np.cos(difference[a, b]) for a, b in cross]))
    return {
        "definition": "circular phase coherence of continuous per-leg engine phase; canonical tripod groups used only for reporting",
        "within_phase_cosine": within_cos,
        "cross_phase_cosine": cross_cos,
        "coherence_scalar": within_cos - cross_cos,
        "phase_difference_matrix_cycles": difference_cycles.tolist(),
        "phase_concentration_matrix": concentration.tolist(),
        "within_pairs_in_phase": int(within_ok.sum()),
        "within_pair_count": len(within),
        "cross_pairs_antiphase": int(cross_ok.sum()),
        "cross_pair_count": len(cross),
        "phase_verdict": "M2-PHASE-PRESENT" if bool(within_ok.all() and cross_ok.all()) else "M2-PHASE-ABSENT",
        "mean_swing_duty": float((np.sin(values) > 0.0).mean()),
        "swing_duty_per_leg": (np.sin(values) > 0.0).mean(axis=0).tolist(),
    }


def feet_planted_fraction(data, floor_id: int, foot_ids: tuple[int, ...]) -> float:
    planted = [False] * len(foot_ids)
    for index in range(data.ncon):
        geom1, geom2 = int(data.contact[index].geom1), int(data.contact[index].geom2)
        foot = geom2 if geom1 == floor_id else geom1 if geom2 == floor_id else -1
        if foot in foot_ids:
            planted[foot_ids.index(foot)] = True
    return float(np.mean(planted))


def run_engine(phase: np.ndarray, episode: dict, video: Path | None = None, secs: float = 8.0) -> dict:
    if len(phase) != int(round(secs * CTRL_HZ)):
        raise ValueError(f"phase length {len(phase)} does not match {secs:g}s at {CTRL_HZ:g}Hz")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    yaw = float(episode["initial_yaw_rad"])
    data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
    engine = StepGaitEngine(stance=None, phase_group=())
    stance = engine.primitive.stance.reshape(18).astype(np.float32)
    physics_steps = max(1, int(round(DT_S / model.opt.timestep)))
    for _ in range(int(SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, stance, 3.0, 0.05)
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
    for step, leg_phase in enumerate(np.asarray(phase, dtype=np.float64)):
        target = engine.continuous_targets(leg_phase, FORWARD, TURN)
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
    phase_metrics = continuous_phase_metrics(phase)
    result = {
        "episode": int(episode["episode"]),
        "episode_seed": int(episode["episode_seed"]),
        "initial_yaw_rad": yaw,
        "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "path_m": float(path),
        "min_height_m": float(heights.min()),
        "mean_feet_planted": float(np.mean(planted)),
        "mean_swing_duty": phase_metrics["mean_swing_duty"],
        "swing_duty_per_leg": phase_metrics["swing_duty_per_leg"],
        "fell": bool(heights.min() < H_MIN),
        "coherence_scalar": phase_metrics["coherence_scalar"],
        "continuous_phase": phase_metrics,
        "target_temporal_std_mean_rad": float(targets.std(axis=0).mean()),
        "protocol": {
            "secs": secs,
            "ctrl_hz": CTRL_HZ,
            "forward": FORWARD,
            "turn": TURN,
            "settle_s": SETTLE_S,
            "kp": 3.0,
            "kd": 0.05,
            "h_min_m": H_MIN,
            "engine": "StepGaitEngine.continuous_targets over fixed Path-1 HexapodGait shape",
            "phase_group": [],
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
    for name in ("path_m", "coherence_scalar", "mean_feet_planted", "mean_swing_duty"):
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
    parser.add_argument("--out", type=Path, default=RESULTS / "continuous_phase_method2")
    parser.add_argument("--video", type=Path, default=RESULTS / "continuous_phase_method2_real.mp4")
    args = parser.parse_args()
    if args.n < 20 or int(round(args.secs * CTRL_HZ)) != BODY_STEPS:
        raise ValueError("Method 2 requires n>=20 and 8 seconds at 50 Hz")

    real = vnc_phase_source(args.data, args.weights_name, args.index_name, False)
    shuffled = vnc_phase_source(args.data, args.weights_name, args.index_name, True)
    random = random_phase_source(real, args.seed + RANDOM_PHASE_SEED_OFFSET)
    sources = {"real": real, "shuffled": shuffled, "random": random}
    specs = episode_specs(args.n, args.seed)
    rows = []
    for condition, source in sources.items():
        for index, spec in enumerate(specs):
            row = run_engine(source["phase"], spec, args.video if condition == "real" and index == 0 else None, args.secs)
            row["condition"] = condition
            rows.append(row)
            print(f"{condition} {index + 1}/{args.n} path={row['path_m']:.4f} coherence={row['coherence_scalar']:+.4f} duty={row['mean_swing_duty']:.3f} feet={row['mean_feet_planted']:.3f} fell={row['fell']}", flush=True)

    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in sources}
    shared_starts = all(
        [(row["episode"], row["episode_seed"], row["initial_yaw_rad"]) for row in groups[condition]]
        == [(row["episode"], row["episode_seed"], row["initial_yaw_rad"]) for row in groups["real"]]
        for condition in sources
    )
    protocols = [row["protocol"] for row in rows]
    same_engine_parameters = all(protocol == protocols[0] for protocol in protocols)
    finite_metrics = all(np.isfinite(row[name]) for row in rows for name in ("path_m", "coherence_scalar", "mean_feet_planted", "mean_swing_duty"))
    if not shared_starts or not same_engine_parameters or not finite_metrics or any(len(group) != args.n for group in groups.values()):
        raise AssertionError("Method 2 control integrity failed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {condition: summarize(groups[condition], rng) for condition in sources}
    comparisons = [compare(groups["real"], groups["shuffled"], "real_minus_shuffled", rng), compare(groups["real"], groups["random"], "real_minus_random", rng)]
    adjusted = holm_adjust([row["p_exact_two_sided_paired_sign_flip"] for row in comparisons])
    for row, pvalue in zip(comparisons, adjusted):
        row["p_holm_across_two_method2_tests"] = pvalue
    method1_path = RESULTS / "connectome_coordination_method1.json"
    method1 = json.loads(method1_path.read_text(encoding="utf-8"))
    result = {
        "method": "M2 continuous VNC phase drive",
        "protocol": {
            "n_per_condition": args.n,
            "secs": args.secs,
            "episode_seed": args.seed,
            "shared_episode_starts": shared_starts,
            "same_engine_parameters": same_engine_parameters,
            "only_condition_difference": "continuous phase source",
            "bandpass_hz": list(BANDPASS_HZ),
            "phase_to_engine_offset_rad": PHASE_TO_ENGINE_OFFSET_RAD,
            "phase_to_engine_offset_rule": "positive flexor-minus-extensor activity maps to positive sin(engine phase), the swing half-cycle",
            "zero_learning": True,
        },
        "method1_reference": {
            "artifact": str(method1_path),
            "real_grouping": method1["groupings"]["real"],
            "combined": False,
            "reason": "continuous per-leg phase already carries inter-leg phase offsets; adding Method-1 group offset would duplicate them",
        },
        "sources": {condition: source["record"] for condition, source in sources.items()},
        "summary": summary,
        "comparisons": comparisons,
        "gate_c": {
            "shared_episode_starts": shared_starts,
            "same_engine_parameters": same_engine_parameters,
            "finite_metrics": finite_metrics,
            "rows": len(rows),
            "verdict": "GATE C: PASS",
        },
        "verdict": "M2-APPLIED",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = ("condition", "episode", "episode_seed", "initial_yaw_rad", "path_m", "forward_distance_m", "displacement_m", "coherence_scalar", "mean_swing_duty", "mean_feet_planted", "min_height_m", "fell")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    print(json.dumps({"summary": summary, "comparisons": comparisons, "sources": {condition: source["record"] for condition, source in sources.items()}}, sort_keys=True))
    print(f"GATE C: PASS (n={args.n} per condition; shared starts={shared_starts}; same engine={same_engine_parameters}; finite metrics={finite_metrics})")
    print(f"method 2 -> {args.out.with_suffix('.json')}")
    print(f"video -> {args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
