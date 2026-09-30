"""Iteration runner: frozen VNC motor activity modulates an NMF CPG.

The CPG is a hand-written prior.  It supplies rhythm and tripod coordination;
the strongest honest claim here is only that the frozen connectome modulates a
biologically-derived gait.  This runner intentionally reports the CPG-only
and connectome-ablation rows beside the full arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from nmf_cpg import NMFCPGPreprogrammedGait, v1_amplitudes

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
EXPECTED_FILE_HASHES = {
    "all_legs_weight_csr.pkl": "018a4890e01f3dd608693e8a040103829e8523d134290ae918e6e5dd4d815506",
    "all_legs_io_index.npz": "3f65f5b3ebed0d63161dca416bffa1a55afc0be4906113103426b16492d09449",
}
EXPECTED_TENSOR_HASH = "128b5ceddec4d8e14f3a04fbb66a08a2e83564c674624741eb03fb2bf01517fc"
SECS = 8.0
DRIVE = 100.0
FREQUENCY_HZ = 1.2  # [NMF] existing reference-ceiling operating point; swept next.
V1_SCALE = 0.5       # declared affine map scale; not selected from body distance.
TIME_SHUFFLE_SEED = 20260906
CPG_SEED = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spec_hash(specs: list[dict]) -> str:
    payload = json.dumps(specs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def joint_ranges(model, mujoco) -> np.ndarray:
    ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg_{leg}_joint_{joint}")
        for leg in range(6) for joint in range(3)
    ], dtype=np.int64)
    if np.any(ids < 0):
        raise AssertionError("body model is missing named leg joints")
    return np.asarray(model.jnt_range[ids], dtype=np.float64).reshape(6, 3, 2)


def collect_history(SC, noise_seed: int):
    """Run one frozen LIF realization at the fixed open-loop descending drive."""
    if SC.CORE_KIND != "lif":
        raise AssertionError("CPG loop must use the published Shiu LIF core")
    vnc = SC.base_vnc(SC.VNC_DATA)
    vnc.core.generator.manual_seed(int(noise_seed))
    vnc.reset()
    current = vnc.descending_current(DRIVE)
    steps = int(round(SECS / SC.SIGNAL_SAMPLE_DT))
    history = np.stack([
        vnc.step(current).cpu().numpy() for _ in range(steps)
    ]).astype(np.float32)
    if history.shape != (steps, len(vnc.io["motor_out"])) or not np.isfinite(history).all():
        raise AssertionError(f"invalid VNC history shape={history.shape}")
    return history, vnc, float(vnc.core.rate_ceiling_hz)


def make_gait(SC, model, step_path: Path | None, frequency_hz: float):
    import mujoco
    home = np.asarray(model.key("home").qpos[7:25], dtype=np.float64).reshape(6, 3)
    return NMFCPGPreprogrammedGait(
        timestep=SC.SIGNAL_SAMPLE_DT,
        stance=home,
        joint_ranges=joint_ranges(model, mujoco),
        step_path=step_path,
        intrinsic_frequency=float(frequency_hz),
        seed=CPG_SEED,
    )


def target_trace(SC, model, leg_score_trace: np.ndarray, rate_ceiling_hz: float,
                 step_path: Path | None, frequency_hz: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    gait = make_gait(SC, model, step_path, frequency_hz)
    amplitudes = []
    targets = []
    for scores in leg_score_trace:
        amplitudes.append(v1_amplitudes(scores, rate_ceiling_hz, scale))
        targets.append(gait.targets(amplitudes[-1]))
    return np.asarray(targets, dtype=np.float32), np.asarray(amplitudes, dtype=np.float64)


def cpg_only_trace(SC, model, step_path: Path | None, frequency_hz: float) -> np.ndarray:
    gait = make_gait(SC, model, step_path, frequency_hz)
    steps = int(round(SECS / SC.SIGNAL_SAMPLE_DT))
    return np.stack([gait.targets() for _ in range(steps)]).astype(np.float32)


def horizon_row(trace: dict, secs: float) -> dict:
    return min(trace["horizons"].values(), key=lambda row: abs(float(row["horizon_s"]) - secs))


def run_arm(SC, targets: np.ndarray, io, specs: list[dict], name: str) -> list[dict]:
    rows = []
    for spec in specs:
        measured = horizon_row(SC.run_body_trace(targets, io, spec, SECS), SECS)
        forward = float(measured["forward_distance_m"])
        path = float(measured["path_m"])
        rows.append({
            "arm": name,
            "episode": int(spec["episode"]),
            **measured,
            "path_per_forward": float(path / forward) if forward > 0.0 else None,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    metrics = ("forward_distance_m", "path_m", "displacement_m", "min_height_m", "mean_feet_planted")
    return {
        "n": len(rows),
        "falls": int(sum(bool(row["fell"]) for row in rows)),
        "metrics": {
            metric: {
                "mean": float(np.mean([row[metric] for row in rows])),
                "sd": float(np.std([row[metric] for row in rows])),
            }
            for metric in metrics
        },
        "path_per_forward_mean": float(np.mean([
            row["path_per_forward"] for row in rows if row["path_per_forward"] is not None
        ])) if any(row["path_per_forward"] is not None for row in rows) else None,
    }


def signal_summary(history: np.ndarray, rate_ceiling_hz: float) -> dict:
    return {
        "frac_active": float(np.mean(history > 0.0)),
        "motor_hz": float(history.mean() * rate_ceiling_hz),
        "rate_ceiling_hz": float(rate_ceiling_hz),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise-seed", type=int, default=20260906)
    parser.add_argument("--frequency-hz", type=float, default=FREQUENCY_HZ)
    parser.add_argument("--v1-scale", type=float, default=V1_SCALE)
    parser.add_argument("--step-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS / "cpg_v1_amplitude.json")
    args = parser.parse_args()
    if args.episodes < 1 or args.frequency_hz <= 0.0:
        raise ValueError("episodes and frequency must be positive")

    # Set before import: specificity_campaign binds core/device at module load.
    os.environ["SC_CORE"] = "lif"
    os.environ.setdefault("SC_DEVICE", "cuda")
    import mujoco
    import specificity_campaign as SC
    from audit_vnc_legs import leg_scores

    graph_hashes_pre = {name: file_sha256(SC.VNC_DATA / name) for name in EXPECTED_FILE_HASHES}
    if graph_hashes_pre != EXPECTED_FILE_HASHES:
        raise AssertionError(f"frozen graph hash mismatch: {graph_hashes_pre}")
    history, vnc, rate_ceiling_hz = collect_history(SC, args.noise_seed)
    from reservoir_3way import tensor_sha256
    tensor_hash_pre = tensor_sha256(vnc.W)
    if tensor_hash_pre != EXPECTED_TENSOR_HASH:
        raise AssertionError(f"frozen tensor hash mismatch: {tensor_hash_pre}")

    scores = leg_scores(history, vnc.io)
    if scores.shape != (len(history), 6, 5) or np.isinf(scores).any():
        raise AssertionError(f"invalid leg score bridge: {scores.shape}")
    constant_history = np.repeat(history.mean(axis=0, keepdims=True), len(history), axis=0)
    shuffled_history = history[np.random.default_rng(TIME_SHUFFLE_SEED).permutation(len(history))]
    score_traces = {
        "full": scores,
        "constant_connectome": leg_scores(constant_history, vnc.io),
        "time_shuffled_connectome": leg_scores(shuffled_history, vnc.io),
    }
    model, body_record = SC.make_body_model()
    target_traces, amplitude_traces = {}, {}
    for name, trace in score_traces.items():
        target_traces[name], amplitude_traces[name] = target_trace(
            SC, model, trace, rate_ceiling_hz, args.step_path, args.frequency_hz, args.v1_scale
        )
    target_traces["cpg_only"] = cpg_only_trace(SC, model, args.step_path, args.frequency_hz)
    amplitude_traces["cpg_only"] = np.ones((len(history), 6), dtype=np.float64)

    specs = SC.episode_specs(args.episodes, SC.EPISODE_SEED)
    episode_hash = spec_hash(specs)
    rows = []
    summaries = {}
    for name, targets in target_traces.items():
        arm_rows = run_arm(SC, targets, vnc.io, specs, name)
        rows.extend(arm_rows)
        summaries[name] = {
            **summarize(arm_rows),
            "signal": signal_summary(history, rate_ceiling_hz) if name != "cpg_only" else {
                "frac_active": 0.0, "motor_hz": 0.0, "rate_ceiling_hz": rate_ceiling_hz,
            },
            "cpg_parameter_range_fraction": {
                "intrinsic_amplitude": float(
                    (amplitude_traces[name].max(axis=0) - amplitude_traces[name].min(axis=0)).max()
                ),
            },
        }

    full_forward = summaries["full"]["metrics"]["forward_distance_m"]["mean"]
    constant_forward = summaries["constant_connectome"]["metrics"]["forward_distance_m"]["mean"]
    cpg_forward = summaries["cpg_only"]["metrics"]["forward_distance_m"]["mean"]
    graph_hashes_post = {name: file_sha256(SC.VNC_DATA / name) for name in EXPECTED_FILE_HASHES}
    tensor_hash_post = tensor_sha256(vnc.W)
    if graph_hashes_post != graph_hashes_pre or tensor_hash_post != tensor_hash_pre:
        raise AssertionError("frozen graph changed during loop")

    videos = {}
    render_order = ["full", "constant_connectome"] if full_forward >= constant_forward else ["cpg_only", "full"]
    for name in render_order:
        video = args.out.with_name(f"{args.out.stem}_{name}_multiview.mp4")
        rendered = SC.run_body_trace(target_traces[name], vnc.io, specs[0], SECS, video=video)
        videos[name] = {"path": str(video), "metrics": horizon_row(rendered, SECS)}

    ablation = {
        name: {"summary": summaries[name], "video": videos.get(name)}
        for name in ("full", "constant_connectome", "time_shuffled_connectome", "cpg_only")
    }
    artifact = {
        "experiment": "cpg_connectome_walk",
        "status": "iteration_1_v1_complete",
        "pre_registered_null": "CPG-interface forward drive is no greater with the real connectome than with degree-preserving rewires.",
        "protocol": {
            "secs": SECS,
            "drive": DRIVE,
            "frequency_hz": args.frequency_hz,
            "v1_scale": args.v1_scale,
            "v1_affine": "clip(0.5 + scale * mean(signed_leg_score_hz / rate_ceiling_hz), 0, 1)",
            "episodes": args.episodes,
            "noise_seed": args.noise_seed,
            "cpg_seed": CPG_SEED,
            "time_shuffle_seed": TIME_SHUFFLE_SEED,
            "body": body_record,
            "learning": False,
            "cpg_role": "hand-written prior supplies rhythm and tripod coordination",
        },
        "zero_learning": True,
        "zero_learning_audit": {"learning": False, "parameter_updates": 0, "training_markers": []},
        "integrity": {
            "file_hashes_pre": graph_hashes_pre,
            "file_hashes_post": graph_hashes_post,
            "file_hashes_equal": graph_hashes_pre == graph_hashes_post,
            "real_tensor_sha256_pre": tensor_hash_pre,
            "real_tensor_sha256_post": tensor_hash_post,
            "real_tensor_hash_equal": tensor_hash_pre == tensor_hash_post,
            "expected_real_tensor_sha256": EXPECTED_TENSOR_HASH,
        },
        "episode_spec_hash": episode_hash,
        "rows": rows,
        "ablation_ladder": ablation,
        "summaries": summaries,
        "self_interrogation": {
            "a_full_minus_constant_forward_m": float(full_forward - constant_forward),
            "b_parameter_range_fraction": summaries["full"]["cpg_parameter_range_fraction"],
            "c_constant_selected_from_forward": False,
            "d_identical_episode_specs": True,
            "d_episode_spec_hash": episode_hash,
            "e_weight_hashes_match": tensor_hash_pre == tensor_hash_post == EXPECTED_TENSOR_HASH,
            "f_realizations_per_arm": 1,
            "f_real_arm_sd": summaries["full"]["metrics"]["forward_distance_m"]["sd"],
            "g_rewire_replacement_number": "not run in iteration 1",
            "h_claim_guard": "connectome modulates a hand-written CPG gait; CPG supplies rhythm and coordination",
        },
        "reference_departures": [
            {
                "reference": "flygym/src/flygym_demo/complex_terrain/preprogrammed.py:40-116",
                "departure": "seven fly DOFs retargeted to three hexapod DOFs by fixed joint role",
                "reason": "PhantomX has coxa/femur/tibia only; mapping is declared, not fitted",
            },
            {
                "reference": "flygym/src/flygym_demo/complex_terrain/preprogrammed.py:88-92",
                "departure": "adhesion flags omitted",
                "reason": "hexapod model has no adhesion actuators",
            },
            {
                "reference": "flygym/src/flygym_demo/complex_terrain/preprogrammed.py:123-143",
                "departure": "mapped excursions are normalised into MuJoCo joint range around home stance",
                "reason": "fly and hexapod joint limits/geometries differ",
            },
            {
                "reference": "flygym/src/flygym_demo/complex_terrain/cpg_controller.py:166-180",
                "departure": "controller is sampled at the 20 ms project control grid and uses 1.2 Hz",
                "reason": "hexapod-scale physical operating point; full frequency sweep is next iteration",
            },
        ],
        "numeric_decision": {
            "full_forward_m": float(full_forward),
            "constant_forward_m": float(constant_forward),
            "cpg_only_forward_m": float(cpg_forward),
            "full_beats_constant": bool(full_forward > constant_forward),
            "success_gate_complete": False,
            "decision": "CONTINUE_TO_FREQUENCY_SWEEP",
        },
        "videos": videos,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["numeric_decision"], sort_keys=True), flush=True)
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
