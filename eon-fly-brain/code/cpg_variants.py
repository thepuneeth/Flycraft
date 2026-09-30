"""Evaluate all frozen-connectome-to-CPG interfaces at fixed 1.2 Hz.

V1 changes amplitude, V2 changes each oscillator's intrinsic frequency, and V3
replaces the hand-written tripod phase bias with measured inter-leg phase. V1
should walk most often but says least about connectome specificity; V3 makes
the strongest claim and is expected to fail. Every declared scale and every
ablation is reported. No training, optimization, or feedback is present.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from cpg_connectome_walk import (
    CPG_SEED,
    DRIVE,
    EXPECTED_FILE_HASHES,
    EXPECTED_TENSOR_HASH,
    RESULTS,
    SECS,
    TIME_SHUFFLE_SEED,
    file_sha256,
    joint_ranges,
    run_arm,
    signal_summary,
    spec_hash,
    summarize,
)

BASE_FREQUENCY_HZ = 1.2  # fixed by corrected Iteration 6; no re-sweep here
SCALE_GRID = (0.25, 0.5, 1.0)
V2_FREQUENCY_MIN_HZ = 0.6
V2_FREQUENCY_MAX_HZ = 1.8
V2_FREQUENCY_HALF_RANGE_HZ = 0.6
VARIANTS = ("v1_amplitude", "v2_frequency", "v3_phase")
ARMS = ("full", "constant_connectome", "time_shuffled_connectome", "cpg_only")


def normalized_leg_activity(leg_scores: np.ndarray, rate_ceiling_hz: float) -> np.ndarray:
    scores = np.asarray(leg_scores, dtype=np.float64)
    if scores.shape != (6, 5) or np.isinf(scores).any():
        raise ValueError(f"leg_scores have shape {scores.shape}; expected (6, 5)")
    ceiling = float(rate_ceiling_hz)
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        raise ValueError("rate_ceiling_hz must be positive and finite")
    # Signed activity -> Hz -> the core's own ceiling. Units cancel, but the
    # explicit conversion keeps the mapping auditable at the interface.
    return np.nan_to_num(scores, nan=0.0).mean(axis=1) * ceiling / ceiling


def v2_frequencies(leg_scores: np.ndarray, rate_ceiling_hz: float,
                   scale: float) -> np.ndarray:
    """Declared affine score-to-frequency map, clipped to physical range."""
    normalized = normalized_leg_activity(leg_scores, rate_ceiling_hz)
    return np.clip(
        BASE_FREQUENCY_HZ + float(scale) * V2_FREQUENCY_HALF_RANGE_HZ * normalized,
        V2_FREQUENCY_MIN_HZ,
        V2_FREQUENCY_MAX_HZ,
    )


def wrap_phase(value: np.ndarray | float) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(value, dtype=np.float64)))


def phase_biases_from_scores(leg_score_trace: np.ndarray, sample_dt: float,
                             scale: float) -> tuple[np.ndarray, dict]:
    """Map measured left-minus-right phase to the CPG's right-minus-left bias."""
    from nmf_cpg import tripod_phase_biases
    from probe_vnc_phase import phase_metrics

    signal = np.nan_to_num(np.asarray(leg_score_trace, dtype=np.float64).mean(axis=2), nan=0.0)
    diagnostic = phase_metrics(signal, sample_dt, present_label="MEASURED", absent_label="MEASURED_PHASE_NOT_TRIPOD")
    # phase_metrics stores phase[i] - phase[j]; oscillator coupling needs
    # phase[j] - phase[i]. Blend once toward measured phase using declared scale.
    measured = -np.asarray(diagnostic["phase_difference_matrix_rad"], dtype=np.float64)
    tripod = tripod_phase_biases()
    bias = wrap_phase(tripod + float(scale) * wrap_phase(measured - tripod))
    np.fill_diagonal(bias, 0.0)
    return bias, diagnostic


def gait_for(SC, model, step_path: Path, phase_biases: np.ndarray | None):
    import mujoco
    from nmf_cpg import NMFCPGPreprogrammedGait

    stance = np.asarray(model.key("home").qpos[7:25], dtype=np.float64).reshape(6, 3)
    return NMFCPGPreprogrammedGait(
        timestep=SC.SIGNAL_SAMPLE_DT,
        stance=stance,
        joint_ranges=joint_ranges(model, mujoco),
        step_path=step_path,
        intrinsic_frequency=BASE_FREQUENCY_HZ,
        phase_biases=phase_biases,
        seed=CPG_SEED,
    )


def variant_trace(SC, model, score_trace: np.ndarray | None, rate_ceiling_hz: float,
                  step_path: Path, variant: str, scale: float) -> tuple[np.ndarray, dict]:
    from nmf_cpg import tripod_phase_biases, v1_amplitudes

    phase_diagnostic = None
    if variant == "v3_phase" and score_trace is not None:
        phase_biases, phase_diagnostic = phase_biases_from_scores(
            score_trace, SC.SIGNAL_SAMPLE_DT, scale
        )
    else:
        phase_biases = tripod_phase_biases()
    gait = gait_for(SC, model, step_path, phase_biases if variant == "v3_phase" else None)
    steps = int(round(SECS / SC.SIGNAL_SAMPLE_DT))
    if score_trace is None:
        # CPG-only is open-loop: leave amplitude/frequency at declared defaults
        # instead of routing synthetic zero activity through a modulation map.
        targets = np.stack([gait.targets() for _ in range(steps)]).astype(np.float32)
        assert targets.shape == (steps, 18) and np.isfinite(targets).all()
        return targets, {"parameter_range_fraction": 0.0, "phase_diagnostic": None}
    targets = []
    parameter_trace = []
    for scores in score_trace:
        if variant == "v1_amplitude":
            parameter = v1_amplitudes(scores, rate_ceiling_hz, scale)
            target = gait.targets(parameter)
        elif variant == "v2_frequency":
            parameter = v2_frequencies(scores, rate_ceiling_hz, scale)
            gait.network.intrinsic_freqs = parameter
            target = gait.targets()
        elif variant == "v3_phase":
            parameter = phase_biases.copy()
            target = gait.targets()
        else:
            raise ValueError(f"unknown variant {variant}")
        targets.append(target)
        parameter_trace.append(parameter)
    targets = np.asarray(targets, dtype=np.float32)
    assert targets.shape == (steps, 18) and np.isfinite(targets).all()
    if variant == "v3_phase":
        parameter_range_fraction = float(np.max(np.abs(wrap_phase(phase_biases - tripod_phase_biases()))) / np.pi)
    elif variant == "v2_frequency":
        parameter_range_fraction = float(
            np.max(np.ptp(np.asarray(parameter_trace), axis=0))
            / (V2_FREQUENCY_MAX_HZ - V2_FREQUENCY_MIN_HZ)
        )
    else:
        parameter_range_fraction = float(np.max(np.ptp(np.asarray(parameter_trace), axis=0)))
    return targets, {
        "parameter_range_fraction": parameter_range_fraction,
        "phase_diagnostic": phase_diagnostic,
    }


def self_check() -> None:
    assert np.allclose(normalized_leg_activity(np.zeros((6, 5)), 250.0), 0.0)
    assert np.allclose(v2_frequencies(np.zeros((6, 5)), 250.0, 1.0), BASE_FREQUENCY_HZ)
    assert np.all((v2_frequencies(np.full((6, 5), 100.0), 250.0, 10.0) >= V2_FREQUENCY_MIN_HZ))
    assert np.all((v2_frequencies(np.full((6, 5), -100.0), 250.0, 10.0) <= V2_FREQUENCY_MAX_HZ))
    assert np.allclose(wrap_phase(np.array([0.0, np.pi, -np.pi])), [0.0, np.pi, -np.pi])
    print("cpg_variants: affine/phase self-check passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise-seed", type=int, default=20260906)
    parser.add_argument("--step-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=RESULTS / "cpg_variants_iteration3.json")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.step_path is None:
        raise ValueError("--step-path is required for rollout")
    if args.episodes < 8:
        raise ValueError("variant evaluation requires at least 8 episodes per arm")

    os.environ["SC_CORE"] = "lif"
    os.environ.setdefault("SC_DEVICE", "cuda")
    import mujoco  # noqa: F401
    import specificity_campaign as SC
    from audit_vnc_legs import leg_scores
    from nmf_cpg import tripod_phase_biases
    from reservoir_3way import tensor_sha256
    from cpg_connectome_walk import collect_history

    graph_hashes_pre = {name: file_sha256(SC.VNC_DATA / name) for name in EXPECTED_FILE_HASHES}
    if graph_hashes_pre != EXPECTED_FILE_HASHES:
        raise AssertionError(f"frozen graph hash mismatch: {graph_hashes_pre}")
    history, vnc, rate_ceiling_hz = collect_history(SC, args.noise_seed)
    tensor_hash_pre = tensor_sha256(vnc.W)
    if tensor_hash_pre != EXPECTED_TENSOR_HASH:
        raise AssertionError(f"frozen tensor hash mismatch: {tensor_hash_pre}")
    scores = leg_scores(history, vnc.io)
    constant_history = np.repeat(history.mean(axis=0, keepdims=True), len(history), axis=0)
    shuffled_history = history[np.random.default_rng(TIME_SHUFFLE_SEED).permutation(len(history))]
    score_traces = {
        "full": scores,
        "constant_connectome": leg_scores(constant_history, vnc.io),
        "time_shuffled_connectome": leg_scores(shuffled_history, vnc.io),
    }
    model, body_record = SC.make_body_model()
    specs = SC.episode_specs(args.episodes, SC.EPISODE_SEED)
    episode_hash = spec_hash(specs)
    rows = []
    variants = {}
    targets_for_video = {}

    for variant in VARIANTS:
        scale_records = {}
        for scale in SCALE_GRID:
            summaries = {}
            phase_diagnostics = {}
            for arm in ARMS:
                trace = None if arm == "cpg_only" else score_traces[arm]
                targets, params = variant_trace(
                    SC, model, trace, rate_ceiling_hz, args.step_path, variant, scale
                )
                arm_rows = run_arm(SC, targets, vnc.io, specs, arm)
                for row in arm_rows:
                    row.update({"variant": variant, "scale": scale, "frequency_hz": BASE_FREQUENCY_HZ})
                rows.extend(arm_rows)
                summaries[arm] = {
                    **summarize(arm_rows),
                    "signal": signal_summary(history, rate_ceiling_hz) if arm != "cpg_only" else {
                        "frac_active": 0.0,
                        "motor_hz": 0.0,
                        "rate_ceiling_hz": rate_ceiling_hz,
                    },
                    "cpg_parameter_range_fraction": {
                        "selected_parameter": params["parameter_range_fraction"],
                    },
                }
                phase_diagnostics[arm] = params["phase_diagnostic"]
                targets_for_video[(variant, scale, arm)] = targets
                print(json.dumps({
                    "variant": variant,
                    "scale": scale,
                    "arm": arm,
                    "forward_m": summaries[arm]["metrics"]["forward_distance_m"]["mean"],
                }, sort_keys=True), flush=True)
            scale_records[str(scale)] = {
                "scale": scale,
                "ablation_ladder": summaries,
                "phase_diagnostics": phase_diagnostics,
            }

        best = max(
            scale_records.values(),
            key=lambda record: record["ablation_ladder"]["full"]["metrics"]["forward_distance_m"]["mean"],
        )
        best_scale = best["scale"]
        variants[variant] = {
            "mapping": {
                "normalized_input": "mean(nan_to_num(signed_leg_scores)) * rate_ceiling_hz / rate_ceiling_hz",
                "scale_grid": list(SCALE_GRID),
                "selected_scale_rule": "maximum full-arm mean forward over declared grid; no narrowed follow-up",
                "frequency_hz": BASE_FREQUENCY_HZ,
            },
            "ablation_ladder": scale_records,
            "best": {
                "scale": best_scale,
                "full_forward_m": best["ablation_ladder"]["full"]["metrics"]["forward_distance_m"]["mean"],
                "full_path_m": best["ablation_ladder"]["full"]["metrics"]["path_m"]["mean"],
                "full_path_per_forward": best["ablation_ladder"]["full"]["path_per_forward_mean"],
            },
            "self_interrogation": {
                "a_full_minus_constant_forward_m": float(
                    best["ablation_ladder"]["full"]["metrics"]["forward_distance_m"]["mean"]
                    - best["ablation_ladder"]["constant_connectome"]["metrics"]["forward_distance_m"]["mean"]
                ),
                "b_parameter_range_fraction": best["ablation_ladder"]["full"]["cpg_parameter_range_fraction"],
                "c_constant_selected_from_forward": False,
                "d_identical_episode_specs": True,
                "d_episode_spec_hash": episode_hash,
                "e_weight_hashes_match": tensor_hash_pre == EXPECTED_TENSOR_HASH,
                "f_realizations_per_arm": 1,
                "f_real_arm_sd": best["ablation_ladder"]["full"]["metrics"]["forward_distance_m"]["sd"],
                "g_rewire_replacement_number": "not run; degree-preserving controls reserved for specificity if walking gate opens",
                "h_claim_guard": "connectome modulates a hand-written CPG prior; V1 timing, V2 coupling, V3 measured phase claim are reported separately",
            },
        }

    graph_hashes_post = {name: file_sha256(SC.VNC_DATA / name) for name in EXPECTED_FILE_HASHES}
    tensor_hash_post = tensor_sha256(vnc.W)
    if graph_hashes_post != graph_hashes_pre or tensor_hash_post != tensor_hash_pre:
        raise AssertionError("frozen graph changed during variant evaluation")

    videos = {}
    for variant, content in variants.items():
        scale = content["best"]["scale"]
        for arm in ("full", "constant_connectome"):
            video = args.out.with_name(f"{args.out.stem}_{variant}_{arm}_multiview.mp4")
            rendered = SC.run_body_trace(targets_for_video[(variant, scale, arm)], vnc.io, specs[0], SECS, video=video)
            videos[f"{variant}:{arm}"] = {
                "path": str(video),
                "metrics": min(rendered["horizons"].values(), key=lambda row: abs(float(row["horizon_s"]) - SECS)),
            }

    full_candidates = [
        (content["best"]["full_forward_m"], variant, content["best"]["scale"])
        for variant, content in variants.items()
    ]
    global_best = max(full_candidates)
    artifact = {
        "experiment": "cpg_connectome_variants",
        "status": "all_three_variants_complete",
        "pre_registered_null": "CPG-interface forward drive is no greater with the real connectome than with degree-preserving rewires.",
        "protocol": {
            "secs": SECS,
            "drive": DRIVE,
            "frequency_hz": BASE_FREQUENCY_HZ,
            "scale_grid": list(SCALE_GRID),
            "episodes_per_arm_per_scale": args.episodes,
            "noise_seed": args.noise_seed,
            "cpg_seed": CPG_SEED,
            "time_shuffle_seed": TIME_SHUFFLE_SEED,
            "body": body_record,
            "learning": False,
            "cpg_role": "hand-written prior supplies base rhythm and recorded-step coordination",
            "variant_tension": "V1 most likely to walk/least specific; V3 strongest specificity claim/most likely to fail",
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
        "ablation_ladder": {variant: content["ablation_ladder"] for variant, content in variants.items()},
        "self_interrogation": {variant: content["self_interrogation"] for variant, content in variants.items()},
        "variants": variants,
        "videos": videos,
        "numeric_decision": {
            "best_by_full_forward": {
                "variant": global_best[1],
                "scale": global_best[2],
                "forward_m": global_best[0],
            },
            "success_gate_complete": False,
            "decision": "CONTINUE_TO_20_EPISODE_CONFIRMATION_AND_SPECIFICITY_CONTROLS" if global_best[0] >= 0.5 else "CONTINUE_BUG_LOOP_BEFORE_SPECIFICITY",
        },
        "reference_departures": [
            {
                "reference": "flygym/src/flygym_demo/complex_terrain/preprogrammed.py:40-116",
                "departure": "seven fly DOFs retargeted to three hexapod DOFs by fixed role; adhesion omitted",
                "reason": "PhantomX has coxa/femur/tibia only and no adhesion actuators",
            },
            {
                "reference": "flygym/src/flygym_demo/complex_terrain/cpg_controller.py:166-180",
                "departure": "fixed 1.2 Hz operating point and three interface-specific maps",
                "reason": "Iteration 2 declared the one-shot frequency grid; variant scale grid is declared, not fitted",
            },
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    for variant, content in variants.items():
        variant_artifact = dict(artifact)
        variant_artifact["variant"] = variant
        variant_artifact["variants"] = {variant: content}
        variant_artifact["ablation_ladder"] = {variant: content["ablation_ladder"]}
        variant_artifact["self_interrogation"] = {variant: content["self_interrogation"]}
        variant_artifact["rows"] = [row for row in rows if row["variant"] == variant]
        path = args.out.with_name(f"cpg_{variant}_iteration3.json")
        path.write_text(json.dumps(variant_artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["numeric_decision"], sort_keys=True), flush=True)
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
