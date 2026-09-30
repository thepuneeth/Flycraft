"""One-shot frequency sweep for the recorded-step CPG translation.

Iteration 1's least-bad forward arm was CPG-only, so this calibrates the
hand-written prior before testing connectome interfaces.  The full ablation
ladder is still run at every frequency.  Frequencies are lower than the fly's
12 Hz reference because Froude scaling predicts a slower natural stride for a
larger, heavier hexapod; this is a declared physical rationale, not a fitted
parameter.  No learning or parameter updates occur.
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
    cpg_only_trace,
    file_sha256,
    make_gait,
    run_arm,
    signal_summary,
    spec_hash,
    summarize,
    target_trace,
)

FREQUENCIES_HZ = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--noise-seed", type=int, default=20260906)
    parser.add_argument("--step-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RESULTS / "cpg_frequency_sweep.json")
    args = parser.parse_args()
    if args.episodes < 8:
        raise ValueError("frequency sweep requires at least 8 episodes per frequency")

    os.environ["SC_CORE"] = "lif"
    os.environ.setdefault("SC_DEVICE", "cuda")
    import mujoco
    import specificity_campaign as SC
    from audit_vnc_legs import leg_scores
    from reservoir_3way import tensor_sha256

    graph_hashes_pre = {name: file_sha256(SC.VNC_DATA / name) for name in EXPECTED_FILE_HASHES}
    if graph_hashes_pre != EXPECTED_FILE_HASHES:
        raise AssertionError(f"frozen graph hash mismatch: {graph_hashes_pre}")
    from cpg_connectome_walk import collect_history
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
    curves = {name: [] for name in (*score_traces, "cpg_only")}
    ablation_ladder = {}

    for frequency_hz in FREQUENCIES_HZ:
        targets = {}
        amplitudes = {}
        for name, trace in score_traces.items():
            targets[name], amplitudes[name] = target_trace(
                SC, model, trace, rate_ceiling_hz, args.step_path, frequency_hz, 0.5
            )
        targets["cpg_only"] = cpg_only_trace(SC, model, args.step_path, frequency_hz)
        amplitudes["cpg_only"] = np.ones((len(history), 6), dtype=np.float64)

        summaries = {}
        for name, trace in targets.items():
            arm_rows = run_arm(SC, trace, vnc.io, specs, name)
            for row in arm_rows:
                row["frequency_hz"] = frequency_hz
            rows.extend(arm_rows)
            summaries[name] = {
                **summarize(arm_rows),
                "signal": signal_summary(history, rate_ceiling_hz) if name != "cpg_only" else {
                    "frac_active": 0.0,
                    "motor_hz": 0.0,
                    "rate_ceiling_hz": rate_ceiling_hz,
                },
                "cpg_parameter_range_fraction": {
                    "intrinsic_amplitude": float(
                        (amplitudes[name].max(axis=0) - amplitudes[name].min(axis=0)).max()
                    ),
                },
            }
            curves[name].append({
                "frequency_hz": frequency_hz,
                "summary": summaries[name],
            })
            print(json.dumps({
                "frequency_hz": frequency_hz,
                "arm": name,
                "forward_m": summaries[name]["metrics"]["forward_distance_m"]["mean"],
            }, sort_keys=True), flush=True)
        ablation_ladder[str(frequency_hz)] = summaries

    graph_hashes_post = {name: file_sha256(SC.VNC_DATA / name) for name in EXPECTED_FILE_HASHES}
    tensor_hash_post = tensor_sha256(vnc.W)
    if graph_hashes_post != graph_hashes_pre or tensor_hash_post != tensor_hash_pre:
        raise AssertionError("frozen graph changed during frequency sweep")

    winner_curve = curves["cpg_only"]
    best = max(winner_curve, key=lambda row: row["summary"]["metrics"]["forward_distance_m"]["mean"])
    best_index = winner_curve.index(best)
    neighbour_indices = [index for index in (best_index - 1, best_index + 1)
                          if 0 <= index < len(winner_curve)]
    neighbours = [winner_curve[index] for index in neighbour_indices]
    best_frequency = best["frequency_hz"]
    best_targets = {}
    best_targets["cpg_only"] = cpg_only_trace(SC, model, args.step_path, best_frequency)
    best_targets["full"], _ = target_trace(
        SC, model, scores, rate_ceiling_hz, args.step_path, best_frequency, 0.5
    )
    videos = {}
    for name in ("cpg_only", "full"):
        video = args.out.with_name(f"{args.out.stem}_{name}_multiview.mp4")
        videos[name] = {
            "path": str(video),
            "metrics": min(
                SC.run_body_trace(best_targets[name], vnc.io, specs[0], SECS, video=video)["horizons"].values(),
                key=lambda row: abs(float(row["horizon_s"]) - SECS),
            ),
        }

    artifact = {
        "experiment": "cpg_frequency_sweep",
        "status": "frequency_sweep_complete",
        "pre_registered_null": "CPG-interface forward drive is no greater with the real connectome than with degree-preserving rewires.",
        "protocol": {
            "secs": SECS,
            "drive": DRIVE,
            "frequencies_hz": list(FREQUENCIES_HZ),
            "episodes_per_frequency_per_arm": args.episodes,
            "noise_seed": args.noise_seed,
            "cpg_seed": CPG_SEED,
            "time_shuffle_seed": TIME_SHUFFLE_SEED,
            "v1_scale": 0.5,
            "body": body_record,
            "learning": False,
            "frequency_rationale": "Froude scaling motivates lower hexapod stride frequency than fly reference 12 Hz; grid declared once and not narrowed.",
            "winner_from_iteration_1": "cpg_only",
            "winner_basis": "highest iteration-1 mean forward among reported arms",
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
        "ablation_ladder": ablation_ladder,
        "curves": curves,
        "self_interrogation": {
            "a_full_minus_constant_forward_m_at_best_frequency": float(
                ablation_ladder[str(best_frequency)]["full"]["metrics"]["forward_distance_m"]["mean"]
                - ablation_ladder[str(best_frequency)]["constant_connectome"]["metrics"]["forward_distance_m"]["mean"]
            ),
            "b_parameter_range_fraction_at_best_frequency": ablation_ladder[str(best_frequency)]["full"]["cpg_parameter_range_fraction"],
            "c_constant_selected_from_forward": False,
            "d_identical_episode_specs": True,
            "d_episode_spec_hash": episode_hash,
            "e_weight_hashes_match": tensor_hash_pre == tensor_hash_post == EXPECTED_TENSOR_HASH,
            "f_realizations_per_arm": 1,
            "f_real_arm_sd_at_best_frequency": ablation_ladder[str(best_frequency)]["full"]["metrics"]["forward_distance_m"]["sd"],
            "g_rewire_replacement_number": "time-shuffled connectome included; degree-preserving rewire controls reserved for specificity campaign",
            "h_claim_guard": "frequency calibrates a hand-written CPG prior; connectome remains a modulation substrate",
        },
        "reference_departures": [
            {
                "reference": "flygym/src/flygym_demo/complex_terrain/preprogrammed.py:40-116",
                "departure": "seven fly DOFs retargeted to three hexapod DOFs by fixed joint role; adhesion omitted",
                "reason": "PhantomX has coxa/femur/tibia only and no adhesion actuators",
            },
            {
                "reference": "flygym/src/flygym_demo/complex_terrain/cpg_controller.py:166-180",
                "departure": "12 Hz fly reference replaced by declared 0.6–3.0 Hz hexapod grid",
                "reason": "Froude-scaled body size/mass makes lower stride frequency physically plausible; no fitted narrow sweep",
            },
        ],
        "numeric_decision": {
            "winner_arm": "cpg_only",
            "best_frequency_hz": best_frequency,
            "best_forward_m": best["summary"]["metrics"]["forward_distance_m"]["mean"],
            "best_path_m": best["summary"]["metrics"]["path_m"]["mean"],
            "neighbour_forward_m": [
                row["summary"]["metrics"]["forward_distance_m"]["mean"] for row in neighbours
            ],
            "neighbour_frequencies_hz": [row["frequency_hz"] for row in neighbours],
            "knife_edge": bool(neighbours and min(
                row["summary"]["metrics"]["forward_distance_m"]["mean"] for row in neighbours
            ) < 0.5 * best["summary"]["metrics"]["forward_distance_m"]["mean"]),
            "success_gate_complete": False,
            "decision": "CONTINUE_TO_ALL_THREE_VARIANTS",
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
