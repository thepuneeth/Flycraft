"""Final no-training specificity test for the frozen-connectome CPG bridge.

The CPG remains a hand-written [NMF] prior.  This confirmation asks whether
its declared V1/V2/V3 interfaces retain any real-connectome advantage after
amplitude matching, graph controls, and Poisson realization averaging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
SECS = 8.0  # [TASK] frozen §2/§7 episode duration.
DRIVE = 100.0  # [SH] existing descending-drive protocol, specificity_campaign.py.
EPISODES = 20  # [TASK] frozen minimum in §2/§8.3.
REAL_REALIZATIONS = 30  # [SH] default_params n_run=30; §7 requirement.
CONTROL_REALIZATIONS = 10  # [TASK] §7 minimum per control realization set.
TIME_SHUFFLE_SEED = 20260906  # [TASK] declared C-arm permutation seed.
CPG_SEED = 0  # [NMF] existing deterministic oscillator seed.
REAL_NOISE_SEEDS = tuple(20261000 + i for i in range(REAL_REALIZATIONS))  # [TASK] declared.
CONTROL_NOISE_SEEDS = tuple(20262000 + i for i in range(CONTROL_REALIZATIONS))  # [TASK] declared.
VARIANTS = ("v1_amplitude", "v2_frequency", "v3_phase")  # [TASK] §5.3.
ARMS = ("full", "constant_connectome", "time_shuffled_connectome", "cpg_only")  # [TASK] §8.1.
# [TASK] scales are frozen winners from corrected Iteration 7's one-shot grid;
# no new scale is selected here.
SELECTED_SCALES = {"v1_amplitude": 0.25, "v2_frequency": 1.0, "v3_phase": 0.5}

EXPECTED_FILE_HASHES = {
    "all_legs_weight_csr.pkl": "018a4890e01f3dd608693e8a040103829e8523d134290ae918e6e5dd4d815506",
    "all_legs_io_index.npz": "3f65f5b3ebed0d63161dca416bffa1a55afc0be4906113103426b16492d09449",
}
EXPECTED_TENSOR_HASH = "128b5ceddec4d8e14f3a04fbb66a08a2e83564c674624741eb03fb2bf01517fc"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spec_hash(specs: list[dict]) -> str:
    return hashlib.sha256(json.dumps(specs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def clean_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0)
    if scores.ndim != 3 or scores.shape[1:] != (6, 5) or not np.isfinite(scores).all():
        raise AssertionError(f"bad leg-score trace {scores.shape}")
    return scores


def collect_realization(SC, substrate, noise_seed: int, leg_scores):
    """Collect one stochastic [SH] history without changing its frozen graph."""
    vnc = SC.substrate_vnc(substrate)
    vnc.core.generator.manual_seed(int(noise_seed))
    vnc.reset()
    current = vnc.descending_current(DRIVE)
    steps = int(round(SECS / SC.SIGNAL_SAMPLE_DT))
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)]).astype(np.float32)
    if history.shape != (steps, len(vnc.io["motor_out"])) or not np.isfinite(history).all():
        raise AssertionError(f"invalid VNC realization {substrate.name}:{substrate.seed}:{noise_seed}")
    scores = clean_scores(leg_scores(history, vnc.io))
    activity = {
        "frac_active": float(np.mean(history > 0.0)),
        "motor_hz": float(history.mean() * float(vnc.core.rate_ceiling_hz)),
        "rate_ceiling_hz": float(vnc.core.rate_ceiling_hz),
    }
    return scores, activity, float(vnc.core.rate_ceiling_hz), vnc.io


def score_diagnostics(scores: np.ndarray, SC) -> dict:
    from probe_vnc_phase import band_power_share, phase_metrics

    joint = scores[..., 0]
    phase = phase_metrics(
        joint,
        SC.SIGNAL_SAMPLE_DT,
        present_label="MEASURED",
        absent_label="MEASURED_PHASE_NOT_TRIPOD",
    )
    band = band_power_share(joint, SC.SIGNAL_SAMPLE_DT)
    return {
        "tripod_coherence_scalar": float(phase["tripod_coherence_scalar"]),
        "band_power_share": float(band["band_power_share_mean"]),
        "phase_verdict": phase["verdict"],
    }


def matched_scores(scores: np.ndarray, target_sd: np.ndarray):
    # Reuse registered H1 machinery at the actual CPG input: each interface
    # consumes the mean of five joint scores for each of six legs.  Mean and
    # temporal pattern stay intact; irrelevant absent joint pools cannot make a
    # live leg look dead.
    from audit_amplitude_control import amplitude_match

    leg_activity = np.nan_to_num(scores.mean(axis=2), nan=0.0)
    flat, record = amplitude_match(
        leg_activity, target_sd.reshape(-1), allow_unmatchable=False
    )
    record["domain"] = "6 fixed per-leg mean leg-score channels"
    result = clean_scores(np.repeat(flat[:, :, None], 5, axis=2))
    record["matched_score_sd_mean"] = float(result.mean(axis=2).std(axis=0).mean())
    return result, record


def assert_targets(targets: np.ndarray, model, SC, mujoco) -> None:
    steps = int(round(SECS / SC.SIGNAL_SAMPLE_DT))
    if targets.shape != (steps, 18) or not np.isfinite(targets).all():
        raise AssertionError(f"bad target shape/values {targets.shape}")
    ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg_{leg}_joint_{joint}")
        for leg in range(6) for joint in range(3)
    ])
    if np.any(ids < 0):
        raise AssertionError("body model missing named leg joints")
    ranges = np.asarray(model.jnt_range[ids], dtype=np.float64)
    flat = targets.astype(np.float64)
    if np.any(flat < ranges[:, 0] - 1e-6) or np.any(flat > ranges[:, 1] + 1e-6):
        raise AssertionError("joint targets outside MuJoCo ranges")


def summary(rows: list[dict], signal: dict | None, diagnostics: dict | None, n_realizations: int) -> dict:
    if not rows:
        raise AssertionError("empty rollout arm")
    metrics = ("forward_distance_m", "path_m", "displacement_m", "min_height_m", "mean_feet_planted", "path_per_forward")
    result = {
        "n": len(rows),
        "falls": int(sum(bool(row["fell"]) for row in rows)),
        "fall_rate": float(np.mean([bool(row["fell"]) for row in rows])),
        "metrics": {},
        "poisson_realizations": int(n_realizations),
    }
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows if row[metric] is not None], dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():
            result["metrics"][metric] = {"mean": None, "sd": None, "n": int(len(values))}
        else:
            result["metrics"][metric] = {"mean": float(values.mean()), "sd": float(values.std()), "n": int(len(values))}
    if signal is not None:
        result["signal"] = signal
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    return result


def mean_signal(records: list[dict]) -> dict:
    keys = ("frac_active", "motor_hz", "rate_ceiling_hz")
    return {key: float(np.mean([record["activity"][key] for record in records])) for key in keys}


def mean_diagnostics(records: list[dict]) -> dict:
    return {
        "tripod_coherence_scalar": float(np.mean([record["diagnostics"]["tripod_coherence_scalar"] for record in records])),
        "band_power_share": float(np.mean([record["diagnostics"]["band_power_share"] for record in records])),
        "phase_verdict_counts": {
            verdict: int(sum(record["diagnostics"]["phase_verdict"] == verdict for record in records))
            for verdict in sorted({record["diagnostics"]["phase_verdict"] for record in records})
        },
    }


def body_rows(SC, targets, io, specs, model, body_record, budget, metadata: dict) -> list[dict]:
    rows = []
    for spec in specs:
        measured = min(
            SC.run_body_trace(targets, io, spec, SECS, budget=budget, model=model, body_record=body_record)["horizons"].values(),
            key=lambda row: abs(float(row["horizon_s"]) - SECS),
        )
        forward = float(measured["forward_distance_m"])
        path = float(measured["path_m"])
        rows.append({
            **spec,
            **metadata,
            **measured,
            "path_per_forward": float(path / forward) if forward > 0.0 else None,
        })
    return rows


def self_check() -> None:
    assert len(REAL_NOISE_SEEDS) == REAL_REALIZATIONS == 30
    assert len(CONTROL_NOISE_SEEDS) == CONTROL_REALIZATIONS == 10
    assert set(SELECTED_SCALES) == set(VARIANTS)
    assert all(float(value) > 0.0 for value in SELECTED_SCALES.values())
    assert EPISODES >= 20
    print("cpg_specificity: protocol self-check passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--step-path", type=Path, required=False)
    parser.add_argument("--out", type=Path, default=RESULTS / "cpg_specificity_variants.json")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.step_path is None:
        raise ValueError("--step-path is required for rollout")
    if args.episodes < EPISODES:
        raise ValueError("final confirmation requires at least 20 episodes")

    os.environ["SC_CORE"] = "lif"
    os.environ.setdefault("SC_DEVICE", "cuda")
    import mujoco
    import specificity_campaign as SC
    from audit_vnc_legs import leg_scores
    from cpg_variants import variant_trace
    from cpg_connectome_walk import signal_summary
    from reservoir_3way import tensor_sha256

    graph_hashes_pre = {name: file_sha256(SC.VNC_DATA / name) for name in EXPECTED_FILE_HASHES}
    if graph_hashes_pre != EXPECTED_FILE_HASHES:
        raise AssertionError(f"frozen graph hash mismatch: {graph_hashes_pre}")
    reference = SC.base_vnc(SC.VNC_DATA)
    tensor_hash_pre = tensor_sha256(reference.W)
    if tensor_hash_pre != EXPECTED_TENSOR_HASH:
        raise AssertionError(f"frozen tensor hash mismatch: {tensor_hash_pre}")
    if SC.CTRL_HZ != 50.0 or SC.SIGNAL_SAMPLE_DT != 0.02 or SC.SETTLE_S != 0.7:
        raise AssertionError("body/control protocol drifted")
    specs = SC.episode_specs(args.episodes, SC.EPISODE_SEED)
    episode_hash = spec_hash(specs)
    model, body_record = SC.make_body_model()
    substrates, substrate_record = SC.build_vnc_substrates(SC.VNC_DATA, SC.SUBSTRATE_CACHE)
    real_substrate = next(item for item in substrates if item.name == "real")
    controls = [item for item in substrates if item.name != "real"]
    if len(controls) != 24:
        raise AssertionError(f"expected 24 control graphs, got {len(controls)}")

    # The 30-realization pooled SD is fixed before any body rollout is measured.
    real_records = []
    rate_ceiling = None
    for noise_seed in REAL_NOISE_SEEDS:
        scores, activity, rate, io = collect_realization(SC, real_substrate, noise_seed, leg_scores)
        if rate_ceiling is None:
            rate_ceiling = rate
        if rate != rate_ceiling:
            raise AssertionError("rate ceiling changed across realizations")
        real_records.append({
            "scores": scores,
            "activity": activity,
            "diagnostics": score_diagnostics(scores, SC),
            "noise_seed": int(noise_seed),
        })
    real_score_sd = np.concatenate([record["scores"].mean(axis=2) for record in real_records], axis=0).std(axis=0)
    real_score_mean = np.mean(np.stack([record["scores"] for record in real_records]), axis=0)

    # Ten Poisson realizations are averaged per graph before the 20 shared body
    # episodes.  Individual probes below retain the realization distribution.
    control_records = {}
    for substrate in controls:
        records = []
        for noise_seed in CONTROL_NOISE_SEEDS:
            scores, activity, rate, io = collect_realization(SC, substrate, noise_seed, leg_scores)
            if rate != rate_ceiling:
                raise AssertionError("control rate ceiling changed")
            matched, amplitude_record = matched_scores(scores, real_score_sd)
            records.append({
                "scores": matched,
                "activity": activity,
                "diagnostics": score_diagnostics(matched, SC),
                "raw_diagnostics": score_diagnostics(scores, SC),
                "noise_seed": int(noise_seed),
                "amplitude_match": amplitude_record,
            })
        control_records[(substrate.name, substrate.seed)] = records
        print(f"prepared {substrate.name} graph_seed={substrate.seed} with {len(records)} Poisson realizations", flush=True)

    target_reference = {
        "constant_connectome": np.repeat(real_score_mean.mean(axis=0, keepdims=True), len(real_score_mean), axis=0),
        "time_shuffled_connectome": real_score_mean[np.random.default_rng(TIME_SHUFFLE_SEED).permutation(len(real_score_mean))],
    }
    budget = SC.Budget()
    planned_rollouts = len(VARIANTS) * (
        args.episodes * (1 + len(controls) + 3) + len(REAL_NOISE_SEEDS) + len(controls) * len(CONTROL_NOISE_SEEDS)
    ) + 3
    if planned_rollouts > SC.MAX_ROLLOUTS:
        raise AssertionError(f"planned rollout count {planned_rollouts} exceeds {SC.MAX_ROLLOUTS}")

    rows = []
    variant_reports = {}
    control_summaries_by_variant = {}
    video_targets = {}
    for variant in VARIANTS:
        scale = SELECTED_SCALES[variant]
        ladder = {}
        control_summaries = {"random": [], "rewired": []}
        realization_probe_rows = []

        # Primary real arm uses the mean of 30 frozen [SH] realizations.
        real_target, real_params = variant_trace(SC, model, real_score_mean, rate_ceiling, args.step_path, variant, scale)
        assert_targets(real_target, model, SC, mujoco)
        real_metadata = {
            "variant": variant,
            "scale": scale,
            "arm": "full",
            "substrate": "real",
            "graph_seed": None,
            "noise_seed": "mean_of_30_realizations",
            "poisson_realizations": REAL_REALIZATIONS,
            "amplitude_matched": False,
            **mean_signal(real_records),
            **mean_diagnostics(real_records),
        }
        real_rows = body_rows(SC, real_target, io, specs, model, body_record, budget, real_metadata)
        rows.extend(real_rows)
        real_ladder = {
            "summary": summary(real_rows, mean_signal(real_records), mean_diagnostics(real_records), REAL_REALIZATIONS),
            "parameter_range_fraction": float(real_params["parameter_range_fraction"]),
        }
        ladder["full"] = real_ladder
        video_targets[(variant, "full", None)] = real_target

        # Realization probes use fixed episode 0, only to estimate the real arm's
        # own Poisson SD beside each control comparison.
        for record in real_records:
            target, params = variant_trace(SC, model, record["scores"], rate_ceiling, args.step_path, variant, scale)
            assert_targets(target, model, SC, mujoco)
            probe_metadata = {
                "variant": variant,
                "scale": scale,
                "arm": "realization_probe",
                "substrate": "real",
                "graph_seed": None,
                "noise_seed": record["noise_seed"],
                "poisson_realizations": 1,
                "amplitude_matched": False,
                **record["activity"],
                **record["diagnostics"],
            }
            probe = body_rows(SC, target, io, specs[:1], model, body_record, budget, probe_metadata)
            realization_probe_rows.extend(probe)
            rows.extend(probe)

        for arm, score_trace in target_reference.items():
            target, params = variant_trace(SC, model, score_trace, rate_ceiling, args.step_path, variant, scale)
            assert_targets(target, model, SC, mujoco)
            metadata = {
                "variant": variant,
                "scale": scale,
                "arm": arm,
                "substrate": "real",
                "graph_seed": None,
                "noise_seed": "mean_of_30_realizations",
                "poisson_realizations": REAL_REALIZATIONS,
                "amplitude_matched": False,
                **mean_signal(real_records),
                **mean_diagnostics(real_records),
            }
            arm_rows = body_rows(SC, target, io, specs, model, body_record, budget, metadata)
            rows.extend(arm_rows)
            ladder[arm] = {
                "summary": summary(arm_rows, mean_signal(real_records), mean_diagnostics(real_records), REAL_REALIZATIONS),
                "parameter_range_fraction": float(params["parameter_range_fraction"]),
            }
            video_targets[(variant, arm, None)] = target

        target, params = variant_trace(SC, model, None, rate_ceiling, args.step_path, variant, scale)
        assert_targets(target, model, SC, mujoco)
        metadata = {
            "variant": variant,
            "scale": scale,
            "arm": "cpg_only",
            "substrate": "none",
            "graph_seed": None,
            "noise_seed": None,
            "poisson_realizations": 0,
            "amplitude_matched": False,
            "frac_active": 0.0,
            "motor_hz": 0.0,
            "rate_ceiling_hz": rate_ceiling,
            "tripod_coherence_scalar": None,
            "band_power_share": None,
        }
        cpg_rows = body_rows(SC, target, io, specs, model, body_record, budget, metadata)
        rows.extend(cpg_rows)
        ladder["cpg_only"] = {
            "summary": summary(cpg_rows, {"frac_active": 0.0, "motor_hz": 0.0, "rate_ceiling_hz": rate_ceiling}, None, 0),
            "parameter_range_fraction": float(params["parameter_range_fraction"]),
        }
        video_targets[(variant, "cpg_only", None)] = target

        # One averaged target per graph is the primary control arm.  Ten
        # realization probes keep control noise visible without spending a body
        # rollout for every point in the 12k harness budget.
        for substrate in controls:
            records = control_records[(substrate.name, substrate.seed)]
            matched_mean = np.mean(np.stack([record["scores"] for record in records]), axis=0)
            target, params = variant_trace(SC, model, matched_mean, rate_ceiling, args.step_path, variant, scale)
            assert_targets(target, model, SC, mujoco)
            metadata = {
                "variant": variant,
                "scale": scale,
                "arm": "full",
                "substrate": substrate.name,
                "graph_seed": int(substrate.seed),
                "noise_seed": "mean_of_10_realizations",
                "poisson_realizations": CONTROL_REALIZATIONS,
                "amplitude_matched": True,
                **mean_signal(records),
                **mean_diagnostics(records),
            }
            control_rows = body_rows(SC, target, io, specs, model, body_record, budget, metadata)
            rows.extend(control_rows)
            graph_summary = {
                "condition": metadata,
                "summary": summary(control_rows, mean_signal(records), mean_diagnostics(records), CONTROL_REALIZATIONS),
                "parameter_range_fraction": float(params["parameter_range_fraction"]),
                "amplitude_match": {
                    "max_sd_error_all_channels": float(max(record["amplitude_match"]["max_sd_error_all_channels"] for record in records)),
                    "mean_preserved_max_error": float(max(record["amplitude_match"]["mean_preserved_max_error"] for record in records)),
                    "unmatchable_channels": int(max(record["amplitude_match"]["unmatchable_channels"] for record in records)),
                },
            }
            control_summaries[substrate.name].append(graph_summary)
            video_targets[(variant, "full", (substrate.name, substrate.seed))] = target

            for record in records:
                target, _ = variant_trace(SC, model, record["scores"], rate_ceiling, args.step_path, variant, scale)
                assert_targets(target, model, SC, mujoco)
                probe_metadata = {
                    "variant": variant,
                    "scale": scale,
                    "arm": "control_realization_probe",
                    "substrate": substrate.name,
                    "graph_seed": int(substrate.seed),
                    "noise_seed": record["noise_seed"],
                    "poisson_realizations": 1,
                    "amplitude_matched": True,
                    **record["activity"],
                    **record["diagnostics"],
                }
                rows.extend(body_rows(SC, target, io, specs[:1], model, body_record, budget, probe_metadata))

        realization_values = np.asarray([row["forward_distance_m"] for row in realization_probe_rows], dtype=np.float64)
        real_forward = ladder["full"]["summary"]["metrics"]["forward_distance_m"]["mean"]
        real_sd = float(realization_values.std())
        comparisons = {}
        for kind in ("random", "rewired"):
            values = np.asarray([
                item["summary"]["metrics"]["forward_distance_m"]["mean"]
                for item in control_summaries[kind]
            ], dtype=np.float64)
            comparisons[kind] = {
                "n_graph_seeds": int(len(values)),
                "n_poisson_realizations_per_graph": CONTROL_REALIZATIONS,
                "control_graph_seed_values_m": values.tolist(),
                "control_mean_m": float(values.mean()),
                "control_sd_across_graph_means_m": float(values.std()),
                "real_mean_m": float(real_forward),
                "real_minus_control_mean_m": float(real_forward - values.mean()),
                "real_realization_sd_m": real_sd,
                "real_beats_by_more_than_real_sd": bool(real_forward - values.mean() > real_sd),
            }
        a_minus_b = real_forward - ladder["constant_connectome"]["summary"]["metrics"]["forward_distance_m"]["mean"]
        ratio = ladder["full"]["summary"]["metrics"]["path_per_forward"]["mean"]
        beats_controls = all(item["real_beats_by_more_than_real_sd"] for item in comparisons.values())
        walking = real_forward >= 0.5 and ratio <= 2.5
        success = bool(walking and beats_controls and a_minus_b > 0.0)
        variant_report = {
            "scale": scale,
            "ablation_ladder": ladder,
            "control_comparison": comparisons,
            "realization_probe": {
                "n": REAL_REALIZATIONS,
                "episode": 0,
                "forward_sd_m": real_sd,
                "forward_mean_m": float(realization_values.mean()),
            },
            "gates": {
                "forward_m_ge_0_5": bool(real_forward >= 0.5),
                "path_per_forward_le_2_5": bool(ratio <= 2.5),
                "beats_both_controls_by_real_sd": beats_controls,
                "full_beats_constant_connectome": bool(a_minus_b > 0.0),
                "success": success,
            },
            "self_interrogation": {
                "a_full_minus_constant_forward_m": float(a_minus_b),
                "b_parameter_range_fraction": {
                    "real_full_mean_trace": ladder["full"]["parameter_range_fraction"],
                    "constant": ladder["constant_connectome"]["parameter_range_fraction"],
                    "time_shuffled": ladder["time_shuffled_connectome"]["parameter_range_fraction"],
                    "cpg_only": ladder["cpg_only"]["parameter_range_fraction"],
                },
                "c_constant_selected_from_forward": False,
                "c_constant_selection_note": "scale fixed from corrected Iteration 7 grid; no confirmation tuning",
                "d_identical_episode_specs": True,
                "d_episode_spec_hash": episode_hash,
                "e_weight_hashes_match": tensor_hash_pre == EXPECTED_TENSOR_HASH,
                "f_realizations_per_arm": {
                    "real": REAL_REALIZATIONS,
                    "random_per_graph": CONTROL_REALIZATIONS,
                    "rewired_per_graph": CONTROL_REALIZATIONS,
                },
                "f_real_arm_sd_m": real_sd,
                "g_rewire_replacement_number": comparisons["rewired"]["control_mean_m"],
                "g_random_replacement_number": comparisons["random"]["control_mean_m"],
                "h_claim_guard": "connectome modulates a hand-written biologically-derived CPG gait; it does not generate the gait",
            },
        }
        variant_reports[variant] = variant_report
        control_summaries_by_variant[variant] = control_summaries
        print(json.dumps({"variant": variant, "forward_m": real_forward, "path_per_forward": ratio, "success": success}, sort_keys=True), flush=True)

    graph_hashes_post = {name: file_sha256(SC.VNC_DATA / name) for name in EXPECTED_FILE_HASHES}
    tensor_hash_post = tensor_sha256(reference.W)
    if graph_hashes_post != graph_hashes_pre or tensor_hash_post != tensor_hash_pre:
        raise AssertionError("frozen graph changed during final confirmation")

    # Video choice is post-numeric and never feeds back into any gate.
    videos = {}
    best_variant = max(VARIANTS, key=lambda name: variant_reports[name]["ablation_ladder"]["full"]["summary"]["metrics"]["forward_distance_m"]["mean"])
    for kind in ("random", "rewired"):
        candidates = [item for item in control_summaries_by_variant[best_variant][kind]]
        selected = candidates[len(candidates) // 2]
        key = (best_variant, "full", (kind, selected["condition"]["graph_seed"]))
        path = args.out.with_name(f"{args.out.stem}_{best_variant}_{kind}_median_multiview.mp4")
        rendered = SC.run_body_trace(video_targets[key], io, specs[0], SECS, video=path, budget=budget, model=model, body_record=body_record)
        videos[kind] = {"path": str(path), "graph_seed": selected["condition"]["graph_seed"], "metrics": min(rendered["horizons"].values(), key=lambda row: abs(float(row["horizon_s"]) - SECS))}
    real_path = args.out.with_name(f"{args.out.stem}_{best_variant}_real_multiview.mp4")
    rendered = SC.run_body_trace(video_targets[(best_variant, "full", None)], io, specs[0], SECS, video=real_path, budget=budget, model=model, body_record=body_record)
    videos["real"] = {"path": str(real_path), "metrics": min(rendered["horizons"].values(), key=lambda row: abs(float(row["horizon_s"]) - SECS))}

    winner = max(
        ((variant_reports[name]["ablation_ladder"]["full"]["summary"]["metrics"]["forward_distance_m"]["mean"], name) for name in VARIANTS),
        key=lambda item: item[0],
    )
    any_success = any(variant_reports[name]["gates"]["success"] for name in VARIANTS)
    walking_any = any(
        variant_reports[name]["gates"]["forward_m_ge_0_5"] and variant_reports[name]["gates"]["path_per_forward_le_2_5"]
        for name in VARIANTS
    )
    decision = "CPG-INTERFACE-SPECIFICITY-EARNED" if any_success else "CPG-INTERFACE-SPECIFICITY-NULL" if walking_any else "CPG-INTERFACE-WALKING-NOT-SUPPORTED"
    protocol = {
        "secs": SECS,
        "drive": DRIVE,
        "episodes": args.episodes,
        "episode_seed": SC.EPISODE_SEED,
        "episode_spec_hash": episode_hash,
        "control_hz": SC.CTRL_HZ,
        "signal_sample_dt_s": SC.SIGNAL_SAMPLE_DT,
        "settle_s": SC.SETTLE_S,
        "kp": SC.KP,
        "kd": SC.KD,
        "h_min_m": SC.H_MIN_EXPECTED,
        "map_gains": list(SC.MAP_GAINS),
        "drive_core": "Shiu et al. published stochastic LIF, SC_CORE=lif",
        "real_noise_seeds": list(REAL_NOISE_SEEDS),
        "control_noise_seeds_per_graph": list(CONTROL_NOISE_SEEDS),
        "real_poisson_realizations": REAL_REALIZATIONS,
        "control_poisson_realizations_per_graph": CONTROL_REALIZATIONS,
        "control_graph_seeds": list(SC.RNG_SEEDS),
        "amplitude_match": "audit_amplitude_control.amplitude_match on six per-leg mean leg-score channels, the actual CPG input; per-channel SD matched to pooled real scores; mean preserved",
        "amplitude_match_allow_unmatchable": False,
        "variants": {name: {"scale": SELECTED_SCALES[name], "frequency_hz": 1.2} for name in VARIANTS},
        "time_shuffle_seed": TIME_SHUFFLE_SEED,
        "cpg_seed": CPG_SEED,
        "body": body_record,
        "same_body_episode_specs": True,
        "run_body_trace_unmodified": True,
        "learning": False,
        "sensory_feedback": False,
        "cpg_role": "hand-written prior supplies rhythm and inter-leg coordination; connectome only modulates it",
        "primary_control_aggregation": "mean matched leg-score trace over 10 Poisson realizations per graph seed",
        "realization_probe": "episode 0 per realization, used only for real-arm SD and control-noise audit",
    }
    reference_departures = [
        {"reference": "flygym/src/flygym_demo/complex_terrain/preprogrammed.py:29-37", "departure": "seven fly DOFs mapped by fixed coxa/femur/tibia role to three hexapod DOFs", "reason": "PhantomX leg has three actuated joints; role mapping is anatomical and fixed before rollout"},
        {"reference": "flygym/src/flygym_demo/complex_terrain/preprogrammed.py:74-80", "departure": "adhesion DOFs omitted", "reason": "PhantomX XML has no adhesion actuators"},
        {"reference": "flygym/src/flygym_demo/complex_terrain/preprogrammed.py:47-57", "departure": "recorded excursions retain source angular size and clip at MuJoCo ranges", "reason": "source-sized retarget is the deterministic corrected port; no locomotion-fit scale"},
        {"reference": "flygym/src/flygym_demo/complex_terrain/cpg_controller.py:166-180", "departure": "hexapod leg order rebuilt from hexapod_gait tripod {0,3,4}", "reason": "literal fly-order phase matrix would prescribe wrong PhantomX tripod"},
        {"reference": "flygym/src/flygym_demo/complex_terrain/cpg_controller.py:166-180", "departure": "V1 amplitude, V2 frequency, and V3 measured-phase affine interfaces", "reason": "declared connectome bridge variants required by task; CPG constants remain [NMF] values"},
        {"reference": "flygym/src/flygym_demo/complex_terrain/cpg_controller.py:80-92", "departure": "V3 couples all off-diagonal measured phase pairs", "reason": "phase matrix is an exploratory interface variant; default tripod coupling remains unchanged"},
    ]
    artifact = {
        "experiment": "cpg_specificity_confirmation",
        "status": "complete",
        "pre_registered_null": "CPG-interface forward drive is no greater with the real connectome than with degree-preserving rewires.",
        "decision": decision,
        "protocol": protocol,
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
        "ablation_ladder": {name: report["ablation_ladder"] for name, report in variant_reports.items()},
        "self_interrogation": {name: report["self_interrogation"] for name, report in variant_reports.items()},
        "variants": variant_reports,
        "substrates": substrate_record,
        "videos": videos,
        "numeric_decision": {
            "best_by_full_forward": {"variant": winner[1], "forward_m": winner[0]},
            "any_variant_success": any_success,
            "walking_achieved": walking_any,
            "decision": decision,
            "kill_rule": "walking achieved but specificity null across all three variants" if decision == "CPG-INTERFACE-SPECIFICITY-NULL" else None,
        },
        "reference_departures": reference_departures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else (_ for _ in ()).throw(TypeError(type(value).__name__))) + "\n", encoding="utf-8")
    for variant in VARIANTS:
        filtered = dict(artifact)
        filtered["variant"] = variant
        filtered["rows"] = [row for row in rows if row["variant"] == variant]
        filtered["variants"] = {variant: variant_reports[variant]}
        filtered["ablation_ladder"] = {variant: variant_reports[variant]["ablation_ladder"]}
        filtered["self_interrogation"] = {variant: variant_reports[variant]["self_interrogation"]}
        path = args.out.with_name(f"cpg_{variant}_specificity.json")
        path.write_text(json.dumps(filtered, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else (_ for _ in ()).throw(TypeError(type(value).__name__))) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "best_variant": winner[1], "best_forward_m": winner[0]}, sort_keys=True), flush=True)
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
