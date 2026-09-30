"""E3: amplitude-matched real-vs-random body generalization."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
import specificity_campaign as SC  # noqa: E402
from audit_amplitude_control import amplitude_match, array_sha256  # noqa: E402


# `SC.ARM_SUFFIX` is empty for the rate core and "_lif" for the published one.
# Without it a run under SC_CORE=lif silently overwrites the recorded
# rate-core artifact of the same name, which is exactly what happened on the
# pod before this line existed -- only audit_amplitude_control had it.
PREFIX = RESULTS / f"e3_body_generalization_amplitude_matched{SC.ARM_SUFFIX}"
SECS = 8.0
METRICS = ("path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m")
MATCHED_MODE = "per_channel_joint_target_sd_matched_to_real"
PERTURBATIONS = SC.B6_PERTURBATIONS


def perturbation_label(perturbation: tuple[str, float]) -> str:
    return f"{perturbation[0]}_x{perturbation[1]:g}"


def run_condition(
    substrate: SC.Substrate,
    signal: np.ndarray,
    io,
    specs: list[dict],
    perturbation: tuple[str, float],
    budget: SC.Budget,
    rng: np.random.Generator,
    mode: str,
) -> tuple[dict, list[dict]]:
    model, body_record = SC.make_body_model(perturbation)
    rows = []
    condition = f"{substrate.name}_{substrate.seed if substrate.seed is not None else 'real'}_{perturbation_label(perturbation)}"
    for spec in specs:
        trace = SC.run_body_trace(signal, io, spec, SECS, perturbation=perturbation, budget=budget, model=model, body_record=body_record)
        rows.append({
            **spec,
            "condition": condition,
            "substrate": substrate.name,
            "kind": "real" if substrate.name == "real" else "random-network",
            "seed": substrate.seed,
            "tensor_sha256": substrate.tensor_hash,
            "amplitude_mode": mode,
            "perturbation_kind": perturbation[0],
            "perturbation_factor": float(perturbation[1]),
            **trace["horizons"]["8s"],
        })
    return {
        "condition": condition,
        "substrate": substrate.name,
        "kind": "real" if substrate.name == "real" else "random-network",
        "seed": substrate.seed,
        "tensor_sha256": substrate.tensor_hash,
        "amplitude_mode": mode,
        "perturbation": {"kind": perturbation[0], "factor": float(perturbation[1]), "label": perturbation_label(perturbation)},
        "signal_sha256": array_sha256(signal),
        "summary": SC.summarize_rows(rows, METRICS, rng),
    }, rows


def summary_metric(record: dict, metric: str = "forward_distance_m") -> float:
    return float(record["summary"]["metrics"][metric]["mean"])


def write_csv(rows: list[dict]) -> None:
    fields = (
        "condition", "substrate", "kind", "seed", "tensor_sha256", "amplitude_mode",
        "perturbation_kind", "perturbation_factor", "episode", "episode_seed",
        "initial_yaw_rad", "prey_x_m", "prey_y_m", "path_m", "forward_distance_m",
        "displacement_m", "mean_feet_planted", "min_height_m", "final_height_m", "fell",
    )
    with PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def make_figure(per_perturbation: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [item["label"] for item in per_perturbation]
    real = [item["real_value"] for item in per_perturbation]
    random = [item["random_mean"] for item in per_perturbation]
    low = [item["random_distribution_ci_low"] for item in per_perturbation]
    high = [item["random_distribution_ci_high"] for item in per_perturbation]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
    axis.errorbar(x, real, fmt="o-", color="#0072B2", label="real")
    axis.errorbar(x, random, yerr=[np.asarray(random) - np.asarray(low), np.asarray(high) - np.asarray(random)], fmt="s--", color="#D55E00", label="matched random mean ± seed percentile")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xticks=x, xticklabels=labels, ylabel="forward distance (m; 8 s)", title="E3 body generalization: amplitude-matched random controls")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(frameon=False)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def mark_videos_reviewed() -> int:
    path = PREFIX.with_suffix(".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["videos"]["reviewed"] = True
    report["videos"]["review_note"] = "nominal real and matched-random four-view sequences visually reviewed"
    report["decision"] = report["numeric_decision"]
    SC.atomic_json(path, report)
    write_csv(report["rows"])
    print(json.dumps({"decision": report["decision"], "videos_reviewed": True}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark-videos-reviewed", action="store_true")
    args = parser.parse_args()
    if args.mark_videos_reviewed:
        return mark_videos_reviewed()
    if PREFIX.with_suffix(".json").exists():
        raise FileExistsError(f"{PREFIX.with_suffix('.json')} exists")

    file_hashes_pre = SC.file_hashes(SC.VNC_DATA)
    substrates, substrate_record = SC.build_vnc_substrates(SC.VNC_DATA, SC.SUBSTRATE_CACHE)
    io = SC.base_vnc(SC.VNC_DATA).io
    specs = SC.episode_specs(SC.N_EPISODES, SC.EPISODE_SEED)
    history_cache = {}
    summary_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    comparison_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    budget = SC.Budget()
    real = next(item for item in substrates if item.name == "real")
    randoms = [item for item in substrates if item.name == "random"]
    real_history = SC.collect_history(real, SECS, SC.BASE_DRIVE, history_cache)
    real_targets = np.stack([SC.translate(row, io, gains=SC.MAP_GAINS) for row in real_history]).astype(np.float32)
    real_sd = real_targets.std(axis=0)
    signals = {"real": real_targets}
    matched_records = []
    records = []
    rows = []
    real_by_label = {}
    random_by_label = {}

    for perturbation in PERTURBATIONS:
        label = perturbation_label(perturbation)
        real_record, real_rows = run_condition(real, real_targets, io, specs, perturbation, budget, summary_rng, "raw")
        records.append(real_record)
        rows.extend(real_rows)
        real_by_label[label] = real_record
        for substrate in randoms:
            history = SC.collect_history(substrate, SECS, SC.BASE_DRIVE, history_cache)
            source_targets = np.stack([SC.translate(row, io, gains=SC.MAP_GAINS) for row in history]).astype(np.float32)
            matched_targets, match_record = amplitude_match(source_targets, real_sd)
            match_record["source_motor_history_sha256"] = array_sha256(history)
            match_record["source_joint_target_history_sha256"] = array_sha256(source_targets)
            match_record["reference"] = "real baseline joint-target temporal SD"
            name = f"matched_random_{substrate.seed}"
            matched_substrate = SC.Substrate(name, substrate.seed, substrate.weight, substrate.tensor_hash)
            matched_record, matched_rows = run_condition(matched_substrate, matched_targets, io, specs, perturbation, budget, summary_rng, MATCHED_MODE)
            matched_record["amplitude_match"] = match_record
            records.append(matched_record)
            rows.extend(matched_rows)
            matched_records.append(matched_record)
            random_by_label.setdefault(label, []).append(matched_record)
            signals[name] = matched_targets
        print(f"completed {label}", flush=True)

    per_perturbation = []
    for perturbation in PERTURBATIONS:
        label = perturbation_label(perturbation)
        real_record = real_by_label[label]
        random_records = random_by_label[label]
        real_metric = real_record["summary"]["metrics"]["forward_distance_m"]
        comparison = SC.substrate_comparison(summary_metric(real_record), [summary_metric(item) for item in random_records], f"forward_{label}", comparison_rng, (real_metric["ci_low"], real_metric["ci_high"]))
        per_perturbation.append({"label": label, "perturbation": {"kind": perturbation[0], "factor": float(perturbation[1])}, **comparison})
    for item, p_holm in zip(per_perturbation, SC.holm_adjust([item["p_raw"] for item in per_perturbation])):
        item["p_holm"] = float(p_holm)

    real_curve = np.asarray([item["real_value"] for item in per_perturbation])
    random_curves = np.asarray([[item["random_seed_values"][index] for item in per_perturbation] for index in range(SC.K)])
    real_worst = float(real_curve.min())
    random_worst = random_curves.min(axis=1)
    real_mean = float(real_curve.mean())
    random_mean = random_curves.mean(axis=1)
    worst_comparison = SC.substrate_comparison(real_worst, random_worst.tolist(), "worst_case_forward_distance_m", comparison_rng, None)
    mean_comparison = SC.substrate_comparison(real_mean, random_mean.tolist(), "mean_forward_distance_m", comparison_rng, None)
    significant = [item for item in per_perturbation if item["p_holm"] < 0.05 and item["real_outside_random_distribution"]]
    numeric_decision = "BODY-GENERALIZATION-EARNED" if worst_comparison["p_raw"] < 0.05 and worst_comparison["real_outside_random_distribution"] else "BODY-GENERALIZATION-NULL"
    selected_seed = int(np.median([item.seed for item in randoms]))
    nominal = perturbation_label(("leg_length", 1.0))
    selected = next(item for item in randoms if item.seed == selected_seed)
    selected_signal = signals[f"matched_random_{selected_seed}"]
    video_paths = {
        "real_nominal": RESULTS / "e3_body_real_nominal_multiview.mp4",
        "matched_random_nominal": RESULTS / "e3_body_matched_random_nominal_multiview.mp4",
    }
    for signal, perturbation, path in ((real_targets, ("leg_length", 1.0), video_paths["real_nominal"]), (selected_signal, ("leg_length", 1.0), video_paths["matched_random_nominal"])):
        model, body_record = SC.make_body_model(perturbation)
        SC.run_body_trace(signal, io, specs[0], SECS, perturbation=perturbation, video=path, budget=budget, model=model, body_record=body_record)
    figure = RESULTS / "e3_body_generalization_amplitude_matched.png"
    make_figure(per_perturbation, figure)
    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    max_sd_error = max(record["amplitude_match"]["max_sd_error_all_channels"] for record in matched_records)
    max_mean_error = max(record["amplitude_match"]["mean_preserved_max_error"] for record in matched_records)
    report = {
        "experiment": "E3",
        "status": "complete",
        "pre_registered_null": "real forward-drive advantage is body-specific or no more robust than matched random",
        "protocol": {
            "secs": SECS,
            "drive": SC.BASE_DRIVE,
            "perturbations": [{"kind": kind, "factor": factor} for kind, factor in PERTURBATIONS],
            "n_random_seeds": SC.K,
            "n_episodes": SC.N_EPISODES,
            "shared_episode_starts": True,
            "amplitude_rule": "per-channel temporal SD of fixed-map joint targets set to real baseline SD",
            "primary": "worst-case and mean forward distance across nine body perturbations",
            "learning": False,
        },
        "substrates": substrate_record,
        "records": records,
        "per_perturbation": per_perturbation,
        "robustness": {
            "real_worst_case_forward_distance_m": real_worst,
            "random_worst_case_forward_distance_m": random_worst.tolist(),
            "worst_case_comparison": worst_comparison,
            "real_mean_forward_distance_m": real_mean,
            "random_mean_forward_distance_m": random_mean.tolist(),
            "mean_comparison": mean_comparison,
            "significant_perturbation_count": len(significant),
            "significant_perturbations": [item["label"] for item in significant],
        },
        "figure": str(figure),
        "videos": {
            "real_nominal": {"path": str(video_paths["real_nominal"]), "exists": video_paths["real_nominal"].exists(), "perturbation": {"kind": "leg_length", "factor": 1.0}},
            "matched_random_nominal": {"path": str(video_paths["matched_random_nominal"]), "exists": video_paths["matched_random_nominal"].exists(), "seed": selected_seed, "perturbation": {"kind": "leg_length", "factor": 1.0}},
            "reviewed": False,
            "review_note": "review required before final decision",
        },
        "integrity": {
            "file_hashes_pre": file_hashes_pre,
            "file_hashes_post": file_hashes_post,
            "file_hashes_equal": bool(file_hashes_pre == file_hashes_post),
            "real_tensor_sha256_pre": real.tensor_hash,
            "real_tensor_sha256_post": SC.tensor_sha256(real.weight),
            "real_tensor_hash_equal": bool(real.tensor_hash == SC.tensor_sha256(real.weight)),
            "condition_count": len(records),
            "row_count": len(rows),
            "amplitude_match_count": len(matched_records),
            "max_sd_error_all_channels": max_sd_error,
            "max_mean_preserved_error": max_mean_error,
            "all_channels_match": bool(max_sd_error <= 1e-4),
            "learning": False,
        },
        "resource": {"rollouts_completed": budget.rollouts, "max_rollouts": SC.MAX_ROLLOUTS},
        "zero_learning": True,
        "zero_learning_audit": {"learning": False, "parameter_updates": 0, "training_markers": []},
        "numeric_decision": numeric_decision,
        "decision": "BODY-GENERALIZATION-PENDING-VIDEO" if numeric_decision.endswith("EARNED") else numeric_decision,
        "rows": rows,
    }
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    write_csv(rows)
    print(json.dumps({"experiment": "E3", "numeric_decision": numeric_decision, "worst_case": worst_comparison, "mean": mean_comparison, "resource": report["resource"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
