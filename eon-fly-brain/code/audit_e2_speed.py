"""E2: preregistered speed-modulation audit with amplitude-matched controls."""
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


PREFIX = RESULTS / "e2_speed"
SECS = 8.0
DRIVES = SC.DRIVES
METRICS = ("path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m")
MATCHED_MODE = "per_channel_joint_target_sd_matched_to_real_at_same_drive"


def run_condition(substrate, mode, signal, drive, io, specs, target_cache, budget, rng, mapped=False):
    targets = signal if mapped else SC.target_history(signal, io, target_cache)
    model, body_record = SC.make_body_model()
    rows = []
    condition_id = f"{substrate.name}_{substrate.seed if substrate.seed is not None else 'real'}_{mode}_drive_{drive:g}"
    for spec in specs:
        trace = SC.run_body_trace(targets, io, spec, SECS, budget=budget, model=model, body_record=body_record)
        rows.append({
            **spec,
            "condition": condition_id,
            "substrate": substrate.name,
            "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
            "seed": substrate.seed,
            "drive": float(drive),
            "amplitude_mode": mode,
            "signal_domain": "joint_targets" if mapped else "vnc_motor_output",
            "tensor_sha256": substrate.tensor_hash,
            **trace["horizons"]["8s"],
        })
    return {
        "condition": condition_id,
        "substrate": substrate.name,
        "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
        "seed": substrate.seed,
        "drive": float(drive),
        "amplitude_mode": mode,
        "signal_domain": "joint_targets" if mapped else "vnc_motor_output",
        "tensor_sha256": substrate.tensor_hash,
        "signal_sha256": array_sha256(signal),
        "summary": SC.summarize_rows(rows, METRICS, rng),
    }, rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def monotonicity(values: list[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    if np.std(values) <= 1e-12:
        return 0.0
    return float(np.corrcoef(np.arange(len(values), dtype=np.float64), average_ranks(values))[0, 1])


def slope(values: list[float]) -> float:
    return float(np.polyfit(np.asarray(DRIVES, dtype=np.float64), np.asarray(values, dtype=np.float64), 1)[0])


def write_csv(rows: list[dict]) -> None:
    fields = (
        "condition", "substrate", "kind", "seed", "drive", "amplitude_mode", "signal_domain",
        "tensor_sha256", "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m",
        "final_height_m", "fell",
    )
    with PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def mark_videos_reviewed() -> int:
    path = PREFIX.with_suffix(".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["videos"]["reviewed"] = True
    report["videos"]["review_note"] = "real and median same-drive joint-target-amplitude-matched random four-view sequences visually reviewed"
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
    b3_path = RESULTS / "specificity_B3.json"
    b3_hash_pre = SC.sha256(b3_path)
    b3 = json.loads(b3_path.read_text(encoding="utf-8"))
    if b3["protocol"]["learning"] is not False or b3["zero_learning"] is not True:
        raise AssertionError("E2 source B3 is not a zero-learning run")
    substrates, substrate_record = SC.build_vnc_substrates(SC.VNC_DATA, SC.SUBSTRATE_CACHE)
    io = SC.base_vnc(SC.VNC_DATA).io
    specs = SC.episode_specs(SC.N_EPISODES, SC.EPISODE_SEED)
    histories = {}
    target_cache = {}
    summary_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    comparison_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    budget = SC.Budget()
    records = []
    rows = []
    for drive in DRIVES:
        real = next(substrate for substrate in substrates if substrate.name == "real")
        real_motor = SC.collect_history(real, SECS, float(drive), histories)
        real_targets = SC.target_history(real_motor, io, target_cache)
        real_record, real_rows = run_condition(real, "raw", real_motor, drive, io, specs, target_cache, budget, summary_rng)
        records.append(real_record)
        rows.extend(real_rows)
        for substrate in substrates:
            if substrate.name == "real":
                continue
            motor_history = SC.collect_history(substrate, SECS, float(drive), histories)
            raw_record, raw_rows = run_condition(substrate, "raw", motor_history, drive, io, specs, target_cache, budget, summary_rng)
            records.append(raw_record)
            rows.extend(raw_rows)
            source_targets = SC.target_history(motor_history, io, target_cache)
            matched_targets, match_record = amplitude_match(source_targets, real_targets.std(axis=0))
            match_record["source_motor_history_sha256"] = array_sha256(motor_history)
            match_record["source_joint_target_history_sha256"] = array_sha256(source_targets)
            match_record["reference_drive"] = float(drive)
            matched_record, matched_rows = run_condition(substrate, MATCHED_MODE, matched_targets, drive, io, specs, target_cache, budget, summary_rng, mapped=True)
            matched_record["amplitude_match"] = match_record
            records.append(matched_record)
            rows.extend(matched_rows)
        print(f"completed drive={drive:g}", flush=True)

    def record(substrate_name, seed, drive, mode):
        return next(row for row in records if row["substrate"] == substrate_name and row["seed"] == seed and row["drive"] == float(drive) and row["amplitude_mode"] == mode)

    point_comparisons = []
    real_curve = []
    matched_random_curves = {seed: [] for seed in SC.RNG_SEEDS}
    matched_rewired_curves = {seed: [] for seed in SC.RNG_SEEDS}
    raw_curve_verification = []
    for drive in DRIVES:
        real_record = record("real", None, drive, "raw")
        matched_random = [record("random", seed, drive, MATCHED_MODE) for seed in SC.RNG_SEEDS]
        matched_rewired = [record("rewired", seed, drive, MATCHED_MODE) for seed in SC.RNG_SEEDS]
        real_metric = real_record["summary"]["metrics"]["forward_distance_m"]
        comparison = SC.substrate_comparison(
            real_metric["mean"],
            [row["summary"]["metrics"]["forward_distance_m"]["mean"] for row in matched_random],
            f"forward_distance_m_at_{drive:g}drive",
            comparison_rng,
            (real_metric["ci_low"], real_metric["ci_high"]),
        )
        comparison["drive"] = float(drive)
        point_comparisons.append(comparison)
        real_curve.append(real_metric["mean"] / SECS)
        for seed in SC.RNG_SEEDS:
            matched_random_curves[seed].append(record("random", seed, drive, MATCHED_MODE)["summary"]["metrics"]["forward_distance_m"]["mean"] / SECS)
            matched_rewired_curves[seed].append(record("rewired", seed, drive, MATCHED_MODE)["summary"]["metrics"]["forward_distance_m"]["mean"] / SECS)
        old = b3["conditions"]
        old_real = next(row for row in old if row["substrate"] == "real")["sweep"][str(float(drive))]["metrics"]["forward_distance_m"]["mean"]
        raw_curve_verification.append({"drive": float(drive), "rerun": real_metric["mean"], "existing_b3": old_real, "absolute_difference": abs(real_metric["mean"] - old_real)})

    random_curve_values = [values for values in matched_random_curves.values()]
    rewired_curve_values = [values for values in matched_rewired_curves.values()]
    real_monotonicity = monotonicity(real_curve)
    random_monotonicity = [monotonicity(values) for values in random_curve_values]
    rewired_monotonicity = [monotonicity(values) for values in rewired_curve_values]
    monotonicity_comparison = SC.substrate_comparison(real_monotonicity, random_monotonicity, "forward_speed_monotonicity", comparison_rng)
    monotonicity_comparison["definition"] = "Spearman correlation between declared descending drive and forward speed across 50/100/150/200"
    # E2 earns only if real speed is monotone and every matched-random curve is flat or less monotone.
    numeric_gate = bool(
        monotonicity_comparison["p_raw"] < 0.05
        and monotonicity_comparison["real_outside_random_distribution"]
        and real_monotonicity > 0.0
        and real_monotonicity > max(random_monotonicity)
    )
    matched_records = [row for row in records if row["amplitude_mode"] == MATCHED_MODE]
    max_sd_error = max(row["amplitude_match"]["max_sd_error_all_channels"] for row in matched_records)
    max_mean_error = max(row["amplitude_match"]["mean_preserved_max_error"] for row in matched_records)

    drive_100_random = [row for row in records if row["substrate"] == "random" and row["drive"] == 100.0 and row["amplitude_mode"] == MATCHED_MODE]
    selected_random = min(drive_100_random, key=lambda row: abs(row["summary"]["metrics"]["forward_distance_m"]["mean"] - np.median([x["summary"]["metrics"]["forward_distance_m"]["mean"] for x in drive_100_random])))
    video_paths = {
        "real": RESULTS / "e2_speed_real_drive100_multiview.mp4",
        "random_median": RESULTS / "e2_speed_matched_random_drive100_median_multiview.mp4",
    }
    real = next(substrate for substrate in substrates if substrate.name == "real")
    for substrate, path in ((real, video_paths["real"]), (next(x for x in substrates if x.name == "random" and x.seed == selected_random["seed"]), video_paths["random_median"])):
        motor = SC.collect_history(substrate, SECS, 100.0, histories)
        if substrate.name == "real":
            signal = SC.target_history(motor, io, target_cache)
        else:
            signal = amplitude_match(SC.target_history(motor, io, target_cache), SC.target_history(SC.collect_history(real, SECS, 100.0, histories), io, target_cache).std(axis=0))[0]
        model, body_record = SC.make_body_model()
        SC.run_body_trace(signal, io, specs[0], SECS, video=path, budget=budget, model=model, body_record=body_record)

    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    b3_hash_post = SC.sha256(b3_path)
    real_tensor = next(substrate for substrate in substrates if substrate.name == "real")
    report = {
        "experiment": "E2",
        "status": "complete",
        "pre_registered_null": "no real-specific monotonic speed control; SPEED-MODULATION-NULL",
        "protocol": {
            "secs": SECS,
            "drives": list(DRIVES),
            "n_episodes": SC.N_EPISODES,
            "episode_seed": SC.EPISODE_SEED,
            "shared_episode_starts": True,
            "primary": "forward speed monotonicity across declared drive curve",
            "speed_definition": "8 s forward_distance_m / 8 s",
            "amplitude_rule": "per-channel temporal SD of fixed-map joint targets set to real joint-target SD at same drive",
            "controls": ["raw random-network", MATCHED_MODE, "raw rewired", "per-channel-SD amplitude-matched rewired"],
            "learning": False,
        },
        "real_graph_hashes": file_hashes_pre,
        "substrates": substrate_record,
        "records": records,
        "rows": rows,
        "point_forward_comparisons": point_comparisons,
        "point_forward_holm_p": SC.holm_adjust([row["p_raw"] for row in point_comparisons]),
        "speed_curves_m_per_s": {
            "real": real_curve,
            "matched_random": {str(seed): values for seed, values in matched_random_curves.items()},
            "matched_rewired": {str(seed): values for seed, values in matched_rewired_curves.items()},
        },
        "monotonicity": {
            "real_spearman": real_monotonicity,
            "matched_random_spearman": random_monotonicity,
            "matched_rewired_spearman": rewired_monotonicity,
            "real_slope_m_per_s_per_drive": slope(real_curve),
            "matched_random_slopes": [slope(values) for values in random_curve_values],
            "matched_rewired_slopes": [slope(values) for values in rewired_curve_values],
            "real_vs_matched_random": monotonicity_comparison,
        },
        "raw_b3_rerun_check": raw_curve_verification,
        "videos": {
            "real": {"path": str(video_paths["real"]), "exists": video_paths["real"].exists()},
            "random_median": {"path": str(video_paths["random_median"]), "exists": video_paths["random_median"].exists(), "seed": int(selected_random["seed"])},
            "reviewed": False,
            "review_note": "review required before final decision",
        },
        "integrity": {
            "file_hashes_pre": file_hashes_pre,
            "file_hashes_post": file_hashes_post,
            "file_hashes_equal": bool(file_hashes_pre == file_hashes_post),
            "b3_input_sha256_pre": b3_hash_pre,
            "b3_input_sha256_post": b3_hash_post,
            "b3_input_hash_equal": bool(b3_hash_pre == b3_hash_post),
            "real_tensor_sha256_pre": real_tensor.tensor_hash,
            "real_tensor_sha256_post": SC.tensor_sha256(real_tensor.weight),
            "random_seeds": list(SC.RNG_SEEDS),
            "rewired_seeds": list(SC.RNG_SEEDS),
            "condition_count": len(records),
            "row_count": len(rows),
            "amplitude_match_count": len(matched_records),
            "max_sd_error_all_channels": max_sd_error,
            "max_mean_preserved_error": max_mean_error,
            "all_channels_match": bool(max_sd_error <= 1e-4),
            "learning": False,
        },
        "zero_learning": True,
        "zero_learning_audit": {"learning": False, "parameter_updates": 0, "training_markers": []},
        "numeric_decision": "SPEED-MODULATION-EARNED" if numeric_gate else "SPEED-MODULATION-NULL",
        "decision": "SPEED-MODULATION-PENDING-VIDEO" if numeric_gate else "SPEED-MODULATION-NULL",
    }
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    write_csv(rows)
    print(json.dumps({"experiment": "E2", "numeric_decision": report["numeric_decision"], "monotonicity": report["monotonicity"], "resource": {"rollouts_completed": budget.rollouts, "max_rollouts": SC.MAX_ROLLOUTS}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
