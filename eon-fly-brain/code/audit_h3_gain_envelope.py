"""H3: fixed two-dimensional motor-map gain envelope."""
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


PREFIX = RESULTS / "h3_gain_envelope"
SECS = 8.0
COXA_GAINS = (-0.85, -0.70, -0.55, -0.40, -0.25)
DISTAL_GAINS = (0.35, 0.50, 0.65, 0.80, 0.95)
GAIN_GRID = tuple((coxa, distal, distal) for coxa in COXA_GAINS for distal in DISTAL_GAINS)
METRICS = ("path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m")
MATCHED_MODE = "per_channel_joint_target_sd_matched_to_real_at_gain_cell"


def map_history(history: np.ndarray, io, gains: tuple[float, float, float], cache: dict) -> np.ndarray:
    key = (id(history), gains)
    if key not in cache:
        targets = np.stack([SC.translate(row, io, gains=gains) for row in history]).astype(np.float32)
        if targets.shape != (len(history), 18) or not np.isfinite(targets).all():
            raise AssertionError(f"invalid target history for gains {gains}")
        cache[key] = targets
    return cache[key]


def run_condition(substrate, mode, signal, gains, io, specs, budget, rng, mapped=False):
    targets = signal if mapped else map_history(signal, io, gains, {})
    model, body_record = SC.make_body_model()
    rows = []
    cell = f"coxa_{gains[0]:+.2f}_distal_{gains[1]:.2f}"
    condition = f"{substrate.name}_{substrate.seed if substrate.seed is not None else 'real'}_{mode}_{cell}"
    for spec in specs:
        trace = SC.run_body_trace(targets, io, spec, SECS, budget=budget, model=model, body_record=body_record)
        rows.append({
            **spec,
            "condition": condition,
            "substrate": substrate.name,
            "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
            "seed": substrate.seed,
            "coxa_gain": float(gains[0]),
            "distal_gain": float(gains[1]),
            "amplitude_mode": mode,
            "tensor_sha256": substrate.tensor_hash,
            **trace["horizons"]["8s"],
        })
    return {
        "condition": condition,
        "substrate": substrate.name,
        "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
        "seed": substrate.seed,
        "coxa_gain": float(gains[0]),
        "distal_gain": float(gains[1]),
        "gains": list(gains),
        "amplitude_mode": mode,
        "signal_domain": "joint_targets" if mapped else "vnc_motor_output",
        "tensor_sha256": substrate.tensor_hash,
        "signal_sha256": array_sha256(signal),
        "summary": SC.summarize_rows(rows, METRICS, rng),
    }, rows


def make_figure(points: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray([[next(row["real_minus_random_mean"] for row in points if row["coxa_gain"] == coxa and row["distal_gain"] == distal) for distal in DISTAL_GAINS] for coxa in COXA_GAINS])
    wins = np.asarray([[next(row["earned_cell"] for row in points if row["coxa_gain"] == coxa and row["distal_gain"] == distal) for distal in DISTAL_GAINS] for coxa in COXA_GAINS])
    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    image = axis.imshow(matrix, cmap="RdBu_r", aspect="auto")
    for i in range(len(COXA_GAINS)):
        for j in range(len(DISTAL_GAINS)):
            label = "win" if wins[i, j] else "—"
            axis.text(j, i, label, ha="center", va="center", color="black", fontsize=9)
    axis.set(xticks=range(len(DISTAL_GAINS)), xticklabels=[f"{value:.2f}" for value in DISTAL_GAINS], xlabel="distal gain (femur = tibia)", yticks=range(len(COXA_GAINS)), yticklabels=[f"{value:.2f}" for value in COXA_GAINS], ylabel="coxa gain", title="H3 real-minus-matched-random forward distance")
    fig.colorbar(image, ax=axis, label="m; 8 s")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def write_csv(rows: list[dict]) -> None:
    fields = (
        "condition", "substrate", "kind", "seed", "coxa_gain", "distal_gain", "amplitude_mode", "tensor_sha256",
        "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m", "path_m", "forward_distance_m",
        "displacement_m", "mean_feet_planted", "min_height_m", "final_height_m", "fell",
    )
    with PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def mark_videos_reviewed() -> int:
    path = PREFIX.with_suffix(".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["videos"]["reviewed"] = True
    report["videos"]["review_note"] = "real and median matched-random baseline gain-cell four-view sequences visually reviewed"
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
    target_cache = {}
    mapped_cache = {}
    summary_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    comparison_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    budget = SC.Budget()
    real = next(substrate for substrate in substrates if substrate.name == "real")
    random_substrates = [substrate for substrate in substrates if substrate.name == "random"]
    records = []
    rows = []
    for coxa, distal, tibia in GAIN_GRID:
        gains = (coxa, distal, tibia)
        real_motor = SC.collect_history(real, SECS, SC.BASE_DRIVE, history_cache)
        real_targets = map_history(real_motor, io, gains, mapped_cache)
        real_record, real_rows = run_condition(real, "raw", real_motor, gains, io, specs, budget, summary_rng)
        records.append(real_record)
        rows.extend(real_rows)
        for substrate in random_substrates:
            motor_history = SC.collect_history(substrate, SECS, SC.BASE_DRIVE, history_cache)
            source_targets = map_history(motor_history, io, gains, mapped_cache)
            matched_targets, match_record = amplitude_match(source_targets, real_targets.std(axis=0))
            match_record["source_motor_history_sha256"] = array_sha256(motor_history)
            match_record["source_joint_target_history_sha256"] = array_sha256(source_targets)
            match_record["reference_gains"] = list(gains)
            matched_record, matched_rows = run_condition(substrate, MATCHED_MODE, matched_targets, gains, io, specs, budget, summary_rng, mapped=True)
            matched_record["amplitude_match"] = match_record
            records.append(matched_record)
            rows.extend(matched_rows)
        print(f"completed coxa={coxa:+.2f} distal={distal:.2f}", flush=True)

    def get(name, seed, gains, mode):
        return next(row for row in records if row["substrate"] == name and row["seed"] == seed and tuple(row["gains"]) == tuple(gains) and row["amplitude_mode"] == mode)

    points = []
    for coxa, distal, tibia in GAIN_GRID:
        gains = (coxa, distal, tibia)
        real_record = get("real", None, gains, "raw")
        metric = real_record["summary"]["metrics"]["forward_distance_m"]
        matched = [get("random", seed, gains, MATCHED_MODE) for seed in SC.RNG_SEEDS]
        point = SC.substrate_comparison(metric["mean"], [row["summary"]["metrics"]["forward_distance_m"]["mean"] for row in matched], f"forward_distance_m_coxa_{coxa:+.2f}_distal_{distal:.2f}", comparison_rng, (metric["ci_low"], metric["ci_high"]))
        point.update({"coxa_gain": float(coxa), "distal_gain": float(distal), "gains": list(gains)})
        points.append(point)
    adjusted = SC.holm_adjust([point["p_raw"] for point in points])
    for point, p_holm in zip(points, adjusted):
        point["p_holm"] = float(p_holm)
        point["earned_cell"] = bool(point["p_holm"] < 0.05 and point["real_outside_random_distribution"] and point["nonoverlap_bootstrap_ci"])

    matched_records = [row for row in records if row["amplitude_mode"] == MATCHED_MODE]
    max_sd_error = max(row["amplitude_match"]["max_sd_error_all_channels"] for row in matched_records)
    max_mean_error = max(row["amplitude_match"]["mean_preserved_max_error"] for row in matched_records)
    baseline = next(point for point in points if point["coxa_gain"] == -0.55 and point["distal_gain"] == 0.65)
    median_seed = int(np.argsort([get("random", seed, (-0.55, 0.65, 0.65), MATCHED_MODE)["summary"]["metrics"]["forward_distance_m"]["mean"] for seed in SC.RNG_SEEDS])[len(SC.RNG_SEEDS) // 2])
    selected_seed = SC.RNG_SEEDS[median_seed]
    random_median = next(substrate for substrate in random_substrates if substrate.seed == selected_seed)
    real_motor = SC.collect_history(real, SECS, SC.BASE_DRIVE, history_cache)
    random_motor = SC.collect_history(random_median, SECS, SC.BASE_DRIVE, history_cache)
    real_signal = map_history(real_motor, io, (-0.55, 0.65, 0.65), mapped_cache)
    random_signal = amplitude_match(map_history(random_motor, io, (-0.55, 0.65, 0.65), mapped_cache), real_signal.std(axis=0))[0]
    video_paths = {
        "real": RESULTS / "h3_gain_real_baseline_multiview.mp4",
        "random_median": RESULTS / "h3_gain_matched_random_baseline_median_multiview.mp4",
    }
    for substrate, signal, path in ((real, real_signal, video_paths["real"]), (random_median, random_signal, video_paths["random_median"])):
        model, body_record = SC.make_body_model()
        SC.run_body_trace(signal, io, specs[0], SECS, video=path, budget=budget, model=model, body_record=body_record)
    figure = RESULTS / "h3_gain_envelope.png"
    make_figure(points, figure)
    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    real_tensor_hash = real.tensor_hash
    report = {
        "experiment": "H3",
        "status": "complete",
        "pre_registered_null": "gain win occupies a thin arbitrary sliver; disclose severe limitation",
        "protocol": {
            "secs": SECS,
            "drive": SC.BASE_DRIVE,
            "grid": "5x5 coxa gain by shared femur/tibia distal gain",
            "coxa_gains": list(COXA_GAINS),
            "distal_gains": list(DISTAL_GAINS),
            "shared_episode_starts": True,
            "n_episodes": SC.N_EPISODES,
            "primary": "cellwise real-minus-random forward-distance comparison",
            "amplitude_rule": "per-channel temporal SD of fixed-map joint targets set to real joint-target SD at same gain cell",
            "controls": [MATCHED_MODE + " across 12 random-network seeds"],
            "learning": False,
        },
        "real_graph_hashes": file_hashes_pre,
        "substrates": substrate_record,
        "points": points,
        "envelope_summary": {
            "cell_count": len(points),
            "earned_cell_count": int(sum(point["earned_cell"] for point in points)),
            "earned_cells": [{"coxa_gain": point["coxa_gain"], "distal_gain": point["distal_gain"]} for point in points if point["earned_cell"]],
            "baseline_cell": baseline,
            "interpretation": "whole predeclared envelope reported; no gain cell selected for headline claim",
        },
        "figure": str(figure),
        "videos": {
            "real": {"path": str(video_paths["real"]), "exists": video_paths["real"].exists(), "coxa_gain": -0.55, "distal_gain": 0.65},
            "random_median": {"path": str(video_paths["random_median"]), "exists": video_paths["random_median"].exists(), "seed": int(selected_seed), "coxa_gain": -0.55, "distal_gain": 0.65},
            "reviewed": False,
            "review_note": "review required before final decision",
        },
        "integrity": {
            "file_hashes_pre": file_hashes_pre,
            "file_hashes_post": file_hashes_post,
            "file_hashes_equal": bool(file_hashes_pre == file_hashes_post),
            "real_tensor_sha256_pre": real_tensor_hash,
            "real_tensor_sha256_post": SC.tensor_sha256(real.weight),
            "real_tensor_hash_equal": bool(real_tensor_hash == SC.tensor_sha256(real.weight)),
            "random_seeds": list(SC.RNG_SEEDS),
            "condition_count": len(records),
            "row_count": len(rows),
            "amplitude_match_count": len(matched_records),
            "max_sd_error_all_channels": max_sd_error,
            "max_mean_preserved_error": max_mean_error,
            "all_channels_match": bool(max_sd_error <= 1e-4),
            "learning": False,
        },
        "resource": {"rollouts_completed": budget.rollouts, "max_rollouts": SC.MAX_ROLLOUTS, "wallclock_ceiling_s": SC.WALLCLOCK_CEILING_S},
        "zero_learning": True,
        "zero_learning_audit": {"learning": False, "parameter_updates": 0, "training_markers": []},
        "numeric_decision": "GAIN-ENVELOPE-REPORTED",
        "decision": "GAIN-ENVELOPE-PENDING-VIDEO",
        "rows": rows,
    }
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    write_csv(rows)
    print(json.dumps({"experiment": "H3", "envelope_summary": report["envelope_summary"], "resource": report["resource"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
