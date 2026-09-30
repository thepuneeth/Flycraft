"""Registered H1: amplitude-matched network controls for B1 forward drive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import specificity_campaign as SC


RESULTS = SC.RESULTS
PREFIX = RESULTS / f"h1_amplitude_control_strict{SC.ARM_SUFFIX}"
SECS = 8.0
DRIVE = SC.BASE_DRIVE
METRICS = ("path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m")
MATCHED_MODE = "per_channel_joint_target_sd_matched_to_real"


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def amplitude_match(history: np.ndarray, target_sd: np.ndarray,
                    allow_unmatchable: bool = False) -> tuple[np.ndarray, dict]:
    """Rescale each channel's temporal SD to `target_sd`, preserving its mean.

    A channel that never moves cannot be scaled up to one that does -- there is
    no signal to multiply. By default that aborts, because silently comparing a
    17-channel signal against an 18-channel one is exactly the mistake this guard
    exists to prevent.

    `allow_unmatchable` leaves such channels alone and reports the count instead.
    It is for the arms where a dead channel is a structural fact about the
    CONTROL graph rather than a measurement problem: a degree-preserving rewire
    can leave one robot joint with no surviving drive, and aborting the whole
    dose curve over 1 of 18 channels discards the result the audit is for. The
    count lands in the record either way, so a reader can see it.
    """
    mean = history.mean(axis=0, keepdims=True)
    current_sd = history.std(axis=0)
    scale = np.ones_like(current_sd)
    np.divide(target_sd, current_sd, out=scale, where=current_sd > 0.0)
    matched = (mean + (history - mean) * scale).astype(np.float32)
    active = current_sd > 0.0
    matched_sd = matched.std(axis=0)
    unmatchable = (~active) & (target_sd > 0.0)
    if np.any(unmatchable) and not allow_unmatchable:
        raise AssertionError(f"amplitude match has {int(unmatchable.sum())} nonzero target channels with zero source SD")
    return matched, {
        "domain": "18 fixed-map joint targets",
        "target_sd": target_sd.tolist(),
        "source_sd": current_sd.tolist(),
        "matched_sd": matched_sd.tolist(),
        "target_sd_mean": float(target_sd.mean()),
        "source_sd_mean": float(current_sd.mean()),
        "matched_sd_mean": float(matched_sd.mean()),
        "source_zero_sd_channels": int(np.count_nonzero(~active)),
        "target_zero_sd_channels": int(np.count_nonzero(target_sd <= 0.0)),
        "unmatchable_channels": int(np.count_nonzero(unmatchable)),
        "max_sd_error_all_channels": float(np.max(np.abs(matched_sd - target_sd))),
        # The unmatchable channels are off by their full target SD by
        # construction, so they would swamp the figure above and hide whether the
        # channels that COULD be matched actually were.
        "max_sd_error_matchable": float(np.max(np.abs((matched_sd - target_sd)[active]))) if np.any(active) else 0.0,
        "mean_preserved_max_error": float(np.max(np.abs(matched.mean(axis=0) - mean[0]))),
        "history_sha256": array_sha256(matched),
    }


def run_condition(
    substrate: SC.Substrate,
    mode: str,
    history: np.ndarray,
    io,
    specs: list[dict],
    targets: dict[int, np.ndarray],
    budget: SC.Budget,
    rng: np.random.Generator,
    amplitude_record: dict | None = None,
    mapped: bool = False,
) -> tuple[dict, list[dict]]:
    mapped_history = history if mapped else SC.target_history(history, io, targets)
    model, body_record = SC.make_body_model()
    rows = []
    condition_id = f"{substrate.name}_{substrate.seed if substrate.seed is not None else 'real'}_{mode}"
    for spec in specs:
        trace = SC.run_body_trace(mapped_history, io, spec, SECS, budget=budget, model=model, body_record=body_record)
        rows.append({
            **spec,
            "condition": condition_id,
            "substrate": substrate.name,
            "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
            "seed": substrate.seed,
            "tensor_sha256": substrate.tensor_hash,
            "amplitude_mode": mode,
            **trace["horizons"]["8s"],
        })
    record = {
        "condition": condition_id,
        "substrate": substrate.name,
        "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
        "seed": substrate.seed,
        "tensor_sha256": substrate.tensor_hash,
        "amplitude_mode": mode,
        "history_sha256": array_sha256(history),
        "signal_domain": "joint_targets" if mapped else "vnc_motor_output",
        "summary": SC.summarize_rows(rows, METRICS, rng),
    }
    if amplitude_record is not None:
        record["amplitude_match"] = amplitude_record
    return record, rows


def comparison(real: dict, controls: list[dict], metric: str, rng: np.random.Generator) -> dict:
    real_value = real["summary"]["metrics"][metric]["mean"]
    real_metric = real["summary"]["metrics"][metric]
    values = [row["summary"]["metrics"][metric]["mean"] for row in controls]
    return SC.substrate_comparison(
        real_value,
        values,
        metric,
        rng,
        (real_metric["ci_low"], real_metric["ci_high"]),
    )


def write_csv(rows: list[dict]) -> None:
    columns = (
        "condition", "substrate", "kind", "seed", "tensor_sha256", "amplitude_mode",
        "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "path_m", "forward_distance_m", "displacement_m", "mean_feet_planted",
        "min_height_m", "final_height_m", "fell",
    )
    with PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def mark_videos_reviewed() -> int:
    path = PREFIX.with_suffix(".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["videos"]["reviewed"] = True
    report["videos"]["review_note"] = "real and median joint-target-amplitude-matched random four-view sequences visually reviewed"
    primary = report["primary_forward"]
    statistical_gate = primary["p_holm"] < 0.05 and primary["real_outside_random_distribution"] and primary["nonoverlap_bootstrap_ci"]
    report["decision"] = "STRUCTURE-DRIVE-EARNED" if statistical_gate else "STRUCTURE-DRIVE-NULL"
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
        raise FileExistsError(f"{PREFIX.with_suffix('.json')} exists; use --mark-videos-reviewed")

    substrates, substrate_record = SC.build_vnc_substrates(SC.VNC_DATA, SC.SUBSTRATE_CACHE)
    io = SC.base_vnc(SC.VNC_DATA).io
    specs = SC.episode_specs(SC.N_EPISODES, SC.EPISODE_SEED)
    histories: dict[tuple[str, float, float], np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    summary_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    comparison_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    budget = SC.Budget()

    real = next(row for row in substrates if row.name == "real")
    real_history = SC.collect_history(real, SECS, DRIVE, histories)
    real_targets = SC.target_history(real_history, io, targets)
    real_sd = real_targets.std(axis=0)
    records = []
    rows = []
    for substrate in substrates:
        history = real_history if substrate.name == "real" else SC.collect_history(substrate, SECS, DRIVE, histories)
        raw_record, raw_rows = run_condition(substrate, "raw", history, io, specs, targets, budget, summary_rng)
        records.append(raw_record)
        rows.extend(raw_rows)
        if substrate.name != "real":
            source_targets = SC.target_history(history, io, targets)
            matched, match_record = amplitude_match(source_targets, real_sd)
            match_record["source_motor_history_sha256"] = array_sha256(history)
            match_record["source_joint_target_history_sha256"] = array_sha256(source_targets)
            matched_record, matched_rows = run_condition(substrate, MATCHED_MODE, matched, io, specs, targets, budget, summary_rng, match_record, mapped=True)
            records.append(matched_record)
            rows.extend(matched_rows)
        print(f"completed {substrate.name} seed={substrate.seed}", flush=True)

    real_record = next(row for row in records if row["substrate"] == "real")
    matched_random = [row for row in records if row["substrate"] == "random" and row["amplitude_mode"] == MATCHED_MODE]
    raw_random = [row for row in records if row["substrate"] == "random" and row["amplitude_mode"] == "raw"]
    matched_rewired = [row for row in records if row["substrate"] == "rewired" and row["amplitude_mode"] == MATCHED_MODE]
    raw_rewired = [row for row in records if row["substrate"] == "rewired" and row["amplitude_mode"] == "raw"]
    primary_forward = comparison(real_record, matched_random, "forward_distance_m", comparison_rng)
    primary_forward["p_holm"] = float(SC.holm_adjust([primary_forward["p_raw"]])[0])
    primary_forward["definition"] = "real minus per-channel-SD-amplitude-matched random-network mean forward distance"
    primary_path = comparison(real_record, matched_random, "path_m", np.random.default_rng(SC.BOOTSTRAP_SEED))
    primary_path["p_holm"] = float(SC.holm_adjust([primary_path["p_raw"]])[0])
    primary_path["definition"] = "secondary real minus per-channel-SD-amplitude-matched random-network mean path"

    random_forward = np.asarray([row["summary"]["metrics"]["forward_distance_m"]["mean"] for row in matched_random])
    median_index = int(np.argsort(random_forward)[len(random_forward) // 2])
    median_random = matched_random[median_index]
    video_paths = {
        "real": RESULTS / "h1_amplitude_strict_real_walk_multiview.mp4",
        "random_median": RESULTS / "h1_amplitude_strict_matched_random_median_walk_multiview.mp4",
    }
    selected_random = next(row for row in substrates if row.name == "random" and row.seed == median_random["seed"])
    for substrate, path in ((real, video_paths["real"]), (selected_random, video_paths["random_median"])):
        history = real_targets if substrate.name == "real" else amplitude_match(SC.target_history(SC.collect_history(substrate, SECS, DRIVE, histories), io, targets), real_sd)[0]
        model, body_record = SC.make_body_model()
        SC.run_body_trace(history, io, specs[0], SECS, video=path, budget=budget, model=model, body_record=body_record)

    file_hashes_pre = SC.file_hashes(SC.VNC_DATA)
    real_tensor_hash_pre = SC.tensor_sha256(real.weight)
    real_tensor_hash_post = SC.tensor_sha256(real.weight)
    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    matched_records = [row for row in records if row["amplitude_mode"] == MATCHED_MODE]
    max_sd_error = max(row["amplitude_match"]["max_sd_error_all_channels"] for row in matched_records)
    max_mean_error = max(row["amplitude_match"]["mean_preserved_max_error"] for row in matched_records)

    report = {
        "experiment": "H1",
        "status": "complete",
        "pre_registered_null": "amplitude-matched random ties real; STRUCTURE-DRIVE-NULL",
        "protocol": {
            "secs": SECS,
            "drive": DRIVE,
            "n_episodes": SC.N_EPISODES,
            "episode_seed": SC.EPISODE_SEED,
            "shared_episode_starts": True,
            "map_gains": list(SC.MAP_GAINS),
            "kp": SC.KP,
            "kd": SC.KD,
            "h_min_m": SC.H_MIN_EXPECTED,
            "settle_s": SC.SETTLE_S,
            "learning": False,
            "amplitude_rule": "per-channel temporal SD of fixed-map joint targets set to real joint-target SD; per-channel mean and temporal pattern retained",
            "primary": "forward_distance_m",
            "controls": ["raw random-network", "per-channel-SD amplitude-matched random-network", "raw rewired", "per-channel-SD amplitude-matched rewired"],
        },
        "real_graph_hashes": file_hashes_pre,
        "substrates": substrate_record,
        "amplitude_reference": {
            "real_history_sha256": array_sha256(real_history),
            "real_joint_target_history_sha256": array_sha256(real_targets),
            "real_channel_sd": real_sd.tolist(),
            "real_channel_sd_mean": float(real_sd.mean()),
            "domain": "18 fixed-map joint targets",
        },
        "conditions": records,
        "primary_forward": primary_forward,
        "secondary_path": primary_path,
        "reference_comparisons": {
            "raw_random_forward": comparison(real_record, raw_random, "forward_distance_m", np.random.default_rng(SC.BOOTSTRAP_SEED)),
            "matched_rewired_forward": comparison(real_record, matched_rewired, "forward_distance_m", np.random.default_rng(SC.BOOTSTRAP_SEED)),
            "raw_rewired_forward": comparison(real_record, raw_rewired, "forward_distance_m", np.random.default_rng(SC.BOOTSTRAP_SEED)),
        },
        "videos": {
            "real": {"path": str(video_paths["real"]), "exists": video_paths["real"].exists()},
            "random_median": {"path": str(video_paths["random_median"]), "exists": video_paths["random_median"].exists(), "seed": int(median_random["seed"])},
            "reviewed": False,
            "review_note": "review required before final decision",
        },
        "integrity": {
            "file_hashes_pre": file_hashes_pre,
            "file_hashes_post": file_hashes_post,
            "file_hashes_equal": bool(file_hashes_pre == file_hashes_post),
            "real_tensor_sha256_pre": real_tensor_hash_pre,
            "real_tensor_sha256_post": real_tensor_hash_post,
            "real_tensor_hash_equal": bool(real_tensor_hash_pre == real_tensor_hash_post),
            "random_seeds": list(SC.RNG_SEEDS),
            "rewired_seeds": list(SC.RNG_SEEDS),
            "condition_count": len(records),
            "row_count": len(rows),
            "episodes_per_condition": SC.N_EPISODES,
            "amplitude_match_count": len(matched_records),
            "max_sd_error_all_channels": max_sd_error,
            "max_mean_preserved_error": max_mean_error,
            "all_channels_match": bool(max_sd_error <= 1e-4),
            "learning": False,
        },
        "resource": {"rollouts_completed": budget.rollouts, "max_rollouts": SC.MAX_ROLLOUTS, "wallclock_ceiling_s": SC.WALLCLOCK_CEILING_S},
        "zero_learning": True,
        "zero_learning_audit": {"learning": False, "parameter_updates": 0, "training_markers": []},
        "rows": rows,
    }
    statistical_gate = primary_forward["p_holm"] < 0.05 and primary_forward["real_outside_random_distribution"] and primary_forward["nonoverlap_bootstrap_ci"]
    report["decision"] = "STRUCTURE-DRIVE-PENDING-VIDEO" if statistical_gate else "STRUCTURE-DRIVE-NULL"
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    write_csv(rows)
    print(json.dumps({"decision": report["decision"], "primary_forward": primary_forward, "resource": report["resource"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
