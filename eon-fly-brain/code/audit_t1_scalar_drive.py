"""T1: one-scalar drive controller calibrated from the fixed E2 speed curves."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
import specificity_campaign as SC  # noqa: E402


INPUT = RESULTS / "e2_speed.json"
PREFIX = RESULTS / "t1_scalar_drive"
TARGET_SPEEDS = np.asarray((0.005, 0.010, 0.020, 0.025), dtype=np.float64)
DRIVE_LIMITS = (0.0, 200.0)
THETA_GRID = np.linspace(0.0, DRIVE_LIMITS[1] / TARGET_SPEEDS.min(), 20_001)
CALIBRATION_SAMPLES = 3


def fit_controller(curve: np.ndarray, held_out: int, drives: np.ndarray) -> dict:
    train = np.asarray([index for index in range(len(TARGET_SPEEDS)) if index != held_out], dtype=np.int64)
    targets = TARGET_SPEEDS[train]
    predicted_drive = np.clip(THETA_GRID[:, None] * targets[None, :], *DRIVE_LIMITS)
    predicted_speed = np.stack([np.interp(predicted_drive[:, index], drives, curve) for index in range(len(train))], axis=1)
    objective = np.mean((predicted_speed - targets[None, :]) ** 2, axis=1)
    best = int(np.argmin(objective))
    theta = float(THETA_GRID[best])
    drive = float(np.clip(theta * TARGET_SPEEDS[held_out], *DRIVE_LIMITS))
    achieved = float(np.interp(drive, drives, curve))
    return {
        "held_out_target_speed_m_per_s": float(TARGET_SPEEDS[held_out]),
        "theta_drive_per_m_per_s": theta,
        "selected_drive": drive,
        "achieved_speed_m_per_s": achieved,
        "absolute_speed_error_m_per_s": abs(achieved - TARGET_SPEEDS[held_out]),
        "calibration_target_indices": train.tolist(),
        "calibration_samples": CALIBRATION_SAMPLES,
    }


def controller_summary(curve: np.ndarray, drives: np.ndarray) -> dict:
    folds = [fit_controller(curve, held_out, drives) for held_out in range(len(TARGET_SPEEDS))]
    errors = np.asarray([row["absolute_speed_error_m_per_s"] for row in folds])
    selected_drives = np.asarray([row["selected_drive"] for row in folds])
    return {
        "folds": folds,
        "mean_absolute_speed_error_m_per_s": float(errors.mean()),
        "speed_error_ci_m_per_s": [float(np.percentile(errors, 2.5)), float(np.percentile(errors, 97.5))],
        "mean_selected_drive": float(selected_drives.mean()),
        "scalar_parameter_count": 1,
        "calibration_samples_per_fold": CALIBRATION_SAMPLES,
    }


def lower_is_better_comparison(real_value: float, control_values: list[float], metric: str, rng: np.random.Generator) -> dict:
    real = -float(real_value)
    controls = [-float(value) for value in control_values]
    result = SC.substrate_comparison(real, controls, metric, rng)
    result["real_error"] = float(real_value)
    result["control_errors"] = [float(value) for value in control_values]
    result["real_minus_control_mean_in_error_units"] = float(np.mean(control_values) - real_value)
    result["higher_is_better"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    if PREFIX.with_suffix(".json").exists():
        raise FileExistsError(f"{PREFIX.with_suffix('.json')} exists")
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    if source["experiment"] != "E2" or source["status"] != "complete" or not source["integrity"]["file_hashes_equal"]:
        raise AssertionError("T1 requires complete, hash-clean E2 input")
    drives = np.asarray([0.0, *source["protocol"]["drives"]], dtype=np.float64)
    curves = {
        "real": np.asarray([0.0, *source["speed_curves_m_per_s"]["real"]], dtype=np.float64),
        "matched_random": {seed: np.asarray([0.0, *curve], dtype=np.float64) for seed, curve in source["speed_curves_m_per_s"]["matched_random"].items()},
        "matched_rewired": {seed: np.asarray([0.0, *curve], dtype=np.float64) for seed, curve in source["speed_curves_m_per_s"]["matched_rewired"].items()},
    }
    prereg = {
        "experiment": "T1",
        "input": str(INPUT),
        "controller": "drive = clip(theta * target_speed, 0, 200)",
        "scalar_parameter_count": 1,
        "target_speeds_m_per_s": TARGET_SPEEDS.tolist(),
        "calibration": "leave-one-target-speed-out; three fixed calibration targets per fold",
        "response_source": "completed E2 amplitude-matched 8 s forward-speed curves; drive=0 fixed stance response is 0 m/s",
        "primary": "leave-one-out mean absolute target-speed error",
        "learning_scope": "one scalar controller coefficient; VNC graph frozen",
        "body_outcome_used": "E2 speeds only; no new evaluation selection",
    }
    SC.atomic_json(PREFIX.with_name(PREFIX.name + "_preregistered.json"), prereg)
    summaries = {"real": controller_summary(curves["real"], drives)}
    summaries.update({"matched_random": {seed: controller_summary(curve, drives) for seed, curve in curves["matched_random"].items()}})
    summaries.update({"matched_rewired": {seed: controller_summary(curve, drives) for seed, curve in curves["matched_rewired"].items()}})
    rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    real_error = summaries["real"]["mean_absolute_speed_error_m_per_s"]
    random_errors = [value["mean_absolute_speed_error_m_per_s"] for value in summaries["matched_random"].values()]
    rewired_errors = [value["mean_absolute_speed_error_m_per_s"] for value in summaries["matched_rewired"].values()]
    comparisons = [
        lower_is_better_comparison(real_error, random_errors, "scalar_controller_error_vs_matched_random", rng),
        lower_is_better_comparison(real_error, rewired_errors, "scalar_controller_error_vs_rewired", rng),
    ]
    for item, p_holm in zip(comparisons, SC.holm_adjust([item["p_raw"] for item in comparisons])):
        item["p_holm"] = float(p_holm)
    numeric_decision = "T1-SCALAR-CONTROL-EARNED" if all(item["real_minus_control_mean_in_error_units"] > 0.0 and item["p_holm"] < 0.05 for item in comparisons) else "T1-SCALAR-CONTROL-NULL"
    report = {
        "experiment": "T1",
        "status": "complete",
        "pre_registered_null": "connectome requires equal-or-more scalar control error than matched random",
        "protocol": prereg,
        "source_integrity": {
            "input_status": source["status"],
            "input_file_hashes_equal": source["integrity"]["file_hashes_equal"],
            "input_zero_learning": source["zero_learning"],
            "input_video_reviewed": source["videos"]["reviewed"],
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "zero_learning": False,
        "zero_learning_audit": {"learning": True, "learned_object": "one scalar theta per substrate", "vnc_parameter_updates": 0, "training_markers": ["scalar calibration from E2 response curve"]},
        "numeric_decision": numeric_decision,
        "decision": numeric_decision,
    }
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    print(json.dumps({"experiment": "T1", "numeric_decision": numeric_decision, "real_error": real_error, "comparisons": comparisons}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
