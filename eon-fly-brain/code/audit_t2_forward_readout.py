"""T2: identical memoryless readouts trained for forward-progress targets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
import specificity_campaign as SC  # noqa: E402
from collect_reservoir_states import collect  # noqa: E402
from rung1_readout import (  # noqa: E402
    BATCH_SIZE,
    CONDITIONS,
    EPOCHS,
    HIDDEN_UNITS,
    LEARNING_RATE,
    N_EPISODES,
    TRAIN_FRACTION,
    VAL_FRACTION,
    condition_stepper,
    load_readout,
    predict_standardized,
    save_readout,
    split_bounds,
    standardize,
    train_readout,
)
from hexapod_gait import HexapodGait  # noqa: E402


PREFIX = RESULTS / "t2_forward_readout"
WEIGHTS_NAME = "all_legs_weight_csr.pkl"
INDEX_NAME = "all_legs_io_index.npz"
VNC_DRIVE = 100.0
BURN_STEPS = 3000
RECORD_STEPS = 3000
EVAL_STEPS = 400
FORWARD_TARGET = 0.8
TURN_TARGET = 0.0
TRAIN_SEED = 20260829
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260829
KP = 3.0
KD = 0.05
DT_S = 1.0 / 50.0


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def forward_target(steps: int) -> np.ndarray:
    gait = HexapodGait(stance=None)
    target = np.stack([gait.targets(FORWARD_TARGET, TURN_TARGET, DT_S) for _ in range(steps)]).astype(np.float32)
    if target.shape != (steps, 18) or not np.isfinite(target).all():
        raise AssertionError("invalid forward target")
    return target


def one_sided_sign_flip(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    if len(diff) > 22:
        raise ValueError("exact sign-flip capped at n=22")
    masks = np.arange(1 << len(diff), dtype=np.uint32)[:, None]
    signs = ((masks >> np.arange(len(diff), dtype=np.uint32)) & 1).astype(np.int8) * 2 - 1
    return float(np.mean((signs @ diff) / len(diff) >= diff.mean() - 1e-12))


def paired_forward(real_rows: list[dict], other_rows: list[dict], rng: np.random.Generator) -> dict:
    if [row["episode"] for row in real_rows] != [row["episode"] for row in other_rows]:
        raise AssertionError("body episode starts are not paired")
    real = np.asarray([row["forward_distance_m"] for row in real_rows], dtype=np.float64)
    other = np.asarray([row["forward_distance_m"] for row in other_rows], dtype=np.float64)
    diff = real - other
    indices = rng.integers(0, len(diff), size=(BOOTSTRAP_REPS, len(diff)))
    low, high = np.percentile(diff[indices].mean(axis=1), (2.5, 97.5))
    return {
        "comparison": "connectome_minus_" + other_rows[0]["condition"].split("_")[0],
        "metric": "forward_distance_m",
        "n_pairs": len(diff),
        "mean_difference_m": float(diff.mean()),
        "ci_low_m": float(low),
        "ci_high_m": float(high),
        "p_raw": one_sided_sign_flip(diff),
        "paired_test": "exact one-sided sign-flip over shared episode starts",
    }


def metric_summary(rows: list[dict], rng: np.random.Generator) -> dict:
    return SC.summarize_rows(rows, ("path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m"), rng)


def write_csv(rows: list[dict]) -> None:
    fields = (
        "condition", "evaluation_condition", "episode", "episode_seed", "initial_yaw_rad",
        "prey_x_m", "prey_y_m", "path_m", "forward_distance_m", "displacement_m",
        "mean_feet_planted", "min_height_m", "final_height_m", "fell",
    )
    with PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def mark_videos_reviewed() -> int:
    path = PREFIX.with_suffix(".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["videos"]["reviewed"] = True
    report["videos"]["review_note"] = "real, random, and rewired forward-readout four-view sequences visually reviewed"
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
    target = forward_target(RECORD_STEPS)
    bounds = split_bounds(RECORD_STEPS)
    train_end = bounds["train"][1]
    validation_end = bounds["validation"][1]
    prereg = {
        "experiment": "T2",
        "target": "fixed forward-progress joint target: HexapodGait forward=0.8, turn=0 at 50 Hz",
        "conditions": list(CONDITIONS),
        "architecture": {"class": "Linear-Tanh-Linear", "hidden_units": HIDDEN_UNITS, "memoryless": True, "input": "current standardized VNC population state only", "output_dim": 18},
        "loss": "mean squared error on held-out forward-progress joint-target signal only",
        "selection": "lowest validation target MSE only",
        "body_outcome_used_for_training_or_selection": False,
        "evaluation": "open-loop body forward distance on held-out VNC state segment",
        "static_input_leak_detector": True,
        "learning_scope": "readout only; VNC weights frozen",
    }
    SC.atomic_json(PREFIX.with_name(PREFIX.name + "_preregistered.json"), prereg)

    raw = {}
    graph_before = {}
    for condition in CONDITIONS[:3]:
        vnc = condition_stepper(SC.VNC_DATA, condition)
        graph_before[condition] = SC.tensor_sha256(vnc.W)
        raw[condition] = collect(vnc, VNC_DRIVE, BURN_STEPS, RECORD_STEPS)
    active = np.flatnonzero(raw["connectome"][:train_end].std(axis=0) > 1e-7).astype(np.int64)
    if len(active) == 0:
        raise AssertionError("no varying connectome state columns")
    raw["static-input"] = np.repeat(raw["connectome"][:train_end].mean(axis=0, keepdims=True), RECORD_STEPS, axis=0)
    normalized, means, scales = {}, {}, {}
    for condition in CONDITIONS:
        normalized[condition], means[condition], scales[condition] = standardize(raw[condition][:, active], train_end)
    dataset_path = PREFIX.with_name(PREFIX.name + "_dataset.npz")
    arrays = {
        "target": target,
        "active_indices": active,
        "train_end": np.asarray(train_end),
        "validation_end": np.asarray(validation_end),
    }
    for condition in CONDITIONS:
        arrays[f"x_{condition}"] = normalized[condition]
        arrays[f"train_mean_{condition}"] = means[condition]
        arrays[f"train_scale_{condition}"] = scales[condition]
    np.savez_compressed(dataset_path, **arrays)

    setup = {
        "architecture": {"class": "Linear-Tanh-Linear", "input_dim": len(active), "hidden_units": HIDDEN_UNITS, "output_dim": 18, "memoryless": True},
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epoch_cap": EPOCHS,
        "seed": TRAIN_SEED,
        "target_sha256": array_hash(target),
        "active_indices_sha256": array_hash(active),
        "split": {key: list(value) for key, value in bounds.items()},
        "body_outcome_used": False,
        "vnc_weights_trained": False,
    }
    setup_hash = json_hash(setup)
    training_rows = []
    training_reports = {}
    readout_paths = {}
    for condition in CONDITIONS:
        model, fit = train_readout(normalized[condition], target, bounds)
        path = PREFIX.with_name(PREFIX.name + f"_readout_{condition}.npz")
        save_readout(path, model, active, means[condition], scales[condition])
        readout = load_readout(path)
        if readout["active_indices"].shape != active.shape:
            raise AssertionError("readout input shape changed")
        row = {
            "condition": condition,
            "train_mse_rad2": fit["errors"]["train_mse_rad2"],
            "validation_mse_rad2": fit["errors"]["validation_mse_rad2"],
            "test_mse_rad2": fit["errors"]["test_mse_rad2"],
            "best_epoch": fit["best_epoch"],
            "setup_hash": setup_hash,
            "artifact": str(path),
        }
        training_rows.append(row)
        training_reports[condition] = {**row, "history": fit["history"]}
        readout_paths[condition] = path
        print(json.dumps(row, sort_keys=True), flush=True)
    if len({row["setup_hash"] for row in training_rows}) != 1:
        raise AssertionError("training setup differed across conditions")

    eval_start = bounds["test"][0]
    eval_slice = slice(eval_start, eval_start + EVAL_STEPS)
    predicted = {}
    prediction_records = {}
    for condition in CONDITIONS:
        readout = load_readout(readout_paths[condition])
        signal = np.stack([predict_standardized(row, readout) for row in normalized[condition][eval_slice]]).astype(np.float32)
        if signal.shape != (EVAL_STEPS, 18) or not np.isfinite(signal).all():
            raise AssertionError(f"invalid predicted forward signal for {condition}")
        predicted[condition] = signal
        prediction_records[condition] = {
            "shape": list(signal.shape),
            "sha256": array_hash(signal),
            "temporal_sd_mean_rad": float(signal.std(axis=0).mean()),
            "temporal_sd_max_rad": float(signal.std(axis=0).max()),
            "constant_signal_max_abs_jitter_rad": float(np.max(np.abs(signal - signal[0]))),
            "mean_abs_rad": float(np.abs(signal).mean()),
        }

    specs = SC.episode_specs(N_EPISODES, SC.EPISODE_SEED)
    io = SC.base_vnc(SC.VNC_DATA).io
    budget = SC.Budget()
    summary_rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    evaluations = {}
    for condition in CONDITIONS:
        condition_rows = []
        for spec in specs:
            model, body_record = SC.make_body_model()
            trace = SC.run_body_trace(predicted[condition], io, spec, 8.0, budget=budget, model=model, body_record=body_record)
            condition_rows.append({**spec, "condition": f"{condition}_episode_{spec['episode']}", "evaluation_condition": condition, **trace["horizons"]["8s"]})
        rows.extend(condition_rows)
        evaluations[condition] = {"summary": metric_summary(condition_rows, summary_rng), "rows": condition_rows, "prediction": prediction_records[condition]}
        print(f"evaluated {condition}", flush=True)

    real_rows = evaluations["connectome"]["rows"]
    comparisons = []
    for condition in ("random", "rewired"):
        comparisons.append(paired_forward(real_rows, evaluations[condition]["rows"], np.random.default_rng(BOOTSTRAP_SEED)))
    for comparison, p_holm in zip(comparisons, [value for value in SC.holm_adjust([row["p_raw"] for row in comparisons])]):
        comparison["p_holm"] = float(p_holm)
    static = evaluations["static-input"]
    static_leak = {
        "pass": bool(prediction_records["static-input"]["constant_signal_max_abs_jitter_rad"] <= 1e-8 and static["summary"]["metrics"]["path_m"]["mean"] < 1.0 and static["summary"]["metrics"]["forward_distance_m"]["mean"] < 0.5),
        "constant_target_max_abs_jitter_rad": prediction_records["static-input"]["constant_signal_max_abs_jitter_rad"],
        "mean_path_m": static["summary"]["metrics"]["path_m"]["mean"],
        "mean_forward_m": static["summary"]["metrics"]["forward_distance_m"]["mean"],
        "threshold": {"target_temporal_sd_max_rad_le": 1e-8, "path_m_lt": 1.0, "forward_m_lt": 0.5},
    }
    real_forward = evaluations["connectome"]["summary"]["metrics"]["forward_distance_m"]
    numeric_decision = "T2-FORWARD-READOUT-EARNED" if all(row["mean_difference_m"] > 0.0 and row["p_holm"] < 0.05 for row in comparisons) else "T2-FORWARD-READOUT-NULL"
    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    graph_after = {condition: SC.tensor_sha256(condition_stepper(SC.VNC_DATA, condition).W) for condition in CONDITIONS[:3]}
    video_paths = {
        "connectome": RESULTS / "t2_forward_readout_connectome_multiview.mp4",
        "random": RESULTS / "t2_forward_readout_random_multiview.mp4",
        "rewired": RESULTS / "t2_forward_readout_rewired_multiview.mp4",
    }
    for condition, path in video_paths.items():
        model, body_record = SC.make_body_model()
        SC.run_body_trace(predicted[condition], io, specs[0], 8.0, video=path, budget=budget, model=model, body_record=body_record)
    report = {
        "experiment": "T2",
        "status": "complete",
        "pre_registered_null": "connectome ties amplitude-matched random under identical forward-target training",
        "protocol": prereg | {"bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": BOOTSTRAP_SEED, "body_kp": KP, "body_kd": KD},
        "setup": setup,
        "setup_hash": setup_hash,
        "dataset": str(dataset_path),
        "training": training_reports,
        "predictions": prediction_records,
        "evaluations": {condition: {"summary": value["summary"], "prediction": value["prediction"]} for condition, value in evaluations.items()},
        "comparisons": comparisons,
        "static_input_leak_check": static_leak,
        "readout_artifacts": {condition: str(path) for condition, path in readout_paths.items()},
        "videos": {
            condition: {"path": str(path), "exists": path.exists()} for condition, path in video_paths.items()
        } | {"reviewed": False, "review_note": "review required before final decision"},
        "integrity": {
            "file_hashes_pre": file_hashes_pre,
            "file_hashes_post": file_hashes_post,
            "file_hashes_equal": bool(file_hashes_pre == file_hashes_post),
            "graph_tensor_sha256_before": graph_before,
            "graph_tensor_sha256_after": graph_after,
            "graph_tensor_hash_equal": bool(graph_before == graph_after),
            "identical_setup_hash": len({row["setup_hash"] for row in training_rows}) == 1,
            "body_outcome_used_for_training_or_selection": False,
            "vnc_weights_trained": False,
        },
        "resource": {"rollouts_completed": budget.rollouts, "max_rollouts": SC.MAX_ROLLOUTS},
        "zero_learning": False,
        "zero_learning_audit": {"learning": True, "learned_objects": "four readout MLPs only", "vnc_parameter_updates": 0, "training_markers": ["readout optimizer steps"]},
        "numeric_decision": numeric_decision,
        "decision": "T2-FORWARD-READOUT-PENDING-VIDEO",
        "rows": rows,
    }
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    write_csv(rows)
    print(json.dumps({"experiment": "T2", "numeric_decision": numeric_decision, "comparisons": comparisons, "static_input_leak_check": static_leak, "resource": report["resource"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
