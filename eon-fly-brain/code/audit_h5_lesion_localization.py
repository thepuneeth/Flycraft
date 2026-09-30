"""H5: localize forward-drive contribution by source-edge lesions."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
DATA = HERE.parent.parent / "data"
sys.path.insert(0, str(HERE))
import specificity_campaign as SC  # noqa: E402
from audit_amplitude_control import amplitude_match, array_sha256  # noqa: E402


# `SC.ARM_SUFFIX` is empty for the rate core and "_lif" for the published one.
# Without it a run under SC_CORE=lif silently overwrites the recorded
# rate-core artifact of the same name, which is exactly what happened on the
# pod before this line existed -- only audit_amplitude_control had it.
PREFIX = RESULTS / f"h5_lesion_localization{SC.ARM_SUFFIX}"
SECS = 8.0
METRICS = ("path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m")
MATCHED_MODE = "per_channel_joint_target_sd_matched_to_real"
LESION_CANDIDATES = (
    ("segment", "somaNeuromere", "T1"),
    ("segment", "somaNeuromere", "T2"),
    ("segment", "somaNeuromere", "T3"),
    ("pathway", "superclass", "vnc_intrinsic"),
    ("pathway", "superclass", "ascending_neuron"),
    ("pathway", "superclass", "vnc_sensory"),
    ("pathway", "superclass", "descending_neuron"),
)


def sparse_source_lesion(weight: torch.Tensor, source_mask: np.ndarray) -> tuple[torch.Tensor, dict]:
    """Remove all stored edges whose presynaptic node is in source_mask."""
    base = weight.to_sparse_coo().coalesce().cpu()
    edge = base.indices().numpy()
    values = base.values().numpy()
    remove = source_mask[edge[1]]
    keep = ~remove
    lesion = torch.sparse_coo_tensor(edge[:, keep], torch.from_numpy(values[keep]), weight.shape).coalesce().to_sparse_csr()
    return lesion, {
        "base_edge_count": int(len(values)),
        "removed_edge_count": int(remove.sum()),
        "remaining_edge_count": int(keep.sum()),
        "removed_absolute_weight_mass": float(np.abs(values[remove]).sum()),
        "remaining_absolute_weight_mass": float(np.abs(values[keep]).sum()),
    }


def map_history(history: np.ndarray, io) -> np.ndarray:
    return np.stack([SC.translate(row, io, gains=SC.MAP_GAINS) for row in history]).astype(np.float32)


def paired_sign_flip(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    n = len(differences)
    if n == 0 or not np.isfinite(differences).all():
        raise AssertionError("invalid paired differences")
    masks = np.arange(1 << n, dtype=np.uint32)[:, None]
    signs = ((masks >> np.arange(n, dtype=np.uint32)) & 1).astype(np.int8) * 2 - 1
    null_means = (signs @ differences) / n
    return float(np.mean(null_means >= differences.mean() - 1e-12))


def paired_loss(baseline: list[dict], lesion: list[dict], rng: np.random.Generator) -> dict:
    base = np.asarray([row["forward_distance_m"] for row in baseline], dtype=np.float64)
    damaged = np.asarray([row["forward_distance_m"] for row in lesion], dtype=np.float64)
    if len(base) != len(damaged) or not np.array_equal([row["episode"] for row in baseline], [row["episode"] for row in lesion]):
        raise AssertionError("lesion and baseline episode pairing changed")
    loss = base - damaged
    indices = rng.integers(0, len(loss), size=(SC.BOOTSTRAP_REPS, len(loss)))
    boot = loss[indices].mean(axis=1)
    base_indices = rng.integers(0, len(base), size=(SC.BOOTSTRAP_REPS, len(base)))
    lesion_indices = rng.integers(0, len(damaged), size=(SC.BOOTSTRAP_REPS, len(damaged)))
    base_ci = np.percentile(base[base_indices].mean(axis=1), (2.5, 97.5))
    lesion_ci = np.percentile(damaged[lesion_indices].mean(axis=1), (2.5, 97.5))
    return {
        "metric": "forward_drive_loss_m",
        "baseline_mean_m": float(base.mean()),
        "baseline_ci_m": [float(base_ci[0]), float(base_ci[1])],
        "lesion_mean_m": float(damaged.mean()),
        "lesion_ci_m": [float(lesion_ci[0]), float(lesion_ci[1])],
        "loss_per_episode_m": loss.tolist(),
        "loss_mean_m": float(loss.mean()),
        "loss_ci_m": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_raw": paired_sign_flip(loss),
        "n": int(len(loss)),
        "paired_test": "exact one-sided sign-flip over shared episode starts",
        "nonoverlap_bootstrap_ci": bool(lesion_ci[1] < base_ci[0]),
        "all_episode_losses_positive": bool(np.all(loss > 0.0)),
    }


def run_condition(substrate: SC.Substrate, signal: np.ndarray, io, specs: list[dict], budget: SC.Budget, rng: np.random.Generator, mode: str) -> tuple[dict, list[dict]]:
    model, body_record = SC.make_body_model()
    rows = []
    condition = substrate.name
    for spec in specs:
        trace = SC.run_body_trace(signal, io, spec, SECS, budget=budget, model=model, body_record=body_record)
        rows.append({
            **spec,
            "condition": condition,
            "substrate": substrate.name,
            "kind": "real" if substrate.name == "real" else "lesion",
            "seed": substrate.seed,
            "tensor_sha256": substrate.tensor_hash,
            "amplitude_mode": mode,
            **trace["horizons"]["8s"],
        })
    return {
        "condition": condition,
        "substrate": substrate.name,
        "kind": "real" if substrate.name == "real" else "lesion",
        "seed": substrate.seed,
        "tensor_sha256": substrate.tensor_hash,
        "amplitude_mode": mode,
        "signal_sha256": array_sha256(signal),
        "summary": SC.summarize_rows(rows, METRICS, rng),
    }, rows


def annotations_for_nodes(io) -> pd.DataFrame:
    table = feather.read_table(DATA / "manc" / "body-annotations-male-cns-v1.0-minconf-0.5.feather", columns=["bodyId", "somaNeuromere", "superclass"])
    annotations = table.to_pandas().drop_duplicates("bodyId").set_index("bodyId")
    nodes = io["node_body_ids"].astype(np.int64)
    aligned = annotations.reindex(nodes)
    if len(aligned) != len(nodes) or aligned.index.isna().any():
        raise AssertionError("node annotation alignment failed")
    return aligned


def write_csv(rows: list[dict]) -> None:
    fields = (
        "condition", "substrate", "kind", "seed", "tensor_sha256", "amplitude_mode",
        "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "path_m", "forward_distance_m", "displacement_m", "mean_feet_planted",
        "min_height_m", "final_height_m", "fell",
    )
    with PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def mark_videos_reviewed() -> int:
    path = PREFIX.with_suffix(".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["videos"]["reviewed"] = True
    report["videos"]["review_note"] = "real and largest-loss lesion four-view sequences visually reviewed"
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
    io = SC.base_vnc(SC.VNC_DATA).io
    base_vnc = SC.base_vnc(SC.VNC_DATA)
    real_weight = base_vnc.W.cpu()
    real_tensor_hash = SC.tensor_sha256(real_weight)
    annotations = annotations_for_nodes(io)
    specs = SC.episode_specs(SC.N_EPISODES, SC.EPISODE_SEED)
    histories = {}
    real_history = SC.collect_history(SC.Substrate("real", None, real_weight, real_tensor_hash), SECS, SC.BASE_DRIVE, histories)
    real_targets = map_history(real_history, io)
    target_sd = real_targets.std(axis=0)
    summary_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    comparison_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    budget = SC.Budget()
    records = []
    rows = []
    lesion_records = []
    signals = {"real": real_targets}
    real_substrate = SC.Substrate("real", None, real_weight, real_tensor_hash)
    real_record, real_rows = run_condition(real_substrate, real_targets, io, specs, budget, summary_rng, "raw")
    records.append(real_record)
    rows.extend(real_rows)
    with (SC.VNC_DATA / "all_legs_weight_csr.pkl").open("rb") as stream:
        frozen_weight = pickle.load(stream).to_sparse_csr().cpu()
    if SC.tensor_sha256(frozen_weight) != real_tensor_hash:
        raise AssertionError("frozen weight hash changed before lesion construction")

    for kind, field, label in LESION_CANDIDATES:
        source_values = annotations[field].fillna("NA").to_numpy(dtype=str)
        source_mask = source_values == label
        lesion_weight, graph_record = sparse_source_lesion(frozen_weight, source_mask)
        name = f"h5_{kind}_{label}"
        substrate = SC.Substrate(name, None, lesion_weight, SC.tensor_sha256(lesion_weight))
        motor_history = SC.collect_history(substrate, SECS, SC.BASE_DRIVE, histories)
        source_targets = map_history(motor_history, io)
        source_sd = source_targets.std(axis=0)
        unmatchable = (source_sd <= 0.0) & (target_sd > 0.0)
        candidate = {
            "name": name,
            "kind": kind,
            "annotation_field": field,
            "label": label,
            "source_node_count": int(source_mask.sum()),
            "graph": graph_record,
            "tensor_sha256": substrate.tensor_hash,
            "source_motor_history_sha256": array_sha256(motor_history),
            "source_joint_target_history_sha256": array_sha256(source_targets),
            "source_zero_sd_channels": int(np.count_nonzero(source_sd <= 0.0)),
            "target_zero_sd_channels": int(np.count_nonzero(target_sd <= 0.0)),
            "unmatchable_channels": int(unmatchable.sum()),
        }
        if unmatchable.any():
            candidate["status"] = "excluded_unmatchable"
            candidate["exclusion"] = "strict all-channel amplitude matching impossible because lesion silenced nonzero real target channels"
            lesion_records.append(candidate)
            continue
        matched_targets, match_record = amplitude_match(source_targets, target_sd)
        match_record["source_motor_history_sha256"] = array_sha256(motor_history)
        match_record["source_joint_target_history_sha256"] = array_sha256(source_targets)
        match_record["reference"] = "real alpha=0 target temporal SD"
        lesion_record, lesion_rows = run_condition(substrate, matched_targets, io, specs, budget, summary_rng, MATCHED_MODE)
        lesion_record["lesion"] = candidate | {"status": "included", "amplitude_match": match_record}
        records.append(lesion_record)
        rows.extend(lesion_rows)
        signals[name] = matched_targets
        lesion_records.append(lesion_record["lesion"])
        print(f"completed {name}", flush=True)

    baseline_rows = real_rows
    contrasts = []
    for record in records[1:]:
        lesion_rows = [row for row in rows if row["condition"] == record["condition"]]
        loss = paired_loss(baseline_rows, lesion_rows, comparison_rng)
        contrasts.append({"lesion": record["condition"], **loss})
    for contrast, p_holm in zip(contrasts, SC.holm_adjust([item["p_raw"] for item in contrasts])):
        contrast["p_holm"] = float(p_holm)
    contrasts.sort(key=lambda item: item["loss_mean_m"], reverse=True)
    top = contrasts[0] if contrasts else None
    included_names = {record["condition"] for record in records[1:]}
    if top is None:
        numeric_decision = "LESION-LOCALIZATION-NULL"
    else:
        significant = [item for item in contrasts if item["p_holm"] < 0.05 and item["loss_ci_m"][0] > 0.0]
        numeric_decision = "LESION-LOCALIZATION-EARNED" if significant else "LESION-LOCALIZATION-NULL"
    top_name = top["lesion"] if top else None
    video_paths = {
        "real": RESULTS / "h5_lesion_real_multiview.mp4",
        "largest_loss": RESULTS / "h5_lesion_largest_loss_multiview.mp4",
    }
    if top_name not in signals:
        raise AssertionError("largest-loss signal missing")
    for name, path in (("real", video_paths["real"]), (top_name, video_paths["largest_loss"])):
        model, body_record = SC.make_body_model()
        SC.run_body_trace(signals[name], io, specs[0], SECS, video=path, budget=budget, model=model, body_record=body_record)
    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    report = {
        "experiment": "H5",
        "status": "complete",
        "pre_registered_null": "forward-drive loss is diffuse; no source segment/pathway class is localizable",
        "protocol": {
            "secs": SECS,
            "drive": SC.BASE_DRIVE,
            "lesion_semantics": "remove all stored edges whose presynaptic node annotation matches candidate",
            "candidate_groups": [list(item) for item in LESION_CANDIDATES],
            "n_episodes": SC.N_EPISODES,
            "shared_episode_starts": True,
            "amplitude_rule": "per-channel temporal SD of fixed-map joint targets set to real baseline SD; strict all-channel matching",
            "primary": "paired forward-drive loss, exact sign-flip with Holm across included lesions",
            "learning": False,
        },
        "annotation_source": str(DATA / "manc" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"),
        "real_graph_hashes": file_hashes_pre,
        "real_tensor_sha256": real_tensor_hash,
        "records": records,
        "lesion_candidates": lesion_records,
        "contrasts_ranked_by_loss": contrasts,
        "videos": {
            "real": {"path": str(video_paths["real"]), "exists": video_paths["real"].exists()},
            "largest_loss": {"path": str(video_paths["largest_loss"]), "exists": video_paths["largest_loss"].exists(), "lesion": top_name},
            "reviewed": False,
            "review_note": "review required before final decision",
        },
        "integrity": {
            "file_hashes_pre": file_hashes_pre,
            "file_hashes_post": file_hashes_post,
            "file_hashes_equal": bool(file_hashes_pre == file_hashes_post),
            "real_tensor_sha256_pre": real_tensor_hash,
            "real_tensor_sha256_post": SC.tensor_sha256(frozen_weight),
            "real_tensor_hash_equal": bool(real_tensor_hash == SC.tensor_sha256(frozen_weight)),
            "candidate_count": len(LESION_CANDIDATES),
            "included_count": len(included_names),
            "excluded_count": len(LESION_CANDIDATES) - len(included_names),
            "row_count": len(rows),
            "rollout_count": budget.rollouts,
            "all_included_channels_match": bool(all(item.get("unmatchable_channels", 1) == 0 for item in lesion_records if item.get("status") == "included")),
            "learning": False,
        },
        "resource": {"rollouts_completed": budget.rollouts, "max_rollouts": SC.MAX_ROLLOUTS},
        "zero_learning": True,
        "zero_learning_audit": {"learning": False, "parameter_updates": 0, "training_markers": []},
        "numeric_decision": numeric_decision,
        "decision": "LESION-LOCALIZATION-PENDING-VIDEO" if numeric_decision.endswith("EARNED") else numeric_decision,
        "rows": rows,
    }
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    write_csv(rows)
    print(json.dumps({"experiment": "H5", "numeric_decision": numeric_decision, "contrasts": contrasts, "resource": report["resource"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
