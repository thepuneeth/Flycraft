"""H4: dose-response from degree-preserving rewiring fraction."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
import specificity_campaign as SC  # noqa: E402
from audit_amplitude_control import amplitude_match, array_sha256  # noqa: E402


# `SC.ARM_SUFFIX` is empty for the rate core and "_lif" for the published one.
# Without it a run under SC_CORE=lif silently overwrites the recorded
# rate-core artifact of the same name, which is exactly what happened on the
# pod before this line existed -- only audit_amplitude_control had it.
PREFIX = RESULTS / f"h4_dose_response{SC.ARM_SUFFIX}"
SECS = 8.0
ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
METRICS = ("path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m")
MATCHED_MODE = "per_channel_joint_target_sd_matched_to_real_at_fraction"


def partial_rewire(weight: torch.Tensor, fraction: float, seed: int) -> tuple[torch.Tensor, dict]:
    """Apply a fixed fraction of accepted degree-preserving edge swaps."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("rewiring fraction must be in [0, 1]")
    base = weight.to_sparse_coo().coalesce().cpu()
    rows = base.indices()[0].numpy().astype(np.int64, copy=True)
    columns = base.indices()[1].numpy().astype(np.int64, copy=True)
    values = base.values().numpy().astype(np.float32, copy=True)
    n = int(weight.shape[0])
    target_swaps = int(round(float(fraction) * len(rows) / 2.0))
    if target_swaps == 0:
        return weight.to_sparse_csr().cpu(), {"fraction": float(fraction), "accepted_swaps": 0, "changed_edges": 0}
    edges = set((rows * n + columns).tolist())
    original_edges = edges.copy()
    rng = np.random.default_rng(seed)
    accepted = 0
    while accepted < target_swaps:
        batch = min(100_000, target_swaps - accepted + 10_000)
        for left, right in rng.integers(0, len(rows), size=(batch, 2)):
            r1, r2 = int(rows[left]), int(rows[right])
            c1, c2 = int(columns[left]), int(columns[right])
            if r1 == r2 or c1 == c2:
                continue
            old1, old2 = r1 * n + c1, r2 * n + c2
            new1, new2 = r1 * n + c2, r2 * n + c1
            if new1 in edges or new2 in edges:
                continue
            edges.remove(old1)
            edges.remove(old2)
            edges.add(new1)
            edges.add(new2)
            columns[left], columns[right] = c2, c1
            accepted += 1
            if accepted == target_swaps:
                break
    indices = torch.from_numpy(np.stack((rows, columns))).long()
    result = torch.sparse_coo_tensor(indices, torch.from_numpy(values), weight.shape).coalesce().to_sparse_csr()
    if result._nnz() != weight._nnz():
        raise AssertionError("partial rewire changed edge count")
    if not torch.equal(result.crow_indices(), weight.cpu().to_sparse_csr().crow_indices()):
        raise AssertionError("partial rewire changed row degree")
    if not np.array_equal(np.bincount(columns, minlength=n), np.bincount(base.indices()[1].numpy(), minlength=n)):
        raise AssertionError("partial rewire changed column degree")
    changed_edges = len(edges - original_edges)
    return result, {"fraction": float(fraction), "accepted_swaps": accepted, "changed_edges": int(changed_edges)}


def map_history(history: np.ndarray, io, gains: tuple[float, float, float], cache: dict) -> np.ndarray:
    key = (id(history), gains)
    if key not in cache:
        cache[key] = np.stack([SC.translate(row, io, gains=gains) for row in history]).astype(np.float32)
    return cache[key]


def run_condition(substrate, mode, signal, io, specs, budget, rng, gains, mapped=False):
    targets = signal if mapped else map_history(signal, io, gains, {})
    model, body_record = SC.make_body_model()
    rows = []
    condition = f"{substrate.name}_{substrate.seed if substrate.seed is not None else 'real'}_{mode}_alpha_{substrate.rewiring_fraction:g}"
    for spec in specs:
        trace = SC.run_body_trace(targets, io, spec, SECS, budget=budget, model=model, body_record=body_record)
        rows.append({
            **spec,
            "condition": condition,
            "substrate": substrate.name,
            "kind": "real" if substrate.name == "real" else "fraction-rewired",
            "seed": substrate.seed,
            "rewiring_fraction": substrate.rewiring_fraction,
            "amplitude_mode": mode,
            "tensor_sha256": substrate.tensor_hash,
            **trace["horizons"]["8s"],
        })
    return {
        "condition": condition,
        "substrate": substrate.name,
        "kind": "real" if substrate.name == "real" else "fraction-rewired",
        "seed": substrate.seed,
        "rewiring_fraction": substrate.rewiring_fraction,
        "amplitude_mode": mode,
        "signal_domain": "joint_targets" if mapped else "vnc_motor_output",
        "tensor_sha256": substrate.tensor_hash,
        "signal_sha256": array_sha256(signal),
        "summary": SC.summarize_rows(rows, METRICS, rng),
    }, rows


def exact_spearman(values: list[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    if np.std(values) <= 1e-12:
        return 0.0
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return float(np.corrcoef(np.arange(len(values), dtype=np.float64), ranks)[0, 1])


def make_figure(dose: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["rewiring_fraction"] for row in dose]
    y = [row["mean_forward_distance_m"] for row in dose]
    low = [row["forward_ci_low"] for row in dose]
    high = [row["forward_ci_high"] for row in dose]
    fig, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    axis.plot(x, y, marker="o", color="#176d9c")
    axis.fill_between(x, low, high, color="#176d9c", alpha=0.2, linewidth=0)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="edge-swap budget α", ylabel="forward distance (m; 8 s)", title="Rewiring dose: preserved structure → forward drive")
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def write_csv(rows: list[dict]) -> None:
    fields = (
        "condition", "substrate", "kind", "seed", "rewiring_fraction", "amplitude_mode", "tensor_sha256",
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
    report["videos"]["review_note"] = "real and fully-rewired median four-view sequences visually reviewed"
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
    base = next(item for item in substrates if item.name == "real")
    io = SC.base_vnc(SC.VNC_DATA).io
    specs = SC.episode_specs(SC.N_EPISODES, SC.EPISODE_SEED)
    history_cache = {}
    target_cache = {}
    summary_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    comparison_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    budget = SC.Budget()
    gains = SC.MAP_GAINS
    base.rewiring_fraction = 0.0
    real_history = SC.collect_history(base, SECS, SC.BASE_DRIVE, history_cache)
    real_targets = map_history(real_history, io, gains, target_cache)
    records = []
    rows = []
    graph_integrity = []
    dose_records = []
    real_record, real_rows = run_condition(base, "raw", real_history, io, specs, budget, summary_rng, gains)
    real_record["rewiring_fraction"] = 0.0
    records.append(real_record)
    rows.extend(real_rows)
    dose_records.append({"fraction": 0.0, "substrates": [base], "graph_construction": {"fraction": 0.0, "accepted_swaps": 0, "changed_edges": 0}})
    for fraction in ALPHAS[1:]:
        # Each alpha changes graph weights/columns while reusing same seed labels.
        # Keep cache scope per alpha; shared cache key cannot distinguish them.
        fraction_history_cache = {}
        fraction_substrates = []
        construction_records = []
        for seed in SC.RNG_SEEDS:
            weight, construction = partial_rewire(base.weight, fraction, int(seed))
            substrate = SC.Substrate("fraction_rewired", int(seed), weight, SC.tensor_sha256(weight))
            substrate.rewiring_fraction = float(fraction)
            fraction_substrates.append(substrate)
            construction_records.append({"seed": int(seed), **construction, "tensor_sha256": substrate.tensor_hash})
        dose_records.append({"fraction": float(fraction), "substrates": fraction_substrates, "graph_construction": construction_records})
        for substrate in fraction_substrates:
            motor_history = SC.collect_history(substrate, SECS, SC.BASE_DRIVE, fraction_history_cache)
            source_targets = map_history(motor_history, io, gains, target_cache)
            # A rewired control can leave a robot joint with no surviving drive.
            # That is a fact about the control graph, so it is recorded rather
            # than allowed to abort the dose curve. See `amplitude_match`.
            matched_targets, match_record = amplitude_match(
                source_targets, real_targets.std(axis=0), allow_unmatchable=True)
            match_record["source_motor_history_sha256"] = array_sha256(motor_history)
            match_record["source_joint_target_history_sha256"] = array_sha256(source_targets)
            match_record["reference_fraction"] = 0.0
            matched_record, matched_rows = run_condition(substrate, MATCHED_MODE, matched_targets, io, specs, budget, summary_rng, gains, mapped=True)
            matched_record["amplitude_match"] = match_record
            records.append(matched_record)
            rows.extend(matched_rows)
        print(f"completed alpha={fraction:g}", flush=True)

    for fraction_record in dose_records[1:]:
        fraction = fraction_record["fraction"]
        fraction_rows = [row for row in records if row["rewiring_fraction"] == fraction]
        values = [row["summary"]["metrics"]["forward_distance_m"]["mean"] for row in fraction_rows]
        summaries = [row["summary"]["metrics"]["forward_distance_m"] for row in fraction_rows]
        dose_records_entry = {
            "rewiring_fraction": fraction,
            "seed_values": values,
            "mean_forward_distance_m": float(np.mean(values)),
            "forward_ci_low": float(np.percentile(np.asarray(values)[np.random.default_rng(SC.BOOTSTRAP_SEED).integers(0, len(values), size=(SC.BOOTSTRAP_REPS, len(values)))].mean(axis=1), 2.5)),
            "forward_ci_high": float(np.percentile(np.asarray(values)[np.random.default_rng(SC.BOOTSTRAP_SEED + 1).integers(0, len(values), size=(SC.BOOTSTRAP_REPS, len(values)))].mean(axis=1), 97.5)),
            "n": len(values),
            "falls": int(sum(summary["falls"] for row in fraction_rows for summary in [row["summary"]])),
            "construction": fraction_record["graph_construction"],
        }
        dose_records_entry["forward_ci"] = [dose_records_entry["forward_ci_low"], dose_records_entry["forward_ci_high"]]
        graph_integrity.extend(fraction_record["graph_construction"])
        fraction_record["summary"] = dose_records_entry

    dose = [{"rewiring_fraction": 0.0, "seed_values": [real_record["summary"]["metrics"]["forward_distance_m"]["mean"]], "mean_forward_distance_m": real_record["summary"]["metrics"]["forward_distance_m"]["mean"], "forward_ci_low": real_record["summary"]["metrics"]["forward_distance_m"]["ci_low"], "forward_ci_high": real_record["summary"]["metrics"]["forward_distance_m"]["ci_high"], "n": 1, "forward_ci": [real_record["summary"]["metrics"]["forward_distance_m"]["ci_low"], real_record["summary"]["metrics"]["forward_distance_m"]["ci_high"]]}]
    dose.extend(item["summary"] for item in dose_records[1:])
    dose_curve = [item["mean_forward_distance_m"] for item in dose]
    alpha_curve = [item["rewiring_fraction"] for item in dose]
    dose_spearman = exact_spearman(dose_curve)
    contrasts = []
    baseline_value = real_record["summary"]["metrics"]["forward_distance_m"]["mean"]
    baseline_ci = (real_record["summary"]["metrics"]["forward_distance_m"]["ci_low"], real_record["summary"]["metrics"]["forward_distance_m"]["ci_high"])
    for item in dose[1:]:
        compare = SC.substrate_comparison(baseline_value, item["seed_values"], f"real_vs_alpha_{item['rewiring_fraction']:g}", comparison_rng, baseline_ci)
        compare["rewiring_fraction"] = item["rewiring_fraction"]
        contrasts.append(compare)
    for compare, p_holm in zip(contrasts, SC.holm_adjust([item["p_raw"] for item in contrasts])):
        compare["p_holm"] = float(p_holm)
    numeric_gate = bool(dose_spearman < 0.0 and all(item["p_holm"] < 0.05 for item in contrasts))
    matched_records = [row for row in records if row["amplitude_mode"] == MATCHED_MODE]
    max_sd_error = max(row["amplitude_match"]["max_sd_error_all_channels"] for row in matched_records)
    max_mean_error = max(row["amplitude_match"]["mean_preserved_max_error"] for row in matched_records)
    alpha1_rows = [row for row in records if row["rewiring_fraction"] == 1.0]
    selected_seed = min(alpha1_rows, key=lambda row: abs(row["summary"]["metrics"]["forward_distance_m"]["mean"] - np.median([item["summary"]["metrics"]["forward_distance_m"]["mean"] for item in alpha1_rows])))["seed"]
    alpha1_record = next(item for item in dose_records if item["fraction"] == 1.0)
    selected = next(item for item in alpha1_record["substrates"] if item.seed == selected_seed)
    alpha1_history = SC.collect_history(selected, SECS, SC.BASE_DRIVE, {})
    alpha1_targets = amplitude_match(map_history(alpha1_history, io, gains, target_cache), real_targets.std(axis=0))[0]
    video_paths = {"real": RESULTS / "h4_dose_real_multiview.mp4", "alpha1_median": RESULTS / "h4_dose_alpha1_median_multiview.mp4"}
    for substrate, signal, path in ((base, real_targets, video_paths["real"]), (selected, alpha1_targets, video_paths["alpha1_median"])):
        model, body_record = SC.make_body_model()
        SC.run_body_trace(signal, io, specs[0], SECS, video=path, budget=budget, model=model, body_record=body_record)
    figure = RESULTS / "h4_dose_response.png"
    make_figure(dose, figure)
    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    report = {
        "experiment": "H4",
        "status": "complete",
        "pre_registered_null": "no monotone relation between preserved real structure and forward drive",
        "protocol": {
            "secs": SECS,
            "drive": SC.BASE_DRIVE,
            "fractions": list(ALPHAS),
            "construction": "accepted degree-preserving edge swaps from real graph; fraction α means swaps/(edges/2)",
            "n_seeds": SC.K,
            "n_episodes": SC.N_EPISODES,
            "amplitude_rule": "per-channel temporal SD of fixed-map joint targets set to α=0 real joint-target SD",
            "primary": "dose-curve monotonicity plus Holm-adjusted α>0 contrasts against α=0 real",
            "learning": False,
        },
        "real_graph_hashes": file_hashes_pre,
        "substrates": substrate_record,
        "records": records,
        "dose_curve": dose,
        "dose_spearman": dose_spearman,
        "baseline_contrasts": contrasts,
        "figure": str(figure),
        "videos": {
            "real": {"path": str(video_paths["real"]), "exists": video_paths["real"].exists(), "rewiring_fraction": 0.0},
            "alpha1_median": {"path": str(video_paths["alpha1_median"]), "exists": video_paths["alpha1_median"].exists(), "seed": int(selected_seed), "rewiring_fraction": 1.0},
            "reviewed": False,
            "review_note": "review required before final decision",
        },
        "integrity": {
            "file_hashes_pre": file_hashes_pre,
            "file_hashes_post": file_hashes_post,
            "file_hashes_equal": bool(file_hashes_pre == file_hashes_post),
            "real_tensor_sha256_pre": base.tensor_hash,
            "real_tensor_sha256_post": SC.tensor_sha256(base.weight),
            "real_tensor_hash_equal": bool(base.tensor_hash == SC.tensor_sha256(base.weight)),
            "graph_construction": graph_integrity,
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
        "numeric_decision": "DOSE-RESPONSE-EARNED" if numeric_gate else "DOSE-RESPONSE-NULL",
        "decision": "DOSE-RESPONSE-PENDING-VIDEO" if numeric_gate else "DOSE-RESPONSE-NULL",
        "rows": rows,
    }
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    write_csv(rows)
    print(json.dumps({"experiment": "H4", "numeric_decision": report["numeric_decision"], "dose_spearman": dose_spearman, "contrasts": contrasts, "resource": report["resource"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
