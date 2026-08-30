"""Measure inter-leg connectivity and coordination in an all-leg MANC run."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.ipc as ipc

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from sweep_vnc import autocorrelation_metrics, balance, correlation, spectral_metrics  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


TRIPOD_A = (0, 3, 4)
TRIPOD_B = (1, 2, 5)


def leg_scores(history: np.ndarray, io: np.lib.npyio.NpzFile) -> np.ndarray:
    legs = io["motor_leg"].astype(np.int64)
    joints = io["motor_joint"].astype(np.int64)
    signs = io["motor_sign"].astype(np.float64)
    scores = np.full((len(history), 6, 5), np.nan, dtype=np.float64)
    for leg in range(6):
        for joint in range(5):
            mask = (legs == leg) & (joints == joint)
            if not mask.any():
                continue
            positive = mask & (signs > 0)
            negative = mask & (signs < 0)
            score = np.zeros(len(history), dtype=np.float64)
            if positive.any():
                score += history[:, positive].mean(axis=1)
            if negative.any():
                score -= history[:, negative].mean(axis=1)
            scores[:, leg, joint] = score
    return scores


def phase_matrix(scores: np.ndarray, burn_fraction: float, joint: int = 0) -> np.ndarray:
    tail = scores[int(len(scores) * burn_fraction):, :, joint]
    matrix = np.full((6, 6), np.nan, dtype=np.float64)
    for left in range(6):
        for right in range(6):
            if np.isfinite(tail[:, left]).all() and np.isfinite(tail[:, right]).all():
                matrix[left, right] = correlation(tail[:, left], tail[:, right])
    np.fill_diagonal(matrix, 1.0)
    return matrix


def pair_mean(matrix: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    values = [matrix[left, right] for left, right in pairs if np.isfinite(matrix[left, right])]
    return float(np.mean(values)) if values else float("nan")


def published_coordination_report(io, output: Path) -> dict:
    """Persist the exact published candidate cells present in the rebuilt graph."""
    required = {
        "published_coordination",
        "published_coordination_body_ids",
        "published_coordination_types",
        "published_coordination_neuromeres",
        "published_coordination_sides",
    }
    if not required.issubset(io.files):
        return {"available": False}
    indices = io["published_coordination"].astype(np.int64)
    body_ids = io["published_coordination_body_ids"].astype(np.int64)
    node_ids = io["node_body_ids"].astype(np.int64)
    if not np.array_equal(node_ids[indices], body_ids):
        raise ValueError("published coordination index/body-id mismatch")
    records = pd.DataFrame({
        "body_id": body_ids,
        "selected_index": indices,
        "type": io["published_coordination_types"].astype("U16"),
        "soma_neuromere": io["published_coordination_neuromeres"].astype("U3"),
        "soma_side": io["published_coordination_sides"].astype("U2"),
        "selected_in_graph": True,
    }).sort_values(["soma_neuromere", "soma_side", "type", "body_id"])
    csv_output = output.with_name(output.name + "_published_coordination.csv")
    records.to_csv(csv_output, index=False)
    missing = io["published_coordination_types_missing"].astype("U16").tolist() if "published_coordination_types_missing" in io.files else []
    return {
        "available": True,
        "source": str(io["published_coordination_source"]) if "published_coordination_source" in io.files else "not recorded",
        "rule": str(io["published_coordination_rule"]) if "published_coordination_rule" in io.files else "not recorded",
        "candidate_count": int(len(records)),
        "unique_type_count": int(records["type"].nunique()),
        "type_labels": sorted(records["type"].unique().tolist()),
        "missing_published_type_labels": missing,
        "csv_output": str(csv_output),
    }


def connectivity(data: Path, io: np.lib.npyio.NpzFile, weights_name: str) -> np.ndarray:
    with (data / weights_name).open("rb") as f:
        weight = pickle.load(f).to_sparse_coo().coalesce()
    annotations = pd.read_feather(
        DATA / "manc" / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        columns=["bodyId", "somaNeuromere"],
    ).drop_duplicates("bodyId").set_index("bodyId")
    body_ids = io["node_body_ids"].astype(np.int64)
    neuromeres = annotations.reindex(body_ids)["somaNeuromere"].fillna("NA").to_numpy()
    edge = weight.indices().cpu().numpy()
    values = weight.values().cpu().numpy()
    names = ("T1", "T2", "T3", "NA")
    result = np.zeros((4, 4), dtype=np.float64)
    for row, column, value in zip(neuromeres[edge[0]], neuromeres[edge[1]], values):
        if row not in names or column not in names:
            continue
        result[names.index(row), names.index(column)] += abs(float(value))
    return result


def _annotation_codes() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    annotations = feather.read_table(
        DATA / "manc" / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        columns=["bodyId", "somaNeuromere", "somaSide", "superclass", "subclass", "type", "status"],
    ).to_pandas().dropna(subset=["bodyId"]).drop_duplicates("bodyId")
    annotations["bodyId"] = annotations["bodyId"].astype("int64")
    annotations = annotations.set_index("bodyId")
    keys = np.asarray(sorted(annotations.index), dtype=np.int64)
    codes = annotations.reindex(keys)["somaNeuromere"].map({"T1": 0, "T2": 1, "T3": 2}).fillna(-1).to_numpy(dtype=np.int8)
    return annotations, keys, codes


def _codes(ids: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(keys, ids)
    valid = positions < len(keys)
    hit = np.zeros(ids.shape, dtype=bool)
    hit[valid] = keys[positions[valid]] == ids[valid]
    result = np.full(ids.shape, -1, dtype=np.int8)
    result[hit] = values[positions[hit]]
    return result


def _selected_mask(ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(selected, ids)
    valid = positions < len(selected)
    result = np.zeros(ids.shape, dtype=bool)
    result[valid] = selected[positions[valid]] == ids[valid]
    return result


def _add_stats(store: defaultdict, ids: np.ndarray, values: np.ndarray) -> None:
    if not len(ids):
        return
    unique, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse)
    masses = np.bincount(inverse, weights=values)
    for body, count, mass in zip(unique, counts, masses):
        store[int(body)][0] += int(count)
        store[int(body)][1] += float(mass)


def _text(value) -> str:
    return "NA" if pd.isna(value) else str(value)


def _records(stats: dict, annotations: pd.DataFrame, spans: dict[int, int] | None = None) -> list[dict]:
    records = []
    for body, (count, mass) in stats.items():
        row = annotations.loc[body] if body in annotations.index else None
        record = {
            "body_id": int(body),
            "cross_edge_count": int(count),
            "absolute_synapse_mass": float(mass),
            "soma_neuromere": _text(row["somaNeuromere"]) if row is not None else "NA",
            "soma_side": _text(row["somaSide"]) if row is not None else "NA",
            "superclass": _text(row["superclass"]) if row is not None else "NA",
            "subclass": _text(row["subclass"]) if row is not None else "NA",
            "type": _text(row["type"]) if row is not None else "NA",
            "status": _text(row["status"]) if row is not None else "NA",
        }
        if spans is not None:
            record["target_neuromere_count"] = int(spans.get(int(body), 0))
        records.append(record)
    return sorted(records, key=lambda record: (-record["absolute_synapse_mass"], -record["cross_edge_count"], record["body_id"]))


def coupling_report(data: Path, io: np.lib.npyio.NpzFile, weights_name: str, output: Path) -> dict:
    """Separate retained cross-neuromere edges from traced candidates omitted by selection."""
    annotations, annotation_keys, annotation_codes = _annotation_codes()
    selected = np.sort(io["node_body_ids"].astype(np.int64))
    with (data / weights_name).open("rb") as f:
        weight = pickle.load(f).to_sparse_coo().coalesce()
    node_ids = io["node_body_ids"].astype(np.int64)
    edge = weight.indices().cpu().numpy()
    values = np.abs(weight.values().cpu().numpy())
    pre = node_ids[edge[1]]
    post = node_ids[edge[0]]
    pre_code = _codes(pre, annotation_keys, annotation_codes)
    post_code = _codes(post, annotation_keys, annotation_codes)
    selected_cross = (pre_code >= 0) & (post_code >= 0) & (pre_code != post_code)
    retained_sources: defaultdict = defaultdict(lambda: [0, 0.0])
    retained_targets: defaultdict = defaultdict(lambda: [0, 0.0])
    _add_stats(retained_sources, pre[selected_cross], values[selected_cross])
    _add_stats(retained_targets, post[selected_cross], values[selected_cross])
    if selected_cross.any():
        source_target_pairs = np.unique(np.stack([pre[selected_cross], post_code[selected_cross]], axis=1), axis=0)
        source_ids, source_spans = np.unique(source_target_pairs[:, 0], return_counts=True)
        spans = {int(body): int(count) for body, count in zip(source_ids, source_spans)}
    else:
        spans = {}

    raw_path = DATA / "manc" / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw MANC weights required for coupling audit: {raw_path}")
    raw_cross_edges = 0
    raw_cross_mass = 0.0
    boundary_edges = 0
    boundary_mass = 0.0
    omitted: defaultdict = defaultdict(lambda: [0, 0.0])
    reader = ipc.open_file(raw_path)
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        batch_pre = batch.column("body_pre").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        batch_post = batch.column("body_post").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        batch_values = batch.column("weight").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        floor = np.abs(batch_values) >= 5
        batch_pre = batch_pre[floor]
        batch_post = batch_post[floor]
        batch_values = np.abs(batch_values[floor])
        batch_pre_code = _codes(batch_pre, annotation_keys, annotation_codes)
        batch_post_code = _codes(batch_post, annotation_keys, annotation_codes)
        cross = (batch_pre_code >= 0) & (batch_post_code >= 0) & (batch_pre_code != batch_post_code)
        if not cross.any():
            continue
        batch_pre = batch_pre[cross]
        batch_post = batch_post[cross]
        batch_values = batch_values[cross]
        raw_cross_edges += len(batch_pre)
        raw_cross_mass += float(batch_values.sum())
        selected_pre = _selected_mask(batch_pre, selected)
        selected_post = _selected_mask(batch_post, selected)
        boundary = selected_pre ^ selected_post
        if boundary.any():
            boundary_edges += int(boundary.sum())
            boundary_mass += float(batch_values[boundary].sum())
            omitted_ids = np.where(selected_pre[boundary], batch_post[boundary], batch_pre[boundary])
            _add_stats(omitted, omitted_ids, batch_values[boundary])

    source_records = _records(retained_sources, annotations, spans)
    target_records = _records(retained_targets, annotations)
    omitted_records = _records(omitted, annotations)
    class_stats: defaultdict = defaultdict(lambda: {"cross_edge_count": 0, "absolute_synapse_mass": 0.0, "node_ids": set()})
    for record in omitted_records:
        key = (record["soma_neuromere"], record["superclass"], record["subclass"])
        class_stats[key]["cross_edge_count"] += record["cross_edge_count"]
        class_stats[key]["absolute_synapse_mass"] += record["absolute_synapse_mass"]
        class_stats[key]["node_ids"].add(record["body_id"])
    class_records = [
        {
            "soma_neuromere": key[0],
            "superclass": key[1],
            "subclass": key[2],
            "cross_edge_count": value["cross_edge_count"],
            "absolute_synapse_mass": value["absolute_synapse_mass"],
            "unique_nodes": len(value["node_ids"]),
        }
        for key, value in class_stats.items()
    ]
    class_records.sort(key=lambda record: (-record["absolute_synapse_mass"], -record["cross_edge_count"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_records).to_csv(output.with_name(output.name + "_coupling_sources.csv"), index=False)
    pd.DataFrame(target_records).to_csv(output.with_name(output.name + "_coupling_targets.csv"), index=False)
    pd.DataFrame(omitted_records).to_csv(output.with_name(output.name + "_coupling_omitted.csv"), index=False)
    pd.DataFrame(class_records).to_csv(output.with_name(output.name + "_coupling_omitted_classes.csv"), index=False)
    return {
        "floor_synapses": 5,
        "raw_floored_cross_edge_count": int(raw_cross_edges),
        "raw_floored_cross_absolute_synapse_mass": float(raw_cross_mass),
        "boundary_floored_cross_edge_count": int(boundary_edges),
        "boundary_floored_cross_absolute_synapse_mass": float(boundary_mass),
        "retained_cross_edge_count": int(selected_cross.sum()),
        "retained_cross_absolute_synapse_mass": float(values[selected_cross].sum()),
        "retained_cross_source_count": len(source_records),
        "retained_cross_target_count": len(target_records),
        "retained_sources_reaching_multiple_neuromeres": int(sum(count >= 2 for count in spans.values())),
        "top_retained_sources": source_records[:20],
        "not_retained_boundary_candidate_count": len(omitted_records),
        "top_not_retained_boundary_candidates": omitted_records[:20],
        "not_retained_boundary_classes": class_records[:30],
        "csv_outputs": [
            str(output.with_name(output.name + "_coupling_sources.csv")),
            str(output.with_name(output.name + "_coupling_targets.csv")),
            str(output.with_name(output.name + "_coupling_omitted.csv")),
            str(output.with_name(output.name + "_coupling_omitted_classes.csv")),
        ],
    }


def make_figure(scores: np.ndarray, phase: np.ndarray, edge_mass: np.ndarray, io, dt_sample: float, burn_fraction: float, output: Path, drive: float, gW: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    names = [str(x) for x in io["leg_names"]]
    tail = scores[int(len(scores) * burn_fraction):, :, 0]
    window = tail[-min(len(tail), max(8, int(round(8.0 / dt_sample)))):]
    time_s = np.arange(len(window)) * dt_sample
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), layout="constrained")
    for leg in range(6):
        axes[0, 0].plot(time_s, window[:, leg], color=colors[leg], linewidth=1.1, label=names[leg])
    axes[0, 0].set(xlabel="Time in validation window (s)", ylabel="Coxa score (normalized)", title="Unsmoothed coxa-pool traces")
    axes[0, 0].legend(ncol=3, frameon=False, fontsize=8)
    image = axes[0, 1].imshow(phase, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0, 1].set(xticks=range(6), xticklabels=names, yticks=range(6), yticklabels=names, title="Coxa phase correlation")
    axes[0, 1].tick_params(axis="x", rotation=35)
    fig.colorbar(image, ax=axes[0, 1], label="Pearson correlation")
    pairs = {
        "tripod A\nwithin": [(a, b) for i, a in enumerate(TRIPOD_A) for b in TRIPOD_A[i + 1:]],
        "tripod B\nwithin": [(a, b) for i, a in enumerate(TRIPOD_B) for b in TRIPOD_B[i + 1:]],
        "cross-tripod": [(a, b) for a in TRIPOD_A for b in TRIPOD_B],
    }
    values = [pair_mean(phase, pair_list) for pair_list in pairs.values()]
    bars = axes[1, 0].bar(tuple(pairs), values, color=("#0072B2", "#D55E00", "#009E73"), edgecolor="black", hatch=("///", "\\\\", ".."))
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_ylim(-1.05, 1.05)
    axes[1, 0].set_ylabel("Mean pairwise correlation")
    axes[1, 0].set_title("Tripod coordination proxy")
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value + (0.06 if value >= 0 else -0.12), f"{value:.2f}", ha="center", fontsize=9)
    im2 = axes[1, 1].imshow(edge_mass, vmin=0, vmax=max(1.0, edge_mass.max()), cmap="cividis")
    axes[1, 1].set(xticks=range(4), xticklabels=("T1", "T2", "T3", "NA"), yticks=range(4), yticklabels=("T1", "T2", "T3", "NA"), title="Absolute signed edge mass")
    fig.colorbar(im2, ax=axes[1, 1], label="Absolute signed edge mass")
    fig.suptitle(f"All-leg MANC coordination — paper model, drive={drive:g}, gW={gW:g}", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=RESULTS)
    ap.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    ap.add_argument("--index_name", default="all_legs_io_index.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "all_legs_coordination")
    ap.add_argument("--model", choices=("bounded", "paper"), default="paper")
    ap.add_argument("--drive", type=float, default=100.0)
    ap.add_argument("--gW", type=float, default=0.03)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dnb08_gain", type=float, default=1.0)
    ap.add_argument("--heterogeneity", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--burn_fraction", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=7.5)
    ap.add_argument("--activation_gain", type=float, default=1.0)
    ap.add_argument("--fr_cap", type=float, default=200.0)
    a = ap.parse_args()
    if a.steps < 32 or not 0 <= a.burn_fraction < 1:
        raise ValueError("invalid steps/burn_fraction")
    io = np.load(a.data / a.index_name)
    vnc = VNCStepper(a.data, substeps=a.substeps, tau=a.tau, dt=a.dt, gW=a.gW, weights_name=a.weights_name, index_name=a.index_name, model=a.model, threshold=a.threshold, activation_gain=a.activation_gain, fr_cap=a.fr_cap, heterogeneity=a.heterogeneity, seed=a.seed)
    current = vnc.descending_current(a.drive, dnb08_gain=a.dnb08_gain)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(a.steps)])
    scores = leg_scores(history, io)
    matrix = phase_matrix(scores, a.burn_fraction)
    edge_mass = connectivity(a.data, io, a.weights_name)
    output = a.out
    coupling = coupling_report(a.data, io, a.weights_name, output)
    published_coordination = published_coordination_report(io, output)
    make_figure(scores, matrix, edge_mass, io, a.dt * a.substeps, a.burn_fraction, output, a.drive, a.gW)
    names = [str(x) for x in io["leg_names"]]
    tripod_pairs = {
        "tripod_A_within_mean": pair_mean(matrix, [(a, b) for i, a in enumerate(TRIPOD_A) for b in TRIPOD_A[i + 1:]]),
        "tripod_B_within_mean": pair_mean(matrix, [(a, b) for i, a in enumerate(TRIPOD_B) for b in TRIPOD_B[i + 1:]]),
        "cross_tripod_mean": pair_mean(matrix, [(a, b) for a in TRIPOD_A for b in TRIPOD_B]),
    }
    manifest = {
        "data": str(a.data),
        "weights_name": a.weights_name,
        "index_name": a.index_name,
        "model": a.model,
        "drive": a.drive,
        "gW": a.gW,
        "tau": a.tau,
        "dt_s": a.dt,
        "substeps": a.substeps,
        "sample_dt_s": a.dt * a.substeps,
        "steps": a.steps,
        "burn_fraction": a.burn_fraction,
        "constant_input_temporal_std": 0.0,
        "heterogeneity": a.heterogeneity,
        "seed": a.seed,
        "overall_rhythm": spectral_metrics(history, a.dt * a.substeps, a.burn_fraction) | autocorrelation_metrics(history, a.dt * a.substeps, a.burn_fraction),
        "coxa_phase_matrix": matrix.tolist(),
        "leg_names": names,
        "tripod_phase": tripod_pairs,
        "neuromere_edge_mass_rows_post_cols_pre": edge_mass.tolist(),
        "matrix_balance": balance(a.data, a.weights_name),
        "coupling": coupling,
        "published_coordination": published_coordination,
        "alt_text": "All-leg MANC constant-drive audit with unsmoothed coxa traces, coxa phase-correlation matrix, tripod correlation bars, and absolute signed edge mass across T1/T2/T3.",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    np.savetxt(output.with_suffix(".csv"), matrix, delimiter=",", header=",".join(names), comments="")
    print(json.dumps({"overall_rhythm": manifest["overall_rhythm"], "tripod_phase": tripod_pairs, "edge_mass": edge_mass.tolist(), "coupling": {key: coupling[key] for key in ("raw_floored_cross_edge_count", "retained_cross_edge_count", "retained_cross_source_count", "retained_cross_target_count", "not_retained_boundary_candidate_count")}}, sort_keys=True))
    print(f"audit -> {output.with_suffix('.json')}")
    print(f"figure -> {output.with_suffix('.png')} / {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
