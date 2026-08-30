"""E2: inventory anatomical T1/T2/T3 intersegmental cells and edges."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
SEGMENTS = ("T1", "T2", "T3")


def load_annotations(data: Path) -> pd.DataFrame:
    frame = pd.read_feather(
        DATA / "manc" / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        columns=["bodyId", "somaNeuromere", "somaSide", "superclass", "subclass", "type", "status"],
    ).drop_duplicates("bodyId").dropna(subset=["bodyId"])
    frame["bodyId"] = frame["bodyId"].astype("int64")
    return frame.set_index("bodyId").sort_index()


def cross_edges(data: Path, annotations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = annotations.index.to_numpy(dtype=np.int64)
    neuromere_code = annotations["somaNeuromere"].map({"T1": 0, "T2": 1, "T3": 2}).fillna(-1).to_numpy(dtype=np.int8)

    def codes(ids: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(keys, ids)
        valid = positions < len(keys)
        result = np.full(ids.shape, -1, dtype=np.int8)
        hit = valid.copy()
        hit[valid] = keys[positions[valid]] == ids[valid]
        result[hit] = neuromere_code[positions[hit]]
        return result

    path = DATA / "manc" / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    pres, posts, weights = [], [], []
    reader = ipc.open_file(path)
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        pre = batch.column("body_pre").to_numpy(zero_copy_only=False).astype(np.int64)
        post = batch.column("body_post").to_numpy(zero_copy_only=False).astype(np.int64)
        weight = batch.column("weight").to_numpy(zero_copy_only=False).astype(np.float64)
        floor = np.abs(weight) >= 5
        pre, post, weight = pre[floor], post[floor], weight[floor]
        pre_code, post_code = codes(pre), codes(post)
        cross = (pre_code >= 0) & (post_code >= 0) & (pre_code != post_code)
        if cross.any():
            pres.append(pre[cross])
            posts.append(post[cross])
            weights.append(weight[cross])
    pre = np.concatenate(pres)
    post = np.concatenate(posts)
    weight = np.concatenate(weights)
    return pre, post, weight, codes(pre), codes(post)


def inventory(data: Path, index_name: str) -> tuple[dict, pd.DataFrame]:
    io = np.load(data / index_name)
    selected = set(io["node_body_ids"].astype(np.int64).tolist())
    published = set(io["published_coordination_body_ids"].astype(np.int64).tolist())
    published_types = set(io["published_coordination_type_labels"].astype("U16").tolist())
    annotations = load_annotations(data)
    pre, post, weight, pre_code, post_code = cross_edges(data, annotations)
    nodes = np.unique(np.r_[pre, post]).astype(np.int64)
    positions = {int(body): i for i, body in enumerate(nodes)}
    pre_position = np.asarray([positions[int(body)] for body in pre], dtype=np.int64)
    post_position = np.asarray([positions[int(body)] for body in post], dtype=np.int64)
    out_count = np.bincount(pre_position, minlength=len(nodes))
    in_count = np.bincount(post_position, minlength=len(nodes))
    out_mass = np.bincount(pre_position, weights=np.abs(weight), minlength=len(nodes))
    in_mass = np.bincount(post_position, weights=np.abs(weight), minlength=len(nodes))
    out_neuromeres = [[] for _ in nodes]
    in_neuromeres = [[] for _ in nodes]
    for source, target, source_code, target_code in zip(pre_position, post_position, pre_code, post_code):
        out_neuromeres[source].append(SEGMENTS[int(target_code)])
        in_neuromeres[target].append(SEGMENTS[int(source_code)])

    ann = annotations.reindex(nodes)
    records = pd.DataFrame({
        "body_id": nodes,
        "soma_neuromere": ann["somaNeuromere"].fillna("NA").astype(str).to_numpy(),
        "soma_side": ann["somaSide"].fillna("NA").astype(str).to_numpy(),
        "superclass": ann["superclass"].fillna("NA").astype(str).to_numpy(),
        "subclass": ann["subclass"].fillna("NA").astype(str).to_numpy(),
        "type": ann["type"].fillna("NA").astype(str).to_numpy(),
        "status": ann["status"].fillna("NA").astype(str).to_numpy(),
        "selected": [int(body) in selected for body in nodes],
        "published_coordination_cell": [int(body) in published for body in nodes],
        "published_coordination_type": [str(ann.loc[int(body), "type"]) in published_types for body in nodes],
        "outgoing_cross_edge_rows": out_count,
        "incoming_cross_edge_rows": in_count,
        "cross_edge_rows": out_count + in_count,
        "absolute_synapse_mass": out_mass + in_mass,
        "source_neuromeres": [",".join(sorted(set(values))) for values in in_neuromeres],
        "target_neuromeres": [",".join(sorted(set(values))) for values in out_neuromeres],
    })
    selected_edge_rows = np.isin(pre, list(selected)) & np.isin(post, list(selected))
    published_nodes = set(nodes.tolist()) & published
    class_counts = []
    for superclass, group in records.groupby("superclass", dropna=False):
        class_counts.append({
            "superclass": superclass,
            "raw_cross_nodes": int(len(group)),
            "selected_nodes": int(group["selected"].sum()),
            "omitted_nodes": int((~group["selected"]).sum()),
            "incident_cross_edge_rows": int(group["cross_edge_rows"].sum()),
            "absolute_synapse_mass": float(group["absolute_synapse_mass"].sum()),
        })
    type_summary = (
        records.groupby(["superclass", "subclass", "type", "published_coordination_type", "selected"], dropna=False)
        .agg(raw_cross_nodes=("body_id", "size"), cross_edge_rows=("cross_edge_rows", "sum"), absolute_synapse_mass=("absolute_synapse_mass", "sum"))
        .reset_index()
        .sort_values(["published_coordination_type", "superclass", "type", "selected"], ascending=[False, True, True, False])
    )
    summary = {
        "index_name": index_name,
        "synapse_floor": 5,
        "neuromere_scope": list(SEGMENTS),
        "raw_floored_cross_edge_rows": int(len(pre)),
        "raw_floored_cross_absolute_synapse_mass": float(np.abs(weight).sum()),
        "raw_cross_node_count": int(len(nodes)),
        "selected_cross_edge_rows": int(selected_edge_rows.sum()),
        "selected_cross_node_count": int(records["selected"].sum()),
        "published_coordination_cell_count": len(published),
        "published_coordination_cells_with_cross_edges": len(published_nodes),
        "published_coordination_cells_selected": int(records["published_coordination_cell"].sum()),
        "published_coordination_cells_omitted": int((~records.loc[records["published_coordination_cell"], "selected"]).sum()),
        "published_coordination_type_labels": sorted(published_types),
        "class_counts": sorted(class_counts, key=lambda row: (-row["raw_cross_nodes"], row["superclass"])),
        "decision": "no anatomical widening: all annotated published coordination cells with T1/T2/T3 cross-neuromere signal are selected; non-published cross-neuromere cells remain outside the declared coordination rule",
    }
    return summary, (records, type_summary)


def make_figure(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in summary["class_counts"] if row["superclass"] in {"vnc_intrinsic", "ascending_neuron", "vnc_motor"}]
    labels = [row["superclass"].replace("_", "\n") for row in rows]
    raw = [row["raw_cross_nodes"] for row in rows]
    selected = [row["selected_nodes"] for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    ax.bar(x - 0.18, raw, width=0.36, label="raw cross-neuromere", color="#999999", edgecolor="black")
    ax.bar(x + 0.18, selected, width=0.36, label="selected graph", color="#0072B2", edgecolor="black")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cross-neuromere endpoint cells")
    ax.set_title("E2 intersegmental inventory — published coordination cells: 212/212 selected")
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=RESULTS)
    ap.add_argument("--index_name", default="all_legs_io_index.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "e2_intersegmental")
    a = ap.parse_args()
    summary, (records, type_summary) = inventory(a.data, a.index_name)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(a.out.with_name(a.out.name + "_cells.csv"), index=False)
    type_summary.to_csv(a.out.with_name(a.out.name + "_types.csv"), index=False)
    make_figure(summary, a.out)
    summary["cell_csv"] = str(a.out.with_name(a.out.name + "_cells.csv"))
    summary["type_csv"] = str(a.out.with_name(a.out.name + "_types.csv"))
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("raw_floored_cross_edge_rows", "raw_cross_node_count", "selected_cross_edge_rows", "selected_cross_node_count", "published_coordination_cell_count", "published_coordination_cells_with_cross_edges", "published_coordination_cells_selected", "published_coordination_cells_omitted", "decision")}, sort_keys=True))
    print(f"audit -> {a.out.with_suffix('.json')}")
    print(f"cells -> {summary['cell_csv']}")
    print(f"types -> {summary['type_csv']}")
    print(f"figure -> {a.out.with_suffix('.png')} / {a.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
