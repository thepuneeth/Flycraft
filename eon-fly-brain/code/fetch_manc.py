"""Build a signed MANC VNC graph from Janelia's public flat files.

Default scope follows the published MANC construction: traced leg motor
neurons <- direct non-descending premotor neurons <- traced descending neurons,
plus every traced T1/T2/T3 intrinsic neuron in the published leg-coordination
candidate type set. The graph stores rows as postsynaptic neurons and columns
as presynaptic neurons, matching ``W @ r`` in the brain stepper.

No neuPrint token is needed. Download the three official Feather files first
into ``data/manc``; this script only writes the compact, runnable artifacts.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import torch


HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
MANC = DATA / "manc"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

FLY_JOINTS = ("coxa", "trochanter", "femur", "tibia", "tarsus")
HEX_LEGS = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")

# Fixed, declared reduction used later by mn_to_joint.py:
# coxa -> coxa, trochanter+femur -> femur, tibia+tarsus -> tibia.
NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    # Paper-consistent MANC construction drops unclassified/modulatory signs.
}

MOTOR_SUBCLASS = {"T1": "fl", "T2": "ml", "T3": "hl"}

# Cheong et al., eLife 96084, Supplementary file 8. These are the intrinsic
# neuron types appearing in the published sequential, ipsilateral-all-leg,
# and bilateral leg-coordination groups. The type set is applied by current
# MANC annotation, then restricted to the requested T1/T2/T3 neuromeres.
PUBLISHED_COORDINATION_SOURCE = (
    "https://cdn.elifesciences.org/articles/96084/elife-96084-supp8-v1.csv"
)
PUBLISHED_COORDINATION_TYPES = frozenset({
    "IN00A001", "IN00A002", "IN01A001", "IN01A002", "IN01A007", "IN01A009",
    "IN01A010", "IN01A011", "IN01A018", "IN02A003", "IN03A001", "IN03A009",
    "IN03B015", "IN03B021", "IN03B025", "IN06A028", "IN06B001", "IN06B012",
    "IN06B018", "IN06B022", "IN06B028", "IN07B001", "IN07B002", "IN07B006",
    "IN07B007", "IN07B009", "IN07B010", "IN07B104", "IN08B001", "IN08B004",
    "IN09A004", "IN09A009", "IN10B004", "IN10B014", "IN11A003", "IN12A001",
    "IN12B002", "IN12B005", "IN12B013", "IN14B001", "IN14B005",
    "IN14B006", "IN17A007", "IN18B006", "IN18B011", "IN19A006", "IN19A009",
    "IN19A010", "IN19A012", "IN19A014", "IN19A019", "IN19B004", "IN19B005", "IN19B021",
    "IN19B027", "IN27X001", "IN27X002", "IN27X005", "INXXX003", "INXXX008",
    "INXXX025", "INXXX036", "INXXX058", "INXXX115", "LBL40", "vMS17",
})


def log(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def require_columns(frame: pd.DataFrame, *columns: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"MANC annotations missing columns: {missing}")


def load_annotations(path: Path) -> pd.DataFrame:
    ann = feather.read_table(path).to_pandas()
    require_columns(
        ann,
        "bodyId", "superclass", "subclass", "somaNeuromere", "somaSide",
        "rootSide", "entryNerve", "class", "status", "type", "instance",
    )
    ann = ann.dropna(subset=["bodyId"]).copy()
    ann["bodyId"] = ann["bodyId"].astype("int64")
    ann = ann.drop_duplicates("bodyId", keep="first")
    return ann


def batches(path: Path):
    reader = ipc.open_file(path)
    require_columns(pd.DataFrame(columns=reader.schema.names), "body_pre", "body_post", "weight")
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        yield (
            batch.column("body_pre").to_numpy(zero_copy_only=False),
            batch.column("body_post").to_numpy(zero_copy_only=False),
            batch.column("weight").to_numpy(zero_copy_only=False),
        )


def isin_sorted(values: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    """Membership test without allocating a second hash table per batch."""
    positions = np.searchsorted(sorted_ids, values)
    valid = positions < len(sorted_ids)
    hit = np.zeros(values.shape, dtype=bool)
    hit[valid] = sorted_ids[positions[valid]] == values[valid]
    return hit


def graph_predecessors(weights: Path, targets: set[int]) -> tuple[set[int], int]:
    target_ids = np.asarray(sorted(targets), dtype=np.int64)
    predecessors: set[int] = set()
    edge_count = 0
    for pre, post, _ in batches(weights):
        hit = isin_sorted(post, target_ids)
        if hit.any():
            edge_count += int(hit.sum())
            predecessors.update(pre[hit].tolist())
    return predecessors, edge_count


def compact_members(ids: set[int], body_to_idx: dict[int, int]) -> np.ndarray:
    return np.asarray(sorted(body_to_idx[x] for x in ids if x in body_to_idx), dtype=np.int64)


def text(row: pd.Series) -> str:
    return " ".join(str(row.get(c, "")) for c in ("type", "instance") if pd.notna(row.get(c)))


def fly_joint(label: str) -> int:
    """Map MANC motor labels to five fixed fly joint groups.

    Proximal coxa labels are the promotor/remotor/rotator/adductor groups;
    abbreviated Tr/Fe/Ti/Ta labels map to their named distal joint. Generic
    ltm neurons remain excluded rather than being silently guessed.
    """
    s = label.lower()
    if "ltm2-femur" in s or re.search(r"\bfe\b|femur", s):
        return 2
    if "ltm1-tibia" in s or re.search(r"\bti\b|tibia", s):
        return 3
    if re.search(r"\bta\b|tarsus", s):
        return 4
    if "tergotr" in s or "sternotrochanter" in s or re.search(r"\btr\b|\bacc\. tr\b", s):
        return 1
    if any(token in s for token in ("promotor", "remotor", "abductor", "adductor", "rotator")):
        return 0
    return -1


def motor_polarity(label: str) -> float:
    """Fixed flexor/extensor-style sign for the motor-pool readout."""
    s = label.lower()
    # Resolve compound labels by their first named action: remotor/abductor
    # is treated as remotor here, not silently as the opposite abductor sign.
    if "remotor/abductor" in s:
        return -1.0
    if "flexor" in s or "promotor" in s or "abductor" in s or "anterior rotator" in s or "levator" in s:
        return 1.0
    if "extensor" in s or "reductor" in s or "remotor" in s or "adductor" in s or "posterior rotator" in s or "depressor" in s:
        return -1.0
    return 0.0


def nt_signs(nt_path: Path, source_ids: np.ndarray) -> dict[int, float]:
    # MANC paper uses per-neuron predictedNt for synaptic signs.
    nt = feather.read_table(nt_path, columns=["body", "predicted_nt"]).to_pandas()
    nt["body"] = nt["body"].astype("int64")
    nt = nt.assign(_sign=nt["predicted_nt"].map(NT_SIGN).fillna(0.0)).drop_duplicates("body", keep="first")
    wanted = set(source_ids.tolist())
    return {int(body): float(sign) for body, sign in nt.loc[nt.body.isin(wanted), ["body", "_sign"]].itertuples(index=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--neuromeres", default="T1", help="comma-separated MANC neuromeres; default T1")
    ap.add_argument("--premotor_scope", choices=("intrinsic", "all"), default="all")
    ap.add_argument(
        "--include_sensory_feedback",
        action="store_true",
        help="include traced T1/T2/T3 leg proprioceptors for the E4 feedback experiment",
    )
    ap.add_argument("--synapse_floor", type=int, default=5, help="drop connections below this synapse count; paper default 5")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--weights_out", type=Path, default=None)
    a = ap.parse_args()

    manc = a.data / "manc"
    ann_path, nt_path, weights_path = (manc / name for name in (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS))
    for path in (ann_path, nt_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; download official MANC flat files into data/manc")

    neuromeres = tuple(x.strip() for x in a.neuromeres.split(",") if x.strip())
    if not neuromeres:
        raise ValueError("--neuromeres cannot be empty")
    unknown_neuromeres = sorted(set(neuromeres) - set(MOTOR_SUBCLASS))
    if unknown_neuromeres:
        raise ValueError(f"unsupported leg neuromeres {unknown_neuromeres}; expected T1, T2, or T3")
    if a.synapse_floor < 0:
        raise ValueError("--synapse_floor must be non-negative")
    t0 = time.time()
    ann = load_annotations(ann_path)
    traced = ann["status"].eq("Traced")
    motor_subclasses = [MOTOR_SUBCLASS[neuromere] for neuromere in neuromeres]
    motor_mask = (
        traced
        & ann["superclass"].eq("vnc_motor")
        & ann["subclass"].isin(motor_subclasses)
        & ann["somaNeuromere"].isin(neuromeres)
    )
    motor_ids = set(ann.loc[motor_mask, "bodyId"].astype("int64"))
    if not motor_ids:
        raise ValueError(f"no traced mapped leg motor neurons found for {neuromeres}")
    log(f"motor neurons: {len(motor_ids)} ({neuromeres})")

    direct_pre, direct_edges = graph_predecessors(weights_path, motor_ids)
    known_traced = set(ann.loc[traced, "bodyId"].astype("int64"))
    if a.premotor_scope == "intrinsic":
        premotor_mask = traced & ann["superclass"].eq("vnc_intrinsic")
        premotor_ids = direct_pre & set(ann.loc[premotor_mask, "bodyId"].astype("int64"))
    else:
        descending_mask = traced & ann["superclass"].fillna("").str.contains("descending", case=False)
        non_descending = known_traced - set(ann.loc[descending_mask, "bodyId"].astype("int64"))
        non_descending -= set(ann.loc[traced & ann["superclass"].eq("vnc_motor"), "bodyId"].astype("int64"))
        premotor_ids = direct_pre & non_descending
    log(f"direct edges to MNs: {direct_edges}; traced premotor candidates: {len(premotor_ids)}")

    second_pre, second_edges = graph_predecessors(weights_path, premotor_ids)
    desc_mask = traced & ann["superclass"].fillna("").str.contains("descending", case=False)
    known_desc = set(ann.loc[desc_mask, "bodyId"].astype("int64"))
    descending_ids = second_pre & known_desc

    exact_dng = set(ann.loc[traced & ann["type"].eq("DNg100"), "bodyId"].astype("int64"))
    exact_dnb = set(ann.loc[traced & ann["type"].eq("DNb08"), "bodyId"].astype("int64"))
    descending_ids |= exact_dng | exact_dnb

    core_labels = ("IN17A001", "INXXX466", "IN16B036")
    core_mask = traced & ann["type"].isin(core_labels) & ann["somaNeuromere"].isin(neuromeres)
    rhythm_core = set(ann.loc[core_mask, "bodyId"].astype("int64"))
    coordination_mask = (
        traced
        & ann["superclass"].eq("vnc_intrinsic")
        & ann["type"].isin(PUBLISHED_COORDINATION_TYPES)
        & ann["somaNeuromere"].isin(neuromeres)
    )
    coordination_ids = set(ann.loc[coordination_mask, "bodyId"].astype("int64"))
    present_coordination_types = sorted(
        set(ann.loc[
            traced
            & ann["superclass"].eq("vnc_intrinsic")
            & ann["somaNeuromere"].isin(neuromeres),
            "type",
        ].dropna())
        & PUBLISHED_COORDINATION_TYPES
    )
    missing_coordination_types = sorted(PUBLISHED_COORDINATION_TYPES - set(present_coordination_types))
    sensory_nerves = {"T1": "ProLN", "T2": "MesoLN", "T3": "MetaLN"}
    sensory_mask = (
        traced
        & ann["superclass"].eq("vnc_sensory")
        & ann["class"].eq("mechanosensory_proprioceptive")
        & ann["subclass"].isin(("campaniform sensilla", "hair plate", "chordotonal organ", "leg"))
        & ann["entryNerve"].isin([sensory_nerves[neuromere] for neuromere in neuromeres])
        & ann["rootSide"].isin(("L", "R"))
    )
    sensory_ids = set(ann.loc[sensory_mask, "bodyId"].astype("int64")) if a.include_sensory_feedback else set()
    selected_ids = motor_ids | premotor_ids | descending_ids | rhythm_core | coordination_ids | sensory_ids
    log(f"edges to premotor candidates: {second_edges}; descending inputs: {len(descending_ids)}")
    log(
        f"published coordination candidates: {len(coordination_ids)} "
        f"({len(present_coordination_types)}/{len(PUBLISHED_COORDINATION_TYPES)} types present; "
        f"missing={missing_coordination_types})"
    )
    log(f"selected nodes: {len(selected_ids)}; rhythm-core nodes: {len(rhythm_core)}")
    if a.include_sensory_feedback:
        log(f"sensory feedback cells: {len(sensory_ids)} (fixed proprioceptive anatomy rule)")

    node_body_ids = np.asarray(sorted(selected_ids), dtype=np.int64)
    body_to_idx = {int(body): i for i, body in enumerate(node_body_ids)}
    selected_sorted = node_body_ids
    edge_rows: list[np.ndarray] = []
    edge_cols: list[np.ndarray] = []
    edge_weights: list[np.ndarray] = []
    induced_edges = 0
    for pre, post, weight in batches(weights_path):
        pre_hit = isin_sorted(pre, selected_sorted)
        post_hit = isin_sorted(post, selected_sorted)
        hit = pre_hit & post_hit
        if hit.any():
            induced_edges += int(hit.sum())
            edge_rows.append(post[hit].astype(np.int64, copy=False))
            edge_cols.append(pre[hit].astype(np.int64, copy=False))
            edge_weights.append(weight[hit].astype(np.float32, copy=False))
    if not edge_rows:
        raise ValueError("selected MANC nodes have no induced edges")

    pre_body = np.concatenate(edge_cols)
    post_body = np.concatenate(edge_rows)
    raw_values = np.concatenate(edge_weights)
    floor_keep = np.abs(raw_values) >= float(a.synapse_floor)
    signs = nt_signs(nt_path, np.unique(pre_body))
    sign = np.asarray([signs.get(int(body), 0.0) for body in pre_body], dtype=np.float32)
    values = raw_values * sign
    keep = floor_keep & (values != 0.0)
    rows = np.asarray([body_to_idx[int(x)] for x in post_body[keep]], dtype=np.int64)
    cols = np.asarray([body_to_idx[int(x)] for x in pre_body[keep]], dtype=np.int64)
    vals = values[keep].astype(np.float32, copy=False)
    coo = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([rows, cols])),
        torch.from_numpy(vals),
        size=(len(node_body_ids), len(node_body_ids)),
    ).coalesce()
    weight_csr = coo.to_sparse_csr()

    motor_rows = ann.loc[motor_mask].copy()
    motor_rows["label"] = motor_rows.apply(text, axis=1)
    motor_rows["fly_joint"] = motor_rows["label"].map(fly_joint)
    motor_rows["polarity"] = motor_rows["label"].map(motor_polarity)
    motor_rows["leg"] = (motor_rows["somaNeuromere"] + motor_rows["somaSide"]).map({name: i for i, name in enumerate(HEX_LEGS)})
    usable = motor_rows["fly_joint"].ge(0) & motor_rows["polarity"].ne(0) & motor_rows["leg"].notna()
    motor_rows = motor_rows.loc[usable].sort_values(["leg", "fly_joint", "bodyId"])
    motor_out = motor_rows["bodyId"].map(body_to_idx).to_numpy(dtype=np.int64)
    motor_leg = motor_rows["leg"].to_numpy(dtype=np.int64)
    motor_joint = motor_rows["fly_joint"].to_numpy(dtype=np.int64)
    motor_sign = motor_rows["polarity"].to_numpy(dtype=np.float32)
    motor_body_ids = motor_rows["bodyId"].to_numpy(dtype=np.int64)
    motor_labels = motor_rows["label"].to_numpy(dtype="U96")
    coordination_rows = ann.loc[coordination_mask, ["bodyId", "type", "somaNeuromere", "somaSide"]].sort_values("bodyId")
    coordination_body_ids = coordination_rows["bodyId"].to_numpy(dtype=np.int64)
    coordination_out = compact_members(coordination_ids, body_to_idx)
    sensory_rows = ann.loc[sensory_mask, ["bodyId", "rootSide", "entryNerve", "subclass", "type"]].sort_values("bodyId")
    sensory_body_ids = sensory_rows["bodyId"].to_numpy(dtype=np.int64) if a.include_sensory_feedback else np.asarray([], dtype=np.int64)
    sensory_legs = sensory_rows["entryNerve"].map({"ProLN": "T1", "MesoLN": "T2", "MetaLN": "T3"}).to_numpy(dtype="U3") if a.include_sensory_feedback else np.asarray([], dtype="U3")
    sensory_sides = sensory_rows["rootSide"].to_numpy(dtype="U1") if a.include_sensory_feedback else np.asarray([], dtype="U1")
    sensory_subclasses = sensory_rows["subclass"].to_numpy(dtype="U32") if a.include_sensory_feedback else np.asarray([], dtype="U32")
    pool_sizes = np.zeros((len(HEX_LEGS), len(FLY_JOINTS)), dtype=np.int64)
    for leg, joint in zip(motor_leg, motor_joint):
        pool_sizes[leg, joint] += 1

    def compact_side(label: str, side: str) -> np.ndarray:
        mask = traced & ann["type"].eq(label) & ann["somaSide"].eq(side)
        return np.asarray(sorted(body_to_idx[int(x)] for x in ann.loc[mask, "bodyId"] if int(x) in body_to_idx), dtype=np.int64)

    out = {
        "n": np.asarray(len(node_body_ids), dtype=np.int64),
        "node_body_ids": node_body_ids,
        "descending_in": compact_members(descending_ids, body_to_idx),
        "dng100_left": compact_side("DNg100", "L"),
        "dng100_right": compact_side("DNg100", "R"),
        "dnb08_left": compact_side("DNb08", "L"),
        "dnb08_right": compact_side("DNb08", "R"),
        "premotor": compact_members(premotor_ids, body_to_idx),
        "motor_out": motor_out,
        "motor_body_ids": motor_body_ids,
        "motor_leg": motor_leg,
        "motor_joint": motor_joint,
        "motor_sign": motor_sign,
        "motor_labels": motor_labels,
        "motor_pool_sizes": pool_sizes,
        "leg_names": np.asarray(HEX_LEGS, dtype="U3"),
        "joint_names": np.asarray(FLY_JOINTS, dtype="U10"),
        "rhythm_core": compact_members(rhythm_core, body_to_idx),
        "rhythm_core_labels": np.asarray(core_labels, dtype="U12"),
        "sensory_feedback": compact_members(sensory_ids, body_to_idx),
        "sensory_feedback_body_ids": sensory_body_ids,
        "sensory_feedback_legs": sensory_legs,
        "sensory_feedback_sides": sensory_sides,
        "sensory_feedback_subclasses": sensory_subclasses,
        "sensory_feedback_rule": np.asarray(
            "traced vnc_sensory mechanosensory_proprioceptive; subclass in campaniform sensilla/hair plate/chordotonal organ/leg; entryNerve ProLN/MesoLN/MetaLN; rootSide L/R"
        ),
        "published_coordination": coordination_out,
        "published_coordination_body_ids": coordination_body_ids,
        "published_coordination_types": coordination_rows["type"].to_numpy(dtype="U16"),
        "published_coordination_neuromeres": coordination_rows["somaNeuromere"].to_numpy(dtype="U3"),
        "published_coordination_sides": coordination_rows["somaSide"].to_numpy(dtype="U2"),
        "published_coordination_type_labels": np.asarray(sorted(PUBLISHED_COORDINATION_TYPES), dtype="U16"),
        "published_coordination_types_present": np.asarray(present_coordination_types, dtype="U16"),
        "published_coordination_types_missing": np.asarray(missing_coordination_types, dtype="U16"),
        "published_coordination_rule": np.asarray(
            "traced vnc_intrinsic; type in eLife 96084 Supplementary file 8; soma neuromere in requested scope"
        ),
        "published_coordination_source": np.asarray(PUBLISHED_COORDINATION_SOURCE),
        "induced_edge_count": np.asarray(induced_edges, dtype=np.int64),
        "floored_edge_count": np.asarray(int(floor_keep.sum()), dtype=np.int64),
        "signed_edge_count": np.asarray(int(keep.sum()), dtype=np.int64),
        "synapse_floor": np.asarray(a.synapse_floor, dtype=np.int64),
        "nt_sign_column": np.asarray("predicted_nt"),
        "premotor_scope": np.asarray(a.premotor_scope),
        "include_sensory_feedback": np.asarray(a.include_sensory_feedback),
        "neuromeres": np.asarray(neuromeres, dtype="U3"),
    }
    out_path = a.out or (a.data / "vnc_io_index.npz")
    weight_path = a.weights_out or out_path.with_name("vnc_weight_csr.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with weight_path.open("wb") as f:
        pickle.dump(weight_csr, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez(out_path, **out)

    vals_np = weight_csr.values().numpy()
    log(f"induced edges: {induced_edges}; >=floor: {int(floor_keep.sum())}; signed nonzero edges: {len(vals_np)}")
    log(f"motor outputs: {len(motor_out)}; pool sizes T1L/T1R: {pool_sizes[:2].tolist()}")
    log(f"signed weight range: [{vals_np.min():.0f}, {vals_np.max():.0f}]")
    log(f"saved {weight_path} and {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
