"""Build the fixed descending-command expansion map (4 latent commands -> DN inputs).

The embodiment interface must not learn one independent gain per descending
neuron; that lets the encoder program the core directly and makes any later
connectivity result an interface result. This script writes a FIXED,
anatomy-derived matrix that expands a 4-dimensional latent command into the
1325 descending inputs. It is declared once, hashed, and never trained.

Channels (rows), all derived from MANC annotations, none tuned:
  0 go      uniform drive to every descending input
  1 steer   +1 left soma, -1 right soma, 0 midline/unknown (bilateral see-saw)
  2 family  +1 DNg/DNge family, -1 DNp/DNpe family, 0 otherwise
  3 walkdn  +1 on the DNg100 and DNb08 classes already declared in the io index

Run:  build_dn_command_map.py [--io all_legs_io_index.npz] [--out dn_command_map.npz]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULTS = HERE.parent / "results" / "path2"
ANNOTATIONS = DATA / "manc" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

CHANNEL_NAMES = ("go", "steer", "family", "walkdn")
WALK_TYPES = ("DNg100", "DNb08")
LEFT_FAMILY = ("DNg", "DNge")
RIGHT_FAMILY = ("DNp", "DNpe")


def array_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def type_prefix(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.extract(r"^(DN[a-z]*)")[0].fillna("other")


def build(io_path: Path, annotations: Path = ANNOTATIONS) -> tuple[np.ndarray, dict]:
    io = np.load(io_path)
    descending = io["descending_in"].astype(np.int64)
    body_ids = io["node_body_ids"][descending]
    frame = pd.read_feather(annotations)
    frame = frame[frame.bodyId.isin(body_ids)].drop_duplicates("bodyId").set_index("bodyId")
    missing = int(len(body_ids) - frame.index.isin(body_ids).sum())
    if missing:
        raise ValueError(f"{missing} descending body ids absent from {annotations.name}")
    ordered = frame.loc[body_ids]

    side = ordered["somaSide"].fillna("U").astype(str).to_numpy()
    prefix = type_prefix(ordered["type"]).to_numpy()
    cell_type = ordered["type"].fillna("").astype(str).to_numpy()

    n = len(body_ids)
    matrix = np.zeros((len(CHANNEL_NAMES), n), dtype=np.float32)
    matrix[0, :] = 1.0
    matrix[1, side == "L"] = 1.0
    matrix[1, side == "R"] = -1.0
    matrix[2, np.isin(prefix, LEFT_FAMILY)] = 1.0
    matrix[2, np.isin(prefix, RIGHT_FAMILY)] = -1.0
    matrix[3, np.isin(cell_type, WALK_TYPES)] = 1.0

    for row, name in enumerate(CHANNEL_NAMES):
        if not np.any(matrix[row] != 0):
            raise ValueError(f"channel {name!r} is all zero")

    provenance = {
        "io_index": io_path.name,
        "annotations": annotations.name,
        "n_descending": n,
        "channels": list(CHANNEL_NAMES),
        "channel_support": {name: int(np.count_nonzero(matrix[row])) for row, name in enumerate(CHANNEL_NAMES)},
        "side_counts": {key: int((side == key).sum()) for key in sorted(set(side))},
        "walk_types": list(WALK_TYPES),
        "family_positive": list(LEFT_FAMILY),
        "family_negative": list(RIGHT_FAMILY),
        "trainable": False,
        "matrix_sha256": array_sha256(matrix),
        "descending_sha256": array_sha256(descending),
    }
    return matrix, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--io", type=Path, default=RESULTS / "all_legs_io_index.npz")
    parser.add_argument("--out", type=Path, default=RESULTS / "dn_command_map.npz")
    args = parser.parse_args()

    matrix, provenance = build(args.io)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        matrix=matrix,
        channels=np.asarray(provenance["channels"], dtype=object),
        provenance=np.asarray([repr(provenance)], dtype=object),
    )
    print(f"DN COMMAND MAP: shape={matrix.shape} sha256={provenance['matrix_sha256']}")
    for name, support in provenance["channel_support"].items():
        print(f"  {name:8s} support={support}")
    return 0


def self_check() -> dict:
    matrix, provenance = build(RESULTS / "all_legs_io_index.npz")
    assert matrix.shape == (4, provenance["n_descending"])
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    assert np.all(matrix[0] == 1.0)
    assert np.any(matrix[1] > 0) and np.any(matrix[1] < 0)
    assert np.linalg.matrix_rank(matrix) == 4, "command channels must be linearly independent"
    return provenance


if __name__ == "__main__":
    raise SystemExit(main())
