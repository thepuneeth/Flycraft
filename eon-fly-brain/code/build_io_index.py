"""Build the sensory-in / motor-out index map for the live Flycraft loop.

Connectome matrix rows/cols are ordered by 2025_Completeness_783.csv index.
We join the FlyWire annotations on root_id to tag each matrix index with
super_class + side, then dump the left/right visual-input and descending-output
index arrays the real-time stepper injects into / reads out of.
"""
import pandas as pd, numpy as np
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data"

comp = pd.read_csv(D / "2025_Completeness_783.csv", index_col=0)
comp_ids = comp.index.astype("int64").to_numpy()            # matrix index i -> root_id
pos = {rid: i for i, rid in enumerate(comp_ids)}            # root_id -> matrix index

ann = pd.read_csv(D / "flywire_annotations.tsv", sep="\t", low_memory=False)
ann = ann.dropna(subset=["root_id"])
ann["root_id"] = ann["root_id"].astype("int64")
ann = ann[ann["root_id"].isin(pos)]
ann["idx"] = ann["root_id"].map(pos)

def idx_for(super_classes, side):
    m = ann["super_class"].isin(super_classes) & (ann["side"] == side)
    return ann.loc[m, "idx"].to_numpy(dtype=np.int64)

visual = ["optic", "visual_projection"]

def dn_idx(cell_type, side):
    m = (ann["cell_type"] == cell_type) & (ann["side"] == side)
    return ann.loc[m, "idx"].to_numpy(dtype=np.int64)

out = dict(
    n=len(comp_ids),
    vis_left=idx_for(visual, "left"),
    vis_right=idx_for(visual, "right"),
    desc_left=idx_for(["descending"], "left"),
    desc_right=idx_for(["descending"], "right"),
    # identified command neurons: DNa02/DNa01 steer (ipsilateral turn), DNp09 walks fwd
    dna02_left=dn_idx("DNa02", "left"),  dna02_right=dn_idx("DNa02", "right"),
    dna01_left=dn_idx("DNa01", "left"),  dna01_right=dn_idx("DNa01", "right"),
    dnp09_left=dn_idx("DNp09", "left"),  dnp09_right=dn_idx("DNp09", "right"),
)
np.savez(D / "io_index.npz", **out)
for k, v in out.items():
    print(k, v if np.isscalar(v) else len(v))
print("saved ->", D / "io_index.npz")
