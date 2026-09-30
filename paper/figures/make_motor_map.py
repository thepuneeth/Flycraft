"""Render the actual fixed map; run from any directory. No fitted parameters."""
from pathlib import Path
import csv
import hashlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eon-fly-brain/code"))
from mn_to_joint import translate

source = ROOT / "eon-fly-brain/results/path2/all_legs_io_index.npz"
with np.load(source, allow_pickle=False) as archive:
    io = dict(archive)
n = len(io["motor_out"])
gains = (-0.55, 0.65, 0.65)
matrix = np.column_stack([
    translate(row, io, stance=np.zeros(18), gains=gains) for row in np.eye(n)
])

# Check the full affine map, not only the displayed ordering.
rng = np.random.default_rng(2026)
for activity in rng.random((20, n)):
    stance = rng.uniform(-0.4, 0.4, 18)
    np.testing.assert_allclose(
        stance + matrix @ activity,
        translate(activity, io, stance=stance, gains=gains), atol=1e-7,
    )
assert matrix.shape == (18, 260)
for leg in range(6):
    assert not np.any(matrix[3 * leg:3 * leg + 3, io["motor_leg"] != leg])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(7.1, 2.8))
fig.subplots_adjust(left=.085, right=.985, bottom=.30, top=.9,
                    hspace=.75, wspace=.25)
limit = float(np.abs(matrix).max())
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("white")
for leg, ax in zip((0, 2, 4, 1, 3, 5), axes.flat):
    indices = np.flatnonzero(io["motor_leg"] == leg)
    indices = indices[np.lexsort((io["motor_sign"][indices], io["motor_joint"][indices]))]
    values = matrix[3 * leg:3 * leg + 3, indices]
    im = ax.imshow(np.ma.masked_equal(values, 0), cmap=cmap, vmin=-limit, vmax=limit,
                   aspect="auto", interpolation="nearest")
    ax.set_title(f"{io['leg_names'][leg]} · {len(indices)} channels", fontsize=8, pad=5)
    ax.set_yticks(range(3), ["Coxa", "Femur", "Tibia"] if leg in (0, 1) else ["", "", ""])
    ax.set_xticks([])
    ax.tick_params(length=0, pad=3)
    for boundary in np.flatnonzero(np.diff(io["motor_joint"][indices])):
        ax.axvline(boundary + .5, color="#777777", linewidth=.35)
    for spine in ax.spines.values():
        spine.set_linewidth(.4)
        spine.set_color("#888888")
fig.text(.53, .22, "Motor channels, grouped by fly joint", ha="center", fontsize=8)
cax = fig.add_axes([.36, .115, .34, .027])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit])
cb.ax.set_xticklabels([f"−{limit:.2f}", "0", f"+{limit:.2f}"])
cb.ax.tick_params(length=2, pad=2)
fig.text(.53, .015, "Target-angle change per unit motor rate (rad)", ha="center", fontsize=8)
fig.savefig(HERE / "fixed_motor_map.pdf")
fig.savefig(HERE / "fixed_motor_map.png", dpi=400)
plt.close(fig)

with (HERE / "fixed_motor_map.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["motor_index", "body_id", "leg", "fly_joint", "pool_sign"] +
                    [f"{leg}_{joint}" for leg in io["leg_names"] for joint in ("coxa", "femur", "tibia")])
    for index in range(n):
        writer.writerow([index, io["motor_body_ids"][index], io["leg_names"][io["motor_leg"][index]],
                         io["joint_names"][io["motor_joint"][index]], io["motor_sign"][index],
                         *matrix[:, index]])
print(f"Verified {matrix.shape} affine map on 20 random inputs; gains={gains}")
print(f"Source SHA256: {hashlib.sha256(source.read_bytes()).hexdigest()}")
