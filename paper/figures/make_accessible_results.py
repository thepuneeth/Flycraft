"""Rebuild the two student-paper figures with the project's local venv.

Run: eon-fly-brain/.venv/bin/python paper/figures/make_accessible_results.py

Exact sources (relative to repository root):
  eon-fly-brain/results/path2/h1_amplitude_control_strict.json:
    primary_forward.{real_value,random_seed_values,random_mean,
    real_minus_random_mean}; checked against conditions and episode rows.
  eon-fly-brain/results/path2/e1_turning.json:
    primary.{real_value,random_seed_values,random_mean,real_minus_random_mean};
    checked against turn_response_records and paired episode yaw changes.
  eon-fly-brain/results/path2/bodyfix_vnc_phase_probe_e0.json:
    within_pairs_in_phase/within_pair_count; cross_pairs_antiphase/cross_pair_count;
    checked against phase_difference_matrix_cycles and pair arrays using
    protocol.phase_tolerance_cycles (0.15). Groups match probe_vnc_phase.py.

Forward and turning each show one measured-graph mean and all 12 independent
weight-shuffle seed means (20260830--20260841). Every mean uses the same 20
starts, 8 s per rollout. Turning is half [yaw(+0.5) - yaw(-0.5)] per paired
start, then averaged. Controls preserve sparse structure and signed weight
multiset; per-channel joint-target temporal SD is matched to measured wiring
(at the same turn for E1). Raw and rewired controls are not plotted.
The black tick is the unweighted mean of 12 control means. Horizontal offsets
only separate seed dots; no intervals or pooled episode/seed uncertainty.

The leg outline is an expected-pattern schematic, not measured motion or
anatomical scale. A = front L, middle R, rear L; B = front R, middle L, rear R.
Bars show fractions of unique phase pairs passing, not uncertainty or independent
replicates. Required: every within pair near 0 and every cross pair near 0.5
cycles, each within 0.15 cycles. Probe: seed 1, 1.5--4 Hz bandpass, final half
of 1500 samples at 0.02 s/sample. E0-ABSENT; neither figure establishes walking.

Exports: vector PDF with embedded TrueType text; opaque 600-dpi PNG; fonts >=8 pt.
General manuscript dimensions, not certification against a journal's rules.
"""

import json
from itertools import combinations, product
from math import isclose
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

OUT = Path(__file__).resolve().parent
SOURCES = OUT.parents[1] / "eon-fly-brain/results/path2"
BLUE, ORANGE, INK = "#0072B2", "#B44D00", "#222222"
GROUPS = ((0, 3, 4), (1, 2, 5))


def check_sources(forward, turning, phase):
    """One runnable check: recompute every displayed source value before export."""
    for data, key, expected, digits in (
        (forward, "primary_forward", (0.22407021664125976, -0.011718073934225191, 0.236), 3),
        (turning, "primary", (7.440824229455197, 0.525278050753374, 6.92), 2),
    ):
        p = data[key]
        assert data["protocol"]["secs"] == 8
        assert data["protocol"]["n_episodes"] == 20
        assert data["protocol"]["shared_episode_starts"] is True
        records = data["conditions"] if key == "primary_forward" else data["turn_response_records"]
        selected = [r for r in records if r["substrate"] == "real" or
                    (r["substrate"] == "random" and r["amplitude_mode"].startswith("per_channel_joint_target_sd_matched"))]
        assert len(selected) == 13
        assert [r["seed"] for r in selected[1:]] == list(range(20260830, 20260842))
        starts = None
        for record, value in zip(selected, [p["real_value"], *p["random_seed_values"]], strict=True):
            if key == "primary_forward":
                rows = [r for r in data["rows"] if r["condition"] == record["condition"]]
                values = [r["forward_distance_m"] for r in rows]
                reported = record["summary"]["metrics"]["forward_distance_m"]
                assert all(r["horizon_s"] == 8 for r in rows)
            else:
                rows = [r for r in data["rows"] if r["condition"] == record["positive_turn_condition"]]
                negative = {r["episode"]: r for r in data["rows"] if r["condition"] == record["negative_turn_condition"]}
                assert len(negative) == 20
                assert {r["episode"] for r in rows} == set(negative)
                values = [(r["yaw_change_deg"] - negative[r["episode"]]["yaw_change_deg"]) / 2 for r in rows]
                assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(values, record["paired_episode_values_deg"], strict=True))
                assert all(r["initial_yaw_rad"] == negative[r["episode"]]["initial_yaw_rad"] for r in rows)
                reported = record["turn_response_deg"]
            current = sorted((r["episode"], r["initial_yaw_rad"]) for r in rows)
            starts = current if starts is None else starts
            assert current == starts
            assert len(values) == reported["n"] == 20
            assert isclose(mean(values), value, abs_tol=1e-12)
            assert isclose(reported["mean"], value, abs_tol=1e-12)
        assert len(p["random_seed_values"]) == 12
        assert isclose(mean(p["random_seed_values"]), p["random_mean"], abs_tol=1e-12)
        assert isclose(p["real_value"], expected[0], abs_tol=1e-12)
        assert isclose(p["random_mean"], expected[1], abs_tol=1e-12)
        delta = p["real_value"] - p["random_mean"]
        assert isclose(delta, p["real_minus_random_mean"], abs_tol=1e-12)
        assert round(delta, digits) == expected[2]
    assert turning["protocol"]["turns"] == [-0.5, 0.5]
    assert phase["leg_names"] == ["T1L", "T1R", "T2L", "T2R", "T3L", "T3R"]
    assert phase["protocol"]["phase_tolerance_cycles"] == 0.15
    pairs = ([p for g in GROUPS for p in combinations(g, 2)], list(product(*GROUPS)))
    for prefix, pair_set, total, target, count_key in (
        ("within", pairs[0], 6, 0, "within_pairs_in_phase"),
        ("cross", pairs[1], 9, 0.5, "cross_pairs_antiphase"),
    ):
        values = [phase["phase_difference_matrix_cycles"][a][b] for a, b in pair_set]
        assert values == phase[f"{prefix}_pair_phase_cycles"]
        assert len(values) == phase[f"{prefix}_pair_count"] == total
        passed = sum(abs(abs((v + 0.5) % 1 - 0.5) - target) <= 0.15 for v in values)
        assert passed == phase[count_key] == 3
    assert phase["verdict"] == "E0-ABSENT"
    assert (phase["protocol"]["seed"], phase["protocol"]["steps"], phase["protocol"]["sample_dt_s"], phase["protocol"]["burn_fraction"]) == (1, 1500, 0.02, 0.5)
    assert phase["protocol"]["bandpass_hz"] == [1.5, 4.0]


def save(fig, name, note):
    fig.savefig(OUT / f"{name}.pdf", metadata={"Title": name, "Subject": note}, facecolor="white")
    fig.savefig(OUT / f"{name}.png", dpi=600, metadata={"Description": note}, facecolor="white")
    plt.close(fig)


def forward_turning(forward, turning):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.25))
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.32, top=0.80, wspace=0.47)
    for ax, p, title, label, digits, unit, limits, ticks in (
        (axes[0], forward["primary_forward"], "A  Forward", "Forward displacement\n(m; 8 s)", 3, "m", (-0.21, 0.28), [-0.2, 0, 0.2]),
        (axes[1], turning["primary"], "B  Turning", "Turning response\n(degrees; 8 s)", 2, "°", (-6.5, 9), [-5, 0, 5]),
    ):
        ax.axhline(0, color="#BBBBBB", lw=0.7, zorder=0)
        ax.scatter([0], [p["real_value"]], marker="D", s=38, color=BLUE, zorder=3)
        ax.scatter([1 + (i - 5.5) * 0.05 for i in range(12)], p["random_seed_values"],
                   marker="o", s=19, color=ORANGE, edgecolors="white", linewidths=0.35, zorder=3)
        ax.plot([0.65, 1.35], [p["random_mean"]] * 2, color=INK, lw=1.3, zorder=2)
        ax.set(xlim=(-0.45, 1.48), ylim=limits, ylabel=label, yticks=ticks)
        ax.set_xticks([0, 1], ["Measured wiring", "Shuffled weights"])
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9, pad=21)
        delta = p["real_minus_random_mean"]
        ax.text(0.5, 1.04, f"+{delta:.{digits}f}{' ' if unit == 'm' else ''}{unit} vs control mean",
                transform=ax.transAxes, ha="center", fontsize=8)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.text(0.5, 0.105, "Each point: mean of 20 starts. Black line: shuffled-control mean.", ha="center", fontsize=8)
    fig.text(0.5, 0.035, "Joint-target variation matched; turning uses opposite inputs.", ha="center", fontsize=8)
    save(fig, "forward_turning", __doc__)


def leg_coordination(phase):
    fig = plt.figure(figsize=(3.4, 2.8))
    fig.text(0.05, 0.955, "A  Expected pattern (schematic)", fontsize=9, weight="bold")
    fig.text(0.05, 0.895, "Same group moves together; groups alternate", fontsize=8)
    ax = fig.add_axes([0.04, 0.555, 0.92, 0.315])
    ax.set(xlim=(-2.2, 2.2), ylim=(-1.25, 1.4), aspect="equal")
    ax.axis("off")
    ax.add_patch(Ellipse((0, -0.05), 0.72, 1.95, fill=False, edgecolor=INK, lw=1))
    ax.add_patch(Ellipse((0, 1.04), 0.55, 0.48, facecolor="white", edgecolor=INK, lw=1))
    for i, (name, y) in enumerate((("front", 0.75), ("middle", 0), ("rear", -0.75))):
        for side, side_label, offset in ((-1, "L", 0), (1, "R", 1)):
            group_a = 2 * i + offset in GROUPS[0]
            color, marker = (BLUE, "o") if group_a else (ORANGE, "^")
            ax.plot([side * 0.32, side * 0.68, side * 1.05], [y * 0.7, y, y], color=color, lw=1.2)
            ax.plot(side * 1.05, y, marker=marker, color=color, markersize=5)
            ax.text(side * 1.23, y, f"{name} {side_label}", va="center", ha="right" if side < 0 else "left", fontsize=8)
    fig.legend(handles=[Line2D([], [], marker=m, color=c, ls="", label=f"Group {g}")
                        for g, c, m in (("A", BLUE, "o"), ("B", ORANGE, "^"))],
               loc="center", bbox_to_anchor=(0.5, 0.52), ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.6)
    fig.text(0.05, 0.437, "B  Measured timing: test failed", fontsize=9, weight="bold")
    bars = fig.add_axes([0.37, 0.175, 0.56, 0.21])
    bars.barh([1, 0], [1, 1], height=0.48, color="white", edgecolor="#777777", linewidth=0.7)
    for y, passed, total in ((1, phase["within_pairs_in_phase"], phase["within_pair_count"]),
                             (0, phase["cross_pairs_antiphase"], phase["cross_pair_count"])):
        bars.barh(y, passed / total, height=0.48, color="#555555")
        bars.text(passed / total + 0.04, y, f"{passed}/{total}", va="center", fontsize=8)
    bars.axvline(1, color=INK, ls="--", lw=0.9)
    bars.set(xlim=(0, 1.04), ylim=(-0.65, 1.65))
    bars.set_yticks([1, 0], ["Within group", "Across groups"])
    bars.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    bars.tick_params(length=0, pad=4)
    bars.spines[:].set_visible(False)
    fig.text(0.05, 0.074, "Pairs passing; every pair required.", fontsize=8)
    fig.text(0.05, 0.022, "Motor signals only; footfalls not measured.", fontsize=8)
    save(fig, "leg_coordination", __doc__)


def main():
    forward, turning, phase = [json.loads((SOURCES / f"{name}.json").read_text()) for name in
                               ("h1_amplitude_control_strict", "e1_turning", "bodyfix_vnc_phase_probe_e0")]
    check_sources(forward, turning, phase)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.7,
                         "text.color": INK, "axes.labelcolor": INK}):
        forward_turning(forward, turning)
        leg_coordination(phase)
    print("Source checks passed; wrote forward_turning.pdf/png and leg_coordination.pdf/png.")
    print(__doc__)


if __name__ == "__main__":
    main()
