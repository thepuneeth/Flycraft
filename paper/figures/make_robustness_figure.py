"""Build Figure 2 from the frozen E3 and H4 result records."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "../eon-fly-brain/results/path2"
OUTPUT = Path(__file__).with_name("robustness_rewiring.png")


def read(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    e3 = read("e3_body_generalization_amplitude_matched.json")
    h4 = read("h4_dose_response.json")
    body = e3["per_perturbation"]
    dose = h4["dose_curve"]

    blue = "#0072B2"
    orange = "#D55E00"
    purple = "#6F42C1"
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.5, 2.8), gridspec_kw={"wspace": 0.28}
    )

    x = np.arange(len(body))
    real = np.array([row["real_value"] for row in body])
    null = np.array([row["random_mean"] for row in body])
    low = np.array([row["random_distribution_ci_low"] for row in body])
    high = np.array([row["random_distribution_ci_high"] for row in body])
    left.errorbar(x, real, fmt="o", color=blue, markersize=4, label="measured wiring")
    left.errorbar(
        x,
        null,
        yerr=[null - low, high - null],
        fmt="s",
        color=orange,
        markersize=4,
        capsize=2,
        label="shuffled weights",
    )
    left.axhline(0, color="black", linewidth=0.7)
    left.axvline(2.5, color="#cccccc", linewidth=0.6)
    left.axvline(5.5, color="#cccccc", linewidth=0.6)
    left.set_title("A  Body-parameter changes", loc="left", fontsize=8.5, fontweight="bold", pad=12)
    left.text(
        0.0, 1.01, "Ranges: central 95% of 12 control means",
        transform=left.transAxes, fontsize=6.5, va="bottom"
    )
    left.set_ylabel("forward displacement (m; 8 s)", fontsize=7.5)
    left.set_xticks(x, ["0.85", "1", "1.15", "0.85", "1", "1.15", "0.7", "1", "1.3"], fontsize=6.5)
    left.tick_params(axis="y", labelsize=6.5)
    left.set_ylim(-0.23, 0.28)
    left.set_xlim(-0.5, 8.5)
    for center, label in ((1, "leg length"), (4, "body mass"), (7, "ground friction")):
        left.text(center, -0.19, label, ha="center", va="top", fontsize=6.5)
    left.set_xlabel("parameter value", fontsize=7.5, labelpad=10)

    x = np.array([row["rewiring_fraction"] for row in dose])
    y = np.array([row["mean_forward_distance_m"] for row in dose])
    low = np.array([row["forward_ci_low"] for row in dose])
    high = np.array([row["forward_ci_high"] for row in dose])
    right.plot(x[1:], y[1:], marker="s", markersize=4, color=purple, label="rewired connections")
    right.fill_between(x[1:], low[1:], high[1:], color=purple, alpha=0.14, linewidth=0)
    right.plot(x[0], y[0], marker="o", markersize=4, color=blue, linestyle="", label="measured wiring")
    right.axhline(0, color="black", linewidth=0.7)
    right.set_title("B  Rewiring dose", loc="left", fontsize=8.5, fontweight="bold", pad=12)
    right.text(0.0, 1.01, "Shading: 95% confidence interval across controls", transform=right.transAxes, fontsize=6.5, va="bottom")
    right.set_xlabel(r"edge-swap fraction $\alpha$", fontsize=7.5)
    right.set_ylabel("forward displacement (m; 8 s)", fontsize=7.5)
    right.tick_params(axis="both", labelsize=6.5)
    right.set_xticks(x, ["0", "0.1", "0.25", "0.5", "0.75", "1"])
    right.set_ylim(-0.08, 0.27)

    handles = [
        plt.Line2D([], [], marker="o", color=blue, linestyle="", markersize=4, label="measured wiring"),
        plt.Line2D([], [], marker="s", color=orange, linestyle="", markersize=4, label="shuffled weights"),
        plt.Line2D([], [], marker="s", color=purple, linestyle="", markersize=4, label="rewired connections"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.25, top=0.88, left=0.07, right=0.99)
    fig.savefig(OUTPUT, dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
