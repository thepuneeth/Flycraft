"""Path 3 paired real/shuffled/random timing controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from analyze_control_study import bootstrap_ci, exact_sign_flip, holm_adjust, paired_bootstrap_ci, wilson_interval  # noqa: E402
from run_stepwalk import StepWalkContext, run_episode  # noqa: E402


CONDITIONS = ("real", "shuffled", "random")
N = 20
SEED = 20260829
EVALUATION_SEEDS = (20260829, 20260830, 20260831, 20260832)
BOOTSTRAP_REPS = 100_000
BOOTSTRAP_SEED = 20260829


def output_prefix(output: Path) -> str:
    return output.stem[:-len("_controls")] if output.stem.endswith("_controls") else output.stem


def condition_metrics_path(output: Path, condition: str) -> Path:
    return output.with_name(f"{output_prefix(output)}_metrics_{condition}.json")


def condition_video_path(output: Path, condition: str) -> Path:
    suffix = "multiview.mp4" if condition == "real" else f"multiview_{condition}.mp4"
    return output.with_name(f"{output_prefix(output)}_{suffix}")


def episode_specs(n: int = N, seed: int = SEED) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    result = []
    for episode in range(n):
        yaw = float(rng.uniform(-np.pi, np.pi))
        distance = float(rng.uniform(1.0, 1.8))
        angle = float(rng.uniform(-np.pi, np.pi))
        result.append({
            "episode": episode,
            "initial_yaw_rad": yaw,
            "prey_x_m": distance * np.cos(angle),
            "prey_y_m": distance * np.sin(angle),
            "episode_seed": int(seed + episode),
        })
    return result


def summarize(group: list[dict], rng: np.random.Generator, reps: int) -> dict:
    metrics = ("forward_distance_m", "path_m", "displacement_m", "tripod_coherence_scalar", "min_height_m", "mean_feet_planted", "mean_commanded_stance_legs")
    out = {
        "n": len(group),
        "caught_count": int(sum(row["caught"] for row in group)),
        "caught_rate": float(np.mean([row["caught"] for row in group])),
        "caught_wilson_95ci": wilson_interval(int(sum(row["caught"] for row in group)), len(group)),
        "fell_count": int(sum(row["fell"] for row in group)),
        "fell_rate": float(np.mean([row["fell"] for row in group])),
        "fell_wilson_95ci": wilson_interval(int(sum(row["fell"] for row in group)), len(group)),
        "upright_count": int(sum(row["upright"] for row in group)),
        "metrics": {},
    }
    for metric in metrics:
        values = np.asarray([row[metric] for row in group], dtype=np.float64)
        low, high = bootstrap_ci(values, rng, reps)
        out["metrics"][metric] = {
            "mean": float(values.mean()),
            "ci_low": low,
            "ci_high": high,
            "ci": "95% percentile bootstrap",
        }
    return out


def paired_comparison(left: list[dict], right: list[dict], name: str, rng: np.random.Generator, reps: int) -> dict:
    diff_forward = np.asarray([a["forward_distance_m"] - b["forward_distance_m"] for a, b in zip(left, right)], dtype=np.float64)
    diff_path = np.asarray([a["path_m"] - b["path_m"] for a, b in zip(left, right)], dtype=np.float64)
    low, high = paired_bootstrap_ci(diff_forward, rng, reps)
    path_low, path_high = paired_bootstrap_ci(diff_path, rng, reps)
    return {
        "comparison": name,
        "n_pairs": len(diff_forward),
        "forward_difference_m": {
            "mean": float(diff_forward.mean()),
            "ci_low": low,
            "ci_high": high,
            "p_exact_two_sided_paired_sign_flip": exact_sign_flip(diff_forward),
        },
        "path_difference_m": {
            "mean": float(diff_path.mean()),
            "ci_low": path_low,
            "ci_high": path_high,
            "p_exact_two_sided_paired_sign_flip": exact_sign_flip(diff_path),
        },
    }


def make_figure(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("real", "shuffled", "random")
    colors = ("#0072B2", "#D55E00", "#009E73")
    x = np.arange(len(labels))
    forward = [summary[name]["metrics"]["forward_distance_m"]["mean"] for name in labels]
    forward_low = [summary[name]["metrics"]["forward_distance_m"]["ci_low"] for name in labels]
    forward_high = [summary[name]["metrics"]["forward_distance_m"]["ci_high"] for name in labels]
    coherence = [summary[name]["metrics"]["tripod_coherence_scalar"]["mean"] for name in labels]
    coherence_low = [summary[name]["metrics"]["tripod_coherence_scalar"]["ci_low"] for name in labels]
    coherence_high = [summary[name]["metrics"]["tripod_coherence_scalar"]["ci_high"] for name in labels]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for axis, values, lows, highs, title, ylabel in (
        (axes[0], forward, forward_low, forward_high, "Forward distance", "m"),
        (axes[1], coherence, coherence_low, coherence_high, "Realized tripod scalar", "within − cross"),
    ):
        axis.bar(x, values, color=colors, edgecolor="black")
        axis.errorbar(x, values, yerr=[np.asarray(values) - lows, np.asarray(highs) - values], fmt="none", ecolor="black", capsize=4)
        axis.set(xticks=x, xticklabels=labels, title=title, ylabel=ylabel)
        axis.axhline(0.0, color="black", linewidth=0.8)
    fig.suptitle("Path 3 fixed gait engine: timing-source controls")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=HERE.parent.parent / "data")
    parser.add_argument("--vnc_data", type=Path, default=RESULTS)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument("--secs", type=float, default=8.0)
    parser.add_argument("--bootstrap_reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--timing_reference", type=Path, default=RESULTS / "step_decode_p1.json")
    parser.add_argument("--out", type=Path, default=RESULTS / "stepwalk_controls")
    args = parser.parse_args()
    if args.n != N:
        raise ValueError(f"Path 3 paired protocol is fixed at n={N}")
    if args.bootstrap_reps < 10_000:
        raise ValueError("use at least 10,000 bootstrap replicates")
    reference = json.loads(args.timing_reference.read_text(encoding="utf-8"))
    specs = episode_specs(args.n, SEED)
    rows = []
    for condition in CONDITIONS:
        context = StepWalkContext(condition, args.vnc_data, args.data)
        for index, spec in enumerate(specs):
            video = None
            if index == 0:
                video = condition_video_path(args.out, condition)
            result = run_episode(
                condition,
                int(spec["episode_seed"]),
                context=context,
                secs=args.secs,
                prey_offset=(float(spec["prey_x_m"]), float(spec["prey_y_m"])),
                initial_yaw=float(spec["initial_yaw_rad"]),
                timing_reference=reference,
                video=video,
            )
            rows.append({**spec, **result})
            print(f"{condition} {index + 1}/{args.n} forward={result['forward_distance_m']:+.4f} path={result['path_m']:.4f} fell={result['fell']} coherence={result['tripod_coherence_scalar']:+.4f}", flush=True)

    groups = {condition: sorted([row for row in rows if row["condition"] == condition], key=lambda row: row["episode"]) for condition in CONDITIONS}
    if any(len(group) != N for group in groups.values()):
        raise AssertionError("condition row count mismatch")
    keys = ("episode", "initial_yaw_rad", "prey_x_m", "prey_y_m", "episode_seed")
    reference_keys = [[group["episode"], group["initial_yaw_rad"], group["prey_x_m"], group["prey_y_m"], group["episode_seed"]] for group in groups["real"]]
    for condition in CONDITIONS[1:]:
        actual_keys = [[row[key] for key in keys] for row in groups[condition]]
        if not np.allclose(np.asarray(actual_keys, dtype=float), np.asarray(reference_keys, dtype=float)):
            raise AssertionError(f"episode starts not paired for {condition}")

    rng = np.random.default_rng(args.bootstrap_seed)
    summary = {condition: summarize(groups[condition], rng, args.bootstrap_reps) for condition in CONDITIONS}
    comparisons = [
        paired_comparison(groups["real"], groups["shuffled"], "real_minus_shuffled", rng, args.bootstrap_reps),
        paired_comparison(groups["real"], groups["random"], "real_minus_random", rng, args.bootstrap_reps),
    ]
    path_p = [row["path_difference_m"]["p_exact_two_sided_paired_sign_flip"] for row in comparisons]
    path_holm = holm_adjust(path_p)
    for row, adjusted in zip(comparisons, path_holm):
        row["path_difference_m"]["p_holm_across_two_path_tests"] = adjusted

    output = args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_json = {
        "protocol": {
            "n_per_condition": N,
            "secs": args.secs,
            "paired_episode_seed": SEED,
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "timing_reference": str(args.timing_reference),
            "timing_conditions": list(CONDITIONS),
            "same_engine_decode_steering": True,
            "only_condition_difference": "timing source",
            "random_process": "independent alternating-renewal swing/stance durations; mean frequency and swing duty matched per leg to P1 real decode; random initial phase",
            "learning": False,
        },
        "summary": summary,
        "comparisons": comparisons,
        "bootstrap": {"reps": args.bootstrap_reps, "seed": args.bootstrap_seed, "ci": "95% percentile bootstrap"},
        "paired_test": {"method": "exact two-sided paired sign-flip randomization", "primary_endpoint": "path_m", "multiplicity": "Holm adjustment across two predeclared Path 3 comparisons"},
        "binary_rate_ci": "95% Wilson interval",
        "condition_metric_files": {condition: str(condition_metrics_path(output, condition)) for condition in CONDITIONS},
        "video_files": {condition: str(condition_video_path(output, condition)) for condition in CONDITIONS},
    }
    output.with_suffix(".json").write_text(json.dumps(metric_json, indent=2) + "\n", encoding="utf-8")
    for condition in CONDITIONS:
        condition_output = condition_metrics_path(output, condition)
        condition_output.write_text(json.dumps({"condition": condition, "protocol": metric_json["protocol"], "summary": summary[condition], "episodes": groups[condition]}, indent=2) + "\n", encoding="utf-8")

    columns = ["condition", "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m", "forward_distance_m", "path_m", "displacement_m", "tripod_coherence_scalar", "min_height_m", "mean_ride_height_m", "mean_feet_planted", "mean_commanded_stance_legs", "fell", "upright", "caught"]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})
    make_figure(summary, output.with_suffix(".png"))
    print(json.dumps({"summary": summary, "comparisons": comparisons}, sort_keys=True))
    print(f"json -> {output.with_suffix('.json')}")
    print(f"csv -> {output.with_suffix('.csv')}")
    print(f"figure -> {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
