"""Add confidence intervals and paired tests to a frozen control study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
CONDITIONS = ("vnc_open_loop", "vnc_brain", "shuffled_vnc", "random_descending", "path1_hand_cpg")
METRICS = (
    "path_m",
    "forward_displacement_m",
    "displacement_m",
    "min_distance_m",
    "motor_temporal_std_mean",
    "mean_abs_az_deg",
)
PAIRS = (
    ("real_vnc_minus_shuffled", "vnc_open_loop", "shuffled_vnc"),
    ("brain_minus_random", "vnc_brain", "random_descending"),
    ("open_loop_minus_path1", "vnc_open_loop", "path1_hand_cpg"),
)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exact_sign_flip(diff: np.ndarray) -> float:
    """Exact two-sided paired sign-flip p-value under conditional independent, uniform sign symmetry."""
    n = len(diff)
    if n > 22:
        raise ValueError("exact sign-flip test is capped at n=22")
    masks = np.arange(1 << n, dtype=np.uint32)[:, None]
    bits = (masks >> np.arange(n, dtype=np.uint32)) & 1
    signs = bits.astype(np.int8) * 2 - 1
    null_means = (signs @ diff) / n
    observed = abs(float(diff.mean()))
    return float(np.mean(np.abs(null_means) >= observed - 1e-12))


def paired_bootstrap_ci(diff: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    indices = rng.integers(0, len(diff), size=(reps, len(diff)))
    means = diff[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    z = 1.96
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def paired_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if set(frame["condition"]) != set(CONDITIONS):
        raise ValueError(f"expected conditions {CONDITIONS}, got {sorted(frame['condition'].unique())}")
    groups = {}
    key_columns = ("episode", "initial_yaw_rad", "prey_x_m", "prey_y_m", "episode_seed")
    reference = None
    for condition in CONDITIONS:
        group = frame[frame.condition.eq(condition)].sort_values("episode").reset_index(drop=True)
        if len(group) != 20:
            raise ValueError(f"{condition} has {len(group)} episodes; expected 20")
        if not np.all(group.returncode.to_numpy() == 0):
            raise ValueError(f"{condition} contains nonzero return codes")
        if reference is None:
            reference = group[list(key_columns)]
        elif not np.allclose(group[list(key_columns)].to_numpy(dtype=float), reference.to_numpy(dtype=float)):
            raise ValueError(f"episode starts are not paired for {condition}")
        groups[condition] = group
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=RESULTS / "bodyfix_vnc_control_study_3s.csv")
    ap.add_argument("--out", type=Path, default=RESULTS / "bodyfix_vnc_control_study_3s_stats")
    ap.add_argument("--bootstrap_reps", type=int, default=100_000)
    ap.add_argument("--bootstrap_seed", type=int, default=20260829)
    a = ap.parse_args()
    if a.bootstrap_reps < 10_000:
        raise ValueError("use at least 10,000 bootstrap replicates")
    frame = pd.read_csv(a.input)
    groups = paired_frames(frame)
    rng = np.random.default_rng(a.bootstrap_seed)

    summary: dict[str, dict] = {}
    summary_rows = []
    for condition in CONDITIONS:
        group = groups[condition]
        condition_summary = {
            "n": len(group),
            "caught_count": int(group["caught"].sum()),
            "caught_rate": float(group["caught"].mean()),
            "caught_wilson_95ci": wilson_interval(int(group["caught"].sum()), len(group)),
            "fell_count": int(group["fell"].sum()),
            "fell_rate": float(group["fell"].mean()),
            "fell_wilson_95ci": wilson_interval(int(group["fell"].sum()), len(group)),
            "metrics": {},
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            if np.isnan(values).all():
                continue
            if not np.isfinite(values).all():
                raise ValueError(f"{condition}/{metric} contains partial non-finite values")
            ci_low, ci_high = bootstrap_ci(values, rng, a.bootstrap_reps)
            condition_summary["metrics"][metric] = {
                "mean": float(values.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci": "95% percentile bootstrap",
            }
            summary_rows.append({
                "condition": condition,
                "metric": metric,
                "n": len(values),
                "mean": float(values.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })
        summary[condition] = condition_summary

    comparisons = []
    primary_p = []
    for name, left_name, right_name in PAIRS:
        left = groups[left_name]
        right = groups[right_name]
        for metric in METRICS:
            diff = left[metric].to_numpy(dtype=float) - right[metric].to_numpy(dtype=float)
            if np.isnan(diff).all():
                continue
            if not np.isfinite(diff).all():
                raise ValueError(f"{name}/{metric} contains partial non-finite values")
            ci_low, ci_high = paired_bootstrap_ci(diff, rng, a.bootstrap_reps)
            pvalue = exact_sign_flip(diff)
            comparisons.append({
                "comparison": name,
                "left_condition": left_name,
                "right_condition": right_name,
                "metric": metric,
                "n_pairs": len(diff),
                "mean_difference_left_minus_right": float(diff.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_exact_two_sided_paired_sign_flip": pvalue,
            })
            if metric == "path_m":
                primary_p.append(pvalue)
    adjusted = holm_adjust(primary_p)
    for row in comparisons:
        if row["metric"] == "path_m":
            row["p_holm_across_three_primary_path_tests"] = adjusted[
                [name for name, _, _ in PAIRS].index(row["comparison"])
            ]

    output = a.out
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name(output.name + "_summary.csv")
    comparisons_path = output.with_name(output.name + "_comparisons.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(comparisons).to_csv(comparisons_path, index=False)
    manifest = {
        "input": str(a.input),
        "conditions": list(CONDITIONS),
        "n_per_condition": 20,
        "metrics": list(METRICS),
        "summary": summary,
        "comparisons": comparisons,
        "bootstrap": {
            "reps": a.bootstrap_reps,
            "seed": a.bootstrap_seed,
            "ci": "95% percentile bootstrap",
        },
        "paired_test": {
            "method": "exact two-sided paired sign-flip under conditional independent, uniform sign-symmetry assumption",
            "pairing": "same episode starts across conditions",
            "primary_endpoint": "path_m",
            "multiplicity": "Holm adjustment across three predeclared path comparisons",
        },
        "binary_rate_ci": "95% Wilson interval",
        "summary_csv": str(summary_path),
        "comparisons_csv": str(comparisons_path),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "primary_path_comparisons": [row for row in comparisons if row["metric"] == "path_m"]}, sort_keys=True))
    print(f"json -> {output.with_suffix('.json')}")
    print(f"summary csv -> {summary_path}")
    print(f"comparisons csv -> {comparisons_path}")


if __name__ == "__main__":
    main()
