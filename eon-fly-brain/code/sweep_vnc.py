"""Characterize full-T1 MANC dynamics without fitting to body performance.

Runs constant descending drive over two grids:
  1. gW x drive at fixed tau
  2. gW x tau at fixed drive

Outputs a CSV of unsmoothed summary metrics and a two-by-two heatmap figure.
The model source artifacts remain untouched.
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from vnc_stepper import VNCStepper  # noqa: E402


DEFAULT_GW = (0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060)
DEFAULT_DRIVE = (25.0, 50.0, 100.0, 150.0, 250.0, 350.0, 500.0)
DEFAULT_TAU = (0.005, 0.010, 0.020, 0.040, 0.080)


def numbers(value: str) -> tuple[float, ...]:
    parsed = tuple(float(x) for x in value.split(",") if x.strip())
    if not parsed or any(x <= 0 for x in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return parsed


def balance(data: Path, weights_name: str) -> dict[str, float | int]:
    with (data / weights_name).open("rb") as f:
        W = pickle.load(f)
    values = W.values().cpu().numpy()
    positive = values[values > 0]
    negative = values[values < 0]
    pos_mass = float(positive.sum())
    neg_mass = float(-negative.sum())
    return {
        "n": int(W.shape[0]),
        "nnz": int(len(values)),
        "positive_edges": int(len(positive)),
        "negative_edges": int(len(negative)),
        "positive_mass": pos_mass,
        "negative_mass": neg_mass,
        "positive_to_negative_mass": pos_mass / neg_mass if neg_mass else float("inf"),
    }


def spectral_metrics(history: np.ndarray, dt_sample: float, burn_fraction: float) -> dict[str, float]:
    cut = int(len(history) * burn_fraction)
    tail = history[cut:]
    if len(tail) < 8:
        raise ValueError("need at least 8 post-burn-in samples")
    std = tail.std(axis=0)
    centered = tail - tail.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    power[0] = 0.0
    aggregate = power.sum(axis=1)
    if aggregate.sum() == 0.0:
        dominant_frequency = 0.0
        concentration = 0.0
    else:
        peak = int(np.argmax(aggregate))
        dominant_frequency = float(np.fft.rfftfreq(len(tail), dt_sample)[peak])
        concentration = float(aggregate[peak] / aggregate.sum())
    return {
        "tail_motor_std_mean": float(std.mean()),
        "tail_motor_std_max": float(std.max()),
        "tail_motor_mean": float(tail.mean()),
        "dominant_frequency_hz": dominant_frequency,
        "spectral_concentration": concentration,
    }


def autocorrelation_metrics(history: np.ndarray, dt_sample: float, burn_fraction: float, std_threshold: float = 0.01) -> dict[str, float]:
    """Report per-motor periodicity, guarding against one dominant output."""
    tail = history[int(len(history) * burn_fraction):]
    std = tail.std(axis=0)
    active = std >= std_threshold
    scores = []
    frequencies = []
    # Ignore sample-scale smoothness; search for a cycle between 100 ms and 2 s.
    min_lag = max(2, int(np.ceil(0.10 / dt_sample)))
    max_lag = max(min_lag + 1, min(len(tail) // 2, int(np.floor(2.0 / dt_sample))))
    for trace in tail[:, active].T:
        centered = trace - trace.mean()
        denominator = float(np.dot(centered, centered))
        if denominator == 0.0:
            continue
        autocorr = np.correlate(centered, centered, mode="full")[len(trace) - 1:] / denominator
        if max_lag <= min_lag:
            continue
        lag = min_lag + int(np.argmax(autocorr[min_lag:max_lag]))
        scores.append(float(autocorr[lag]))
        frequencies.append(1.0 / (lag * dt_sample))
    return {
        "active_motor_fraction": float(active.mean()) if len(active) else 0.0,
        "autocorr_peak_mean": float(np.mean(scores)) if scores else 0.0,
        "autocorr_peak_median": float(np.median(scores)) if scores else 0.0,
        "autocorr_frequency_hz": float(np.median(frequencies)) if frequencies else 0.0,
        "autocorr_search_min_s": min_lag * dt_sample,
        "autocorr_search_max_s": max_lag * dt_sample,
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def phase_metrics(history: np.ndarray, io, burn_fraction: float) -> dict[str, float | int]:
    tail = history[int(len(history) * burn_fraction):]
    legs = io["motor_leg"].astype(np.int64)
    joints = io["motor_joint"].astype(np.int64)
    signs = io["motor_sign"].astype(np.float64)
    scores: dict[tuple[int, int], np.ndarray] = {}
    opposing = []
    for leg in range(6):
        for joint in range(5):
            positive = (legs == leg) & (joints == joint) & (signs > 0)
            negative = (legs == leg) & (joints == joint) & (signs < 0)
            if positive.any() and negative.any():
                p = tail[:, positive].mean(axis=1)
                n = tail[:, negative].mean(axis=1)
                opposing.append(correlation(p, n))
                scores[(leg, joint)] = p - n
            elif positive.any():
                scores[(leg, joint)] = tail[:, positive].mean(axis=1)
            elif negative.any():
                scores[(leg, joint)] = -tail[:, negative].mean(axis=1)
    left_right = [correlation(scores[(0, joint)], scores[(1, joint)]) for joint in range(5) if (0, joint) in scores and (1, joint) in scores]
    opposing = np.asarray(opposing, dtype=float)
    left_right = np.asarray(left_right, dtype=float)
    return {
        "opposing_pool_corr": float(np.nanmean(opposing)) if np.isfinite(opposing).any() else float("nan"),
        "opposing_pool_pairs": int(np.isfinite(opposing).sum()),
        "left_right_corr": float(np.nanmean(left_right)) if np.isfinite(left_right).any() else float("nan"),
        "left_right_pairs": int(np.isfinite(left_right).sum()),
    }


def run_condition(data: Path, weights_name: str, index_name: str, model: str, drive: float, gW: float, tau: float, steps: int, substeps: int, dt: float, dnb08_gain: float, burn_fraction: float, threshold: float, activation_gain: float, fr_cap: float, heterogeneity: bool, seed: int) -> dict[str, float | int | str]:
    vnc = VNCStepper(data, substeps=substeps, tau=tau, dt=dt, gW=gW, weights_name=weights_name, index_name=index_name, model=model, threshold=threshold, activation_gain=activation_gain, fr_cap=fr_cap, heterogeneity=heterogeneity, seed=seed)
    current = vnc.descending_current(drive, dnb08_gain=dnb08_gain)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)])
    row: dict[str, float | int | str] = {
        "drive": drive,
        "gW": gW,
        "tau": tau,
        "model": model,
        "steps": steps,
        "sample_dt_s": dt * substeps,
        "burn_fraction": burn_fraction,
        "dnb08_gain": dnb08_gain,
    }
    row.update(spectral_metrics(history, dt * substeps, burn_fraction))
    row.update(autocorrelation_metrics(history, dt * substeps, burn_fraction))
    row.update(phase_metrics(history, vnc.io, burn_fraction))
    return row


def log_edges(values: tuple[float, ...]) -> np.ndarray:
    values = np.asarray(sorted(set(values)), dtype=float)
    if len(values) == 1:
        return np.asarray([values[0] / 2, values[0] * 2])
    middle = np.sqrt(values[:-1] * values[1:])
    return np.r_[values[0] ** 2 / middle[0], middle, values[-1] ** 2 / middle[-1]]


def heatmap(ax, rows, x_name, y_name, metric, cmap, norm, colorbar_label, threshold, threshold_label):
    from matplotlib.ticker import NullFormatter

    x_values = tuple(sorted({float(row[x_name]) for row in rows}))
    y_values = tuple(sorted({float(row[y_name]) for row in rows}))
    grid = np.full((len(y_values), len(x_values)), np.nan)
    x_index = {value: i for i, value in enumerate(x_values)}
    y_index = {value: i for i, value in enumerate(y_values)}
    for row in rows:
        grid[y_index[float(row[y_name])], x_index[float(row[x_name])]] = float(row[metric])
    image = ax.pcolormesh(log_edges(x_values), log_edges(y_values), grid, shading="auto", cmap=cmap, norm=norm)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_xticks(x_values, labels=[f"{x:g}" for x in x_values])
    ax.set_yticks(y_values, labels=[f"{y:g}" for y in y_values])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=8, pad=8)
    ax.tick_params(axis="y", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_horizontalalignment("center")
    X, Y = np.meshgrid(x_values, y_values)
    finite = grid[np.isfinite(grid)]
    if finite.size and finite.min() < threshold < finite.max():
        ax.contour(X, Y, grid, levels=[threshold], colors="black", linewidths=1.1, linestyles="--")
        ax.text(0.02, 0.97, threshold_label, transform=ax.transAxes, va="top", fontsize=8, color="black", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})
    return image, colorbar_label


def make_figure(rows: list[dict[str, float | int | str]], fixed_tau: float, fixed_drive: float, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    std_values = np.asarray([float(row["tail_motor_std_mean"]) for row in rows])
    concentration_values = np.asarray([float(row["spectral_concentration"]) for row in rows])
    std_positive = std_values[std_values > 0]
    std_norm = LogNorm(vmin=max(float(std_positive.min()), 1e-5), vmax=max(float(std_positive.max()), 1e-4))
    concentration_norm = Normalize(vmin=0.0, vmax=max(float(concentration_values.max()), 0.05))

    drive_rows = [row for row in rows if np.isclose(float(row["tau"]), fixed_tau)]
    tau_rows = [row for row in rows if np.isclose(float(row["drive"]), fixed_drive)]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), layout="constrained")
    panels = (
        (axes[0, 0], drive_rows, "drive", "gW", "tail_motor_std_mean", "cividis", std_norm, "Mean motor-rate SD (tail)", 0.01, "SD = 0.01"),
        (axes[0, 1], drive_rows, "drive", "gW", "spectral_concentration", "viridis", concentration_norm, "Dominant-frequency power fraction", 0.20, "power = 0.20"),
        (axes[1, 0], tau_rows, "tau", "gW", "tail_motor_std_mean", "cividis", std_norm, "Mean motor-rate SD (tail)", 0.01, "SD = 0.01"),
        (axes[1, 1], tau_rows, "tau", "gW", "spectral_concentration", "viridis", concentration_norm, "Dominant-frequency power fraction", 0.20, "power = 0.20"),
    )
    for ax, panel, x_name, y_name, metric, cmap, norm, label, threshold, threshold_label in panels:
        heatmap(ax, panel, x_name, y_name, metric, cmap, norm, label, threshold, threshold_label)
    axes[0, 0].set_title(f"Drive sweep (tau={fixed_tau:g} s)")
    axes[0, 1].set_title(f"Drive sweep (tau={fixed_tau:g} s) — spectral")
    axes[1, 0].set_title(f"Tau sweep (drive={fixed_drive:g})")
    axes[1, 1].set_title(f"Tau sweep (drive={fixed_drive:g}) — spectral")
    fig.suptitle("Full T1 MANC network under constant DNg100/DNb08 drive", fontsize=12)
    for ax in axes.flat:
        ax.grid(False)
    fig.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label="Mean motor-rate SD (tail)")
    fig.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label="Dominant-frequency power fraction")
    fig.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label="Mean motor-rate SD (tail)")
    fig.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label="Dominant-frequency power fraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=RESULTS / "vnc_bifurcation_sweep")
    ap.add_argument("--weights_name", default="vnc_weight_csr.pkl")
    ap.add_argument("--index_name", default="vnc_io_index.npz")
    ap.add_argument("--model", choices=("bounded", "paper"), default="paper")
    ap.add_argument("--threshold", type=float, default=7.5)
    ap.add_argument("--activation_gain", type=float, default=1.0)
    ap.add_argument("--fr_cap", type=float, default=200.0)
    ap.add_argument("--heterogeneity", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gW", type=numbers, default=DEFAULT_GW)
    ap.add_argument("--drive", type=numbers, default=DEFAULT_DRIVE)
    ap.add_argument("--tau", type=numbers, default=DEFAULT_TAU)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--dnb08_gain", type=float, default=1.0)
    ap.add_argument("--burn_fraction", type=float, default=0.5)
    a = ap.parse_args()
    if a.steps < 16 or not 0.0 <= a.burn_fraction < 1.0:
        raise ValueError("--steps must be >=16 and --burn_fraction must be in [0,1)")
    started = time.time()
    rows: list[dict[str, float | int | str]] = []
    # gW x drive at the central tau; gW x tau at the central drive.
    fixed_tau = min(a.tau, key=lambda x: abs(x - 0.02))
    reference_drive = 250.0 if a.model == "paper" else 1.0
    fixed_drive = min(a.drive, key=lambda x: abs(x - reference_drive))
    for gW in a.gW:
        for drive in a.drive:
            rows.append(run_condition(a.data, a.weights_name, a.index_name, a.model, drive, gW, fixed_tau, a.steps, a.substeps, a.dt, a.dnb08_gain, a.burn_fraction, a.threshold, a.activation_gain, a.fr_cap, a.heterogeneity, a.seed) | {"sweep": "drive"})
        for tau in a.tau:
            rows.append(run_condition(a.data, a.weights_name, a.index_name, a.model, fixed_drive, gW, tau, a.steps, a.substeps, a.dt, a.dnb08_gain, a.burn_fraction, a.threshold, a.activation_gain, a.fr_cap, a.heterogeneity, a.seed) | {"sweep": "tau"})

    output = a.out
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with output.with_suffix(".csv").open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row[column]) for column in columns) + "\n")
    metadata = {
        "data": str(a.data),
        "weights_name": a.weights_name,
        "index_name": a.index_name,
        "model": a.model,
        "threshold": a.threshold,
        "activation_gain": a.activation_gain,
        "fr_cap": a.fr_cap,
        "heterogeneity": a.heterogeneity,
        "seed": a.seed,
        "matrix_balance": balance(a.data, a.weights_name),
        "gW_values": list(a.gW),
        "drive_values": list(a.drive),
        "tau_values": list(a.tau),
        "fixed_tau_for_drive_sweep": fixed_tau,
        "fixed_drive_for_tau_sweep": fixed_drive,
        "steps": a.steps,
        "substeps": a.substeps,
        "dt_s": a.dt,
        "dnb08_gain": a.dnb08_gain,
        "burn_fraction": a.burn_fraction,
        "metric": "tail motor-rate standard deviation and aggregate non-DC FFT power; no smoothing",
        "alt_text": "Four heatmaps show full-T1 MANC motor-rate variance and dominant-frequency power across constant-drive by gW and tau by gW sweeps; dashed contours mark SD 0.01 and spectral power 0.20.",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_s": time.time() - started,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    make_figure(rows, fixed_tau, fixed_drive, output)
    print(f"conditions={len(rows)} elapsed={metadata['elapsed_s']:.1f}s")
    print(json.dumps(metadata["matrix_balance"], sort_keys=True))
    print(f"csv -> {output.with_suffix('.csv')}")
    print(f"figure -> {output.with_suffix('.png')} / {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
