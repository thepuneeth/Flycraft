"""Validate a selected full-T1 regime as rhythm rather than broadband motion."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
import sys
sys.path.insert(0, str(HERE))
from sweep_vnc import autocorrelation_metrics, balance, phase_metrics, spectral_metrics  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


def pool(history, io, leg, joint, sign=None):
    legs = io["motor_leg"].astype(np.int64)
    joints = io["motor_joint"].astype(np.int64)
    signs = io["motor_sign"].astype(np.float64)
    mask = (legs == leg) & (joints == joint)
    if sign is not None:
        mask &= signs == sign
    if not mask.any():
        return None
    return history[:, mask].mean(axis=1)


def dominant_spectrum(signal, dt):
    centered = signal - signal.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    power[0] = 0.0
    frequency = np.fft.rfftfreq(len(signal), dt)
    return frequency, power


def make_figure(history, io, dt_sample, burn_fraction, gW, drive, tau, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tail = history[int(len(history) * burn_fraction):]
    window = tail[-min(len(tail), max(8, int(round(8.0 / dt_sample)))):]
    time_s = np.arange(len(window)) * dt_sample
    specs = (
        (0, 0, 1.0, "T1L coxa +"),
        (0, 0, -1.0, "T1L coxa −"),
        (0, 2, 1.0, "T1L femur +"),
        (0, 2, -1.0, "T1L femur −"),
        (1, 0, 1.0, "T1R coxa +"),
        (1, 0, -1.0, "T1R coxa −"),
    )
    traces = [(label, pool(window, io, leg, joint, sign)) for leg, joint, sign, label in specs]
    traces = [(label, trace) for label, trace in traces if trace is not None]
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#004488", "#882255")
    styles = ("-", "--", "-", "--", "-", "--")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), layout="constrained", height_ratios=(1.3, 1.0, 0.8))
    for (label, trace), color, style in zip(traces, colors, styles):
        axes[0].plot(time_s, trace, color=color, linestyle=style, linewidth=1.2, label=label)
    axes[0].set(xlabel="Time in validation window (s)", ylabel="Motor rate (normalized)", title="Constant descending drive: motor-pool traces")
    axes[0].legend(ncol=3, fontsize=8, frameon=False)

    for (label, trace), color, style in zip(traces, colors, styles):
        frequency, power = dominant_spectrum(trace, dt_sample)
        keep = frequency > 0
        if keep.any() and power[keep].sum() > 0:
            axes[1].plot(frequency[keep], power[keep] / power[keep].sum(), color=color, linestyle=style, linewidth=1.0, label=label)
    axes[1].set(xlim=(0, min(15.0, 0.5 / dt_sample)), xlabel="Frequency (Hz)", ylabel="Relative non-DC power", title="Unsmooth spectra")
    axes[1].legend(ncol=3, fontsize=8, frameon=False)

    phase = phase_metrics(history, io, burn_fraction)
    labels = ("Opposing pools\n(flexor/extensor sign)", "T1 left/right\npool scores")
    values = (phase["opposing_pool_corr"], phase["left_right_corr"])
    bars = axes[2].bar(labels, values, color=("#0072B2", "#D55E00"), edgecolor="black", hatch=("///", "\\\\"))
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_ylabel("Pearson correlation")
    axes[2].set_title("Phase proxy (−1 = antiphase, +1 = in-phase)")
    for bar, value in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + (0.06 if value >= 0 else -0.12), f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.suptitle(f"Full T1 MANC rhythm validation — gW={gW:g}, drive={drive:g}, tau={tau:g} s", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return traces, phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--weights_name", default="vnc_weight_csr.pkl")
    ap.add_argument("--index_name", default="vnc_io_index.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "vnc_rhythm_validation")
    ap.add_argument("--model", choices=("bounded", "paper"), default="paper")
    ap.add_argument("--gW", type=float, default=0.03)
    ap.add_argument("--drive", type=float, default=250.0)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--dnb08_gain", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--substeps", type=int, default=12)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--burn_fraction", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=7.5)
    ap.add_argument("--activation_gain", type=float, default=1.0)
    ap.add_argument("--fr_cap", type=float, default=200.0)
    ap.add_argument("--heterogeneity", action="store_true")
    ap.add_argument("--shuffle_vnc", action="store_true", help="permute presynaptic columns for matched topology control")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.steps < 32 or not 0.0 <= a.burn_fraction < 1.0:
        raise ValueError("--steps must be >=32 and --burn_fraction must be in [0,1)")

    vnc = VNCStepper(a.data, substeps=a.substeps, tau=a.tau, dt=a.dt, gW=a.gW, weights_name=a.weights_name, index_name=a.index_name, model=a.model, threshold=a.threshold, activation_gain=a.activation_gain, fr_cap=a.fr_cap, heterogeneity=a.heterogeneity, seed=a.seed, shuffle=a.shuffle_vnc)
    current = vnc.descending_current(a.drive, dnb08_gain=a.dnb08_gain)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(a.steps)])
    metrics = spectral_metrics(history, a.dt * a.substeps, a.burn_fraction)
    metrics.update(autocorrelation_metrics(history, a.dt * a.substeps, a.burn_fraction))
    metrics.update(phase_metrics(history, vnc.io, a.burn_fraction))
    metrics.update({"model": a.model, "gW": a.gW, "drive": a.drive, "tau": a.tau, "steps": a.steps, "sample_dt_s": a.dt * a.substeps, "dnb08_gain": a.dnb08_gain, "constant_drive": True, "heterogeneity": a.heterogeneity, "shuffle_vnc": a.shuffle_vnc, "seed": a.seed})
    output = a.out
    output.parent.mkdir(parents=True, exist_ok=True)
    traces, _phase = make_figure(history, vnc.io, a.dt * a.substeps, a.burn_fraction, a.gW, a.drive, a.tau, output)
    trace_length = len(traces[0][1]) if traces else 0
    trace_columns = {"time_s": np.arange(trace_length) * a.dt * a.substeps}
    for label, trace in traces:
        trace_columns[label] = trace
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(trace_columns.keys())
        writer.writerows(zip(*trace_columns.values()))
    manifest = {
        "data": str(a.data),
        "weights_name": a.weights_name,
        "index_name": a.index_name,
        "model": a.model,
        "threshold": a.threshold,
        "activation_gain": a.activation_gain,
        "fr_cap": a.fr_cap,
        "heterogeneity": a.heterogeneity,
        "shuffle_vnc": a.shuffle_vnc,
        "seed": a.seed,
        "matrix_balance": balance(a.data, a.weights_name),
        "metrics": metrics,
        "transformations": ["constant DNg100/DNb08 current", f"discarded first {a.burn_fraction:.2f} of samples", "no smoothing", "FFT on post-burn-in traces"],
        "alt_text": "Three-panel figure: constant descending drive produces motor traces, their unsmoothed spectra, and opposing-pool/left-right phase correlations for the full T1 MANC network.",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    print(f"trace csv -> {output.with_suffix('.csv')}")
    print(f"figure -> {output.with_suffix('.png')} / {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
