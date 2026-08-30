"""E0: probe tripod phase directly at the all-leg VNC motor output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from audit_vnc_legs import leg_scores  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
TRIPOD_A = (0, 3, 4)
TRIPOD_B = (1, 2, 5)
BANDPASS_HZ = (1.5, 4.0)
PHASE_TOLERANCE_CYCLES = 0.15


def fft_bandpass(signal: np.ndarray, dt: float, low: float, high: float) -> np.ndarray:
    frequency = np.fft.rfftfreq(len(signal), dt)
    spectrum = np.fft.rfft(signal - signal.mean(axis=0, keepdims=True), axis=0)
    spectrum[(frequency < low) | (frequency > high), :] = 0.0
    return np.fft.irfft(spectrum, n=len(signal), axis=0)


def analytic_signal(signal: np.ndarray) -> np.ndarray:
    n = len(signal)
    spectrum = np.fft.fft(signal, axis=0)
    multiplier = np.zeros(n, dtype=float)
    if n % 2 == 0:
        multiplier[[0, n // 2]] = 1.0
        multiplier[1:n // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier[:, None], axis=0)


def pair_list(groups: tuple[tuple[int, ...], ...]) -> list[tuple[int, int]]:
    return [(left, right) for group in groups for i, left in enumerate(group) for right in group[i + 1:]]


def wrap_cycles(value: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(value) + 0.5) % 1.0 - 0.5


def run(data: Path, weights_name: str, index_name: str) -> dict:
    vnc = VNCStepper(
        data,
        substeps=4,
        tau=0.02,
        dt=0.005,
        gW=0.03,
        weights_name=weights_name,
        index_name=index_name,
        model="paper",
        threshold=7.5,
        activation_gain=1.0,
        fr_cap=200.0,
        heterogeneity=True,
        seed=1,
    )
    steps = 1500
    sample_dt = 0.005 * 4
    history = np.stack([vnc.step(vnc.descending_current(100.0)).cpu().numpy() for _ in range(steps)])
    scores = leg_scores(history, vnc.io)[..., 0]
    tail = scores[steps // 2:]
    filtered = fft_bandpass(tail, sample_dt, *BANDPASS_HZ)
    phase = np.angle(analytic_signal(filtered))
    difference_rad = np.zeros((6, 6), dtype=float)
    concentration = np.zeros((6, 6), dtype=float)
    for left in range(6):
        for right in range(6):
            vector = np.exp(1j * (phase[:, left] - phase[:, right]))
            mean_vector = np.mean(vector)
            difference_rad[left, right] = np.angle(mean_vector)
            concentration[left, right] = abs(mean_vector)

    within = pair_list((TRIPOD_A, TRIPOD_B))
    cross = [(left, right) for left in TRIPOD_A for right in TRIPOD_B]
    difference_cycles = difference_rad / (2 * np.pi)
    within_cycles = np.asarray([difference_cycles[left, right] for left, right in within])
    cross_cycles = np.asarray([difference_cycles[left, right] for left, right in cross])
    within_ok = np.abs(wrap_cycles(within_cycles)) <= PHASE_TOLERANCE_CYCLES
    cross_ok = (0.5 - np.abs(wrap_cycles(cross_cycles))) <= PHASE_TOLERANCE_CYCLES
    within_cos = float(np.mean([np.cos(difference_rad[left, right]) for left, right in within]))
    cross_cos = float(np.mean([np.cos(difference_rad[left, right]) for left, right in cross]))
    result = {
        "protocol": {
            "weights_name": weights_name,
            "index_name": index_name,
            "model": "paper",
            "drive": 100.0,
            "gW": 0.03,
            "tau": 0.02,
            "dt": 0.005,
            "substeps": 4,
            "sample_dt_s": sample_dt,
            "steps": steps,
            "burn_fraction": 0.5,
            "bandpass_hz": list(BANDPASS_HZ),
            "heterogeneity": True,
            "seed": 1,
            "phase_tolerance_cycles": PHASE_TOLERANCE_CYCLES,
        },
        "leg_names": list(LEG_NAMES),
        "phase_difference_matrix_rad": difference_rad.tolist(),
        "phase_difference_matrix_cycles": difference_cycles.tolist(),
        "phase_concentration_matrix": concentration.tolist(),
        "within_phase_cosine": within_cos,
        "cross_phase_cosine": cross_cos,
        "tripod_coherence_scalar": within_cos - cross_cos,
        "within_pair_count": len(within),
        "within_pairs_in_phase": int(within_ok.sum()),
        "cross_pair_count": len(cross),
        "cross_pairs_antiphase": int(cross_ok.sum()),
        "within_pair_phase_cycles": within_cycles.tolist(),
        "cross_pair_phase_cycles": cross_cycles.tolist(),
        "verdict": "E0-PRESENT" if bool(within_ok.all() and cross_ok.all()) else "E0-ABSENT",
    }
    assert difference_rad.shape == (6, 6)
    assert np.allclose(np.diag(difference_rad), 0.0)
    return result


def make_figure(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = result["leg_names"]
    phase = np.asarray(result["phase_difference_matrix_cycles"])
    concentration = np.asarray(result["phase_concentration_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    image = axes[0].imshow(phase, vmin=-0.5, vmax=0.5, cmap="twilight")
    axes[0].set(xticks=range(6), xticklabels=names, yticks=range(6), yticklabels=names, title="Mean phase difference (cycles)")
    axes[0].tick_params(axis="x", rotation=35)
    fig.colorbar(image, ax=axes[0], label="left − right phase")
    image = axes[1].imshow(concentration, vmin=0.0, vmax=1.0, cmap="viridis")
    axes[1].set(xticks=range(6), xticklabels=names, yticks=range(6), yticklabels=names, title="Circular phase concentration")
    axes[1].tick_params(axis="x", rotation=35)
    fig.colorbar(image, ax=axes[1], label="|mean exp(iΔφ)|")
    fig.suptitle(f"E0 VNC motor-output phase probe — {result['verdict']}", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=RESULTS)
    ap.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    ap.add_argument("--index_name", default="all_legs_io_index.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "vnc_phase_probe_e0")
    a = ap.parse_args()
    result = run(a.data, a.weights_name, a.index_name)
    make_figure(result, a.out)
    a.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savetxt(a.out.with_suffix(".csv"), np.asarray(result["phase_difference_matrix_rad"]), delimiter=",", header=",".join(result["leg_names"]), comments="")
    print(json.dumps({key: result[key] for key in ("within_phase_cosine", "cross_phase_cosine", "tripod_coherence_scalar", "within_pairs_in_phase", "cross_pairs_antiphase", "verdict")}, sort_keys=True))
    print("phase difference matrix (cycles)")
    print(np.asarray(result["phase_difference_matrix_cycles"]))
    print(f"audit -> {a.out.with_suffix('.json')}")
    print(f"figure -> {a.out.with_suffix('.png')} / {a.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
