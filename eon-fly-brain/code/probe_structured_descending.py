"""E3: feed computed FlyWire DNg100/DNb08 output into the frozen MANC VNC."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from audit_vnc_legs import leg_scores  # noqa: E402
from mn_to_joint import translate  # noqa: E402
from probe_vnc_phase import (  # noqa: E402
    BANDPASS_HZ,
    PHASE_TOLERANCE_CYCLES,
    TRIPOD_A,
    TRIPOD_B,
    analytic_signal,
    fft_bandpass,
    pair_list,
    wrap_cycles,
)
from run_hexapod_chase import retina  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
DESC_KEYS = ("dng100_left", "dng100_right", "dnb08_left", "dnb08_right")


def brain_type_indices(data: Path) -> dict[str, np.ndarray]:
    complete = pd.read_csv(data / "2025_Completeness_783.csv")
    root_ids = complete.iloc[:, 0].astype("int64").to_numpy()
    positions = {int(root): index for index, root in enumerate(root_ids)}
    annotations = pd.read_csv(data / "flywire_annotations.tsv", sep="\t", low_memory=False)
    result = {}
    for key in DESC_KEYS:
        cell_type, side = ("DNg100", "left") if key == "dng100_left" else \
            ("DNg100", "right") if key == "dng100_right" else \
            ("DNb08", "left") if key == "dnb08_left" else ("DNb08", "right")
        roots = annotations.loc[
            annotations["cell_type"].eq(cell_type) & annotations["side"].eq(side), "root_id"
        ].dropna().astype("int64")
        indices = np.asarray(sorted(positions[root] for root in roots if int(root) in positions), dtype=np.int64)
        expected = 1 if cell_type == "DNg100" else 2
        if len(indices) != expected:
            raise ValueError(f"expected {expected} {cell_type} {side} cells in brain matrix, found {len(indices)}")
        result[key] = indices
    return result


def phase_metrics(scores: np.ndarray, sample_dt: float) -> dict:
    tail = scores[len(scores) // 2:]
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
    return {
        "phase_difference_matrix_rad": difference_rad.tolist(),
        "phase_difference_matrix_cycles": difference_cycles.tolist(),
        "phase_concentration_matrix": concentration.tolist(),
        "within_phase_cosine": within_cos,
        "cross_phase_cosine": cross_cos,
        "tripod_coherence_scalar": within_cos - cross_cos,
        "within_pairs_in_phase": int(within_ok.sum()),
        "cross_pairs_antiphase": int(cross_ok.sum()),
        "within_pair_phase_cycles": within_cycles.tolist(),
        "cross_pair_phase_cycles": cross_cycles.tolist(),
        "verdict": "E3-PRESENT" if bool(within_ok.all() and cross_ok.all()) else "E3-ABSENT",
    }


def run(vnc_data: Path, brain_data: Path, weights_name: str, index_name: str) -> dict:
    vnc = VNCStepper(
        vnc_data,
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
    with (brain_data / "weight_csr.pkl").open("rb") as handle:
        brain_weight = pickle.load(handle)
    if brain_weight.layout != torch.sparse_csr:
        brain_weight = brain_weight.to_sparse_csr()
    brain_weight = brain_weight.float()
    brain_io = np.load(brain_data / "io_index.npz")
    brain_n = int(brain_weight.shape[0])
    visual_left = torch.from_numpy(brain_io["vis_left"]).long()
    visual_right = torch.from_numpy(brain_io["vis_right"]).long()
    type_indices = brain_type_indices(brain_data)
    brain_state = torch.zeros(brain_n, dtype=torch.float32)
    brain_decay = 1.0 - 0.005 / 0.02
    brain_dt_tau = 0.005 / 0.02
    present, az, distance, size = retina((0.0, 0.0), 0.0, (1.2, 0.8))
    visual_current = torch.zeros(brain_n, dtype=torch.float32)
    visual_mag = 1.2 * (0.5 + min(1.0, size))
    if present:
        visual_current[visual_left] = visual_mag * (max(0.0, az) + 0.15)
        visual_current[visual_right] = visual_mag * (max(0.0, -az) + 0.15)

    motor_history = []
    joint_history = []
    brain_history = {key: [] for key in DESC_KEYS}
    steps = 1500
    for _ in range(steps):
        with torch.no_grad():
            for _ in range(12):
                rates = torch.relu(torch.tanh(brain_state))
                brain_state.mul_(brain_decay).add_(brain_dt_tau * (0.006 * (brain_weight @ rates) + visual_current))
                brain_state.clamp_(-5.0, 5.0)
            rates = torch.relu(torch.tanh(brain_state))
        outputs = {key: float(rates[indices].mean()) for key, indices in type_indices.items()}
        for key, value in outputs.items():
            brain_history[key].append(value)
        current = np.zeros(vnc.n, dtype=np.float32)
        for key in DESC_KEYS:
            current[vnc.io[key]] = 100.0 * outputs[key]
        motor = vnc.step(current).cpu().numpy()
        motor_history.append(motor)
        joint_history.append(translate(motor, vnc.io))

    motor_history = np.asarray(motor_history)
    joint_history = np.asarray(joint_history).reshape(steps, 6, 3)
    motor_scores = leg_scores(motor_history, vnc.io)[..., 0]
    joint_scores = joint_history[..., 0]
    result = {
        "protocol": {
            "weights_name": weights_name,
            "index_name": index_name,
            "brain_model": "bounded rate model",
            "brain_tau": 0.02,
            "brain_dt": 0.005,
            "brain_substeps": 12,
            "brain_gW": 0.006,
            "brain_drive_in": 1.2,
            "vnc_model": "paper",
            "vnc_tau": 0.02,
            "vnc_dt": 0.005,
            "vnc_substeps": 4,
            "vnc_gW": 0.03,
            "vnc_threshold": 7.5,
            "vnc_activation_gain": 1.0,
            "vnc_fr_cap": 200.0,
            "vnc_heterogeneity": True,
            "vnc_seed": 1,
            "brain_to_vnc_current_scale": 100.0,
            "steps": steps,
            "sample_dt_s": 0.02,
            "burn_fraction": 0.5,
            "visual_scene": {"present": bool(present), "azimuth_rad": float(az), "distance_m": float(distance), "size": float(size)},
            "bandpass_hz": list(BANDPASS_HZ),
            "phase_tolerance_cycles": PHASE_TOLERANCE_CYCLES,
            "motor_map": "mn_to_joint.translate, fixed stance/gains",
        },
        "leg_names": list(LEG_NAMES),
        "brain_descending_type_counts": {key: int(len(type_indices[key])) for key in DESC_KEYS},
        "brain_descending_tail_mean": {key: float(np.mean(values[steps // 2:])) for key, values in brain_history.items()},
        "brain_descending_tail_std": {key: float(np.std(values[steps // 2:])) for key, values in brain_history.items()},
        "vnc_motor_output": phase_metrics(motor_scores, 0.02),
        "joint_target_coxa": phase_metrics(joint_scores, 0.02),
    }
    assert motor_history.shape == (steps, len(vnc.motor_out))
    assert joint_history.shape == (steps, 6, 3)
    return result


def make_figure(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for row, (label, key) in enumerate((("VNC motor output", "vnc_motor_output"), ("joint-target coxa", "joint_target_coxa"))):
        phase = np.asarray(result[key]["phase_difference_matrix_cycles"])
        concentration = np.asarray(result[key]["phase_concentration_matrix"])
        image = axes[row, 0].imshow(phase, vmin=-0.5, vmax=0.5, cmap="twilight")
        axes[row, 0].set(xticks=range(6), xticklabels=LEG_NAMES, yticks=range(6), yticklabels=LEG_NAMES, title=f"{label}: phase (cycles)")
        axes[row, 0].tick_params(axis="x", rotation=35)
        fig.colorbar(image, ax=axes[row, 0], label="left − right")
        image = axes[row, 1].imshow(concentration, vmin=0.0, vmax=1.0, cmap="viridis")
        axes[row, 1].set(xticks=range(6), xticklabels=LEG_NAMES, yticks=range(6), yticklabels=LEG_NAMES, title=f"{label}: concentration")
        axes[row, 1].tick_params(axis="x", rotation=35)
        fig.colorbar(image, ax=axes[row, 1], label="|mean exp(iΔφ)|")
    fig.suptitle("E3 structured descending drive — fixed FlyWire DNg100/DNb08 output", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=RESULTS)
    ap.add_argument("--brain_data", type=Path, default=DATA)
    ap.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    ap.add_argument("--index_name", default="all_legs_io_index.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "e3_structured_descending")
    a = ap.parse_args()
    result = run(a.data, a.brain_data, a.weights_name, a.index_name)
    make_figure(result, a.out)
    a.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key in ("vnc_motor_output", "joint_target_coxa"):
        np.savetxt(a.out.with_name(a.out.name + "_" + key + ".csv"), np.asarray(result[key]["phase_difference_matrix_rad"]), delimiter=",", header=",".join(LEG_NAMES), comments="")
    print(json.dumps({key: result[key] for key in ("vnc_motor_output", "joint_target_coxa")}, sort_keys=True))
    print(f"audit -> {a.out.with_suffix('.json')}")
    print(f"figure -> {a.out.with_suffix('.png')} / {a.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
