"""Named motor-neuron readouts for the frozen VNC graph."""
from __future__ import annotations

import numpy as np

from probe_vnc_phase import (
    BANDPASS_HZ,
    PHASE_TOLERANCE_CYCLES,
    analytic_signal,
    band_power_share,
    fft_bandpass,
    phase_metrics,
    wrap_cycles,
)


CONCENTRATION_FLOOR = 0.3
EXPECTED_JOINTS = ("coxa", "trochanter", "femur", "tibia", "tarsus")
TESTABLE_JOINTS = {
    "coxa": tuple(range(6)),
    "trochanter": tuple(range(6)),
    "tibia": tuple(range(6)),
    "tarsus": (0, 1),
}


def motor_types(io) -> list[str]:
    return sorted({" ".join(str(label).split(" ")[:-1]) for label in io["motor_labels"]})


def build_index(io) -> dict:
    labels = np.asarray(io["motor_labels"])
    legs = np.asarray(io["motor_leg"], dtype=np.int64)
    joints = np.asarray(io["motor_joint"], dtype=np.int64)
    signs = np.asarray(io["motor_sign"], dtype=np.float32)
    leg_names = tuple(str(value) for value in io["leg_names"])
    joint_names = tuple(str(value) for value in io["joint_names"])
    types = np.asarray([" ".join(str(label).split(" ")[:-1]) for label in labels])
    if labels.shape != (260,):
        raise AssertionError(f"motor_out must contain 260 labels, got {labels.shape}")
    if len(motor_types(io)) != 14:
        raise AssertionError("motor labels must parse to exactly 14 named types")
    if joint_names != EXPECTED_JOINTS:
        raise AssertionError(f"joint order changed: {joint_names}")
    cells = []
    for joint, leg_ids in TESTABLE_JOINTS.items():
        joint_id = joint_names.index(joint)
        for leg_id in leg_ids:
            mask = joints == joint_id
            positive = int(np.count_nonzero(mask & (legs == leg_id) & (signs > 0)))
            negative = int(np.count_nonzero(mask & (legs == leg_id) & (signs < 0)))
            if positive == 0 or negative == 0:
                raise AssertionError(f"unpaired antagonist cell: {leg_names[leg_id]} {joint}")
            cells.append({
                "leg": int(leg_id), "leg_name": leg_names[leg_id],
                "joint": int(joint_id), "joint_name": joint,
                "positive_pool_size": positive, "negative_pool_size": negative,
            })
    if len(cells) != 20:
        raise AssertionError(f"expected 20 testable cells, got {len(cells)}")
    if any(cell["joint_name"] == "femur" for cell in cells):
        raise AssertionError("femur entered antagonist test")
    return {
        "motor_types": motor_types(io),
        "motor_type_index": types,
        "motor_leg": legs,
        "motor_joint": joints,
        "motor_sign": signs,
        "leg_names": leg_names,
        "joint_names": joint_names,
        "testable_cells": cells,
    }


def _check_history(history: np.ndarray, index: dict) -> np.ndarray:
    value = np.asarray(history, dtype=np.float32)
    if value.ndim == 2:
        value = value[None, ...]
    if value.ndim != 3 or value.shape[2] != len(index["motor_leg"]):
        raise AssertionError(f"history shape {value.shape}; expected (realization,time,260)")
    if not np.isfinite(value).all():
        raise AssertionError("motor history contains non-finite values")
    return value


def pair_phase(agonist: np.ndarray, antagonist: np.ndarray, sample_dt: float) -> dict:
    pair = np.stack((np.asarray(agonist, dtype=np.float64), np.asarray(antagonist, dtype=np.float64)), axis=1)
    tail = pair[len(pair) // 2:]
    filtered = fft_bandpass(tail, sample_dt, *BANDPASS_HZ)
    fa, fb = filtered.T
    band = band_power_share(tail, sample_dt)
    if max(float(np.max(np.abs(fa))), float(np.max(np.abs(fb)))) <= np.finfo(float).eps:
        return {
            "phase_cycles": None,
            "concentration": 0.0,
            "band_power_share": float(band["band_power_share_mean"]),
        }
    analytic = analytic_signal(filtered)
    z = np.mean(np.exp(1j * (np.angle(analytic[:, 0]) - np.angle(analytic[:, 1]))))
    concentration = float(abs(z))
    phase = None if concentration <= 0.0 else float(wrap_cycles(np.angle(z) / (2.0 * np.pi)))
    return {
        "phase_cycles": phase,
        "concentration": concentration,
        "band_power_share": float(band["band_power_share_mean"]),
    }


def phase_table(history: np.ndarray, index: dict, sample_dt: float) -> dict:
    value = _check_history(history, index)
    rows = []
    for cell in index["testable_cells"]:
        mask = (index["motor_leg"] == cell["leg"]) & (index["motor_joint"] == cell["joint"])
        ago = value[:, :, mask & (index["motor_sign"] > 0)].mean(axis=2)
        anta = value[:, :, mask & (index["motor_sign"] < 0)].mean(axis=2)
        per_realization = [pair_phase(a, b, sample_dt) for a, b in zip(ago, anta)]
        concentrations = np.asarray([item["concentration"] for item in per_realization], dtype=float)
        band_shares = np.asarray([item["band_power_share"] for item in per_realization], dtype=float)
        valid_phases = np.asarray([
            item["phase_cycles"] for item in per_realization
            if item["phase_cycles"] is not None and item["concentration"] >= CONCENTRATION_FLOOR
        ], dtype=float)
        phase_mean = None if not len(valid_phases) else float(np.angle(np.mean(np.exp(2j * np.pi * valid_phases))) / (2.0 * np.pi))
        rows.append({
            **cell,
            "phase_cycles": phase_mean,
            "phase_cycles_sd": float(np.std(valid_phases, ddof=1)) if len(valid_phases) > 1 else 0.0,
            "concentration": float(concentrations.mean()),
            "concentration_sd": float(concentrations.std(ddof=1)) if len(concentrations) > 1 else 0.0,
            "in_band_power_share": float(band_shares.mean()),
            "in_band_power_share_sd": float(band_shares.std(ddof=1)) if len(band_shares) > 1 else 0.0,
            "valid_realizations": int(len(valid_phases)),
            "missing_realizations": int(len(per_realization) - len(valid_phases)),
            "pass_count": int(sum(
                item["phase_cycles"] is not None
                and item["concentration"] >= CONCENTRATION_FLOOR
                and abs(abs(float(item["phase_cycles"])) - 0.5) <= PHASE_TOLERANCE_CYCLES
                for item in per_realization
            )),
            "per_realization": per_realization,
        })
    pass_counts = []
    for r in range(len(value)):
        count = 0
        for cell in index["testable_cells"]:
            mask = (index["motor_leg"] == cell["leg"]) & (index["motor_joint"] == cell["joint"])
            item = pair_phase(
                value[r][:, mask & (index["motor_sign"] > 0)].mean(axis=1),
                value[r][:, mask & (index["motor_sign"] < 0)].mean(axis=1),
                sample_dt,
            )
            count += int(
                item["phase_cycles"] is not None
                and item["concentration"] >= CONCENTRATION_FLOOR
                and abs(abs(float(item["phase_cycles"])) - 0.5) <= PHASE_TOLERANCE_CYCLES
            )
        pass_counts.append(count)
    pass_counts = np.asarray(pass_counts, dtype=float)
    return {
        "cells": rows,
        "testable_cell_count": len(rows),
        "pass_count_mean": float(pass_counts.mean()),
        "pass_count_sd": float(pass_counts.std(ddof=1)) if len(pass_counts) > 1 else 0.0,
        "pass_counts_per_realization": pass_counts.astype(int).tolist(),
        "concentration_floor": CONCENTRATION_FLOOR,
        "phase_tolerance_cycles": PHASE_TOLERANCE_CYCLES,
    }


def leg_phase(history: np.ndarray, index: dict, sample_dt: float) -> dict:
    value = _check_history(history, index)
    scores = np.stack([
        value[:, :, index["motor_leg"] == leg].sum(axis=2)
        for leg in range(6)
    ], axis=2)
    records = [phase_metrics(scores[r], sample_dt) for r in range(len(value))]
    band = [band_power_share(scores[r], sample_dt) for r in range(len(value))]
    scalar = np.asarray([record["tripod_coherence_scalar"] for record in records], dtype=float)
    return {
        "tripod_coherence_mean": float(scalar.mean()),
        "tripod_coherence_sd": float(scalar.std(ddof=1)) if len(scalar) > 1 else 0.0,
        "verdict_counts": {
            label: int(sum(record["verdict"] == label for record in records))
            for label in sorted({record["verdict"] for record in records})
        },
        "band_power_share_mean": float(np.mean([item["band_power_share_mean"] for item in band])),
        "band_power_share_sd": float(np.std([item["band_power_share_mean"] for item in band], ddof=1)) if len(band) > 1 else 0.0,
        "per_realization": [
            {
                "tripod_coherence_scalar": float(record["tripod_coherence_scalar"]),
                "verdict": record["verdict"],
                "band_power_share_mean": float(band_item["band_power_share_mean"]),
            }
            for record, band_item in zip(records, band)
        ],
    }


def _type_values(value: np.ndarray, index: dict, ceiling_hz: float) -> np.ndarray:
    return np.asarray([
        value[:, :, index["motor_type_index"] == name].mean(axis=(1, 2)) * ceiling_hz
        if np.any(index["motor_type_index"] == name) else np.zeros(len(value))
        for name in index["motor_types"]
    ], dtype=np.float64).T


def selectivity(value: np.ndarray, index: dict, ceiling_hz: float) -> dict:
    value = _check_history(value, index)
    per_realization = _type_values(value, index, ceiling_hz)
    raw = per_realization.mean(axis=0)
    z = np.zeros_like(per_realization)
    row_mean = per_realization.mean(axis=1, keepdims=True)
    row_sd = per_realization.std(axis=1, keepdims=True)
    np.divide(per_realization - row_mean, row_sd, out=z, where=row_sd > 0.0)
    return {
        "type_names": index["motor_types"],
        "raw_rate_hz": raw.tolist(),
        "z_rate": z.mean(axis=0).tolist(),
        "z_rate_sd": z.std(axis=0, ddof=1).tolist() if len(z) > 1 else np.zeros(z.shape[1]).tolist(),
        "per_realization_raw_rate_hz": per_realization.tolist(),
        "per_realization_z_rate": z.tolist(),
    }


def rate_table(value: np.ndarray, index: dict, ceiling_hz: float) -> list[dict]:
    value = _check_history(value, index)
    rows = []
    for name in index["motor_types"]:
        for leg, leg_name in enumerate(index["leg_names"]):
            mask = (index["motor_type_index"] == name) & (index["motor_leg"] == leg)
            per_realization = value[:, :, mask].mean(axis=(1, 2)) * ceiling_hz if np.any(mask) else np.zeros(len(value))
            rows.append({
                "motor_type": name, "leg": leg, "leg_name": leg_name,
                "motor_count": int(mask.sum()),
                "rate_hz": float(per_realization.mean()),
                "rate_hz_sd": float(per_realization.std(ddof=1)) if len(per_realization) > 1 else 0.0,
            })
    return rows


def activity(value: np.ndarray, ceiling_hz: float) -> dict:
    value = np.asarray(value, dtype=np.float32)
    per = value.mean(axis=(1, 2)) * ceiling_hz
    active = (value.mean(axis=1) > 1e-6).mean(axis=1)
    return {
        "frac_active": float(active.mean()),
        "frac_active_sd": float(active.std(ddof=1)) if len(active) > 1 else 0.0,
        "motor_hz": float(per.mean()),
        "motor_hz_sd": float(per.std(ddof=1)) if len(per) > 1 else 0.0,
        "rate_ceiling_hz": float(ceiling_hz),
        "active_sample_fraction": float((value > 0.0).mean()),
    }


def summarize(value: np.ndarray, index: dict, ceiling_hz: float, sample_dt: float) -> dict:
    value = _check_history(value, index)
    return {
        "n_realizations": int(len(value)),
        "activity": activity(value, ceiling_hz),
        "rate_table": rate_table(value, index, ceiling_hz),
        "selectivity": selectivity(value, index, ceiling_hz),
        "antagonist_phase": phase_table(value, index, sample_dt),
        "inter_leg_phase": leg_phase(value, index, sample_dt),
    }


def self_check() -> None:
    dt = 0.02
    time = np.arange(2000) * dt
    wave = np.sin(2.0 * np.pi * 2.0 * time)
    anti = pair_phase(wave, -wave, dt)
    assert anti["concentration"] > 0.99 and abs(abs(anti["phase_cycles"]) - 0.5) < 0.01, anti
    rng = np.random.default_rng(7)
    noise = pair_phase(rng.normal(size=len(time)), rng.normal(size=len(time)), dt)
    assert noise["concentration"] < CONCENTRATION_FLOOR, noise
    constant = pair_phase(np.ones(len(time)), np.ones(len(time)), dt)
    assert constant["concentration"] == 0.0 and constant["phase_cycles"] is None, constant
    print("labeled_line: synthetic anti-phase/noise/constant checks passed")


if __name__ == "__main__":
    self_check()
