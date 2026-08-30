"""DC-1 delayed phase coupling checks; later phases run from this module."""
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
from last_three_jobs import (  # noqa: E402
    BODY_STEPS,
    CLEAN_MAX_PHASE_STEP_RAD,
    CLEAN_MAX_STEP_CHANGE_RAD,
    CTRL_HZ,
    GAIN_CANDIDATES,
    build_coupling_matrices,
    coupled_phases,
    delayed_coupled_phases,
)


CLOCK_HZ = 0.8 + 0.5 * 0.8
DC2_MAX_PER_HOP_STEPS = int(round(CTRL_HZ / (2.0 * CLOCK_HZ)))
DC2_RANDOM_PHASE_SEED = 20260830
DC2_PHASE_LOCK_VARIANCE_RAD2 = 0.25
DC2_SELECTION_RULE = (
    "Among admissible cells, select the one with the smallest per_hop_steps, "
    "breaking ties by the largest gain. Selection uses only the neutral "
    "phase-locked and bounded criteria across all three conditions."
)


def anatomical_delay_matrix(per_hop_steps: int) -> np.ndarray:
    if int(per_hop_steps) != per_hop_steps or per_hop_steps < 0:
        raise ValueError("per_hop_steps must be a non-negative integer")
    segments = np.arange(6) // 2
    return np.rint(int(per_hop_steps) * np.abs(segments[:, None] - segments[None, :])).astype(int)


def dc1_self_check(data: Path) -> None:
    matrices = build_coupling_matrices(data)["matrices"]
    rng = np.random.default_rng(DC2_RANDOM_PHASE_SEED)
    source = rng.normal(size=(32, 6)).cumsum(axis=0)
    expected = coupled_phases(source, matrices["real"], GAIN_CANDIDATES[7])
    actual = delayed_coupled_phases(source, matrices["real"], GAIN_CANDIDATES[7], 0)
    assert np.allclose(actual, expected)

    neutral = np.arange(BODY_STEPS, dtype=np.float64)[:, None] * (2.0 * np.pi * CLOCK_HZ / CTRL_HZ)
    neutral = np.broadcast_to(neutral, (BODY_STEPS, 6)).copy()
    neutral += rng.uniform(0.0, 2.0 * np.pi, size=6)
    for per_hop_steps in range(DC2_MAX_PER_HOP_STEPS + 1):
        delay = anatomical_delay_matrix(per_hop_steps)
        result = delayed_coupled_phases(neutral, matrices["real"], GAIN_CANDIDATES[-1], delay)
        steps = np.diff(result, axis=0)
        assert np.isfinite(result).all()
        assert np.abs(steps).max() <= CLEAN_MAX_PHASE_STEP_RAD
        assert np.abs(np.diff(steps, axis=0)).max() <= CLEAN_MAX_STEP_CHANGE_RAD

    expected_delay = np.asarray(
        [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [1, 1, 0, 0, 1, 1],
         [1, 1, 0, 0, 1, 1], [2, 2, 1, 1, 0, 0], [2, 2, 1, 1, 0, 0]],
        dtype=int,
    )
    assert np.array_equal(anatomical_delay_matrix(1), expected_delay)
    print("DC-1 regression: PASS")
    print("DC-1 warmup boundedness: PASS")
    print("DC-1 anatomical delay matrix (per_hop_steps=1):")
    print(anatomical_delay_matrix(1))
    print("GATE DC-1: PASS")


def neutral_phase_source() -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(DC2_RANDOM_PHASE_SEED)
    offsets = rng.uniform(0.0, 2.0 * np.pi, size=6)
    phase = np.arange(BODY_STEPS, dtype=np.float64)[:, None] * (2.0 * np.pi * CLOCK_HZ / CTRL_HZ)
    phase = np.broadcast_to(phase, (BODY_STEPS, 6)).copy()
    phase += offsets
    return phase, {
        "definition": "neutral common-frequency clock with seeded independent initial phases; no tripod offsets",
        "frequency_hz": CLOCK_HZ,
        "steps": BODY_STEPS,
        "ctrl_hz": CTRL_HZ,
        "random_phase_seed": DC2_RANDOM_PHASE_SEED,
        "initial_phase_rad": offsets.tolist(),
    }


def neutral_convergence(result: np.ndarray) -> dict:
    pair_difference = result[:, :, None] - result[:, None, :]
    tail = pair_difference[-max(1, len(result) // 3):]
    variance = np.var(tail, axis=0)
    upper = variance[np.triu_indices(6, 1)]
    steps = np.diff(result, axis=0)
    step_changes = np.diff(steps, axis=0)
    max_phase_step = float(np.abs(steps).max())
    max_step_change = float(np.abs(step_changes).max())
    max_pair_variance = float(upper.max())
    phase_locked = max_pair_variance < DC2_PHASE_LOCK_VARIANCE_RAD2
    bounded = max_phase_step <= CLEAN_MAX_PHASE_STEP_RAD and max_step_change <= CLEAN_MAX_STEP_CHANGE_RAD
    return {
        "max_pair_phase_difference_variance_rad2": max_pair_variance,
        "phase_lock_variance_threshold_rad2": DC2_PHASE_LOCK_VARIANCE_RAD2,
        "phase_locked": bool(phase_locked),
        "max_phase_step_rad": max_phase_step,
        "max_step_change_rad": max_step_change,
        "bounded": bool(bounded),
        "admissible": bool(phase_locked and bounded),
    }


def blind_calibration(matrices: dict[str, np.ndarray]) -> dict:
    source, source_record = neutral_phase_source()
    table = []
    for per_hop_steps in range(DC2_MAX_PER_HOP_STEPS + 1):
        delay = anatomical_delay_matrix(per_hop_steps)
        for gain in GAIN_CANDIDATES:
            conditions = {
                condition: neutral_convergence(delayed_coupled_phases(source, matrices[condition], gain, delay))
                for condition in ("real", "shuffled", "random")
            }
            table.append({
                "gain": float(gain),
                "per_hop_steps": per_hop_steps,
                "delay_matrix": delay.tolist(),
                "conditions": conditions,
                "admissible_all_conditions": bool(all(item["admissible"] for item in conditions.values())),
            })
    admissible = [row for row in table if row["admissible_all_conditions"]]
    selected = min(admissible, key=lambda row: (row["per_hop_steps"], -row["gain"])) if admissible else None
    return {
        "candidate_gains": list(GAIN_CANDIDATES),
        "candidate_per_hop_steps": list(range(DC2_MAX_PER_HOP_STEPS + 1)),
        "half_period_ceiling": {
            "clock_frequency_hz": CLOCK_HZ,
            "half_period_steps": CTRL_HZ / (2.0 * CLOCK_HZ),
            "max_per_hop_steps": DC2_MAX_PER_HOP_STEPS,
            "rule": "search through the nearest integer control-step half-period; no larger delay searched",
        },
        "source": source_record,
        "criterion": {
            "phase_locked": "maximum ordinary variance of every pairwise phase difference over final third is below threshold",
            "bounded": "maximum absolute phase step and step change reuse existing clean thresholds",
            "phase_lock_variance_threshold_rad2": DC2_PHASE_LOCK_VARIANCE_RAD2,
            "maximum_phase_step_rad": CLEAN_MAX_PHASE_STEP_RAD,
            "maximum_step_change_rad": CLEAN_MAX_STEP_CHANGE_RAD,
            "all_three_conditions_required": True,
        },
        "selection_rule": DC2_SELECTION_RULE,
        "conditions": ["real", "shuffled", "random"],
        "table": table,
        "admissible_cell_count": len(admissible),
        "selected": selected,
    }


def write_calibration(output: Path, matrices: dict, calibration: dict) -> None:
    result = {
        "method": "DC-2 connectome-blind delayed coupling calibration",
        "coupling": matrices,
        "calibration": calibration,
        "zero_learning": True,
        "gate_dc2": {
            "admissibility_table_complete": True,
            "selection_uses_only_neutral_criterion": True,
            "verdict": "GATE DC-2: PASS" if calibration["selected"] else "DC-2 NULL: no connectome-blind delayed regime",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = ("per_hop_steps", "gain", "admissible_all_conditions", "real_admissible", "shuffled_admissible", "random_admissible")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({
            "per_hop_steps": row["per_hop_steps"],
            "gain": row["gain"],
            "admissible_all_conditions": row["admissible_all_conditions"],
            **{f"{condition}_admissible": row["conditions"][condition]["admissible"] for condition in ("real", "shuffled", "random")},
        } for row in calibration["table"])


def run_dc2(data: Path, output: Path) -> int:
    matrices = build_coupling_matrices(data)
    calibration = blind_calibration(matrices["matrices"])
    write_calibration(output, matrices["record"], calibration)
    print(f"PRE-REGISTERED DC-2 SELECTION RULE: {DC2_SELECTION_RULE}")
    print(f"DC-2 admissible cells: {calibration['admissible_cell_count']}/{len(calibration['table'])}")
    print("DC-2 admissibility table:")
    for row in calibration["table"]:
        print(row["per_hop_steps"], row["gain"], row["admissible_all_conditions"], flush=True)
    if calibration["selected"] is None:
        print("DC-2 NULL: no connectome-blind delayed regime")
        return 0
    selected = calibration["selected"]
    print(f"GATE DC-2: PASS (gain={selected['gain']:.2f}, per_hop_steps={selected['per_hop_steps']})")
    print(f"dc2 json -> {output.with_suffix('.json')}")
    print(f"dc2 csv -> {output.with_suffix('.csv')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RESULTS)
    parser.add_argument("--phase", choices=("dc1", "dc2"), default="dc1")
    parser.add_argument("--out", type=Path, default=RESULTS / "delay_coupling_dc2")
    args = parser.parse_args()
    if args.phase == "dc1":
        dc1_self_check(args.data)
        return 0
    return run_dc2(args.data, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
