"""DC-1 delayed phase coupling checks; later phases run from this module."""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RESULTS)
    args = parser.parse_args()
    dc1_self_check(args.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
