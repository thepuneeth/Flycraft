"""Path 3 fixed per-leg wrapper around the unchanged Path-1 gait primitive."""
from __future__ import annotations

import numpy as np

from hexapod_gait import HexapodGait, LEFT, TRIPOD_A, home_stance


STANCE = home_stance()
COXA_WORLD_SIGN = np.asarray([1.0 if leg in LEFT else -1.0 for leg in range(6)], dtype=np.float64)


def coxa_phase_signal(targets: np.ndarray) -> np.ndarray:
    """Undo mirrored joint-coordinate signs before applying the canonical phase metric."""
    values = np.asarray(targets, dtype=np.float64).reshape(-1, 6, 3)[..., 0]
    return values * COXA_WORLD_SIGN


def advance_leg(phase: float, swing: bool, forward: float, dt: float, primitive: HexapodGait) -> float:
    """Advance one fixed Path-1 half-cycle without changing its shape."""
    if dt < 0:
        raise ValueError("dt must be non-negative")
    forward = float(np.clip(forward, 0.0, 1.0))
    frequency = (primitive.base_freq + primitive.drive_freq * forward) if forward > 0.02 else 0.0
    phase = float(phase)
    if swing:
        phase = float(np.clip(phase, 0.0, np.pi))
        return min(np.pi, phase + 2.0 * np.pi * frequency * dt)
    phase = float(np.clip(phase, np.pi, 2.0 * np.pi))
    return min(2.0 * np.pi, phase + 2.0 * np.pi * frequency * dt)


def path1_leg_target(primitive: HexapodGait, phase: float, leg: int, forward: float, turn: float) -> np.ndarray:
    """Ask Path-1's own target function for one arbitrary leg phase."""
    primitive.phi = float(phase) - (np.pi if leg in TRIPOD_A else 0.0)
    target = primitive.targets(forward, turn, 0.0)
    return target[3 * leg:3 * leg + 3].copy()


class StepGaitEngine:
    """Per-leg state machine; neural input chooses half-cycle, Path 1 chooses shape."""

    def __init__(self, stance: np.ndarray | None = None):
        self.primitive = HexapodGait(stance=stance)
        self.phases = np.zeros(6, dtype=np.float64)
        self.states = np.zeros(6, dtype=bool)
        self.initialized = False

    def reset(self) -> None:
        self.phases.fill(0.0)
        self.states.fill(False)
        self.initialized = False

    def targets(self, swing_states: np.ndarray, forward: float, turn: float, dt: float) -> np.ndarray:
        states = np.asarray(swing_states, dtype=bool).reshape(-1)
        if states.shape != (6,):
            raise ValueError(f"swing state shape {states.shape}; expected (6,)")
        if not np.isfinite(float(forward)) or not np.isfinite(float(turn)):
            raise ValueError("forward and turn must be finite")
        targets = np.empty(18, dtype=np.float64)
        for leg, swing in enumerate(states):
            if not self.initialized or bool(swing) != bool(self.states[leg]):
                self.phases[leg] = 0.0 if swing else np.pi
            self.phases[leg] = advance_leg(self.phases[leg], bool(swing), forward, dt, self.primitive)
            targets[3 * leg:3 * leg + 3] = path1_leg_target(self.primitive, self.phases[leg], leg, forward, turn)
        self.states[:] = states
        self.initialized = True
        return targets.astype(np.float32)


def clean_tripod_states(phase: float) -> np.ndarray:
    """External ideal tripod clock used only for the P2 engine ceiling."""
    return np.asarray([np.sin(float(phase) + (np.pi if leg in TRIPOD_A else 0.0)) > 0.0 for leg in range(6)], dtype=bool)


def protocol() -> dict:
    primitive = HexapodGait(stance=STANCE)
    return {
        "source": "eon-fly-brain/code/hexapod_gait.py:HexapodGait",
        "stance": list(STANCE),
        "swing_amp": primitive.swing_amp,
        "lift_amp": primitive.lift_amp,
        "base_freq_hz": primitive.base_freq,
        "drive_freq_hz": primitive.drive_freq,
        "turn_scale": primitive.turn_scale,
        "phase_rule": "state transition starts swing at phase 0 or stance at phase pi; fixed Path-1 rate advances and clamps at half-cycle endpoint",
    }


def self_check() -> None:
    engine = StepGaitEngine()
    target = engine.targets(np.asarray([True, False, True, False, True, False]), 0.8, 0.0, 0.02)
    assert target.shape == (18,)
    assert np.isfinite(target).all()
    assert np.array_equal(engine.primitive.stance, STANCE)
    assert np.isclose(engine.primitive.swing_amp, 0.35)
    assert np.isclose(engine.primitive.lift_amp, 0.7)


if __name__ == "__main__":
    self_check()
    print(protocol())
