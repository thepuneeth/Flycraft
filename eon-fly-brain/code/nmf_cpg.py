"""NeuroMechFly v2's CPG, ported to the hexapod as a REFERENCE arm.

[NMF] Wang-Chen, Stimpfling, Lam, Ozdil, Genoud, Hurtak, Ramdya,
      "NeuroMechFly v2: simulating embodied sensorimotor control in adult
      Drosophila", Nature Methods 21:2353-2362 (2024).
      Source vendored at fly_field_refs/flygym and fly_field_refs/nmf2-paper;
      the oscillator, the tripod phase-bias matrix and the default parameters
      below are transcribed from
      flygym/src/flygym_demo/complex_terrain/cpg_controller.py.

WHAT THIS IS FOR, AND WHAT IT IS NOT
------------------------------------
This is a hand-built central pattern generator. It supplies the rhythm and the
inter-leg coordination outright -- which is precisely the thing E0 says our
frozen connectome does not produce. So a walk produced here is NOT evidence
about the connectome, and this must never be mixed into a connectome claim.

It is worth having as a REFERENCE: it measures what this body does under a
published controller, which separates "the body or the reward is the problem"
from "the connectome is the problem". Report it as a labelled ceiling beside the
zero-training numbers, never as an arm of the same experiment.

WHAT TRANSFERS AND WHAT DOES NOT
--------------------------------
The oscillator network transfers exactly: it is morphology-independent, six
phases and a coupling matrix, so [NMF]'s parameters carry over unchanged.

[NMF]'s preprogrammed leg kinematics do NOT transfer, and this file deliberately
does not fake it. Their steps are measured on a fly leg with SEVEN degrees of
freedom (thorax-coxa pitch/roll/yaw, coxa-femur pitch/roll, femur-tibia pitch,
tibia-tarsus pitch; preprogrammed.py:29-37). The PhantomX leg has three. Any
7-to-3 retarget is a design decision of ours, which would defeat the point of
borrowing a published controller, so the swing shape here stays the one
hexapod_gait already uses and only the timing and coordination come from [NMF].
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from hexapod_gait import LEFT, resolve_phase_group, resolve_stance

# [NMF] make_tripod_cpg_network defaults, cpg_controller.py:166-180.
INTRINSIC_FREQUENCY_HZ = 12.0
INTRINSIC_AMPLITUDE = 1.0
COUPLING_STRENGTH = 10.0
CONVERGENCE_COEF = 20.0

# [NMF] preprogrammed.py:47-57, 74-80.  The local copy is used on the pod;
# the vendored reference copy is the laptop fallback for the math self-check.
STEP_ASSET = "single_steps_untethered.pkl"
FLY_LEG_NAMES = ("lf", "rf", "lm", "rm", "lh", "rh")
LEGACY_LEG_NAMES = {leg: leg.upper() for leg in FLY_LEG_NAMES}
LEGACY_DOF_NAMES = (
    "Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll", "Tibia", "Tarsus1"
)
# [TRANSLATION] fixed role mapping: protraction/retraction, elevation/depression,
# and extension/flexion.  Names, not hand-picked array positions, define indices.
HEX_ROLE_TO_FLY_DOF = {"coxa": "Coxa", "femur": "Femur", "tibia": "Tibia"}
HEX_ROLE_INDICES = tuple(LEGACY_DOF_NAMES.index(HEX_ROLE_TO_FLY_DOF[name]) for name in ("coxa", "femur", "tibia"))
RIGHT_SIGN_FLIPPED_DOFS = frozenset(("Coxa_roll", "Coxa_yaw", "Femur_roll"))
V1_AMPLITUDE_BIAS = 0.5       # affine midpoint of [NMF] intrinsic amplitude range [0, 1]
V1_AMPLITUDE_SCALE = 0.5      # maps a normalised signed leg score into [0, 1]
TRANSLATED_PHASE_DIRECTION = -1.0  # fixed body/source phase convention from bug loop


def tripod_phase_biases(phase_group=None) -> np.ndarray:
    """[NMF] get_cpg_biases("tripod"), expressed in this body's leg indexing.

    [NMF] hard-codes a parity checkerboard times pi, which is a tripod under
    THEIR leg order. Ours is not their order -- hexapod_gait's tripod is
    {0, 3, 4} against {1, 2, 5} -- so the matrix is rebuilt from the same rule
    (0 within a tripod, pi across) rather than copied index for index. Copying
    the literal array would silently prescribe a different gait.
    """
    group = resolve_phase_group(phase_group)
    same = np.array([[(i in group) == (j in group) for j in range(6)] for i in range(6)])
    return np.where(same, 0.0, np.pi)


class NMFCPGNetwork:
    """[NMF] CPGNetwork: coupled phase-amplitude oscillators, Euler integrated.

        dtheta_i/dt = 2*pi*nu_i + sum_j r_j w_ij sin(theta_j - theta_i - phi_ij)
        dr_i/dt     = alpha_i (R_i - r_i)

    Transcribed from cpg_controller.py:12-27 and 80-92.
    """

    def __init__(self, timestep: float, phase_biases: np.ndarray | None = None,
                 intrinsic_frequency: float = INTRINSIC_FREQUENCY_HZ,
                 intrinsic_amplitude: float = INTRINSIC_AMPLITUDE,
                 coupling_strength: float = COUPLING_STRENGTH,
                 convergence_coef: float = CONVERGENCE_COEF,
                 seed: int = 0) -> None:
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        self.timestep = float(timestep)
        custom_phase_biases = phase_biases is not None
        self.phase_biases = tripod_phase_biases() if phase_biases is None else np.asarray(phase_biases, float)
        if self.phase_biases.shape != (6, 6):
            raise ValueError(f"phase_biases has shape {self.phase_biases.shape}; expected (6, 6)")
        if not np.isfinite(self.phase_biases).all():
            raise ValueError("phase_biases must be finite")
        self.intrinsic_freqs = np.full(6, float(intrinsic_frequency))
        self.intrinsic_amps = np.full(6, float(intrinsic_amplitude))
        # [NMF] tripod keeps its published cross-group mask. V3 supplies a
        # measured phase for every off-diagonal pair, so all such pairs couple.
        self.coupling_weights = (
            (self.phase_biases > 0) if not custom_phase_biases
            else ~np.eye(6, dtype=bool)
        ) * float(coupling_strength)
        self.convergence_coefs = np.full(6, float(convergence_coef))
        self.random_state = np.random.RandomState(seed)
        self.reset()

    def reset(self, init_phases=None, init_magnitudes=None) -> None:
        self.curr_phases = (self.random_state.random(6) * 2 * np.pi
                            if init_phases is None else np.asarray(init_phases, float).copy())
        self.curr_magnitudes = (np.zeros(6) if init_magnitudes is None
                                else np.asarray(init_magnitudes, float).copy())

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        theta, r = self.curr_phases, self.curr_magnitudes
        phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        coupling = (r * self.coupling_weights * np.sin(phase_diff - self.phase_biases)).sum(axis=1)
        dtheta = 2 * np.pi * self.intrinsic_freqs + coupling
        dr = self.convergence_coefs * (self.intrinsic_amps - r)
        self.curr_phases = theta + dtheta * self.timestep
        self.curr_magnitudes = r + dr * self.timestep
        return self.curr_phases.copy(), self.curr_magnitudes.copy()


class NMFCPGGait:
    """[NMF] timing driving this body's swing shape.

    The joint mapping is hexapod_gait's, unchanged: coxa swings with sin(phase),
    femur lifts on the positive half. Only WHEN each leg does that -- the phase
    of each oscillator and the coupling that holds them a half cycle apart --
    comes from [NMF].
    """

    def __init__(self, timestep: float, stance=None, swing_amp: float = 0.35,
                 lift_amp: float = 0.7, seed: int = 0, **cpg_kwargs) -> None:
        self.stance = resolve_stance(stance)
        self.swing_amp, self.lift_amp = float(swing_amp), float(lift_amp)
        self.network = NMFCPGNetwork(timestep=timestep, seed=seed, **cpg_kwargs)
        self.side = np.array([1.0 if leg in LEFT else -1.0 for leg in range(6)])

    def targets(self, amplitude: float = 1.0) -> np.ndarray:
        phases, magnitudes = self.network.step()
        gain = np.clip(magnitudes, 0.0, None) * float(amplitude)
        sin_phase = np.sin(phases)
        coxa = self.stance[:, 0] - self.side * self.swing_amp * gain * sin_phase
        femur = self.stance[:, 1] + self.side * self.lift_amp * np.maximum(0.0, sin_phase) * gain
        tibia = self.stance[:, 2]
        return np.stack((coxa, femur, tibia), axis=-1).reshape(18)


def default_step_asset() -> Path:
    """Find the deployed asset, then the vendored reference asset."""
    here = Path(__file__).resolve()
    candidates = (
        here.parent.parent / "data" / "nmf" / STEP_ASSET,
        here.parents[2] / "fly_field_refs" / "flygym" / "src" /
        "flygym_demo" / "complex_terrain" / "assets" / STEP_ASSET,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find " + STEP_ASSET + " in deployed data or vendored reference")


class PreprogrammedHexapodSteps:
    """Retarget [NMF]'s recorded fly steps to the PhantomX's three joints.

    [NMF] preprogrammed.py:40-116 loads seven recorded fly-leg angles and
    interpolates each with a periodic cubic spline.  This port keeps that
    spline and right-leg roll/yaw convention, then makes one fixed, declared
    role mapping to coxa/femur/tibia.  Source excursions retain their recorded
    angular size and are clipped by each target joint's available MuJoCo range
    around the author-declared home stance.  No amplitude is fitted to
    locomotion.

    Adhesion is intentionally omitted: the PhantomX model has no adhesion
    actuators.  ``swing_period`` is retained for diagnostics and phase checks.
    """

    leg_names = FLY_LEG_NAMES
    legacy_dof_names = LEGACY_DOF_NAMES
    role_to_fly_dof = HEX_ROLE_TO_FLY_DOF

    def __init__(self, path: str | Path | None = None, stance=None,
                 joint_ranges: np.ndarray | None = None) -> None:
        path = default_step_asset() if path is None else Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            data = pickle.load(handle)

        self.path = path
        self._length = len(data["joint_LFCoxa"])
        self._timestep = float(data["meta"]["timestep"])
        self.duration = self._length * self._timestep
        if self._length < 2 or self._timestep <= 0 or self.duration <= 0:
            raise ValueError("recorded step asset has invalid length or timestep")

        self.stance = np.asarray(resolve_stance(stance), dtype=np.float64)
        if self.stance.shape != (6, 3) or not np.isfinite(self.stance).all():
            raise ValueError(f"stance has shape {self.stance.shape}; expected (6, 3)")
        if joint_ranges is None:
            raise ValueError("joint_ranges must come from the MuJoCo model")
        self.joint_ranges = np.asarray(joint_ranges, dtype=np.float64)
        if self.joint_ranges.shape != (6, 3, 2) or not np.isfinite(self.joint_ranges).all():
            raise ValueError(f"joint_ranges has shape {self.joint_ranges.shape}; expected (6, 3, 2)")
        low, high = self.joint_ranges[..., 0], self.joint_ranges[..., 1]
        if np.any(low >= high) or np.any(self.stance < low) or np.any(self.stance > high):
            raise ValueError("stance must lie inside every MuJoCo joint range")

        # [NMF] preprogrammed.py:78-96, including its periodic endpoint and
        # right-leg roll/yaw sign correction.
        phase_grid = np.linspace(0.0, 2.0 * np.pi, self._length)
        self._psi_funcs: list[CubicSpline] = []
        neutral = np.empty((6, len(LEGACY_DOF_NAMES)), dtype=np.float64)
        source_peak = np.empty((6, 3), dtype=np.float64)
        source_indices = np.asarray(HEX_ROLE_INDICES, dtype=np.int64)
        for leg_idx, leg in enumerate(self.leg_names):
            legacy_leg = LEGACY_LEG_NAMES[leg]
            angles = np.asarray(
                [data[f"joint_{legacy_leg}{dof}"] for dof in LEGACY_DOF_NAMES],
                dtype=np.float64,
            )
            if angles.shape != (len(LEGACY_DOF_NAMES), self._length) or not np.isfinite(angles).all():
                raise ValueError(f"invalid recorded angles for {leg}: {angles.shape}")
            if leg.startswith("r"):
                for dof_idx, dof in enumerate(LEGACY_DOF_NAMES):
                    if dof in RIGHT_SIGN_FLIPPED_DOFS:
                        angles[dof_idx] *= -1.0
            if not np.allclose(angles[:, 0], angles[:, -1]):
                raise ValueError(f"recorded step for {leg} is not periodic")
            spline = CubicSpline(phase_grid, angles, axis=1, bc_type="periodic")
            self._psi_funcs.append(spline)
            neutral[leg_idx] = spline(np.pi)
            offsets = spline(phase_grid)[source_indices] - neutral[leg_idx, source_indices, None]
            source_peak[leg_idx] = np.max(np.abs(offsets), axis=1)
        if np.any(source_peak <= np.finfo(np.float64).eps):
            raise ValueError("recorded step has zero excursion in a mapped joint")
        self._neutral_source = neutral
        self._source_peak = source_peak
        self.swing_period = np.empty((6, 2), dtype=np.float64)
        stance_durations = data["swing_stance_time"]["stance"]
        for leg_idx, leg in enumerate(self.leg_names):
            duration = float(stance_durations[LEGACY_LEG_NAMES[leg]])
            if not 0.0 <= duration <= self.duration:
                raise ValueError(f"invalid stance duration for {leg}: {duration}")
            # [NMF] preprogrammed.py:102-116: phase 0 starts swing; the
            # recorded stance duration determines swing_end.
            swing_end = duration / self.duration * 2.0 * np.pi
            if TRANSLATED_PHASE_DIRECTION < 0.0:
                self.swing_period[leg_idx] = (2.0 * np.pi - swing_end, 2.0 * np.pi)
            else:
                self.swing_period[leg_idx] = (0.0, swing_end)

    @classmethod
    def from_mujoco_model(cls, model, path: str | Path | None = None, stance=None):
        """Construct ranges directly from named hinge joints in ``model``."""
        import mujoco

        joint_ids = np.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"leg_{leg}_joint_{joint}")
            for leg in range(6) for joint in range(3)
        ], dtype=np.int64)
        if np.any(joint_ids < 0):
            raise ValueError("hexapod model is missing one or more named leg joints")
        ranges = model.jnt_range[joint_ids].reshape(6, 3, 2)
        if stance is None:
            stance = np.asarray(model.key("home").qpos[7:25], dtype=np.float64).reshape(6, 3)
        return cls(path=path, stance=stance, joint_ranges=ranges)

    @property
    def step_cycle_frequency_hz(self) -> float:
        return 1.0 / self.duration

    def get_joint_angles(self, leg: int | str, phase: float | np.ndarray,
                         magnitude: float | np.ndarray = 1.0) -> np.ndarray:
        """Return one retargeted leg's three angles, preserving phase shape."""
        if isinstance(leg, str):
            try:
                leg_idx = self.leg_names.index(leg.lower())
            except ValueError as exc:
                raise ValueError(f"unknown leg {leg!r}; expected {self.leg_names}") from exc
        else:
            leg_idx = int(leg)
            if leg_idx not in range(6):
                raise ValueError(f"leg index {leg_idx} outside 0..5")
        phase, magnitude = np.broadcast_arrays(np.asarray(phase, dtype=np.float64),
                                               np.asarray(magnitude, dtype=np.float64))
        if not np.isfinite(phase).all() or not np.isfinite(magnitude).all():
            raise ValueError("phase and magnitude must be finite")
        source_phase = TRANSLATED_PHASE_DIRECTION * phase
        values = np.asarray(self._psi_funcs[leg_idx](source_phase % (2.0 * np.pi)), dtype=np.float64)
        indices = np.asarray(HEX_ROLE_INDICES, dtype=np.int64)
        neutral = self._neutral_source[leg_idx, indices]
        if phase.ndim == 0:
            offset = values[indices] - neutral
            normalised = offset / self._source_peak[leg_idx]
            low, high = self.joint_ranges[leg_idx, :, 0], self.joint_ranges[leg_idx, :, 1]
            room = np.where(normalised >= 0.0, high - self.stance[leg_idx],
                            self.stance[leg_idx] - low)
            excursion = np.minimum(room, self._source_peak[leg_idx])
            result = self.stance[leg_idx] + np.clip(magnitude, 0.0, 1.0) * np.clip(normalised, -1.0, 1.0) * excursion
            return np.clip(result, low, high)
        offset = values[indices] - neutral[:, None]
        normalised = offset / self._source_peak[leg_idx, :, None]
        low, high = self.joint_ranges[leg_idx, :, 0], self.joint_ranges[leg_idx, :, 1]
        room = np.where(normalised >= 0.0, high[:, None] - self.stance[leg_idx, :, None],
                        self.stance[leg_idx, :, None] - low[:, None])
        excursion = np.minimum(room, self._source_peak[leg_idx, :, None])
        result = self.stance[leg_idx, :, None] + np.clip(magnitude, 0.0, 1.0) * np.clip(normalised, -1.0, 1.0) * excursion
        result = np.clip(result, low[:, None], high[:, None])
        return result

    def targets(self, phases: np.ndarray, magnitudes: np.ndarray | None = None) -> np.ndarray:
        phases = np.asarray(phases, dtype=np.float64).reshape(-1)
        if phases.shape != (6,) or not np.isfinite(phases).all():
            raise ValueError(f"phases have shape {phases.shape}; expected six finite values")
        if magnitudes is None:
            magnitudes = np.ones(6, dtype=np.float64)
        magnitudes = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
        if magnitudes.shape != (6,) or not np.isfinite(magnitudes).all():
            raise ValueError(f"magnitudes have shape {magnitudes.shape}; expected six finite values")
        return np.stack([
            self.get_joint_angles(leg, phases[leg], magnitudes[leg])
            for leg in range(6)
        ], axis=0).reshape(18)

    def is_swing(self, leg: int, phase: float) -> bool:
        phase = float(phase) % (2.0 * np.pi)
        start, end = self.swing_period[int(leg)]
        return bool(start < phase < end)


def v1_amplitudes(leg_scores: np.ndarray, rate_ceiling_hz: float,
                  scale: float = V1_AMPLITUDE_SCALE) -> np.ndarray:
    """[SH] rate-normalised affine leg-score -> NMF amplitude mapping.

    ``leg_scores`` is the fixed ``audit_vnc_legs.leg_scores`` bridge.  Missing
    pools are zero-filled, then the signed five-joint mean is converted to Hz
    and divided by the core's own ceiling before the declared affine map.
    """
    scores = np.asarray(leg_scores, dtype=np.float64)
    if scores.shape != (6, 5) or np.isinf(scores).any():
        raise ValueError(f"leg_scores have shape {scores.shape}; expected (6, 5)")
    ceiling = float(rate_ceiling_hz)
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        raise ValueError("rate_ceiling_hz must be positive and finite")
    if not np.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    signed_mean = np.nan_to_num(scores, nan=0.0).mean(axis=1)
    score_hz = signed_mean * ceiling
    normalised_score = score_hz / ceiling
    return np.clip(V1_AMPLITUDE_BIAS + float(scale) * normalised_score, 0.0, 1.0)


class NMFCPGPreprogrammedGait:
    """NMF oscillator plus recorded-step retargeter; V1 is amplitude-only."""

    def __init__(self, timestep: float, joint_ranges: np.ndarray,
                 stance=None, step_path: str | Path | None = None, seed: int = 0,
                 **cpg_kwargs) -> None:
        self.steps = PreprogrammedHexapodSteps(path=step_path, stance=stance,
                                               joint_ranges=joint_ranges)
        self.network = NMFCPGNetwork(timestep=timestep, seed=seed, **cpg_kwargs)

    def targets(self, amplitudes: np.ndarray | None = None) -> np.ndarray:
        if amplitudes is not None:
            amplitudes = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
            if amplitudes.shape != (6,) or not np.isfinite(amplitudes).all():
                raise ValueError("amplitudes must contain six finite values")
            self.network.intrinsic_amps = np.clip(amplitudes, 0.0, 1.0)
        phases, magnitudes = self.network.step()
        return self.steps.targets(phases, magnitudes)

    def targets_from_v1(self, leg_scores: np.ndarray, rate_ceiling_hz: float,
                        scale: float = V1_AMPLITUDE_SCALE) -> np.ndarray:
        amplitudes = v1_amplitudes(leg_scores, rate_ceiling_hz, scale)
        return self.targets(amplitudes)


def demo() -> None:
    """Self-check: the ported network must actually lock into a tripod."""
    biases = tripod_phase_biases()
    group = resolve_phase_group(None)
    for i in range(6):
        for j in range(6):
            expected = 0.0 if ((i in group) == (j in group)) else np.pi
            assert biases[i, j] == expected, (i, j, biases[i, j])
    # Diagonal is zero and the matrix is symmetric, as [NMF]'s tripod array is.
    assert np.allclose(np.diag(biases), 0.0)
    assert np.allclose(biases, biases.T)

    # From random phases, the coupling must converge to the prescribed tripod:
    # legs inside a group in phase, legs across it half a cycle apart.
    net = NMFCPGNetwork(timestep=1e-4, seed=0)
    for _ in range(20_000):
        net.step()
    phases = net.curr_phases
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            delta = (phases[j] - phases[i]) % (2 * np.pi)
            target = 0.0 if ((i in group) == (j in group)) else np.pi
            error = min(abs(delta - target), abs(delta - target - 2 * np.pi))
            assert error < 0.15, f"legs {i},{j} settled {delta:.3f} rad, wanted {target}"
    assert np.allclose(net.curr_magnitudes, INTRINSIC_AMPLITUDE, atol=1e-3), net.curr_magnitudes

    # And the joint output has to be a real 18-vector that moves.
    # [TEST] explicit synthetic stance keeps this check pure NumPy; production
    # callers pass the MuJoCo model's home stance/ranges.
    gait = NMFCPGGait(timestep=1e-3, stance=np.zeros((6, 3)), seed=0)
    frames = np.stack([gait.targets() for _ in range(500)])
    assert frames.shape == (500, 18)
    assert np.isfinite(frames).all()
    assert frames.std(axis=0).max() > 0.05, "the ported CPG is not moving any joint"
    test_ranges = np.broadcast_to(np.asarray([[-1.0, 1.0]]), (6, 3, 2)).copy()
    recorded = PreprogrammedHexapodSteps(
        path=default_step_asset(), stance=np.zeros((6, 3)), joint_ranges=test_ranges
    )
    assert recorded.swing_period.shape == (6, 2)
    if TRANSLATED_PHASE_DIRECTION < 0.0:
        assert (recorded.swing_period[:, 0] > 0.0).all()
        assert np.isclose(recorded.swing_period[:, 1], 2.0 * np.pi).all()
    else:
        assert np.isclose(recorded.swing_period[:, 0], 0.0).all()
    mapped = recorded.targets(np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False))
    assert mapped.shape == (18,)
    assert np.isfinite(mapped).all()
    assert (mapped.reshape(6, 3) >= -1.0).all() and (mapped.reshape(6, 3) <= 1.0).all()
    assert np.allclose(v1_amplitudes(np.zeros((6, 5)), 250.0), V1_AMPLITUDE_BIAS)
    recorded_gait = NMFCPGPreprogrammedGait(
        timestep=1e-3, stance=np.zeros((6, 3)), joint_ranges=test_ranges, seed=0
    )
    recorded_frames = np.stack([
        recorded_gait.targets_from_v1(np.zeros((6, 5)), 250.0) for _ in range(500)
    ])
    assert recorded_frames.shape == (500, 18)
    assert np.isfinite(recorded_frames).all()
    assert recorded_frames.std(axis=0).max() > 0.01, "recorded-step retargeter is not moving"
    print(f"nmf_cpg: all checks passed (nu={INTRINSIC_FREQUENCY_HZ} Hz, "
          f"w={COUPLING_STRENGTH}, alpha={CONVERGENCE_COEF}, "
          f"recorded_cycle={recorded.step_cycle_frequency_hz:.3f} Hz, tripod {sorted(group)})")


if __name__ == "__main__":
    demo()
