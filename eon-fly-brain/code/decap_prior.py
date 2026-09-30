"""Decaying action priors (DecAP) with a batched real-time motion generator.

[DA] Sood, Sun, Li, Sartoretti, "DecAP: Decaying Action Priors for Accelerated
     Imitation Learning of Torque-Based Legged Locomotion Policies", IROS 2024
     (arXiv:2310.05714). Code: github.com/marmotlab/decaying_action_priors
[K]  Kim, Choi, Choi, Yoon, Choi, "Deep Reinforcement Learning Control of a
     Hexapod Robot", Actuators 15(1):33 (2026). Applies [DA] to a hexapod, with
     the prior supplied by a Real-Time Motion Generator (RTMG) rather than a
     pretrained position policy: at each step the RTMG synthesises end-effector
     trajectories from commanded translational and yaw velocity, maps them to
     joint angles by inverse kinematics, and the result is used as a torque bias
     that is annealed away during training.

Why this is here
----------------
Every PPO run so far failed the same way: the policy never stayed upright long
enough for the locomotion reward to carry information, so the update had nothing
to climb. [DA] exists for exactly that failure -- it biases early exploration
toward a stable gait and then removes the bias.

Our RTMG is `hexapod_gait.HexapodGait`, which already does [K]'s job: joint
angles synthesised in real time from a commanded forward speed and yaw rate. The
class below is that same generator, vectorised over the environment axis, since
PPO steps 512 envs at once and a per-env Python object costs more than the
physics. `demo()` asserts it reproduces `HexapodGait.targets` exactly.

The constraint this touches, stated plainly
-------------------------------------------
A permanent teacher residual is on the forbidden list, and it should stay there.
This prior is not that: `prior_weight` returns EXACTLY 0.0 from `ZERO_AFTER`
onward, and evaluation runs at weight 0, so nothing the generator supplies
survives into the reported policy. It is a training curriculum, not an
inference-time assist. That is still a change to the agreed constraints and
needs the owner's sign-off, not an assumption.
"""

from __future__ import annotations

import numpy as np

from hexapod_gait import LEFT, resolve_phase_group, resolve_stance

# WHERE THIS DEPARTS FROM [DA]'s OWN CODE
# ---------------------------------------
# The reference implementation was read after this was written, and it differs on
# two points that a negative result here has to be reported against. Both are
# recorded rather than fixed: the PPO path they belong to is abandoned, and a
# faithful port would have to be re-run to say anything about [DA] itself.
#
# 1. WHERE THE PRIOR IS APPLIED. `yuna.py:78` (their hexapod, closest to ours):
#        torques = p_gains*(actions_scaled + default_dof_pos - dof_pos)
#                  - d_gains*dof_vel
#                  + decap*(p_gains*(dof_imit - dof_pos) - d_gains*dof_vel)
#    The prior is a SEPARATE PD torque carrying its own damping term, summed with
#    the policy's. `apply_prior` below instead offsets the position target that
#    one shared PD servos to, so the prior contributes no damping of its own and
#    its stiffness does not add. Their default `decap_torque` is further away
#    still -- the action there IS a torque, and ours is a joint angle.
#
# 2. WHAT THE EXPONENT COUNTS. `param_config.yaml` sets gamma = 0.99 and k = 100,
#    and `legged_robot.py:129` increments `torque_ref_decay_factor` once per
#    environment STEP, so the factor is gamma**(steps/k) -- at 24 steps per
#    iteration that is 0.99**(0.24 * iteration), which still sits at 0.146 by
#    iteration 800. Ours reads the exponent as the iteration count, so it anneals
#    roughly four times faster than theirs at the same nominal gamma.
GAMMA = 0.99
# Ours clamps to exactly zero at ZERO_AFTER, which [DA] do not do, so "the policy
# ran with no prior" is a property of the code rather than an argument about how
# small a small number is. That part is deliberately stricter than the reference.
ZERO_AFTER = 800


def prior_weight(iteration: int, gamma: float = GAMMA, zero_after: int = ZERO_AFTER) -> float:
    """Decay factor on the generator bias for a given policy iteration."""
    if iteration >= zero_after:
        return 0.0
    return float(gamma ** int(iteration))


def apply_prior(action: np.ndarray, teacher: np.ndarray, stance: np.ndarray,
                weight: float) -> np.ndarray:
    """Blend the generator bias into the executed action.

    The policy's action is already in ABSOLUTE joint-angle space: `_to_joints`
    applies the `stance + action_scale * deviation` anchor inside the policy, and
    the env PD-servos to whatever angles it is handed. So the bias is simply the
    generator's departure from stance, and at weight 1.0 with a policy sitting at
    stance the executed action is the generator's pose exactly.

    Only the EXECUTED action is biased. The action PPO stores, scores and
    replays is the policy's own, which is what [DA] does -- the decaying bias is
    part of the environment as far as the learner is concerned.
    """
    if weight == 0.0:
        return action
    return action + weight * (teacher - stance)


class BatchRTMG:
    """`hexapod_gait.HexapodGait` vectorised over the environment axis.

    Mirrors `HexapodGait.targets` term for term; `demo()` checks the two agree to
    floating-point tolerance on random commands.
    """

    def __init__(self, batch_size: int, stance=None, swing_amp: float = 0.35,
                 lift_amp: float = 0.7, base_freq: float = 0.8,
                 drive_freq: float = 0.5, turn_scale: float = 1.0,
                 phase_group=None) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.stance = resolve_stance(stance)                      # (6, 3)
        group = resolve_phase_group(phase_group)
        self.swing_amp, self.lift_amp = float(swing_amp), float(lift_amp)
        self.base_freq, self.drive_freq = float(base_freq), float(drive_freq)
        self.turn_scale = float(turn_scale)
        # Per-leg constants, broadcast against the (B, 6) phase array.
        self.offset = np.array([np.pi if leg in group else 0.0 for leg in range(6)])
        self.side = np.array([1.0 if leg in LEFT else -1.0 for leg in range(6)])
        self.is_left = np.array([leg in LEFT for leg in range(6)])
        self.phi = np.zeros(self.batch_size, dtype=np.float64)

    def reset(self, done: np.ndarray | None = None) -> None:
        """Restart the gait cycle for environments that just reset."""
        if done is None:
            self.phi.fill(0.0)
            return
        mask = np.asarray(done, dtype=bool).reshape(-1)
        if mask.shape != (self.batch_size,):
            raise ValueError(f"done has shape {mask.shape}; expected ({self.batch_size},)")
        self.phi[mask] = 0.0

    def targets(self, forward: np.ndarray, turn: np.ndarray, dt: float) -> np.ndarray:
        """Advance the generator one control step and return (B, 18) joint angles."""
        forward = np.clip(np.asarray(forward, dtype=np.float64).reshape(-1), 0.0, 1.0)
        turn = np.asarray(turn, dtype=np.float64).reshape(-1)
        if forward.shape != (self.batch_size,) or turn.shape != (self.batch_size,):
            raise ValueError("forward and turn must both have length batch_size")

        freq = np.where(forward > 0.02, self.base_freq + self.drive_freq * forward, 0.0)
        self.phi = (self.phi + 2.0 * np.pi * freq * dt) % (2.0 * np.pi)

        phase = self.phi[:, None] + self.offset[None, :]                   # (B, 6)
        bias = np.clip(turn / self.turn_scale, -0.9, 0.9)[:, None]         # (B, 1)
        # Inside legs take a shorter stride; that differential is the turn.
        side_gain = np.where(self.is_left[None, :], 1.0 - bias, 1.0 + bias)
        sin_phase = np.sin(phase)

        coxa = self.stance[:, 0][None, :] - (
            self.side[None, :] * self.swing_amp * side_gain * forward[:, None] * sin_phase
        )
        femur = self.stance[:, 1][None, :] + (
            self.side[None, :] * self.lift_amp * np.maximum(0.0, sin_phase)
        )
        tibia = np.broadcast_to(self.stance[:, 2][None, :], coxa.shape)
        return np.stack((coxa, femur, tibia), axis=-1).reshape(self.batch_size, 18)


def commands_from_observation(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read the RTMG's commanded velocity out of the observation's target block.

    Channels 9..11 are (cos bearing, sin bearing, clipped distance) -- see
    `embody_env_v4.HexapodTargetEnv._observation`. [K]'s RTMG is driven by
    commanded translational and yaw velocity; ours walks at full speed and steers
    by the egocentric bearing to the prey, which is the same two commands read
    off the task instead of a joystick.
    """
    observation = np.asarray(observation, dtype=np.float64)
    if observation.ndim != 2 or observation.shape[1] < 12:
        raise ValueError(f"observation has shape {observation.shape}; expected (B, >=12)")
    bearing = np.arctan2(observation[:, 10], observation[:, 9])
    forward = np.ones(observation.shape[0])
    return forward, bearing


def demo() -> None:
    from hexapod_gait import HexapodGait

    # Schedule: decays, and is exactly zero -- not merely small -- at the cutoff.
    assert prior_weight(0) == 1.0
    assert prior_weight(ZERO_AFTER) == 0.0 and prior_weight(ZERO_AFTER + 400) == 0.0
    assert prior_weight(700) < prior_weight(300) < prior_weight(1) < 1.0
    assert prior_weight(700) < 1e-2, prior_weight(700)

    # The batched generator must be the published one, not a lookalike.
    rng = np.random.default_rng(0)
    batch = 5
    batched = BatchRTMG(batch)
    singles = [HexapodGait() for _ in range(batch)]
    dt = 0.02
    worst = 0.0
    for _ in range(50):
        forward = rng.uniform(0.0, 1.0, batch)
        turn = rng.uniform(-1.5, 1.5, batch)
        ours = batched.targets(forward, turn, dt)
        theirs = np.stack([
            gait.targets(forward[i], turn[i], dt) for i, gait in enumerate(singles)
        ])
        worst = max(worst, float(np.abs(ours - theirs).max()))
    assert worst < 1e-12, f"batched RTMG diverges from HexapodGait by {worst:.3g}"

    # Resetting one environment restarts only that gait cycle.
    done = np.zeros(batch, dtype=bool)
    done[2] = True
    before = batched.phi.copy()
    batched.reset(done)
    assert batched.phi[2] == 0.0
    assert np.array_equal(batched.phi[~done], before[~done])

    # At weight 0 the action is untouched; that is what evaluation runs.
    stance = resolve_stance().reshape(18)
    action = rng.normal(size=(batch, 18)) * 0.1 + stance
    assert np.array_equal(apply_prior(action, np.zeros((batch, 18)), stance, 0.0), action)
    # At weight 1 a policy sitting at stance executes the generator's pose exactly.
    teacher = batched.targets(np.ones(batch), np.zeros(batch), dt)
    at_stance = np.broadcast_to(stance, (batch, 18)).copy()
    assert np.allclose(apply_prior(at_stance, teacher, stance, 1.0), teacher)

    forward, bearing = commands_from_observation(
        np.tile(np.concatenate(([0.0] * 9, [np.cos(0.4), np.sin(0.4), 0.5], [0.0] * 54)), (batch, 1))
    )
    assert np.allclose(bearing, 0.4) and np.allclose(forward, 1.0)

    print("decap_prior: all checks passed")


if __name__ == "__main__":
    demo()
