"""Goal-directed hexapod env (v4): walk to a target object.

This is a near-verbatim port of the reward stack that is known to make legged
robots walk, applied to our body. v2 and v3 both invented their own objective and
both were gamed:

  v2  reward = unbounded forward displacement          -> the dive (12/12 falls)
  v3  reward = tolerance(instantaneous speed) * upright -> the rock (0.05 m in 8 s)

Neither failure was a connectome result. Both were reward-specification bugs, and
both were mine. v4 stops inventing. Sources, and what is taken from each:

  [LG] leggedrobotics/legged_gym, legged_gym/envs/base/legged_robot.py and
       legged_robot_config.py (Rudin et al., CoRL 2022). Every term below marked
       [LG] is that file's `_reward_*` method transcribed to numpy, with that
       file's default weight from `LeggedRobotCfg.rewards.scales`. Also taken:
       heading-mode command generation, `only_positive_rewards` clipping, and the
       `scale *= dt` convention, so the weights here are literally theirs.
       This is the recipe Kim et al. (Actuators 15(1):33, 2026) use for an 18-DoF
       hexapod with PPO, i.e. our exact morphology.

  [FB] google-deepmind/flybody, flybody/tasks/vision_flight.py (Vaxenburg et al.,
       Nature 643:1312, 2025). The bounded `rewards.tolerance` ride-height factor
       is theirs. Kept from v3 because it is the term that stopped the diving.

The two differ on combination: [FB] multiplies bounded factors, [LG] sums weighted
terms and clips the total at zero. v3 used the [FB] product and produced a rocker,
because a product of two factors that oscillation can both satisfy is still
satisfiable by oscillation. v4 uses the [LG] sum. The term that actually forbids
rocking is not the combination rule -- it is `feet_air_time`, which pays only on a
foot's first contact after leaving the ground, so a policy that never unloads a
foot earns nothing from it.

What is new here and copied from nobody: the target object. `commands.heading` in
[LG] is sampled at random each episode; here it is the bearing to the prey sphere
that already exists in the hexapod XML. Reaching it pays a bonus and respawns it.
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
from dm_rewards import tolerance  # noqa: E402
from hexapod_gait import H_MIN, XML, home_stance, pd_torque  # noqa: E402

# --- task geometry -----------------------------------------------------------
TARGET_SPEED_MPS = 0.1          # matches the Stage-1 teacher's speed_mps
# Reference yaw rate, the angular counterpart of TARGET_SPEED_MPS. The heading law
# emits clip(0.5 * bearing, +/-1) and teacher_v4's widest command is 0.4 rad/s,
# described there as "a target sitting 0.8 rad off the nose -- a normal approach
# angle". So 0.4, not 1.0, is the magnitude this task's yaw commands actually have.
TARGET_YAW_RATE_RPS = 0.4
# Target respawn annulus, measured body-centre to target.
#
# These were 0.4-1.0 m with a 0.15 m reach radius, and that made the task
# reachable without walking. Measured on this model: the feet sit 0.259-0.284 m
# from the body centre at stance, so the FOOTPRINT WAS WIDER THAN THE REACH ZONE.
# A target at the 0.4 m inner edge needed only 0.25 m of body travel -- less than
# one footprint diameter -- and the prey sphere is a ground-level ball (radius
# 0.07 at z=0.07), so the cheapest way to close the last stretch was to extend a
# leg toward it rather than walk. That is what run 2's video shows.
#
# The inner edge is now 3.5x the footprint radius, so no amount of leg extension
# substitutes for locomotion: the body itself has to cover at least 0.80 m.
TARGET_SPAWN_MIN_M = 1.0
TARGET_SPAWN_MAX_M = 1.8
# Slightly inside the footprint, so "reached" means the body is over the target
# rather than merely near it.
TARGET_REACH_M = 0.20
# One-shot, deliberately not dt-scaled. Sized against the per-step terms: an
# episode of standing still banks ~12 return (see smoke_test), so a single reach
# has to be worth more than that or the task term is decoration.
TARGET_REACH_BONUS = 20.0
# 30 s at 50 Hz, which is 3.2 m of travel at the teacher's measured 0.106 m/s --
# enough for the far edge of the annulus (1.8 m) plus a respawn. This was 600
# steps (12 s, 1.2 m), which could not have reached the new far edge at all.
# Episode length is free: PPO rollout segments are PPO_ROLLOUT_STEPS regardless,
# so this changes how often environments reset, not the cost of an iteration.
EPISODE_STEPS = 1500
OBS_TARGET_DIST_SCALE = 2.0     # divides the distance channel into ~[0, 1]
# [LG] LeggedRobotCfg.normalization.obs_scales, verbatim.
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
OBS_CLIP = 100.0                # [LG] normalization.clip_observations

# --- reward constants --------------------------------------------------------
# [LG] LeggedRobotCfg.rewards.tracking_sigma = 0.25, tuned for ~1 m/s commands.
# Our command is 0.1 m/s. The tracking term is exp(-squared velocity error/sigma),
# so holding the same relative sensitivity means scaling sigma by the square of
# the speed ratio: 0.25 * (0.1/1.0)^2. Without this rescale a policy standing
# still would score exp(-0.01/0.25) = 0.96 and rocking would be free again.
TRACKING_SIGMA_LIN = 0.25 * (TARGET_SPEED_MPS / 1.0) ** 2   # 0.0025
# The same rescale, for the same reason, on the angular term. v4 asserted that
# "yaw-rate commands stay on [LG]'s own scale (order 1 rad/s), so sigma is theirs"
# and left sigma at 0.25. Measured, that is the single largest free lunch in the
# reward: a body holding its stance, never turning, scored 0.790 on this term --
# 96% of everything a motionless policy earned -- because at sigma 0.25 a 0.24
# rad/s command is well inside the tolerance, so not turning reads as tracking.
# Rescaled by the square of the command-magnitude ratio, exactly as the linear
# term is, the same body scores 0.229.
TRACKING_SIGMA_ANG = 0.25 * (TARGET_YAW_RATE_RPS / 1.0) ** 2   # 0.04
# [LG] _reward_feet_air_time subtracts 0.5 s, calibrated for ANYmal's stride. A
# hexapod at ~2 Hz swings a leg for ~0.25 s, so the same term with 0.5 would be
# negative for every stride this body can physically produce.
# ponytail: 0.2 s is a stride-rate guess, not a measurement; retune from the
# realised duty factor once an arm walks.
FEET_AIR_TIME_THRESHOLD_S = 0.2

# [LG] LeggedRobotCfg.rewards.scales, verbatim, except the three noted.
REWARD_SCALES = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "feet_air_time": 1.0,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "action_rate": -0.01,
    # [LG] ships these two at non-zero defaults and v4 had dropped both. They are
    # the smoothness regularisers: without them nothing costs a buzzing, high-
    # torque policy that satisfies the tracking terms by vibrating.
    "torques": -0.00001,
    "dof_acc": -2.5e-7,
    "collision": -1.0,
    # [LG] ships orientation at -0.0 because ANYmal's flat-base prior comes from
    # its terrain curriculum. We have no curriculum and a body that already
    # learned to fall over once, so this is turned on.
    "orientation": -1.0,
    # Declared deviation from [LG], added after run 2. [LG] has no stance term
    # because ANYmal's flat-base prior comes from its terrain curriculum; we have
    # none, and run 2 exploited the gap. It trained for 24.9M steps and learned to
    # BRACE: joints driven 0.636 rad from the home stance on average, against the
    # CEM teacher's 0.103 rad, at a ride height above nominal standing. Splayed
    # wide enough that it cannot fall, and it never falls -- 12 s upright on every
    # seed -- while translating at 0.026 m/s against the teacher's 0.106.
    #
    # Nothing scored that. `upright` is a flat 1.0 anywhere above 0.13 m, so a
    # 0.636 rad contortion and a clean stance were worth exactly the same.
    #
    # The whole v4 design assumes the policy modulates AROUND the home stance:
    # the observation is (qpos - stance), the action is compared to stance, and
    # Stage 1 fits a teacher that stays within 0.103 rad of it. A configuration
    # 0.636 rad away is outside the envelope the interface was fit for. Quadratic,
    # so the teacher pays ~0.19 of it and the brace ~7.3 -- a 38x separation.
    "stance_deviation": -0.05,
}
# [FB] ride-height factor, applied the way [FB] applies it: as a multiplicative
# gate on the summed [LG] reward, not as a term inside the sum.
#
# This file's own header says [FB] multiplies bounded factors and [LG] sums
# weighted terms, and that v4 uses the [LG] sum. It then put `upright: 0.5` into
# that sum anyway, which is neither source's design. A multiplicative factor can
# only ever reduce a reward; an additive one pays for existing. Measured: holding
# the stance perfectly still scored 0.01825 per step against 0.02253 for the CEM
# teacher walking 0.424 m at 0.106 m/s -- a ratio of 1.23, with `upright`
# contributing 55% of the standing score for no effort. PPO finds standing at
# that contrast, which is the same failure that produced the v3 rocker.
#
# As a gate it keeps the property it was kept for -- it still collapses the
# reward as the body sinks, which is what stopped the v2 dive -- and pays nothing
# for merely being upright.
UPRIGHT_GATE = "upright"
# [FB] vision_flight height factor. STAND_HEIGHT_M is the measured settle height
# of the home stance held for 400 steps on this model.
#
# This gate used to read bounds=(0.13, inf), which scores a flat 1.0 at ANY height
# above 0.13 -- one-sided, so riding high was free. Run 2 found that: it braced
# into a splayed posture at 0.165 m, above nominal standing, and collected a full
# 1.0 gate for it. [FB] does not do this. vision_flight.py:167 passes
# bounds=(target, target) -- a POINT, with a linear falloff either side -- so
# departing the target height in EITHER direction costs reward. Matching that
# closes the hole; the margin runs to the fall threshold, so the factor reaches
# zero exactly where an episode would terminate anyway.
UPRIGHT_HEIGHT_M = 0.13         # kept: legacy floor, still the H_MIN-side reference
STAND_HEIGHT_M = 0.154
UPRIGHT_MARGIN_M = STAND_HEIGHT_M - H_MIN   # 0.074

FOOT_GEOMS = tuple(f"foot_{index}" for index in range(6))
TRUNK_GEOMS = tuple(f"trunk_{index}" for index in range(1, 8))
PREY_GEOM = "prey"


def wrap_to_pi(angle: float) -> float:
    """[LG] helpers.wrap_to_pi."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class HexapodTargetEnv:
    """Hexapod that walks to a target object, scored by the legged_gym reward set."""

    def __init__(self, control_hz: float = 50.0, kp: float = 3.0, kd: float = 0.05,
                 max_steps: int = EPISODE_STEPS):
        if control_hz <= 0 or kp < 0 or kd < 0 or max_steps < 1:
            raise ValueError("control_hz, max_steps must be positive; kp and kd must be non-negative")
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.control_hz = float(control_hz)
        self.dt = 1.0 / self.control_hz
        self.kp, self.kd = float(kp), float(kd)
        self.max_steps = int(max_steps)
        self.physics_steps = max(1, int(round(self.dt / self.model.opt.timestep)))
        if self.model.nu != 18 or self.model.nq != 25 or self.model.nv != 24:
            raise ValueError(f"unexpected hexapod dimensions nq={self.model.nq} nv={self.model.nv} nu={self.model.nu}")
        self.stance = home_stance(self.model).reshape(18).astype(np.float64)
        self.joint_range = np.asarray(self.model.jnt_range[1:], dtype=np.float64).copy()
        if self.joint_range.shape != (18, 2):
            raise ValueError(f"unexpected hinge range shape {self.joint_range.shape}")

        def geom_ids(names) -> np.ndarray:
            ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names])
            if (ids < 0).any():
                raise ValueError(f"hexapod XML is missing one of the geoms {names}")
            return ids

        self.foot_geom_ids = geom_ids(FOOT_GEOMS)
        self.trunk_geom_ids = geom_ids(TRUNK_GEOMS)
        self.prey_geom_id = int(geom_ids((PREY_GEOM,))[0])
        # The prey sphere is a static worldbody geom, so its episode position is
        # written into the model rather than the data. Keep the authored height.
        self.prey_z = float(self.model.geom_pos[self.prey_geom_id][2])

        self._done = True
        self._steps = 0
        self._target = np.zeros(2)
        self._min_height = float("inf")
        self._last_action = self.stance.copy()
        self._feet_air_time = np.zeros(6)
        self._last_dof_vel = np.zeros(18)
        self._last_contacts = np.zeros(6, dtype=bool)
        self._targets_reached = 0
        self._timeout = False
        self._last_terms = {name: 0.0 for name in list(REWARD_SCALES) + [UPRIGHT_GATE]}
        self._rng = np.random.default_rng()

    # --- observation ---------------------------------------------------------

    # 3 base linear + 3 base angular + 3 projected gravity + 3 target
    # + 18 joint positions + 18 joint velocities + 18 previous actions.
    OBS_DIM = 66

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM

    def _base_rotation(self) -> np.ndarray:
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, self.data.qpos[3:7])
        return matrix.reshape(3, 3)

    def heading(self) -> float:
        """Body yaw in the world frame. Exact for a tilted body, not just a level one."""
        w, x, y, z = self.data.qpos[3:7]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _target_polar(self) -> tuple[float, float]:
        """Egocentric bearing and distance to the target."""
        delta = self._target - self.data.qpos[:2]
        distance = float(np.linalg.norm(delta))
        bearing = wrap_to_pi(float(np.arctan2(delta[1], delta[0])) - self.heading())
        return bearing, distance

    def _observation(self) -> np.ndarray:
        """[LG] compute_observations, in its order, with our target block as the command.

        v2 and v3 fed the raw MuJoCo state -- `qpos` and `qvel` concatenated. That
        carries three things [LG] deliberately excludes, and each is a defect:

          * world XY position (`qpos[0:2]`), which drifts without bound. A linear
            encoder reading a growing input emits a drifting descending command.
          * absolute world yaw, which makes the policy non-equivariant: the same
            situation rotated 90 degrees is a different observation, so the body
            has to relearn walking once per heading.
          * world-frame base velocity, so "forward" depends on where north is.

        None of the three carries task information the target block does not
        already carry egocentrically, and dropping them costs nothing. [LG] also
        normalises, which matters here more than there: the encoder is a single
        `Linear(obs, 4)` and unscaled inputs put the whole descending drive at the
        mercy of whichever channel happens to be largest.

        The target block replaces [LG]'s `commands` slot. That placement is the
        seam the design turns on: a real fly's VNC does not see the world, it
        receives descending commands from the brain, where vision and olfaction
        live. The VNC still has to turn a steering command into eighteen
        coordinated joints, and that part stays frozen.
        """
        rotation = self._base_rotation()
        bearing, distance = self._target_polar()
        observation = np.concatenate((
            (rotation.T @ self.data.qvel[:3]) * OBS_SCALE_LIN_VEL,
            np.asarray(self.data.qvel[3:6]) * OBS_SCALE_ANG_VEL,
            rotation.T @ np.array([0.0, 0.0, -1.0]),
            (np.cos(bearing), np.sin(bearing), min(distance / OBS_TARGET_DIST_SCALE, 1.0)),
            (self.data.qpos[7:] - self.stance) * OBS_SCALE_DOF_POS,
            self.data.qvel[6:] * OBS_SCALE_DOF_VEL,
            self._last_action - self.stance,
        ))
        return np.clip(observation, -OBS_CLIP, OBS_CLIP).astype(np.float32, copy=False)

    # --- episode -------------------------------------------------------------

    def _respawn_target(self) -> None:
        angle = self._rng.uniform(-np.pi, np.pi)
        radius = self._rng.uniform(TARGET_SPAWN_MIN_M, TARGET_SPAWN_MAX_M)
        self._target = self.data.qpos[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
        self.model.geom_pos[self.prey_geom_id] = (self._target[0], self._target[1], self.prey_z)

    def reset(self, seed: int | None = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.key("home").qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._rng = np.random.default_rng(seed)
        yaw = float(self._rng.uniform(-np.pi, np.pi))
        self.data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        self._respawn_target()
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._done = False
        self._min_height = float(self.data.qpos[2])
        self._last_action = self.stance.copy()
        self._feet_air_time.fill(0.0)
        self._last_dof_vel.fill(0.0)
        self._last_contacts.fill(False)
        self._targets_reached = 0
        self._timeout = False
        self._last_terms = {name: 0.0 for name in list(REWARD_SCALES) + [UPRIGHT_GATE]}
        return self._observation()

    def _foot_contacts(self) -> np.ndarray:
        """Which feet touch anything this control step.

        [LG] reads PhysX contact forces per foot; MuJoCo exposes the contact list
        directly, so this reads that instead. Sampled at the control rate, which
        can miss a contact shorter than 20 ms -- the same rate [LG] samples at.
        """
        count = self.data.ncon
        if count == 0:
            return np.zeros(6, dtype=bool)
        geom1 = self.data.contact.geom1[:count]
        geom2 = self.data.contact.geom2[:count]
        touching = np.union1d(geom1, geom2)
        return np.isin(self.foot_geom_ids, touching)

    def _trunk_contacts(self) -> int:
        """[LG] _reward_collision, over the trunk geoms."""
        count = self.data.ncon
        if count == 0:
            return 0
        geom1 = self.data.contact.geom1[:count]
        geom2 = self.data.contact.geom2[:count]
        touching = np.union1d(geom1, geom2)
        return int(np.isin(self.trunk_geom_ids, touching).sum())

    # --- reward --------------------------------------------------------------

    def reward_terms(self, action: np.ndarray, contacts: np.ndarray) -> dict[str, float]:
        """One dict entry per [LG] `_reward_*` method, before weighting."""
        rotation = self._base_rotation()
        base_lin_vel = rotation.T @ self.data.qvel[:3]      # free-joint linear qvel is global
        base_ang_vel = np.asarray(self.data.qvel[3:6])      # free-joint angular qvel is already local
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])

        # [LG] heading-mode command: hold target speed straight ahead, and steer
        # with a yaw rate proportional to the heading error, clipped to +/-1.
        bearing, _ = self._target_polar()
        lin_vel_command = np.array([TARGET_SPEED_MPS, 0.0])
        ang_vel_command = float(np.clip(0.5 * bearing, -1.0, 1.0))

        # [LG] _reward_tracking_lin_vel / _reward_tracking_ang_vel
        lin_vel_error = float(np.sum(np.square(lin_vel_command - base_lin_vel[:2])))
        ang_vel_error = float(np.square(ang_vel_command - base_ang_vel[2]))

        # [LG] _reward_feet_air_time: credit a foot's airborne time only at the
        # moment it lands, so holding a foot up forever pays nothing and never
        # unloading a foot pays nothing either. This is the anti-rocking term.
        contact_filt = np.logical_or(contacts, self._last_contacts)
        self._last_contacts = contacts
        first_contact = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time += self.dt
        air_time = float(np.sum((self._feet_air_time - FEET_AIR_TIME_THRESHOLD_S) * first_contact))
        self._feet_air_time *= ~contact_filt

        # [FB] vision_flight ride-height factor.
        upright = float(tolerance(
            float(self.data.qpos[2]),
            bounds=(STAND_HEIGHT_M, STAND_HEIGHT_M),
            margin=UPRIGHT_MARGIN_M,
            sigmoid="linear",
            value_at_margin=0.0,
        ))

        # [LG] _reward_torques / _reward_dof_acc. `data.ctrl` holds the last
        # substep's PD torques, which is the control-rate sample [LG] takes too.
        dof_vel = np.asarray(self.data.qvel[6:])
        dof_acc = float(np.sum(np.square((self._last_dof_vel - dof_vel) / self.dt)))
        self._last_dof_vel = dof_vel.copy()

        return {
            "tracking_lin_vel": float(np.exp(-lin_vel_error / TRACKING_SIGMA_LIN)),
            "tracking_ang_vel": float(np.exp(-ang_vel_error / TRACKING_SIGMA_ANG)),
            "feet_air_time": air_time,
            "lin_vel_z": float(np.square(base_lin_vel[2])),
            "ang_vel_xy": float(np.sum(np.square(base_ang_vel[:2]))),
            "action_rate": float(np.sum(np.square(self._last_action - action))),
            "torques": float(np.sum(np.square(self.data.ctrl))),
            "dof_acc": dof_acc,
            "collision": float(self._trunk_contacts()),
            "orientation": float(np.sum(np.square(projected_gravity[:2]))),
            "stance_deviation": float(np.sum(np.square(self.data.qpos[7:] - self.stance))),
            "upright": upright,
        }

    # --- step ----------------------------------------------------------------

    def step(self, action_18: np.ndarray) -> tuple[np.ndarray, float, bool]:
        if self._done:
            raise RuntimeError("step called after episode is done; call reset()")
        action = np.asarray(action_18, dtype=np.float64).reshape(-1)
        if action.shape != (18,):
            raise ValueError(f"action has shape {action.shape}; expected (18,)")
        if not np.isfinite(action).all():
            raise ValueError("action must contain only finite values")
        target = np.clip(action, self.joint_range[:, 0], self.joint_range[:, 1])
        for _ in range(self.physics_steps):
            self.data.ctrl[:] = pd_torque(self.model, self.data, target, self.kp, self.kd)
            mujoco.mj_step(self.model, self.data)
            self._min_height = min(self._min_height, float(self.data.qpos[2]))

        terms = self.reward_terms(target, self._foot_contacts())
        # Kept so the vector wrapper can report which behaviour is improving.
        # Total return alone cannot distinguish "tracks the heading better" from
        # "fell over less", and those want different fixes.
        self._last_terms = terms
        # [LG] _prepare_reward_function multiplies every scale by dt, then
        # compute_reward sums and (only_positive_rewards) clips the total at zero.
        # The [FB] ride-height factor then gates that total; see UPRIGHT_GATE.
        reward = self.dt * sum(scale * terms[name] for name, scale in REWARD_SCALES.items())
        reward = max(reward, 0.0) * terms[UPRIGHT_GATE]

        _, distance = self._target_polar()
        if distance < TARGET_REACH_M:
            reward += TARGET_REACH_BONUS
            self._targets_reached += 1
            self._respawn_target()

        self._last_action = target
        self._steps += 1
        fell = bool(self._min_height < H_MIN)
        self._timeout = bool(self._steps >= self.max_steps and not fell)
        self._done = bool(self._timeout or fell)
        return self._observation(), float(reward), self._done


class VectorHexapodTargetEnv:
    """CPU vector wrapper; terminal rows are reset before returning observations."""

    def __init__(self, batch_size: int, **env_kwargs):
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.envs = [HexapodTargetEnv(**env_kwargs) for _ in range(self.batch_size)]
        self.last_reset_mask = np.ones(self.batch_size, dtype=bool)
        # `step` resets a terminal env, which zeroes that env's own reach counter.
        # Reading `env._targets_reached` after a rollout that ends on a terminal
        # step therefore always returned 0 -- the primary progress signal would
        # have been flat zero for the whole run. Bank the count before the reset.
        self.reaches = np.zeros(self.batch_size, dtype=np.int64)
        # Ending on the clock is not the same event as falling over, and the
        # observation to bootstrap a truncated episode from is the one the
        # auto-reset is about to throw away. Both are kept for the GAE mask.
        self.last_timeout_mask = np.zeros(self.batch_size, dtype=bool)
        self.last_terminal_obs = np.zeros((self.batch_size, self.obs_dim), dtype=np.float32)
        # Running sum of every reward term over the rollout segment. `targets_reached`
        # is the endpoint but it stays at 0.000 for hours while the arm gets steadily
        # better, so on its own it cannot tell a run that is working from run 1.
        self.term_sums = {name: 0.0 for name in list(REWARD_SCALES) + [UPRIGHT_GATE]}
        self.term_steps = 0
        # Run 2 braced for 24.9M steps and every logged scalar improved while the
        # behaviour got worse. These are the two that would have shown it: how far
        # the joints sit from the home stance, and the actual ride height.
        self.posture_sums = {"stance_dev": 0.0, "height": 0.0}

    @property
    def obs_dim(self) -> int:
        return self.envs[0].obs_dim

    @property
    def action_dim(self) -> int:
        return 18

    def mean_terms(self) -> dict[str, float]:
        """Mean of each unweighted reward term per environment step, then cleared."""
        steps = max(self.term_steps, 1)
        means = {name: value / steps for name, value in self.term_sums.items()}
        means.update({name: value / steps for name, value in self.posture_sums.items()})
        self.term_sums = {name: 0.0 for name in self.term_sums}
        self.posture_sums = {name: 0.0 for name in self.posture_sums}
        self.term_steps = 0
        return means

    def reset(self, seeds: np.ndarray | list[int | None] | None = None) -> np.ndarray:
        if seeds is None:
            seeds = [None] * self.batch_size
        if len(seeds) != self.batch_size:
            raise ValueError(f"seeds has length {len(seeds)}; expected {self.batch_size}")
        observations = np.stack([env.reset(seed) for env, seed in zip(self.envs, seeds)])
        self.last_reset_mask.fill(True)
        self.last_timeout_mask.fill(False)
        self.last_terminal_obs.fill(0.0)
        self.reaches.fill(0)
        return observations

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.batch_size, self.action_dim):
            raise ValueError(f"actions has shape {actions.shape}; expected ({self.batch_size}, {self.action_dim})")
        if not np.isfinite(actions).all():
            raise ValueError("actions must contain only finite values")
        observations, rewards, done = [], [], np.zeros(self.batch_size, dtype=bool)
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            reaches_before = env._targets_reached
            observation, reward, terminal = env.step(action)
            self.reaches[index] += env._targets_reached - reaches_before
            done[index] = terminal
            self.last_timeout_mask[index] = env._timeout
            for name, value in env._last_terms.items():
                self.term_sums[name] += value
            self.posture_sums["stance_dev"] += float(np.sum(np.square(env.data.qpos[7:] - env.stance)))
            self.posture_sums["height"] += float(env.data.qpos[2])
            if terminal:
                self.last_terminal_obs[index] = observation
                observation = env.reset()
            observations.append(observation)
            rewards.append(reward)
        self.term_steps += self.batch_size
        self.last_reset_mask = done.copy()
        result = np.stack(observations).astype(np.float32, copy=False)
        assert result.shape == (self.batch_size, self.obs_dim)
        return result, np.asarray(rewards, dtype=np.float32), done


def smoke_test() -> dict[str, float | int]:
    """One runnable check that fails if any of the three known failure modes returns."""
    start = time.perf_counter()
    env = HexapodTargetEnv(max_steps=50)
    obs = env.reset(20260903)
    assert obs.shape == (env.obs_dim,) and env.obs_dim == 66
    rng = np.random.default_rng(20260903)
    steps, done = 0, False
    while not done:
        obs, reward, done = env.step(env.stance + rng.normal(0.0, 0.04, size=18))
        assert obs.shape == (env.obs_dim,)
        assert np.isfinite(obs).all() and np.isfinite(reward) and reward >= 0.0
        steps += 1
    assert 0 < steps <= env.max_steps

    # The three failure modes, scored directly. A probe body is posed by hand and
    # its reward terms read, so each assertion names one thing that must not pay.
    probe = HexapodTargetEnv()
    probe.reset(20260903)

    def score(vx: float, vy: float, height: float, contacts: np.ndarray) -> float:
        probe.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)          # zero yaw
        probe._target = probe.data.qpos[:2] + np.array([1.0, 0.0])  # dead ahead
        probe.data.qvel[:3] = (vx, vy, 0.0)
        probe.data.qpos[2] = height
        mujoco.mj_forward(probe.model, probe.data)
        probe._last_action = probe.stance.copy()
        terms = probe.reward_terms(probe.stance, contacts)
        total = max(probe.dt * sum(scale * terms[k] for k, scale in REWARD_SCALES.items()), 0.0)
        return total * terms[UPRIGHT_GATE]

    all_down = np.ones(6, dtype=bool)
    walk = score(TARGET_SPEED_MPS, 0.0, STAND_HEIGHT_M, all_down)
    dive = score(0.588, 0.0, 0.0724, all_down)   # measured v2 connectome arm
    rock = score(0.0, 0.0, STAND_HEIGHT_M, all_down)  # measured v3 connectome arm
    assert walk > rock, "on-target walking must outscore standing still"
    assert walk > dive, "the v2 dive must not outscore the walk"
    # [LG] pays a standing robot for tracking a zero yaw-rate command and for
    # every penalty it is not incurring, so a per-step standing floor of roughly
    # half the walking rate is inherited from the source, not a bug here. What
    # closes the gap is the episode: only a walker collects reach bonuses.
    assert rock < 0.6 * walk, f"standing still scores {rock:.4f} against a walk at {walk:.4f}"
    rocking_return = EPISODE_STEPS * rock
    walking_return = EPISODE_STEPS * walk + TARGET_REACH_BONUS  # >= 1 reach at target speed
    assert walking_return > 3.0 * rocking_return, (
        f"a walking episode returns {walking_return:.1f} against {rocking_return:.1f} for rocking; "
        "raise TARGET_REACH_BONUS until the task term dominates"
    )

    # feet_air_time must pay a stepping foot and nothing else. Lift one foot for
    # ten control steps, then land it; a foot that never leaves the ground and a
    # foot that never lands must both earn zero.
    def air_time_reward(lift_steps: int, land: bool) -> float:
        probe._feet_air_time.fill(0.0)
        probe._last_contacts = np.ones(6, dtype=bool)
        total = 0.0
        for _ in range(lift_steps):
            contacts = np.ones(6, dtype=bool)
            contacts[0] = False
            total += probe.reward_terms(probe.stance, contacts)["feet_air_time"]
        if land:
            total += probe.reward_terms(probe.stance, np.ones(6, dtype=bool))["feet_air_time"]
        return total

    # [LG] filters contacts against the previous step, so a foot needs two
    # consecutive airborne steps before its air time starts accumulating.
    assert air_time_reward(20, land=True) > 0.0, "a foot that swings and lands must be paid"
    assert air_time_reward(20, land=False) == 0.0, "a foot held in the air must not be paid"
    assert air_time_reward(0, land=True) == 0.0, "a foot that never leaves the ground must not be paid"
    # [LG] does not clip this term, so a stride shorter than the threshold is a
    # penalty. That is the property that makes shuffling cost more than standing.
    assert air_time_reward(5, land=True) < 0.0, "a stride shorter than the threshold must cost"

    # The reach counter must survive the vector wrapper's auto-reset, or the only
    # endpoint the run reports is permanently zero.
    vector = VectorHexapodTargetEnv(2, max_steps=3)
    vector.reset(seeds=[20260903, 20260904])
    for one in vector.envs:                       # park the target under the body
        one._target = one.data.qpos[:2].copy()
    for _ in range(3):
        vector.step(np.stack([one.stance for one in vector.envs]))
    assert (vector.reaches > 0).all(), f"reaches lost across the auto-reset: {vector.reaches}"

    return {
        "steps": steps,
        "obs_dim": env.obs_dim,
        "walk_step_reward": walk,
        "dive_step_reward": dive,
        "rock_step_reward": rock,
        "wall_time_s": time.perf_counter() - start,
    }


if __name__ == "__main__":
    result = smoke_test()
    print(
        "ENV V4 SMOKE: PASS "
        f"(steps={result['steps']} obs_dim={result['obs_dim']} "
        f"walk={result['walk_step_reward']:.5f} dive={result['dive_step_reward']:.5f} "
        f"rock={result['rock_step_reward']:.5f} wall_time_s={result['wall_time_s']:.3f})"
    )
