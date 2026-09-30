"""CPU-only gym-style hexapod wrapper for the frozen VNC stack."""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
from dm_rewards import tolerance  # noqa: E402
from hexapod_gait import H_MIN, XML, home_stance, pd_torque  # noqa: E402

# Reward shaping, ported from flybody (google-deepmind/flybody). Two properties
# of theirs matter and the previous unbounded `dx*cos+dy*sin` reward had neither:
#
#   1. Bounded target tracking, not maximisation. flybody's walk_on_ball rewards
#      tolerance(ball_qvel - target_ball_qvel, ...): moving faster than target
#      scores *worse*, so a ballistic lunge cannot outscore a walk. Maximising
#      raw displacement has no fixed point and the lunge always wins it -- the
#      v2 connectome arm hit 0.588 m/s for 0.52 s and fell in 12/12 eval seeds.
#   2. Multiplicative combination. flybody's Walking.get_reward returns
#      np.prod(self.get_reward_factors(physics)), so no factor can be bought
#      with another; one bad factor zeroes the step. Summing lets a policy trade
#      stability for distance, which is exactly the trade v2 made.
#
# TARGET_SPEED_MPS matches the Stage-1 CEM teacher's speed_mps, so both stages
# now optimise the same locomotion objective. In v2 Stage 1 taught 0.1 m/s and
# Stage 2 rewarded unbounded distance; the stages disagreed and Stage 2 won.
TARGET_SPEED_MPS = 0.1
# Home stance held for 400 control steps settles at 0.154 m (measured, this
# model). Full upright credit above UPRIGHT_HEIGHT_M leaves room for normal gait
# bob; credit decays linearly to zero at the H_MIN fall floor, so losing ride
# height costs reward continuously instead of only at the terminal cliff.
UPRIGHT_HEIGHT_M = 0.13
STAND_HEIGHT_M = 0.154


class HexapodEnv:
    """Minimal state/action wrapper around the existing MuJoCo hexapod."""

    def __init__(self, control_hz: float = 50.0, kp: float = 3.0, kd: float = 0.05, max_steps: int = 400):
        if control_hz <= 0 or kp < 0 or kd < 0 or max_steps < 1:
            raise ValueError("control_hz, max_steps must be positive; kp and kd must be non-negative")
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.control_hz = float(control_hz)
        self.kp, self.kd = float(kp), float(kd)
        self.max_steps = int(max_steps)
        self.physics_steps = max(1, int(round((1.0 / self.control_hz) / self.model.opt.timestep)))
        if self.model.nu != 18 or self.model.nq != 25 or self.model.nv != 24:
            raise ValueError(f"unexpected hexapod dimensions nq={self.model.nq} nv={self.model.nv} nu={self.model.nu}")
        self.stance = home_stance(self.model).reshape(18).astype(np.float64)
        self.joint_range = np.asarray(self.model.jnt_range[1:], dtype=np.float64).copy()
        if self.joint_range.shape != (18, 2):
            raise ValueError(f"unexpected hinge range shape {self.joint_range.shape}")
        self._done = True
        self._steps = 0
        self._heading = 0.0
        self._x0 = self._y0 = 0.0
        self._min_height = float("inf")

    @property
    def obs_dim(self) -> int:
        return self.model.nq + self.model.nv

    def _observation(self) -> np.ndarray:
        return np.concatenate((self.data.qpos, self.data.qvel)).astype(np.float32, copy=True)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset home pose; an optional seed gives the episode a yaw."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.key("home").qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        if seed is not None:
            yaw = float(np.random.default_rng(int(seed)).uniform(-np.pi, np.pi))
            self.data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._done = False
        self._x0, self._y0 = float(self.data.qpos[0]), float(self.data.qpos[1])
        self._heading = float(2.0 * np.arctan2(self.data.qpos[6], self.data.qpos[3]))
        self._min_height = float(self.data.qpos[2])
        return self._observation()

    def reward_factors(self) -> tuple[float, float]:
        """Bounded [0, 1] factors: forward-speed tracking, then upright ride height."""
        velocity = np.asarray(self.data.qvel[:3], dtype=np.float64)
        forward_speed = velocity[0] * np.cos(self._heading) + velocity[1] * np.sin(self._heading)
        speed = tolerance(
            forward_speed,
            bounds=(TARGET_SPEED_MPS, TARGET_SPEED_MPS),
            margin=TARGET_SPEED_MPS,
            sigmoid="linear",
            value_at_margin=0.0,
        )
        upright = tolerance(
            float(self.data.qpos[2]),
            bounds=(UPRIGHT_HEIGHT_M, float("inf")),
            margin=UPRIGHT_HEIGHT_M - H_MIN,
            sigmoid="linear",
            value_at_margin=0.0,
        )
        return float(speed), float(upright)

    def step(self, action_18: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Apply 18 absolute joint targets; reward is the product of the bounded factors."""
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

        # Product, not sum: one collapsed factor zeroes the step (flybody Walking.get_reward).
        speed, upright = self.reward_factors()
        reward = speed * upright
        self._steps += 1
        self._done = bool(self._steps >= self.max_steps or self._min_height < H_MIN)
        return self._observation(), float(reward), self._done


class VectorHexapodEnv:
    """CPU vector wrapper; terminal rows are reset before returning observations."""

    def __init__(self, batch_size: int, **env_kwargs):
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.envs = [HexapodEnv(**env_kwargs) for _ in range(self.batch_size)]
        self.last_reset_mask = np.ones(self.batch_size, dtype=bool)

    @property
    def obs_dim(self) -> int:
        return self.envs[0].obs_dim

    @property
    def action_dim(self) -> int:
        return 18

    def reset(self, seeds: np.ndarray | list[int | None] | None = None) -> np.ndarray:
        if seeds is None:
            seeds = [None] * self.batch_size
        if len(seeds) != self.batch_size:
            raise ValueError(f"seeds has length {len(seeds)}; expected {self.batch_size}")
        observations = np.stack([env.reset(seed) for env, seed in zip(self.envs, seeds)])
        self.last_reset_mask.fill(True)
        return observations

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.batch_size, self.action_dim):
            raise ValueError(f"actions has shape {actions.shape}; expected ({self.batch_size}, {self.action_dim})")
        if not np.isfinite(actions).all():
            raise ValueError("actions must contain only finite values")
        observations, rewards, done = [], [], np.zeros(self.batch_size, dtype=bool)
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            observation, reward, terminal = env.step(action)
            done[index] = terminal
            if terminal:
                observation = env.reset()
            observations.append(observation)
            rewards.append(reward)
        self.last_reset_mask = done.copy()
        result = np.stack(observations).astype(np.float32, copy=False)
        assert result.shape == (self.batch_size, self.obs_dim)
        return result, np.asarray(rewards, dtype=np.float32), done


def smoke_test() -> dict[str, float | int]:
    start = time.perf_counter()
    env = HexapodEnv(max_steps=50)
    obs = env.reset()
    rng = np.random.default_rng(20260830)
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = env.stance + rng.normal(0.0, 0.04, size=18)
        obs, reward, done = env.step(action)
        assert obs.shape == (env.obs_dim,)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert isinstance(done, bool)
        total_reward += reward
        steps += 1
    assert steps > 0 and steps <= env.max_steps

    # The v2 failure in one check: the dive must not outscore the walk.
    probe = HexapodEnv()
    probe.reset(20260830)
    probe._heading = 0.0

    def factors_at(speed_mps: float, height_m: float) -> tuple[float, float]:
        probe.data.qvel[:3] = (speed_mps, 0.0, 0.0)
        probe.data.qpos[2] = height_m
        return probe.reward_factors()

    walk = factors_at(TARGET_SPEED_MPS, STAND_HEIGHT_M)
    dive = factors_at(0.588, 0.0724)  # measured v2 connectome arm: 0.588 m/s, fell to 0.0724 m
    stand = factors_at(0.0, STAND_HEIGHT_M)
    for speed, upright in (walk, dive, stand):
        assert 0.0 <= speed <= 1.0 and 0.0 <= upright <= 1.0
    assert walk[0] * walk[1] > 0.99, "on-target upright walking must score ~1 per step"
    assert dive[0] * dive[1] == 0.0, "overspeed dive below the fall floor must score 0"
    assert stand[0] * stand[1] == 0.0, "standing still must not earn forward-speed credit"
    assert factors_at(TARGET_SPEED_MPS, 0.105)[1] < walk[1], "losing ride height must cost reward before H_MIN"
    assert factors_at(2 * TARGET_SPEED_MPS, STAND_HEIGHT_M)[0] == 0.0, "exceeding target speed must not pay more"

    return {
        "steps": steps,
        "obs_dim": env.obs_dim,
        "total_reward": total_reward,
        "walk_step_reward": walk[0] * walk[1],
        "dive_step_reward": dive[0] * dive[1],
        "wall_time_s": time.perf_counter() - start,
    }


if __name__ == "__main__":
    result = smoke_test()
    print(
        "ENV PORT SMOKE: PASS "
        f"(steps={result['steps']} obs_dim={result['obs_dim']} "
        f"total_reward={result['total_reward']:+.6f} wall_time_s={result['wall_time_s']:.3f})"
    )
