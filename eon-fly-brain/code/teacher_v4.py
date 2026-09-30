"""Command-conditioned Stage-1 teacher for the v4 goal-directed task.

The v2/v3 teacher (`audit_embody_3way.build_teacher`) searched one gait and
scored it against a straight constant-speed path. That taught the interface
exactly one thing: walk forwards. The encoder's `steer` latent had no reason to
mean anything, and steering is the whole of "walk to a target".

This is the fix, copied from Kim et al., "Deep Reinforcement Learning Control of
a Hexapod Robot", Actuators 15(1):33 (2026), which trains an 18-DoF hexapod with
PPO on top of *command-conditioned kinematic trajectories* as the imitation
prior, rather than a single reference gait. Here the commands are yaw rates: one
independent CEM search per command, each scored against the arc that command
traces at the target speed.

Everything else -- the periodic joint parametrisation, the CEM hyperparameters,
the fall penalty -- is carried over unchanged from the v2/v3 teacher so that the
only difference between the two Stage 1 datasets is the command conditioning.
"""
from __future__ import annotations

import math

import numpy as np

from embody_env_v4 import HexapodTargetEnv, TARGET_SPEED_MPS

# Carried over from audit_embody_3way, except the search budget.
TEACHER_SEED = 20260903
TEACHER_FREQ_HZ = 1.20
# v2/v3 ran population 24 for 3 rounds and shipped whatever that found. Measured
# on the straight command over 200 steps, that budget selects a teacher with path
# MSE 1.25e-2 that travels 0.166 m against a 0.400 m reference -- a shuffle, not a
# gait. v3's own recorded teacher agrees: best_path_mse_m2 = 0.0117. Every arm
# then spent Stage 1 imitating that, so the interface was initialised to shuffle
# before PPO ever ran, in all three conditions.
#
# Population 64 for 12 rounds reaches MSE 2.77e-4 and 0.424 m on the same command
# and seed -- a 45x better fit and a teacher that actually covers the reference
# distance. The cost is about a minute of CEM per command.
TEACHER_POPULATION = 64
TEACHER_ELITES = 12
TEACHER_ROUNDS = 12
TEACHER_FALL_PENALTY = 1.0
TEACHER_PARAM_LOW = (-0.35, 0.0, 0.0, -math.pi)
TEACHER_PARAM_HIGH = (0.35, 0.70, 0.70, math.pi)
TEACHER_INITIAL_MEAN = (0.05, 0.20, 0.10, 0.0)
TEACHER_INITIAL_STD = (0.10, 0.20, 0.20, math.pi)
TEACHER_STD_FLOOR = (0.02, 0.05, 0.05, 0.20)
STAGE1_STEPS = 200

# Yaw-rate commands, rad/s. Symmetric so neither turn direction is favoured, and
# bounded by what the [LG] heading law can ask for: it emits
# ang_vel_cmd = clip(0.5 * bearing, -1, 1), so 0.4 rad/s corresponds to a target
# sitting 0.8 rad off the nose -- a normal approach angle, not a pirouette.
# Narrowed from (-0.4, -0.2, 0.0, 0.2, 0.4).
#
# The five-command set is what the 4-latent anatomical map cannot carry. Same
# architecture, same fixed gradients, two measurements: one command fits to 0.00611
# against a 0.01109 constant (beats it 1.8x), five commands plateau at 0.0209
# against 0.0145 (loses). Rhythm, sigmoid saturation and the optimizer were all
# ruled out by direct measurement; the command count is what is left.
#
# Three commands keep both turn directions -- so steering is still conditioned,
# and the left/right assertion below still has something to check -- while asking
# the 4 latents to separate three gaits instead of five. Widening the map to hold
# more commands is the forbidden fix; narrowing what we ask of it is not.
TEACHER_COMMANDS = (-0.2, 0.0, 0.2)
# Distance at which the synthetic target is held during teacher rollouts. Only
# the distance *channel* of the observation depends on it; the commanded arc does
# not. Held constant so Stage 1 never sees a reach event.
TEACHER_TARGET_DISTANCE_M = 1.0


def teacher_action(params: np.ndarray, step: int, stance: np.ndarray, dt: float) -> np.ndarray:
    """Generic periodic joint controller, unchanged from the v2/v3 teacher."""
    action = stance.reshape(6, 3).copy()
    phase = 2.0 * math.pi * TEACHER_FREQ_HZ * step * dt + params[:, 3]
    swing = np.sin(phase)
    lift = np.maximum(swing, 0.0)
    action[:, 0] += params[:, 0] * swing
    action[:, 1] += params[:, 1] * lift
    action[:, 2] += params[:, 2] * lift
    return action.reshape(18).astype(np.float32)


def arc_reference(yaw_rate: float, steps: int, dt: float) -> np.ndarray:
    """Body path traced by TARGET_SPEED_MPS at a constant yaw rate, in the start frame.

    Straight line as yaw_rate -> 0, circle of radius v/omega otherwise. This is the
    only place the command enters the objective.
    """
    time = dt * np.arange(1, steps + 1, dtype=np.float64)
    if abs(yaw_rate) < 1e-9:
        return np.stack((TARGET_SPEED_MPS * time, np.zeros_like(time)), axis=1)
    radius = TARGET_SPEED_MPS / yaw_rate
    return np.stack((radius * np.sin(yaw_rate * time), radius * (1.0 - np.cos(yaw_rate * time))), axis=1)


def rollout(params: np.ndarray, yaw_rate: float, steps: int) -> tuple[float, bool, np.ndarray, np.ndarray]:
    """Run one candidate under one command; return loss, fall flag, actions, observations.

    The synthetic target is re-placed every step at the bearing the [LG] heading
    law would need to emit this command, so the three target observation channels
    stay consistent with the arc being imitated for the whole rollout.
    """
    env = HexapodTargetEnv(max_steps=steps)
    observation = env.reset(TEACHER_SEED)
    x0, y0 = float(env.data.qpos[0]), float(env.data.qpos[1])
    yaw0 = env.heading()
    cos0, sin0 = math.cos(yaw0), math.sin(yaw0)
    bearing = 2.0 * yaw_rate  # inverse of ang_vel_cmd = clip(0.5 * bearing, -1, 1)

    def place_target() -> None:
        # Same heading definition the observation uses, or the bearing channel the
        # interface is fitted on would not be the bearing that was commanded.
        angle = env.heading() + bearing
        env._target = env.data.qpos[:2] + TEACHER_TARGET_DISTANCE_M * np.array([math.cos(angle), math.sin(angle)])

    place_target()
    observation = env._observation()

    observations, actions, path = [], [], []
    fell = False
    for step in range(steps):
        observations.append(observation)
        action = teacher_action(params, step, env.stance, env.dt)
        actions.append(action)
        observation, _, done = env.step(action)
        place_target()
        observation = env._observation()
        dx, dy = float(env.data.qpos[0]) - x0, float(env.data.qpos[1]) - y0
        path.append((dx * cos0 + dy * sin0, -dx * sin0 + dy * cos0))
        if done and step + 1 < steps:
            fell = True
            break

    path = np.asarray(path, dtype=np.float64)
    if len(path) < steps:  # hold the last pose so a fall is scored, not truncated
        path = np.vstack((path, np.repeat(path[-1:], steps - len(path), axis=0)))
        observations.extend([observations[-1]] * (steps - len(observations)))
        actions.extend([actions[-1]] * (steps - len(actions)))
    loss = float(np.mean((path - arc_reference(yaw_rate, steps, env.dt)) ** 2) + TEACHER_FALL_PENALTY * fell)
    return loss, fell, np.asarray(actions, dtype=np.float32), np.asarray(observations, dtype=np.float32)


def search_command(yaw_rate: float, steps: int = STAGE1_STEPS) -> tuple[np.ndarray, np.ndarray, dict]:
    """One independent CEM search for one command. Hyperparameters are the v2/v3 ones."""
    rng = np.random.default_rng(TEACHER_SEED + int(round(1000 * yaw_rate)))
    mean = np.tile(np.asarray(TEACHER_INITIAL_MEAN, dtype=np.float64), (6, 1))
    std = np.tile(np.asarray(TEACHER_INITIAL_STD, dtype=np.float64), (6, 1))
    low = np.tile(np.asarray(TEACHER_PARAM_LOW, dtype=np.float64), (6, 1))
    high = np.tile(np.asarray(TEACHER_PARAM_HIGH, dtype=np.float64), (6, 1))
    floor = np.tile(np.asarray(TEACHER_STD_FLOOR, dtype=np.float64), (6, 1))
    best = None
    for _ in range(TEACHER_ROUNDS):
        samples = np.clip(rng.normal(mean, std, size=(TEACHER_POPULATION, 6, 4)), low, high)
        # Keyed on the loss: the tuples carry numpy arrays, which raise rather
        # than compare if two candidates ever tie on loss.
        scored = sorted(((rollout(params, yaw_rate, steps), params) for params in samples),
                        key=lambda entry: entry[0][0])
        elites = scored[:TEACHER_ELITES]
        mean = np.mean([params for _, params in elites], axis=0)
        std = np.maximum(np.std([params for _, params in elites], axis=0), floor)
        if best is None or elites[0][0][0] < best[0][0]:
            best = elites[0]
    assert best is not None
    (loss, fell, actions, observations), best_params = best
    # The params come back too, so a relabelling pass can ask "what would the
    # teacher have done at this phase?" for a state the teacher never visited.
    # Without them the teacher is only replayable as a fixed action sequence.
    return observations, actions, {"yaw_rate": yaw_rate, "path_mse_m2": loss - TEACHER_FALL_PENALTY * fell,
                                   "fell": bool(fell), "params": np.asarray(best_params, dtype=np.float64).tolist()}


def build_teacher(steps: int = STAGE1_STEPS) -> tuple[np.ndarray, np.ndarray, dict]:
    """One CEM-selected demonstration per command.

    Returned with a leading command axis, (commands, steps, dim), NOT concatenated.
    The demonstrations are independent episodes that each start from a reset body,
    so anything replaying them has to reset the recurrent state between them; a
    flat (commands * steps, dim) array silently loses that boundary.
    """
    observations, actions, per_command = [], [], []
    for yaw_rate in TEACHER_COMMANDS:
        obs, act, record = search_command(yaw_rate, steps)
        observations.append(obs)
        actions.append(act)
        per_command.append(record)
    observations = np.stack(observations, axis=0)
    actions = np.stack(actions, axis=0)
    record = {
        "method": "command-conditioned CEM periodic joint controller, one search per yaw-rate command, "
                  "scored by constant-speed arc path MSE plus declared fall penalty",
        "source": "Kim et al., Actuators 15(1):33 (2026), command-conditioned kinematic imitation prior",
        "commands_rad_s": list(TEACHER_COMMANDS),
        "speed_mps": TARGET_SPEED_MPS,
        "frequency_hz": TEACHER_FREQ_HZ,
        "population": TEACHER_POPULATION,
        "elites": TEACHER_ELITES,
        "rounds": TEACHER_ROUNDS,
        "seed": TEACHER_SEED,
        "fall_penalty": TEACHER_FALL_PENALTY,
        "steps_per_command": steps,
        "target_distance_m": TEACHER_TARGET_DISTANCE_M,
        "per_command": per_command,
    }
    return observations, actions, record


def self_check() -> dict:
    """Short search on a short horizon; asserts the command actually changes the demo."""
    steps = 40
    observations, actions, record = build_teacher(steps)
    assert observations.shape == (len(TEACHER_COMMANDS), steps, HexapodTargetEnv.OBS_DIM), observations.shape
    assert actions.shape == (len(TEACHER_COMMANDS), steps, 18), actions.shape
    assert np.isfinite(observations).all() and np.isfinite(actions).all()

    # The bearing channels must differ across commands, or the conditioning is
    # invisible to the encoder and this whole module bought nothing.
    # Observation layout: 3 base linear, 3 base angular, 3 projected gravity,
    # then cos(bearing), sin(bearing), distance.
    bearings = observations[:, :, 10].mean(axis=1)  # sin(bearing) channel
    assert bearings[0] < bearings[2] < bearings[-1], f"sin(bearing) not ordered by command: {bearings}"

    # A left command and a right command must not produce the same gait.
    left = actions[0]
    right = actions[-1]
    assert float(np.abs(left - right).max()) > 1e-3, "left and right commands produced identical actions"

    # The failure that shipped in v2 and v3: a "teacher" that scores acceptably
    # while barely moving. Distance travelled is the check that catches it, and
    # it is asserted on the straight command at full horizon, not the short one.
    straight_obs, straight_act, straight_record = search_command(0.0, STAGE1_STEPS)
    env = HexapodTargetEnv(max_steps=STAGE1_STEPS)
    env.reset(TEACHER_SEED)
    start = env.data.qpos[:2].copy()
    for action in straight_act:
        env.step(action)
    travelled = float(np.linalg.norm(env.data.qpos[:2] - start))
    expected = TARGET_SPEED_MPS * STAGE1_STEPS * env.dt
    assert travelled > 0.8 * expected, (
        f"teacher travelled {travelled:.3f} m against a {expected:.3f} m reference; "
        "the search budget is too small and Stage 1 would imitate a shuffle"
    )
    record["straight_travelled_m"] = travelled
    record["straight_reference_m"] = expected

    # The arc reference must curve the way the command says.
    assert arc_reference(0.4, 100, 0.02)[-1, 1] > 0.0, "positive yaw rate must curve left"
    assert arc_reference(-0.4, 100, 0.02)[-1, 1] < 0.0, "negative yaw rate must curve right"
    assert abs(arc_reference(0.0, 100, 0.02)[-1, 1]) < 1e-12, "zero yaw rate must go straight"
    return record


if __name__ == "__main__":
    import json

    print("TEACHER V4 SELF CHECK: PASS")
    print(json.dumps(self_check(), indent=2, sort_keys=True))
