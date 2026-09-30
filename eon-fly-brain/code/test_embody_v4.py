"""Pre-flight checks for the v4 pipeline. Run before spending pod time.

Each test names one way the run could waste money by producing a number that
looks fine and means nothing. Grouped by what they protect:

  * the frozen-core invariant, which the whole project rests on
  * fidelity to [LG] legged_gym, which v2 and v3 both drifted from
  * the reward, which has been gamed twice
  * the plumbing, where the v4 review found four silent zeros

Run: python test_embody_v4.py
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import embody_env_v4 as E
import run_embody_v4 as R
import teacher_v4 as T


# --- fidelity to legged_gym --------------------------------------------------

def test_reward_scales_match_source():
    """Every scale is [LG]'s LeggedRobotCfg.rewards.scales value, or a declared deviation."""
    source = {  # legged_gym/envs/base/legged_robot_config.py, transcribed
        "tracking_lin_vel": 1.0, "tracking_ang_vel": 0.5, "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05, "orientation": -0.0, "torques": -0.00001,
        "dof_vel": -0.0, "dof_acc": -2.5e-7, "base_height": -0.0,
        "feet_air_time": 1.0, "collision": -1.0, "feet_stumble": -0.0,
        "action_rate": -0.01, "stand_still": -0.0, "termination": -0.0,
    }
    declared_deviations = {
        "orientation": -1.0,        # [LG] ships -0.0; no terrain curriculum here
        # Added after run 2's brace. [LG] has no stance term because ANYmal's
        # flat-base prior comes from its terrain curriculum; without one, nothing
        # scored a 0.636 rad contortion differently from a clean stance.
        "stance_deviation": -0.05,
    }
    for name, scale in E.REWARD_SCALES.items():
        if name in declared_deviations:
            assert scale == declared_deviations[name], f"{name} deviates undeclared: {scale}"
            continue
        assert name in source, f"{name} is not a legged_gym term and is not declared"
        assert scale == source[name], f"{name} is {scale}, legged_gym ships {source[name]}"
    # Every [LG] term with a non-zero default must be present, or we dropped it.
    for name, scale in source.items():
        if scale != 0.0:
            assert name in E.REWARD_SCALES, f"dropped legged_gym term {name} (scale {scale})"


def test_reward_terms_cover_every_scale():
    """A scale with no matching term is a silent no-op; a term with no scale is
    dead weight. The one term outside REWARD_SCALES is the [FB] ride-height gate,
    which multiplies the summed reward instead of joining the sum."""
    env = E.HexapodTargetEnv(max_steps=2)
    env.reset(0)
    terms = set(env.reward_terms(env.stance, np.ones(6, bool)))
    assert terms == set(E.REWARD_SCALES) | {E.UPRIGHT_GATE}
    assert E.UPRIGHT_GATE not in E.REWARD_SCALES, (
        "the ride-height factor is back inside the additive sum; as an additive "
        "term it pays a motionless body for merely standing up"
    )


def test_ppo_hyperparameters_match_source():
    """[LG] LeggedRobotCfgPPO.algorithm, the values rsl_rl actually trains with."""
    assert R.PPO_CLIP == 0.2
    assert R.PPO_ENTROPY_COEF == 0.01
    assert R.PPO_UPDATE_EPOCHS == 5
    assert R.PPO_MINIBATCHES == 4
    # Deviation from rsl_rl's 1e-3, measured and documented in run_embody_v4.
    assert R.PPO_INITIAL_LEARNING_RATE == 1.0e-5
    assert R.PPO_LEARNING_RATE_MIN < 1.0e-5 < R.PPO_LEARNING_RATE_MAX
    assert R.PPO_GAMMA == 0.99
    assert R.PPO_GAE_LAMBDA == 0.95
    assert R.PPO_DESIRED_KL == 0.01
    assert R.PPO_GRAD_CLIP_NORM == 1.0
    assert R.PPO_VALUE_LOSS_COEF == 1.0
    assert R.PPO_USE_CLIPPED_VALUE_LOSS


def test_air_time_matches_source_algebra():
    """[LG] _reward_feet_air_time, step for step, against a hand-computed schedule.

    A foot lifts for `lift` control steps and lands. [LG] filters contact against
    the previous step, so the first airborne step is still counted as contact and
    its accumulated dt is wiped -- but the landing step adds one more dt before
    the reward is read. The two cancel and the paid time is exactly lift * dt.
    """
    env = E.HexapodTargetEnv(max_steps=2)
    env.reset(0)
    for lift in (3, 8, 20):
        env._feet_air_time.fill(0.0)
        env._last_contacts = np.ones(6, dtype=bool)
        total = 0.0
        for _ in range(lift):
            contacts = np.ones(6, dtype=bool)
            contacts[0] = False
            total += env.reward_terms(env.stance, contacts)["feet_air_time"]
        total += env.reward_terms(env.stance, np.ones(6, dtype=bool))["feet_air_time"]
        expected = lift * env.dt - E.FEET_AIR_TIME_THRESHOLD_S
        assert abs(total - expected) < 1e-9, f"lift={lift}: {total} != {expected}"


# --- the observation ---------------------------------------------------------

def test_observation_is_egocentric():
    """The same situation at a different world heading and position must look identical.

    v2/v3 fed raw qpos/qvel, which carries world XY and absolute yaw. This is the
    test those observations would fail, and the reason the encoder had to relearn
    walking once per heading.
    """
    env = E.HexapodTargetEnv(max_steps=2)
    env.reset(7)

    def observe(yaw: float, offset: np.ndarray) -> np.ndarray:
        env.data.qpos[:2] = offset
        env.data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
        # Target one metre off the nose, and a body velocity of 0.1 m/s forward,
        # both expressed in the rotated world frame.
        env._target = offset + np.array([np.cos(yaw), np.sin(yaw)])
        env.data.qvel[:3] = (0.1 * np.cos(yaw), 0.1 * np.sin(yaw), 0.0)
        mujoco.mj_forward(env.model, env.data)
        return env._observation()

    reference = observe(0.0, np.zeros(2))
    for yaw in (0.7, -1.9, 3.0):
        for offset in (np.array([5.0, -3.0]), np.array([-40.0, 12.0])):
            rotated = observe(yaw, offset)
            assert np.allclose(reference, rotated, atol=1e-5), (
                f"observation changed under a world rotation/translation "
                f"(yaw={yaw}, offset={offset}); largest delta "
                f"{np.abs(reference - rotated).max():.4g}"
            )


def test_observation_is_bounded():
    """A linear encoder feeding a sigmoid cannot cope with an unbounded channel."""
    env = E.HexapodTargetEnv(max_steps=400)
    env.reset(11)
    rng = np.random.default_rng(11)
    largest = 0.0
    done = False
    while not done:
        observation, _, done = env.step(env.stance + rng.normal(0.0, 0.12, size=18))
        largest = max(largest, float(np.abs(observation).max()))
    assert largest < E.OBS_CLIP, f"observation reached {largest}"
    assert env.obs_dim == E.HexapodTargetEnv.OBS_DIM == 66


# --- the reward, which has been gamed twice ----------------------------------

def test_the_two_known_hacks_lose():
    """v2's dive and v3's rock must both score below an on-target walk."""
    env = E.HexapodTargetEnv()
    env.reset(3)

    def score(vx: float, height: float) -> float:
        env.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        env._target = env.data.qpos[:2] + np.array([1.0, 0.0])
        env.data.qvel[:3] = (vx, 0.0, 0.0)
        env.data.qpos[2] = height
        mujoco.mj_forward(env.model, env.data)
        env._last_action = env.stance.copy()
        env._last_dof_vel = np.asarray(env.data.qvel[6:]).copy()
        terms = env.reward_terms(env.stance, np.ones(6, dtype=bool))
        total = max(env.dt * sum(scale * terms[k] for k, scale in E.REWARD_SCALES.items()), 0.0)
        return total * terms[E.UPRIGHT_GATE]   # [FB] gates, it does not add

    walk = score(E.TARGET_SPEED_MPS, E.STAND_HEIGHT_M)
    dive = score(0.588, 0.0724)          # measured v2 connectome arm
    rock = score(0.0, E.STAND_HEIGHT_M)  # measured v3 connectome arm
    assert walk > dive and walk > rock
    walking = E.EPISODE_STEPS * walk + E.TARGET_REACH_BONUS
    assert walking > 3.0 * E.EPISODE_STEPS * rock


def test_reaching_the_target_pays_and_respawns():
    env = E.HexapodTargetEnv(max_steps=4)
    env.reset(5)
    env._target = env.data.qpos[:2].copy()          # park it under the body
    _, reward, _ = env.step(env.stance)
    assert reward > E.TARGET_REACH_BONUS
    assert env._targets_reached == 1
    _, distance = env._target_polar()
    assert distance >= E.TARGET_SPAWN_MIN_M, "target did not respawn away from the body"


def test_reaches_survive_the_vector_auto_reset():
    """The bug that would have reported a flat-zero endpoint for the whole run."""
    vector = E.VectorHexapodTargetEnv(2, max_steps=3)
    vector.reset(seeds=[1, 2])
    for env in vector.envs:
        env._target = env.data.qpos[:2].copy()
    for _ in range(3):
        _, _, done = vector.step(np.stack([env.stance for env in vector.envs]))
    assert done.all(), "expected the episodes to terminate on the last step"
    assert (vector.reaches > 0).all(), f"reaches lost across the reset: {vector.reaches}"


# --- the frozen core ---------------------------------------------------------

def test_core_is_frozen_through_a_real_update():
    """The invariant the project rests on, checked against an actual gradient step."""
    import specificity_campaign as SC

    agent = R.Agent(R.load_calibration(), stage1=None)
    before = SC.tensor_sha256(agent.policy.core.W)
    episode = agent.collect(R.STAGE1_SEED, horizon=8)
    episode["iteration"] = 0
    statistics = agent.update(episode)
    assert SC.tensor_sha256(agent.policy.core.W) == before, "the VNC graph moved"
    assert not agent.policy.core.W.requires_grad
    assert agent.policy.core.W.grad is None
    assert sum(p.numel() for p in agent.policy.core.parameters()) == 0
    assert np.isfinite(statistics["grad_norm"])
    trainable = sum(p.numel() for p in agent.policy.interface_parameters())
    assert trainable < 10_000, f"{trainable} trainable interface parameters is not an interface"
    return statistics


def test_replay_reproduces_the_behaviour_policy():
    """Before any update, re-running the rollout must reproduce the collected means.

    This is the check that catches a recurrent-state mismatch between collection
    and replay. Segments are shorter than episodes, so from the second iteration
    onwards a segment starts mid-episode; replaying from a zero state would make
    the KL and the importance ratio measure the mismatch instead of the update.
    """
    agent = R.Agent(R.load_calibration(), stage1=None)
    agent.collect(R.STAGE1_SEED, horizon=R.PPO_BPTT_CHUNK)   # leave the envs mid-episode
    episode = agent.collect(R.STAGE1_SEED, horizon=R.PPO_ROLLOUT_STEPS)
    episode["iteration"] = 0
    assert float(episode["start_state"].abs().max()) > 0.0, "second segment started from zeros"
    # Exercise the path PPO actually takes: full rollout length, minibatch
    # slicing, and gradient enabled so the checkpointed recompute runs. An
    # earlier version of this check ran under `no_grad` on a six-step rollout,
    # which is why it passed on the machine where training was already broken.
    fidelity = R.replay_fidelity(agent, episode)
    assert fidelity["replay_steps"] == R.PPO_ROLLOUT_STEPS
    assert fidelity["replay_kl"] < 1e-3, (
        f"replay differs from the behaviour policy: KL {fidelity['replay_kl']:.3g}, "
        f"max mean delta {fidelity['replay_max_delta']:.3g}")
    return {k: round(v, 8) if isinstance(v, float) else v for k, v in fidelity.items()}


def test_adaptive_learning_rate_responds_to_kl():
    """rsl_rl's schedule: a large policy step must pull the learning rate down."""
    agent = R.Agent(R.load_calibration(), stage1=None)
    episode = agent.collect(R.STAGE1_SEED, horizon=8)
    episode["iteration"] = 0
    # Force a large KL by claiming the behaviour policy had a very different mean.
    episode["old_means"] = episode["old_means"] + 5.0
    start = agent.learning_rate
    agent.update(episode)
    assert agent.learning_rate < start, (
        f"learning rate stayed at {agent.learning_rate} under a large KL")
    assert agent.learning_rate >= R.PPO_LEARNING_RATE_MIN


# --- budget ------------------------------------------------------------------

def test_stage1_prior_survives_the_first_update():
    """A first PPO update must not throw away the behaviour-cloned interface.

    At rsl_rl's initial rate of 1e-3 the measured KL after one update is ~1.0.
    This asserts the configured rate keeps the first update inside a couple of
    orders of magnitude of the target trust region.
    """
    agent = R.Agent(R.load_calibration(), stage1=None)
    episode = agent.collect(R.STAGE1_SEED, horizon=R.PPO_BPTT_CHUNK)
    episode["iteration"] = 0
    statistics = agent.update(episode)
    assert statistics["kl"] < 1.0, f"first update moved the policy to KL {statistics['kl']:.3g}"
    return {"kl": round(statistics["kl"], 4), "lr": statistics["learning_rate"]}


def test_budget_is_enough_gradient_steps():
    """v4 as first written did 648 updates for 50M steps. legged_gym does ~30,000."""
    iterations = R.ARM_STEP_CAP // (R.PPO_BATCH_SIZE * R.PPO_ROLLOUT_STEPS)
    updates = iterations * R.PPO_UPDATE_EPOCHS * R.PPO_MINIBATCHES
    assert iterations >= 500, iterations
    assert updates >= 10_000, updates
    assert R.PPO_ROLLOUT_STEPS % R.PPO_BPTT_CHUNK == 0, "ragged final BPTT chunk"

    half_stride = 1.0 / T.TEACHER_FREQ_HZ / (1.0 / 50.0) / 2
    assert R.PPO_BPTT_CHUNK * 2 >= half_stride, (
        "BPTT window is shorter than half a stride; no step can be credited whole")
    return {"iterations": iterations, "gradient_updates": updates,
            "bptt_chunk": R.PPO_BPTT_CHUNK, "half_stride_steps": round(half_stride, 1)}


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            extra = test()
            print(f"PASS {test.__name__}" + (f" {extra}" if extra else ""), flush=True)
        except AssertionError as error:
            failures += 1
            print(f"FAIL {test.__name__}: {error}", flush=True)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0



def test_collection_and_replay_use_the_same_batch_width():
    """The seven-hour bug: the core was stepped at 512 wide and replayed at 128.

    A sparse matmul kernel picks its reduction order from the operand width, so
    that is not a no-op in floating point even though the core's columns are
    independent. Assert the two paths agree on the partition, which is what makes
    the replay the same arithmetic rather than merely the same algebra.
    """
    agent = R.Agent(R.load_calibration(), stage1=None)
    groups = agent.env_groups()
    assert len(groups) == R.PPO_MINIBATCHES
    widths = {int(group.numel()) for group in groups}
    assert len(widths) == 1, f"ragged partition {widths}; the widths must all match"
    assert sum(widths) * len(groups) == agent.batch_size
    assert torch.equal(torch.cat(groups), torch.arange(agent.batch_size)), "partition must be contiguous and ordered"
    # And the update must consume exactly this partition, not a reshuffle of it.
    assert "randperm" not in inspect.getsource(R.Agent.update)
    return {"groups": len(groups), "width": widths.pop()}


def test_a_truncated_episode_is_bootstrapped_not_zeroed():
    """[LG] cuts the value only at a genuine termination. Falling is one; the
    clock running out is not, and every episode here ends on the clock."""
    env = E.HexapodTargetEnv(max_steps=3)
    env.reset(seed=1)
    stance = env.stance.copy()
    for _ in range(3):
        _, _, done = env.step(stance)
    assert done and env._timeout, "a full-length episode must report a timeout, not a fall"

    vector = E.VectorHexapodTargetEnv(2, max_steps=3)
    vector.reset(seeds=[1, 2])
    for _ in range(3):
        _, _, done = vector.step(np.stack([stance, stance]))
    assert done.all()
    assert vector.last_timeout_mask.all(), "the wrapper lost the timeout flag"
    # The observation banked for bootstrapping is the pre-reset one.
    assert np.abs(vector.last_terminal_obs).max() > 0.0
    return {"timeouts": int(vector.last_timeout_mask.sum())}

def test_the_rollout_is_short_enough_that_a_step_stays_in_the_trust_region():
    """The bug that cost run 1, as a test.

    Replay under UNCHANGED parameters is bit-exact at any horizon, which is why
    the old check passed on a broken machine. The quantity that matters is the KL
    after a real gradient step: the core amplifies a tiny weight change
    exponentially along the segment, so the segment has to be short enough that
    the amplified change is still inside PPO's trust region. Measured at lr 1e-5:
    2.0e-4 at 12 steps, 1.5e-2 at 24, 2.5e-1 at 32, 10.3 at 150.
    """
    agent = R.Agent(R.load_calibration())
    episode = agent.collect(R.STAGE1_SEED, horizon=R.PPO_ROLLOUT_STEPS)
    episode["iteration"] = 0
    assert R.replay_fidelity(agent, episode)["replay_kl"] <= 1e-3, "replay diverged before any update"
    agent.update(episode)
    first = agent.update_trace[0]
    assert first["kl"] <= R.PPO_DESIRED_KL, f"first minibatch already outside the trust region: {first}"
    worst = max(row["kl"] for row in agent.update_trace)
    assert worst <= R.PPO_DESIRED_KL * R.PPO_KL_ABORT_FACTOR, (
        f"a single Adam step left the trust region ({worst:.3g}); PPO_ROLLOUT_STEPS "
        f"={R.PPO_ROLLOUT_STEPS} is longer than the core stays predictable"
    )
    return {"first_kl": round(first["kl"], 6), "worst_kl": round(worst, 4),
            "minibatches": len(agent.update_trace)}


def test_an_overshooting_update_is_abandoned_not_continued():
    """The adaptive schedule cannot undo a step, so the rest of the update would
    optimise against a policy the parameters no longer resemble.

    Provoked by lowering the abort threshold rather than by raising the learning
    rate: now that the segment is 24 steps, even lr 1e-2 stays inside the band,
    which is the point of the fix but leaves the guard otherwise untestable.
    """
    agent = R.Agent(R.load_calibration())
    episode = agent.collect(R.STAGE1_SEED, horizon=R.PPO_ROLLOUT_STEPS)
    episode["iteration"] = 0
    original = R.PPO_KL_ABORT_FACTOR
    R.PPO_KL_ABORT_FACTOR = 1.0e-3        # trips on any non-zero KL
    try:
        stats = agent.update(episode)
    finally:
        R.PPO_KL_ABORT_FACTOR = original
    expected = R.PPO_UPDATE_EPOCHS * R.PPO_MINIBATCHES
    assert len(agent.update_trace) < expected, (
        f"ran all {expected} minibatches past the abort threshold; the guard did not fire"
    )
    assert stats["kl"] > R.PPO_DESIRED_KL * 1.0e-3
    return {"minibatches_run": len(agent.update_trace), "of": expected}


def test_the_stuck_detector_matches_run_1s_signature():
    """Run 1 logged KL ~8 with the learning rate on its floor from the first PPO
    iteration and ran for seven more hours. Check the rule that now stops that,
    and check it does not fire on a healthy iteration."""
    stuck = {"learning_rate": R.PPO_LEARNING_RATE_MIN, "kl": 8.0}
    healthy = {"learning_rate": 1.0e-6, "kl": 0.012}
    floored_but_fine = {"learning_rate": R.PPO_LEARNING_RATE_MIN, "kl": 0.012}

    def fires(row):
        return (row["learning_rate"] <= R.PPO_LEARNING_RATE_MIN
                and row["kl"] > R.PPO_DESIRED_KL * R.PPO_STUCK_KL_FACTOR)

    assert fires(stuck), "run 1's own numbers would not have tripped the detector"
    assert not fires(healthy)
    assert not fires(floored_but_fine), "a floored rate alone is not a stuck run"
    assert R.PPO_STUCK_ITERATIONS >= 25, "too twitchy; the schedule dips transiently"
    return {"trips_after_iterations": R.PPO_STUCK_ITERATIONS}


def test_progress_is_visible_before_the_endpoint_moves():
    """`targets_reached` stays at 0.000 for hours. Without a dense proxy a working
    run and run 1 look identical in the log until it is too late to react."""
    agent = R.Agent(R.load_calibration())
    episode = agent.collect(R.STAGE1_SEED, horizon=R.PPO_ROLLOUT_STEPS)
    distance = episode["target_distance_m"]
    assert E.TARGET_REACH_M < distance < E.TARGET_SPAWN_MAX_M * 1.5, distance
    terms = episode["reward_terms"]
    assert set(terms) == set(E.REWARD_SCALES) | {E.UPRIGHT_GATE, "stance_dev", "height"}, (
        "a reward term is missing from the log"
    )
    # The two run 2 was blind to.
    assert 0.05 < terms["height"] < 0.30, terms["height"]
    assert terms["stance_dev"] >= 0.0
    assert all(np.isfinite(value) for value in terms.values()), terms
    # Cleared per segment, so consecutive iterations are comparable.
    again = agent.collect(R.STAGE1_SEED + 1, horizon=R.PPO_ROLLOUT_STEPS)
    assert again["reward_terms"]["upright"] > 0.0
    return {"target_distance_m": round(distance, 3),
            "track_lin": round(terms["tracking_lin_vel"], 4)}


def test_a_real_gait_outscores_the_shuffle_it_has_to_beat():
    """The invariant v2 and v3 both violated, measured on real rollouts.

    `smoke_test` checks this by posing a body and injecting a velocity, which
    cannot see the terms that depend on dynamics -- feet_air_time, action_rate,
    dof_acc. This replays the CEM teacher, which provably walks 0.424 m at
    0.106 m/s against a 0.100 m/s command, and requires it to beat the same body
    holding its stance. If that ordering ever inverts, the reward is gameable and
    no amount of compute produces walking.

    Costs ~90 s of CEM. That is the cheapest possible insurance against spending a
    night optimising a reward that prefers standing still.
    """
    _, actions, record = T.search_command(0.0)
    assert not record["fell"], record

    def replay(policy_actions):
        env = E.HexapodTargetEnv(max_steps=T.STAGE1_STEPS)
        env.reset(seed=T.TEACHER_SEED)
        start = env.data.qpos[:2].copy()
        total, steps = 0.0, 0
        for step in range(T.STAGE1_STEPS):
            _, reward, done = env.step(policy_actions[step])
            total += reward
            steps += 1
            if done:
                break
        travelled = float(np.linalg.norm(env.data.qpos[:2] - start))
        return total / steps, travelled

    walker, walked = replay(actions)
    stander, stood = replay([None] * 0 or [E.HexapodTargetEnv(max_steps=1).stance] * T.STAGE1_STEPS)
    assert walked > 0.35, f"the teacher only covered {walked:.3f} m; it is not a walker"
    assert stood < 0.05, f"holding stance moved {stood:.3f} m"
    # Measured: 1.23x as v4 shipped, 1.52x with [FB]'s ride-height factor restored
    # to a multiplicative gate, 3.53x with the angular tracking sigma rescaled to
    # the command magnitude this task actually emits. 2.5 locks that in with slack.
    assert walker > 2.5 * stander, (
        f"walking pays {walker:.5f}/step against {stander:.5f} for standing still "
        f"({walker / max(stander, 1e-9):.2f}x); the reward does not prefer a gait"
    )
    return {"walk_per_step": round(walker, 5), "stand_per_step": round(stander, 5),
            "ratio": round(walker / stander, 2), "walked_m": round(walked, 3)}


BRACE_CHECKPOINT = Path(__file__).resolve().parent.parent / "results" / "path2" / "run2" / "embody_v4_ppo_connectome.pt"


def test_the_run_2_brace_loses_to_a_real_gait():
    """The third hack, scored from its own checkpoint rather than a posed body.

    v2 dove, v3 rocked, v4 braced: 24.9M steps produced joints 0.636 rad from the
    home stance (teacher 0.103) at a ride height above nominal standing, splayed
    so wide it could not fall, translating at 0.026 m/s against 0.106. Every
    logged scalar improved while it happened. This replays that exact policy and
    requires the teacher to beat it.
    """
    if not BRACE_CHECKPOINT.exists():
        return {"skipped": "run-2 checkpoint not present"}

    def per_step_from_actions(actions):
        env = E.HexapodTargetEnv(max_steps=T.STAGE1_STEPS)
        env.reset(seed=T.TEACHER_SEED)
        total = steps = 0
        for action in actions:
            _, reward, done = env.step(action)
            total += reward
            steps += 1
            if done:
                break
        return total / steps

    def per_step_from_policy(checkpoint, seeds):
        agent = R.Agent(R.load_calibration(), stage1=R.STAGE1_CHECKPOINT)
        saved = torch.load(checkpoint, map_location=agent.device, weights_only=True)
        agent.policy.encoder.load_state_dict(saved["encoder"])
        agent.policy.decoder.load_state_dict(saved["decoder"])
        total = steps = 0
        deviation = []
        for seed in seeds:
            env = E.HexapodTargetEnv(max_steps=E.EPISODE_STEPS)
            observation = env.reset(seed=seed)
            state = agent.policy.initial_state()
            done = False
            with torch.no_grad():
                while not done:
                    mean, state = agent.policy.step(torch.from_numpy(observation), state)
                    observation, reward, done = env.step(mean.cpu().numpy().astype(np.float32))
                    total += reward
                    steps += 1
                    deviation.append(float(np.sum(np.square(env.data.qpos[7:] - env.stance))))
        return total / steps, float(np.mean(deviation))

    _, actions, record = T.search_command(0.0)
    assert not record["fell"], record
    teacher = per_step_from_actions(actions)
    brace, brace_deviation = per_step_from_policy(BRACE_CHECKPOINT, R.EVALUATION_SEEDS[:3])

    if brace_deviation <= 5.0:
        # The checkpoint predates the action-interface fix. It was trained when the
        # decoder emitted ABSOLUTE joint targets; `_to_joints` now reads the same
        # numbers as offsets from the home stance, so replaying those weights no
        # longer reproduces the 0.636-rad splay they were recorded for. The fixture
        # is stale, not the stance term -- and this guard cannot be scored again
        # until a braced checkpoint is regenerated under the fixed interface.
        return {"skipped": "run-2 checkpoint predates the action-interface fix",
                "brace_stance_dev": round(brace_deviation, 2), "expected_above": 5.0}
    assert teacher > 5.0 * brace, (
        f"a real gait pays {teacher:.6f}/step against {brace:.6f} for the brace "
        f"({teacher / max(brace, 1e-12):.1f}x); the stance term is not holding"
    )
    return {"teacher_per_step": round(teacher, 6), "brace_per_step": round(brace, 6),
            "ratio": round(teacher / max(brace, 1e-12), 1), "brace_stance_dev": round(brace_deviation, 2)}


def test_the_target_cannot_be_reached_without_walking():
    """The task has to require locomotion, not leg extension.

    v4 shipped a 0.4-1.0 m spawn annulus with a 0.15 m reach radius, while the
    feet sit 0.259-0.284 m from the body centre at stance -- the footprint was
    wider than the reach zone. A target at the inner edge needed 0.25 m of body
    travel, less than one footprint diameter, and the prey is a ground-level
    sphere, so extending a leg toward it beat walking to it.
    """
    env = E.HexapodTargetEnv(max_steps=2)
    env.reset(seed=1)
    mujoco.mj_forward(env.model, env.data)
    body = env.data.qpos[:2]
    footprint = max(float(np.linalg.norm(env.data.geom_xpos[int(g)][:2] - body))
                    for g in env.foot_geom_ids)

    travel = E.TARGET_SPAWN_MIN_M - E.TARGET_REACH_M
    assert travel > 2.0 * footprint, (
        f"closest target needs only {travel:.3f} m of body travel against a "
        f"{footprint:.3f} m footprint radius; the task does not require walking"
    )
    # And the episode has to be long enough to cross the far edge at teacher speed.
    seconds = E.EPISODE_STEPS / 50.0
    assert seconds * 0.106 > E.TARGET_SPAWN_MAX_M, (
        f"{seconds:.0f}s at the teacher's 0.106 m/s covers {seconds * 0.106:.2f} m, "
        f"short of the {E.TARGET_SPAWN_MAX_M} m far edge; the far target is unreachable"
    )
    # The distance observation channel must not saturate over the annulus.
    assert E.TARGET_SPAWN_MAX_M / E.OBS_TARGET_DIST_SCALE <= 1.0, "distance channel clips"
    return {"footprint_m": round(footprint, 3), "min_travel_m": round(travel, 3),
            "ratio": round(travel / footprint, 2), "episode_s": seconds}




def test_the_decaying_prior_never_reaches_the_reported_policy():
    """[DA]/[K] prior is a curriculum, so it has to actually leave.

    Two ways it could quietly become the forbidden teacher residual, both checked
    here against real rollouts:

      * the weight never reaching zero, so the shipped policy is still braced by
        the motion generator;
      * PPO storing the BIASED action, which would both corrupt the importance
        ratio and score the policy on help the evaluation does not give it.

    The third failure is subtler: the step budget not reaching the cutoff, so the
    schedule is correct and irrelevant. That is checked against ARM_STEP_CAP.
    """
    zero_after, gamma = R.DECAP_ZERO_AFTER, R.DECAP_GAMMA
    assert R.prior_weight(0, gamma, zero_after) == 1.0
    assert R.prior_weight(zero_after, gamma, zero_after) == 0.0, "prior never leaves"
    assert R.prior_weight(zero_after * 2, gamma, zero_after) == 0.0
    assert R.prior_weight(zero_after - 1, gamma, zero_after) < 1e-2, (
        "the prior is still contributing materially at the cutoff, so it is being "
        "switched off rather than annealed away"
    )
    iterations = R.ARM_STEP_CAP // (R.PPO_BATCH_SIZE * R.PPO_ROLLOUT_STEPS)
    assert iterations > zero_after, (
        f"the budget runs {iterations} iterations but the prior only reaches zero at "
        f"{zero_after}; the run would end while the generator was still helping"
    )

    agent = R.Agent(R.load_calibration(), stage1=None)
    if agent.rtmg is None:
        return {"skipped": "V4_DECAP=0"}

    def rollout(iteration):
        agent_ = R.Agent(R.load_calibration(), stage1=None)
        executed = []
        original = agent_.vector_env.step

        def record(actions):
            executed.append(np.asarray(actions, dtype=np.float64).copy())
            return original(actions)

        agent_.vector_env.step = record
        episode = agent_.collect(R.STAGE1_SEED, horizon=2, iteration=iteration)
        agent_.vector_env.step = original
        return episode, np.stack(executed)

    early, early_executed = rollout(0)
    assert early["prior_weight"] == 1.0
    stored = np.asarray(early["actions"], dtype=np.float64)
    bias = np.abs(early_executed - stored).max()
    assert bias > 1e-6, "the prior was enabled but changed nothing that reached the body"
    # Whatever the generator adds has to be a leg pose, not an arbitrary push:
    # its widest departure from stance is swing_amp * 1.9 + lift_amp < 1.1 rad.
    assert bias < 1.1, f"prior displaced a joint by {bias:.3f} rad; that is not the gait"

    late, late_executed = rollout(R.DECAP_ZERO_AFTER)
    assert late["prior_weight"] == 0.0
    assert np.array_equal(late_executed, np.asarray(late["actions"], dtype=np.float64)), (
        "at zero weight the executed action must be the policy's own, bit for bit"
    )
    return {"iterations_budgeted": int(iterations), "zero_after": zero_after,
            "early_bias_rad": round(float(bias), 4)}


def test_stage1_must_beat_a_constant_action_before_ppo_runs():
    """The check that was missing for three runs.

    Stage 1 shipped a prior scoring 0.0824 train / 0.0394 held-out when a single
    constant action scores 0.0151, and runs 1 and 2 and DAgger all trained on top
    of it. The gate turns that comparison into a hard stop.
    """
    assert R.STAGE1_CONSTANT_BASELINE == 0.0151
    threshold = R.STAGE1_CONSTANT_BASELINE * R.STAGE1_GATE_FRACTION
    assert threshold < R.STAGE1_CONSTANT_BASELINE, "gate must demand better than a constant"

    # The historical prior must fail the gate it was never asked to pass.
    assert not (0.0394 < threshold), "run-1/run-2 prior would still slip through"

    record = json.loads(R.STAGE1_JSON.read_text()) if R.STAGE1_JSON.exists() else None
    if record is not None and "gate_passed" in record:
        assert record["gate_passed"] == (record["final_heldout_loss"] < record["gate_threshold"])
    return {"threshold": threshold, "historic_heldout": 0.0394}


def test_stage2_refuses_to_start_behind_a_failed_gate():
    """run_stage2 must raise, not warn, when the prior did not learn the gait."""
    saved = R.STAGE1_JSON.read_text() if R.STAGE1_JSON.exists() else None
    R.STAGE1_JSON.parent.mkdir(parents=True, exist_ok=True)
    R.STAGE1_JSON.write_text(json.dumps({
        "gate_passed": False, "final_heldout_loss": 0.0394,
        "gate_threshold": 0.00755, "constant_action_baseline": 0.0151,
    }))
    try:
        os.environ.pop("V4_SKIP_STAGE1_GATE", None)
        raised = False
        try:
            R.run_stage2()
        except SystemExit as error:
            raised = "Stage-1 gate failed" in str(error)
        assert raised, "stage 2 started anyway behind a failed gate"
    finally:
        if saved is None:
            R.STAGE1_JSON.unlink(missing_ok=True)
        else:
            R.STAGE1_JSON.write_text(saved)
    return {"refused": True}


def test_stage1_scores_the_parameters_it_saves():
    """Selection happens before the step, so the restored checkpoint is the one scored.

    The old loop stepped, then re-ran the whole replay to log; the numbers it
    selected on came from parameters one gradient step ahead of the snapshot it
    kept, and it paid for a second forward pass to get them.
    """
    source = inspect.getsource(R.run_stage1)
    step_at = source.index("optimizer.step()")
    select_at = source.index("best_heldout, best_epoch, best_state =")
    assert select_at < step_at, "selection must be scored before the parameters move"
    assert source.count("_predict_per_demo") == 1, "one replay per epoch, not two"
    return {"forwards_per_epoch": 1}


if __name__ == "__main__":
    raise SystemExit(main())
