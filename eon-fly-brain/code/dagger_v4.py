"""DAgger for the v4 body interface: the decision gate before any more compute.

Why this exists
---------------
v4 run 2 trained cleanly for 24.9M steps and did not walk. Two measurements say
the blocker is the training procedure rather than the connectome:

  * The CEM teacher walks 0.424 m at 0.106 m/s on the identical body, PD gains,
    action space and control rate. The body is not the limit.
  * Stage 1 fits that teacher's actions to MSE 2.8e-4 through the 4-latent
    anatomical map -- it reproduces a real gait almost exactly ON THE TEACHER'S
    OWN TRAJECTORY -- and then walks at 0.034 m/s closed loop.

A model that predicts expert actions well on expert states and fails when it
drives is the textbook signature of compounding error, and DAgger (Ross, Gordon
and Bagnell, AISTATS 2011) is the textbook fix: roll out the *current policy*,
label the states *it* visits with what the expert would have done there, add
those to the dataset, refit, repeat.

This is a Tier 1 assist under `EMBODY_V5_PLAN.md`. It adds no capacity, touches
no weights outside the encoder and decoder, and changes nothing about the task.
The frozen core stays frozen and is asserted so every round.

The gate
--------
If closed-loop speed moves materially toward the teacher's 0.106 m/s, the
interface was never the ceiling and v5 proceeds to a pod run. If a good fit still
leaves it near 0.034 m/s, the 4-latent anatomical map genuinely cannot drive this
gait closed loop -- which is a connectome result worth reporting rather than
engineering around.

Labelling note
--------------
The teacher is an open-loop periodic controller, a function of gait phase rather
than of state, so the label for a policy-visited state at step t is the teacher's
action at that same step index. That is DAgger with a time-indexed expert, which
is what locomotion imitation normally uses: the label says "be at the gait phase
you should be at by now", and the policy is what drifted.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

import specificity_campaign as SC
import teacher_v4 as T
from audit_embody_3way import STAGE1_LEARNING_RATE, STAGE1_MIN_DELTA, STAGE1_PATIENCE, STAGE1_SEED
from embody_env_v4 import EPISODE_STEPS, HexapodTargetEnv
from run_embody_v4 import (
    RESULTS,
    STAGE1_CHECKPOINT,
    Agent,
    _predict_per_demo,
    assert_frozen,
    load_calibration,
)

DAGGER_ROUNDS = 4
DAGGER_EPOCHS = 40           # per round; each epoch is now several gradient steps
# Demos per gradient step. The core state is (11000, demos) and the graph holds
# 200 steps x 4 substeps of it, so peak memory is linear in this number. Stage 1
# runs 5 demos and is known to fit; the aggregated DAgger set reaches 25, and
# batching all of them OOM-killed the first attempt. Chunking here keeps peak
# memory at Stage 1's known-good level regardless of how many rounds accumulate.
DAGGER_DEMO_BATCH = 5
DAGGER_STEPS = T.STAGE1_STEPS
DAGGER_CHECKPOINT = RESULTS / "embody_v4_dagger_connectome.pt"
DAGGER_JSON = RESULTS / "embody_v4_dagger.json"
SPEED_SEEDS = (20260903, 20260904, 20260905, 20260906)


def _teacher_params(record: dict) -> dict[float, np.ndarray]:
    return {float(row["yaw_rate"]): np.asarray(row["params"], dtype=np.float64).reshape(6, 4)
            for row in record["per_command"]}


def policy_rollout_with_teacher_labels(agent: Agent, params: np.ndarray, yaw_rate: float,
                                       steps: int = DAGGER_STEPS) -> tuple[np.ndarray, np.ndarray]:
    """Drive with the POLICY; label every visited state with the teacher's action.

    Target placement is copied from `teacher_v4.rollout` so the three target
    observation channels mean the same thing they meant during Stage 1. Without
    that the relabelled states would carry a bearing the teacher was never
    conditioned on.
    """
    env = HexapodTargetEnv(max_steps=steps)
    env.reset(T.TEACHER_SEED)
    bearing = 2.0 * yaw_rate

    def place_target() -> None:
        angle = env.heading() + bearing
        env._target = env.data.qpos[:2] + T.TEACHER_TARGET_DISTANCE_M * np.array(
            [math.cos(angle), math.sin(angle)]
        )

    place_target()
    observation = env._observation()
    state = agent.policy.initial_state()
    observations, labels = [], []
    with torch.no_grad():
        for step in range(steps):
            observations.append(observation)
            labels.append(T.teacher_action(params, step, env.stance, env.dt))
            mean, state = agent.policy.step(torch.from_numpy(observation), state)
            observation, _, done = env.step(mean.cpu().numpy().astype(np.float32))
            place_target()
            observation = env._observation()
            if done and step + 1 < steps:
                # A fall ends the rollout. Hold the terminal state so every demo
                # keeps the same length and the batched refit stays rectangular;
                # the labels keep advancing, which is the correction we want.
                observations.extend([observations[-1]] * (steps - len(observations)))
                labels.extend([T.teacher_action(params, index, env.stance, env.dt)
                               for index in range(step + 1, steps)])
                break
    return (np.asarray(observations, dtype=np.float32)[:steps],
            np.asarray(labels, dtype=np.float32)[:steps])


def closed_loop_speed(agent: Agent, seeds=SPEED_SEEDS) -> dict[str, float]:
    """The number the gate turns on. Teacher reference: 0.424 m at 0.106 m/s."""
    distances, speeds, uprights, deviations = [], [], [], []
    for seed in seeds:
        env = HexapodTargetEnv(max_steps=EPISODE_STEPS)
        observation = env.reset(seed=seed)
        state = agent.policy.initial_state()
        start = env.data.qpos[:2].copy()
        steps, done = 0, False
        with torch.no_grad():
            while not done:
                mean, state = agent.policy.step(torch.from_numpy(observation), state)
                observation, _, done = env.step(mean.cpu().numpy().astype(np.float32))
                steps += 1
                deviations.append(float(np.sum(np.square(env.data.qpos[7:] - env.stance))))
        distance = float(np.linalg.norm(env.data.qpos[:2] - start))
        seconds = steps / env.control_hz
        distances.append(distance)
        speeds.append(distance / seconds)
        uprights.append(seconds)
    return {"distance_m": float(np.mean(distances)), "speed_mps": float(np.mean(speeds)),
            "upright_s": float(np.mean(uprights)), "stance_dev": float(np.mean(deviations))}


def fit(agent: Agent, observations: torch.Tensor, targets: torch.Tensor, epochs: int) -> float:
    """Refit encoder and decoder on the aggregated set. Core stays frozen."""
    optimizer = torch.optim.Adam(agent.policy.interface_parameters(), lr=STAGE1_LEARNING_RATE)
    assert_frozen(agent.policy, optimizer)
    best, best_epoch, best_state = float("inf"), 0, None
    generator = torch.Generator().manual_seed(STAGE1_SEED)
    demos = observations.shape[0]
    for epoch in range(1, epochs + 1):
        order = torch.randperm(demos, generator=generator)
        losses = []
        for chunk in order.split(DAGGER_DEMO_BATCH):
            prediction = _predict_per_demo(agent.policy, observations[chunk])
            loss = torch.mean((prediction - targets[chunk]) ** 2)
            if not torch.isfinite(loss):
                raise AssertionError(f"non-finite DAgger loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_loss = float(np.mean(losses))
        if epoch_loss < best - STAGE1_MIN_DELTA:
            best, best_epoch = epoch_loss, epoch
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in agent.policy.encoder.state_dict().items()},
                "decoder": {k: v.detach().clone() for k, v in agent.policy.decoder.state_dict().items()},
            }
        elif epoch - best_epoch >= STAGE1_PATIENCE:
            break
    if best_state is not None:
        agent.policy.encoder.load_state_dict(best_state["encoder"])
        agent.policy.decoder.load_state_dict(best_state["decoder"])
    assert_frozen(agent.policy, optimizer)
    return best


def run_dagger(rounds: int = DAGGER_ROUNDS) -> dict:
    started = time.perf_counter()
    torch.manual_seed(STAGE1_SEED)
    calibration = load_calibration()
    agent = Agent(calibration, stage1=STAGE1_CHECKPOINT)   # start from the BC fit
    core_before = SC.tensor_sha256(agent.policy.core.W)
    device = agent.device

    print(json.dumps({"stage": "teacher"}), flush=True)
    base_observations, base_actions, record = T.build_teacher(DAGGER_STEPS)
    params_by_command = _teacher_params(record)

    observations = torch.from_numpy(base_observations).to(device)
    targets = torch.from_numpy(base_actions).to(device)

    baseline = closed_loop_speed(agent)
    print(json.dumps({"round": 0, "note": "behaviour-cloned prior", **baseline}, sort_keys=True), flush=True)
    history = [{"round": 0, "demos": int(observations.shape[0]), **baseline}]

    for round_index in range(1, rounds + 1):
        new_observations, new_targets = [], []
        for yaw_rate, params in params_by_command.items():
            visited, labels = policy_rollout_with_teacher_labels(agent, params, yaw_rate)
            new_observations.append(visited)
            new_targets.append(labels)
        observations = torch.cat(
            (observations, torch.from_numpy(np.asarray(new_observations, dtype=np.float32)).to(device)), dim=0
        )
        targets = torch.cat(
            (targets, torch.from_numpy(np.asarray(new_targets, dtype=np.float32)).to(device)), dim=0
        )
        loss = fit(agent, observations, targets, DAGGER_EPOCHS)
        measured = closed_loop_speed(agent)
        row = {"round": round_index, "demos": int(observations.shape[0]), "fit_mse": loss, **measured}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if SC.tensor_sha256(agent.policy.core.W) != core_before:
        raise AssertionError("VNC graph changed during DAgger")

    DAGGER_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": agent.policy.encoder.state_dict(),
                "decoder": agent.policy.decoder.state_dict()}, DAGGER_CHECKPOINT)
    final = history[-1]
    result = {
        "rounds": rounds,
        "teacher_speed_mps": 0.1059,
        "baseline_speed_mps": baseline["speed_mps"],
        "final_speed_mps": final["speed_mps"],
        "speed_gain": final["speed_mps"] / max(baseline["speed_mps"], 1e-9),
        "fraction_of_teacher": final["speed_mps"] / 0.1059,
        "core_frozen": True,
        "core_tensor_sha256": core_before,
        "history": history,
        "wall_time_s": time.perf_counter() - started,
    }
    DAGGER_JSON.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def self_check() -> dict:
    """One cheap runnable check: labels advance in phase and the core stays frozen."""
    agent = Agent(load_calibration(), stage1=STAGE1_CHECKPOINT)
    before = SC.tensor_sha256(agent.policy.core.W)
    params = np.tile(np.asarray(T.TEACHER_INITIAL_MEAN, dtype=np.float64), (6, 1))
    visited, labels = policy_rollout_with_teacher_labels(agent, params, 0.0, steps=20)
    assert visited.shape == (20, HexapodTargetEnv.OBS_DIM), visited.shape
    assert labels.shape == (20, 18), labels.shape
    assert np.isfinite(visited).all() and np.isfinite(labels).all()
    assert float(np.abs(labels[0] - labels[5]).max()) > 0.0, "teacher labels are not advancing in phase"
    assert SC.tensor_sha256(agent.policy.core.W) == before
    return {"visited": visited.shape, "labels": labels.shape, "core_frozen": True}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--rounds", type=int, default=DAGGER_ROUNDS)
    arguments = parser.parse_args()
    if arguments.command == "check":
        print("DAGGER SELF-CHECK:", self_check())
    else:
        output = run_dagger(arguments.rounds)
        print("DAGGER:", json.dumps({k: v for k, v in output.items() if k != "history"},
                                    indent=2, sort_keys=True, default=str))
