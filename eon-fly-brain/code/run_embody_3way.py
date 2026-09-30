"""PPO stage, fixed-checkpoint evaluation, and final reporting for EMBODY-3WAY v3."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.distributions import Normal

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))

import specificity_campaign as SC  # noqa: E402
from analyze_control_study import bootstrap_ci, wilson_interval  # noqa: E402
from audit_embody_3way import (  # noqa: E402
    CONDITIONS,
    DECISION_LABELS,
    EVALUATION_SEEDS,
    PPO_BATCH_SIZE,
    PPO_CHECKPOINT_INTERVAL,
    PPO_CLIP,
    PPO_ENTROPY_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_STEPS_PER_ARM,
    PPO_VALUE_COEF,
    PPO_VALUE_WARMUP_STEPS,
    PREREGISTERED,
    RESULTS as EMBODY_RESULTS,
    STAGE1_CHECKPOINTS,
    STAGE1_SEED,
    TAG,
    array_sha256,
    calibration,
    json_hash,
    _policy,
)
from embody_env import TARGET_SPEED_MPS, UPRIGHT_HEIGHT_M, HexapodEnv, VectorHexapodEnv  # noqa: E402
from hexapod_gait import H_MIN  # noqa: E402
from embody_policy import COMMAND_MAP_SHA256, EmbodyPolicy, resolve_device  # noqa: E402
from audit_t2_forward_readout import one_sided_sign_flip, paired_forward  # noqa: E402

# v3 artifact names; see the TAG comment in audit_embody_3way.
ADDENDUM = RESULTS / f"embody_3way_{TAG}_stage2_addendum.json"
FINAL_JSON = RESULTS / f"embody_3way_{TAG}.json"
FINAL_CSV = RESULTS / f"embody_3way_{TAG}.csv"
RUN_BRIEF = RESULTS / f"RUN_EMBODY_3WAY_{TAG.upper()}.md"

PPO_EPISODE_STEPS = 400
PPO_GAE_LAMBDA = 0.95
PPO_VALUE_HIDDEN = 256
PPO_UPDATE_EPOCHS = 4
# Backpropagation through the full 400-step rollout stores one set of activations
# per step for a (11000, B) recurrent state, which is ~19 GB at B=128 and OOMs a
# 24 GB card. Chunked gradient checkpointing keeps the gradient exact -- every step
# is still differentiated -- and trades one extra forward pass per chunk for memory
# proportional to the chunk, not the episode.
PPO_BPTT_CHUNK = 25
# Backpropagating through the full 400-step episode overflows float32. The frozen VNC's
# recurrent Jacobian has spectral radius above 1, so the gradient norm grows
# exponentially with depth -- measured on this core at 2.6e3 (30 steps), 1.4e7 (60),
# 4.5e11 (100), 9.5e14 (150), inf (200+), and nan by 400, while the forward pass stays
# finite because the core clamps its state. That nan is what killed the first v3 Stage 2.
# Clipping alone cannot repair it, since clipping an inf yields nan.
#
# The recurrent state is therefore detached between BPTT chunks: credit assignment is
# truncated to PPO_BPTT_CHUNK steps, which is where the gradient is still finite and
# clippable. Truncated BPTT plus clipping is the standard recipe for recurrent PPO.
# Both apply identically to all three arms, so the contrast stays fair; what is lost is
# credit assignment beyond 25 steps, equally in every arm.
PPO_TRUNCATED_BPTT = True
PPO_GRAD_CLIP_NORM = 1.0
PPO_INITIAL_STD = 0.20
PPO_LOG_STD_MIN = -5.0
PPO_LOG_STD_MAX = 2.0
PPO_ADAM_EPS = 1e-8
PPO_EMA_ALPHA = 0.05
MIN_STABLE_FORWARD_M = 0.20
# The training-progress threshold has to move with the reward. In v2 ema_return was
# summed signed displacement in metres, so MIN_STABLE_FORWARD_M was directly
# comparable to it. The v3 reward is a per-step factor in [0, 1], so an episode return
# runs 0..PPO_EPISODE_STEPS and 0.20 would be cleared at initialisation (the PPO smoke
# returns 6.29 untrained). Express the same bar in return units: the number of on-target
# control steps needed to cover MIN_STABLE_FORWARD_M at TARGET_SPEED_MPS.
MIN_STABLE_RETURN = MIN_STABLE_FORWARD_M / TARGET_SPEED_MPS * 50.0
VIDEO_SEED = EVALUATION_SEEDS[0]
# ponytail: whole vector batches leave <B tail; add partial-batch accounting if exact cap matters.
ARM_STEP_CAP = PPO_STEPS_PER_ARM - PPO_STEPS_PER_ARM % PPO_BATCH_SIZE

CHECKPOINTS = {condition: RESULTS / f"embody_3way_{TAG}_ppo_{condition}_final.pt" for condition in CONDITIONS}
VIDEO_PATHS = {
    "connectome": RESULTS / f"embody_3way_{TAG}_connectome_final_multiview.mp4",
    "random": RESULTS / f"embody_3way_{TAG}_random_final_multiview.mp4",
}


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, PPO_VALUE_HIDDEN), nn.Tanh(), nn.Linear(PPO_VALUE_HIDDEN, 1))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation).reshape(-1)


def addendum_payload() -> dict:
    preregistered = json.loads(PREREGISTERED.read_text(encoding="utf-8"))
    return {
        "experiment": "EMBODY_3WAY_STAGE2_PROTOCOL_ADDENDUM",
        "parent_preregistration": str(PREREGISTERED),
        "parent_config_hash": json.loads(PREREGISTERED.read_text(encoding="utf-8"))["config_hash"],
        "device": preregistered["device"],
        "latent_dim": preregistered["latent_dim"],
        "command_map_sha256": preregistered["command_map"]["sha256"],
        "batch_size": preregistered["training"]["stage2"]["batch_size"],
        "effective_steps_per_arm": ARM_STEP_CAP,
        "vector_environment": True,
        "episode_steps": PPO_EPISODE_STEPS,
        "gae_lambda": PPO_GAE_LAMBDA,
        "value_network": {"class": "Linear-Tanh-Linear", "hidden_units": PPO_VALUE_HIDDEN},
        "update_epochs": PPO_UPDATE_EPOCHS,
        "bptt": {
            "mode": "chunked_gradient_checkpointing_with_truncation",
            "chunk_steps": PPO_BPTT_CHUNK,
            "truncated": PPO_TRUNCATED_BPTT,
            "grad_clip_norm": PPO_GRAD_CLIP_NORM,
            "amends": "v2 backpropagated through the full 400-step episode untruncated and unclipped",
            "reason": "the frozen core's recurrent Jacobian explodes with depth (2.6e3 at 30 steps, inf by 200, nan by 400 on this core), which produced a non-finite policy mean and aborted the first v3 Stage 2; truncation and clipping are applied identically to all three arms",
        },
        "reward": {
            "form": "multiplicative bounded factors, ported from flybody Walking.get_reward",
            "factors": ["forward_speed_tracking", "upright_ride_height"],
            "combination": "product",
            "per_step_range": [0.0, 1.0],
            "tolerance_implementation": "dm_control.utils.rewards.tolerance, vendored verbatim in dm_rewards.py",
            "target_speed_mps": TARGET_SPEED_MPS,
            "upright_height_m": UPRIGHT_HEIGHT_M,
            "fall_floor_m": H_MIN,
            "amends": "v2 used unbounded per-step signed forward displacement summed over the episode",
            "reason": "unbounded displacement has no fixed point; the v2 connectome arm maximised it by diving forward at 0.588 m/s and falling in 12/12 evaluation seeds",
        },
        "initial_action_std": PPO_INITIAL_STD,
        "log_std_clamp": [PPO_LOG_STD_MIN, PPO_LOG_STD_MAX],
        "optimizer": {"class": "Adam", "learning_rate": PPO_LEARNING_RATE, "eps": PPO_ADAM_EPS},
        "ema": {"formula": "ema=(1-alpha)*ema+alpha*episode_return", "alpha": PPO_EMA_ALPHA},
        "training_seed": STAGE1_SEED,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "evaluation_policy": "deterministic mean action at the last fixed checkpoint",
        "action_handling": "Gaussian samples are logged before HexapodEnv joint-range clipping; log-probability uses unclipped samples",
        "stable_forward_rule": {
            "mean_forward_displacement_m_at_least": MIN_STABLE_FORWARD_M,
            "all_evaluation_episodes_fall_free": True,
        },
        "training_ema_threshold_return": MIN_STABLE_RETURN,
        "evaluation_seed_disclosure": "EVALUATION_SEEDS are a subset of the 128 training initialisation seeds, so no arm is evaluated on unseen initial yaw; the overlap is identical across arms and does not bias the paired contrast",
        "video_seed": VIDEO_SEED,
        "decision_labels": list(DECISION_LABELS),
        "protocol_hash": json_hash({
            "episode_steps": PPO_EPISODE_STEPS,
            "gae_lambda": PPO_GAE_LAMBDA,
            "value_hidden": PPO_VALUE_HIDDEN,
            "update_epochs": PPO_UPDATE_EPOCHS,
            "bptt_chunk_steps": PPO_BPTT_CHUNK,
            "bptt_truncated": PPO_TRUNCATED_BPTT,
            "grad_clip_norm": PPO_GRAD_CLIP_NORM,
            "initial_std": PPO_INITIAL_STD,
            "log_std_clamp": [PPO_LOG_STD_MIN, PPO_LOG_STD_MAX],
            "optimizer": [PPO_LEARNING_RATE, PPO_ADAM_EPS],
            "ema_alpha": PPO_EMA_ALPHA,
            "reward_form": "product(speed_tracking, upright)",
            "reward_target_speed_mps": TARGET_SPEED_MPS,
            "reward_upright_height_m": UPRIGHT_HEIGHT_M,
            "training_seed": STAGE1_SEED,
            "evaluation_seeds": EVALUATION_SEEDS,
            "stable_forward_threshold": MIN_STABLE_FORWARD_M,
            "training_ema_threshold_return": MIN_STABLE_RETURN,
        }),
    }


def write_addendum() -> None:
    if not PREREGISTERED.exists():
        raise FileNotFoundError(PREREGISTERED)
    if ADDENDUM.exists():
        raise FileExistsError(f"{ADDENDUM} exists; Stage-2 protocol is immutable")
    SC.atomic_json(ADDENDUM, addendum_payload())
    print(json.dumps({"stage2_addendum": str(ADDENDUM), "protocol_hash": addendum_payload()["protocol_hash"]}, sort_keys=True))


def require_protocol() -> dict:
    if not PREREGISTERED.exists() or not ADDENDUM.exists():
        raise FileNotFoundError("write the preregistration and Stage-2 addendum before PPO")
    prereg = json.loads(PREREGISTERED.read_text(encoding="utf-8"))
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    expected = addendum_payload()
    if prereg["latent_dim"] != 4 or prereg["command_map"]["sha256"] != COMMAND_MAP_SHA256:
        raise AssertionError("command-map registration is invalid")
    if resolve_device(prereg["device"]) != prereg["device"]:
        raise AssertionError("registered device is not available or changed")
    if prereg["training"]["stage2"]["batch_size"] != PPO_BATCH_SIZE:
        raise AssertionError("registered batch size differs from runner")
    if addendum != expected:
        raise AssertionError("Stage-2 addendum changed after registration")
    if prereg["config_hash"] != expected["parent_config_hash"]:
        raise AssertionError("parent preregistration changed")
    return addendum


def assert_frozen(agent: "PPOAgent") -> None:
    core = agent.policy.core
    assert not tuple(core.parameters())
    assert not isinstance(core.W, nn.Parameter)
    assert not core.W.requires_grad
    assert core.W.grad is None
    assert all(not buffer.requires_grad for buffer in core.buffers())
    optimizer_ids = {id(parameter) for group in agent.optimizer.param_groups for parameter in group["params"]}
    assert not optimizer_ids.intersection({id(parameter) for parameter in core.parameters()})


class PPOAgent:
    def __init__(self, condition: str, calibration_record: dict, checkpoint: Path | None = None):
        torch.manual_seed(STAGE1_SEED)
        self.condition = condition
        self.device = torch.device(calibration_record["device"])
        self.batch_size = PPO_BATCH_SIZE
        self.policy = _policy(condition, calibration_record, STAGE1_SEED)
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
            self.policy.encoder.load_state_dict(saved["encoder"])
            self.policy.decoder.load_state_dict(saved["decoder"])
        self.log_std = nn.Parameter(torch.full((18,), math.log(PPO_INITIAL_STD), dtype=torch.float32, device=self.device))
        self.value = ValueNet(HexapodEnv(max_steps=1).obs_dim).to(self.device)
        self.vector_env = VectorHexapodEnv(PPO_BATCH_SIZE, max_steps=PPO_EPISODE_STEPS)
        self.optimizer = torch.optim.Adam(
            list(self.policy.interface_parameters()) + [self.log_std] + list(self.value.parameters()),
            lr=PPO_LEARNING_RATE,
            eps=PPO_ADAM_EPS,
        )
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=PPO_LEARNING_RATE, eps=PPO_ADAM_EPS)
        assert_frozen(self)

    def distribution(self, mean: torch.Tensor) -> Normal:
        log_std = self.log_std.clamp(PPO_LOG_STD_MIN, PPO_LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def policy_sequence(self, observations: torch.Tensor) -> torch.Tensor:
        state = self.policy.initial_state()
        means = []
        for observation in observations.to(self.device):
            mean, state = self.policy.step(observation, state)
            means.append(mean)
        return torch.stack(means)

    def _policy_batch_chunk(self, state: torch.Tensor, observations: torch.Tensor, dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means = []
        for observation, done in zip(observations, dones):
            mean, state = self.policy.step_batch(observation, state)
            means.append(mean)
            state = self.policy.reset_state_batch(state, done)
        return torch.stack(means), state

    def policy_batch_sequence(self, observations: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        state = self.policy.initial_state_batch(self.batch_size)
        means = []
        for start in range(0, observations.shape[0], PPO_BPTT_CHUNK):
            observation_chunk = observations[start:start + PPO_BPTT_CHUNK]
            done_chunk = dones[start:start + PPO_BPTT_CHUNK]
            if torch.is_grad_enabled():
                chunk_means, state = torch.utils.checkpoint.checkpoint(
                    self._policy_batch_chunk, state, observation_chunk, done_chunk, use_reentrant=False
                )
            else:
                chunk_means, state = self._policy_batch_chunk(state, observation_chunk, done_chunk)
            if PPO_TRUNCATED_BPTT:
                state = state.detach()
            means.append(chunk_means)
        return torch.cat(means)

    def collect_batch(self, seed: int, stochastic: bool = True, max_steps: int = PPO_EPISODE_STEPS) -> dict:
        horizon = min(PPO_EPISODE_STEPS, int(max_steps) // self.batch_size)
        if horizon < 1:
            raise ValueError("rollout horizon must be positive")
        seeds = (int(seed) + np.arange(self.batch_size)).tolist()
        observation = self.vector_env.reset(seeds=seeds)
        state = self.policy.initial_state_batch(self.batch_size)
        observations, actions, rewards, values, old_log_probs, dones = [], [], [], [], [], []
        with torch.no_grad():
            for _ in range(horizon):
                observation_t = torch.from_numpy(observation).to(self.device)
                mean, state = self.policy.step_batch(observation_t, state)
                distribution = self.distribution(mean)
                action = distribution.sample() if stochastic else mean
                old_log_prob = distribution.log_prob(action).sum(dim=-1)
                value = self.value(observation_t)
                next_observation, reward, done = self.vector_env.step(action.cpu().numpy())
                observations.append(observation.copy())
                actions.append(action.cpu().numpy().astype(np.float32, copy=False))
                rewards.append(reward.copy())
                values.append(value.cpu().numpy())
                old_log_probs.append(old_log_prob.cpu().numpy())
                dones.append(done.copy())
                state = self.policy.reset_state_batch(state, done)
                observation = next_observation

            next_value = self.value(torch.from_numpy(observation).to(self.device)).cpu().numpy().astype(np.float64)
        values_np = np.asarray(values, dtype=np.float64)
        rewards_np = np.asarray(rewards, dtype=np.float64)
        dones_np = np.asarray(dones, dtype=bool)
        advantages = np.zeros_like(rewards_np)
        returns = np.zeros_like(rewards_np)
        next_advantage = np.zeros(self.batch_size, dtype=np.float64)
        for index in range(horizon - 1, -1, -1):
            nonterminal = 1.0 - dones_np[index]
            delta = rewards_np[index] + PPO_GAMMA * next_value * nonterminal - values_np[index]
            next_advantage = delta + PPO_GAMMA * PPO_GAE_LAMBDA * next_advantage * nonterminal
            advantages[index] = next_advantage
            returns[index] = advantages[index] + values_np[index]
            next_value = values_np[index]
        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": rewards_np.astype(np.float32),
            "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
            "returns": returns.astype(np.float32),
            "advantages": advantages.astype(np.float32),
            "dones": dones_np,
            "episode_return": float(rewards_np.sum(axis=0).mean()),
            "steps": int(horizon * self.batch_size),
            "fell": bool(any(env._min_height < H_MIN for env in self.vector_env.envs)),
        }

    def value_warmup(self, episode: dict) -> float:
        observations = torch.from_numpy(episode["observations"]).to(self.device).reshape(-1, self.policy.obs_dim)
        returns = torch.from_numpy(episode["returns"]).to(self.device).reshape(-1)
        prediction = self.value(observations)
        loss = torch.mean((prediction - returns) ** 2)
        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), PPO_GRAD_CLIP_NORM)
        if not torch.isfinite(grad_norm):
            raise AssertionError(f"non-finite value-warmup gradient for {self.condition}")
        self.value_optimizer.step()
        assert_frozen(self)
        return float(loss.detach())

    def ppo_update(self, episode: dict) -> dict[str, float]:
        observations = torch.from_numpy(episode["observations"]).to(self.device)
        actions = torch.from_numpy(episode["actions"]).to(self.device)
        old_log_probs = torch.from_numpy(episode["old_log_probs"]).to(self.device)
        returns = torch.from_numpy(episode["returns"]).to(self.device)
        advantages = torch.from_numpy(episode["advantages"]).to(self.device)
        dones = torch.from_numpy(episode["dones"]).to(self.device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        last = {}
        for _ in range(PPO_UPDATE_EPOCHS):
            means = self.policy_batch_sequence(observations, dones)
            distribution = self.distribution(means)
            log_probs = distribution.log_prob(actions).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1).mean()
            values = self.value(observations.reshape(-1, observations.shape[-1])).reshape_as(returns)
            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.mean((values - returns) ** 2)
            loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy
            if not torch.isfinite(loss):
                raise AssertionError(f"non-finite PPO loss for {self.condition}")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for group in self.optimizer.param_groups for parameter in group["params"]],
                PPO_GRAD_CLIP_NORM,
            )
            # A non-finite gradient must stop the run, not be clipped into a plausible
            # looking number and stepped on.
            if not torch.isfinite(grad_norm):
                raise AssertionError(f"non-finite PPO gradient for {self.condition}")
            self.optimizer.step()
            last = {
                "grad_norm": float(grad_norm),
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "mean_ratio": float(ratio.detach().mean()),
            }
        assert_frozen(self)
        return last

    def checkpoint(self, path: Path, steps: int, episode_index: int, ema_return: float | None, history: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "condition": self.condition,
                "steps": steps,
                "episode_index": episode_index,
                "ema_return": ema_return,
                "encoder": self.policy.encoder.state_dict(),
                "decoder": self.policy.decoder.state_dict(),
                "log_std": self.log_std.detach().cpu(),
                "value": self.value.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                "history": history,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict:
        saved = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.encoder.load_state_dict(saved["encoder"])
        self.policy.decoder.load_state_dict(saved["decoder"])
        self.log_std.data.copy_(saved["log_std"])
        self.value.load_state_dict(saved["value"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.value_optimizer.load_state_dict(saved["value_optimizer"])
        assert_frozen(self)
        return saved


def completed_training_record(condition: str, agent: "PPOAgent", saved: dict) -> dict:
    """Rebuild a training record from an arm whose final checkpoint is already at the cap.

    The first v3 Stage 2 was killed by the runner's own step timeout partway through the
    third arm, after the first two had finished their full preregistered budget. Their
    final checkpoints are complete and were produced by this code at these seeds, so they
    are reused rather than retrained; only the unfinished arm runs again. Fields that
    live only in the training loop (warm-up loss, wall time) are null here and the record
    is flagged, so the report never implies they were measured in this session.
    """
    history = saved.get("history", [])
    threshold_rows = [row for row in history if row["ema_return"] >= MIN_STABLE_RETURN]
    return {
        "condition": condition,
        "steps": int(saved["steps"]),
        "hard_cap_steps": PPO_STEPS_PER_ARM,
        "batch_size": agent.batch_size,
        "episodes": int(saved["episode_index"]),
        "value_warmup_steps": PPO_VALUE_WARMUP_STEPS,
        "warmup_value_loss_mean": None,
        "final_ema_return": float(saved["ema_return"]),
        "sample_efficiency_threshold_return": MIN_STABLE_RETURN,
        "steps_to_first_training_ema_threshold": int(threshold_rows[0]["steps"]) if threshold_rows else None,
        "checkpoint": str(CHECKPOINTS[condition]),
        "interface_parameter_count": int(sum(p.numel() for p in agent.policy.interface_parameters()) + agent.log_std.numel()),
        "value_parameter_count": int(sum(p.numel() for p in agent.value.parameters())),
        "core_parameter_count": 0,
        "core_tensor_sha256_pre": SC.tensor_sha256(agent.policy.core.W),
        "core_tensor_sha256_post": SC.tensor_sha256(agent.policy.core.W),
        "vnc_weights_trained": False,
        "wall_time_s": None,
        "checkpoint_history": history,
        "reused_completed_checkpoint": True,
        "reuse_reason": "arm reached the full preregistered step cap in an earlier pod session that was cut short by the runner's step timeout during a later arm",
    }


def train_condition(condition: str, calibration_record: dict) -> dict:
    existing = CHECKPOINTS[condition]
    if existing.exists():
        probe = PPOAgent(condition, calibration_record, STAGE1_CHECKPOINTS[condition])
        saved = torch.load(existing, map_location=probe.device, weights_only=True)
        if int(saved["steps"]) == ARM_STEP_CAP:
            probe.load_checkpoint(existing)
            assert_frozen(probe)
            print(json.dumps({"condition": condition, "reused_completed_checkpoint": True, "steps": int(saved["steps"])}, sort_keys=True), flush=True)
            return completed_training_record(condition, probe, saved)
        print(json.dumps({"condition": condition, "discarding_partial_checkpoint_steps": int(saved["steps"])}, sort_keys=True), flush=True)
        existing.unlink()
    agent = PPOAgent(condition, calibration_record, STAGE1_CHECKPOINTS[condition])
    started = time.perf_counter()
    steps = 0
    episode_index = 0
    ema_return = None
    history = []
    warmup_loss = []
    while steps < PPO_VALUE_WARMUP_STEPS:
        episode = agent.collect_batch(STAGE1_SEED, stochastic=True, max_steps=PPO_VALUE_WARMUP_STEPS - steps)
        steps += episode["steps"]
        episode_index += 1
        warmup_loss.append(agent.value_warmup(episode))
        ema_return = episode["episode_return"] if ema_return is None else (1.0 - PPO_EMA_ALPHA) * ema_return + PPO_EMA_ALPHA * episode["episode_return"]
    next_checkpoint = PPO_CHECKPOINT_INTERVAL * (steps // PPO_CHECKPOINT_INTERVAL + 1)
    while steps < ARM_STEP_CAP:
        # Collect against the remaining budget, not the next checkpoint boundary: the
        # boundary is rarely a whole number of batches away, and a remainder below one
        # batch used to floor the rollout horizon to zero. Checkpoints still fire on the
        # steps >= next_checkpoint test below.
        episode = agent.collect_batch(STAGE1_SEED, stochastic=True, max_steps=ARM_STEP_CAP - steps)
        update = agent.ppo_update(episode)
        steps += episode["steps"]
        episode_index += 1
        ema_return = episode["episode_return"] if ema_return is None else (1.0 - PPO_EMA_ALPHA) * ema_return + PPO_EMA_ALPHA * episode["episode_return"]
        if steps >= next_checkpoint or steps == ARM_STEP_CAP:
            row = {"steps": steps, "episode": episode_index, "episode_return": episode["episode_return"], "ema_return": ema_return, **update}
            history.append(row)
            agent.checkpoint(CHECKPOINTS[condition], steps, episode_index, ema_return, history)
            print(json.dumps({"condition": condition, **row}, sort_keys=True), flush=True)
            while next_checkpoint <= steps:
                next_checkpoint += PPO_CHECKPOINT_INTERVAL
    if not CHECKPOINTS[condition].exists():
        agent.checkpoint(CHECKPOINTS[condition], steps, episode_index, ema_return, history)
    threshold_rows = [row for row in history if row["ema_return"] >= MIN_STABLE_RETURN]
    assert_frozen(agent)
    return {
        "condition": condition,
        "steps": steps,
        "hard_cap_steps": PPO_STEPS_PER_ARM,
        "batch_size": agent.batch_size,
        "episodes": episode_index,
        "value_warmup_steps": PPO_VALUE_WARMUP_STEPS,
        "warmup_value_loss_mean": float(np.mean(warmup_loss)),
        "final_ema_return": float(ema_return),
        "sample_efficiency_threshold_return": MIN_STABLE_RETURN,
        "steps_to_first_training_ema_threshold": int(threshold_rows[0]["steps"]) if threshold_rows else None,
        "checkpoint": str(CHECKPOINTS[condition]),
        "interface_parameter_count": int(sum(parameter.numel() for parameter in agent.policy.interface_parameters()) + agent.log_std.numel()),
        "value_parameter_count": int(sum(parameter.numel() for parameter in agent.value.parameters())),
        "core_parameter_count": 0,
        "core_tensor_sha256_pre": SC.tensor_sha256(agent.policy.core.W),
        "core_tensor_sha256_post": SC.tensor_sha256(agent.policy.core.W),
        "vnc_weights_trained": False,
        "wall_time_s": time.perf_counter() - started,
        "checkpoint_history": history,
    }


def evaluate(agent: PPOAgent, seed: int, video_path: Path | None = None) -> dict:
    import mujoco
    import imageio.v2 as imageio

    env = HexapodEnv(max_steps=PPO_EPISODE_STEPS)
    observation = env.reset(seed=seed)
    state = agent.policy.initial_state()
    x0, y0 = float(observation[0]), float(observation[1])
    path = 0.0
    last_xy = (x0, y0)
    actions = []
    heights = [float(observation[2])]
    writer = None
    renderer = None
    cameras = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video_path), fps=25, codec="libx264")
        renderer = mujoco.Renderer(env.model, 480, 640)
        cameras = SC.make_cameras(env.model)

    def frame():
        views = []
        for camera in cameras:
            renderer.update_scene(env.data, camera=camera)
            views.append(renderer.render())
        return np.vstack((np.hstack((views[0], views[1])), np.hstack((views[2], views[3]))))

    done = False
    episode_reward = 0.0
    with torch.no_grad():
        while not done:
            mean, state = agent.policy.step(torch.from_numpy(observation), state)
            action = mean.cpu().numpy().astype(np.float32)
            observation, reward, done = env.step(action)
            episode_reward += reward
            actions.append(action)
            now = (float(observation[0]), float(observation[1]))
            path += float(np.hypot(now[0] - last_xy[0], now[1] - last_xy[1]))
            last_xy = now
            heights.append(float(observation[2]))
            if writer is not None and len(actions) % 2 == 0:
                writer.append_data(frame())
    if writer is not None:
        writer.close()
    dx, dy = float(observation[0]) - x0, float(observation[1]) - y0
    forward = float((dx * math.cos(env._heading)) + (dy * math.sin(env._heading)))
    # _heading is the seeded initial heading, unchanged during the episode.
    return {
        "condition": agent.condition,
        "episode": int(EVALUATION_SEEDS.index(seed)),
        "episode_seed": int(seed),
        "forward_distance_m": forward,
        "path_m": float(path),
        "displacement_m": float(np.hypot(dx, dy)),
        "steps": len(actions),
        # Time normalisation. v2 compared a 26-step connectome episode against
        # 400-step control episodes on distance alone, which flattered the arm
        # that ended early by falling. Report the rate and the upright duration
        # alongside the distance so unequal episode lengths stay visible.
        "upright_s": float(len(actions) / env.control_hz),
        "mean_forward_speed_mps": float(forward / (len(actions) / env.control_hz)) if actions else 0.0,
        # env._min_height is tracked every physics step (10 per control step); `heights`
        # only samples the control rate and can miss the dip that ended the episode,
        # which would log a short episode as a non-fall.
        "min_height_m": float(env._min_height),
        "final_height_m": float(heights[-1]),
        "fell": bool(env._min_height < H_MIN),
        "episode_return": float(episode_reward),
        "action_temporal_std_mean": float(np.asarray(actions).std(axis=0).mean()) if actions else 0.0,
    }


def evaluation_agent(condition: str, calibration_record: dict) -> PPOAgent:
    agent = PPOAgent(condition, calibration_record, STAGE1_CHECKPOINTS[condition])
    saved = agent.load_checkpoint(CHECKPOINTS[condition])
    if int(saved["steps"]) != ARM_STEP_CAP:
        raise AssertionError(f"{condition} final checkpoint has {saved['steps']} steps")
    return agent


def final_report(training: list[dict], rows: list[dict], calibration_record: dict, addendum: dict) -> dict:
    rng = np.random.default_rng(20260829)
    grouped = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    summaries = {}
    for condition in CONDITIONS:
        values = np.asarray([row["forward_distance_m"] for row in grouped[condition]], dtype=np.float64)
        low, high = bootstrap_ci(values, rng, 100_000)
        falls = int(sum(row["fell"] for row in grouped[condition]))
        summaries[condition] = {
            "n": len(values),
            "mean_forward_displacement_m": float(values.mean()),
            "bootstrap_ci_95_low_m": low,
            "bootstrap_ci_95_high_m": high,
            "fall_count": falls,
            "fall_rate": float(falls / len(values)),
            "fall_wilson_95ci": wilson_interval(falls, len(values)),
            "mean_upright_s": float(np.mean([row["upright_s"] for row in grouped[condition]])),
            "mean_forward_speed_mps": float(np.mean([row["mean_forward_speed_mps"] for row in grouped[condition]])),
            "mean_episode_return": float(np.mean([row["episode_return"] for row in grouped[condition]])),
            "stable_forward": bool(falls == 0 and values.mean() >= MIN_STABLE_FORWARD_M),
        }
    real_rows = grouped["connectome"]
    contrasts = {}
    raw_p = []
    for condition in ("random", "rewired"):
        contrast = paired_forward(real_rows, grouped[condition], np.random.default_rng(20260829))
        contrast["n_over_K"] = f"{int(sum(row['forward_distance_m'] >= real_rows[i]['forward_distance_m'] for i, row in enumerate(grouped[condition])))}/{len(real_rows)}"
        contrast["controls_match_or_beat_real_count"] = int(sum(row["forward_distance_m"] >= real_rows[i]["forward_distance_m"] for i, row in enumerate(grouped[condition])))
        contrasts[f"connectome_minus_{condition}"] = contrast
        raw_p.append(contrast["p_raw"])
    holm = SC.holm_adjust(raw_p)
    for index, condition in enumerate(("random", "rewired")):
        contrasts[f"connectome_minus_{condition}"]["p_holm"] = float(holm[index])
    all_stable = any(summaries[condition]["stable_forward"] for condition in CONDITIONS)
    real_wins = all(
        contrasts[f"connectome_minus_{condition}"]["mean_difference_m"] > 0.0
        and contrasts[f"connectome_minus_{condition}"]["ci_low_m"] > 0.0
        and contrasts[f"connectome_minus_{condition}"]["p_holm"] < 0.05
        for condition in ("random", "rewired")
    )
    decision = "EMBODY-PILOT-INCONCLUSIVE" if not all_stable else "EMBODY-MOAT-EARNED" if real_wins else "EMBODY-MOAT-NULL"
    file_pre = calibration_record["file_hashes_pre"]
    file_post = SC.file_hashes(SC.VNC_DATA)
    if file_post != file_pre:
        raise AssertionError("real graph files changed during EMBODY evaluation")
    tensor_pre = calibration_record["graph_tensor_sha256"]
    tensor_post = {condition: SC.tensor_sha256(_policy(condition, calibration_record, STAGE1_SEED).core.W) for condition in CONDITIONS}
    if tensor_post != tensor_pre:
        raise AssertionError("graph tensor changed during EMBODY evaluation")
    for row in training:
        if row["core_tensor_sha256_pre"] != row["core_tensor_sha256_post"] or row["core_tensor_sha256_post"] != tensor_pre[row["condition"]]:
            raise AssertionError(f"training graph hash invariant failed for {row['condition']}")
    return {
        "experiment": "EMBODY_3WAY",
        "status": "complete_owner_claim_gate",
        "decision_label": decision,
        "preregistered": str(PREREGISTERED),
        "stage2_addendum": str(ADDENDUM),
        "conditions": list(CONDITIONS),
        "training": training,
        "evaluation": summaries,
        "contrasts": contrasts,
        "rows": rows,
        "graph_tensor_sha256_pre": tensor_pre,
        "graph_tensor_sha256_post": tensor_post,
        "file_hashes_pre": file_pre,
        "file_hashes_post": file_post,
        "amplitude_match": calibration_record["amplitude_match"],
        "amplitude_active_indices": calibration_record["active_indices"],
        "amplitude_excluded_indices": calibration_record["excluded_indices"],
        "vnc_weights_trained": False,
        "core_gradients_seen": False,
        "protocol": addendum,
        "videos": {condition: str(path) for condition, path in VIDEO_PATHS.items()},
        "honest_note": "Numbers and decision label are reported without a final claim; owner approval remains required.",
    }


def write_outputs(report: dict) -> None:
    SC.atomic_json(FINAL_JSON, report)
    fields = (
        "condition", "episode", "episode_seed", "forward_distance_m", "path_m", "displacement_m", "steps",
        "upright_s", "mean_forward_speed_mps",
        "min_height_m", "final_height_m", "fell", "episode_return", "action_temporal_std_mean",
    )
    with FINAL_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in report["rows"])


def write_brief(report: dict) -> None:
    lines = [
        "# EMBODY-3WAY run",
        "",
        "Stage 2 uses the preregistered B=128 vector environment per frozen core, with identical budget and final-checkpoint rule.",
        "",
        f"Decision label: `{report['decision_label']}`. Final claim remains owner-gated.",
        "",
        "## Forward displacement",
        "",
        "| arm | mean m | bootstrap 95% CI | upright s | m/s | falls | stable |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for condition in CONDITIONS:
        summary = report["evaluation"][condition]
        lines.append(
            f"| {condition} | {summary['mean_forward_displacement_m']:.6f} | "
            f"[{summary['bootstrap_ci_95_low_m']:.6f}, {summary['bootstrap_ci_95_high_m']:.6f}] | "
            f"{summary['mean_upright_s']:.2f} | {summary['mean_forward_speed_mps']:.4f} | "
            f"{summary['fall_count']}/{summary['n']} | {summary['stable_forward']} |"
        )
    lines.extend(["", "## Paired contrasts", ""])
    for name, contrast in report["contrasts"].items():
        lines.append(f"- `{name}`: {contrast['mean_difference_m']:+.6f} m, CI [{contrast['ci_low_m']:+.6f}, {contrast['ci_high_m']:+.6f}], one-sided exact p={contrast['p_raw']:.6g}, Holm p={contrast['p_holm']:.6g}, n/K={contrast['n_over_K']}")
    lines.extend(["", "## Limitations", "", f"- Amplitude calibration retained {len(report['amplitude_active_indices'])}/260 motor channels and explicitly excluded {len(report['amplitude_excluded_indices'])}.", "- This is forward locomotion under one body, interface, optimizer, and PPO budget; it is not a coordination proof.", "- Final interpretation awaits owner approval.", ""])
    RUN_BRIEF.write_text("\n".join(lines), encoding="utf-8")


def run_stage2() -> None:
    addendum = require_protocol()
    if FINAL_JSON.exists():
        raise FileExistsError(f"{FINAL_JSON} exists")
    for condition in CONDITIONS:
        if not STAGE1_CHECKPOINTS[condition].exists():
            raise FileNotFoundError(STAGE1_CHECKPOINTS[condition])
    prereg = json.loads(PREREGISTERED.read_text(encoding="utf-8"))
    calibration_record = calibration(prereg["device"])
    training = []
    for condition in CONDITIONS:
        training.append(train_condition(condition, calibration_record))
    agents = {condition: evaluation_agent(condition, calibration_record) for condition in CONDITIONS}
    rows = []
    for condition in CONDITIONS:
        for seed in EVALUATION_SEEDS:
            rows.append(evaluate(agents[condition], seed, VIDEO_PATHS[condition] if condition in VIDEO_PATHS and seed == VIDEO_SEED else None))
    report = final_report(training, rows, calibration_record, addendum)
    write_outputs(report)
    write_brief(report)
    print(json.dumps({"result": str(FINAL_JSON), "decision_label": report["decision_label"], "evaluation": report["evaluation"], "contrasts": report["contrasts"]}, sort_keys=True))


def run_smoke() -> None:
    require_protocol()
    prereg = json.loads(PREREGISTERED.read_text(encoding="utf-8"))
    calibration_record = calibration(prereg["device"])
    reports = []
    for condition in CONDITIONS:
        agent = PPOAgent(condition, calibration_record)
        episode = agent.collect_batch(STAGE1_SEED, stochastic=True, max_steps=8 * PPO_BATCH_SIZE)
        update = agent.ppo_update(episode)
        assert_frozen(agent)
        reports.append({"condition": condition, "steps": episode["steps"], "ppo": update, "core_grad": agent.policy.core.W.grad is None})
    print(json.dumps({"ppo_smoke": "PASS", "reports": reports}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-addendum", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if sum((args.write_addendum, args.stage2, args.smoke)) != 1:
        parser.error("choose exactly one of --write-addendum, --stage2, or --smoke")
    if args.write_addendum:
        write_addendum()
    elif args.stage2:
        run_stage2()
    else:
        run_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
