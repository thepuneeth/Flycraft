"""EMBODY v4: train the frozen-VNC interface to walk a hexapod to a target object.

Deliberately leaner than run_embody_3way.py. The owner's call is to get one arm
walking first and lock the recipe, then run the random and rewired controls once
against that locked recipe. So this script trains and evaluates ONE condition and
carries none of the three-arm preregistration, decision-label, or paired-test
machinery -- that lives in run_embody_3way.py and is where the controls go back
when the recipe stops moving. Nothing here writes a v2 or v3 artifact.

What changed from v3, and why, in order of how much each mattered:

  1. The teacher never walked. v2/v3 ran CEM at population 24 for 3 rounds and
     shipped the result; on the straight command that selects a gait covering
     0.166 m against a 0.400 m reference (v3's own recorded teacher path MSE was
     0.0117). Stage 1 fit the interface to a shuffle, in every arm, before PPO
     started. teacher_v4 raises the search budget until the teacher covers the
     reference distance, and asserts it.
  2. The teacher only ever taught straight-line walking, so the encoder's `steer`
     latent was never given a meaning -- and steering is the whole of reaching a
     target. teacher_v4 is command-conditioned over five yaw rates.
  3. The reward was invented here twice and gamed twice. v4 uses the legged_gym
     reward set as published; see embody_env_v4 for the term-by-term provenance.
  4. There was no target. The task is now to reach the prey sphere.

Usage:
    python run_embody_v4.py stage1     # command-conditioned teacher + BC fit
    python run_embody_v4.py smoke      # one PPO update; checks gradients are finite
    python run_embody_v4.py stage2     # PPO to the step cap, resumable
    python run_embody_v4.py evaluate   # fixed-seed evaluation + one video
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
sys.path.insert(0, str(HERE.parent))

import specificity_campaign as SC  # noqa: E402
import teacher_v4  # noqa: E402
from audit_embody_3way import (  # noqa: E402
    STAGE1_LEARNING_RATE,
    STAGE1_TRAIN_FRACTION,
    PPO_CLIP,
    PPO_ENTROPY_COEF,
    PPO_GAMMA,
    STAGE1_SEED,
    calibration,
    _policy,
)
from embody_env_v4 import (  # noqa: E402
    EPISODE_STEPS,
    OBS_TARGET_DIST_SCALE,
    TARGET_REACH_M,
    HexapodTargetEnv,
    VectorHexapodTargetEnv,
)
from hexapod_gait import H_MIN  # noqa: E402
from decap_prior import (  # noqa: E402
    GAMMA as DECAP_DEFAULT_GAMMA,
    ZERO_AFTER as DECAP_DEFAULT_ZERO_AFTER,
    BatchRTMG,
    apply_prior,
    commands_from_observation,
    prior_weight,
)

TAG = "v4"
CONDITION = "connectome"

# Artifacts are namespaced by architecture. A Stage-1 checkpoint is a set of
# interface weights, so a linear one cannot be loaded into an MLP -- attempting
# it raises inside load_state_dict and takes the whole run with it. Suffixing the
# filenames means the arms simply never collide, and each is forced to earn its
# own Stage 1, which is correct: they are different fits, not the same fit.
# The default arm (linear interface, rate core) keeps the original bare names, so
# every artifact recorded before this existed is still found where it was.
_ARM = "".join(
    part for part in (
        "" if (os.environ.get("V4_INTERFACE") or "linear").strip().lower() == "linear" else "_mlp",
        "" if (os.environ.get("V4_CORE") or "rate").strip().lower() == "rate" else "_lif",
    )
)
CALIBRATION_JSON = RESULTS / f"embody_{TAG}_calibration.json"
STAGE1_JSON = RESULTS / f"embody_{TAG}_stage1{_ARM}.json"
STAGE1_CHECKPOINT = RESULTS / f"embody_{TAG}_stage1_{CONDITION}{_ARM}.pt"
PPO_CHECKPOINT = RESULTS / f"embody_{TAG}_ppo_{CONDITION}{_ARM}.pt"
EVAL_JSON = RESULTS / f"embody_{TAG}_eval_{CONDITION}{_ARM}.json"
VIDEO_PATH = RESULTS / f"embody_{TAG}_{CONDITION}{_ARM}_multiview.mp4"

# --- Stage 2 budget ----------------------------------------------------------
# The measured throughput report has core matmul at 8.5 ms of a 68.9 ms batch
# step at B=128, so 88% of wall time is MuJoCo and Python, not the frozen core.
# That makes CPU cores the lever, not a GPU. B=512 measured 1,935 env steps/s
# against 1,859 at B=128 on four threads, and the gap widens with core count.
PPO_BATCH_SIZE = 512
# v2 and v3 both ran 5M steps/arm. Rudin et al. (CoRL 2022) reach ANYmal
# locomotion at roughly 100-300M steps with 4,096 environments, for a quadruped
# with no recurrent core in the loop. 5M was about 2% of that, so neither prior
# null could distinguish "this core cannot walk" from "this was never trained".
PPO_STEPS = 50_000_000
# Rollout segment, NOT episode length. The episode is still EPISODE_STEPS; the
# vector env persists across iterations and GAE bootstraps off the value net at
# the segment boundary, exactly as [LG] does.
#
# This is the number that cost run 1. It was 600 (a whole episode), then 150.
# PPO's importance ratio and the adaptive learning rate are both computed by
# replaying the segment under the updated parameters, so the segment can only be
# as long as the policy stays predictable under a small parameter change -- and
# the policy contains an 11,000-node recurrent core with a gain of 100 on its
# input, which amplifies any perturbation exponentially. Measured here: KL after
# a single Adam step at 1e-5, against the behaviour policy, by replay horizon --
#
#   1 step 1.8e-4    12 steps 2.0e-4    24 steps 1.5e-2    32 steps 2.5e-1
#   48 steps 1.35    75 steps 4.62     100 steps 6.84     150 steps 10.3
#
# At 150 the replayed policy is decorrelated from the behaviour policy before a
# single gradient has been applied, so the KL controller reads ~11 on its first
# minibatch, drives the learning rate to its floor, and stays there. That is
# exactly what run 1 logged for seven hours, and it is reproducible on CPU in
# five minutes -- the earlier CUDA hypothesis was wrong, and the check that
# missed it only ever compared a replay under *unchanged* parameters, which is
# bit-exact at any horizon and therefore proves nothing about this.
#
# 24 is [LG]'s num_steps_per_env, and 24 is where the measurement puts the KL at
# 0.0146 against a trust region of 0.01. The source value was right; the
# reasoning that overrode it was not.
PPO_ROLLOUT_STEPS = 24
PPO_GAE_LAMBDA = 0.95
# [LG] LeggedRobotCfgPPO.algorithm.num_learning_epochs = 5.
PPO_UPDATE_EPOCHS = 5
# Minibatch over the environment axis only -- the time axis has to stay contiguous
# for BPTT. Also the memory lever: the core state is (11000, B) float32 and the
# checkpointed chunk holds ~50 x 4 substeps of it, so B=512 x chunk 50 is roughly
# 18 GB of graph. At B=128 per minibatch it is ~4.5 GB, near what v3 ran.
PPO_MINIBATCHES = 4
PPO_VALUE_HIDDEN = 256
# [LG] algorithm block, verbatim. v4 had inherited v3's fixed 1e-4 and a value
# coefficient of 0.5, and had no KL schedule at all. `schedule = 'adaptive'` is a
# large part of why legged_gym trains in minutes: it re-tunes the learning rate
# every minibatch to hold the policy step at a fixed KL, which is exactly the
# control this run needs -- the measured gradient norm is ~3e5 against a clip of
# 1.0, so the step size, not the gradient, is what has to be governed.
# rsl_rl starts at 1e-3. Measured here, a single update at that rate moves the
# policy to KL ~1.0 against the behaviour policy -- a hundred times the trust
# region, and enough to destroy the Stage-1 behaviour-cloned prior on the first
# update, before the controller has seen anything to react to. The schedule
# raises the rate by 1.5x whenever KL comes in under half the target, so starting
# low costs a few iterations of ramp and starting high costs the prior.
# Stage-1 budget, overriding the v3 values imported above.
#
# v3 ran 600 epochs / patience 60 / min_delta 1e-4 and v4 inherited it. Measured,
# that stopped Stage 1 at epoch 158 with train 0.0723 and held-out 0.0550 STILL
# DESCENDING -- it ran out of patience, it did not converge. Predicting a single
# constant action for the whole gait scores 0.0151, so the shipped prior was 5x
# worse than a constant, and every run so far started from it. An overfit probe on
# one command reached 0.00611 by epoch 400 and was still falling.
#
# One epoch is one full-batch gradient step over the five demos, so the epoch
# count IS the gradient-step count. min_delta 1e-4 is 0.2% of a 0.05 loss, which
# reads a real descent as a plateau; at 1e-6 the patience counter measures the
# tail instead of the noise floor.
# Measured on the 3090 pod: 1.45 s/epoch (550 epochs in ~800 s after the CEM
# search), so epochs are cheap and the budget is the only thing rationing them.
#
# The first pod run made the descent shape measurable for the first time. Held-out
# went 0.1045 / 0.0701 / 0.0467 / 0.0375 / 0.0322 at epochs 75 / 175 / 275 / 400 /
# 525. That is NOT geometric -- the per-100-epoch ratio drifts 0.67, 0.67, 0.84,
# 0.89. It is a power law, loss ~ epoch^-b with b = 0.62 and 0.58 measured across
# the two intervals.
#
# That fit predicts 0.0118 at epoch 3000: better than the 0.0151 constant action,
# but short of the 0.00755 gate, which would have burned 72 minutes to fail the
# gate. It predicts the gate crossing at ~6500 and 0.0059 by 10000. 12000 gives
# margin for b drifting further, costs ~4.8 h at the measured pace, and stops on
# patience if the curve flattens first -- so the ceiling is only paid if it helps.
STAGE1_MAX_EPOCHS = 12000
# The early-stop patience has to outlast several scheduler drops, or the run stops
# while the decay is still working. 150 was tuned against a flat learning rate.
STAGE1_LR_PATIENCE = 60
# Same value PPO uses here and rsl_rl uses upstream (max_grad_norm 1.0).
STAGE1_GRAD_CLIP_NORM = 1.0
# Truncation window for Stage-1 BPTT, in control steps. 0 disables truncation
# (the original full-rollout behaviour, which measured 2.6e15 gradient norms).
#
# Must span more than one gait cycle. The teacher runs at 1.2 Hz and the control
# step is 0.02 s, so a cycle is about 42 steps. The first truncated run used 20 --
# under half a cycle -- which bounds the explosion but also cuts credit assignment
# below the periodicity the policy has to reproduce. It reached held-out 0.0209
# (better than the 0.0249 of the exploding runs, still worse than the 0.0145 a
# constant action scores) and, tellingly, ended UNDERFIT: train 0.029 against
# held-out 0.021. 50 steps covers a full cycle with margin.
# Measured both ways. 20 held the pre-clip norm at 3.2 and reached held-out
# 0.0209; 50 pushed it to 14599 at epoch 25 and 425 at epoch 300, and tracked
# WORSE (0.0419 at epoch 300 against 0.030). The gait-cycle argument for a wider
# window is real but loses to the explosion it re-admits, so the measurement wins.
STAGE1_TBPTT_STEPS = 20
STAGE1_MIN_LR = 1.0e-6
STAGE1_PATIENCE = 600
STAGE1_MIN_DELTA = 1.0e-6
# Stage 1 has to clear the constant-action baseline before PPO is worth paying
# for. Measured on the v4 teacher: per-command variance about the cycle mean is
# 0.0111-0.0194, mean 0.0143, and about a single global mean 0.0151. A fit above
# that has not learned the gait, it has learned the average pose -- which is
# exactly the standing-still behaviour runs 1 and 2 started from.
STAGE1_CONSTANT_BASELINE = 0.0151
STAGE1_GATE_FRACTION = 0.5      # require comfortably better, not marginally

PPO_INITIAL_LEARNING_RATE = 1.0e-5
# rsl_rl floors the adaptive rate at 1e-5, calibrated for an MLP actor. Measured
# here, the KL stays above 1.2 with the rate pinned at that floor: the encoder
# feeds a sigmoid scaled by 100 into an 11,000-node recurrent core, so a change
# of 1e-5 in an interface weight is not a small change in the policy. The floor
# is lowered so the controller can actually reach the trust region it is for.
# Measured at the 24-step horizon: KL after one step is 1.6e-2 at lr 1e-5,
# 8.4e-4 at 1e-6, and 1.82e-4 at both 1e-8 and 1e-9 -- below 1e-8 the rate has no
# measurable effect on the policy step, so a lower floor buys nothing and only
# lets the controller hide a horizon bug by winding down forever. rsl_rl floors
# at 1e-5; the extra three decades are the deviation, and they are bounded.
PPO_LEARNING_RATE_MIN = 1.0e-8
PPO_LEARNING_RATE_MAX = 1.0e-2
PPO_DESIRED_KL = 0.01
# Abandon the rest of an update once the policy is this far outside the trust
# region; see the guard in update().
PPO_KL_ABORT_FACTOR = 50.0
PPO_VALUE_LOSS_COEF = 1.0
PPO_USE_CLIPPED_VALUE_LOSS = True
# The whole segment, which at PPO_ROLLOUT_STEPS = 24 is one chunk. Measured at
# the old 150-step segment, gradient norm went 1.4e1 at chunk 25, 1.9e2 at 50 and
# 9.8e4 at 75, so a chunk much longer than this explodes; 24 sits under the knee.
# Credit still crosses segment boundaries through the value function and GAE,
# which is how [LG] does it -- its num_steps_per_env is this same 24.
PPO_BPTT_CHUNK = 24
# Both truncation windows are overridable, because the right value is a property
# of the CORE rather than of PPO, and the two cores could have needed different
# ones. They do not: the [SH] LIF core is stable at this same 24 once its
# surrogate slope is set for the measured weight scale (see vnc_lif_core._spike),
# which is a backward-pass constant and leaves the firing rate untouched.
PPO_BPTT_CHUNK = int(os.environ.get("V4_BPTT_CHUNK", PPO_BPTT_CHUNK))
STAGE1_TBPTT_STEPS = int(os.environ.get("V4_TBPTT_STEPS", STAGE1_TBPTT_STEPS))
PPO_GRAD_CLIP_NORM = 1.0
PPO_INITIAL_STD = 0.20
PPO_LOG_STD_MIN = -5.0
PPO_LOG_STD_MAX = 2.0
PPO_ADAM_EPS = 1e-8
PPO_EMA_ALPHA = 0.05
PPO_CHECKPOINT_INTERVAL = 1_000_000
# The per-step cost is measured on a laptop; the speedup on a many-core pod is
# not. The first few logged iterations settle it, and stage2 resumes from its
# checkpoint, so the budget can be retuned from the pod's own numbers without
# losing work and without editing a script bash is still reading.
PPO_STEPS = int(os.environ.get("V4_PPO_STEPS", PPO_STEPS))
PPO_UPDATE_EPOCHS = int(os.environ.get("V4_UPDATE_EPOCHS", PPO_UPDATE_EPOCHS))
PPO_MINIBATCHES = int(os.environ.get("V4_MINIBATCHES", PPO_MINIBATCHES))
# Wall-clock budget for stage2, hours. An unattended overnight run must leave a
# deliverable whatever the pod's throughput turns out to be: without this, a run
# that does not reach ARM_STEP_CAP never returns, so `evaluate` never runs and
# there is no video and no endpoint in the morning -- only a checkpoint. Stage 2
# stops cleanly on whichever bound it reaches first and checkpoints before it
# returns, so the run is still resumable if the budget was simply too small.
PPO_MAX_HOURS = float(os.environ.get("V4_MAX_HOURS", "0")) or None
# The value head starts random, so the first advantages are noise. v3's threshold
# was 102_400 steps, which one v4 iteration used to blow past on its own -- the
# warmup branch could never fire. Count iterations instead of steps.
PPO_VALUE_WARMUP_ITERATIONS = 5
# Run 1's signature, as a stopping rule. The learning rate pinned at its floor
# while KL sits far outside the trust region means the replayed policy is not the
# behaviour policy, so the importance ratio is measuring a discrepancy rather than
# an update and no amount of further wall clock will help. Run 1 spent seven hours
# in that state. Fifty consecutive iterations is well beyond anything the adaptive
# schedule does in normal operation -- it is a stuck detector, not a patience knob.
PPO_STUCK_KL_FACTOR = 10.0
PPO_STUCK_ITERATIONS = 50
ARM_STEP_CAP = PPO_STEPS - PPO_STEPS % (PPO_BATCH_SIZE * PPO_ROLLOUT_STEPS)

# [DA]/[K] decaying action prior; see decap_prior for the citations and for why
# this is a curriculum rather than a residual. Enabled by default: every run so
# far collapsed before the locomotion reward could carry information, which is
# the failure this exists to remove. V4_DECAP=0 turns it off for the control run.
# The weight reaches EXACTLY zero at DECAP_ZERO_AFTER, and evaluate never applies
# it at all, so no reported policy is ever assisted by the generator.
DECAP_ENABLED = os.environ.get("V4_DECAP", "1") != "0"
DECAP_GAMMA = float(os.environ.get("V4_DECAP_GAMMA", DECAP_DEFAULT_GAMMA))
DECAP_ZERO_AFTER = int(os.environ.get("V4_DECAP_ZERO_AFTER", DECAP_DEFAULT_ZERO_AFTER))

EVALUATION_SEEDS = tuple(20260903 + index for index in range(12))
VIDEO_SEED = EVALUATION_SEEDS[0]


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, PPO_VALUE_HIDDEN), nn.Tanh(),
            nn.Linear(PPO_VALUE_HIDDEN, PPO_VALUE_HIDDEN), nn.Tanh(),
            nn.Linear(PPO_VALUE_HIDDEN, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


def obs_dim() -> int:
    return HexapodTargetEnv(max_steps=1).obs_dim


def load_calibration() -> dict:
    """Calibration is condition-independent and env-independent; cache it."""
    if CALIBRATION_JSON.exists():
        return json.loads(CALIBRATION_JSON.read_text())
    record = calibration("auto")
    CALIBRATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_JSON.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    return json.loads(CALIBRATION_JSON.read_text())


def assert_frozen(policy, optimizer=None) -> None:
    """The one invariant the whole project rests on: the measured weights never move."""
    if policy.core.W.requires_grad or policy.core.W.grad is not None:
        raise AssertionError("VNC core weight is trainable or accumulated a gradient")
    if any(parameter.requires_grad for parameter in policy.core.parameters()):
        raise AssertionError("VNC core exposes a trainable parameter")
    if optimizer is not None:
        core_ids = {id(tensor) for tensor in policy.core.buffers()} | {id(policy.core.W)}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) in core_ids:
                    raise AssertionError("optimizer holds a core tensor")


# --- Stage 1 -----------------------------------------------------------------
#
# Not audit_embody_3way._stage1_fit. That function is correct for the v2/v3
# dataset, which was one demonstration, and both of its structural assumptions
# break on a multi-command dataset:
#
#   * `_predict` carries one recurrent state across the whole array. With five
#     independent demonstrations stacked, demos 2..5 would start from the
#     previous demo's terminal core state instead of a reset one.
#   * `STAGE1_TRAIN_FRACTION` splits the array sequentially. On a per-command
#     dataset that puts whole commands in the held-out set, so the "held-out
#     loss" would measure extrapolation to unseen commands rather than fit,
#     and the early-stopping rule would be selecting on the wrong quantity.
#
# Everything else -- optimiser, learning rate, patience, best-epoch restore --
# is the v3 rule, imported rather than retyped.


def _predict_per_demo(policy, observations: torch.Tensor) -> torch.Tensor:
    """Replay every demonstration from a reset core state. (C, T, obs) -> (C, T, 18).

    The demonstrations are the same length, so they run as one batch down the time
    axis rather than one loop per demo. Each row keeps its own core state, which is
    what "reset between demonstrations" requires, and the sparse core costs almost
    the same for five rows as for one: the throughput report measures 2.54 ms per
    core matmul at batch 1 against 2.89 ms at batch 8. Looping the demos instead
    made Stage 1 about 33 s per epoch on CPU; batching them makes it about 7.
    """
    demos, steps = observations.shape[0], observations.shape[1]
    state = policy.initial_state_batch(demos)
    outputs = []
    for index in range(steps):
        action, state = policy.step_batch(observations[:, index], state)
        outputs.append(action)
        # Truncated BPTT. Backpropagating through all 200 steps (each 4 core
        # substeps, so 800 recurrent applications) explodes: the measured pre-clip
        # gradient norm was 2.6e15 at epoch 125 and 2.8e15 at epoch 150. Every
        # earlier Stage-1 run optimised through that, which is why none of them
        # beat a constant action no matter how long they ran or how the learning
        # rate was scheduled -- more epochs and LR decay were both treating a
        # symptom. Clipping alone does not fix it either: it rescales an exploded
        # direction rather than preventing the explosion.
        #
        # Detaching the state every STAGE1_TBPTT_STEPS bounds the credit-assignment
        # window. The forward pass is unchanged -- the core still runs its full
        # trajectory, and the frozen weights are untouched -- only the gradient
        # path is cut.
        if STAGE1_TBPTT_STEPS and (index + 1) % STAGE1_TBPTT_STEPS == 0:
            state = state.detach()
    return torch.stack(outputs, dim=1)


def run_stage1() -> dict:
    calibration_record = load_calibration()
    started = time.perf_counter()
    observations, actions, teacher_record = teacher_v4.build_teacher()
    print(json.dumps({"teacher": teacher_record["per_command"]}, sort_keys=True), flush=True)

    policy = _policy(CONDITION, calibration_record, STAGE1_SEED, obs_dim=observations.shape[2])
    optimizer = torch.optim.Adam(policy.interface_parameters(), lr=STAGE1_LEARNING_RATE)
    # Decay the learning rate when held-out stops improving.
    #
    # Measured on the first pod run: held-out fell to 0.0249 by epoch 606 and then
    # OSCILLATED rather than converged -- 0.0390, 0.0284, 0.0306, 0.0293, 0.0335,
    # 0.0323, 0.0307 at epochs 450..750. A bouncing tail with a lucky minimum is a
    # step size too large for the late phase, not a capacity ceiling, and no epoch
    # budget fixes it: patience early-stopped that run at 756 regardless.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=STAGE1_LR_PATIENCE, min_lr=STAGE1_MIN_LR)
    assert_frozen(policy, optimizer)
    core_before = SC.tensor_sha256(policy.core.W)
    device = policy.core.W.device
    observations_t = torch.from_numpy(observations).to(device)
    targets_t = torch.from_numpy(actions).to(device)
    # Held out within every command, not across commands: the tail of each demo.
    train_end = int(round(observations.shape[1] * STAGE1_TRAIN_FRACTION))

    def interface_state() -> dict:
        return {
            "encoder": {key: value.detach().cpu().clone() for key, value in policy.encoder.state_dict().items()},
            "decoder": {key: value.detach().cpu().clone() for key, value in policy.decoder.state_dict().items()},
        }

    best_heldout, best_epoch, best_state = float("inf"), 0, interface_state()
    history, stopped_early = [], False
    for epoch in range(1, STAGE1_MAX_EPOCHS + 1):
        # One forward pass, read for both splits, scored BEFORE the step. The
        # previous version ran a second full replay after optimizer.step() purely
        # to log, which doubled the cost of every epoch; the held-out columns of
        # this same tensor are the same quantity, one gradient step earlier, and
        # `interface_state()` is snapshotted here so the restored checkpoint is
        # the parameters that actually scored `heldout_loss`.
        prediction = _predict_per_demo(policy, observations_t)
        loss = torch.mean((prediction[:, :train_end] - targets_t[:, :train_end]) ** 2)
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite Stage-1 loss at epoch {epoch}")
        train_loss = float(loss.detach())
        heldout_loss = float(torch.mean(
            (prediction[:, train_end:].detach() - targets_t[:, train_end:]) ** 2))
        history.append({"epoch": epoch, "train_loss": train_loss, "heldout_loss": heldout_loss})
        if epoch % 25 == 0 or epoch == 1:
            print(json.dumps({"stage1": CONDITION, "epoch": epoch, "train_loss": train_loss,
                              "heldout_loss": heldout_loss}, sort_keys=True), flush=True)
        if heldout_loss < best_heldout - STAGE1_MIN_DELTA:
            best_heldout, best_epoch, best_state = heldout_loss, epoch, interface_state()
        elif epoch - best_epoch >= STAGE1_PATIENCE:
            stopped_early = True
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip before stepping. PPO in this file already clips both networks at
        # PPO_GRAD_CLIP_NORM, matching rsl_rl ppo.py:309-310 (max_grad_norm 1.0),
        # but Stage 1 shipped with no clip at all -- and Stage 1 is the half that
        # backpropagates through a 200-step recurrent core rollout, which is far
        # more explosion-prone than PPO's feedforward update.
        #
        # Unclipped BPTT is what the oscillating tail looks like: held-out went
        # 0.0283 at epoch 550 to 0.0460 at 800 on the run before this, and
        # 0.0390/0.0284/0.0306/0.0293/0.0335/0.0323/0.0307 across 450..750 on the
        # one before that. A single spiked step undoes hundreds of good ones.
        # grad_norm is recorded so the spikes are visible rather than assumed.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            policy.interface_parameters(), STAGE1_GRAD_CLIP_NORM))
        history[-1]["grad_norm"] = grad_norm
        # The PRE-clip norm, printed live. Clipping alone was not diagnostic: with
        # the clip in, held-out sat at 1.60 for 175 epochs -- not slow learning but
        # no learning, which is what an enormous norm looks like after clipping,
        # because clip_coef = max_norm/total_norm underflows the gradients to zero
        # in float32 and Adam then steps m/(sqrt(v)+eps) = 0. This number separates
        # that from every other explanation, so it is printed rather than only
        # stored in an artifact written at the end of the run.
        if epoch % 25 == 0 or epoch == 1:
            print(json.dumps({"stage1_grad": CONDITION, "epoch": epoch,
                              "grad_norm_preclip": grad_norm}, sort_keys=True), flush=True)
        optimizer.step()
        scheduler.step(heldout_loss)
    policy.encoder.load_state_dict(best_state["encoder"])
    policy.decoder.load_state_dict(best_state["decoder"])
    assert_frozen(policy, optimizer)
    if SC.tensor_sha256(policy.core.W) != core_before:
        raise AssertionError("VNC graph changed during Stage 1")

    STAGE1_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, STAGE1_CHECKPOINT)
    result = {
        "condition": CONDITION, "seed": STAGE1_SEED, "optimizer": "Adam",
        "learning_rate": STAGE1_LEARNING_RATE, "max_epochs": STAGE1_MAX_EPOCHS,
        "grad_clip_norm": STAGE1_GRAD_CLIP_NORM,
        "tbptt_steps": STAGE1_TBPTT_STEPS,
        "lr_schedule": f"ReduceLROnPlateau factor 0.5 patience {STAGE1_LR_PATIENCE} min_lr {STAGE1_MIN_LR}",
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "patience": STAGE1_PATIENCE, "min_delta": STAGE1_MIN_DELTA,
        "epochs_run": len(history), "stopped_early": stopped_early, "selected_epoch": best_epoch,
        "selection_rule": "best held-out epoch, restored; held out within each command",
        "commands": list(teacher_v4.TEACHER_COMMANDS),
        "steps_per_command": int(observations.shape[1]),
        "train_steps_per_command": train_end,
        "heldout_steps_per_command": int(observations.shape[1]) - train_end,
        "final_train_loss": history[best_epoch - 1]["train_loss"] if best_epoch else history[-1]["train_loss"],
        "final_heldout_loss": best_heldout,
        "interface_parameter_count": int(sum(p.numel() for p in policy.interface_parameters())),
        "core_parameters": int(sum(p.numel() for p in policy.core.parameters())),
        "core_tensor_sha256_pre": core_before,
        "core_tensor_sha256_post": SC.tensor_sha256(policy.core.W),
        "vnc_weights_trained": False,
        "observation_dim": int(observations.shape[2]),
        "history": history,
        "teacher": teacher_record,
        "wall_time_s": time.perf_counter() - started,
    }
    # The gate. Run 1, run 2 and DAgger all trained for hours on top of a prior
    # that scored worse than a constant action, and nothing in the pipeline
    # noticed, because nothing ever compared the fit to a baseline. Twenty
    # minutes of Stage 1 answers whether the next ten hours are worth buying.
    # Baseline measured on the SAME held-out slice the policy is scored on, and on
    # the SAME commands actually in this dataset. The shipped 0.0151 was the
    # variance about a global mean over all timesteps of a five-command set: a
    # different quantity from the held-out loss, and wrong outright once the
    # command set changes. Verified equal to 0.01459 (global) / 0.01452 (per
    # command) on the five-command data, so this reproduces the old number there
    # rather than moving the goalposts.
    held_targets = actions[:, train_end:]
    per_command_mean = actions[:, :train_end].mean(axis=1, keepdims=True)
    measured_baseline = float(np.mean((held_targets - per_command_mean) ** 2))
    result["constant_action_baseline"] = measured_baseline
    result["constant_action_baseline_shipped"] = STAGE1_CONSTANT_BASELINE
    result["gate_threshold"] = measured_baseline * STAGE1_GATE_FRACTION
    result["gate_passed"] = bool(best_heldout < result["gate_threshold"])
    STAGE1_JSON.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"stage1_gate": result["gate_passed"],
                      "heldout": best_heldout,
                      "threshold": result["gate_threshold"]}, sort_keys=True), flush=True)
    return result


# --- Stage 2 -----------------------------------------------------------------

class Agent:
    def __init__(self, calibration_record: dict, stage1: Path | None = STAGE1_CHECKPOINT):
        torch.manual_seed(STAGE1_SEED)
        self.condition = CONDITION
        self.device = torch.device(calibration_record["device"])
        self.batch_size = PPO_BATCH_SIZE
        self.obs_dim = obs_dim()
        self.policy = _policy(CONDITION, calibration_record, STAGE1_SEED, obs_dim=self.obs_dim)
        if stage1 is not None and stage1.exists():
            saved = torch.load(stage1, map_location=self.device, weights_only=True)
            try:
                self.policy.encoder.load_state_dict(saved["encoder"])
                self.policy.decoder.load_state_dict(saved["decoder"])
            except RuntimeError as error:
                # Belt and braces behind the _ARM suffix: a prior silently dropped
                # is worse than a run that refuses to start, because stage 2 would
                # then train from random initialisation while the log claims a
                # prior was loaded.
                raise SystemExit(
                    f"{stage1.name} does not fit this interface "
                    f"(hidden dims {list(self.policy.interface_hidden)}, core "
                    f"{self.policy.core_kind}): {error}\n"
                    "Run `stage1` for this architecture before `stage2`."
                ) from error
        self.log_std = nn.Parameter(
            torch.full((18,), math.log(PPO_INITIAL_STD), dtype=torch.float32, device=self.device)
        )
        self.value = ValueNet(self.obs_dim).to(self.device)
        self.vector_env = VectorHexapodTargetEnv(PPO_BATCH_SIZE, max_steps=EPISODE_STEPS)
        self.optimizer = torch.optim.Adam(
            list(self.policy.interface_parameters()) + [self.log_std] + list(self.value.parameters()),
            lr=PPO_INITIAL_LEARNING_RATE, eps=PPO_ADAM_EPS,
        )
        # Rollout segments are shorter than an episode, so the envs and the core
        # state have to survive between collect() calls or the policy would only
        # ever see the first PPO_ROLLOUT_STEPS of an episode.
        self._observation = None
        self._state = None
        # [DA]/[K] real-time motion generator supplying the decaying prior. It is
        # the same tripod generator hexapod_gait exposes, vectorised over the env
        # axis, and it holds only a phase -- nothing it produces reaches the
        # reported policy, because the weight is zero by then.
        self.stance = self.vector_env.envs[0].stance.astype(np.float64).copy()
        self.control_dt = float(self.vector_env.envs[0].dt)
        self.rtmg = BatchRTMG(PPO_BATCH_SIZE, stance=self.stance) if DECAP_ENABLED else None
        # [LG] adaptive schedule state; re-tuned every minibatch in update().
        self.learning_rate = PPO_INITIAL_LEARNING_RATE
        assert_frozen(self.policy, self.optimizer)

    def env_groups(self) -> tuple[torch.Tensor, ...]:
        """The fixed contiguous partition of the environment axis.

        The core's columns are independent, so slicing the batch is exact in real
        arithmetic. It is not exact in floating point: a sparse matmul kernel
        picks its tiling and its reduction order from the operand width, so the
        same column stepped at width 512 and at width 128 does not round the same
        way. Collection ran the whole batch at 512 and the update replayed each
        minibatch at 128, and 600 recurrent sparse matmuls amplified that
        rounding difference until the replayed policy was not the behaviour
        policy -- which is what PPO's importance ratio and the adaptive learning
        rate then measured, for seven hours, instead of the update.

        Both paths now step through this same partition at the same width, so the
        replay is the same arithmetic on any device. It also drops the per-epoch
        reshuffle, which moves toward the source rather than away from it: rsl_rl
        shuffles only in its feed-forward generator; its recurrent generator
        splits the environment axis into fixed contiguous blocks exactly here.
        """
        return torch.arange(self.batch_size).chunk(PPO_MINIBATCHES)

    def distribution(self, mean: torch.Tensor) -> Normal:
        return Normal(mean, self.log_std.clamp(PPO_LOG_STD_MIN, PPO_LOG_STD_MAX).exp())

    def _chunk(self, state, observations, dones):
        means = []
        for observation, done in zip(observations, dones):
            mean, state = self.policy.step_batch(observation, state)
            means.append(mean)
            state = self.policy.reset_state_batch(state, done)
        return torch.stack(means), state

    def policy_batch_sequence(self, observations: torch.Tensor, dones: torch.Tensor,
                              start_state: torch.Tensor) -> torch.Tensor:
        """Re-run the rollout under the current parameters, from the state it began in.

        `start_state` is not optional and must not be zeros. Rollout segments are
        shorter than an episode, so from the second iteration onwards a segment
        begins mid-episode with a non-zero core state. Replaying from zeros here
        would make the recomputed means differ from the behaviour means for a
        reason that has nothing to do with the parameter update -- inflating the
        KL, corrupting the importance ratio, and driving the adaptive learning
        rate down in response to a discrepancy the optimiser cannot fix.
        """
        state = start_state
        means = []
        for start in range(0, observations.shape[0], PPO_BPTT_CHUNK):
            chunk_obs = observations[start:start + PPO_BPTT_CHUNK]
            chunk_done = dones[start:start + PPO_BPTT_CHUNK]
            if torch.is_grad_enabled():
                chunk_means, state = torch.utils.checkpoint.checkpoint(
                    self._chunk, state, chunk_obs, chunk_done, use_reentrant=False
                )
            else:
                chunk_means, state = self._chunk(state, chunk_obs, chunk_done)
            state = state.detach()  # truncated BPTT; see PPO_BPTT_CHUNK
            means.append(chunk_means)
        return torch.cat(means)

    def collect(self, seed: int, horizon: int = PPO_ROLLOUT_STEPS, stochastic: bool = True,
                iteration: int = 0) -> dict:
        # [DA] eq. 4: tau = a ~ pi(s) + gamma^(t/k) * beta. Only the EXECUTED
        # action carries the bias; `actions` below stores the policy's own, which
        # is what the update scores and replays, so the prior is part of the
        # environment as far as PPO is concerned and no importance ratio is
        # distorted by it.
        weight = prior_weight(iteration, DECAP_GAMMA, DECAP_ZERO_AFTER) if self.rtmg else 0.0
        if self._observation is None:
            # Stride by batch_size so consecutive iterations do not re-draw almost
            # the same 512 seeds.
            seeds = (int(seed) * self.batch_size + np.arange(self.batch_size)).tolist()
            self._observation = self.vector_env.reset(seeds=seeds)
            self._state = self.policy.initial_state_batch(self.batch_size)
        observation, state = self._observation, self._state
        start_state = state.detach().clone()
        self.vector_env.reaches.fill(0)
        self.vector_env.mean_terms()   # clear; the segment's own sums start here
        groups = self.env_groups()
        observations, actions, rewards, values, old_log_probs, old_means, dones = [], [], [], [], [], [], []
        timeouts, terminal_values = [], []
        with torch.no_grad():
            for _ in range(horizon):
                observation_t = torch.from_numpy(observation).to(self.device)
                # Same widths as the update; see env_groups().
                means_parts, state_parts = [], []
                for group in groups:
                    part_mean, part_state = self.policy.step_batch(observation_t[group], state[:, group])
                    means_parts.append(part_mean)
                    state_parts.append(part_state)
                mean = torch.cat(means_parts, dim=0)
                state = torch.cat(state_parts, dim=1)
                distribution = self.distribution(mean)
                action = distribution.sample() if stochastic else mean
                old_means.append(mean.cpu().numpy())
                old_log_probs.append(distribution.log_prob(action).sum(dim=-1).cpu().numpy())
                values.append(self.value(observation_t).cpu().numpy())
                action_np = action.cpu().numpy()
                executed = action_np
                if weight != 0.0:
                    forward, turn = commands_from_observation(observation)
                    teacher = self.rtmg.targets(forward, turn, self.control_dt)
                    executed = apply_prior(action_np, teacher, self.stance, weight)
                next_observation, reward, done = self.vector_env.step(executed)
                observations.append(observation.copy())
                actions.append(action_np.astype(np.float32, copy=False))
                rewards.append(reward.copy())
                dones.append(done.copy())
                # An episode that ended on the clock has not been shown to be
                # worthless, so its value is bootstrapped rather than cut. The
                # observation to bootstrap from is the one before the auto-reset.
                timeout = self.vector_env.last_timeout_mask.copy()
                timeouts.append(timeout)
                if timeout.any():
                    final = torch.from_numpy(self.vector_env.last_terminal_obs).to(self.device)
                    terminal_values.append(np.where(timeout, self.value(final).cpu().numpy(), 0.0))
                else:
                    terminal_values.append(np.zeros(self.batch_size))
                state = self.policy.reset_state_batch(state, done)
                if self.rtmg is not None:
                    self.rtmg.reset(done)
                observation = next_observation
            next_value = self.value(torch.from_numpy(observation).to(self.device)).cpu().numpy().astype(np.float64)
        self._observation, self._state = observation, state.detach()

        values_np = np.asarray(values, dtype=np.float64)
        rewards_np = np.asarray(rewards, dtype=np.float64)
        dones_np = np.asarray(dones, dtype=bool)
        timeouts_np = np.asarray(timeouts, dtype=bool)
        terminal_values_np = np.asarray(terminal_values, dtype=np.float64)
        advantages = np.zeros_like(rewards_np)
        returns = np.zeros_like(rewards_np)
        next_advantage = np.zeros(self.batch_size, dtype=np.float64)
        for index in range(horizon - 1, -1, -1):
            nonterminal = 1.0 - dones_np[index]
            # [LG] compute_returns bootstraps through the time limit and only
            # zeroes the value of a genuine termination (here, a fall). The GAE
            # recursion still cuts at every boundary, because the next step
            # belongs to a different episode either way.
            bootstrap = np.where(timeouts_np[index], terminal_values_np[index], next_value * nonterminal)
            delta = rewards_np[index] + PPO_GAMMA * bootstrap - values_np[index]
            next_advantage = delta + PPO_GAMMA * PPO_GAE_LAMBDA * next_advantage * nonterminal
            advantages[index] = next_advantage
            returns[index] = advantages[index] + values_np[index]
            next_value = values_np[index]
        observations_np = np.asarray(observations, dtype=np.float32)
        return {
            "observations": observations_np,
            "actions": np.asarray(actions, dtype=np.float32),
            "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
            "old_means": np.asarray(old_means, dtype=np.float32),
            "old_values": values_np.astype(np.float32),
            "old_log_std": self.log_std.detach().cpu().numpy().astype(np.float32),
            "start_state": start_state,
            "returns": returns.astype(np.float32),
            "advantages": advantages.astype(np.float32),
            "dones": dones_np,
            # Segment return, not an episode return: the rollout is a slice of an
            # ongoing episode. Comparable across iterations, not to EPISODE_STEPS.
            "episode_return": float(rewards_np.sum(axis=0).mean()),
            # The endpoint, not the reward: how often a target was actually reached
            # per environment during this segment. Read from the vector wrapper's
            # banked counter, because a terminal step resets the env's own.
            "targets_reached": float(self.vector_env.reaches.mean()),
            "steps": int(horizon * self.batch_size),
            # How much of the executed action came from the motion generator this
            # iteration. Must reach exactly 0.0 and stay there; the evaluation is
            # only meaningful once it has.
            "prior_weight": float(weight),
            # The dense stand-in for the endpoint. Observation channel 11 is the
            # distance to the target, scaled and clipped; a run that is learning to
            # walk toward the sphere drives this down long before `targets_reached`
            # leaves zero. Without it the first useful signal is hours away.
            "target_distance_m": float(observations_np[:, :, 11].mean() * OBS_TARGET_DIST_SCALE),
            "reward_terms": self.vector_env.mean_terms(),
        }

    def value_warmup(self, episode: dict) -> float:
        observations = torch.from_numpy(episode["observations"]).to(self.device).reshape(-1, self.obs_dim)
        returns = torch.from_numpy(episode["returns"]).to(self.device).reshape(-1)
        # The main optimizer already owns the value parameters. A second Adam over
        # the same tensors would keep its own moment estimates, and whichever one
        # stepped second would be correcting with a stale second moment. Adam
        # skips parameters whose grad is None, so the interface and log_std are
        # untouched by a loss that does not reach them.
        loss = torch.mean((self.value(observations) - returns) ** 2)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), PPO_GRAD_CLIP_NORM)
        if not torch.isfinite(grad_norm):
            raise AssertionError("non-finite value-warmup gradient")
        self.optimizer.step()
        assert_frozen(self.policy, self.optimizer)
        return float(loss.detach())

    def update(self, episode: dict) -> dict[str, float]:
        observations = torch.from_numpy(episode["observations"]).to(self.device)
        actions = torch.from_numpy(episode["actions"]).to(self.device)
        old_log_probs = torch.from_numpy(episode["old_log_probs"]).to(self.device)
        returns = torch.from_numpy(episode["returns"]).to(self.device)
        advantages = torch.from_numpy(episode["advantages"]).to(self.device)
        dones = torch.from_numpy(episode["dones"]).to(self.device)
        old_means = torch.from_numpy(episode["old_means"]).to(self.device)
        old_values = torch.from_numpy(episode["old_values"]).to(self.device)
        old_std = torch.from_numpy(episode["old_log_std"]).to(self.device).exp()
        start_state = episode["start_state"].to(self.device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        # Minibatch over the environment axis. The time axis stays whole because
        # the core is recurrent and BPTT needs it contiguous. Four minibatches per
        # epoch is [LG]'s num_mini_batches, and it is also what keeps the
        # checkpointed core graph inside memory (see PPO_MINIBATCHES).
        last: dict[str, float] = {}
        # Per-minibatch trace. The summary row only reports the last minibatch of
        # the last epoch, which is how a learning rate that ramps and then blows
        # up inside a single update looked, from the log, like a flat run.
        self.update_trace = []
        for _ in range(PPO_UPDATE_EPOCHS):
            for slice_index in self.env_groups():
              means = self.policy_batch_sequence(observations[:, slice_index],
                                                 dones[:, slice_index],
                                                 start_state[:, slice_index])
              distribution = self.distribution(means)
              log_probs = distribution.log_prob(actions[:, slice_index]).sum(dim=-1)
              entropy = distribution.entropy().sum(dim=-1).mean()
              minibatch_returns = returns[:, slice_index]
              minibatch_advantages = advantages[:, slice_index]
              values = self.value(
                  observations[:, slice_index].reshape(-1, self.obs_dim)
              ).reshape_as(minibatch_returns)
              # rsl_rl PPO.update, schedule == 'adaptive': hold the policy step at
              # a fixed KL by re-tuning the learning rate, rather than trusting a
              # fixed rate to be right for a recurrent core whose gradient norm
              # spans orders of magnitude.
              std = self.log_std.clamp(PPO_LOG_STD_MIN, PPO_LOG_STD_MAX).exp()
              with torch.no_grad():
                  kl = torch.sum(
                      torch.log(std / old_std + 1.0e-5)
                      + (torch.square(old_std) + torch.square(old_means[:, slice_index] - means))
                      / (2.0 * torch.square(std)) - 0.5,
                      dim=-1,
                  ).mean()
                  if kl > PPO_DESIRED_KL * 2.0:
                      self.learning_rate = max(PPO_LEARNING_RATE_MIN, self.learning_rate / 1.5)
                  elif 0.0 < kl < PPO_DESIRED_KL / 2.0:
                      self.learning_rate = min(PPO_LEARNING_RATE_MAX, self.learning_rate * 1.5)
                  for group in self.optimizer.param_groups:
                      group["lr"] = self.learning_rate

              # Declared deviation from [LG], which has no such guard. The
              # adaptive schedule is reactive: it can shrink the next step but
              # cannot undo one that has already left the trust region, and every
              # later minibatch is then optimising a ratio computed against a
              # policy the parameters no longer resemble. Run 1 spent 13,000
              # updates in that state. Stop the update instead.
              if kl > PPO_DESIRED_KL * PPO_KL_ABORT_FACTOR:
                  print(json.dumps({"aborted_update": "kl", "kl": float(kl),
                                    "learning_rate": self.learning_rate}), flush=True)
                  assert_frozen(self.policy, self.optimizer)
                  return last or {"grad_norm": 0.0, "loss": float("nan"), "policy_loss": float("nan"),
                                  "value_loss": float("nan"), "entropy": float("nan"),
                                  "mean_ratio": float("nan"), "kl": float(kl),
                                  "learning_rate": self.learning_rate}

              ratio = torch.exp(log_probs - old_log_probs[:, slice_index])
              policy_loss = -torch.minimum(
                  ratio * minibatch_advantages,
                  torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * minibatch_advantages,
              ).mean()
              if PPO_USE_CLIPPED_VALUE_LOSS:
                  # rsl_rl PPO.update: the same trust region the policy gets, so a
                  # single minibatch cannot move the critic off the data it was
                  # collected under.
                  clipped = old_values[:, slice_index] + (
                      values - old_values[:, slice_index]
                  ).clamp(-PPO_CLIP, PPO_CLIP)
                  value_loss = torch.max((values - minibatch_returns) ** 2,
                                         (clipped - minibatch_returns) ** 2).mean()
              else:
                  value_loss = torch.mean((values - minibatch_returns) ** 2)
              loss = policy_loss + PPO_VALUE_LOSS_COEF * value_loss - PPO_ENTROPY_COEF * entropy
              if not torch.isfinite(loss):
                  raise AssertionError("non-finite PPO loss")
              self.optimizer.zero_grad(set_to_none=True)
              loss.backward()
              grad_norm = torch.nn.utils.clip_grad_norm_(
                  [parameter for group in self.optimizer.param_groups for parameter in group["params"]],
                  PPO_GRAD_CLIP_NORM,
              )
              if not torch.isfinite(grad_norm):
                  raise AssertionError("non-finite PPO gradient; lower PPO_BPTT_CHUNK")
              self.optimizer.step()
              self.update_trace.append({"kl": float(kl), "learning_rate": self.learning_rate,
                                        "grad_norm": float(grad_norm),
                                        "mean_ratio": float(ratio.detach().mean())})
              last = {
                  "grad_norm": float(grad_norm), "loss": float(loss.detach()),
                  "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                  "entropy": float(entropy.detach()), "mean_ratio": float(ratio.detach().mean()),
                  "kl": float(kl), "learning_rate": self.learning_rate,
              }
        assert_frozen(self.policy, self.optimizer)
        return last

    def save(self, path: Path, steps: int, episode_index: int, ema_return: float | None, history: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "condition": self.condition, "steps": steps, "episode_index": episode_index,
            "ema_return": ema_return, "learning_rate": self.learning_rate,
            "encoder": self.policy.encoder.state_dict(),
            "decoder": self.policy.decoder.state_dict(), "log_std": self.log_std.detach().cpu(),
            "value": self.value.state_dict(), "optimizer": self.optimizer.state_dict(),
            "history": history,
        }, path)

    def load(self, path: Path) -> dict:
        saved = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.encoder.load_state_dict(saved["encoder"])
        self.policy.decoder.load_state_dict(saved["decoder"])
        self.log_std.data.copy_(saved["log_std"])
        self.learning_rate = float(saved.get("learning_rate", PPO_INITIAL_LEARNING_RATE))
        self.value.load_state_dict(saved["value"])
        self.optimizer.load_state_dict(saved["optimizer"])
        assert_frozen(self.policy, self.optimizer)
        return saved


def run_stage2() -> dict:
    # Refuse to spend GPU hours on a prior that has not learned the gait. Runs 1
    # and 2 both did exactly that; see the gate in run_stage1. Set
    # V4_SKIP_STAGE1_GATE=1 to train from a failing prior deliberately.
    if STAGE1_JSON.exists() and not os.environ.get("V4_SKIP_STAGE1_GATE"):
        stage1_record = json.loads(STAGE1_JSON.read_text())
        if stage1_record.get("gate_passed") is False:
            raise SystemExit(
                f"Stage-1 gate failed: held-out {stage1_record['final_heldout_loss']:.5f} "
                f"is not below {stage1_record['gate_threshold']:.5f} "
                f"(a constant action scores {stage1_record['constant_action_baseline']}). "
                "The prior has not learned the gait; fix Stage 1 before training."
            )
    calibration_record = load_calibration()
    agent = Agent(calibration_record)
    core_before = SC.tensor_sha256(agent.policy.core.W)
    steps, episode_index, ema, history = 0, 0, None, []
    if PPO_CHECKPOINT.exists():
        saved = agent.load(PPO_CHECKPOINT)
        steps, episode_index = int(saved["steps"]), int(saved["episode_index"])
        ema, history = saved["ema_return"], saved.get("history", [])
        print(json.dumps({"resumed_at_steps": steps}, sort_keys=True), flush=True)

    started = time.perf_counter()
    next_checkpoint = steps + PPO_CHECKPOINT_INTERVAL
    warmup_losses = []
    stop_reason = "step_cap"
    stuck_for = 0
    while steps < ARM_STEP_CAP:
        if PPO_MAX_HOURS is not None and (time.perf_counter() - started) / 3600.0 >= PPO_MAX_HOURS:
            stop_reason = "wall_clock"
            print(json.dumps({"stopping": stop_reason, "hours": PPO_MAX_HOURS}), flush=True)
            break
        episode_index += 1
        episode = agent.collect(STAGE1_SEED + episode_index, iteration=episode_index)
        episode["iteration"] = episode_index
        steps += episode["steps"]
        if episode_index <= PPO_VALUE_WARMUP_ITERATIONS:
            warmup_losses.append(agent.value_warmup(episode))
            stats = {"phase": "value_warmup", "value_loss": warmup_losses[-1]}
        else:
            stats = agent.update(episode)
            stats["phase"] = "ppo"
        ema = episode["episode_return"] if ema is None else (1 - PPO_EMA_ALPHA) * ema + PPO_EMA_ALPHA * episode["episode_return"]
        row = {
            "iteration": episode_index, "steps": steps, "ema_return": float(ema),
            "episode_return": episode["episode_return"], "targets_reached": episode["targets_reached"],
            "target_distance_m": episode["target_distance_m"],
            "prior_weight": episode["prior_weight"],
            "track_lin": episode["reward_terms"]["tracking_lin_vel"],
            "track_ang": episode["reward_terms"]["tracking_ang_vel"],
            "air_time": episode["reward_terms"]["feet_air_time"],
            "upright": episode["reward_terms"]["upright"],
            "stance_dev": episode["reward_terms"]["stance_dev"],
            "height": episode["reward_terms"]["height"],
            "elapsed_s": time.perf_counter() - started, **stats,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        if stats["phase"] == "ppo":
            pinned = stats["learning_rate"] <= PPO_LEARNING_RATE_MIN
            outside = stats["kl"] > PPO_DESIRED_KL * PPO_STUCK_KL_FACTOR
            stuck_for = stuck_for + 1 if (pinned and outside) else 0
            if stuck_for >= PPO_STUCK_ITERATIONS:
                stop_reason = "stuck"
                print(json.dumps({
                    "stopping": stop_reason, "iterations_stuck": stuck_for,
                    "kl": stats["kl"], "learning_rate": stats["learning_rate"],
                    "diagnosis": "learning rate pinned at its floor with KL outside the "
                                 "trust region: the replayed policy is not the behaviour "
                                 "policy, so further wall clock buys nothing. This is the "
                                 "run 1 failure; check PPO_ROLLOUT_STEPS.",
                }, sort_keys=True), flush=True)
                break
        if steps >= next_checkpoint or steps >= ARM_STEP_CAP:
            agent.save(PPO_CHECKPOINT, steps, episode_index, float(ema), history)
            next_checkpoint = steps + PPO_CHECKPOINT_INTERVAL

    final_ema = None if ema is None else float(ema)
    agent.save(PPO_CHECKPOINT, steps, episode_index, final_ema, history)
    if SC.tensor_sha256(agent.policy.core.W) != core_before:
        raise AssertionError("VNC graph changed during Stage 2")
    if stop_reason == "stuck":
        # Checkpoint first (done above), then fail loudly: the pod runner must not
        # go on to evaluate a policy that never trained and call the night a success.
        raise AssertionError(
            f"stage 2 stopped after {stuck_for} iterations with the learning rate pinned "
            f"at {PPO_LEARNING_RATE_MIN} and KL outside {PPO_DESIRED_KL * PPO_STUCK_KL_FACTOR}; "
            "this is run 1's failure mode and the checkpoint is not worth evaluating"
        )
    # Provenance for the prior. `decap_reached_zero` is the one that matters when
    # the result is written up: false means the run stopped while the motion
    # generator was still contributing, so the policy was trained expecting help
    # that the evaluation does not give it.
    final_prior = prior_weight(episode_index, DECAP_GAMMA, DECAP_ZERO_AFTER) if DECAP_ENABLED else 0.0
    return {"steps": steps, "episodes": episode_index, "final_ema_return": final_ema,
            "decap_enabled": DECAP_ENABLED, "decap_gamma": DECAP_GAMMA,
            "decap_zero_after": DECAP_ZERO_AFTER,
            "decap_final_prior_weight": float(final_prior),
            "decap_reached_zero": bool(final_prior == 0.0),
            "stop_reason": stop_reason, "step_cap": ARM_STEP_CAP,
            "update_epochs": PPO_UPDATE_EPOCHS, "minibatches": PPO_MINIBATCHES,
            "rollout_steps": PPO_ROLLOUT_STEPS, "bptt_chunk": PPO_BPTT_CHUNK,
            "warmup_value_loss_mean": float(np.mean(warmup_losses)) if warmup_losses else None,
            "wall_time_s": time.perf_counter() - started,
            "core_tensor_sha256_pre": core_before,
            "core_tensor_sha256_post": SC.tensor_sha256(agent.policy.core.W)}


# --- evaluation --------------------------------------------------------------

def evaluate_episode(agent: Agent, seed: int, video_path: Path | None = None) -> dict:
    env = HexapodTargetEnv(max_steps=EPISODE_STEPS)
    observation = env.reset(seed=seed)
    state = agent.policy.initial_state()
    start_xy = env.data.qpos[:2].copy()
    last_xy = start_xy.copy()
    path_length = 0.0
    closest = float(np.linalg.norm(env._target - start_xy))
    writer = renderer = cameras = None
    if video_path is not None:
        # The video is a nice-to-have; the numbers are the deliverable. An
        # unattended pod run once reached this line with no EGL library in the
        # image, and the whole evaluation died there -- so a headless box came
        # back with a checkpoint and no endpoint at all. Render if we can.
        try:
            import mujoco
            import imageio.v2 as imageio

            video_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(env.model, 480, 640)
            cameras = SC.make_cameras(env.model)
            writer = imageio.get_writer(str(video_path), fps=25, codec="libx264")
        except Exception as error:  # no GL, no codec, no imageio
            print(json.dumps({"video": "unavailable", "reason": f"{type(error).__name__}: {error}"}), flush=True)
            if writer is not None:
                writer.close()
            writer = renderer = cameras = None

    def frame():
        views = []
        for camera in cameras:
            renderer.update_scene(env.data, camera=camera)
            views.append(renderer.render())
        return np.vstack((np.hstack((views[0], views[1])), np.hstack((views[2], views[3]))))

    steps = 0
    episode_return = 0.0
    done = False
    with torch.no_grad():
        while not done:
            mean, state = agent.policy.step(torch.from_numpy(observation), state)
            action = mean.cpu().numpy().astype(np.float32)
            observation, reward, done = env.step(action)
            episode_return += reward
            steps += 1
            now = env.data.qpos[:2].copy()
            path_length += float(np.linalg.norm(now - last_xy))
            last_xy = now
            closest = min(closest, float(np.linalg.norm(env._target - now)))
            if writer is not None and steps % 2 == 0:
                writer.append_data(frame())
    if writer is not None:
        writer.close()
    return {
        "episode_seed": int(seed),
        "video_written": writer is not None,
        # The preregistered endpoint for this task. Distance walked is secondary:
        # v3 taught us that a body can score well and go nowhere.
        "targets_reached": int(env._targets_reached),
        "closest_approach_m": closest,
        "reach_radius_m": TARGET_REACH_M,
        "path_m": path_length,
        "displacement_m": float(np.linalg.norm(env.data.qpos[:2] - start_xy)),
        # Path over displacement. A rocker has a large ratio, a walker's is near 1.
        "path_over_displacement": float(path_length / max(float(np.linalg.norm(env.data.qpos[:2] - start_xy)), 1e-6)),
        "steps": steps,
        "upright_s": float(steps / env.control_hz),
        "min_height_m": float(env._min_height),
        "fell": bool(env._min_height < H_MIN),
        "episode_return": episode_return,
    }


def run_evaluate() -> dict:
    calibration_record = load_calibration()
    agent = Agent(calibration_record)
    saved = agent.load(PPO_CHECKPOINT)
    rows = [evaluate_episode(agent, seed, VIDEO_PATH if seed == VIDEO_SEED else None)
            for seed in EVALUATION_SEEDS]
    reached = np.array([row["targets_reached"] for row in rows], dtype=float)
    report = {
        "condition": CONDITION,
        "checkpoint_steps": int(saved["steps"]),
        "episodes": len(rows),
        "targets_reached_mean": float(reached.mean()),
        "episodes_reaching_at_least_one": int((reached >= 1).sum()),
        "closest_approach_mean_m": float(np.mean([row["closest_approach_m"] for row in rows])),
        "path_over_displacement_mean": float(np.mean([row["path_over_displacement"] for row in rows])),
        "falls": int(sum(row["fell"] for row in rows)),
        "upright_s_mean": float(np.mean([row["upright_s"] for row in rows])),
        "video": str(VIDEO_PATH) if any(row["video_written"] for row in rows) else None,
        "rows": rows,
    }
    EVAL_JSON.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


# --- smoke -------------------------------------------------------------------

def replay_fidelity(agent: "Agent", episode: dict) -> dict:
    """Does replaying the rollout reproduce the policy that generated it?

    PPO's importance ratio and the adaptive learning rate both compare the current
    policy against the behaviour policy by re-running the rollout, so a replay
    that does not reproduce it before any update makes both quantities measure the
    discrepancy instead of the update.

    Read what this does and does not prove. Under *unchanged* parameters the
    replay is bit-exact at any horizon, on CPU and (as run 1 showed) on CUDA, so
    passing here is necessary and nowhere near sufficient: run 1's earlier version
    of this check passed on a machine that then trained for seven hours without
    moving. What actually broke was divergence under a *changed* parameter -- the
    recurrent core amplifies a 1e-8 weight change to an O(1) action change within
    about 30 control steps. That is what PPO_ROLLOUT_STEPS is set from, and what
    the KL reported by the first minibatch of update() measures.
    """
    observations = torch.from_numpy(episode["observations"]).to(agent.device)
    dones = torch.from_numpy(episode["dones"]).to(agent.device)
    behaviour = torch.from_numpy(episode["old_means"]).to(agent.device)
    start_state = episode["start_state"].to(agent.device)
    index = agent.env_groups()[0]
    means = agent.policy_batch_sequence(observations[:, index], dones[:, index],
                                        start_state[:, index])
    delta = (means.detach() - behaviour[:, index]).abs()
    standard_deviation = agent.log_std.detach().clamp(PPO_LOG_STD_MIN, PPO_LOG_STD_MAX).exp()
    divergence = float(torch.sum(
        (means.detach() - behaviour[:, index]) ** 2 / (2.0 * standard_deviation ** 2), dim=-1
    ).mean())
    return {"replay_max_delta": float(delta.max()), "replay_kl": divergence,
            "replay_steps": int(observations.shape[0])}


def run_smoke() -> dict:
    """Short rollout and one real update. Fails loudly on the three known killers:
    a non-finite gradient from the recurrent core, an unfrozen core, and a replay
    that does not reproduce the behaviour policy on this device."""
    calibration_record = load_calibration()
    agent = Agent(calibration_record)
    core_before = SC.tensor_sha256(agent.policy.core.W)
    # The real rollout length, so the smoke exercises the BPTT depth and the
    # memory footprint stage2 will actually run at.
    episode = agent.collect(STAGE1_SEED, horizon=PPO_ROLLOUT_STEPS)
    episode["iteration"] = 0
    fidelity = replay_fidelity(agent, episode)
    # Before any update the replayed policy IS the behaviour policy, so this is a
    # numerical identity, not a tolerance to be tuned. 1e-3 is slack, not a target.
    if fidelity["replay_kl"] > 1e-3:
        raise AssertionError(
            f"replay does not reproduce the behaviour policy on {agent.device}: "
            f"KL {fidelity['replay_kl']:.4g}, max mean delta "
            f"{fidelity['replay_max_delta']:.4g} over {fidelity['replay_steps']} steps. "
            "PPO's importance ratio would be meaningless; do not train on this device."
        )
    stats = agent.update(episode)
    assert np.isfinite(stats["grad_norm"]), stats
    if SC.tensor_sha256(agent.policy.core.W) != core_before:
        raise AssertionError("core changed during smoke update")
    return {"obs_dim": agent.obs_dim, "episode_return": episode["episode_return"],
            "targets_reached": episode["targets_reached"], "core_frozen": True,
            "device": str(agent.device), **fidelity, **stats}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage1", "smoke", "stage2", "evaluate"))
    arguments = parser.parse_args()
    result = {"stage1": run_stage1, "smoke": run_smoke,
              "stage2": run_stage2, "evaluate": run_evaluate}[arguments.command]()
    printable = {key: value for key, value in result.items() if key not in ("rows", "history", "teacher")}
    print(f"EMBODY {TAG.upper()} {arguments.command.upper()}: OK")
    print(json.dumps(printable, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
