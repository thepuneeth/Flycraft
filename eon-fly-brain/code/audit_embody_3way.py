"""Pre-register and run the v2 frozen-core three-way embodiment experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
PREFIX = RESULTS / "embody_3way"
# v3 writes to its own artifact names so the v2 run stays on disk untouched and the two
# remain directly comparable. v3 differs from v2 in exactly two places: the Stage-1
# stopping rule (convergence, not a fixed 20 steps) and the Stage-2 reward (bounded
# multiplicative factors, not unbounded displacement).
TAG = "v3"
PREREGISTERED = RESULTS / f"embody_3way_{TAG}_preregistered.json"
STAGE1_JSON = RESULTS / f"embody_3way_{TAG}_stage1.json"
STAGE1_CSV = RESULTS / f"embody_3way_{TAG}_stage1.csv"
STAGE1_CHECKPOINTS = {condition: RESULTS / f"embody_3way_{TAG}_stage1_{condition}.pt" for condition in ("connectome", "random", "rewired")}
COMMAND_MAP_PATH = RESULTS / "dn_command_map.npz"
sys.path.insert(0, str(HERE))

import specificity_campaign as SC  # noqa: E402
from hexapod_gait import home_stance  # noqa: E402
from audit_amplitude_control import amplitude_match  # noqa: E402
from collect_reservoir_states import PROTOCOL  # noqa: E402
from embody_env import HexapodEnv  # noqa: E402
from embody_policy import COMMAND_MAP_SHA256, EmbodyPolicy, _assert_frozen, command_map_sha256, load_command_map, resolve_device  # noqa: E402
from rung1_readout import DYNAMIC_CONDITIONS, condition_stepper  # noqa: E402


CONDITIONS = tuple(DYNAMIC_CONDITIONS)
assert CONDITIONS == ("connectome", "random", "rewired")

STAGE1_SEED = 20260829
EVALUATION_SEEDS = tuple(20260830 + index for index in range(12))
# Stage 1 used to stop every arm after STAGE1_EPOCHS full-batch gradient steps. That
# equalises compute, not convergence, and the two are not the same thing here: at 20
# steps the controls had flattened (held-out 0.0755 / 0.0721) while the connectome was
# still descending steeply (0.694 -> 0.614 on the final step) and stopped at 0.6135.
# Stage 2 then started the three arms from an ~8x unequal fit, with the handicap running
# against the connectome, which confounds the whole comparison. Given more steps the
# connectome reaches ~0.030 held-out -- better than either control -- so the gap was a
# budget artifact, not a capability difference.
#
# Equalise convergence instead: every arm trains until its held-out loss stops improving
# for STAGE1_PATIENCE consecutive epochs, under a shared STAGE1_MAX_EPOCHS ceiling, and
# the interface is restored to its best held-out epoch. Identical rule for all arms; the
# realised epoch count per arm is recorded so any residual gap stays visible.
STAGE1_MAX_EPOCHS = 600
STAGE1_PATIENCE = 60
STAGE1_MIN_DELTA = 1e-4
STAGE1_LEARNING_RATE = 1e-3
STAGE1_STEPS = 200
STAGE1_TRAIN_FRACTION = 0.70
STAGE1_HELDOUT_FRACTION = 0.30

# Same practical activity floor used by collect_reservoir_states.scaler.
# Channels below it are declared effectively zero-variance before training.
AMPLITUDE_SD_EPS = 1e-7
# Cap on the amplitude-match multiplier. A control channel whose own SD is far
# below the real channel's needs an enormous factor to "match" it, which amplifies
# noise instead of matching signal and distorts only the control arms (observed up
# to 1.23e6x before this cap). Channels needing more than the cap are declared
# UNMATCHED and keep identity scale; they still feed the decoder.
AMPLITUDE_MAX_SCALE = 10.0

# Option-B teacher: search a generic periodic joint controller against body path
# error only. No HexapodGait target is supplied to the interface.
TEACHER_SEED = 20260830
TEACHER_SPEED_MPS = 0.10
TEACHER_FREQ_HZ = 1.20
TEACHER_POPULATION = 24
TEACHER_ELITES = 6
TEACHER_ROUNDS = 3
TEACHER_FALL_PENALTY = 1.0
TEACHER_PARAM_LOW = (-0.35, 0.0, 0.0, -math.pi)
TEACHER_PARAM_HIGH = (0.35, 0.70, 0.70, math.pi)
TEACHER_INITIAL_MEAN = (0.05, 0.20, 0.10, 0.0)
TEACHER_INITIAL_STD = (0.10, 0.20, 0.20, math.pi)
TEACHER_STD_FLOOR = (0.02, 0.05, 0.05, 0.20)

PPO_STEPS_PER_ARM = 5_000_000
PPO_BATCH_SIZE = 128
PPO_GAMMA = 0.99
PPO_CLIP = 0.20
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.50
PPO_LEARNING_RATE = 1e-4
PPO_VALUE_WARMUP_STEPS = 102_400
PPO_CHECKPOINT_INTERVAL = 100_000

PRE_REGISTERED_NULL = (
    "connectome ties or loses to the amplitude-matched random arm under identical training "
    "→ the structural advantage does not survive a trainable interface; the moat is zero-training-only."
)
DECISION_LABELS = ("EMBODY-MOAT-EARNED", "EMBODY-MOAT-NULL", "EMBODY-PILOT-INCONCLUSIVE")


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _serial(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [_serial(item) for item in value]
    if isinstance(value, dict):
        return {key: _serial(item) for key, item in value.items()}
    return value


def _collect_history(condition: str) -> tuple[object, np.ndarray]:
    vnc = condition_stepper(SC.VNC_DATA, condition)
    current = vnc.descending_current(PROTOCOL["drive"], dnb08_gain=PROTOCOL["dnb08_gain"])
    for _ in range(PROTOCOL["burn_steps"]):
        vnc.step(current)
    history = np.stack(
        [vnc.step(current).cpu().numpy() for _ in range(PROTOCOL["record_steps"])]
    ).astype(np.float32)
    if history.shape != (PROTOCOL["record_steps"], len(vnc.io["motor_out"])):
        raise AssertionError(f"invalid calibration history shape for {condition}: {history.shape}")
    if not np.isfinite(history).all():
        raise AssertionError(f"non-finite calibration history for {condition}")
    return vnc, history


def calibration(device: str = "auto") -> dict:
    device = resolve_device(device)
    file_hashes = SC.file_hashes(SC.VNC_DATA)
    if file_hashes != SC.EXPECTED_SHA256:
        raise AssertionError(f"real graph input hash mismatch before embodiment: {file_hashes}")

    vnc_by_condition = {}
    histories = {}
    graph_hashes = {}
    for condition in CONDITIONS:
        vnc, history = _collect_history(condition)
        vnc_by_condition[condition] = vnc
        histories[condition] = history
        graph_hashes[condition] = SC.tensor_sha256(vnc.W)

    history64 = {condition: value.astype(np.float64) for condition, value in histories.items()}
    temporal_sd = {condition: value.std(axis=0) for condition, value in history64.items()}
    active = temporal_sd["connectome"] > AMPLITUDE_SD_EPS
    for condition in ("random", "rewired"):
        active &= temporal_sd[condition] > AMPLITUDE_SD_EPS
    active_indices = np.flatnonzero(active).astype(np.int64)
    if len(active_indices) == 0:
        raise AssertionError("amplitude matching left no common motor channels")

    target_sd = temporal_sd["connectome"][active]
    amplitude = {}
    affine = {}
    for condition in CONDITIONS:
        source = history64[condition][:, active]
        source_mean = source.mean(axis=0)
        source_sd = source.std(axis=0)
        if condition == "connectome":
            matched = source
            scale = np.ones_like(source_sd)
        else:
            # Required helper is the validation path; float64 affine values are
            # used at runtime so tiny SDs do not create avoidable float32 error.
            helper_matched, helper_record = amplitude_match(source, target_sd)
            if not np.isfinite(helper_matched).all():
                raise AssertionError(f"helper amplitude match produced non-finite values for {condition}")
            if np.any(source_sd <= 0.0):
                raise AssertionError(f"zero-variance source survived active mask for {condition}")
            raw_scale = target_sd / source_sd
            over_cap = raw_scale > AMPLITUDE_MAX_SCALE
            scale = np.where(over_cap, 1.0, raw_scale)
            matched = source_mean + (source - source_mean) * scale
            amplitude[condition] = {
                "helper": helper_record,
                "target_sd": target_sd.tolist(),
                "source_sd": source_sd.tolist(),
                "matched_sd": matched.std(axis=0).tolist(),
                "max_sd_error_all_channels": float(np.max(np.abs(matched.std(axis=0) - target_sd))),
                "max_scale_cap": AMPLITUDE_MAX_SCALE,
                "raw_scale_max": float(np.max(raw_scale)),
                "applied_scale_max": float(np.max(scale)),
                "unmatched_over_cap_count": int(np.count_nonzero(over_cap)),
                "unmatched_over_cap_indices": active_indices[over_cap].tolist(),
                "matched_sd_error_capped_channels_excluded": float(
                    np.max(np.abs(matched.std(axis=0) - target_sd)[~over_cap]) if np.any(~over_cap) else 0.0
                ),
                "mean_preserved_max_error": float(np.max(np.abs(matched.mean(axis=0) - source_mean))),
                "source_mean": source_mean.tolist(),
                "matched_mean": matched.mean(axis=0).tolist(),
                "matched_history_sha256": array_sha256(matched.astype(np.float32)),
            }
        affine[condition] = {
            "mean": source_mean.tolist(),
            "scale": scale.tolist(),
        }
        if not np.isfinite(matched).all():
            raise AssertionError(f"non-finite matched history for {condition}")
        if float(np.max(scale)) > AMPLITUDE_MAX_SCALE:
            raise AssertionError(
                f"{condition} applied amplitude scale {float(np.max(scale)):.4g} exceeds cap {AMPLITUDE_MAX_SCALE}"
            )

    excluded = np.flatnonzero(~active).astype(np.int64)
    calibration_record = {
        "device": device,
        "protocol": {
            "model": PROTOCOL["model"],
            "drive": PROTOCOL["drive"],
            "dnb08_gain": PROTOCOL["dnb08_gain"],
            "burn_steps": PROTOCOL["burn_steps"],
            "record_steps": PROTOCOL["record_steps"],
            "substeps": PROTOCOL["substeps"],
            "tau": PROTOCOL["tau"],
            "dt": PROTOCOL["dt"],
            "gW": PROTOCOL["gW"],
            "threshold": PROTOCOL["threshold"],
            "activation_gain": PROTOCOL["activation_gain"],
            "fr_cap": PROTOCOL["fr_cap"],
            "heterogeneity": PROTOCOL["heterogeneity"],
            "seed": PROTOCOL["seed"],
        },
        "motor_output_dim": int(histories["connectome"].shape[1]),
        "active_indices": active_indices.tolist(),
        "active_indices_sha256": array_sha256(active_indices),
        "excluded_indices": excluded.tolist(),
        "excluded_count": int(len(excluded)),
        "sd_threshold": AMPLITUDE_SD_EPS,
        "amplitude_max_scale": AMPLITUDE_MAX_SCALE,
        "exact_zero_sd_channels": {condition: int(np.count_nonzero(temporal_sd[condition] == 0.0)) for condition in CONDITIONS},
        "below_threshold_channels": {condition: int(np.count_nonzero(temporal_sd[condition] <= AMPLITUDE_SD_EPS)) for condition in CONDITIONS},
        "unmatchable_nonzero_target_channels_before_exclusion": {
            condition: int(np.count_nonzero((temporal_sd["connectome"] > AMPLITUDE_SD_EPS) & (temporal_sd[condition] <= AMPLITUDE_SD_EPS)))
            for condition in ("random", "rewired")
        },
        "real_target_sd_active": target_sd.tolist(),
        "amplitude_match": amplitude,
        "runtime_affine": affine,
        "graph_tensor_sha256": graph_hashes,
        "file_hashes_pre": file_hashes,
    }
    return calibration_record


def _leg_posture_map() -> dict | None:
    """The reduced-agency decoder, off unless V4_LEG_POSTURE=1.

    With it, the decoder emits one scalar per leg and a static map (built by
    build_leg_posture_map.py from the teacher's own joint covariance) expands each
    to three joint targets. Measured, one scalar carries mean 0.892 of a leg's
    joint motion, so the 18-output decoder was mostly fitting detail the gait does
    not contain. Timing stays entirely in the connectome: the map has no clock.
    """
    if os.environ.get("V4_LEG_POSTURE") != "1":
        return None
    path = SC.VNC_DATA / "leg_posture_map.npz"
    if not path.exists():
        raise SystemExit(
            f"V4_LEG_POSTURE=1 but {path} is missing; run build_leg_posture_map.py first")
    with np.load(path) as bundle:
        return {"home": bundle["home"], "direction": bundle["direction"]}


def _policy(condition: str, calibration_record: dict, seed: int, obs_dim: int | None = None) -> EmbodyPolicy:
    """Build the frozen-core policy for one arm.

    `obs_dim` defaults to the v2/v3 proprioception-only observation. v4 passes the
    wider goal-directed observation (see embody_env_v4); nothing else about the
    policy changes between versions, so the two share this factory rather than
    forking it.
    """
    vnc = condition_stepper(SC.VNC_DATA, condition)
    expected_hash = calibration_record["graph_tensor_sha256"][condition]
    if SC.tensor_sha256(vnc.W) != expected_hash:
        raise AssertionError(f"{condition} graph hash changed before training")
    active = np.asarray(calibration_record["active_indices"], dtype=np.int64)
    affine = calibration_record["runtime_affine"][condition]
    torch.manual_seed(seed)
    full_motor_indices = np.arange(len(vnc.motor_out), dtype=np.int64)
    full_mean = np.zeros(len(full_motor_indices), dtype=np.float32)
    full_scale = np.ones(len(full_motor_indices), dtype=np.float32)
    full_mean[active] = np.asarray(affine["mean"], dtype=np.float32)
    full_scale[active] = np.asarray(affine["scale"], dtype=np.float32)
    policy = EmbodyPolicy(
        obs_dim=HexapodEnv(max_steps=1).obs_dim if obs_dim is None else int(obs_dim),
        data=SC.VNC_DATA,
        frozen_weight=vnc.W,
        weights_name="all_legs_weight_csr.pkl",
        index_name="all_legs_io_index.npz",
        motor_indices=full_motor_indices,
        motor_mean=full_mean,
        motor_scale=full_scale,
        command_map=load_command_map(COMMAND_MAP_PATH),
        leg_posture_map=_leg_posture_map(),
        # legged_gym legged_robot.py:368 -- actions are offsets from the default
        # standing pose, scaled by action_scale (0.5 in legged_robot_config.py:95).
        # Absolute targets made a fresh policy collapse in 0.33 s.
        default_pose=np.asarray(home_stance(), dtype=np.float32).reshape(-1),
        action_scale=0.5,
        device=calibration_record["device"],
    )
    if not policy.core.W.is_sparse_csr:
        raise AssertionError(f"{condition} policy core is not CSR")
    if policy.core.W.requires_grad or isinstance(policy.core.W, torch.nn.Parameter):
        raise AssertionError(f"{condition} core weight is trainable")
    for buffer in policy.core.buffers():
        if buffer.requires_grad:
            raise AssertionError(f"{condition} core buffer has gradients")
    return policy


def _teacher_action(params: np.ndarray, step: int, stance: np.ndarray, dt: float) -> np.ndarray:
    action = stance.reshape(6, 3).copy()
    phase = 2.0 * math.pi * TEACHER_FREQ_HZ * step * dt + params[:, 3]
    swing = np.sin(phase)
    lift = np.maximum(swing, 0.0)
    action[:, 0] += params[:, 0] * swing
    action[:, 1] += params[:, 1] * lift
    action[:, 2] += params[:, 2] * lift
    return action.reshape(18).astype(np.float32)


def _teacher_rollout(params: np.ndarray, steps: int) -> tuple[float, bool, np.ndarray, np.ndarray]:
    env = HexapodEnv(max_steps=steps)
    observation = env.reset()
    x0, y0 = float(observation[0]), float(observation[1])
    yaw0 = float(2.0 * math.atan2(observation[6], observation[3]))
    actual = []
    actions = []
    fell = False
    for step in range(steps):
        action = _teacher_action(params, step, env.stance, 1.0 / env.control_hz)
        actions.append(action)
        next_observation, _, done = env.step(action)
        actual.append(
            (float(next_observation[0]) - x0) * math.cos(yaw0)
            + (float(next_observation[1]) - y0) * math.sin(yaw0)
        )
        observation = next_observation
        if done and step + 1 < steps:
            fell = True
            actual.extend([actual[-1]] * (steps - step - 1))
            break
    actual = np.asarray(actual, dtype=np.float64)
    if len(actual) != steps:
        actual = np.pad(actual, (0, steps - len(actual)), mode="edge")
    reference = TEACHER_SPEED_MPS * np.arange(1, steps + 1, dtype=np.float64) / env.control_hz
    loss = float(np.mean((actual - reference) ** 2) + TEACHER_FALL_PENALTY * fell)
    return loss, fell, np.asarray(actions, dtype=np.float32), actual


def build_teacher() -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(TEACHER_SEED)
    mean = np.tile(np.asarray(TEACHER_INITIAL_MEAN, dtype=np.float64), (6, 1))
    std = np.tile(np.asarray(TEACHER_INITIAL_STD, dtype=np.float64), (6, 1))
    low = np.tile(np.asarray(TEACHER_PARAM_LOW, dtype=np.float64), (6, 1))
    high = np.tile(np.asarray(TEACHER_PARAM_HIGH, dtype=np.float64), (6, 1))
    floor = np.tile(np.asarray(TEACHER_STD_FLOOR, dtype=np.float64), (6, 1))
    best = None
    history = []
    for round_index in range(TEACHER_ROUNDS):
        samples = np.clip(rng.normal(mean, std, size=(TEACHER_POPULATION, 6, 4)), low, high)
        scored = []
        for params in samples:
            loss, fell, actions, actual = _teacher_rollout(params, STAGE1_STEPS)
            scored.append((loss, fell, params, actions, actual))
        scored.sort(key=lambda row: row[0])
        elites = scored[:TEACHER_ELITES]
        mean = np.mean([row[2] for row in elites], axis=0)
        std = np.maximum(np.std([row[2] for row in elites], axis=0), floor)
        winner = elites[0]
        best = winner if best is None or winner[0] < best[0] else best
        history.append({"round": round_index + 1, "best_loss": float(winner[0]), "fell": bool(winner[1])})
    if best is None:
        raise AssertionError("teacher search produced no candidate")

    _, fell, _, actions, actual = best
    env = HexapodEnv(max_steps=STAGE1_STEPS)
    observation = env.reset()
    observations = []
    done = False
    for action in actions:
        observations.append(observation)
        if done:
            continue
        observation, _, done = env.step(action)
    observations = np.asarray(observations, dtype=np.float32)
    if observations.shape != (STAGE1_STEPS, env.obs_dim):
        raise AssertionError(f"teacher observation shape changed: {observations.shape}")
    record = {
        "method": "CEM generic periodic joint controller selected only by constant-speed forward body-path MSE plus declared fall penalty",
        "option": "B",
        "speed_mps": TEACHER_SPEED_MPS,
        "frequency_hz": TEACHER_FREQ_HZ,
        "population": TEACHER_POPULATION,
        "elites": TEACHER_ELITES,
        "rounds": TEACHER_ROUNDS,
        "seed": TEACHER_SEED,
        "fall_penalty": TEACHER_FALL_PENALTY,
        "best_path_mse_m2": float(np.mean((actual - TEACHER_SPEED_MPS * np.arange(1, STAGE1_STEPS + 1) / env.control_hz) ** 2)),
        "best_forward_displacement_m": float(actual[-1]),
        "fell": bool(fell),
        "actions_sha256": array_sha256(actions),
        "observations_sha256": array_sha256(observations),
        "search_history": history,
    }
    return observations, actions, record


def _predict(policy: EmbodyPolicy, observations: torch.Tensor) -> torch.Tensor:
    state = policy.initial_state()
    predictions = []
    for observation in observations:
        action, state = policy.step(observation, state)
        predictions.append(action)
    return torch.stack(predictions)


def _stage1_fit(condition: str, observations: np.ndarray, targets: np.ndarray, calibration_record: dict) -> dict:
    policy = _policy(condition, calibration_record, STAGE1_SEED)
    optimizer = torch.optim.Adam(policy.interface_parameters(), lr=STAGE1_LEARNING_RATE)
    _assert_frozen(policy, optimizer)
    before = SC.tensor_sha256(policy.core.W)
    device = policy.core.W.device
    observations_t = torch.from_numpy(observations).to(device)
    targets_t = torch.from_numpy(targets).to(device)
    train_end = int(round(len(targets) * STAGE1_TRAIN_FRACTION))
    history = []
    started = time.perf_counter()
    with torch.no_grad():
        initial_prediction = _predict(policy, observations_t)
        initial_train = float(torch.mean((initial_prediction[:train_end] - targets_t[:train_end]) ** 2))
        initial_heldout = float(torch.mean((initial_prediction[train_end:] - targets_t[train_end:]) ** 2))
    def _interface_state() -> dict:
        return {
            "encoder": {key: value.detach().cpu().clone() for key, value in policy.encoder.state_dict().items()},
            "decoder": {key: value.detach().cpu().clone() for key, value in policy.decoder.state_dict().items()},
        }

    best_heldout = float("inf")
    best_epoch = 0
    best_state = _interface_state()
    stopped_early = False
    for epoch in range(1, STAGE1_MAX_EPOCHS + 1):
        prediction = _predict(policy, observations_t)
        loss = torch.mean((prediction[:train_end] - targets_t[:train_end]) ** 2)
        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite Stage-1 loss for {condition} at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            evaluation = _predict(policy, observations_t)
            train_loss = float(torch.mean((evaluation[:train_end] - targets_t[:train_end]) ** 2))
            heldout_loss = float(torch.mean((evaluation[train_end:] - targets_t[train_end:]) ** 2))
        if not np.isfinite((train_loss, heldout_loss)).all():
            raise AssertionError(f"non-finite Stage-1 evaluation loss for {condition} at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": train_loss, "heldout_loss": heldout_loss})
        # Progress line so a long remote Stage 1 is observable from the log instead of
        # looking identical to a hung process for its whole duration.
        if epoch % 25 == 0 or epoch == 1:
            print(
                json.dumps({"stage1": condition, "epoch": epoch, "train_loss": train_loss, "heldout_loss": heldout_loss}, sort_keys=True),
                flush=True,
            )
        # Full-batch Adam on 140 samples is noisy enough that the last epoch is not
        # reliably the best one, so select the best held-out epoch rather than the last.
        if heldout_loss < best_heldout - STAGE1_MIN_DELTA:
            best_heldout, best_epoch, best_state = heldout_loss, epoch, _interface_state()
        elif epoch - best_epoch >= STAGE1_PATIENCE:
            stopped_early = True
            break
    policy.encoder.load_state_dict(best_state["encoder"])
    policy.decoder.load_state_dict(best_state["decoder"])
    _assert_frozen(policy, optimizer)
    after = SC.tensor_sha256(policy.core.W)
    if before != after:
        raise AssertionError(f"{condition} VNC graph changed during Stage 1")
    result = {
        "condition": condition,
        "seed": STAGE1_SEED,
        "optimizer": "Adam",
        "learning_rate": STAGE1_LEARNING_RATE,
        "max_epochs": STAGE1_MAX_EPOCHS,
        "patience": STAGE1_PATIENCE,
        "min_delta": STAGE1_MIN_DELTA,
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "selected_epoch": best_epoch,
        "selection_rule": "best held-out epoch, restored; identical rule for every arm",
        "train_steps": train_end,
        "heldout_steps": len(targets) - train_end,
        "initial_train_loss": initial_train,
        "initial_heldout_loss": initial_heldout,
        "final_train_loss": history[best_epoch - 1]["train_loss"] if best_epoch else history[-1]["train_loss"],
        "final_heldout_loss": best_heldout,
        "history": history,
        "interface_parameter_count": int(sum(parameter.numel() for parameter in policy.interface_parameters())),
        "core_tensor_sha256_pre": before,
        "core_tensor_sha256_post": after,
        "core_parameters": int(sum(parameter.numel() for parameter in policy.core.parameters())),
        "vnc_weights_trained": False,
        "wall_time_s": time.perf_counter() - started,
    }
    result["_interface_state"] = {
        "encoder": {key: value.detach().cpu().clone() for key, value in policy.encoder.state_dict().items()},
        "decoder": {key: value.detach().cpu().clone() for key, value in policy.decoder.state_dict().items()},
    }
    return result


def prepare(device: str = "auto") -> None:
    if PREREGISTERED.exists():
        raise FileExistsError(f"{PREREGISTERED} exists; preregistration is immutable")
    record = calibration(device)
    command_map = load_command_map(COMMAND_MAP_PATH)
    policies = [_policy(condition, record, STAGE1_SEED) for condition in CONDITIONS]
    parameter_counts = [sum(parameter.numel() for parameter in policy.interface_parameters()) for policy in policies]
    if len(set(parameter_counts)) != 1:
        raise AssertionError(f"interface parameter counts differ: {parameter_counts}")
    if parameter_counts[0] != 4_898:
        raise AssertionError(f"interface parameter count changed: {parameter_counts[0]}")
    record["interface_parameter_count"] = int(parameter_counts[0])
    record["command_map"] = {
        "path": str(COMMAND_MAP_PATH),
        "shape": list(command_map.shape),
        "latent_dim": int(command_map.shape[0]),
        "sha256": command_map_sha256(command_map),
        "channels": ["go", "steer", "family", "walkdn"],
        "trainable": False,
    }
    if record["command_map"]["sha256"] != COMMAND_MAP_SHA256:
        raise AssertionError("command map SHA-256 differs from registration")
    record["interface_architecture"] = {
        "class": "Linear encoder → frozen VNC rate core → Linear motor decoder",
        "observation": "HexapodEnv qpos concatenated with qvel",
        "latent_dim": 4,
        "encoder_output": "4 fixed latent commands expanded by anatomy-derived command map",
        "decoder_input": "all 260 VNC motor-output channels; common-mask sensitivity reported separately",
        "decoder_output": 18,
        "activation": "sigmoid encoder drive; linear decoder",
        "recurrent_state": "one frozen VNC rate per node",
    }
    prereg = {
        "experiment": "EMBODY_3WAY",
        "question": "Under identical training, does frozen measured VNC outperform amplitude-matched random and degree-preserving rewired cores through a 4-command trainable body interface?",
        "status": f"{TAG}-pre-registered-before-training",
        "device": record["device"],
        "latent_dim": 4,
        "command_map": record["command_map"],
        "pilot_clause": "5M environment steps per arm is a compute-limited pilot; if no arm reaches stable forward locomotion, label EMBODY-PILOT-INCONCLUSIVE and make no connectivity conclusion.",
        "conditions": list(CONDITIONS),
        "core_factory": "rung1_readout.condition_stepper(VNC_DATA, condition)",
        "graph_integrity": {
            "data": str(SC.VNC_DATA),
            "file_hash_function": "specificity_campaign.file_hashes(VNC_DATA)",
            "tensor_hash_function": "specificity_campaign.tensor_sha256(vnc.W)",
            "file_hashes": record["file_hashes_pre"],
            "tensor_hashes": record["graph_tensor_sha256"],
        },
        "calibration": record,
        "training": {
            "stage1": {
                "method": "supervised interface fit to an Option-B body-path teacher; core frozen",
                "steps": STAGE1_STEPS,
                "train_fraction": STAGE1_TRAIN_FRACTION,
                "heldout_fraction": STAGE1_HELDOUT_FRACTION,
                "optimizer": "Adam",
                "learning_rate": STAGE1_LEARNING_RATE,
                "max_epochs": STAGE1_MAX_EPOCHS,
                "patience": STAGE1_PATIENCE,
                "min_delta": STAGE1_MIN_DELTA,
                "stopping_rule": "train to held-out convergence, restore best held-out epoch; identical for every arm, realised epoch count recorded per arm",
                "amends": "v2 stopped every arm after 20 full-batch steps, which left the connectome at 0.6135 held-out against 0.0755 and 0.0721 for the controls and confounded Stage 2",
                "seed": STAGE1_SEED,
                "option_a_fallback": "not used",
            },
            "stage2": {
                "method": "batched on-policy PPO with the measured vector-environment throughput choice",
                "steps_per_arm_hard_cap": PPO_STEPS_PER_ARM,
                "batch_size": PPO_BATCH_SIZE,
                "device": record["device"],
                "gamma": PPO_GAMMA,
                "clip": PPO_CLIP,
                "entropy_coefficient": PPO_ENTROPY_COEF,
                "value_coefficient": PPO_VALUE_COEF,
                "learning_rate": PPO_LEARNING_RATE,
                "value_warmup_steps": PPO_VALUE_WARMUP_STEPS,
                "checkpoint_interval_steps": PPO_CHECKPOINT_INTERVAL,
                "selection": "last fixed checkpoint; no evaluation-distance selection",
                "evaluation_seeds": list(EVALUATION_SEEDS),
                "minimum_seeds": 5,
                "requested_seeds": 12,
                "owner_gate": "approval required before Stage 2 budget",
            },
        },
        "endpoints": {
            "primary": "mean forward displacement (m) per held-out-seed evaluation episode at fixed final checkpoint",
            "paired_contrasts": ["connectome_minus_random", "connectome_minus_rewired"],
            "test": "paired exact one-sided sign-flip",
            "multiple_comparison": "Holm across two contrasts",
            "interval": "paired bootstrap 95% CI",
            "secondary": ["sample efficiency", "final EMA return", "interface parameter count"],
        },
        "pre_registered_null": PRE_REGISTERED_NULL,
        "decision_labels": list(DECISION_LABELS),
        "motor_channel_sensitivity": {
            "primary": "all 260 motor channels; excluded channels remain in decoder with declared identity affine",
            "common_mask_sensitivity": "52 channels varying in all three calibration histories; report separately without retuning",
            "excluded_from_common_mask": int(len(record["excluded_indices"])),
        },
        "static_input_diagnostic": {"run": False, "reason": "optional arm deferred; CPU-only host and three required arms take precedence"},
        "no_final_claim_without_owner_approval": True,
        "config_hash": json_hash({"conditions": CONDITIONS, "calibration": record, "stage1_seed": STAGE1_SEED, "stage2_seeds": EVALUATION_SEEDS}),
    }
    SC.atomic_json(PREREGISTERED, _serial(prereg))
    print(json.dumps({"preregistered": str(PREREGISTERED), "device": record["device"], "latent_dim": 4, "interface_parameter_count": parameter_counts[0], "active_motor_channels": len(record["active_indices"]), "motor_channels": record["motor_output_dim"]}, sort_keys=True))


def run_stage1() -> None:
    if not PREREGISTERED.exists():
        raise FileNotFoundError(f"run --prepare first: {PREREGISTERED}")
    if STAGE1_JSON.exists():
        raise FileExistsError(f"{STAGE1_JSON} exists; Stage 1 artifact is immutable")
    prereg = json.loads(PREREGISTERED.read_text(encoding="utf-8"))
    record = calibration(prereg["device"])
    if prereg["graph_integrity"]["file_hashes"] != record["file_hashes_pre"]:
        raise AssertionError("preregistered graph file hashes no longer match")
    if prereg["graph_integrity"]["tensor_hashes"] != record["graph_tensor_sha256"]:
        raise AssertionError("preregistered graph tensor hashes no longer match")
    if prereg["calibration"]["active_indices"] != record["active_indices"]:
        raise AssertionError("preregistered amplitude mask changed")
    if prereg["config_hash"] != json_hash({"conditions": CONDITIONS, "calibration": prereg["calibration"], "stage1_seed": STAGE1_SEED, "stage2_seeds": EVALUATION_SEEDS}):
        raise AssertionError("preregistered config hash is invalid")

    observations, targets, teacher = build_teacher()
    reports = [_stage1_fit(condition, observations, targets, record) for condition in CONDITIONS]
    interface_states = {row["condition"]: row.pop("_interface_state") for row in reports}
    if len({row["interface_parameter_count"] for row in reports}) != 1:
        raise AssertionError("Stage-1 interface parameter counts differ")
    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    if file_hashes_post != record["file_hashes_pre"]:
        raise AssertionError("real graph files changed during Stage 1")
    if any(row["core_tensor_sha256_pre"] != row["core_tensor_sha256_post"] for row in reports):
        raise AssertionError("a core tensor hash changed during Stage 1")
    # Checkpoints are written from the interface states that were just trained.
    # They are never regenerated by a second fit: CUDA sparse matmul reduces in a
    # nondeterministic order, so a rerun does not reproduce the recorded losses bitwise.
    checkpoint_sha256 = {}
    for condition, state in interface_states.items():
        torch.save({"condition": condition, "seed": STAGE1_SEED, "interface_parameter_count": reports[0]["interface_parameter_count"], **state}, STAGE1_CHECKPOINTS[condition])
        checkpoint_sha256[condition] = SC.sha256(STAGE1_CHECKPOINTS[condition])
    result = {
        "experiment": "EMBODY_3WAY",
        "status": "stage1_complete_owner_gate_before_stage2",
        "decision_label": None,
        "preregistered": str(PREREGISTERED),
        "conditions": list(CONDITIONS),
        "teacher": teacher,
        "stage1": {"reports": reports, "target_sha256": array_sha256(targets), "observation_sha256": array_sha256(observations), "checkpoint_sha256": checkpoint_sha256},
        "graph_tensor_sha256_pre": record["graph_tensor_sha256"],
        "graph_tensor_sha256_post": {condition: row["core_tensor_sha256_post"] for condition, row in ((r["condition"], r) for r in reports)},
        "file_hashes_pre": record["file_hashes_pre"],
        "file_hashes_post": file_hashes_post,
        "vnc_weights_trained": False,
        "core_gradients_seen": False,
        "stage2": {"status": "not_started_owner_approval_required"},
        "honest_note": "Stage 1 is an interface-fit readout only; no final claim or decision label is made before Stage 2 and owner approval.",
    }
    SC.atomic_json(STAGE1_JSON, _serial(result))
    fields = ("condition", "seed", "interface_parameter_count", "initial_train_loss", "final_train_loss", "initial_heldout_loss", "final_heldout_loss", "core_tensor_sha256_pre", "core_tensor_sha256_post", "wall_time_s")
    with STAGE1_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in reports)
    print(json.dumps({"stage1": str(STAGE1_JSON), "reports": reports, "teacher": teacher}, sort_keys=True))


def export_stage1_checkpoints() -> None:
    """Verify the Stage-1 checkpoints written by run_stage1; never retrain."""
    if not PREREGISTERED.exists() or not STAGE1_JSON.exists():
        raise FileNotFoundError("prepare and run Stage 1 before verifying checkpoints")
    prereg = json.loads(PREREGISTERED.read_text(encoding="utf-8"))
    existing = json.loads(STAGE1_JSON.read_text(encoding="utf-8"))
    if SC.file_hashes(SC.VNC_DATA) != prereg["graph_integrity"]["file_hashes"]:
        raise AssertionError("graph files changed before Stage-1 checkpoint verification")
    recorded = existing["stage1"].get("checkpoint_sha256")
    if not recorded:
        raise AssertionError("Stage-1 JSON predates in-run checkpoint saving; rerun --stage1")
    expected_count = {row["condition"]: row["interface_parameter_count"] for row in existing["stage1"]["reports"]}
    verified = {}
    for condition in CONDITIONS:
        path = STAGE1_CHECKPOINTS[condition]
        if not path.exists():
            raise FileNotFoundError(path)
        digest = SC.sha256(path)
        if digest != recorded.get(condition):
            raise AssertionError(f"Stage-1 checkpoint for {condition} does not match the hash recorded at training time")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["condition"] != condition or saved["seed"] != STAGE1_SEED:
            raise AssertionError(f"Stage-1 checkpoint metadata mismatch for {condition}")
        if saved["interface_parameter_count"] != expected_count[condition]:
            raise AssertionError(f"Stage-1 checkpoint parameter count mismatch for {condition}")
        loaded = int(sum(tensor.numel() for group in ("encoder", "decoder") for tensor in saved[group].values()))
        if loaded != expected_count[condition]:
            raise AssertionError(f"Stage-1 checkpoint tensor count mismatch for {condition}")
        verified[condition] = {"path": str(path), "sha256": digest, "interface_parameter_count": loaded}
    print(json.dumps({"verified": verified, "verified_against": str(STAGE1_JSON)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--stage1", action="store_true")
    parser.add_argument("--export-stage1", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if sum((args.prepare, args.stage1, args.export_stage1)) != 1:
        parser.error("choose exactly one of --prepare, --stage1, or --export-stage1")
    if args.prepare:
        prepare(args.device)
    elif args.stage1:
        run_stage1()
    else:
        export_stage1_checkpoints()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
