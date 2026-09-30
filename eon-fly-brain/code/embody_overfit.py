"""CPU interface-capacity check; not the imitation dataset or real training."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
from embody_env import HexapodEnv  # noqa: E402
from embody_policy import EmbodyPolicy, load_command_map, _assert_frozen, vnc_graph_sha256  # noqa: E402
from hexapod_gait import HexapodGait  # noqa: E402


TARGET_STEPS = 8
OPTIMIZER_STEPS = 400


def fixed_forward_slice() -> tuple[np.ndarray, np.ndarray]:
    """Collect real body states paired with the existing forward target template."""
    env = HexapodEnv(max_steps=TARGET_STEPS + 1)
    observation = env.reset()
    gait = HexapodGait(stance=env.stance.reshape(6, 3))
    observations, targets = [], []
    for _ in range(TARGET_STEPS):
        target = gait.targets(0.8, 0.0, 1.0 / env.control_hz).astype(np.float32)
        observations.append(observation)
        observation, _, done = env.step(target)
        assert not done
        targets.append(target)
    return np.asarray(observations, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def predict_sequence(policy: EmbodyPolicy, observations: torch.Tensor) -> torch.Tensor:
    state = policy.initial_state()
    predictions = []
    for observation in observations:
        action, state = policy.step(observation, state)
        predictions.append(action)
    return torch.stack(predictions)


def run_overfit(command_map=None) -> dict[str, float | int | str]:
    torch.manual_seed(20260830)
    command_map = load_command_map() if command_map is None else command_map
    observations_np, targets_np = fixed_forward_slice()
    observations = torch.from_numpy(observations_np)
    targets = torch.from_numpy(targets_np)
    policy = EmbodyPolicy(obs_dim=observations.shape[1], substeps=1, command_map=command_map)
    optimizer = torch.optim.Adam(policy.interface_parameters(), lr=0.02)
    _assert_frozen(policy, optimizer)
    before = vnc_graph_sha256(policy.core)

    with torch.no_grad():
        initial_loss = float(torch.mean((predict_sequence(policy, observations) - targets) ** 2))
    start = time.perf_counter()
    for _ in range(OPTIMIZER_STEPS):
        prediction = predict_sequence(policy, observations)
        loss = torch.mean((prediction - targets) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    wall_time = time.perf_counter() - start
    with torch.no_grad():
        final_loss = float(torch.mean((predict_sequence(policy, observations) - targets) ** 2))
    assert final_loss < 1e-3, f"overfit loss stayed high: initial={initial_loss:.6g} final={final_loss:.6g}"
    assert before == vnc_graph_sha256(policy.core)
    return {
        "target_steps": TARGET_STEPS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_time_s": wall_time,
        "core_sha256": before,
    }


if __name__ == "__main__":
    result = run_overfit(load_command_map())
    print(
        "OVERFIT TEST: PASS "
        f"(steps={result['optimizer_steps']} target_steps={result['target_steps']} "
        f"loss={result['initial_loss']:.6g}->{result['final_loss']:.6g} "
        f"wall_time_s={result['wall_time_s']:.3f} core_sha256={result['core_sha256']})"
    )
