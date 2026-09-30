"""Measure batched frozen-VNC throughput without starting Stage 2."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
DEFAULT_JSON = RESULTS / "THROUGHPUT_REPORT.json"
DEFAULT_MD = RESULTS / "THROUGHPUT_REPORT.md"
BATCH_SIZES = (1, 8, 32, 128, 512)
CONDITIONS = ("connectome", "random", "rewired")
PLATEAU_RATIO = 0.90

sys.path.insert(0, str(HERE))

from embody_env import VectorHexapodEnv  # noqa: E402
from embody_policy import (  # noqa: E402
    EmbodyPolicy,
    _assert_frozen,
    load_command_map,
    resolve_device,
    self_check,
    vnc_graph_sha256,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _policy(device: str) -> EmbodyPolicy:
    torch.manual_seed(20260831)
    policy = EmbodyPolicy(
        obs_dim=VectorHexapodEnv(1, max_steps=1).obs_dim,
        data=RESULTS,
        weights_name="all_legs_weight_csr.pkl",
        index_name="all_legs_io_index.npz",
        command_map=load_command_map(),
        device=device,
    )
    if policy.core.n != 11_000 or policy.core.W._nnz() != 485_074:
        raise AssertionError(f"unexpected benchmark core: n={policy.core.n} nnz={policy.core.W._nnz()}")
    if policy.core.W.device.type not in {"cpu", "cuda"}:
        raise AssertionError(f"unsupported benchmark device {policy.core.W.device}")
    optimizer = torch.optim.Adam(policy.interface_parameters(), lr=1e-3)
    _assert_frozen(policy, optimizer)
    return policy


def sparse_mm_benchmark(policy: EmbodyPolicy, batch_size: int, repeats: int) -> dict[str, float | int]:
    device = policy.core.W.device
    generator = torch.Generator(device=device).manual_seed(20260831 + batch_size)
    value = torch.randn((policy.core.n, batch_size), generator=generator, device=device)
    with torch.no_grad():
        for _ in range(5):
            output = torch.sparse.mm(policy.core.W, value)
        _sync(device)
        start = time.perf_counter()
        for _ in range(repeats):
            output = torch.sparse.mm(policy.core.W, value)
        _sync(device)
        elapsed = time.perf_counter() - start
    checksum = float(output.sum().detach().cpu())
    total_ms = 1000.0 * elapsed / repeats
    return {
        "batch_size": batch_size,
        "repeats": repeats,
        "total_ms": total_ms,
        "per_environment_ms": total_ms / batch_size,
        "checksum": checksum,
    }


def reset_contract_check(policy: EmbodyPolicy) -> dict[str, object]:
    vector = VectorHexapodEnv(2, max_steps=400)
    vector.envs[0].max_steps = 1
    observations = vector.reset(seeds=[20260830, 20260831])
    node_state = policy.initial_state_batch(2)
    node_state[:, 0] = 3.0
    node_state[:, 1] = 4.0
    nonterminal_before = node_state[:, 1].clone()
    actions = np.stack([env.stance for env in vector.envs])
    next_observations, rewards, done = vector.step(actions)
    assert next_observations.shape == (2, vector.obs_dim)
    assert rewards.shape == (2,) and np.isfinite(rewards).all()
    assert done.tolist() == [True, False], f"reset probe did not produce selective termination: {done}"
    assert vector.last_reset_mask.tolist() == done.tolist()
    assert not vector.envs[0]._done
    assert np.isfinite(next_observations).all()
    node_state = policy.reset_state_batch(node_state, done)
    reset_max_abs = float(node_state[:, 0].abs().max().detach().cpu())
    assert torch.equal(node_state[:, 1], nonterminal_before)
    assert torch.count_nonzero(node_state[:, 0]).item() == 0
    return {
        "done_mask": done.tolist(),
        "auto_reset_done_env": not vector.envs[0]._done,
        "reset_node_state_max_abs": reset_max_abs,
        "nonterminal_state_unchanged": True,
        "pass": True,
    }


def end_to_end_benchmark(
    policy: EmbodyPolicy,
    batch_size: int,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, float | int]:
    vector = VectorHexapodEnv(batch_size, max_steps=400)
    observations = vector.reset(seeds=(20260830 + np.arange(batch_size)).tolist())
    node_state = policy.initial_state_batch(batch_size)

    def one_step() -> None:
        nonlocal observations, node_state
        observation_tensor = torch.from_numpy(observations)
        with torch.no_grad():
            actions, node_state = policy.step_batch(observation_tensor, node_state)
        observations, _, done = vector.step(actions.detach().cpu().numpy())
        with torch.no_grad():
            node_state = policy.reset_state_batch(node_state, done)

    for _ in range(warmup_steps):
        one_step()
    start = time.perf_counter()
    for _ in range(timed_steps):
        one_step()
    elapsed = time.perf_counter() - start
    return {
        "batch_size": batch_size,
        "warmup_batch_steps": warmup_steps,
        "timed_batch_steps": timed_steps,
        "wall_time_s": elapsed,
        "batch_steps_per_s": timed_steps / elapsed,
        "environment_steps_per_s": batch_size * timed_steps / elapsed,
    }


def _load_protocol() -> tuple[list[str], int]:
    preregistered = json.loads((RESULTS / "embody_3way_preregistered.json").read_text(encoding="utf-8"))
    stage2 = preregistered["training"]["stage2"]
    conditions = preregistered["conditions"]
    budget = int(stage2["steps_per_arm_hard_cap"])
    if tuple(conditions) != CONDITIONS or budget != 5_000_000:
        raise AssertionError(f"registered protocol changed: conditions={conditions}, budget={budget}")
    return conditions, budget


def _hours(steps: int, steps_per_second: float) -> float:
    return steps / steps_per_second / 3600.0


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    sparse_rows = report["sparse_mm"]
    e2e_rows = report["end_to_end"]
    chosen = report["chosen_batch_size"]
    lines = [
        "# Frozen VNC throughput report",
        "",
        "CPU-only throughput plumbing. No Stage 2 training started.",
        "",
        f"- Device: `{report['device']}`; PyTorch `{report['torch_version']}`; torch threads: `{report['torch_threads']}`.",
        f"- Core: `n={report['core']['n']}`, `nnz={report['core']['nnz']}`, density `{report['core']['density']:.5f}`; dense fp32 estimate `{report['core']['dense_fp32_mb']:.1f} MB`.",
        f"- Chosen batch: **B={chosen}**; smallest measured end-to-end batch within 90% of best throughput (`{report['plateau']['threshold_steps_per_s']:.2f}` env steps/s; best `{report['plateau']['best_steps_per_s']:.2f}`).",
        "",
        "## CPU sparse matmul",
        "",
        "| B | total ms | per-env ms |",
        "|---:|---:|---:|",
    ]
    lines.extend(f"| {row['batch_size']} | {row['total_ms']:.3f} | {row['per_environment_ms']:.4f} |" for row in sparse_rows)
    lines.extend([
        "",
        "## End-to-end vector environment + batched policy",
        "",
        "| B | batch steps/s | environment steps/s | wall s |",
        "|---:|---:|---:|---:|",
    ])
    lines.extend(
        f"| {row['batch_size']} | {row['batch_steps_per_s']:.2f} | {row['environment_steps_per_s']:.2f} | {row['wall_time_s']:.3f} |"
        for row in e2e_rows
    )
    lines.extend([
        "",
        "## Required checks",
        "",
        f"- Batched-vs-single core max abs difference: `{report['self_check']['batched_core_max_abs_diff']:.3g}`; backward max abs difference: `{report['self_check']['batched_backward_max_abs_diff']:.3g}`; tolerance: `{report['self_check']['tolerance']:.3g}`.",
        f"- Vector auto-reset probe: done mask `{report['reset_check']['done_mask']}`; reset column max abs `{report['reset_check']['reset_node_state_max_abs']:.1f}`; nonterminal column unchanged: `{report['reset_check']['nonterminal_state_unchanged']}`.",
        f"- Frozen invariants: W hash pre/post equal `{report['invariants']['hash_equal_pre_post']}`; `W.grad is None` `{report['invariants']['W_grad_is_none']}`; W parameter `{report['invariants']['W_is_parameter']}`; command map requires grad `{report['invariants']['command_map_requires_grad']}`; all pass `{report['invariants']['all_pass']}`.",
        f"- Three arms: `{', '.join(report['arms']['conditions'])}`; same B/device/budget assertion: `{report['arms']['same_batch_device_budget']}` (`{chosen}`, `{report['device']}`, `{report['arms']['budget_steps_per_arm']:,}` steps/arm).",
        "- MPS: refused by `FrozenVNC`; no dense fallback. CUDA not benchmarked on this host.",
        "",
        "## Projected wall clock per arm",
        "",
        "Using selected B and measured end-to-end throughput; projections are arithmetic, not runs.",
        "",
        "| budget | hours/arm |",
        "|---:|---:|",
        f"| 5,000,000 | {report['projections']['5M_steps_per_arm_hours']:.3f} h |",
        f"| 50,000,000 | {report['projections']['50M_steps_per_arm_hours']:.3f} h |",
        "",
        "Stage 2 remains owner-gated and unrun.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    resolved = resolve_device(args.device)
    device = torch.device(resolved)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("throughput benchmark permits CPU or CUDA only; MPS is refused")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    conditions, budget = _load_protocol()

    checks = self_check(str(device))
    policy = _policy(str(device))
    reset_check = reset_contract_check(policy)
    graph_before = vnc_graph_sha256(policy.core)
    sparse_rows = [sparse_mm_benchmark(policy, batch_size, args.matmul_repeats) for batch_size in BATCH_SIZES]
    e2e_rows = [
        end_to_end_benchmark(policy, batch_size, args.warmup_steps, args.timed_steps)
        for batch_size in BATCH_SIZES
    ]
    graph_after = vnc_graph_sha256(policy.core)

    best_steps_per_s = max(row["environment_steps_per_s"] for row in e2e_rows)
    plateau_threshold = PLATEAU_RATIO * best_steps_per_s
    plateau_rows = [row for row in e2e_rows if row["environment_steps_per_s"] >= plateau_threshold]
    chosen = min(row["batch_size"] for row in plateau_rows)
    selected = next(row for row in e2e_rows if row["batch_size"] == chosen)
    invariants = {
        "hash_equal_pre_post": graph_before == graph_after == checks["core_sha256"],
        "W_grad_is_none": policy.core.W.grad is None,
        "W_is_parameter": isinstance(policy.core.W, torch.nn.Parameter),
        "command_map_requires_grad": policy.command_map.requires_grad,
    }
    invariants["all_pass"] = bool(
        invariants["hash_equal_pre_post"]
        and invariants["W_grad_is_none"]
        and not invariants["W_is_parameter"]
        and not invariants["command_map_requires_grad"]
    )
    arms = {
        "conditions": conditions,
        "batch_size": chosen,
        "device": str(device),
        "budget_steps_per_arm": budget,
        "configs": [{"condition": condition, "batch_size": chosen, "device": str(device), "budget_steps_per_arm": budget} for condition in conditions],
        "same_batch_device_budget": len({(chosen, str(device), budget) for _ in conditions}) == 1 and tuple(conditions) == CONDITIONS,
    }
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "stage2_started": False,
        "core": {
            "n": policy.core.n,
            "nnz": policy.core.W._nnz(),
            "density": policy.core.W._nnz() / (policy.core.n * policy.core.n),
            "dense_fp32_mb": policy.core.n * policy.core.n * 4 / 1_000_000,
        },
        "benchmark_config": {
            "batch_sizes": list(BATCH_SIZES),
            "warmup_batch_steps": args.warmup_steps,
            "timed_batch_steps": args.timed_steps,
            "matmul_repeats": args.matmul_repeats,
        },
        "sparse_mm": sparse_rows,
        "end_to_end": e2e_rows,
        "plateau": {
            "rule": "smallest measured B with environment steps/s >= 90% of best",
            "ratio": PLATEAU_RATIO,
            "best_batch_size": max(e2e_rows, key=lambda row: row["environment_steps_per_s"])["batch_size"],
            "best_steps_per_s": best_steps_per_s,
            "threshold_steps_per_s": plateau_threshold,
            "plateau_batch_sizes": [row["batch_size"] for row in plateau_rows],
        },
        "chosen_batch_size": chosen,
        "selected_throughput_steps_per_s": selected["environment_steps_per_s"],
        "self_check": {
            "batched_core_max_abs_diff": checks["batched_equivalence_max_abs_diff"],
            "batched_backward_max_abs_diff": checks["batched_backward_max_abs_diff"],
            "tolerance": checks["batched_equivalence_tolerance"],
        },
        "reset_check": reset_check,
        "invariants": invariants,
        "arms": arms,
        "projections": {
            "5M_steps_per_arm_hours": _hours(5_000_000, selected["environment_steps_per_s"]),
            "50M_steps_per_arm_hours": _hours(50_000_000, selected["environment_steps_per_s"]),
        },
    }
    if not invariants["all_pass"] or not reset_check["pass"] or not arms["same_batch_device_budget"]:
        raise AssertionError("throughput invariant failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument("--matmul-repeats", type=int, default=20)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    if min(args.warmup_steps, args.timed_steps, args.matmul_repeats) < 1:
        parser.error("benchmark counts must be positive")
    report = run(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(args.markdown, report)
    print(json.dumps({"json": str(args.json), "markdown": str(args.markdown), "chosen_batch_size": report["chosen_batch_size"], "steps_per_s": report["selected_throughput_steps_per_s"]}, sort_keys=True))


if __name__ == "__main__":
    main()
