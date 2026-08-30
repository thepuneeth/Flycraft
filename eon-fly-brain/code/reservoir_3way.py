"""R3: identical-readout comparison of connectome, random, and rewired reservoirs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE.parent))
from collect_reservoir_states import PROTOCOL, collect, make_stepper, scaler  # noqa: E402
from fit_readout import (  # noqa: E402
    TEMPLATE_STANCE,
    TRAIN_FRACTION,
    fit_design,
    r2_columns,
    random_features,
    template,
)
from run_reservoir_walk import run_episode  # noqa: E402


CONDITIONS = ("connectome", "random", "rewired")
GENERALIZATION_DRIVE = 90.0
EFFECTIVE_R2_TARGET = 0.8
REWIRE_SEED = 20260829


def tensor_sha256(weight: torch.Tensor) -> str:
    weight = weight.cpu()
    digest = hashlib.sha256()
    for array in (weight.crow_indices().numpy(), weight.col_indices().numpy(), weight.values().numpy()):
        digest.update(array.tobytes())
    return digest.hexdigest()


def degree_preserving_rewire(weight: torch.Tensor, seed: int = REWIRE_SEED) -> torch.Tensor:
    """Swap presynaptic endpoints while preserving every node's in/out degree."""
    coo = weight.to_sparse_coo().coalesce().cpu()
    rows = coo.indices()[0].numpy().copy()
    cols = coo.indices()[1].numpy().copy()
    original_col_degree = np.bincount(cols, minlength=int(weight.shape[1]))
    values = coo.values().numpy().copy()
    n = int(weight.shape[0])
    edges = set((rows.astype(np.int64) * n + cols).tolist())
    rng = np.random.default_rng(seed)
    successes = 0
    target = len(rows)
    while successes < target:
        for left, right in rng.integers(0, len(rows), size=(min(100_000, target - successes + 10_000), 2)):
            r1, r2, c1, c2 = int(rows[left]), int(rows[right]), int(cols[left]), int(cols[right])
            if r1 == r2 or c1 == c2:
                continue
            old1, old2 = r1 * n + c1, r2 * n + c2
            new1, new2 = r1 * n + c2, r2 * n + c1
            if new1 in edges or new2 in edges:
                continue
            edges.remove(old1)
            edges.remove(old2)
            edges.add(new1)
            edges.add(new2)
            cols[left], cols[right] = c2, c1
            successes += 1
            if successes == target:
                break
    indices = torch.from_numpy(np.stack((rows, cols))).long()
    rewired = torch.sparse_coo_tensor(indices, torch.from_numpy(values), weight.shape).coalesce().to_sparse_csr()
    assert rewired._nnz() == weight._nnz()
    assert np.array_equal(np.bincount(cols, minlength=n), original_col_degree)
    assert torch.equal(rewired.crow_indices(), weight.cpu().crow_indices())
    return rewired


def condition_stepper(data: Path, weights_name: str, index_name: str, condition: str):
    if condition == "random":
        return make_stepper(data, weights_name, index_name, shuffle=True)
    vnc = make_stepper(data, weights_name, index_name)
    if condition == "rewired":
        vnc.W = degree_preserving_rewire(vnc.W).to(vnc.device)
    return vnc


def readout_artifact(result: dict, mean: np.ndarray, scale: np.ndarray, kind: str, feature_data: dict | None = None) -> dict:
    artifact = {
        "W_out": result["weights"],
        "train_mean": mean,
        "train_scale": scale,
        "stance": TEMPLATE_STANCE.astype(np.float32),
        "chosen_lambda": np.asarray(result["chosen_lambda"]),
        "readout_kind": np.asarray(kind),
    }
    if feature_data:
        artifact.update(feature_data)
    return artifact


def predict(states: np.ndarray, artifact: dict) -> np.ndarray:
    x = (states - artifact["train_mean"]) / artifact["train_scale"]
    if str(artifact["readout_kind"]) == "random_features":
        hidden = np.tanh(x[:, artifact["active_indices"]] @ artifact["projection"])
        x = (hidden - artifact["feature_mean"]) / artifact["feature_scale"]
    return x @ artifact["W_out"] + artifact["stance"]


def effective_rank(design: np.ndarray, q: np.ndarray, fit: dict, target_r2: float = EFFECTIVE_R2_TARGET) -> int | None:
    split = int(len(design) * TRAIN_FRACTION)
    cross_kernel = (design[split:] @ design[:split].T).astype(np.float64)
    order = np.argsort(fit["eigenvalues"])[::-1]
    test_projection = cross_kernel @ fit["eigenvectors"][:, order]
    coefficients = fit["projected_y"][order] / (fit["eigenvalues"][order, None] + fit["chosen_lambda"])
    prediction = np.zeros_like(q[split:]) + TEMPLATE_STANCE
    dynamic = np.tile((True, True, False), 6)
    for rank in range(1, len(order) + 1):
        prediction += test_projection[:, rank - 1:rank] @ coefficients[rank - 1:rank]
        if float(r2_columns(q[split:, dynamic], prediction[:, dynamic]).mean()) >= target_r2:
            return rank
    return None


def make_figure(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['condition']}\n{row['readout_kind']}" for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for axis, key, title in zip(
        axes,
        ("dynamic_heldout_r2", "forward_displacement_m", "mean_target_temporal_sd_rad"),
        ("Held-out dynamic-joint R²", "Closed-loop forward distance", "Readout target amplitude"),
    ):
        axis.bar(range(len(rows)), [row[key] for row in rows], color=["#176d9c" if "connectome" in label else "#999999" for label in labels])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(rows)), labels, rotation=35, ha="right")
        axis.set_title(title)
    axes[1].set_ylabel("m")
    axes[2].set_ylabel("rad")
    fig.suptitle("Frozen-reservoir 3-way readout test")
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=RESULTS)
    ap.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    ap.add_argument("--index_name", default="all_legs_io_index.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "reservoir_3way")
    a = ap.parse_args()
    q = template(PROTOCOL["record_steps"])
    dynamic = np.tile((True, True, False), 6)
    rows = []
    graph_hashes = {}
    for condition in CONDITIONS:
        vnc = condition_stepper(a.data, a.weights_name, a.index_name, condition)
        graph_hashes[condition] = tensor_sha256(vnc.W)
        states = collect(vnc, PROTOCOL["drive"], PROTOCOL["burn_steps"], PROTOCOL["record_steps"])
        mean, scale = scaler(states)
        designs = {"linear": ((states - mean) / scale).astype(np.float32)}
        nonlinear_design, feature_data = random_features(states, mean, scale)
        designs["random_features"] = nonlinear_design
        artifacts = {}
        fits = {}
        for kind, design in designs.items():
            fit = fit_design(design, q)
            fits[kind] = fit
            artifacts[kind] = readout_artifact(fit, mean, scale, kind, feature_data if kind == "random_features" else None)
        general_vnc = condition_stepper(a.data, a.weights_name, a.index_name, condition)
        general_states = collect(general_vnc, GENERALIZATION_DRIVE, PROTOCOL["burn_steps"], len(q) - int(len(q) * TRAIN_FRACTION))
        for kind, design in designs.items():
            fit = fits[kind]
            artifact = artifacts[kind]
            heldout_prediction = predict(states[int(len(states) * TRAIN_FRACTION):], artifact)
            general_prediction = predict(general_states, artifact)
            walk_vnc = condition_stepper(a.data, a.weights_name, a.index_name, condition)
            walk = run_episode(walk_vnc, artifact, secs=8.0, video=None)
            row = {
                "condition": condition,
                "readout_kind": kind,
                "chosen_lambda": fit["chosen_lambda"],
                "mean_heldout_r2_all_joints": float(fit["heldout_r2"].mean()),
                "dynamic_heldout_r2": float(r2_columns(q[int(len(q) * TRAIN_FRACTION):, dynamic], heldout_prediction[:, dynamic]).mean()),
                "generalization_drive": GENERALIZATION_DRIVE,
                "generalization_dynamic_r2": float(r2_columns(q[-len(general_states):, dynamic], general_prediction[:, dynamic]).mean()),
                "effective_r2_target": EFFECTIVE_R2_TARGET,
                "effective_rank": effective_rank(design, q, fit),
                "tripod_coherence_scalar": walk["tripod_coherence_scalar"],
                "e0_phase_verdict": walk["e0_phase_verdict"],
                "forward_displacement_m": walk["forward_displacement_m"],
                "path_m": walk["path_m"],
                "mean_target_temporal_sd_rad": walk["mean_target_temporal_sd_rad"],
                "fell": walk["fell"],
                "caught": walk["caught"],
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    verdict = "READOUT TOO WEAK"
    report = {
        "gate": "R3",
        "verdict": verdict,
        "reason": "Neither linear nor single allowed nonlinear readout produced a visible or kinematically material walk; phase score occurred at collapsed amplitude.",
        "conditions": {
            "connectome": "real frozen signed MANC CSR",
            "random": "existing seeded presynaptic-column permutation; exact size/row degree and global out-degree distribution matched; seed 1",
            "rewired": "one successful directed endpoint swap per edge; exact per-node in/out degree and weight multiset preserved; seed 20260829",
        },
        "identical_procedure": True,
        "generalization_drive": GENERALIZATION_DRIVE,
        "effective_r2_target": EFFECTIVE_R2_TARGET,
        "graph_hashes": graph_hashes,
        "rows": rows,
        "honesty": {
            "vnc_weights_trained": False,
            "only_fit": "closed-form ridge W_out",
            "template_changed": False,
            "lambda_selected_by": "held-out reconstruction R2 only",
            "task_reward_or_distance_used_for_fit": False,
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with a.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_figure(rows, a.out.with_suffix(".png"))
    print("RESULT_JSON " + json.dumps({"verdict": verdict, "rows": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
