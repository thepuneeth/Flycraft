"""R1: fit the sole learned object, a closed-form linear ridge readout."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE.parent))
from hexapod_gait import HexapodGait, home_stance  # noqa: E402


TRAIN_FRACTION = 0.7
SAMPLE_DT = 0.02
TEMPLATE_FORWARD = 0.8
TEMPLATE_TURN = 0.0
TEMPLATE_STANCE = home_stance().reshape(18)
LAMBDAS = np.logspace(-6, 6, 13)
RANDOM_FEATURE_UNITS = 64
RANDOM_FEATURE_SEED = 20260829


def template(steps: int) -> np.ndarray:
    gait = HexapodGait()
    return np.stack([gait.targets(TEMPLATE_FORWARD, TEMPLATE_TURN, SAMPLE_DT) for _ in range(steps)])


def r2_columns(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    residual = np.sum((truth - pred) ** 2, axis=0)
    total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    return 1.0 - residual / np.maximum(total, np.finfo(float).eps)


def fit_design(x: np.ndarray, q: np.ndarray) -> dict:
    split = int(len(x) * TRAIN_FRACTION)
    # Fixed stance is a coordinate convention, not a learned intercept.
    y = (q - TEMPLATE_STANCE).astype(np.float64)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    kernel = (x_train @ x_train.T).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_y = eigenvectors.T @ y_train
    cross_kernel = (x_test @ x_train.T).astype(np.float64)
    path = []
    solutions = []
    for ridge in LAMBDAS:
        alpha = eigenvectors @ (projected_y / (eigenvalues[:, None] + ridge))
        pred_train = kernel @ alpha + TEMPLATE_STANCE
        pred_test = cross_kernel @ alpha + TEMPLATE_STANCE
        train_r2 = r2_columns(q[:split], pred_train)
        test_r2 = r2_columns(q[split:], pred_test)
        path.append({
            "lambda": float(ridge),
            "mean_train_r2": float(train_r2.mean()),
            "mean_heldout_r2": float(test_r2.mean()),
        })
        solutions.append((alpha, train_r2, test_r2))
    chosen = int(np.argmax([row["mean_heldout_r2"] for row in path]))
    alpha, train_r2, test_r2 = solutions[chosen]
    weights = (x_train.T @ alpha).astype(np.float32)
    return {
        "weights": weights,
        "chosen_index": chosen,
        "chosen_lambda": float(LAMBDAS[chosen]),
        "path": path,
        "train_r2": train_r2,
        "heldout_r2": test_r2,
        "split": split,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "projected_y": projected_y,
    }


def random_features(states: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, dict]:
    active = np.flatnonzero(scale != 1.0)
    rng = np.random.default_rng(RANDOM_FEATURE_SEED)
    projection = rng.normal(0.0, 1.0 / np.sqrt(len(active)), (len(active), RANDOM_FEATURE_UNITS)).astype(np.float32)
    hidden = np.tanh(((states[:, active] - mean[active]) / scale[active]) @ projection).astype(np.float32)
    split = int(len(hidden) * TRAIN_FRACTION)
    hidden_mean = hidden[:split].mean(axis=0)
    hidden_scale = hidden[:split].std(axis=0)
    hidden_scale[hidden_scale < 1e-7] = 1.0
    return ((hidden - hidden_mean) / hidden_scale).astype(np.float32), {
        "active_indices": active,
        "projection": projection,
        "feature_mean": hidden_mean,
        "feature_scale": hidden_scale,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, default=RESULTS / "reservoir_states.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "reservoir_readout_W.npz")
    ap.add_argument("--nonlinear", action="store_true", help="single allowed fixed 64-unit tanh random-feature escalation")
    a = ap.parse_args()
    with np.load(a.states) as source:
        states = source["states"]
        mean = source["train_mean"]
        scale = source["train_scale"]
        body_ids = source["node_body_ids"]
    q = template(len(states))
    feature_data = None
    if a.nonlinear:
        design, feature_data = random_features(states, mean, scale)
    else:
        design = ((states - mean) / scale).astype(np.float32)
    result = fit_design(design, q)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    template_out = a.out.with_name("reservoir_template_q.npz")
    np.savez_compressed(template_out, q=q.astype(np.float32), sample_dt_s=SAMPLE_DT)
    artifact = {
        "W_out": result["weights"],
        "train_mean": mean,
        "train_scale": scale,
        "node_body_ids": body_ids,
        "stance": TEMPLATE_STANCE.astype(np.float32),
        "chosen_lambda": result["chosen_lambda"],
        "readout_kind": "random_features" if a.nonlinear else "linear",
    }
    if feature_data:
        artifact.update(feature_data)
    np.savez_compressed(a.out, **artifact)
    suffix = "_nonlinear" if a.nonlinear else ""
    path_out = a.out.with_name(f"reservoir_lambda_path{suffix}.csv")
    with path_out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("lambda", "mean_train_r2", "mean_heldout_r2"))
        writer.writeheader()
        writer.writerows(result["path"])
    report = {
        "gate": "R1",
        "only_learned_object": "W_out",
        "readout_kind": "random_features" if a.nonlinear else "linear",
        "nonlinear_escalation": ({
            "hidden_units": RANDOM_FEATURE_UNITS,
            "activation": "tanh",
            "input_projection": "fixed Gaussian random features",
            "seed": RANDOM_FEATURE_SEED,
            "projection_learned": False,
            "label": "linear capacity insufficient",
        } if a.nonlinear else None),
        "method": "exact closed-form dual ridge; algebraically identical to primal ridge",
        "template": {
            "source": "Path-1 HexapodGait defaults",
            "forward": TEMPLATE_FORWARD,
            "turn": TEMPLATE_TURN,
            "sample_dt_s": SAMPLE_DT,
            "stance": TEMPLATE_STANCE.tolist(),
            "declared_before_fit": True,
            "performance_tuned": False,
        },
        "train_fraction": TRAIN_FRACTION,
        "train_steps": result["split"],
        "heldout_steps": len(states) - result["split"],
        "lambda_grid": LAMBDAS.tolist(),
        "chosen_lambda": result["chosen_lambda"],
        "selection_metric": "mean held-out time-window R2 over 18 joints",
        "per_joint_train_r2": result["train_r2"].tolist(),
        "per_joint_heldout_r2": result["heldout_r2"].tolist(),
        "mean_train_r2": float(result["train_r2"].mean()),
        "mean_heldout_r2": float(result["heldout_r2"].mean()),
        "frozen_vnc": True,
        "reward_or_task_loss": False,
        "lambda_path": str(path_out),
        "template_artifact": str(template_out),
        "readout_artifact": str(a.out),
    }
    report_out = a.out.with_name(f"reservoir_readout_report{suffix}.json")
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
