"""R0: collect full frozen-MANC reservoir activations under tonic drive."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE.parent))
from vnc_stepper import VNCStepper  # noqa: E402


PROTOCOL = {
    "model": "paper",
    "tau": 0.02,
    "dt": 0.005,
    "substeps": 4,
    "sample_dt_s": 0.02,
    "gW": 0.03,
    "drive": 100.0,
    "dnb08_gain": 1.0,
    "threshold": 7.5,
    "activation_gain": 1.0,
    "fr_cap": 200.0,
    "heterogeneity": True,
    "seed": 1,
    "burn_steps": 3000,
    "record_steps": 3000,
    "train_fraction": 0.7,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_stepper(data: Path, weights_name: str, index_name: str, shuffle: bool = False) -> VNCStepper:
    return VNCStepper(
        data,
        substeps=PROTOCOL["substeps"],
        tau=PROTOCOL["tau"],
        dt=PROTOCOL["dt"],
        gW=PROTOCOL["gW"],
        weights_name=weights_name,
        index_name=index_name,
        model=PROTOCOL["model"],
        threshold=PROTOCOL["threshold"],
        activation_gain=PROTOCOL["activation_gain"],
        fr_cap=PROTOCOL["fr_cap"],
        heterogeneity=PROTOCOL["heterogeneity"],
        seed=PROTOCOL["seed"],
        shuffle=shuffle,
    )


def population_rates(vnc: VNCStepper) -> np.ndarray:
    cap = vnc.fr_cap if np.isscalar(vnc.fr_cap) else vnc.fr_cap.detach().cpu().numpy()
    return (vnc.v.detach().cpu().numpy() / cap).astype(np.float32, copy=False)


def collect(vnc: VNCStepper, drive: float, burn_steps: int, record_steps: int) -> np.ndarray:
    current = vnc.descending_current(drive, dnb08_gain=PROTOCOL["dnb08_gain"])
    for _ in range(burn_steps):
        vnc.step(current)
    states = np.empty((record_steps, vnc.n), dtype=np.float32)
    for step in range(record_steps):
        vnc.step(current)
        states[step] = population_rates(vnc)
    assert states.shape == (record_steps, vnc.n)
    assert np.isfinite(states).all()
    return states


def scaler(states: np.ndarray, train_fraction: float = PROTOCOL["train_fraction"]) -> tuple[np.ndarray, np.ndarray]:
    split = int(len(states) * train_fraction)
    mean = states[:split].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = states[:split].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-7] = 1.0
    return mean, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=RESULTS)
    ap.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    ap.add_argument("--index_name", default="all_legs_io_index.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "reservoir_states.npz")
    a = ap.parse_args()
    vnc = make_stepper(a.data, a.weights_name, a.index_name)
    states = collect(vnc, PROTOCOL["drive"], PROTOCOL["burn_steps"], PROTOCOL["record_steps"])
    mean, scale = scaler(states)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        states=states,
        node_body_ids=vnc.io["node_body_ids"],
        train_mean=mean,
        train_scale=scale,
    )
    metadata = {
        "protocol": PROTOCOL,
        "shape": list(states.shape),
        "weights_name": a.weights_name,
        "weights_sha256": file_sha256(a.data / a.weights_name),
        "index_name": a.index_name,
        "active_column_count": int(np.count_nonzero(states.std(axis=0) > 1e-7)),
        "output": str(a.out),
    }
    a.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
