"""Fixed MANC motor-pool to PhantomX joint-coordinate translation.

Fly groups: coxa, trochanter, femur, tibia, tarsus.
Hexapod groups: coxa <- coxa; femur <- mean(trochanter, femur);
tibia <- mean(tibia, tarsus). Each fly pool contributes
``mean(flexor-like) - mean(extensor-like)`` using fixed labels from fetch_manc.
Absent pools produce the neutral stance angle, never a copied or learned gait.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hexapod_gait import home_stance, resolve_stance


HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"


def _pool_score(activity, legs, joints, signs):
    positive_sum = np.zeros((6, 5), dtype=np.float64)
    negative_sum = np.zeros((6, 5), dtype=np.float64)
    positive_n = np.zeros((6, 5), dtype=np.int64)
    negative_n = np.zeros((6, 5), dtype=np.int64)
    positive = signs > 0
    negative = signs < 0
    np.add.at(positive_sum, (legs[positive], joints[positive]), activity[positive])
    np.add.at(negative_sum, (legs[negative], joints[negative]), activity[negative])
    np.add.at(positive_n, (legs[positive], joints[positive]), 1)
    np.add.at(negative_n, (legs[negative], joints[negative]), 1)
    pos = np.divide(positive_sum, positive_n, out=np.zeros_like(positive_sum), where=positive_n != 0)
    neg = np.divide(negative_sum, negative_n, out=np.zeros_like(negative_sum), where=negative_n != 0)
    return pos - neg, positive_n + negative_n


def translate(motor_activity, io, stance=None, gains=(-0.55, 0.65, 0.65)):
    """Return 18 fixed joint targets from one VNC motor-rate vector."""
    activity = np.asarray(motor_activity, dtype=np.float64).reshape(-1)
    expected = len(io["motor_out"])
    if len(activity) != expected:
        raise ValueError(f"motor activity has {len(activity)} values; expected {expected}")
    legs = io["motor_leg"].astype(np.int64)
    joints = io["motor_joint"].astype(np.int64)
    signs = io["motor_sign"].astype(np.float64)
    score, pool_n = _pool_score(activity, legs, joints, signs)

    target = resolve_stance(stance)
    for leg in range(6):
        coxa = score[leg, 0] if pool_n[leg, 0] else 0.0
        femur_values = [score[leg, j] for j in (1, 2) if pool_n[leg, j]]
        tibia_values = [score[leg, j] for j in (3, 4) if pool_n[leg, j]]
        target[leg, 0] += gains[0] * coxa
        if femur_values:
            target[leg, 1] += gains[1] * float(np.mean(femur_values))
        if tibia_values:
            target[leg, 2] += gains[2] * float(np.mean(tibia_values))
    return target.reshape(18).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    a = ap.parse_args()
    io = np.load(a.data / "vnc_io_index.npz")
    zero = translate(np.zeros(len(io["motor_out"])), io)
    assert zero.shape == (18,)
    assert np.allclose(zero.reshape(6, 3), home_stance())
    print(f"motor outputs={len(io['motor_out'])} active pools={int((io['motor_pool_sizes'] > 0).sum())}")
    print("zero-drive targets:", zero.reshape(6, 3).tolist())


if __name__ == "__main__":
    main()
