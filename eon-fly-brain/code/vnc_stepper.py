"""Rate-model stepper for the compact MANC VNC graph."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"


class VNCStepper:
    """Run signed MANC dynamics and return selected motor-neuron rates."""

    def __init__(
        self,
        data=DATA,
        substeps=12,
        tau=0.02,
        dt=0.005,
        gW=6e-3,
        device="cpu",
        weights_name="vnc_weight_csr.pkl",
        index_name="vnc_io_index.npz",
        model="bounded",
        threshold=7.5,
        activation_gain=1.0,
        fr_cap=200.0,
        heterogeneity=False,
        seed=1,
        shuffle=False,
    ):
        if model not in {"bounded", "paper"}:
            raise ValueError(f"unknown VNC model {model!r}; expected bounded or paper")
        if tau <= 0 or dt <= 0 or dt > tau:
            raise ValueError("VNC requires 0 < dt <= tau")
        data = Path(data)
        self.device = torch.device(device)
        with (data / weights_name).open("rb") as f:
            self.W = pickle.load(f)
        if self.W.layout != torch.sparse_csr:
            self.W = self.W.to_sparse_csr()
        self.W = self.W.float().cpu()
        if shuffle:
            generator = torch.Generator().manual_seed(int(seed))
            column_permutation = torch.randperm(self.W.shape[1], generator=generator)
            self.W = torch.sparse_csr_tensor(
                self.W.crow_indices(),
                column_permutation[self.W.col_indices()],
                self.W.values(),
                size=self.W.shape,
            )
        self.W = self.W.to(self.device)
        self.n = int(self.W.shape[0])
        self.io = np.load(data / index_name)
        if int(self.io["n"]) != self.n:
            raise ValueError(f"VNC matrix/index mismatch: {self.n} != {int(self.io['n'])}")
        self.descending_in = self.io["descending_in"].astype(np.int64)
        self.motor_out = self.io["motor_out"].astype(np.int64)
        self.v = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.substeps = int(substeps)
        self.dt = float(dt)
        self.gW = float(gW)
        self.model = model
        self.shuffle = bool(shuffle)
        if fr_cap <= 0 or activation_gain <= 0:
            raise ValueError("activation_gain and fr_cap must be positive")
        if heterogeneity and model != "paper":
            raise ValueError("heterogeneity is defined for paper model only")
        if heterogeneity:
            rng = np.random.default_rng(seed)
            tau_values = np.maximum(rng.normal(float(tau), 0.002, self.n), float(dt))
            gain_values = np.maximum(rng.normal(float(activation_gain), 0.1, self.n), 1e-3)
            threshold_values = np.maximum(rng.normal(float(threshold), 0.6, self.n), 0.0)
            cap_values = np.maximum(rng.normal(float(fr_cap), 10.0, self.n), 1.0)
            self.tau = torch.from_numpy(tau_values.astype(np.float32)).to(self.device)
            self.activation_gain = torch.from_numpy(gain_values.astype(np.float32)).to(self.device)
            self.threshold = torch.from_numpy(threshold_values.astype(np.float32)).to(self.device)
            self.fr_cap = torch.from_numpy(cap_values.astype(np.float32)).to(self.device)
        else:
            self.tau = float(tau)
            self.threshold = float(threshold)
            self.activation_gain = float(activation_gain)
            self.fr_cap = float(fr_cap)
        self.decay = 1.0 - float(dt) / self.tau
        self.dt_tau = float(dt) / self.tau

    def reset(self):
        self.v.zero_()

    def descending_current(self, drive, turn=0.0, dnb08_gain=1.0, turn_scale=1.0):
        """Make fixed DNg100/DNb08 currents from brain go/turn signals."""
        drive = max(0.0, float(drive))
        bias = float(np.clip(float(turn) / max(float(turn_scale), 1e-6), -1.0, 1.0))
        left = drive * (1.0 - bias)
        right = drive * (1.0 + bias)
        current = np.zeros(self.n, dtype=np.float32)
        for key, value in (("dng100_left", left), ("dng100_right", right)):
            idx = self.io[key]
            if len(idx):
                current[idx] = value
        for key, value in (("dnb08_left", left * dnb08_gain), ("dnb08_right", right * dnb08_gain)):
            idx = self.io[key]
            if len(idx):
                current[idx] = value
        return current

    def step(self, I_descending):
        """Advance network and return rates at ``motor_out`` indices.

        ``I_descending`` may be a full compact-N current vector or a vector
        aligned to ``descending_in``.
        """
        x = torch.as_tensor(I_descending, dtype=torch.float32, device=self.device).flatten()
        if x.numel() == self.n:
            current = x
        elif x.numel() == len(self.descending_in):
            current = torch.zeros(self.n, dtype=torch.float32, device=self.device)
            current[self.descending_in] = x
        else:
            raise ValueError(f"descending current has {x.numel()} values; expected {self.n} or {len(self.descending_in)}")
        with torch.no_grad():
            for _ in range(self.substeps):
                if self.model == "paper":
                    # Published MANC model: rate state, half-tanh thresholded
                    # activation, fixed current and connectivity gains.
                    total = self.gW * (self.W @ self.v) + current
                    target = torch.relu(
                        self.fr_cap
                        * torch.tanh((self.activation_gain / self.fr_cap) * (total - self.threshold))
                    )
                    self.v.mul_(self.decay).add_(self.dt_tau * target)
                    self.v.clamp_min_(0.0)
                    if torch.is_tensor(self.fr_cap):
                        self.v.copy_(torch.minimum(self.v, self.fr_cap))
                    else:
                        self.v.clamp_max_(self.fr_cap)
                else:
                    rates = torch.relu(torch.tanh(self.v))
                    self.v.mul_(self.decay).add_(self.dt_tau * (self.gW * (self.W @ rates) + current))
                    self.v.clamp_(-5.0, 5.0)
            if self.model == "paper":
                return (self.v / self.fr_cap)[self.motor_out].detach()
            return torch.relu(torch.tanh(self.v))[self.motor_out].detach()


def core_history(data, steps, substeps, tau, dt, gW, drive, dnb08_gain, model="bounded", threshold=7.5, activation_gain=1.0, fr_cap=200.0):
    """Run only the six annotated T1 rhythm-core cells and their inputs."""
    data = Path(data)
    with (data / "vnc_weight_csr.pkl").open("rb") as f:
        W = pickle.load(f).to_sparse_coo().coalesce()
    io = np.load(data / "vnc_io_index.npz")
    inputs = np.r_[io["dng100_left"], io["dng100_right"], io["dnb08_left"], io["dnb08_right"]]
    active = np.unique(np.r_[io["rhythm_core"], inputs]).astype(np.int64)
    index = {int(node): i for i, node in enumerate(active)}
    edge = W.indices().cpu().numpy()
    value = W.values().cpu().numpy()
    keep = np.isin(edge[0], active) & np.isin(edge[1], active)
    rows = np.asarray([index[int(node)] for node in edge[0, keep]], dtype=np.int64)
    cols = np.asarray([index[int(node)] for node in edge[1, keep]], dtype=np.int64)
    matrix = np.zeros((len(active), len(active)), dtype=np.float32)
    matrix[rows, cols] = value[keep]
    current = np.zeros(len(active), dtype=np.float32)
    bias = float(drive)
    for key in ("dng100_left", "dng100_right"):
        current[[index[int(node)] for node in io[key]]] = bias
    for key in ("dnb08_left", "dnb08_right"):
        current[[index[int(node)] for node in io[key]]] = bias * dnb08_gain
    state = np.zeros(len(active), dtype=np.float32)
    history = []
    decay = 1.0 - dt / tau
    dt_tau = dt / tau
    core_positions = [index[int(node)] for node in io["rhythm_core"]]
    for _ in range(steps):
        for _ in range(substeps):
            if model == "paper":
                total = gW * (matrix @ state) + current
                target = np.maximum(fr_cap * np.tanh((activation_gain / fr_cap) * (total - threshold)), 0.0)
                state = np.clip(decay * state + dt_tau * target, 0.0, fr_cap)
            else:
                rates = np.maximum(np.tanh(state), 0.0)
                state = np.clip(decay * state + dt_tau * (gW * (matrix @ rates) + current), -5.0, 5.0)
        history.append(state[core_positions] / fr_cap if model == "paper" else np.maximum(np.tanh(state[core_positions]), 0.0))
    return np.asarray(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--weights_name", default="vnc_weight_csr.pkl")
    ap.add_argument("--index_name", default="vnc_io_index.npz")
    ap.add_argument("--secs", type=float, default=6.0)
    ap.add_argument("--drive", type=float, default=100.0)
    ap.add_argument("--turn", type=float, default=0.0)
    ap.add_argument("--dnb08_gain", type=float, default=1.0)
    ap.add_argument("--turn_scale", type=float, default=1.0)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gW", type=float, default=0.03)
    ap.add_argument("--model", choices=("bounded", "paper"), default="paper")
    ap.add_argument("--threshold", type=float, default=7.5)
    ap.add_argument("--activation_gain", type=float, default=1.0)
    ap.add_argument("--fr_cap", type=float, default=200.0)
    ap.add_argument("--heterogeneity", action="store_true")
    ap.add_argument("--shuffle_vnc", action="store_true", help="permute presynaptic columns; preserve row degree and sparsity")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--core_only", action="store_true")
    a = ap.parse_args()

    steps = max(1, int(round(a.secs / (a.dt * a.substeps))))
    if a.core_only:
        history = core_history(a.data, steps, a.substeps, a.tau, a.dt, a.gW, a.drive, a.dnb08_gain, a.model, a.threshold, a.activation_gain, a.fr_cap)
        tail = history[len(history) // 2:]
        print(f"core-only cells={history.shape[1]} steps={steps}")
        print(f"drive={a.drive} dnb08_gain={a.dnb08_gain} gW={a.gW}")
        print(f"core temporal std all={history.std(axis=0).mean():.5f} tail={tail.std(axis=0).mean():.5f}")
        return

    vnc = VNCStepper(a.data, a.substeps, a.tau, a.dt, a.gW, a.device, weights_name=a.weights_name, index_name=a.index_name, model=a.model, threshold=a.threshold, activation_gain=a.activation_gain, fr_cap=a.fr_cap, heterogeneity=a.heterogeneity, seed=a.seed, shuffle=a.shuffle_vnc)
    current = vnc.descending_current(a.drive, a.turn, a.dnb08_gain, a.turn_scale)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)])
    temporal_std = history.std(axis=0)
    tail = history[len(history) // 2:]
    print(f"VNC N={vnc.n} motor_outputs={history.shape[1]} steps={steps}")
    print(f"model={a.model} drive={a.drive} turn={a.turn} dnb08_gain={a.dnb08_gain} gW={a.gW}")
    print(f"motor mean={history[-1].mean():.5f} temporal_std all={temporal_std.mean():.5f} tail={tail.std(axis=0).mean():.5f} max={temporal_std.max():.5f}")
    print(f"DNg100 L/R={len(vnc.io['dng100_left'])}/{len(vnc.io['dng100_right'])} DNb08 L/R={len(vnc.io['dnb08_left'])}/{len(vnc.io['dnb08_right'])}")


if __name__ == "__main__":
    main()
