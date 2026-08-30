"""R2: deploy a frozen-reservoir ridge readout on the MuJoCo hexapod."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch


HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE.parent))
from collect_reservoir_states import PROTOCOL, make_stepper, population_rates  # noqa: E402
from hexapod_gait import H_MIN, XML, pd_torque  # noqa: E402
from probe_vnc_phase import (  # noqa: E402
    BANDPASS_HZ,
    PHASE_TOLERANCE_CYCLES,
    TRIPOD_A,
    TRIPOD_B,
    analytic_signal,
    fft_bandpass,
    pair_list,
    wrap_cycles,
)
from run_hexapod_chase import retina  # noqa: E402


def load_readout(path: Path) -> dict:
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def target_from_state(vnc, readout: dict) -> np.ndarray:
    rates = population_rates(vnc)
    x = (rates - readout["train_mean"]) / readout["train_scale"]
    kind = str(readout["readout_kind"])
    if kind == "random_features":
        hidden = np.tanh(x[readout["active_indices"]] @ readout["projection"])
        x = (hidden - readout["feature_mean"]) / readout["feature_scale"]
    elif kind != "linear":
        raise ValueError(f"unknown readout kind {kind!r}")
    return (x @ readout["W_out"] + readout["stance"]).astype(np.float32)


def tripod_metrics(targets: np.ndarray, sample_dt: float = PROTOCOL["sample_dt_s"]) -> dict:
    coxa = targets[:, ::3]
    tail = coxa[len(coxa) // 2:]
    filtered = fft_bandpass(tail, sample_dt, *BANDPASS_HZ)
    phase = np.angle(analytic_signal(filtered))
    difference = np.zeros((6, 6), dtype=float)
    for left in range(6):
        for right in range(6):
            difference[left, right] = np.angle(np.mean(np.exp(1j * (phase[:, left] - phase[:, right]))))
    within = pair_list((TRIPOD_A, TRIPOD_B))
    cross = [(left, right) for left in TRIPOD_A for right in TRIPOD_B]
    cycles = difference / (2 * np.pi)
    within_cycles = np.asarray([cycles[left, right] for left, right in within])
    cross_cycles = np.asarray([cycles[left, right] for left, right in cross])
    within_ok = np.abs(wrap_cycles(within_cycles)) <= PHASE_TOLERANCE_CYCLES
    cross_ok = (0.5 - np.abs(wrap_cycles(cross_cycles))) <= PHASE_TOLERANCE_CYCLES
    within_cos = float(np.mean([np.cos(difference[left, right]) for left, right in within]))
    cross_cos = float(np.mean([np.cos(difference[left, right]) for left, right in cross]))
    return {
        "within_phase_cosine": within_cos,
        "cross_phase_cosine": cross_cos,
        "tripod_coherence_scalar": within_cos - cross_cos,
        "within_pairs_in_phase": int(within_ok.sum()),
        "within_pair_count": len(within),
        "cross_pairs_antiphase": int(cross_ok.sum()),
        "cross_pair_count": len(cross),
        "phase_difference_matrix_cycles": cycles.tolist(),
        "e0_phase_verdict": "PRESENT" if bool(within_ok.all() and cross_ok.all()) else "ABSENT",
    }


class FrozenBrain:
    """Exact unlearned Path-2 visual steering forward pass."""

    def __init__(self, data: Path):
        with (data / "weight_csr.pkl").open("rb") as stream:
            self.W = pickle.load(stream)
        if self.W.layout != torch.sparse_csr:
            self.W = self.W.to_sparse_csr()
        self.W = self.W.float()
        io = np.load(data / "io_index.npz")
        self.vis_left = torch.from_numpy(io["vis_left"]).long()
        self.vis_right = torch.from_numpy(io["vis_right"]).long()
        self.desc_left = torch.from_numpy(io["desc_left"]).long()
        self.desc_right = torch.from_numpy(io["desc_right"]).long()
        self.v = torch.zeros(self.W.shape[0], dtype=torch.float32)

    def step(self, present: bool, azimuth: float, size: float) -> tuple[float, float]:
        current = torch.zeros(len(self.v), dtype=torch.float32)
        magnitude = 1.2 * (0.5 + min(1.0, size))
        if present:
            current[self.vis_left] = magnitude * (max(0.0, azimuth) + 0.15)
            current[self.vis_right] = magnitude * (max(0.0, -azimuth) + 0.15)
        with torch.no_grad():
            for _ in range(12):
                rates = torch.relu(torch.tanh(self.v))
                self.v.mul_(0.75).add_(0.25 * (0.006 * (self.W @ rates) + current))
                self.v.clamp_(-5.0, 5.0)
            rates = torch.relu(torch.tanh(self.v))
        left = float(rates[self.desc_left].mean())
        right = float(rates[self.desc_right].mean())
        turn = -6.0 * (right - left)
        gate = max(0.1, float(np.cos(azimuth))) if present else 0.0
        forward = float(np.clip(12.0 * (left + right), 0.0, 1.0)) * gate
        return forward, turn


def run_episode(
    vnc,
    readout: dict,
    brain_data: Path = DATA,
    secs: float = 8.0,
    prey_offset: tuple[float, float] = (1.2, 0.8),
    video: Path | None = None,
) -> dict:
    constant = vnc.descending_current(PROTOCOL["drive"])
    for _ in range(PROTOCOL["burn_steps"]):
        vnc.step(constant)
    brain = FrozenBrain(brain_data)
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    physics_steps = max(1, int(round(PROTOCOL["sample_dt_s"] / model.opt.timestep)))
    renderer = mujoco.Renderer(model, 480, 640) if video else None
    cameras = []
    if renderer:
        for azimuth, elevation, distance in ((135.0, 25.0, 1.4), (0.0, 80.0, 1.8), (90.0, 25.0, 1.4), (180.0, 25.0, 1.4)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
            cameras.append(camera)
    frames = []
    stance = readout["stance"]
    for _ in range(int(0.7 / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, stance, 3.0, 0.05)
        mujoco.mj_step(model, data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    yaw0 = 2 * np.arctan2(data.qpos[6], data.qpos[3])
    prey = (x0 + prey_offset[0], y0 + prey_offset[1])
    last = (x0, y0)
    path = 0.0
    min_distance = float(np.hypot(*prey_offset))
    targets, heights = [], []
    caught = False
    for step in range(int(round(secs / PROTOCOL["sample_dt_s"]))):
        trunk = (float(data.qpos[0]), float(data.qpos[1]))
        yaw = 2 * np.arctan2(data.qpos[6], data.qpos[3])
        present, azimuth, distance, size = retina(trunk, yaw, prey)
        forward, turn = brain.step(present, azimuth, size)
        # Use the exact unlearned forward/turn brain outputs from run_vnc_chase;
        # the tonic 100 protocol remains fixed for R0 and the reservoir burn-in.
        current = vnc.descending_current(forward * PROTOCOL["drive"], turn, turn_scale=6.0)
        vnc.step(current)
        target = target_from_state(vnc, readout)
        targets.append(target)
        for _ in range(physics_steps):
            data.ctrl[:] = pd_torque(model, data, target, 3.0, 0.05)
            mujoco.mj_step(model, data)
        trunk2 = (float(data.qpos[0]), float(data.qpos[1]))
        path += float(np.hypot(trunk2[0] - last[0], trunk2[1] - last[1]))
        last = trunk2
        min_distance = min(min_distance, distance)
        caught |= distance < 0.25
        heights.append(float(data.qpos[2]))
        if renderer and step % 2 == 0:
            views = []
            for camera in cameras:
                camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.08)
                renderer.update_scene(data, camera=camera)
                views.append(renderer.render())
            frames.append(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))
    targets = np.asarray(targets)
    heights = np.asarray(heights)
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    metrics = {
        **tripod_metrics(targets),
        "readout_kind": str(readout["readout_kind"]),
        "secs": secs,
        "path_m": path,
        "forward_displacement_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "min_prey_distance_m": min_distance,
        "caught": caught,
        "fell": bool(heights.min() < H_MIN),
        "min_height_m": float(heights.min()),
        "final_height_m": float(heights[-1]),
        "mean_target_temporal_sd_rad": float(targets.std(axis=0).mean()),
        "frozen_vnc": True,
        "steering_brain_driven_unlearned": True,
        "brain_forward_used": True,
        "tonic_drive": PROTOCOL["drive"],
    }
    if renderer:
        import imageio.v2 as imageio
        imageio.mimwrite(video, frames, fps=25)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vnc_data", type=Path, default=RESULTS)
    ap.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    ap.add_argument("--index_name", default="all_legs_io_index.npz")
    ap.add_argument("--brain_data", type=Path, default=DATA)
    ap.add_argument("--readout", type=Path, default=RESULTS / "reservoir_readout_W.npz")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=RESULTS / "reservoir_walk_metrics.json")
    ap.add_argument("--video", type=Path, default=RESULTS / "reservoir_walk_multiview.mp4")
    a = ap.parse_args()
    vnc = make_stepper(a.vnc_data, a.weights_name, a.index_name)
    metrics = run_episode(vnc, load_readout(a.readout), a.brain_data, a.secs, video=a.video)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("RESULT_JSON " + json.dumps(metrics, sort_keys=True))
    print(f"video -> {a.video}")


if __name__ == "__main__":
    main()
