"""Path 2: FlyWire brain -> MANC VNC -> fixed MN map -> hexapod physics."""
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
sys.path.insert(0, str(HERE))
from hexapod_gait import XML, pd_torque  # noqa: E402
from mn_to_joint import translate  # noqa: E402
from run_hexapod_chase import retina  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--vnc_data", type=Path, default=None, help="VNC artifact directory; defaults to --data")
    ap.add_argument("--vnc_weights_name", default="vnc_weight_csr.pkl")
    ap.add_argument("--vnc_index_name", default="vnc_io_index.npz")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--prey", type=float, nargs=2, default=[1.2, 0.8])
    ap.add_argument("--initial_yaw", type=float, default=0.0, help="initial body yaw in radians")
    ap.add_argument("--catch", type=float, default=0.25)
    ap.add_argument("--ctrl_hz", type=float, default=50.0)
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--kd", type=float, default=0.05)
    ap.add_argument("--mn_coxa_gain", type=float, default=-0.55)
    ap.add_argument("--mn_femur_gain", type=float, default=0.65)
    ap.add_argument("--mn_tibia_gain", type=float, default=0.65)
    ap.add_argument("--open_loop", action="store_true", help="drive VNC directly; skip brain")
    ap.add_argument("--open_loop_drive", type=float, default=1.0)
    ap.add_argument("--open_loop_turn", type=float, default=0.0)
    ap.add_argument("--random_descending", action="store_true", help="replace constant open-loop drive with seeded matched-scale noise")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shuffle_vnc", action="store_true", help="permute VNC presynaptic columns")
    # FlyWire brain dynamics, copied from Path 1.
    ap.add_argument("--substeps", type=int, default=12)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gW", type=float, default=6e-3)
    ap.add_argument("--drive_in", type=float, default=1.2)
    ap.add_argument("--turn_gain", type=float, default=6.0)
    ap.add_argument("--turn_sign", type=float, default=-1.0)
    ap.add_argument("--fwd_gain", type=float, default=12.0)
    # MANC coupling. Fixed constants, never fit to catch success.
    ap.add_argument("--vnc_drive", type=float, default=100.0)
    ap.add_argument("--vnc_turn_scale", type=float, default=6.0)
    ap.add_argument("--dnb08_gain", type=float, default=1.0)
    ap.add_argument("--vnc_gW", type=float, default=0.03)
    ap.add_argument("--vnc_substeps", type=int, default=4)
    ap.add_argument("--vnc_tau", type=float, default=0.02)
    ap.add_argument("--vnc_dt", type=float, default=0.005)
    ap.add_argument("--vnc_model", choices=("bounded", "paper"), default="paper")
    ap.add_argument("--vnc_threshold", type=float, default=7.5)
    ap.add_argument("--vnc_activation_gain", type=float, default=1.0)
    ap.add_argument("--vnc_fr_cap", type=float, default=200.0)
    ap.add_argument("--vnc_heterogeneity", action="store_true")
    ap.add_argument("--vnc_seed", type=int, default=1)
    ap.add_argument("--video", default=None)
    ap.add_argument("--multi_view", action="store_true", help="tile track, overhead, side, and front views of same episode")
    a = ap.parse_args()

    vnc_data = a.vnc_data or a.data
    vnc = VNCStepper(
        vnc_data,
        substeps=a.vnc_substeps,
        tau=a.vnc_tau,
        dt=a.vnc_dt,
        gW=a.vnc_gW,
        weights_name=a.vnc_weights_name,
        index_name=a.vnc_index_name,
        model=a.vnc_model,
        threshold=a.vnc_threshold,
        activation_gain=a.vnc_activation_gain,
        fr_cap=a.vnc_fr_cap,
        heterogeneity=a.vnc_heterogeneity,
        seed=a.vnc_seed,
        shuffle=a.shuffle_vnc,
    )
    rng = np.random.default_rng(a.seed)
    # Brain is unnecessary for the open-loop milestone.
    if a.open_loop:
        brain_W = brain_v = None
    else:
        with (a.data / "weight_csr.pkl").open("rb") as f:
            brain_W = pickle.load(f)
        if brain_W.layout != torch.sparse_csr:
            brain_W = brain_W.to_sparse_csr()
        brain_W = brain_W.float()
        brain_n = brain_W.shape[0]
        brain_io = np.load(a.data / "io_index.npz")
        vis_left = torch.from_numpy(brain_io["vis_left"]).long()
        vis_right = torch.from_numpy(brain_io["vis_right"]).long()
        desc_left = torch.from_numpy(brain_io["desc_left"]).long()
        desc_right = torch.from_numpy(brain_io["desc_right"]).long()
        brain_v = torch.zeros(brain_n, dtype=torch.float32)
        brain_decay = 1.0 - a.dt / a.tau
        brain_dt_tau = a.dt / a.tau

    def brain(present, az, size):
        if a.open_loop:
            if a.random_descending:
                return max(0.0, float(rng.normal(a.open_loop_drive, 0.25 * max(a.open_loop_drive, 0.1)))), float(rng.normal(a.open_loop_turn, 0.25 * max(a.vnc_turn_scale, 0.1))), 0.0, 0.0
            return a.open_loop_drive, a.open_loop_turn, 0.0, 0.0
        mag = a.drive_in * (0.5 + min(1.0, size))
        current = torch.zeros(brain_n, dtype=torch.float32)
        if present:
            current[vis_left] = mag * (max(0.0, az) + 0.15)
            current[vis_right] = mag * (max(0.0, -az) + 0.15)
        with torch.no_grad():
            for _ in range(a.substeps):
                rates = torch.relu(torch.tanh(brain_v))
                brain_v.mul_(brain_decay).add_(brain_dt_tau * (a.gW * (brain_W @ rates) + current))
                brain_v.clamp_(-5.0, 5.0)
            rates = torch.relu(torch.tanh(brain_v))
        dL, dR = float(rates[desc_left].mean()), float(rates[desc_right].mean())
        turn = a.turn_sign * a.turn_gain * (dR - dL)
        gate = max(0.1, float(np.cos(az))) if present else 0.0
        forward = float(np.clip(a.fwd_gain * (dL + dR), 0.0, 1.0)) * gate
        return forward, turn, dL, dR

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    d.qpos[2] = 0.15
    d.qpos[3:7] = (np.cos(a.initial_yaw / 2), 0.0, 0.0, np.sin(a.initial_yaw / 2))
    phys = max(1, int(round((1.0 / a.ctrl_hz) / m.opt.timestep)))
    dt = 1.0 / a.ctrl_hz
    renderer = mujoco.Renderer(m, 480, 640) if a.video else None
    if a.multi_view:
        cameras = []
        for azimuth, elevation, distance in ((135.0, 25.0, 1.4), (0.0, 80.0, 1.8), (90.0, 25.0, 1.4), (180.0, 25.0, 1.4)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(m, camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = distance
            cameras.append(camera)
    else:
        cameras = ("track",)
    frames = []
    map_gains = (a.mn_coxa_gain, a.mn_femur_gain, a.mn_tibia_gain)

    stance = translate(np.zeros(len(vnc.motor_out)), vnc.io, gains=map_gains)
    for _ in range(int(0.7 / m.opt.timestep)):
        d.ctrl[:] = pd_torque(m, d, stance, a.kp, a.kd)
        mujoco.mj_step(m, d)

    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    yaw0 = 2 * np.arctan2(d.qpos[6], d.qpos[3])
    prey = (x0 + a.prey[0], y0 + a.prey[1])
    d0 = float(np.hypot(*a.prey))
    n = int(a.secs * a.ctrl_hz)
    last = (x0, y0)
    path = 0.0
    mind = d0
    az_hist = []
    motor_hist = []
    height_hist = []
    caught = None

    for k in range(n):
        trunk = (float(d.qpos[0]), float(d.qpos[1]))
        yaw = 2 * np.arctan2(d.qpos[6], d.qpos[3])
        present, az, dist, size = retina(trunk, yaw, prey)
        forward, turn, dL, dR = brain(present, az, size)
        current = vnc.descending_current(forward * a.vnc_drive, turn, a.dnb08_gain, a.vnc_turn_scale)
        motor = vnc.step(current).cpu().numpy()
        target = translate(motor, vnc.io, gains=map_gains)
        for _ in range(phys):
            d.ctrl[:] = pd_torque(m, d, target, a.kp, a.kd)
            mujoco.mj_step(m, d)
        trunk2 = (float(d.qpos[0]), float(d.qpos[1]))
        path += float(np.hypot(trunk2[0] - last[0], trunk2[1] - last[1]))
        last = trunk2
        mind = min(mind, dist)
        az_hist.append(abs(np.rad2deg(az)))
        motor_hist.append(motor)
        height_hist.append(float(d.qpos[2]))
        if renderer and k % 2 == 0:
            views = []
            for camera in cameras:
                if a.multi_view:
                    camera.lookat[:] = (float(d.qpos[0]), float(d.qpos[1]), 0.08)
                renderer.update_scene(d, camera=camera)
                views.append(renderer.render())
            if a.multi_view:
                top = np.concatenate(views[:2], axis=1)
                bottom = np.concatenate(views[2:], axis=1)
                frames.append(np.concatenate((top, bottom), axis=0))
            else:
                frames.append(views[0])
        if k % 25 == 0 or dist < a.catch:
            print(f"t={k*dt:4.1f}s d={dist:4.2f} az={np.rad2deg(az):+5.0f} fwd={forward:.2f} turn={turn:+.2f} dL={dL:.3f} dR={dR:.3f} motor={motor.mean():.3f} h={d.qpos[2]:.3f}")
        if dist < a.catch:
            caught = k * dt
            break

    motor_hist = np.stack(motor_hist)
    heights = np.asarray(height_hist)
    dx, dy = float(d.qpos[0]) - x0, float(d.qpos[1]) - y0
    yaw1 = 2 * np.arctan2(d.qpos[6], d.qpos[3])
    displacement = float(np.hypot(dx, dy))
    forward_displacement = float(dx * np.cos(yaw0) + dy * np.sin(yaw0))
    side_displacement = float(-dx * np.sin(yaw0) + dy * np.cos(yaw0))
    print("=== VNC hexapod chase result ===")
    print(f"mode={'open-loop' if a.open_loop else 'brain-coupled'} vnc_N={vnc.n} motor_outputs={len(vnc.motor_out)}")
    print(f"start dist={d0:.2f}m  min dist={mind:.2f}m  path={path:.2f}m")
    print(f"displacement={displacement:.2f}m  forward={forward_displacement:+.2f}m  side={side_displacement:+.2f}m  yaw={np.rad2deg(yaw1-yaw0):+.1f}deg")
    print(f"motor temporal std={motor_hist.std(axis=0).mean():.5f}  mean|az|={np.mean(az_hist):.1f}deg")
    print(f"final height={heights[-1]:.3f}  min height={heights.min():.3f}  fell={heights.min()<0.02}")
    print(f"VERDICT: {'CAUGHT at '+format(caught,'.1f')+'s' if caught is not None else 'not caught in '+str(a.secs)+'s'}")
    print("RESULT_JSON " + json.dumps({
        "mode": "open_loop" if a.open_loop else "brain_coupled",
        "vnc_model": a.vnc_model,
        "vnc_n": vnc.n,
        "motor_outputs": len(vnc.motor_out),
        "start_distance_m": d0,
        "min_distance_m": mind,
        "path_m": path,
        "displacement_m": displacement,
        "forward_displacement_m": forward_displacement,
        "side_displacement_m": side_displacement,
        "yaw_change_deg": float(np.rad2deg(yaw1 - yaw0)),
        "motor_temporal_std_mean": float(motor_hist.std(axis=0).mean()),
        "mean_abs_az_deg": float(np.mean(az_hist)),
        "final_height_m": float(heights[-1]),
        "min_height_m": float(heights.min()),
        "fell": bool(heights.min() < 0.02),
        "caught": caught is not None,
        "catch_time_s": caught,
        "secs": a.secs,
        "vnc_drive": a.vnc_drive,
        "vnc_gW": a.vnc_gW,
        "vnc_tau": a.vnc_tau,
        "vnc_substeps": a.vnc_substeps,
        "vnc_heterogeneity": a.vnc_heterogeneity,
        "vnc_seed": a.vnc_seed,
        "shuffle_vnc": a.shuffle_vnc,
        "random_descending": a.random_descending,
        "seed": a.seed,
        "mn_gains": map_gains,
    }, sort_keys=True))
    if a.video and frames:
        import imageio.v2 as imageio
        imageio.mimwrite(a.video, frames, fps=25)
        print(f"video -> {a.video}")


if __name__ == "__main__":
    main()
