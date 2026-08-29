"""The honest Flycraft experiment: FlyWire connectome STEERS a physics hexapod.

Biologically-faithful split of labour:
  vision -> CONNECTOME (decides steer + go, ZERO learning) -> {forward, turn}
         -> CPG gait (rhythm = ventral-nerve-cord stand-in) -> PD torque -> hexapod legs

The brain does NOT make the walking rhythm (we proved it settles to a fixed point and
can't -- run_hexapod.py). It supplies only the two descending signals a real fly brain
supplies: how much to go, which way to turn. The CPG (hexapod_gait) makes the legs march.
The body then physically pursues a prey target in MuJoCo.

Readout = same as flyserver: dL/dR = mean firing of left/right descending populations;
turn = sign*gain*(dR-dL), forward = clip(fwd_gain*(dL+dR)). No task training anywhere.

Usage: run_hexapod_chase.py [--secs 25] [--prey 1.2 0.8] [--turn_gain ..] [--video f.mp4]
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
import torch
import mujoco

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
sys.path.insert(0, str(HERE))
from hexapod_gait import HexapodGait, pd_torque, XML


def retina(trunk_xy, yaw, prey):
    dx, dy = prey[0] - trunk_xy[0], prey[1] - trunk_xy[1]
    dist = float(np.hypot(dx, dy))
    ang = np.arctan2(dy, dx)
    az = float(np.arctan2(np.sin(ang - yaw), np.cos(ang - yaw)))   # +left
    present = dist < 12.0 and abs(az) < np.deg2rad(150)
    return present, az, dist, 0.2 / max(dist, 0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=25.0)
    ap.add_argument("--prey", type=float, nargs=2, default=[1.2, 0.8])
    ap.add_argument("--catch", type=float, default=0.25)
    ap.add_argument("--ctrl_hz", type=float, default=50.0)
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--kd", type=float, default=0.05)
    # connectome
    ap.add_argument("--substeps", type=int, default=12)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gW", type=float, default=6e-3)
    ap.add_argument("--drive_in", type=float, default=1.2)
    ap.add_argument("--turn_gain", type=float, default=8.0)
    ap.add_argument("--turn_sign", type=float, default=-1.0)
    ap.add_argument("--fwd_gain", type=float, default=3.0)
    ap.add_argument("--video", default=None)
    a = ap.parse_args()

    W = pickle.load(open(DATA / "weight_csr.pkl", "rb"))
    if W.layout != torch.sparse_csr:
        W = W.to_sparse_csr()
    W = W.float()
    N = W.shape[0]
    io = np.load(DATA / "io_index.npz")
    vis_left = torch.from_numpy(io["vis_left"]).long()
    vis_right = torch.from_numpy(io["vis_right"]).long()
    desc_left = torch.from_numpy(io["desc_left"]).long()
    desc_right = torch.from_numpy(io["desc_right"]).long()
    v = torch.zeros(N, dtype=torch.float32)
    decay, dt_tau = 1.0 - a.dt / a.tau, a.dt / a.tau

    def brain(present, az, size):
        mag = a.drive_in * (0.5 + min(1.0, size))
        I = torch.zeros(N, dtype=torch.float32)
        if present:
            I[vis_left] = mag * (max(0.0, az) + 0.15)
            I[vis_right] = mag * (max(0.0, -az) + 0.15)
        for _ in range(a.substeps):
            r = torch.relu(torch.tanh(v))
            v.mul_(decay).add_(dt_tau * (a.gW * (W @ r) + I))
            v.clamp_(-5.0, 5.0)
        r = torch.relu(torch.tanh(v))
        dL, dR = float(r[desc_left].mean()), float(r[desc_right].mean())
        turn = a.turn_sign * a.turn_gain * (dR - dL)
        # charge only when roughly facing prey; pivot (near-zero forward) when it's abeam.
        # stops the orbit -> intercept. Sensory heading gate, not learning.
        gate = max(0.1, float(np.cos(az))) if present else 0.0
        forward = float(np.clip(a.fwd_gain * (dL + dR), 0, 1)) * gate
        return forward, turn, dL, dR

    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    gait = HexapodGait(stance=(0.0, 0.6, -1.2))
    d.qpos[2] = 0.15
    phys = max(1, int(round((1 / a.ctrl_hz) / m.opt.timestep)))
    dt = 1 / a.ctrl_hz
    renderer = mujoco.Renderer(m, 480, 640) if a.video else None
    frames = []

    stance_q = gait.targets(0, 0, 0)
    for _ in range(int(0.7 / m.opt.timestep)):
        d.ctrl[:] = pd_torque(m, d, stance_q, a.kp, a.kd)
        mujoco.mj_step(m, d)

    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    prey = (x0 + a.prey[0], y0 + a.prey[1])
    d0 = np.hypot(a.prey[0], a.prey[1])
    n = int(a.secs * a.ctrl_hz)
    path = 0.0
    last = (x0, y0)
    mind = d0
    caught = None
    az_hist = []

    for k in range(n):
        trunk = (float(d.qpos[0]), float(d.qpos[1]))
        yaw = 2 * np.arctan2(d.qpos[6], d.qpos[3])
        present, az, dist, size = retina(trunk, yaw, prey)
        forward, turn, dL, dR = brain(present, az, size)
        tgt = gait.targets(forward, turn, dt)
        for _ in range(phys):
            d.ctrl[:] = pd_torque(m, d, tgt, a.kp, a.kd)
            mujoco.mj_step(m, d)
        trunk2 = (float(d.qpos[0]), float(d.qpos[1]))
        path += np.hypot(trunk2[0] - last[0], trunk2[1] - last[1]); last = trunk2
        mind = min(mind, dist)
        az_hist.append(abs(np.rad2deg(az)))
        if renderer and k % 2 == 0:
            renderer.update_scene(d, camera="track"); frames.append(renderer.render())
        if k % 25 == 0 or dist < a.catch:
            print(f"t={k*dt:4.1f}s d={dist:4.2f} az={np.rad2deg(az):+5.0f} fwd={forward:.2f} turn={turn:+.2f} dL={dL:.3f} dR={dR:.3f} h={d.qpos[2]:.3f}")
        if dist < a.catch:
            caught = k * dt; break

    print("=== hexapod chase result ===")
    print(f"start dist={d0:.2f}m  min dist={mind:.2f}m  path={path:.2f}m")
    print(f"mean|az|={np.mean(az_hist):.1f}deg  final height={d.qpos[2]:.3f}  fell={d.qpos[2]<0.02}")
    print(f"VERDICT: {'CAUGHT at '+format(caught,'.1f')+'s' if caught else 'not caught in '+str(a.secs)+'s'}")
    if a.video and frames:
        import imageio.v2 as imageio
        imageio.mimwrite(a.video, frames, fps=25)
        print(f"video -> {a.video}")


if __name__ == "__main__":
    main()
