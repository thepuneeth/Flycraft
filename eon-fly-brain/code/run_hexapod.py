"""Flycraft rich-readout experiment: FlyWire connectome -> hexapod, zero learning.

The connectome's full DESCENDING population (desc_left + desc_right ~1291 neurons)
is mapped to the hexapod's 18 joint targets through a FIXED RANDOM linear readout
(seeded Gaussian, never trained). No CPG, no hand-designed gait: whatever the body
does is produced by the connectome's own dynamics passed through a fixed map.

We measure whether the wiring alone, in a body it never evolved for, produces any
coherent movement -- and where it breaks. A frozen posture / collapse is the
expected (still-publishable) null: it would show the rate model settles to a fixed
point and carries no gait without a pattern generator.

    retina(az,dist,loom) -> visual injection -> connectome rate step
      -> descending vector d -> ctrl = clip(scale * R @ (d - mean d))
      -> MuJoCo hexapod step -> log motion

Usage: run_hexapod.py [--secs 8] [--scale 10] [--seed 0] [--prey 3 0] [--video out.mp4]
"""
import argparse, pickle
from pathlib import Path
import numpy as np
import torch
import mujoco

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
XML = HERE.parent.parent / "assets" / "hexapod" / "hexapod.xml"
import sys
sys.path.insert(0, str(HERE))
from hexapod_gait import H_MIN  # noqa: E402


def retina(trunk_xy, trunk_yaw, prey_xy):
    """Minimal fly-retina encode in the body frame. az>0 = prey in LEFT field."""
    dx, dy = prey_xy[0] - trunk_xy[0], prey_xy[1] - trunk_xy[1]
    dist = float(np.hypot(dx, dy))
    world_ang = np.arctan2(dy, dx)
    az = float(np.arctan2(np.sin(world_ang - trunk_yaw), np.cos(world_ang - trunk_yaw)))
    present = dist < 12.0 and abs(az) < np.deg2rad(150)
    size = 0.2 / max(dist, 0.2)          # angular size ~ 1/dist
    return present, az, dist, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--scale", type=float, default=10.0)   # readout gain (fixed hyperparam, not trained)
    ap.add_argument("--seed", type=int, default=0)         # fixes the random readout -> reproducible
    ap.add_argument("--prey", type=float, nargs=2, default=[3.0, 0.0])
    ap.add_argument("--ctrl_hz", type=float, default=50.0)
    # connectome dynamics (same as flyserver defaults)
    ap.add_argument("--substeps", type=int, default=12)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--gW", type=float, default=6e-3)
    ap.add_argument("--drive_in", type=float, default=1.2)
    ap.add_argument("--video", default=None)
    a = ap.parse_args()

    # --- connectome ---
    W = pickle.load(open(DATA / "weight_csr.pkl", "rb"))
    if W.layout != torch.sparse_csr:
        W = W.to_sparse_csr()
    W = W.float()
    N = W.shape[0]
    io = np.load(DATA / "io_index.npz")
    vis_left = torch.from_numpy(io["vis_left"]).long()
    vis_right = torch.from_numpy(io["vis_right"]).long()
    desc = torch.from_numpy(np.concatenate([io["desc_left"], io["desc_right"]])).long()
    n_desc = len(desc)

    # --- fixed random readout: descending population -> 18 joint targets ---
    rng = np.random.default_rng(a.seed)
    R = torch.from_numpy((rng.standard_normal((18, n_desc)) / np.sqrt(n_desc)).astype(np.float32))

    v = torch.zeros(N, dtype=torch.float32)
    decay, dt_tau = 1.0 - a.dt / a.tau, a.dt / a.tau

    def connectome_step(present, az, size):
        mag = a.drive_in * (0.5 + min(1.0, size))
        I = torch.zeros(N, dtype=torch.float32)
        if present:
            I[vis_left] = mag * (max(0.0, az) + 0.15)
            I[vis_right] = mag * (max(0.0, -az) + 0.15)
        for _ in range(a.substeps):
            r = torch.relu(torch.tanh(v))
            v.mul_(decay).add_(dt_tau * (a.gW * (W @ r) + I))
            v.clamp_(-5.0, 5.0)
        return torch.relu(torch.tanh(v))[desc]

    # --- body ---
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    # lift trunk so it settles onto its feet, then let physics drop it
    d.qpos[2] = 0.20
    ctrl_range = m.actuator_ctrlrange.astype(np.float32).copy()   # (18,2), +/-1.96
    phys_per_ctrl = max(1, int(round((1.0 / a.ctrl_hz) / m.opt.timestep)))

    renderer = None
    frames = []
    if a.video:
        renderer = mujoco.Renderer(m, 240, 320)

    # settle 0.5 s with neutral ctrl so we start from a stance, not a faceplant
    for _ in range(int(0.5 / m.opt.timestep)):
        d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)

    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    prey = (x0 + a.prey[0], y0 + a.prey[1])
    n_ctrl = int(a.secs * a.ctrl_hz)
    log = {"dist": [], "height": [], "dvar": [], "ctrlvar": [], "speed": [], "az": []}
    last = (x0, y0)

    for k in range(n_ctrl):
        trunk_xy = (float(d.qpos[0]), float(d.qpos[1]))
        yaw = float(2 * np.arctan2(d.qpos[6], d.qpos[3]))   # z-yaw from quat (w,x,y,z)=qpos[3:7]
        present, az, dist, size = retina(trunk_xy, yaw, prey)
        dvec = connectome_step(present, az, size)
        dc = dvec - dvec.mean()
        ctrl = torch.clamp(a.scale * (R @ dc), torch.from_numpy(ctrl_range[:, 0]),
                           torch.from_numpy(ctrl_range[:, 1])).numpy()
        d.ctrl[:] = ctrl
        for _ in range(phys_per_ctrl):
            mujoco.mj_step(m, d)
        sp = np.hypot(d.qpos[0] - last[0], d.qpos[1] - last[1]) * a.ctrl_hz
        last = (float(d.qpos[0]), float(d.qpos[1]))
        log["dist"].append(dist); log["height"].append(float(d.qpos[2]))
        log["dvar"].append(float(dvec.std())); log["ctrlvar"].append(float(ctrl.std()))
        log["speed"].append(float(sp)); log["az"].append(np.rad2deg(az))
        if renderer is not None and k % 2 == 0:
            renderer.update_scene(d, camera="track")
            frames.append(renderer.render())

    disp = np.hypot(d.qpos[0] - x0, d.qpos[1] - y0)
    A = {k: np.array(vv) for k, vv in log.items()}
    fell = A["height"].min() < H_MIN
    print("=== rich-readout hexapod result ===")
    print(f"seed={a.seed} scale={a.scale} secs={a.secs} n_desc={n_desc}")
    print(f"start->prey dist: {A['dist'][0]:.2f} m  final: {A['dist'][-1]:.2f} m  min: {A['dist'].min():.2f} m")
    print(f"trunk displacement: {disp:.3f} m   final height: {A['height'][-1]:.3f} m (min {A['height'].min():.3f})")
    print(f"mean body speed: {A['speed'].mean():.3f} m/s   max: {A['speed'].max():.3f} m/s")
    print(f"descending activity std over time: {A['dvar'].std():.5f} (0 = frozen fixed point, >0 = time-varying)")
    print(f"joint-target std: {A['ctrlvar'].mean():.4f} rad   fell/collapsed: {fell}")
    # verdict
    moved = disp > 0.15 and not fell
    toward = (A["dist"][0] - A["dist"].min()) > 0.15
    print(f"VERDICT: {'MOVED' if moved else 'no locomotion'}"
          f"{' + toward prey' if toward else ''}"
          f"{' | COLLAPSED' if fell else ''}")

    if a.video and frames:
        try:
            import imageio.v2 as imageio
            imageio.mimwrite(a.video, frames, fps=25)
            print(f"video -> {a.video}")
        except Exception as e:
            print(f"(video skipped: {e})")


if __name__ == "__main__":
    main()
