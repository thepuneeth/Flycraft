"""Hexapod tripod-gait CPG + PD torque control (the VNC stand-in / rhythm layer).

The PhantomX hexapod uses TORQUE motors (ctrl = joint torque, not target angle), so
we run a PD servo: tau = kp*(target - q) - kd*qdot, clamped to ctrlrange. A tripod CPG
generates the leg targets. This module has NO connectome in it -- it's the pure gait
engine we calibrate to walk straight/turn on a constant {forward,turn} drive, exactly
like a ventral-nerve-cord central pattern generator. The connectome plugs in later,
supplying only {forward,turn}.

Leg mounts (from model): LEFT={0,2,4}, RIGHT={1,3,5}. Alternating tripod:
  tripod A = {0 (front-L), 3 (mid-R), 4 (back-L)}, tripod B = {1,2,5}.

Run directly to test: hexapod_gait.py [--mode stand|walk] [--drive .8] [--turn 0] [--secs 5] [--video f.mp4]
"""
import argparse
import numpy as np
import mujoco

from pathlib import Path
XML = Path(__file__).resolve().parent.parent.parent / "assets" / "hexapod" / "hexapod.xml"

LEFT = {0, 2, 4}
TRIPOD_A = {0, 3, 4}                      # half-cycle out of phase with the rest
H_MIN = 0.08                              # ride-height floor; below this is a fall
# per-leg joint indices in ctrl: leg i -> [3i (coxa), 3i+1 (femur), 3i+2 (tibia)]


def home_stance(model: mujoco.MjModel | None = None) -> np.ndarray:
    """Return the six-leg stance encoded by the model's author-declared home keyframe."""
    model = model or mujoco.MjModel.from_xml_path(str(XML))
    qpos = np.asarray(model.key("home").qpos, dtype=np.float64)
    if qpos.shape != (model.nq,) or model.nq < 25:
        raise ValueError(f"home keyframe qpos shape {qpos.shape}; expected at least 25 values")
    return qpos[7:25].reshape(6, 3).copy()


def resolve_stance(stance: np.ndarray | tuple[float, ...] | None = None) -> np.ndarray:
    """Normalize a uniform, per-leg, or omitted stance to a six-by-three array."""
    values = home_stance() if stance is None else np.asarray(stance, dtype=np.float64)
    if values.size == 3:
        return np.tile(values.reshape(1, 3), (6, 1))
    if values.size == 18:
        return values.reshape(6, 3).copy()
    raise ValueError(f"stance has {values.size} values; expected 3 or 18")


class HexapodGait:
    """Tripod CPG producing 18 joint targets from a {forward,turn} drive."""
    def __init__(self, stance=None, swing_amp=0.35, lift_amp=0.7,
                 base_freq=0.8, drive_freq=0.5, turn_scale=1.0):
        self.stance = resolve_stance(stance)
        self.swing_amp, self.lift_amp = float(swing_amp), float(lift_amp)
        self.base_freq, self.drive_freq = base_freq, drive_freq
        self.turn_scale = turn_scale
        self.phi = 0.0

    def targets(self, forward, turn, dt):
        forward = float(np.clip(forward, 0, 1))
        freq = (self.base_freq + self.drive_freq * forward) if forward > 0.02 else 0.0
        self.phi = (self.phi + 2 * np.pi * freq * dt) % (2 * np.pi)
        bias = float(np.clip(turn / self.turn_scale, -0.9, 0.9))   # >0 steer left
        q = np.zeros(18)
        for leg in range(6):
            p = self.phi + (np.pi if leg in TRIPOD_A else 0.0)
            side = 1.0 if leg in LEFT else -1.0
            # turn: inside legs take a shorter stride (differential drive)
            side_gain = (1 - bias) if leg in LEFT else (1 + bias)
            coxa0, femur0, tibia0 = self.stance[leg]
            coxa = coxa0 - side * self.swing_amp * side_gain * forward * np.sin(p)
            lift = side * self.lift_amp * max(0.0, np.sin(p))     # mirrored lift; stance half stays planted
            femur = femur0 + lift
            q[3 * leg] = coxa
            q[3 * leg + 1] = femur
            q[3 * leg + 2] = tibia0
        return q


def pd_torque(m, d, target, kp, kd):
    # joint qpos/qvel: freejoint takes qpos[0:7]/qvel[0:6]; 18 hinges follow
    q = d.qpos[7:7 + 18]
    qd = d.qvel[6:6 + 18]
    tau = kp * (target - q) - kd * qd
    return np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


def run(mode="walk", drive=0.8, turn=0.0, secs=5.0, stance=None,
        kp=3.0, kd=0.05, video=None, ctrl_hz=100.0, verbose=True):
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    gait = HexapodGait(stance=stance)
    d.qpos[2] = 0.15
    phys = max(1, int(round((1 / ctrl_hz) / m.opt.timestep)))
    n = int(secs * ctrl_hz)
    renderer = mujoco.Renderer(m, 240, 320) if video else None
    frames = []

    # settle onto stance
    stance_q = gait.targets(0, 0, 0)
    for _ in range(int(0.7 / m.opt.timestep)):
        d.ctrl[:] = pd_torque(m, d, stance_q, kp, kd)
        mujoco.mj_step(m, d)

    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    yaw0 = 2 * np.arctan2(d.qpos[6], d.qpos[3])
    heights = []
    for k in range(n):
        tgt = stance_q if mode == "stand" else gait.targets(drive, turn, 1 / ctrl_hz)
        for _ in range(phys):
            d.ctrl[:] = pd_torque(m, d, tgt, kp, kd)
            mujoco.mj_step(m, d)
        heights.append(float(d.qpos[2]))
        if renderer and k % 4 == 0:
            renderer.update_scene(d, camera="track"); frames.append(renderer.render())

    dx, dy = float(d.qpos[0]) - x0, float(d.qpos[1]) - y0
    yaw1 = 2 * np.arctan2(d.qpos[6], d.qpos[3])
    disp = np.hypot(dx, dy)
    fwd_body = dx * np.cos(yaw0) + dy * np.sin(yaw0)     # displacement along initial heading
    H = np.array(heights)
    if verbose:
        print(f"mode={mode} drive={drive} turn={turn} secs={secs} kp={kp} kd={kd}")
        print(f"  displacement={disp:.3f} m  forward(along heading)={fwd_body:+.3f} m  side={dx*-np.sin(yaw0)+dy*np.cos(yaw0):+.3f} m")
        print(f"  yaw change={np.rad2deg(yaw1-yaw0):+.1f} deg  final height={H[-1]:.3f} (min {H.min():.3f})  fell={H.min()<H_MIN}")
    if video and frames:
        import imageio.v2 as imageio
        imageio.mimwrite(video, frames, fps=25)
        print(f"  video -> {video}")
    return dict(disp=disp, fwd=fwd_body, yaw=np.rad2deg(yaw1 - yaw0), height=H[-1], fell=bool(H.min() < H_MIN))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="walk")
    ap.add_argument("--drive", type=float, default=0.8)
    ap.add_argument("--turn", type=float, default=0.0)
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--kd", type=float, default=0.05)
    ap.add_argument("--stance", type=float, nargs=3, default=None)
    ap.add_argument("--video", default=None)
    a = ap.parse_args()
    run(a.mode, a.drive, a.turn, a.secs, None if a.stance is None else tuple(a.stance), a.kp, a.kd, a.video)
