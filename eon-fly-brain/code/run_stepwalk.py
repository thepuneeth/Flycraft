"""Path 3: frozen VNC step timing into the fixed Path-1 gait engine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from gait_engine import STANCE, StepGaitEngine, clean_tripod_states  # noqa: E402
from probe_vnc_phase import phase_metrics  # noqa: E402
from run_hexapod_chase import retina  # noqa: E402
from run_reservoir_walk import FrozenBrain  # noqa: E402
from step_decode import MatchedRandomTiming, decode_vector, step_statistics  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402
from hexapod_gait import HexapodGait, XML, pd_torque  # noqa: E402


CTRL_HZ = 50.0
DT_S = 1.0 / CTRL_HZ
SETTLE_S = 0.7
KP = 3.0
KD = 0.05
VNC_DRIVE = 100.0
VNC_TURN_SCALE = 6.0
H_MIN = 0.02
CATCH_DISTANCE = 0.25
VNC_WEIGHTS = "all_legs_weight_csr.pkl"
VNC_INDEX = "all_legs_io_index.npz"


def run_clean_engine(secs: float = 8.0) -> dict:
    """P2 ceiling: a clean external tripod clock drives the same fixed engine."""
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    engine = StepGaitEngine(stance=tuple(STANCE))
    clean = HexapodGait(stance=tuple(STANCE))
    forward, turn = 0.8, 0.0
    physics_steps = max(1, int(round(DT_S / model.opt.timestep)))
    stance = np.tile(STANCE, 6).astype(np.float32)
    for _ in range(int(SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = pd_torque(model, data, stance, KP, KD)
        mujoco.mj_step(model, data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    yaw0 = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
    phase = 0.0
    targets = []
    heights = []
    path = 0.0
    last = (x0, y0)
    frequency = clean.base_freq + clean.drive_freq * forward
    for _ in range(int(round(secs * CTRL_HZ))):
        phase = (phase + 2.0 * np.pi * frequency * DT_S) % (2.0 * np.pi)
        target = engine.targets(clean_tripod_states(phase), forward, turn, DT_S)
        targets.append(target)
        for _ in range(physics_steps):
            data.ctrl[:] = pd_torque(model, data, target, KP, KD)
            mujoco.mj_step(model, data)
        now = (float(data.qpos[0]), float(data.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        heights.append(float(data.qpos[2]))
    targets = np.asarray(targets, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    phase_result = phase_metrics(targets.reshape(-1, 6, 3)[..., 0], DT_S, "P2-ENGINE-PRESENT", "P2-ENGINE-ABSENT")
    return {
        "protocol": {
            "secs": secs,
            "ctrl_hz": CTRL_HZ,
            "forward": forward,
            "turn": turn,
            "external_clock": "clean alternating tripod at Path-1 HexapodGait rate",
            "gait_engine": "StepGaitEngine over unchanged HexapodGait primitive",
        },
        "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "path_m": float(path),
        "min_height_m": float(heights.min()),
        "final_height_m": float(heights[-1]),
        "fell": bool(heights.min() < H_MIN),
        "tripod_coherence_scalar": float(phase_result["tripod_coherence_scalar"]),
        "phase": phase_result,
    }


def reference_timing(reference: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = reference.get("real", reference)["step_statistics"]["per_leg"]
    frequencies = np.asarray([row["step_frequency_hz"] for row in rows], dtype=np.float64)
    duties = np.asarray([row["duty_factor_swing"] for row in rows], dtype=np.float64)
    if frequencies.shape != (6,) or duties.shape != (6,) or not np.isfinite(frequencies).all() or not np.isfinite(duties).all():
        raise ValueError("invalid P1 timing reference")
    if np.any(frequencies <= 0.0) or np.any((duties <= 0.0) | (duties >= 1.0)):
        raise ValueError("P1 timing reference must have positive frequency and interior duty factor")
    return frequencies, duties


class StepWalkContext:
    def __init__(self, condition: str, vnc_data: Path = RESULTS, brain_data: Path = DATA):
        if condition not in {"real", "shuffled", "random"}:
            raise ValueError(f"unknown timing condition {condition!r}")
        self.condition = condition
        self.brain_data = Path(brain_data)
        self.vnc = VNCStepper(
            vnc_data,
            substeps=4,
            tau=0.02,
            dt=0.005,
            gW=0.03,
            weights_name=VNC_WEIGHTS,
            index_name=VNC_INDEX,
            model="paper",
            threshold=7.5,
            activation_gain=1.0,
            fr_cap=200.0,
            heterogeneity=True,
            seed=1,
            shuffle=condition == "shuffled",
        )
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.physics_steps = max(1, int(round(DT_S / self.model.opt.timestep)))


def run_episode(
    condition: str,
    seed: int,
    context: StepWalkContext | None = None,
    secs: float = 8.0,
    prey_offset: tuple[float, float] = (1.2, 0.8),
    initial_yaw: float = 0.0,
    timing_reference: dict | None = None,
    video: Path | None = None,
) -> dict:
    context = context or StepWalkContext(condition)
    if context.condition != condition:
        raise ValueError("context condition does not match episode condition")
    context.vnc.reset()
    brain = FrozenBrain(context.brain_data)
    engine = StepGaitEngine(stance=tuple(STANCE))
    random_timing = None
    if condition == "random":
        if timing_reference is None:
            raise ValueError("random timing requires the frozen P1 reference")
        frequency, duty = reference_timing(timing_reference)
        random_timing = MatchedRandomTiming(frequency, duty, seed)

    data = mujoco.MjData(context.model)
    data.qpos[2] = 0.15
    data.qpos[3:7] = (np.cos(initial_yaw / 2.0), 0.0, 0.0, np.sin(initial_yaw / 2.0))
    stance = np.tile(STANCE, 6).astype(np.float32)
    for _ in range(int(SETTLE_S / context.model.opt.timestep)):
        data.ctrl[:] = pd_torque(context.model, data, stance, KP, KD)
        mujoco.mj_step(context.model, data)

    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    prey = (x0 + float(prey_offset[0]), y0 + float(prey_offset[1]))
    min_distance = float(np.hypot(*prey_offset))
    last = (x0, y0)
    path = 0.0
    caught = False
    targets = []
    states = []
    phases = []
    motors = []
    heights = []
    renderer = mujoco.Renderer(context.model, 480, 640) if video else None
    cameras = []
    if renderer:
        for azimuth, elevation, distance in ((135.0, 25.0, 1.4), (0.0, 80.0, 1.8), (90.0, 25.0, 1.4), (180.0, 25.0, 1.4)):
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(context.model, camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
            cameras.append(camera)
    frames = []

    for step in range(int(round(secs * CTRL_HZ))):
        trunk = (float(data.qpos[0]), float(data.qpos[1]))
        yaw = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
        present, azimuth, distance, size = retina(trunk, yaw, prey)
        forward, turn = brain.step(present, azimuth, size)
        if condition == "random":
            state = random_timing.step(DT_S)
        else:
            current = context.vnc.descending_current(forward * VNC_DRIVE, turn, dnb08_gain=1.0, turn_scale=VNC_TURN_SCALE)
            motor = context.vnc.step(current).cpu().numpy()
            state, _, _ = decode_vector(motor, context.vnc.io)
            motors.append(motor)
        target = engine.targets(state, forward, turn, DT_S)
        targets.append(target)
        states.append(state)
        phases.append(engine.phases.copy())
        for _ in range(context.physics_steps):
            data.ctrl[:] = pd_torque(context.model, data, target, KP, KD)
            mujoco.mj_step(context.model, data)
        now = (float(data.qpos[0]), float(data.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        min_distance = min(min_distance, distance)
        caught |= distance < CATCH_DISTANCE
        heights.append(float(data.qpos[2]))
        if renderer and step % 2 == 0:
            views = []
            for camera in cameras:
                camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.08)
                renderer.update_scene(data, camera=camera)
                views.append(renderer.render())
            frames.append(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))

    targets = np.asarray(targets, dtype=np.float64)
    states = np.asarray(states, dtype=bool)
    phases = np.asarray(phases, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    realized_phase = phase_metrics(targets.reshape(-1, 6, 3)[..., 0], DT_S, "P3-PRESENT", "P3-ABSENT")
    decoded_phase = phase_metrics(states.astype(np.float64), DT_S, "P1-PRESENT", "P1-ABSENT")
    fell = bool(heights.min() < H_MIN)
    result = {
        "condition": condition,
        "seed": int(seed),
        "secs": float(secs),
        "initial_yaw_rad": float(initial_yaw),
        "prey_offset_m": [float(prey_offset[0]), float(prey_offset[1])],
        "forward_distance_m": float(dx * np.cos(initial_yaw) + dy * np.sin(initial_yaw)),
        "displacement_m": float(np.hypot(dx, dy)),
        "path_m": float(path),
        "min_distance_m": float(min_distance),
        "min_height_m": float(heights.min()),
        "final_height_m": float(heights[-1]),
        "fell": fell,
        "upright": bool(not fell),
        "caught": bool(caught),
        "tripod_coherence_scalar": float(realized_phase["tripod_coherence_scalar"]),
        "realized_phase": realized_phase,
        "decoded_state_phase": decoded_phase,
        "step_statistics": step_statistics(states, DT_S),
        "engine_phase_temporal_std_mean": float(phases.std(axis=0).mean()),
        "target_temporal_std_mean_rad": float(targets.std(axis=0).mean()),
        "motor_temporal_std_mean": float(np.asarray(motors).std(axis=0).mean()) if motors else None,
        "timing_reference_used": condition == "random",
        "vnc_timing_used": condition != "random",
        "protocol": {
            "ctrl_hz": CTRL_HZ,
            "settle_s": SETTLE_S,
            "kp": KP,
            "kd": KD,
            "vnc_drive": VNC_DRIVE,
            "vnc_turn_scale": VNC_TURN_SCALE,
            "vnc_model": "paper",
            "vnc_tau": 0.02,
            "vnc_dt": 0.005,
            "vnc_substeps": 4,
            "vnc_gW": 0.03,
            "vnc_threshold": 7.5,
            "vnc_activation_gain": 1.0,
            "vnc_fr_cap": 200.0,
            "vnc_heterogeneity": True,
            "vnc_seed": 1,
            "vnc_shuffle": condition == "shuffled",
            "vnc_shuffle_seed": 1,
            "decode": "mean trochanter flexor/Acc. tr flexor > mean trochanter extensor; ties stance",
            "gait_engine": "fixed Path-1 HexapodGait primitive",
            "steering": "FrozenBrain from Path-1 run_vnc_chase forward/turn pass",
            "learning": False,
        },
    }
    if renderer:
        import imageio.v2 as imageio
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, frames, fps=25)
        result["video"] = str(video)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("real", "shuffled", "random"), default="real")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--vnc_data", type=Path, default=RESULTS)
    parser.add_argument("--secs", type=float, default=8.0)
    parser.add_argument("--prey", type=float, nargs=2, default=[1.2, 0.8])
    parser.add_argument("--initial_yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--timing_reference", type=Path, default=RESULTS / "step_decode_p1.json")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--engine_only", action="store_true")
    args = parser.parse_args()
    if args.engine_only:
        result = run_clean_engine(args.secs)
        output = args.out or RESULTS / "gait_engine_sanity.json"
        output = output if output.suffix == ".json" else output.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("forward_distance_m", "path_m", "fell", "tripod_coherence_scalar")}, sort_keys=True))
        print(f"engine -> {output}")
        return
    reference = json.loads(args.timing_reference.read_text(encoding="utf-8")) if args.condition == "random" else None
    context = StepWalkContext(args.condition, args.vnc_data, args.data)
    result = run_episode(args.condition, args.seed, context, args.secs, tuple(args.prey), args.initial_yaw, reference, args.video)
    output = args.out or RESULTS / f"stepwalk_metrics_{args.condition}_episode.json"
    output = output if output.suffix == ".json" else output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("condition", "forward_distance_m", "path_m", "fell", "tripod_coherence_scalar")}, sort_keys=True))
    print(f"metrics -> {output}")


if __name__ == "__main__":
    main()
