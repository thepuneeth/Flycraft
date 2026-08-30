"""E4: close the frozen VNC through traced leg proprioceptor populations."""
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
XML = HERE.parent.parent.parent / "assets" / "hexapod" / "hexapod.xml"
sys.path.insert(0, str(HERE))
from audit_vnc_legs import leg_scores  # noqa: E402
from hexapod_gait import H_MIN, home_stance, pd_torque  # noqa: E402
from mn_to_joint import translate  # noqa: E402
from probe_structured_descending import DESC_KEYS, brain_type_indices, phase_metrics  # noqa: E402
from run_hexapod_chase import retina  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
STANCE = home_stance()
MAP_GAINS = (-0.55, 0.65, 0.65)
CTRL_HZ = 50.0
SECS = 8.0
INPUT_SCALE = 100.0
ANGLE_LIMIT_RAD = 2.35619
PROPRIO_SUBCLASSES = ("campaniform sensilla", "hair plate", "chordotonal organ", "leg")


def foot_loads(model, sim, floor_id: int, foot_ids: list[int], mass_per_leg: float) -> np.ndarray:
    normal_force = np.zeros(6, dtype=np.float64)
    force = np.zeros(6, dtype=np.float64)
    for contact_index in range(sim.ncon):
        geom1 = int(sim.contact[contact_index].geom1)
        geom2 = int(sim.contact[contact_index].geom2)
        if geom1 == floor_id:
            foot = geom2
        elif geom2 == floor_id:
            foot = geom1
        else:
            continue
        if foot not in foot_ids:
            continue
        leg = foot_ids.index(foot)
        mujoco.mj_contactForce(model, sim, contact_index, force)
        normal_force[leg] += max(0.0, float(force[0]))
    return np.clip(normal_force / mass_per_leg, 0.0, 1.0)


def sensory_current(vnc, io, qpos: np.ndarray, loads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    body_ids = io["node_body_ids"].astype(np.int64)
    sensory_indices = io["sensory_feedback"].astype(np.int64)
    sensory_body_ids = io["sensory_feedback_body_ids"].astype(np.int64)
    if not np.array_equal(body_ids[sensory_indices], sensory_body_ids):
        raise ValueError("sensory feedback index/body-id mismatch")
    legs = io["sensory_feedback_legs"].astype("U3")
    sides = io["sensory_feedback_sides"].astype("U1")
    subclasses = io["sensory_feedback_subclasses"].astype("U32")
    leg_index = {name: index for index, name in enumerate(LEG_NAMES)}
    angle_signal = np.clip(np.mean(np.abs(qpos.reshape(6, 3) - STANCE), axis=1) / ANGLE_LIMIT_RAD, 0.0, 1.0)
    current = np.zeros(vnc.n, dtype=np.float32)
    for index, (neuromere, side, subclass) in enumerate(zip(legs, sides, subclasses)):
        leg = leg_index[f"{neuromere}{side}"]
        signal = loads[leg] if subclass == "campaniform sensilla" else angle_signal[leg]
        current[sensory_indices[index]] = INPUT_SCALE * signal
    return current, angle_signal


def run(vnc_data: Path, brain_data: Path, weights_name: str, index_name: str) -> dict:
    vnc = VNCStepper(
        vnc_data,
        substeps=4,
        tau=0.02,
        dt=0.005,
        gW=0.03,
        weights_name=weights_name,
        index_name=index_name,
        model="paper",
        threshold=7.5,
        activation_gain=1.0,
        fr_cap=200.0,
        heterogeneity=True,
        seed=1,
    )
    io = vnc.io
    expected_subclasses = set(PROPRIO_SUBCLASSES)
    actual_subclasses = set(io["sensory_feedback_subclasses"].astype("U32"))
    if not actual_subclasses <= expected_subclasses:
        raise ValueError(f"unexpected sensory subclass: {sorted(actual_subclasses - expected_subclasses)}")

    with (brain_data / "weight_csr.pkl").open("rb") as handle:
        brain_weight = pickle.load(handle)
    if brain_weight.layout != torch.sparse_csr:
        brain_weight = brain_weight.to_sparse_csr()
    brain_weight = brain_weight.float()
    brain_io = np.load(brain_data / "io_index.npz")
    brain_n = int(brain_weight.shape[0])
    visual_left = torch.from_numpy(brain_io["vis_left"]).long()
    visual_right = torch.from_numpy(brain_io["vis_right"]).long()
    type_indices = brain_type_indices(brain_data)
    brain_state = torch.zeros(brain_n, dtype=torch.float32)
    brain_decay = 1.0 - 0.005 / 0.02
    brain_dt_tau = 0.005 / 0.02

    model = mujoco.MjModel.from_xml_path(str(XML))
    sim = mujoco.MjData(model)
    sim.qpos[2] = 0.15
    stance = translate(np.zeros(len(vnc.motor_out), dtype=np.float32), io, gains=MAP_GAINS)
    for _ in range(int(0.7 / model.opt.timestep)):
        sim.ctrl[:] = pd_torque(model, sim, stance, 3.0, 0.05)
        mujoco.mj_step(model, sim)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in range(6)]
    mass_per_leg = float(model.body_mass.sum() * 9.81 / 6.0)
    x0, y0 = float(sim.qpos[0]), float(sim.qpos[1])
    prey = (x0 + 1.2, y0 + 0.8)
    phys = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))

    motor_history, target_history, qpos_history = [], [], []
    body_history, load_history, angle_history, brain_history = [], [], [], {key: [] for key in DESC_KEYS}
    path = 0.0
    last = (x0, y0)
    for _ in range(int(SECS * CTRL_HZ)):
        trunk = (float(sim.qpos[0]), float(sim.qpos[1]))
        yaw = 2.0 * np.arctan2(sim.qpos[6], sim.qpos[3])
        present, az, distance, size = retina(trunk, yaw, prey)
        visual_mag = 1.2 * (0.5 + min(1.0, size))
        visual_current = torch.zeros(brain_n, dtype=torch.float32)
        if present:
            visual_current[visual_left] = visual_mag * (max(0.0, az) + 0.15)
            visual_current[visual_right] = visual_mag * (max(0.0, -az) + 0.15)
        with torch.no_grad():
            for _ in range(12):
                rates = torch.relu(torch.tanh(brain_state))
                brain_state.mul_(brain_decay).add_(brain_dt_tau * (0.006 * (brain_weight @ rates) + visual_current))
                brain_state.clamp_(-5.0, 5.0)
            rates = torch.relu(torch.tanh(brain_state))
        outputs = {key: float(rates[indices].mean()) for key, indices in type_indices.items()}
        for key, value in outputs.items():
            brain_history[key].append(value)

        loads = foot_loads(model, sim, floor_id, foot_ids, mass_per_leg)
        sensor_input, angle_signal = sensory_current(vnc, io, sim.qpos[7:25].copy(), loads)
        current = sensor_input
        for key in DESC_KEYS:
            current[vnc.io[key]] = INPUT_SCALE * outputs[key]
        motor = vnc.step(current).cpu().numpy()
        target = translate(motor, io, gains=MAP_GAINS)
        for _ in range(phys):
            sim.ctrl[:] = pd_torque(model, sim, target, 3.0, 0.05)
            mujoco.mj_step(model, sim)
        now = (float(sim.qpos[0]), float(sim.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        motor_history.append(motor)
        target_history.append(target)
        qpos_history.append(sim.qpos[7:25].copy())
        body_history.append(sim.qpos[:3].copy())
        load_history.append(loads)
        angle_history.append(angle_signal)

    motor_history = np.asarray(motor_history)
    target_history = np.asarray(target_history)
    qpos_history = np.asarray(qpos_history)
    body_history = np.asarray(body_history)
    loads = np.asarray(load_history)
    angles = np.asarray(angle_history)
    motor_phase = phase_metrics(leg_scores(motor_history, io)[..., 0], 0.02)
    joint_phase = phase_metrics(target_history.reshape(-1, 6, 3)[..., 0], 0.02)
    motor_phase["verdict"] = "E4-PHASE-PRESENT" if motor_phase["verdict"] == "E3-PRESENT" else "E4-PHASE-ABSENT"
    joint_phase["verdict"] = "E4-PHASE-PRESENT" if joint_phase["verdict"] == "E3-PRESENT" else "E4-PHASE-ABSENT"
    dx, dy = body_history[-1, 0] - x0, body_history[-1, 1] - y0
    min_height = float(body_history[:, 2].min())
    result = {
        "protocol": {
            "weights_name": weights_name,
            "index_name": index_name,
            "secs": SECS,
            "ctrl_hz": CTRL_HZ,
            "settle_s": 0.7,
            "body_servo_kp": 3.0,
            "body_servo_kd": 0.05,
            "stance": STANCE.tolist(),
            "motor_map_gains": list(MAP_GAINS),
            "brain_model": "bounded rate model",
            "brain_tau": 0.02,
            "brain_dt": 0.005,
            "brain_substeps": 12,
            "brain_gW": 0.006,
            "brain_drive_in": 1.2,
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
            "input_scale": INPUT_SCALE,
            "angle_normalization_rad": ANGLE_LIMIT_RAD,
            "sensory_rule": str(io["sensory_feedback_rule"]),
            "sensory_subclass_encoding": {
                "campaniform sensilla": "per-leg normalized foot normal load",
                "hair plate": "per-leg mean absolute joint-angle displacement from stance",
                "chordotonal organ": "per-leg mean absolute joint-angle displacement from stance",
                "leg": "per-leg mean absolute joint-angle displacement from stance",
            },
        },
        "sensory_feedback_cell_count": int(len(io["sensory_feedback"])),
        "sensory_feedback_counts_by_leg": {
            leg: int(np.sum((io["sensory_feedback_legs"].astype("U3") + io["sensory_feedback_sides"].astype("U1")) == leg))
            for leg in LEG_NAMES
        },
        "sensory_feedback_counts_by_subclass": {
            subclass: int(np.sum(io["sensory_feedback_subclasses"].astype("U32") == subclass))
            for subclass in PROPRIO_SUBCLASSES
        },
        "brain_descending_tail_mean": {key: float(np.mean(values[len(values) // 2:])) for key, values in brain_history.items()},
        "brain_descending_tail_std": {key: float(np.std(values[len(values) // 2:])) for key, values in brain_history.items()},
        "sensory_tail_mean": {"load": loads[len(loads) // 2:].mean(axis=0).tolist(), "angle_signal": angles[len(angles) // 2:].mean(axis=0).tolist()},
        "sensory_tail_std": {"load": loads[len(loads) // 2:].std(axis=0).tolist(), "angle_signal": angles[len(angles) // 2:].std(axis=0).tolist()},
        "vnc_motor_output": motor_phase,
        "joint_target_coxa": joint_phase,
        "locomotion": {
            "path_m": path,
            "displacement_m": float(np.hypot(dx, dy)),
            "forward_displacement_m": float(dx),
            "min_height_m": min_height,
            "final_height_m": float(body_history[-1, 2]),
            "fell": bool(min_height < H_MIN),
            "motor_temporal_std_mean": float(motor_history.std(axis=0).mean()),
            "target_temporal_std_mean_rad": float(target_history.std(axis=0).mean()),
            "joint_position_std_mean_rad": float(qpos_history.std(axis=0).mean()),
            "mean_load": float(loads.mean()),
        },
        "pivot_forced_by": "E4",
        "pivot_reason": "E0 remained absent after complete published coordination set; E3 structured descending drive and E4 fixed anatomical proprioceptive feedback both failed to produce the declared tripod phase pattern",
        "verdict": "E4-PRESENT" if motor_phase["verdict"] == "E4-PHASE-PRESENT" and joint_phase["verdict"] == "E4-PHASE-PRESENT" else "E4-ABSENT",
        "zero_learning": True,
    }
    assert motor_history.shape == (int(SECS * CTRL_HZ), len(vnc.motor_out))
    assert target_history.shape == (int(SECS * CTRL_HZ), 18)
    return result


def make_figure(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    for axis, label, key in zip(axes, ("VNC motor output", "joint-target coxa"), ("vnc_motor_output", "joint_target_coxa")):
        phase = np.asarray(result[key]["phase_difference_matrix_cycles"])
        image = axis.imshow(phase, vmin=-0.5, vmax=0.5, cmap="twilight")
        axis.set(xticks=range(6), xticklabels=LEG_NAMES, yticks=range(6), yticklabels=LEG_NAMES, title=f"{label}\n{result[key]['verdict']}")
        axis.tick_params(axis="x", rotation=35)
        fig.colorbar(image, ax=axis, label="left − right phase (cycles)")
    fig.suptitle("E4 proprioceptive sensory feedback — frozen encoding", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=RESULTS)
    ap.add_argument("--brain_data", type=Path, default=DATA)
    ap.add_argument("--weights_name", default="e4_sensory_weight_csr.pkl")
    ap.add_argument("--index_name", default="e4_sensory_io_index.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "e4_sensory_feedback")
    a = ap.parse_args()
    result = run(a.data, a.brain_data, a.weights_name, a.index_name)
    make_figure(result, a.out)
    a.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key in ("vnc_motor_output", "joint_target_coxa"):
        np.savetxt(a.out.with_name(a.out.name + "_" + key + ".csv"), np.asarray(result[key]["phase_difference_matrix_rad"]), delimiter=",", header=",".join(LEG_NAMES), comments="")
    print(json.dumps({key: result[key] for key in ("vnc_motor_output", "joint_target_coxa", "locomotion", "verdict")}, sort_keys=True))
    print(f"audit -> {a.out.with_suffix('.json')}")
    print(f"figure -> {a.out.with_suffix('.png')} / {a.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
