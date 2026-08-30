"""Localize Path-2 failure across VNC, fixed joint map, and body physics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from hexapod_gait import H_MIN, XML, home_stance, pd_torque  # noqa: E402
from mn_to_joint import translate  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


STANCE = home_stance().reshape(18).astype(np.float32)
MAP_GAINS = (-0.55, 0.65, 0.65)
TRIPOD_A = (0, 3, 4)
TRIPOD_B = (1, 2, 5)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def foot_contact_fraction(data, floor_id: int, foot_ids: set[int]) -> float:
    contacts = 0
    for index in range(data.ncon):
        geom1, geom2 = int(data.contact[index].geom1), int(data.contact[index].geom2)
        if (geom1 == floor_id and geom2 in foot_ids) or (geom2 == floor_id and geom1 in foot_ids):
            contacts += 1
    return contacts / len(foot_ids)


def run(data_dir: Path, weights_name: str, index_name: str, secs: float) -> dict:
    # These are the already-frozen open-loop Path-2 constants.
    ctrl_hz = 50.0
    kp, kd = 3.0, 0.05
    vnc = VNCStepper(
        data_dir,
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
    stance = translate(np.zeros(len(vnc.motor_out), dtype=np.float32), io, gains=MAP_GAINS)
    current = vnc.descending_current(100.0)

    model = mujoco.MjModel.from_xml_path(str(XML))
    sim = mujoco.MjData(model)
    sim.qpos[2] = 0.15
    phys = max(1, int(round((1.0 / ctrl_hz) / model.opt.timestep)))
    for _ in range(int(0.7 / model.opt.timestep)):
        sim.ctrl[:] = pd_torque(model, sim, stance, kp, kd)
        mujoco.mj_step(model, sim)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in range(6)}
    x0, y0 = float(sim.qpos[0]), float(sim.qpos[1])
    yaw0 = 2 * np.arctan2(sim.qpos[6], sim.qpos[3])
    n = int(secs * ctrl_hz)
    motor_history, target_history, q_history, ctrl_history = [], [], [], []
    body_history, foot_z_history, contact_history = [], [], []
    path = 0.0
    last = (x0, y0)
    for _ in range(n):
        motor = vnc.step(current).cpu().numpy()
        target = translate(motor, io, gains=MAP_GAINS)
        for _ in range(phys):
            sim.ctrl[:] = pd_torque(model, sim, target, kp, kd)
            mujoco.mj_step(model, sim)
        now = (float(sim.qpos[0]), float(sim.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        motor_history.append(motor)
        target_history.append(target)
        q_history.append(sim.qpos[7:25].copy())
        ctrl_history.append(sim.ctrl.copy())
        body_history.append(sim.qpos[:3].copy())
        foot_z_history.append([float(sim.geom_xpos[geom_id, 2]) for geom_id in sorted(foot_ids)])
        contact_history.append(foot_contact_fraction(sim, floor_id, foot_ids))

    motor = np.asarray(motor_history)
    targets = np.asarray(target_history)
    qpos = np.asarray(q_history)
    controls = np.asarray(ctrl_history)
    body = np.asarray(body_history)
    foot_z = np.asarray(foot_z_history)
    target_delta = targets - stance
    target_std = target_delta.std(axis=0)
    qpos_std = qpos.std(axis=0)
    target_q_corr = np.asarray([corr(target_delta[:, i], qpos[:, i]) for i in range(18)])
    result = {
        "constants": {
            "secs": secs,
            "ctrl_hz": ctrl_hz,
            "settle_s": 0.7,
            "kp": kp,
            "kd": kd,
            "stance": STANCE.tolist(),
            "motor_map_gains": list(MAP_GAINS),
            "vnc_model": "paper",
            "vnc_drive": 100.0,
            "vnc_gW": 0.03,
            "vnc_tau": 0.02,
            "vnc_dt": 0.005,
            "vnc_substeps": 4,
            "vnc_threshold": 7.5,
            "vnc_activation_gain": 1.0,
            "vnc_fr_cap": 200.0,
            "vnc_heterogeneity": True,
            "vnc_seed": 1,
        },
        "layer_i_vnc": {
            "motor_output_count": int(motor.shape[1]),
            "motor_temporal_std_mean": float(motor.std(axis=0).mean()),
            "motor_temporal_std_max": float(motor.std(axis=0).max()),
        },
        "layer_ii_fixed_motor_map": {
            "target_channel_count": int(targets.shape[1]),
            "active_target_channels": int(np.count_nonzero(target_std > 1e-9)),
            "target_temporal_std_mean_rad": float(target_std.mean()),
            "target_temporal_std_max_rad": float(target_std.max()),
            "target_range_mean_rad": float((targets.max(axis=0) - targets.min(axis=0)).mean()),
            "motor_to_target_is_fixed": True,
        },
        "layer_iii_hexapod_physics": {
            "joint_position_std_mean_rad": float(qpos_std.mean()),
            "joint_position_std_max_rad": float(qpos_std.max()),
            "target_joint_corr_mean": float(np.nanmean(target_q_corr)),
            "target_joint_corr_median": float(np.nanmedian(target_q_corr)),
            "target_tracking_mae_mean_rad": float(np.abs(targets - qpos).mean()),
            "control_temporal_std_mean": float(controls.std(axis=0).mean()),
            "body_path_m": path,
            "body_displacement_m": float(np.hypot(body[-1, 0] - x0, body[-1, 1] - y0)),
            "body_forward_displacement_m": float((body[-1, 0] - x0) * np.cos(yaw0) + (body[-1, 1] - y0) * np.sin(yaw0)),
            "min_body_height_m": float(body[:, 2].min()),
            "final_body_height_m": float(body[-1, 2]),
            "fell": bool(body[:, 2].min() < H_MIN),
            "mean_foot_floor_contact_fraction": float(np.mean(contact_history)),
            "foot_z_temporal_std_mean_m": float(foot_z.std(axis=0).mean()),
        },
        "target_joint_correlations": target_q_corr.tolist(),
        "coordination_audit": str(RESULTS / "all_legs_coordination.json"),
        "zero_learning": True,
    }
    assert targets.shape == (n, 18)
    assert np.count_nonzero(target_std > 1e-9) == 18
    return result | {"traces": {"target": targets, "qpos": qpos}}


def make_figure(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = result.pop("traces")
    target = traces["target"]
    qpos = traces["qpos"]
    window = min(len(target), 200)
    time_s = np.arange(window) / 50.0
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), layout="constrained")
    target_coxa = target[-window:].reshape(window, 6, 3)[:, :, 0]
    for leg in range(6):
        style = "-" if leg in TRIPOD_A else "--"
        axes[0, 0].plot(time_s, target_coxa[:, leg], color=colors[leg], linestyle=style, linewidth=1.1, label=f"leg {leg} ({'A' if leg in TRIPOD_A else 'B'})")
    axes[0, 0].set(xlabel="Time in final 4 s (s)", ylabel="Coxa target (rad)", title="Layer ii: fixed map carries VNC output")
    axes[0, 0].legend(ncol=2, fontsize=8, frameon=False)

    t = target[-window:]
    q = qpos[-window:]
    axes[0, 1].plot(time_s, t[:, 0], color="#0072B2", label="T1L coxa target")
    axes[0, 1].plot(time_s, q[:, 0], color="#0072B2", linestyle="--", label="T1L coxa actual")
    axes[0, 1].plot(time_s, t[:, 1], color="#D55E00", label="T1L femur target")
    axes[0, 1].plot(time_s, q[:, 1], color="#D55E00", linestyle="--", label="T1L femur actual")
    axes[0, 1].set(xlabel="Time in final 4 s (s)", ylabel="Joint angle (rad)", title="Layer iii: target-to-body transfer")
    axes[0, 1].legend(fontsize=8, frameon=False)

    phase = json.loads(Path(result["coordination_audit"]).read_text(encoding="utf-8"))["tripod_phase"]
    labels = ("Tripod A\nwithin", "Tripod B\nwithin", "Cross-tripod")
    values = (phase["tripod_A_within_mean"], phase["tripod_B_within_mean"], phase["cross_tripod_mean"])
    bars = axes[1, 0].bar(labels, values, color=colors[:3], edgecolor="black", hatch=("///", "\\\\", ".."))
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_ylim(-1.05, 1.05)
    axes[1, 0].set_ylabel("Mean coxa correlation")
    axes[1, 0].set_title("Layer i: no coherent alternating tripod")
    for bar, value in zip(bars, values):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value + (0.06 if value >= 0 else -0.12), f"{value:.2f}", ha="center", fontsize=9)

    names = ("VNC motor\nstd", "joint target\nstd (rad)", "joint angle\nstd (rad)")
    values = (result["layer_i_vnc"]["motor_temporal_std_mean"], result["layer_ii_fixed_motor_map"]["target_temporal_std_mean_rad"], result["layer_iii_hexapod_physics"]["joint_position_std_mean_rad"])
    bars = axes[1, 1].bar(names, values, color=("#56B4E9", "#009E73", "#E69F00"), edgecolor="black")
    axes[1, 1].set_ylabel("Temporal standard deviation")
    axes[1, 1].set_title("Signal exists through all three stages")
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Path 2 transfer-layer audit — frozen open-loop VNC drive", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    ap.add_argument("--index_name", default="all_legs_io_index.npz")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=RESULTS / "transfer_layers")
    a = ap.parse_args()
    result = run(a.data, a.weights_name, a.index_name, a.secs)
    make_figure(result, a.out)
    a.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("layer_i_vnc", "layer_ii_fixed_motor_map", "layer_iii_hexapod_physics")}, sort_keys=True))
    print(f"audit -> {a.out.with_suffix('.json')}")
    print(f"figure -> {a.out.with_suffix('.png')} / {a.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
