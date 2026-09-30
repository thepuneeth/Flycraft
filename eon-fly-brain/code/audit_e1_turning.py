"""E1: direct descending-drive asymmetry audit with matched controls."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
import specificity_campaign as SC  # noqa: E402
from audit_amplitude_control import amplitude_match, array_sha256  # noqa: E402


# `SC.ARM_SUFFIX` is empty for the rate core and "_lif" for the published one.
# Without it a run under SC_CORE=lif silently overwrites the recorded
# rate-core artifact of the same name, which is exactly what happened on the
# pod before this line existed -- only audit_amplitude_control had it.
PREFIX = RESULTS / f"e1_turning{SC.ARM_SUFFIX}"
SECS = 8.0
TURN_MAG = 0.5
TURNS = (-TURN_MAG, TURN_MAG)
METRICS = ("yaw_change_deg", "yaw_rate_deg_s", "path_m", "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m")
MATCHED_MODE = "per_channel_joint_target_sd_matched_to_real_at_same_turn"


def collect_turn_history(substrate, turn: float, cache: dict) -> np.ndarray:
    key = (substrate.name, substrate.seed, float(turn))
    if key not in cache:
        vnc = SC.substrate_vnc(substrate)
        current = vnc.descending_current(SC.BASE_DRIVE, turn=turn, turn_scale=1.0)
        steps = int(round(SECS / SC.SIGNAL_SAMPLE_DT))
        history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)]).astype(np.float32)
        if history.shape != (steps, len(vnc.io["motor_out"])) or not np.isfinite(history).all():
            raise AssertionError(f"invalid turn history for {key}")
        cache[key] = history
    return cache[key]


def run_episode(targets, io, spec, turn, mode, substrate, budget, model, body_record, video=None) -> dict:
    budget.take()
    data = mujoco.MjData(model)
    data.qpos[2] = 0.15
    yaw = float(spec["initial_yaw_rad"])
    data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
    physics_steps = max(1, int(round(SC.DT_S / model.opt.timestep)))
    stance = SC.translate(np.zeros(len(io["motor_out"]), dtype=np.float32), io, gains=SC.MAP_GAINS)
    if not np.allclose(stance.reshape(6, 3), SC.home_stance(model)):
        raise AssertionError("fixed map zero input is not model home stance")
    for _ in range(int(SC.SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = SC.pd_torque(model, data, stance, SC.KP, SC.KD)
        mujoco.mj_step(model, data)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    yaw0 = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
    last = (x0, y0)
    path = 0.0
    heights = []
    planted = []
    writer = None
    if video:
        import imageio.v2 as imageio
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video), fps=25, codec="libx264")
        cameras = SC.make_cameras(model)
        renderer = mujoco.Renderer(model, 480, 640)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in range(6))
    try:
        for step in range(int(round(SECS * SC.CTRL_HZ))):
            for _ in range(physics_steps):
                data.ctrl[:] = SC.pd_torque(model, data, targets[step], SC.KP, SC.KD)
                mujoco.mj_step(model, data)
            now = (float(data.qpos[0]), float(data.qpos[1]))
            path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
            last = now
            heights.append(float(data.qpos[2]))
            planted.append(SC.feet_planted_fraction(data, floor_id, foot_ids))
            if writer and step % 2 == 0:
                views = []
                for camera in cameras:
                    camera.lookat[:] = (float(data.qpos[0]), float(data.qpos[1]), 0.08)
                    renderer.update_scene(data, camera=camera)
                    views.append(renderer.render())
                writer.append_data(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))
    finally:
        if writer:
            writer.close()
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    yaw1 = 2.0 * np.arctan2(data.qpos[6], data.qpos[3])
    yaw_change = float(np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0)))
    return {
        **spec,
        "substrate": substrate.name,
        "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
        "seed": substrate.seed,
        "turn": float(turn),
        "amplitude_mode": mode,
        "tensor_sha256": substrate.tensor_hash,
        "yaw_change_deg": float(np.rad2deg(yaw_change)),
        "yaw_rate_deg_s": float(np.rad2deg(yaw_change) / SECS),
        "path_m": float(path),
        "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "mean_feet_planted": float(np.mean(planted)),
        "min_height_m": float(np.min(heights)),
        "final_height_m": float(heights[-1]),
        "fell": bool(np.min(heights) < SC.H_MIN),
        "body": body_record,
        "signal_domain": "joint_targets",
        "video": str(video) if video else None,
    }


def run_condition(substrate, turn, mode, motor_history, io, target_cache, specs, budget, rng, mapped=False):
    targets = motor_history if mapped else SC.target_history(motor_history, io, target_cache)
    model, body_record = SC.make_body_model()
    rows = []
    condition = f"{substrate.name}_{substrate.seed if substrate.seed is not None else 'real'}_{mode}_turn_{turn:+.1f}"
    for spec in specs:
        row = run_episode(targets, io, spec, turn, mode, substrate, budget, model, body_record)
        row["condition"] = condition
        rows.append(row)
    return {
        "condition": condition,
        "substrate": substrate.name,
        "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
        "seed": substrate.seed,
        "turn": float(turn),
        "amplitude_mode": mode,
        "signal_domain": "joint_targets" if mapped else "vnc_motor_output",
        "tensor_sha256": substrate.tensor_hash,
        "signal_sha256": array_sha256(motor_history),
        "summary": SC.summarize_rows(rows, METRICS, rng),
    }, rows


def response_record(positive: dict, negative: dict, rng: np.random.Generator) -> dict:
    pos = {row["episode"]: row for row in positive["rows"]}
    neg = {row["episode"]: row for row in negative["rows"]}
    if set(pos) != set(neg):
        raise AssertionError("turn directions do not share episode starts")
    values = [0.5 * (pos[index]["yaw_change_deg"] - neg[index]["yaw_change_deg"]) for index in sorted(pos)]
    return {
        "substrate": positive["record"]["substrate"],
        "kind": positive["record"]["kind"],
        "seed": positive["record"]["seed"],
        "amplitude_mode": positive["record"]["amplitude_mode"],
        "turn_response_definition": "half of positive-turn minus negative-turn yaw change, degrees",
        "turn_response_deg": SC.metric_summary(values, rng),
        "paired_episode_values_deg": values,
        "positive_turn_condition": positive["record"]["condition"],
        "negative_turn_condition": negative["record"]["condition"],
        "falls_any_direction": int(sum(bool(pos[index]["fell"] or neg[index]["fell"]) for index in pos)),
    }


def write_csv(rows: list[dict]) -> None:
    fields = (
        "condition", "substrate", "kind", "seed", "turn", "amplitude_mode", "signal_domain", "tensor_sha256",
        "episode", "episode_seed", "initial_yaw_rad", "yaw_change_deg", "yaw_rate_deg_s", "path_m",
        "forward_distance_m", "displacement_m", "mean_feet_planted", "min_height_m", "final_height_m", "fell",
    )
    with PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def mark_videos_reviewed() -> int:
    path = PREFIX.with_suffix(".json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["videos"]["reviewed"] = True
    report["videos"]["review_note"] = "real and median matched-random positive-turn four-view sequences visually reviewed"
    report["decision"] = report["numeric_decision"]
    SC.atomic_json(path, report)
    write_csv(report["rows"])
    print(json.dumps({"decision": report["decision"], "videos_reviewed": True}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mark-videos-reviewed", action="store_true")
    args = parser.parse_args()
    if args.mark_videos_reviewed:
        return mark_videos_reviewed()
    if PREFIX.with_suffix(".json").exists():
        raise FileExistsError(f"{PREFIX.with_suffix('.json')} exists")

    file_hashes_pre = SC.file_hashes(SC.VNC_DATA)
    substrates, substrate_record = SC.build_vnc_substrates(SC.VNC_DATA, SC.SUBSTRATE_CACHE)
    io = SC.base_vnc(SC.VNC_DATA).io
    specs = SC.episode_specs(SC.N_EPISODES, SC.EPISODE_SEED)
    history_cache = {}
    target_cache = {}
    summary_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    comparison_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    budget = SC.Budget()
    records = []
    rows = []
    for substrate in substrates:
        for turn in TURNS:
            motor_history = collect_turn_history(substrate, turn, history_cache)
            raw_record, raw_rows = run_condition(substrate, turn, "raw", motor_history, io, target_cache, specs, budget, summary_rng)
            records.append(raw_record)
            rows.extend(raw_rows)
            if substrate.name != "real":
                source_targets = SC.target_history(motor_history, io, target_cache)
                real_history = collect_turn_history(next(item for item in substrates if item.name == "real"), turn, history_cache)
                real_targets = SC.target_history(real_history, io, target_cache)
                matched_targets, match_record = amplitude_match(source_targets, real_targets.std(axis=0))
                match_record["source_motor_history_sha256"] = array_sha256(motor_history)
                match_record["source_joint_target_history_sha256"] = array_sha256(source_targets)
                match_record["reference_turn"] = float(turn)
                matched_record, matched_rows = run_condition(substrate, turn, MATCHED_MODE, matched_targets, io, target_cache, specs, budget, summary_rng, mapped=True)
                matched_record["amplitude_match"] = match_record
                records.append(matched_record)
                rows.extend(matched_rows)
        print(f"completed {substrate.name}", flush=True)

    def get_record(name, seed, turn, mode):
        return next(row for row in records if row["substrate"] == name and row["seed"] == seed and row["turn"] == float(turn) and row["amplitude_mode"] == mode)

    response_records = []
    for substrate in substrates:
        for mode in ("raw", MATCHED_MODE) if substrate.name != "real" else ("raw",):
            positive = get_record(substrate.name, substrate.seed, TURN_MAG, mode)
            negative = get_record(substrate.name, substrate.seed, -TURN_MAG, mode)
            response_records.append({"record": response_record({"record": positive, "rows": [row for row in rows if row["condition"] == positive["condition"]]}, {"record": negative, "rows": [row for row in rows if row["condition"] == negative["condition"]]}, summary_rng), "positive": positive, "negative": negative})

    real_response = next(row["record"] for row in response_records if row["record"]["substrate"] == "real")
    matched_random_responses = [row["record"] for row in response_records if row["record"]["substrate"] == "random" and row["record"]["amplitude_mode"] == MATCHED_MODE]
    raw_random_responses = [row["record"] for row in response_records if row["record"]["substrate"] == "random" and row["record"]["amplitude_mode"] == "raw"]
    matched_rewired_responses = [row["record"] for row in response_records if row["record"]["substrate"] == "rewired" and row["record"]["amplitude_mode"] == MATCHED_MODE]
    raw_rewired_responses = [row["record"] for row in response_records if row["record"]["substrate"] == "rewired" and row["record"]["amplitude_mode"] == "raw"]
    real_value = real_response["turn_response_deg"]["mean"]
    real_ci = (real_response["turn_response_deg"]["ci_low"], real_response["turn_response_deg"]["ci_high"])
    primary = SC.substrate_comparison(real_value, [row["turn_response_deg"]["mean"] for row in matched_random_responses], "turn_response_deg", comparison_rng, real_ci)
    primary["definition"] = "real minus same-turn joint-target-amplitude-matched random odd yaw response"
    primary["p_holm"] = float(SC.holm_adjust([primary["p_raw"]])[0])
    references = {
        "raw_random": SC.substrate_comparison(real_value, [row["turn_response_deg"]["mean"] for row in raw_random_responses], "raw_random_turn_response_deg", comparison_rng, real_ci),
        "matched_rewired": SC.substrate_comparison(real_value, [row["turn_response_deg"]["mean"] for row in matched_rewired_responses], "matched_rewired_turn_response_deg", comparison_rng, real_ci),
        "raw_rewired": SC.substrate_comparison(real_value, [row["turn_response_deg"]["mean"] for row in raw_rewired_responses], "raw_rewired_turn_response_deg", comparison_rng, real_ci),
    }
    numeric_gate = bool(primary["p_holm"] < 0.05 and primary["real_outside_random_distribution"] and primary["nonoverlap_bootstrap_ci"] and real_value > 0.0)
    matched_records = [row for row in records if row["amplitude_mode"] == MATCHED_MODE]
    max_sd_error = max(row["amplitude_match"]["max_sd_error_all_channels"] for row in matched_records)
    max_mean_error = max(row["amplitude_match"]["mean_preserved_max_error"] for row in matched_records)

    median_seed = sorted(matched_random_responses, key=lambda row: abs(row["turn_response_deg"]["mean"] - np.median([item["turn_response_deg"]["mean"] for item in matched_random_responses])))[0]["seed"]
    real = next(item for item in substrates if item.name == "real")
    random_median = next(item for item in substrates if item.name == "random" and item.seed == median_seed)
    real_turn_history = collect_turn_history(real, TURN_MAG, history_cache)
    random_turn_history = collect_turn_history(random_median, TURN_MAG, history_cache)
    real_signal = SC.target_history(real_turn_history, io, target_cache)
    random_signal = amplitude_match(SC.target_history(random_turn_history, io, target_cache), real_signal.std(axis=0))[0]
    video_paths = {
        "real": RESULTS / "e1_turning_real_positive_multiview.mp4",
        "random_median": RESULTS / "e1_turning_matched_random_positive_multiview.mp4",
    }
    for substrate, signal, path in ((real, real_signal, video_paths["real"]), (random_median, random_signal, video_paths["random_median"])):
        model, body_record = SC.make_body_model()
        run_episode(signal, io, specs[0], TURN_MAG, "raw" if substrate.name == "real" else MATCHED_MODE, substrate, budget, model, body_record, video=path)

    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    response_records_json = [row["record"] for row in response_records]
    report = {
        "experiment": "E1",
        "status": "complete",
        "pre_registered_null": "real odd yaw response ties matched-random; TURNING-NULL",
        "protocol": {
            "secs": SECS,
            "drive": SC.BASE_DRIVE,
            "turns": list(TURNS),
            "turn_scale": 1.0,
            "n_episodes": SC.N_EPISODES,
            "episode_seed": SC.EPISODE_SEED,
            "shared_episode_starts": True,
            "primary": "odd yaw response = half of positive-turn minus negative-turn yaw change",
            "amplitude_rule": "per-channel temporal SD of fixed-map joint targets set to real joint-target SD at same turn",
            "controls": ["raw random-network", MATCHED_MODE, "raw rewired", "per-channel-SD amplitude-matched rewired"],
            "learning": False,
        },
        "real_graph_hashes": file_hashes_pre,
        "substrates": substrate_record,
        "conditions": records,
        "turn_response_records": response_records_json,
        "primary": primary,
        "reference_comparisons": references,
        "videos": {
            "real": {"path": str(video_paths["real"]), "exists": video_paths["real"].exists(), "turn": TURN_MAG},
            "random_median": {"path": str(video_paths["random_median"]), "exists": video_paths["random_median"].exists(), "seed": int(median_seed), "turn": TURN_MAG},
            "reviewed": False,
            "review_note": "review required before final decision",
        },
        "integrity": {
            "file_hashes_pre": file_hashes_pre,
            "file_hashes_post": file_hashes_post,
            "file_hashes_equal": bool(file_hashes_pre == file_hashes_post),
            "real_tensor_sha256_pre": real.tensor_hash,
            "real_tensor_sha256_post": SC.tensor_sha256(real.weight),
            "real_tensor_hash_equal": bool(real.tensor_hash == SC.tensor_sha256(real.weight)),
            "random_seeds": list(SC.RNG_SEEDS),
            "rewired_seeds": list(SC.RNG_SEEDS),
            "condition_count": len(records),
            "row_count": len(rows),
            "amplitude_match_count": len(matched_records),
            "max_sd_error_all_channels": max_sd_error,
            "max_mean_preserved_error": max_mean_error,
            "all_channels_match": bool(max_sd_error <= 1e-4),
            "learning": False,
        },
        "resource": {"rollouts_completed": budget.rollouts, "max_rollouts": SC.MAX_ROLLOUTS, "wallclock_ceiling_s": SC.WALLCLOCK_CEILING_S},
        "zero_learning": True,
        "zero_learning_audit": {"learning": False, "parameter_updates": 0, "training_markers": []},
        "numeric_decision": "TURNING-EARNED" if numeric_gate else "TURNING-NULL",
        "decision": "TURNING-PENDING-VIDEO" if numeric_gate else "TURNING-NULL",
        "rows": rows,
    }
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    write_csv(rows)
    print(json.dumps({"experiment": "E1", "primary": primary, "numeric_decision": report["numeric_decision"], "resource": report["resource"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
