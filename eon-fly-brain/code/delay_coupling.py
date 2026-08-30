"""DC-1 delayed phase coupling checks; later phases run from this module."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from last_three_jobs import (  # noqa: E402
    BODY_STEPS,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CLEAN_MAX_PHASE_STEP_RAD,
    CLEAN_MAX_STEP_CHANGE_RAD,
    CONDITIONS,
    CTRL_HZ,
    GAIN_CANDIDATES,
    PLANNED_N,
    PLANNED_SECS,
    SUMMARY_METRICS,
    build_coupling_matrices,
    coupled_phases,
    delayed_coupled_phases,
    graph_record,
    run_engine,
    summarize,
    comparisons,
    vnc_phase_source,
)
from stepwalk_controls import episode_specs  # noqa: E402


CLOCK_HZ = 0.8 + 0.5 * 0.8
DC2_MAX_PER_HOP_STEPS = int(round(CTRL_HZ / (2.0 * CLOCK_HZ)))
DC2_RANDOM_PHASE_SEED = 20260830
DC2_PHASE_LOCK_VARIANCE_RAD2 = 0.25
DC2_SELECTION_RULE = (
    "Among admissible cells, select the one with the smallest per_hop_steps, "
    "breaking ties by the largest gain. Selection uses only the neutral "
    "phase-locked and bounded criteria across all three conditions."
)
DELAY_NULL = (
    "real delayed coupling does not produce a walking anti-phase tripod beating shuffled/random "
    "on coherence and feet-planted -> DELAY-COUPLING-NULL; the zero-training coordination ceiling holds"
)
DELAY_POSITIVE = (
    "real edges + the declared delay produce a stable anti-phase tripod that walks, with low falls and "
    "positive forward path, beat both controls on coherence and feet-planted with Holm-corrected p < 0.05 "
    "and non-overlapping bootstrap CIs, and the watched real video shows a genuine tripod walk -> "
    "DELAY-COUPLING-EARNED"
)
DC3_VIDEO_VERDICT_PENDING = "PENDING_MANUAL_WATCH"


def anatomical_delay_matrix(per_hop_steps: int) -> np.ndarray:
    if int(per_hop_steps) != per_hop_steps or per_hop_steps < 0:
        raise ValueError("per_hop_steps must be a non-negative integer")
    segments = np.arange(6) // 2
    return np.rint(int(per_hop_steps) * np.abs(segments[:, None] - segments[None, :])).astype(int)


def dc1_self_check(data: Path) -> None:
    matrices = build_coupling_matrices(data)["matrices"]
    rng = np.random.default_rng(DC2_RANDOM_PHASE_SEED)
    source = rng.normal(size=(32, 6)).cumsum(axis=0)
    expected = coupled_phases(source, matrices["real"], GAIN_CANDIDATES[7])
    actual = delayed_coupled_phases(source, matrices["real"], GAIN_CANDIDATES[7], 0)
    assert np.allclose(actual, expected)

    neutral = np.arange(BODY_STEPS, dtype=np.float64)[:, None] * (2.0 * np.pi * CLOCK_HZ / CTRL_HZ)
    neutral = np.broadcast_to(neutral, (BODY_STEPS, 6)).copy()
    neutral += rng.uniform(0.0, 2.0 * np.pi, size=6)
    for per_hop_steps in range(DC2_MAX_PER_HOP_STEPS + 1):
        delay = anatomical_delay_matrix(per_hop_steps)
        result = delayed_coupled_phases(neutral, matrices["real"], GAIN_CANDIDATES[-1], delay)
        steps = np.diff(result, axis=0)
        assert np.isfinite(result).all()
        assert np.abs(steps).max() <= CLEAN_MAX_PHASE_STEP_RAD
        assert np.abs(np.diff(steps, axis=0)).max() <= CLEAN_MAX_STEP_CHANGE_RAD

    expected_delay = np.asarray(
        [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [1, 1, 0, 0, 1, 1],
         [1, 1, 0, 0, 1, 1], [2, 2, 1, 1, 0, 0], [2, 2, 1, 1, 0, 0]],
        dtype=int,
    )
    assert np.array_equal(anatomical_delay_matrix(1), expected_delay)
    print("DC-1 regression: PASS")
    print("DC-1 warmup boundedness: PASS")
    print("DC-1 anatomical delay matrix (per_hop_steps=1):")
    print(anatomical_delay_matrix(1))
    print("GATE DC-1: PASS")


def neutral_phase_source() -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(DC2_RANDOM_PHASE_SEED)
    offsets = rng.uniform(0.0, 2.0 * np.pi, size=6)
    phase = np.arange(BODY_STEPS, dtype=np.float64)[:, None] * (2.0 * np.pi * CLOCK_HZ / CTRL_HZ)
    phase = np.broadcast_to(phase, (BODY_STEPS, 6)).copy()
    phase += offsets
    return phase, {
        "definition": "neutral common-frequency clock with seeded independent initial phases; no tripod offsets",
        "frequency_hz": CLOCK_HZ,
        "steps": BODY_STEPS,
        "ctrl_hz": CTRL_HZ,
        "random_phase_seed": DC2_RANDOM_PHASE_SEED,
        "initial_phase_rad": offsets.tolist(),
    }


def neutral_convergence(result: np.ndarray) -> dict:
    pair_difference = result[:, :, None] - result[:, None, :]
    tail = pair_difference[-max(1, len(result) // 3):]
    variance = np.var(tail, axis=0)
    upper = variance[np.triu_indices(6, 1)]
    steps = np.diff(result, axis=0)
    step_changes = np.diff(steps, axis=0)
    max_phase_step = float(np.abs(steps).max())
    max_step_change = float(np.abs(step_changes).max())
    max_pair_variance = float(upper.max())
    phase_locked = max_pair_variance < DC2_PHASE_LOCK_VARIANCE_RAD2
    bounded = max_phase_step <= CLEAN_MAX_PHASE_STEP_RAD and max_step_change <= CLEAN_MAX_STEP_CHANGE_RAD
    return {
        "max_pair_phase_difference_variance_rad2": max_pair_variance,
        "phase_lock_variance_threshold_rad2": DC2_PHASE_LOCK_VARIANCE_RAD2,
        "phase_locked": bool(phase_locked),
        "max_phase_step_rad": max_phase_step,
        "max_step_change_rad": max_step_change,
        "bounded": bool(bounded),
        "admissible": bool(phase_locked and bounded),
    }


def blind_calibration(matrices: dict[str, np.ndarray]) -> dict:
    source, source_record = neutral_phase_source()
    table = []
    for per_hop_steps in range(DC2_MAX_PER_HOP_STEPS + 1):
        delay = anatomical_delay_matrix(per_hop_steps)
        for gain in GAIN_CANDIDATES:
            conditions = {
                condition: neutral_convergence(delayed_coupled_phases(source, matrices[condition], gain, delay))
                for condition in ("real", "shuffled", "random")
            }
            table.append({
                "gain": float(gain),
                "per_hop_steps": per_hop_steps,
                "delay_matrix": delay.tolist(),
                "conditions": conditions,
                "admissible_all_conditions": bool(all(item["admissible"] for item in conditions.values())),
            })
    admissible = [row for row in table if row["admissible_all_conditions"]]
    selected = min(admissible, key=lambda row: (row["per_hop_steps"], -row["gain"])) if admissible else None
    return {
        "candidate_gains": list(GAIN_CANDIDATES),
        "candidate_per_hop_steps": list(range(DC2_MAX_PER_HOP_STEPS + 1)),
        "half_period_ceiling": {
            "clock_frequency_hz": CLOCK_HZ,
            "half_period_steps": CTRL_HZ / (2.0 * CLOCK_HZ),
            "max_per_hop_steps": DC2_MAX_PER_HOP_STEPS,
            "rule": "search through the nearest integer control-step half-period; no larger delay searched",
        },
        "source": source_record,
        "criterion": {
            "phase_locked": "maximum ordinary variance of every pairwise phase difference over final third is below threshold",
            "bounded": "maximum absolute phase step and step change reuse existing clean thresholds",
            "phase_lock_variance_threshold_rad2": DC2_PHASE_LOCK_VARIANCE_RAD2,
            "maximum_phase_step_rad": CLEAN_MAX_PHASE_STEP_RAD,
            "maximum_step_change_rad": CLEAN_MAX_STEP_CHANGE_RAD,
            "all_three_conditions_required": True,
        },
        "selection_rule": DC2_SELECTION_RULE,
        "conditions": ["real", "shuffled", "random"],
        "table": table,
        "admissible_cell_count": len(admissible),
        "selected": selected,
    }


def write_calibration(output: Path, matrices: dict, calibration: dict) -> None:
    result = {
        "method": "DC-2 connectome-blind delayed coupling calibration",
        "coupling": matrices,
        "calibration": calibration,
        "zero_learning": True,
        "gate_dc2": {
            "admissibility_table_complete": True,
            "selection_uses_only_neutral_criterion": True,
            "verdict": "GATE DC-2: PASS" if calibration["selected"] else "DC-2 NULL: no connectome-blind delayed regime",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = ("per_hop_steps", "gain", "admissible_all_conditions", "real_admissible", "shuffled_admissible", "random_admissible")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({
            "per_hop_steps": row["per_hop_steps"],
            "gain": row["gain"],
            "admissible_all_conditions": row["admissible_all_conditions"],
            **{f"{condition}_admissible": row["conditions"][condition]["admissible"] for condition in ("real", "shuffled", "random")},
        } for row in calibration["table"])


def run_dc2(data: Path, output: Path) -> int:
    matrices = build_coupling_matrices(data)
    calibration = blind_calibration(matrices["matrices"])
    write_calibration(output, matrices["record"], calibration)
    print(f"PRE-REGISTERED DC-2 SELECTION RULE: {DC2_SELECTION_RULE}")
    print(f"DC-2 admissible cells: {calibration['admissible_cell_count']}/{len(calibration['table'])}")
    print("DC-2 admissibility table:")
    for row in calibration["table"]:
        print(row["per_hop_steps"], row["gain"], row["admissible_all_conditions"], flush=True)
    if calibration["selected"] is None:
        print("DC-2 NULL: no connectome-blind delayed regime")
        return 0
    selected = calibration["selected"]
    print(f"GATE DC-2: PASS (gain={selected['gain']:.2f}, per_hop_steps={selected['per_hop_steps']})")
    print(f"dc2 json -> {output.with_suffix('.json')}")
    print(f"dc2 csv -> {output.with_suffix('.csv')}")
    return 0


def non_overlapping(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    return bool(high_a < low_b or high_b < low_a)


def dc3_positive_criteria(result: dict, watched_video_verdict: str | None = None) -> dict:
    summary = result["summary"]
    paired = {
        (row["metric"], row["comparison"].removeprefix("real_minus_")): row
        for row in result["comparisons"]
    }
    beat = {}
    for metric in ("coherence_scalar", "mean_feet_planted"):
        real_ci = summary["real"]["metrics"][metric]
        beat[metric] = {}
        for condition in ("shuffled", "random"):
            control_ci = summary[condition]["metrics"][metric]
            test = paired[(metric, condition)]
            beat[metric][condition] = {
                "positive_mean_difference": bool(test["mean_difference"] > 0.0),
                "holm_p_below_0_05": bool(test["p_holm_across_two_controls_for_metric"] < 0.05),
                "non_overlapping_summary_bootstrap_cis": non_overlapping(
                    real_ci["ci_low"], real_ci["ci_high"], control_ci["ci_low"], control_ci["ci_high"]
                ),
            }
            beat[metric][condition]["all"] = bool(all(beat[metric][condition].values()))
    video = watched_video_verdict is not None and watched_video_verdict.lower().startswith("genuine tripod")
    return {
        "real_phase_present_all_runs": bool(all(row["continuous_phase"]["phase_verdict"] == "M2-PHASE-PRESENT" for row in result["rows"] if row["condition"] == "real")),
        "real_falls_zero": bool(summary["real"]["falls"] == 0),
        "real_forward_positive": bool(summary["real"]["metrics"]["forward_distance_m"]["mean"] > 0.0),
        "real_path_positive": bool(summary["real"]["metrics"]["path_m"]["mean"] > 0.0),
        "beats_both_controls_on_coherence_and_feet": beat,
        "watched_video_genuine_tripod": video,
        "full_conjunction": bool(
            all((
                all(criteria[condition]["all"] for condition in ("shuffled", "random"))
                for criteria in beat.values()
            ))
            and summary["real"]["falls"] == 0
            and summary["real"]["metrics"]["forward_distance_m"]["mean"] > 0.0
            and summary["real"]["metrics"]["path_m"]["mean"] > 0.0
            and all(row["continuous_phase"]["phase_verdict"] == "M2-PHASE-PRESENT" for row in result["rows"] if row["condition"] == "real")
            and video
        ),
    }


def write_dc3(output: Path, result: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = (
        "condition", "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "forward_distance_m", "path_m", "displacement_m", "coherence_scalar", "mean_swing_duty",
        "mean_feet_planted", "min_height_m", "fell", "phase_verdict",
    )
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({
            **{key: row[key] for key in columns if key != "phase_verdict"},
            "phase_verdict": row["continuous_phase"]["phase_verdict"],
        } for row in result["rows"])


def run_dc3(data: Path, dc2_path: Path, output: Path, video: Path, n: int, secs: float, seed: int) -> int:
    dc2 = json.loads(dc2_path.read_text(encoding="utf-8"))
    calibration = dc2["calibration"]
    selected = calibration["selected"]
    if selected is None or calibration["selection_rule"] != DC2_SELECTION_RULE:
        raise AssertionError("DC-2 selected cell or selection rule is missing")
    matrices = build_coupling_matrices(data)
    graph = graph_record(data)
    if not graph["hash_equal"]:
        raise AssertionError("frozen graph hash changed before DC-3")
    gain = float(selected["gain"])
    per_hop_steps = int(selected["per_hop_steps"])
    delay = anatomical_delay_matrix(per_hop_steps)
    preregistered = {
        "method": "DC-3 delay-coupling generation run",
        "protocol": {
            "planned_n_per_condition": PLANNED_N,
            "planned_secs": PLANNED_SECS,
            "n_per_condition": n,
            "secs": secs,
            "episode_seed": seed,
            "shared_episode_starts": True,
            "same_gain_all_conditions": True,
            "same_delay_all_conditions": True,
            "only_condition_difference": "coupling source matrix; shared real VNC continuous phase source",
            "phase_source": "continuous_phase.vnc_phase_source real VNC phase, shared across conditions",
            "coupling_equation": "theta_i[t] = theta_i[t-1] + source_delta_i[t] + g * sum_j K[i,j] * sin(theta_j[t-1-D[i,j]] - theta_i[t-1])",
            "coupling_gain": gain,
            "per_hop_steps": per_hop_steps,
            "delay_matrix": delay.tolist(),
            "engine": "continuous_phase.run_engine / StepGaitEngine.continuous_targets; fixed Path-1 waveform",
            "forward": 0.8,
            "turn": 0.0,
            "h_min_m": 0.08,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "learning": False,
        },
        "graph": graph,
        "coupling": matrices["record"],
        "dc2_calibration": {"artifact": str(dc2_path), "selected": selected, "selection_rule": calibration["selection_rule"]},
        "pre_registered_null": DELAY_NULL,
        "pre_registered_positive": DELAY_POSITIVE,
        "pre_registered_before_full_run": True,
        "zero_learning": True,
        "rows": [],
    }
    write_dc3(output, preregistered)
    source = vnc_phase_source(data, "all_legs_weight_csr.pkl", "all_legs_io_index.npz", False)
    phases = {
        condition: delayed_coupled_phases(source["phase"], matrices["matrices"][condition], gain, delay)
        for condition in CONDITIONS
    }
    smoke_specs = episode_specs(n, seed)
    smoke = {}
    smoke_steps = int(round(min(1.0, secs) * CTRL_HZ))
    for condition in CONDITIONS:
        smoke_result = run_engine(phases[condition][:smoke_steps], smoke_specs[0], secs=smoke_steps / CTRL_HZ)
        smoke[condition] = {key: smoke_result[key] for key in ("path_m", "coherence_scalar", "mean_feet_planted", "fell")}
        print(
            f"DC-3 smoke {condition}: path={smoke_result['path_m']:.4f} "
            f"coherence={smoke_result['coherence_scalar']:+.4f} feet={smoke_result['mean_feet_planted']:.3f} fell={smoke_result['fell']}",
            flush=True,
        )
    if not all(np.isfinite(value[name]) for value in smoke.values() for name in ("path_m", "coherence_scalar", "mean_feet_planted")):
        raise AssertionError("DC-3 smoke produced non-finite metrics")

    specs = episode_specs(n, seed)
    rows = []
    for condition in CONDITIONS:
        for index, spec in enumerate(specs):
            row = run_engine(phases[condition], spec, video if condition == "real" and index == 0 else None, secs)
            rows.append({**spec, **row, "condition": condition})
            print(
                f"{condition} {index + 1}/{n} path={row['path_m']:.4f} "
                f"coherence={row['coherence_scalar']:+.4f} feet={row['mean_feet_planted']:.3f} fell={row['fell']}",
                flush=True,
            )
    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    reference = [(row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"]) for row in groups["real"]]
    shared_starts = all(
        [(row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"]) for row in groups[condition]] == reference
        for condition in CONDITIONS
    )
    protocols = [row["protocol"] for row in rows]
    same_pipeline = all(protocol == protocols[0] for protocol in protocols)
    finite_metrics = all(np.isfinite(row[name]) for row in rows for name in SUMMARY_METRICS)
    phase_inputs = {condition: source["phase"] for condition in CONDITIONS}
    same_phase_source = all(phase_inputs[condition] is phase_inputs["real"] for condition in CONDITIONS)
    if not shared_starts or not same_pipeline or not finite_metrics or not graph["hash_equal"] or not same_phase_source or any(len(group) != n for group in groups.values()):
        raise AssertionError("DC-3 control integrity failed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {condition: summarize(groups[condition], rng) for condition in CONDITIONS}
    paired = comparisons(groups, rng, SUMMARY_METRICS)
    result = {
        **preregistered,
        "sources": {condition: {"source": "real_vnc_continuous_phase", "shared": True, "record": source["record"]} for condition in CONDITIONS},
        "smoke": smoke,
        "summary": summary,
        "comparisons": paired,
        "rows": rows,
        "gate_dc3": {
            "shared_episode_starts": shared_starts,
            "same_fixed_pipeline": same_pipeline,
            "same_phase_source": True,
            "same_gain": True,
            "same_delay": True,
            "finite_metrics": finite_metrics,
            "graph_hash_equal": graph["hash_equal"],
            "rows": len(rows),
            "video": str(video),
            "watched_video_verdict": DC3_VIDEO_VERDICT_PENDING,
            "verdict": "GATE DC-3: PASS pending watched-video verdict",
        },
        "automatic_positive_criteria": dc3_positive_criteria({"summary": summary, "comparisons": paired, "rows": rows}),
        "verdict": "PENDING_MANUAL_WATCH",
    }
    write_dc3(output, result)
    print("GATE DC-3: PASS (metrics complete; watch real video before verdict)")
    print(f"json -> {output.with_suffix('.json')}")
    print(f"csv -> {output.with_suffix('.csv')}")
    print(f"video -> {video}")
    return 0


def finalize_dc3(output: Path, watched_video_verdict: str) -> int:
    result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    criteria = dc3_positive_criteria(result, watched_video_verdict)
    result["watched_video_verdict"] = watched_video_verdict
    result["automatic_positive_criteria"] = criteria
    result["verdict"] = "DELAY-COUPLING-EARNED" if criteria["full_conjunction"] else "DELAY-COUPLING-NULL"
    result["gate_dc3"]["watched_video_verdict"] = watched_video_verdict
    result["gate_dc3"]["verdict"] = "GATE DC-3: PASS"
    write_dc3(output, result)
    print(f"GATE DC-3: PASS (watched real video: {watched_video_verdict})")
    print(f"{result['verdict']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RESULTS)
    parser.add_argument("--phase", choices=("dc1", "dc2", "dc3", "finalize"), default="dc1")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dc2", type=Path, default=RESULTS / "delay_coupling_dc2.json")
    parser.add_argument("--video", type=Path, default=RESULTS / "delay_coupling_real_multiview.mp4")
    parser.add_argument("--n", type=int, default=PLANNED_N)
    parser.add_argument("--secs", type=float, default=PLANNED_SECS)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--watched-verdict", default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = RESULTS / ("delay_coupling_dc2" if args.phase == "dc2" else "delay_coupling")
    if args.phase == "dc1":
        dc1_self_check(args.data)
        return 0
    if args.phase == "dc2":
        return run_dc2(args.data, args.out)
    if args.phase == "dc3":
        if args.n != PLANNED_N or args.secs != PLANNED_SECS:
            raise ValueError("DC-3 requires n=20 and 8 seconds")
        return run_dc3(args.data, args.dc2, args.out, args.video, args.n, args.secs, args.seed)
    if args.watched_verdict is None:
        raise ValueError("--watched-verdict is required for --phase finalize")
    return finalize_dc3(args.out, args.watched_verdict)


if __name__ == "__main__":
    raise SystemExit(main())
