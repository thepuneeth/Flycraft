"""Untried zero-learning transfer methods, run one preregistered phase at a time."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
ANATOMY_DATA = HERE.parent.parent / "data"
sys.path.insert(0, str(HERE))
from audit_intersegmental import cross_edges, load_annotations  # noqa: E402
from analyze_control_study import exact_sign_flip, holm_adjust, paired_bootstrap_ci  # noqa: E402
from fetch_manc import NEUROTRANSMITTERS, nt_signs  # noqa: E402
from last_three_jobs import (  # noqa: E402
    CONDITIONS,
    DIRECT_SUMMARY_METRICS,
    MAP_GAINS,
    SUMMARY_METRICS,
    calibrate_gain,
    comparisons,
    coupled_phases,
    direct_sources,
    endpoint_legs,
    feet_planted_fraction,
    graph_record,
    make_cameras,
    run_engine,
    run_direct_episode,
    summarize,
    vnc_phase_source,
)
from mn_to_joint import translate, translate_antagonist  # noqa: E402
from preflight import EXPECTED_SHA256, sha256  # noqa: E402
from probe_structured_descending import brain_type_indices  # noqa: E402
from run_hexapod_chase import retina  # noqa: E402
from stepwalk_controls import episode_specs  # noqa: E402
from hexapod_gait import H_MIN, XML, home_stance, pd_torque  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
PLANNED_N = 20
PLANNED_SECS = 8.0
EPISODE_SEED = 20260829
COORD_SHUFFLE_SEED = 20260833
COORD_RANDOM_SEED = 20260834
INHIB_SHUFFLE_SEED = 20260835
INHIB_RANDOM_SEED = 20260836
SMOKE_SECS = 1.0
WAVE_BASELINE = RESULTS / "lastthree_C_direct_mn_waveform.json"
WAVE_MAP_NAME = "mn_to_joint.translate_antagonist(motor_activity, io, gains=(-0.55, 0.65, 0.65))"
VISION_SHUFFLE_SEED = 20260837
VISION_RANDOM_SEED = 20260838
VISION_ANGLES_DEG = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
VISION_PREY_DISTANCE_M = 3.0
VISION_TRIAL_SECS = 2.0
VISION_VIDEO_ANGLES_DEG = (-60.0, 60.0)
VISION_BRAIN_WEIGHT_SHA256 = "f84f8dcd8397a43b8825ba45573cf14105f23d1846c82b58e4f07422f7586a88"
VISION_BRAIN_INDEX_SHA256 = "338cb0160ddeced104f597884ed99be84966a732f89199e876977a18bf84d7b5"
VISION_BRAIN_SUBSTEPS = 12
VISION_BRAIN_TAU = 0.02
VISION_BRAIN_DT = 0.005
VISION_BRAIN_GW = 0.006
VISION_BRAIN_DRIVE_IN = 1.2
VISION_TURN_GAIN = 6.0
VISION_FWD_GAIN = 12.0
VISION_VNC_DRIVE = 100.0
VISION_VNC_TURN_SCALE = 6.0
VISION_VNC_SUBSTEPS = 4
VISION_VNC_TAU = 0.02
VISION_VNC_DT = 0.005
VISION_VNC_GW = 0.03
VISION_SMOKE_SECS = 0.5


def sign_table(matrix: np.ndarray) -> dict:
    values = np.asarray(matrix, dtype=np.float64)
    return {
        "positive_cells": int((values > 0.0).sum()),
        "negative_cells": int((values < 0.0).sum()),
        "zero_cells": int((values == 0.0).sum()),
        "row_sums": values.sum(axis=1).tolist(),
        "absolute_row_masses": np.abs(values).sum(axis=1).tolist(),
    }


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    scale = np.abs(matrix).sum(axis=1)
    if np.any(scale <= 0.0):
        raise AssertionError("coordination matrix has an empty post-leg row")
    return matrix / scale[:, None]


def build_edge_matrices(
    data: Path,
    *,
    published_only: bool,
    inhibitory_only: bool,
    shuffle_seed: int,
    random_seed: int,
    edge_scope: str,
) -> tuple[dict[str, np.ndarray], dict]:
    annotations = load_annotations(data)
    pre, post, weights, _, _ = cross_edges(data, annotations)
    io = np.load(data / "all_legs_io_index.npz")
    published = np.asarray(io["published_coordination_body_ids"], dtype=np.int64)
    if len(published) != 212 or len(np.unique(published)) != 212:
        raise AssertionError("published coordination inventory is not 212 unique cells")
    if len(io["published_coordination"]) != 212:
        raise AssertionError("published coordination cells are not all represented in graph")

    signs = nt_signs(ANATOMY_DATA / "manc" / NEUROTRANSMITTERS, np.unique(pre))
    pre_sign = np.asarray([signs.get(int(body), 0.0) for body in pre], dtype=np.float64)
    pre_leg = endpoint_legs(annotations, pre)
    post_leg = endpoint_legs(annotations, post)
    valid = (pre_leg >= 0) & (post_leg >= 0) & (pre_leg != post_leg) & (pre_sign != 0.0)
    if published_only:
        valid &= np.isin(pre, published) & np.isin(post, published)
    if inhibitory_only:
        valid &= pre_sign < 0.0
    signed_weights = weights * pre_sign
    real = np.zeros((6, 6), dtype=np.float64)
    np.add.at(real, (post_leg[valid], pre_leg[valid]), signed_weights[valid])
    mask = real != 0.0
    if int(mask.sum()) == 0:
        raise AssertionError(f"{edge_scope} has no cross-leg K cells")

    values = real[mask].copy()
    shuffled = np.zeros_like(real)
    shuffled[mask] = np.random.default_rng(shuffle_seed).permutation(values)
    random = np.zeros_like(real)
    random[mask] = (
        np.random.default_rng(random_seed).permutation(np.abs(values))
        * np.random.default_rng(random_seed + 1).permutation(np.sign(values))
    )
    raw_matrices = {"real": real, "shuffled": shuffled, "random": random}
    matrices = {condition: normalized_rows(matrix) for condition, matrix in raw_matrices.items()}
    record = {
        "edge_scope": edge_scope,
        "leg_order": list(LEG_NAMES),
        "matrix_orientation": "K[post_leg, pre_leg]",
        "edge_source": "audit_intersegmental.cross_edges at absolute synapse floor 5",
        "sign_source": "fetch_manc.nt_signs predicted_nt: acetylcholine +1; GABA/glutamate -1; unknown omitted",
        "raw_cross_edge_rows": int(len(pre)),
        "eligible_signed_edge_rows": int(valid.sum()),
        "eligible_edge_sign_rows": {
            "positive": int((signed_weights[valid] > 0.0).sum()),
            "negative": int((signed_weights[valid] < 0.0).sum()),
        },
        "published_coordination_cells": 212,
        "published_coordination_cells_in_graph": 212,
        "raw_nonzero_cells": int(mask.sum()),
        "real_cell_sign_table": sign_table(real),
        "raw_signed_matrices": {condition: matrix.tolist() for condition, matrix in raw_matrices.items()},
        "normalized_matrices": {condition: matrix.tolist() for condition, matrix in matrices.items()},
        "control_rules": {
            "shuffled": "permute real signed nonzero K cell values across the fixed real K mask",
            "random": "independently permute real K magnitudes and signs across the fixed real K mask",
        },
        "shuffle_seed": shuffle_seed,
        "random_seed": random_seed,
        "control_cell_sign_tables": {condition: sign_table(matrix) for condition, matrix in raw_matrices.items()},
    }
    return matrices, record


def build_coordination_matrices(data: Path) -> tuple[dict[str, np.ndarray], dict]:
    matrices, record = build_edge_matrices(
        data,
        published_only=True,
        inhibitory_only=False,
        shuffle_seed=COORD_SHUFFLE_SEED,
        random_seed=COORD_RANDOM_SEED,
        edge_scope="both endpoints in io[published_coordination_body_ids], cross-neuromere only",
    )
    record["eligible_published_signed_edge_rows"] = record.pop("eligible_signed_edge_rows")
    return matrices, record


def build_inhibitory_matrices(data: Path) -> tuple[dict[str, np.ndarray], dict]:
    return build_edge_matrices(
        data,
        published_only=False,
        inhibitory_only=True,
        shuffle_seed=INHIB_SHUFFLE_SEED,
        random_seed=INHIB_RANDOM_SEED,
        edge_scope="all T1/T2/T3 cross-neuromere edges with inhibitory presynaptic sign",
    )


def smoke_test(matrices: dict[str, np.ndarray], gain: dict, phase: np.ndarray, seed: int, label: str) -> dict:
    specs = episode_specs(PLANNED_N, seed)
    steps = int(round(SMOKE_SECS * 50.0))
    rows = {}
    for condition in CONDITIONS:
        phases = coupled_phases(phase, matrices[condition], gain["selected_gain"])
        result = run_engine(phases[:steps], specs[0], secs=SMOKE_SECS)
        rows[condition] = {
            "path_m": result["path_m"],
            "coherence_scalar": result["coherence_scalar"],
            "mean_feet_planted": result["mean_feet_planted"],
            "fell": result["fell"],
            "phase_verdict": result["continuous_phase"]["phase_verdict"],
        }
        print(
            f"smoke {condition}: path={result['path_m']:.4f} "
            f"coherence={result['coherence_scalar']:+.4f} "
            f"feet={result['mean_feet_planted']:.3f} fell={result['fell']}",
            flush=True,
        )
    if not all(np.isfinite(value[name]) for value in rows.values() for name in ("path_m", "coherence_scalar", "mean_feet_planted")):
        raise AssertionError(f"{label} smoke produced non-finite metrics")
    return rows


def run_coordination_phase(
    args: argparse.Namespace,
    phase_name: str,
    matrices: dict[str, np.ndarray],
    coupling: dict,
    null_text: str,
) -> int:
    print(
        f"PRE-REGISTERED NULL {phase_name}: {null_text}",
        flush=True,
    )
    print(
        f"{phase_name} sign table: positive_cells={{}} negative_cells={{}} zero_cells={{}} row_sums={{}}".format(
            coupling["real_cell_sign_table"]["positive_cells"],
            coupling["real_cell_sign_table"]["negative_cells"],
            coupling["real_cell_sign_table"]["zero_cells"],
            [round(value, 3) for value in coupling["real_cell_sign_table"]["row_sums"]],
        ),
        flush=True,
    )
    print(
        f"{phase_name} signed edge rows: positive={{}} negative={{}}".format(
            coupling["eligible_edge_sign_rows"]["positive"],
            coupling["eligible_edge_sign_rows"]["negative"],
        ),
        flush=True,
    )
    gain = calibrate_gain(matrices)
    source = vnc_phase_source(args.data, args.weights_name, args.index_name, False)
    smoke = smoke_test(matrices, gain, source["phase"], args.seed, phase_name)
    phases = {
        condition: coupled_phases(source["phase"], matrices[condition], gain["selected_gain"])
        for condition in CONDITIONS
    }
    specs = episode_specs(args.n, args.seed)
    rows = []
    for condition in CONDITIONS:
        for index, spec in enumerate(specs):
            video = args.video if condition == "real" and index == 0 else None
            result = run_engine(phases[condition][: int(round(args.secs * 50.0))], spec, video=video, secs=args.secs)
            rows.append({**spec, **result, "condition": condition})
            print(
                f"{condition} {index + 1}/{args.n} path={result['path_m']:.4f} "
                f"coherence={result['coherence_scalar']:+.4f} "
                f"feet={result['mean_feet_planted']:.3f} fell={result['fell']}",
                flush=True,
            )

    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    reference = [
        (row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"])
        for row in groups["real"]
    ]
    shared_starts = all(
        [
            (row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"])
            for row in groups[condition]
        ]
        == reference
        for condition in CONDITIONS
    )
    protocols = [row["protocol"] for row in rows]
    same_pipeline = all(protocol == protocols[0] for protocol in protocols)
    finite_metrics = all(np.isfinite(row[name]) for row in rows for name in SUMMARY_METRICS)
    graph = graph_record(args.data)
    if not shared_starts or not same_pipeline or not finite_metrics or not graph["hash_equal"] or any(len(group) != args.n for group in groups.values()):
        raise AssertionError(f"{phase_name} control integrity failed")

    rng = np.random.default_rng(20260829)
    summaries = {condition: summarize(groups[condition], rng, SUMMARY_METRICS) for condition in CONDITIONS}
    paired = comparisons(groups, rng, SUMMARY_METRICS)
    real_phase_present = all(row["continuous_phase"]["phase_verdict"] == "M2-PHASE-PRESENT" for row in groups["real"])
    real_beats_controls = all(
        summaries["real"]["metrics"][metric]["mean"] > max(
            summaries["shuffled"]["metrics"][metric]["mean"],
            summaries["random"]["metrics"][metric]["mean"],
        )
        for metric in ("coherence_scalar", "mean_feet_planted")
    )
    verdict = f"{phase_name}-PRESENT" if real_phase_present and real_beats_controls and summaries["real"]["falls"] == 0 else f"{phase_name}-NULL"
    gate_key = {
        "COORD-212": "gate_coord_212",
        "COORD-INHIB": "gate_coord_inhib",
    }[phase_name]
    result = {
        "method": f"{phase_name} inhibitory/anatomical inter-leg coupling",
        "protocol": {
            "planned_n_per_condition": PLANNED_N,
            "planned_secs": PLANNED_SECS,
            "n_per_condition": args.n,
            "secs": args.secs,
            "episode_seed": args.seed,
            "shared_episode_starts": shared_starts,
            "same_fixed_pipeline": same_pipeline,
            "only_condition_difference": f"{phase_name} K source; real VNC phase source and all body/engine parameters fixed",
            "coupling_equation": "theta_i[t] = theta_i[t-1] + source_delta_i[t] + g * sum_j K[i,j] * sin(theta_j[t-1] - theta_i[t-1])",
            "coupling_gain": gain["selected_gain"],
            "gain_calibration_basis": gain["criterion"]["basis"],
            "engine": "continuous_phase.run_engine / StepGaitEngine.continuous_targets; fixed Path-1 waveform",
            "h_min_m": 0.08,
            "learning": False,
        },
        "graph": graph,
        "coupling": coupling,
        "gain_calibration": gain,
        "phase_source": source["record"],
        "smoke": smoke,
        "summary": summaries,
        "comparisons": paired,
        "pre_registered_null": null_text,
        "verdict": verdict,
        gate_key: {
            "shared_episode_starts": shared_starts,
            "same_fixed_pipeline": same_pipeline,
            "finite_metrics": finite_metrics,
            "graph_hash_equal": graph["hash_equal"],
            "published_coordination_cells": 212,
            "rows": len(rows),
            "smoke_complete": True,
            "verdict": f"GATE {phase_name}: PASS",
        },
        "zero_learning": True,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = (
        "condition", "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "forward_distance_m", "path_m", "displacement_m", "coherence_scalar", "mean_swing_duty",
        "mean_feet_planted", "min_height_m", "fell",
    )
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    print(f"GATE {phase_name}: PASS (shared starts; fixed pipeline; finite; graph hash; 212/212 cells)")
    print(f"RESULT: {verdict}")
    print(f"json -> {args.out.with_suffix('.json')}")
    print(f"csv -> {args.out.with_suffix('.csv')}")
    print(f"video -> {args.video}")
    return 0


def run_coord212(args: argparse.Namespace) -> int:
    matrices, coupling = build_coordination_matrices(args.data)
    return run_coordination_phase(
        args,
        "COORD-212",
        matrices,
        coupling,
        "published-cell K_212 is net-synchronizing and does not produce a stable tripod that beats shuffled/random on coherence and feet-planted",
    )


def run_coord_inhib(args: argparse.Namespace) -> int:
    matrices, coupling = build_inhibitory_matrices(args.data)
    return run_coordination_phase(
        args,
        "COORD-INHIB",
        matrices,
        coupling,
        "inhibitory-only K does not produce a stable anti-phase tripod limit cycle that beats shuffled/random controls",
    )


def wave_comparisons(groups: dict[str, list[dict]], baseline: list[dict], rng: np.random.Generator) -> list[dict]:
    rows = comparisons(groups, rng, DIRECT_SUMMARY_METRICS)
    baseline_by_episode = {int(row["episode"]): row for row in baseline}
    baseline_rows = []
    for metric in DIRECT_SUMMARY_METRICS:
        differences = np.asarray(
            [row[metric] - baseline_by_episode[int(row["episode"])] [metric] for row in groups["real"]],
            dtype=np.float64,
        )
        low, high = paired_bootstrap_ci(differences, rng, 100_000)
        baseline_rows.append({
            "comparison": "wave_real_minus_phase_c_real",
            "metric": metric,
            "n_pairs": len(differences),
            "mean_difference": float(differences.mean()),
            "ci_low": low,
            "ci_high": high,
            "p_exact_two_sided_paired_sign_flip": exact_sign_flip(differences),
        })
    for metric in DIRECT_SUMMARY_METRICS:
        related = [row for row in baseline_rows if row["metric"] == metric]
        related.extend(row for row in rows if row["metric"] == metric)
        adjusted = holm_adjust([row["p_exact_two_sided_paired_sign_flip"] for row in related])
        for row, pvalue in zip(related, adjusted):
            row["p_holm_across_phase_c_and_two_controls_for_metric"] = pvalue
    return rows + baseline_rows


def run_wave_antag(args: argparse.Namespace) -> int:
    null_text = (
        "the fuller antagonist map does not beat Phase C forward distance beyond CI overlap, "
        "or does not beat shuffled/random"
    )
    print(f"PRE-REGISTERED NULL WAVE-ANTAG: {null_text}", flush=True)
    if not WAVE_BASELINE.exists():
        raise FileNotFoundError(f"missing frozen Phase C baseline: {WAVE_BASELINE}")
    baseline_result = json.loads(WAVE_BASELINE.read_text(encoding="utf-8"))
    if baseline_result.get("verdict") != "C1-ABSENT" and baseline_result.get("gate_c1", {}).get("verdict") != "GATE C1: PASS":
        raise AssertionError("Phase C baseline is not the committed C1 record")
    steps = int(round(args.secs * 50.0))
    sources, io, source_records = direct_sources(args.data, args.weights_name, args.index_name, steps)
    zero = translate_antagonist(np.zeros(len(io["motor_out"]), dtype=np.float32), io, gains=(-0.55, 0.65, 0.65))
    assert np.allclose(zero.reshape(6, 3), home_stance())
    specs = episode_specs(args.n, args.seed)
    smoke = {}
    smoke_steps = int(round(SMOKE_SECS * 50.0))
    for condition in CONDITIONS:
        result = run_direct_episode(
            sources[condition][:smoke_steps], io, specs[0], secs=SMOKE_SECS,
            mapper=translate_antagonist, mapper_name=WAVE_MAP_NAME,
        )
        smoke[condition] = {
            key: result[key] for key in ("path_m", "forward_distance_m", "tripod_coherence_scalar", "mean_feet_planted", "fell")
        }
        print(
            f"smoke {condition}: forward={result['forward_distance_m']:.4f} "
            f"path={result['path_m']:.4f} coherence={result['tripod_coherence_scalar']:+.4f} "
            f"feet={result['mean_feet_planted']:.3f} fell={result['fell']}",
            flush=True,
        )
    if not all(np.isfinite(value[key]) for value in smoke.values() for key in ("path_m", "forward_distance_m", "tripod_coherence_scalar", "mean_feet_planted")):
        raise AssertionError("WAVE-ANTAG smoke produced non-finite metrics")

    rows = []
    for condition in CONDITIONS:
        for index, spec in enumerate(specs):
            video = args.video if condition == "real" and index == 0 else None
            result = run_direct_episode(
                sources[condition][:steps], io, spec, video=video, secs=args.secs,
                mapper=translate_antagonist, mapper_name=WAVE_MAP_NAME,
            )
            rows.append({**spec, **result, "condition": condition})
            print(
                f"{condition} {index + 1}/{args.n} forward={result['forward_distance_m']:.4f} "
                f"path={result['path_m']:.4f} coherence={result['tripod_coherence_scalar']:+.4f} "
                f"feet={result['mean_feet_planted']:.3f} fell={result['fell']}",
                flush=True,
            )
    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    reference = [
        (row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"])
        for row in groups["real"]
    ]
    shared_starts = all(
        [
            (row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"])
            for row in groups[condition]
        ]
        == reference
        for condition in CONDITIONS
    )
    protocols = [row["protocol"] for row in rows]
    same_pipeline = all(protocol == protocols[0] for protocol in protocols)
    finite_metrics = all(np.isfinite(row[name]) for row in rows for name in DIRECT_SUMMARY_METRICS)
    graph = graph_record(args.data)
    if not shared_starts or not same_pipeline or not finite_metrics or not graph["hash_equal"] or any(len(group) != args.n for group in groups.values()):
        raise AssertionError("WAVE-ANTAG control integrity failed")

    baseline_rows = baseline_result["rows"]
    baseline_real = [row for row in baseline_rows if row["condition"] == "real"]
    if len(baseline_real) != args.n:
        raise AssertionError("Phase C baseline does not have 20 real episodes")
    for row, old in zip(groups["real"], sorted(baseline_real, key=lambda value: int(value["episode"]))):
        if (row["episode"], row["episode_seed"], row["initial_yaw_rad"], row["prey_x_m"], row["prey_y_m"]) != (old["episode"], old["episode_seed"], old["initial_yaw_rad"], old["prey_x_m"], old["prey_y_m"]):
            raise AssertionError("WAVE-ANTAG and Phase C starts are not paired")
    rng = np.random.default_rng(20260829)
    summary = {condition: summarize(groups[condition], rng, DIRECT_SUMMARY_METRICS) for condition in CONDITIONS}
    comparison_rows = wave_comparisons(groups, baseline_real, rng)
    phase_c_forward = baseline_result["summary"]["real"]["metrics"]["forward_distance_m"]
    wave_forward = summary["real"]["metrics"]["forward_distance_m"]
    beats_phase_c = wave_forward["ci_low"] > phase_c_forward["ci_high"]
    beats_controls = all(
        wave_forward["mean"] > summary[condition]["metrics"]["forward_distance_m"]["mean"]
        for condition in ("shuffled", "random")
    )
    phase_present = all(row["phase"]["verdict"] == "C1-PRESENT" for row in groups["real"])
    verdict = "WAVE-ANTAG-PRESENT" if beats_phase_c and beats_controls and phase_present and summary["real"]["falls"] == 0 else "WAVE-ANTAG-NULL"
    result = {
        "method": "WAVE-ANTAG fuller fixed antagonist MN-to-joint waveform",
        "protocol": {
            "planned_n_per_condition": PLANNED_N,
            "planned_secs": PLANNED_SECS,
            "n_per_condition": args.n,
            "secs": args.secs,
            "episode_seed": args.seed,
            "shared_episode_starts": shared_starts,
            "same_fixed_pipeline": same_pipeline,
            "only_condition_difference": "motor source; fixed fuller antagonist map across all conditions",
            "motor_to_joint": WAVE_MAP_NAME,
            "map_rule": "coxa uses fly coxa pools; femur uses all trochanter+femur pools; tibia uses all tibia+tarsus pools; each target is mean positive sign minus mean negative sign",
            "motor_map_gains": [-0.55, 0.65, 0.65],
            "engine": None,
            "phase_machine": None,
            "h_min_m": 0.08,
            "learning": False,
        },
        "graph": graph,
        "sources": source_records,
        "phase_c_baseline": {
            "artifact": str(WAVE_BASELINE.relative_to(HERE.parents[2])),
            "method": baseline_result["method"],
            "verdict": baseline_result["gate_c1"]["verdict"],
            "zero_learning": baseline_result["zero_learning"],
        },
        "smoke": smoke,
        "summary": summary,
        "comparisons": comparison_rows,
        "beats_phase_c_forward_beyond_ci_overlap": bool(beats_phase_c),
        "beats_shuffled_random_forward_means": bool(beats_controls),
        "real_phase_present": bool(phase_present),
        "pre_registered_null": null_text,
        "verdict": verdict,
        "gate_wave_antag": {
            "shared_episode_starts": shared_starts,
            "same_fixed_pipeline": same_pipeline,
            "finite_metrics": finite_metrics,
            "graph_hash_equal": graph["hash_equal"],
            "phase_c_baseline_paired": True,
            "rows": len(rows),
            "smoke_complete": True,
            "verdict": "GATE WAVE-ANTAG: PASS",
        },
        "zero_learning": True,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = (
        "condition", "episode", "episode_seed", "initial_yaw_rad", "prey_x_m", "prey_y_m",
        "forward_distance_m", "path_m", "displacement_m", "tripod_coherence_scalar", "mean_feet_planted",
        "min_height_m", "motor_temporal_std_mean", "target_temporal_std_mean_rad", "active_target_channels", "fell",
    )
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    print(json.dumps({"summary": summary, "comparisons": comparison_rows}, sort_keys=True))
    print("GATE WAVE-ANTAG: PASS (shared starts; fixed fuller map; finite; Phase C paired)")
    print(f"RESULT: {verdict}")
    print(f"json -> {args.out.with_suffix('.json')}")
    print(f"csv -> {args.out.with_suffix('.csv')}")
    print(f"video -> {args.video}")
    return 0


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        raise AssertionError("vision correlation requires variable azimuth and turn output")
    return float(np.corrcoef(x, y)[0, 1])


def correlation_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, reps: int = 100_000) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    indices = rng.integers(0, len(x), size=(reps, len(x)))
    xs = x[indices]
    ys = y[indices]
    xs -= xs.mean(axis=1, keepdims=True)
    ys -= ys.mean(axis=1, keepdims=True)
    denominator = np.sqrt((xs * xs).sum(axis=1) * (ys * ys).sum(axis=1))
    correlations = (xs * ys).sum(axis=1) / denominator
    correlations = correlations[np.isfinite(correlations)]
    if len(correlations) < reps // 2:
        raise AssertionError("too many degenerate vision bootstrap samples")
    low, high = np.percentile(correlations, (2.5, 97.5))
    return float(low), float(high)


def build_visual_conditions(data: Path) -> tuple[dict[str, dict], dict]:
    weight_path = ANATOMY_DATA / "weight_csr.pkl"
    index_path = ANATOMY_DATA / "io_index.npz"
    with weight_path.open("rb") as handle:
        weight = pickle.load(handle)
    if weight.layout != torch.sparse_csr:
        weight = weight.to_sparse_csr()
    weight = weight.float().cpu()
    io = np.load(index_path)
    type_indices = brain_type_indices(ANATOMY_DATA)
    target_keys = ("dng100_left", "dng100_right", "dnb08_left", "dnb08_right")
    target_indices = np.unique(np.concatenate([type_indices[key] for key in target_keys])).astype(np.int64)
    visual_left = np.asarray(io["vis_left"], dtype=np.int64)
    visual_right = np.asarray(io["vis_right"], dtype=np.int64)
    visual_indices = np.unique(np.concatenate((visual_left, visual_right)))
    columns = weight.col_indices().numpy()
    rows = weight.crow_indices().numpy()
    direct_slots = []
    edge_counts = {}
    edge_sides = {}
    left_set = set(visual_left.tolist())
    right_set = set(visual_right.tolist())
    for target in target_indices:
        start, end = int(rows[target]), int(rows[target + 1])
        local_columns = columns[start:end]
        hits = np.flatnonzero(np.isin(local_columns, visual_indices)) + start
        direct_slots.extend(hits.tolist())
        edge_counts[int(target)] = int(len(hits))
        edge_sides[int(target)] = {
            "left": int(sum(int(columns[slot]) in left_set for slot in hits)),
            "right": int(sum(int(columns[slot]) in right_set for slot in hits)),
        }
    direct_slots = np.asarray(sorted(direct_slots), dtype=np.int64)
    if len(direct_slots) != 27 or len(np.unique(direct_slots)) != 27:
        raise AssertionError(f"expected 27 unique direct visual-to-DN edges, found {len(direct_slots)}")
    base_values = weight.values().clone()
    real_values = base_values.numpy()[direct_slots].copy()
    shuffled_values = np.random.default_rng(VISION_SHUFFLE_SEED).permutation(real_values)
    random_values = (
        np.random.default_rng(VISION_RANDOM_SEED).permutation(np.abs(real_values))
        * np.random.default_rng(VISION_RANDOM_SEED + 1).permutation(np.sign(real_values))
    )
    replacements = {"real": real_values, "shuffled": shuffled_values, "random": random_values}
    conditions = {}
    slot_tensor = torch.from_numpy(direct_slots)
    for condition, replacement in replacements.items():
        values = base_values.clone()
        if condition != "real":
            values[slot_tensor] = torch.from_numpy(replacement.astype(np.float32))
        conditions[condition] = {
            "weight": torch.sparse_csr_tensor(weight.crow_indices(), weight.col_indices(), values, size=weight.shape),
            "replacement_values": replacement.tolist(),
        }
    record = {
        "brain_weight": "eon-fly-brain/data/weight_csr.pkl",
        "brain_index": "eon-fly-brain/data/io_index.npz",
        "brain_n": int(weight.shape[0]),
        "brain_hashes": {
            "weight_csr.pkl": sha256(weight_path),
            "io_index.npz": sha256(index_path),
        },
        "expected_brain_hashes": {
            "weight_csr.pkl": VISION_BRAIN_WEIGHT_SHA256,
            "io_index.npz": VISION_BRAIN_INDEX_SHA256,
        },
        "hash_equal": sha256(weight_path) == VISION_BRAIN_WEIGHT_SHA256 and sha256(index_path) == VISION_BRAIN_INDEX_SHA256,
        "visual_input_counts": {"left": int(len(visual_left)), "right": int(len(visual_right)), "union": int(len(visual_indices))},
        "descending_output_counts": {key: int(len(type_indices[key])) for key in target_keys},
        "direct_visual_to_descending_edge_count": int(len(direct_slots)),
        "direct_edge_counts_by_target_index": edge_counts,
        "direct_edge_sides_by_target_index": edge_sides,
        "fixed_target_rule": "exact DNg100/DNb08 cells from probe_structured_descending.brain_type_indices",
        "shuffle_seed": VISION_SHUFFLE_SEED,
        "random_seed": VISION_RANDOM_SEED,
        "control_rules": {
            "shuffled": "permute real signed values over the fixed 27 visual-to-DNg100/DNb08 edge slots",
            "random": "independently permute real magnitudes and signs over the fixed 27 slots",
        },
        "fixed_csr_structure": True,
        "csr_row_indices_equal": all(torch.equal(weight.crow_indices(), conditions[condition]["weight"].crow_indices()) for condition in CONDITIONS),
        "csr_column_indices_equal": all(torch.equal(weight.col_indices(), conditions[condition]["weight"].col_indices()) for condition in CONDITIONS),
    }
    if not record["hash_equal"] or not record["csr_row_indices_equal"] or not record["csr_column_indices_equal"]:
        raise AssertionError("frozen brain or CSR integrity failed")
    return {condition: {**conditions[condition], "io": io, "type_indices": type_indices} for condition in CONDITIONS}, record


class VisualBrain:
    def __init__(self, condition: dict):
        self.weight = condition["weight"]
        self.io = condition["io"]
        self.type_indices = condition["type_indices"]
        self.n = int(self.weight.shape[0])
        self.visual_left = torch.from_numpy(self.io["vis_left"]).long()
        self.visual_right = torch.from_numpy(self.io["vis_right"]).long()
        self.state = torch.zeros(self.n, dtype=torch.float32)
        self.decay = 1.0 - VISION_BRAIN_DT / VISION_BRAIN_TAU
        self.dt_tau = VISION_BRAIN_DT / VISION_BRAIN_TAU

    def reset(self):
        self.state.zero_()

    def step(self, present: bool, azimuth: float, size: float) -> dict[str, float]:
        magnitude = VISION_BRAIN_DRIVE_IN * (0.5 + min(1.0, size))
        current = torch.zeros(self.n, dtype=torch.float32)
        if present:
            current[self.visual_left] = magnitude * (max(0.0, azimuth) + 0.15)
            current[self.visual_right] = magnitude * (max(0.0, -azimuth) + 0.15)
        with torch.no_grad():
            for _ in range(VISION_BRAIN_SUBSTEPS):
                rates = torch.relu(torch.tanh(self.state))
                self.state.mul_(self.decay).add_(self.dt_tau * (VISION_BRAIN_GW * (self.weight @ rates) + current))
                self.state.clamp_(-5.0, 5.0)
            rates = torch.relu(torch.tanh(self.state))
        dng_left = float(rates[self.type_indices["dng100_left"]].mean())
        dng_right = float(rates[self.type_indices["dng100_right"]].mean())
        dnb_left = float(rates[self.type_indices["dnb08_left"]].mean())
        dnb_right = float(rates[self.type_indices["dnb08_right"]].mean())
        left = float(np.mean((dng_left, dnb_left)))
        right = float(np.mean((dng_right, dnb_right)))
        turn = -VISION_TURN_GAIN * (right - left)
        heading_gate = max(0.1, float(np.cos(azimuth))) if present else 0.0
        forward = float(np.clip(VISION_FWD_GAIN * (left + right), 0.0, 1.0)) * heading_gate
        return {
            "dng100_left": dng_left,
            "dng100_right": dng_right,
            "dnb08_left": dnb_left,
            "dnb08_right": dnb_right,
            "dL": left,
            "dR": right,
            "turn": turn,
            "forward": forward,
        }


def run_vision_trial(
    brain: VisualBrain,
    vnc: VNCStepper,
    azimuth: float,
    secs: float,
    video_frames: list[np.ndarray] | None = None,
) -> dict:
    brain.reset()
    vnc.reset()
    model = mujoco.MjModel.from_xml_path(str(XML))
    body = mujoco.MjData(model)
    body.qpos[2] = 0.15
    body.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    physics_steps = max(1, int(round((1.0 / 50.0) / model.opt.timestep)))
    stance = translate(np.zeros(len(vnc.io["motor_out"]), dtype=np.float32), vnc.io, gains=MAP_GAINS)
    assert np.allclose(stance.reshape(6, 3), home_stance(model))
    for _ in range(int(0.7 / model.opt.timestep)):
        body.ctrl[:] = pd_torque(model, body, stance, 3.0, 0.05)
        mujoco.mj_step(model, body)
    x0, y0 = float(body.qpos[0]), float(body.qpos[1])
    yaw0 = 2.0 * np.arctan2(body.qpos[6], body.qpos[3])
    prey = (x0 + VISION_PREY_DISTANCE_M * np.cos(azimuth), y0 + VISION_PREY_DISTANCE_M * np.sin(azimuth))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{leg}") for leg in range(6))
    renderer = mujoco.Renderer(model, 480, 640) if video_frames is not None else None
    cameras = make_cameras(model) if renderer else []
    n = int(round(secs * 50.0))
    turns, forwards, observed_azimuths, heights, planted = [], [], [], [], []
    dng_left, dng_right, dnb_left, dnb_right = [], [], [], []
    last = (x0, y0)
    path = 0.0
    for step in range(n):
        trunk = (float(body.qpos[0]), float(body.qpos[1]))
        yaw = 2.0 * np.arctan2(body.qpos[6], body.qpos[3])
        present, observed_azimuth, distance, size = retina(trunk, yaw, prey)
        output = brain.step(present, observed_azimuth, size)
        current = vnc.descending_current(output["forward"] * VISION_VNC_DRIVE, output["turn"], 1.0, VISION_VNC_TURN_SCALE)
        motor = vnc.step(current).cpu().numpy()
        target = translate(motor, vnc.io, gains=MAP_GAINS)
        for _ in range(physics_steps):
            body.ctrl[:] = pd_torque(model, body, target, 3.0, 0.05)
            mujoco.mj_step(model, body)
        now = (float(body.qpos[0]), float(body.qpos[1]))
        path += float(np.hypot(now[0] - last[0], now[1] - last[1]))
        last = now
        turns.append(output["turn"])
        forwards.append(output["forward"])
        observed_azimuths.append(observed_azimuth)
        heights.append(float(body.qpos[2]))
        planted.append(float(feet_planted_fraction(body, floor_id, foot_ids)))
        dng_left.append(output["dng100_left"])
        dng_right.append(output["dng100_right"])
        dnb_left.append(output["dnb08_left"])
        dnb_right.append(output["dnb08_right"])
        if renderer and step % 2 == 0:
            views = []
            for camera in cameras:
                camera.lookat[:] = (float(body.qpos[0]), float(body.qpos[1]), 0.08)
                renderer.update_scene(body, camera=camera)
                views.append(renderer.render())
            video_frames.append(np.concatenate((np.concatenate(views[:2], axis=1), np.concatenate(views[2:], axis=1)), axis=0))
    yaw1 = 2.0 * np.arctan2(body.qpos[6], body.qpos[3])
    dx, dy = float(body.qpos[0]) - x0, float(body.qpos[1]) - y0
    turn_mean = float(np.mean(turns))
    toward_sign = float(np.sign(azimuth))
    return {
        "azimuth_deg": float(np.rad2deg(azimuth)),
        "azimuth_rad": float(azimuth),
        "azimuth_label": "left" if azimuth > 0.0 else "right" if azimuth < 0.0 else "center",
        "prey_distance_m": VISION_PREY_DISTANCE_M,
        "initial_x_m": x0,
        "initial_y_m": y0,
        "initial_yaw_rad": yaw0,
        "mean_observed_azimuth_rad": float(np.mean(observed_azimuths)),
        "mean_turn_command": turn_mean,
        "initial_turn_command": float(turns[0]),
        "turn_command_std": float(np.std(turns)),
        "mean_forward_command": float(np.mean(forwards)),
        "toward_turn_score": float(toward_sign * turn_mean),
        "yaw_change_rad": float(yaw1 - yaw0),
        "toward_yaw_score": float(toward_sign * (yaw1 - yaw0)),
        "mean_dng100_left": float(np.mean(dng_left)),
        "mean_dng100_right": float(np.mean(dng_right)),
        "mean_dnb08_left": float(np.mean(dnb_left)),
        "mean_dnb08_right": float(np.mean(dnb_right)),
        "path_m": float(path),
        "forward_distance_m": float(dx * np.cos(yaw0) + dy * np.sin(yaw0)),
        "displacement_m": float(np.hypot(dx, dy)),
        "mean_feet_planted": float(np.mean(planted)),
        "min_height_m": float(np.min(heights)),
        "final_height_m": float(heights[-1]),
        "fell": bool(np.min(heights) < H_MIN),
        "protocol": {
            "secs": secs,
            "ctrl_hz": 50.0,
            "prey_distance_m": VISION_PREY_DISTANCE_M,
            "retina": "run_hexapod_chase.retina; present=dist<12 and abs(az)<150deg; size=0.2/max(dist,0.2)",
            "brain_model": "bounded rate model",
            "brain_tau": VISION_BRAIN_TAU,
            "brain_dt": VISION_BRAIN_DT,
            "brain_substeps": VISION_BRAIN_SUBSTEPS,
            "brain_gW": VISION_BRAIN_GW,
            "brain_drive_in": VISION_BRAIN_DRIVE_IN,
            "brain_readout": "dL/R=mean(exact DNg100,DNb08 left/right); turn=-6*(dR-dL); forward=clip(12*(dL+dR))*max(0.1,cos(az))",
            "vnc_drive": VISION_VNC_DRIVE,
            "vnc_turn_scale": VISION_VNC_TURN_SCALE,
            "vnc_model": "paper",
            "vnc_tau": VISION_VNC_TAU,
            "vnc_dt": VISION_VNC_DT,
            "vnc_substeps": VISION_VNC_SUBSTEPS,
            "vnc_gW": VISION_VNC_GW,
            "vnc_heterogeneity": True,
            "vnc_seed": 1,
            "motor_to_joint": "mn_to_joint.translate(motor_activity, io, gains=(-0.55, 0.65, 0.65))",
            "kp": 3.0,
            "kd": 0.05,
            "h_min_m": H_MIN,
            "learning": False,
        },
    }


def vision_comparisons(groups: dict[str, list[dict]], rng: np.random.Generator) -> list[dict]:
    nonzero = {condition: [row for row in groups[condition] if row["azimuth_rad"] != 0.0] for condition in CONDITIONS}
    rows = []
    for metric in ("toward_turn_score", "toward_yaw_score"):
        metric_rows = []
        for right in ("shuffled", "random"):
            differences = np.asarray(
                [left[metric] - other[metric] for left, other in zip(nonzero["real"], nonzero[right])],
                dtype=np.float64,
            )
            low, high = paired_bootstrap_ci(differences, rng, 100_000)
            metric_rows.append({
                "comparison": f"real_minus_{right}",
                "metric": metric,
                "n_pairs": len(differences),
                "mean_difference": float(differences.mean()),
                "ci_low": low,
                "ci_high": high,
                "p_exact_two_sided_paired_sign_flip": exact_sign_flip(differences),
            })
        adjusted = holm_adjust([row["p_exact_two_sided_paired_sign_flip"] for row in metric_rows])
        for row, pvalue in zip(metric_rows, adjusted):
            row["p_holm_across_two_controls_for_metric"] = pvalue
        rows.extend(metric_rows)
    return rows


def run_vision_eval(args: argparse.Namespace) -> int:
    null_text = "the frozen optic-lobe steering output does not correlate with true prey azimuth better than shuffled/random wiring"
    print(f"PRE-REGISTERED NULL VISION-EVAL: {null_text}", flush=True)
    conditions, brain_record = build_visual_conditions(args.data)
    print(
        "VISION-EVAL direct visual-to-DN slots: {} edges; DNg100 L/R={}/{}; DNb08 L/R={}/{}".format(
            brain_record["direct_visual_to_descending_edge_count"],
            brain_record["descending_output_counts"]["dng100_left"],
            brain_record["descending_output_counts"]["dng100_right"],
            brain_record["descending_output_counts"]["dnb08_left"],
            brain_record["descending_output_counts"]["dnb08_right"],
        ),
        flush=True,
    )
    smoke_angles = (-60.0, 0.0, 60.0)
    smoke = {}
    for condition in CONDITIONS:
        brain = VisualBrain(conditions[condition])
        vnc = VNCStepper(
            args.data,
            substeps=VISION_VNC_SUBSTEPS,
            tau=VISION_VNC_TAU,
            dt=VISION_VNC_DT,
            gW=VISION_VNC_GW,
            weights_name=args.weights_name,
            index_name=args.index_name,
            model="paper",
            threshold=7.5,
            activation_gain=1.0,
            fr_cap=200.0,
            heterogeneity=True,
            seed=1,
        )
        smoke_rows = [run_vision_trial(brain, vnc, np.deg2rad(angle), VISION_SMOKE_SECS) for angle in smoke_angles]
        smoke_turns = [row["mean_turn_command"] for row in smoke_rows]
        smoke[condition] = {"angles_deg": list(smoke_angles), "mean_turn_commands": smoke_turns}
        if not np.isfinite(smoke_turns).all():
            raise AssertionError("VISION-EVAL smoke produced non-finite turn commands")
        print(f"smoke {condition}: turns={['%.4f' % value for value in smoke_turns]}", flush=True)

    angles = tuple(np.deg2rad(value) for value in VISION_ANGLES_DEG)
    rows = []
    video_frames = []
    for condition in CONDITIONS:
        brain = VisualBrain(conditions[condition])
        vnc = VNCStepper(
            args.data,
            substeps=VISION_VNC_SUBSTEPS,
            tau=VISION_VNC_TAU,
            dt=VISION_VNC_DT,
            gW=VISION_VNC_GW,
            weights_name=args.weights_name,
            index_name=args.index_name,
            model="paper",
            threshold=7.5,
            activation_gain=1.0,
            fr_cap=200.0,
            heterogeneity=True,
            seed=1,
        )
        for azimuth in angles:
            capture = video_frames if condition == "real" and any(np.isclose(float(np.rad2deg(azimuth)), video_angle) for video_angle in VISION_VIDEO_ANGLES_DEG) else None
            result = run_vision_trial(brain, vnc, float(azimuth), VISION_TRIAL_SECS, capture)
            rows.append({**result, "condition": condition})
            print(
                f"{condition} az={result['azimuth_deg']:+.0f} turn={result['mean_turn_command']:+.4f} "
                f"yaw={np.rad2deg(result['yaw_change_rad']):+.3f}deg forward={result['forward_distance_m']:+.4f} fell={result['fell']}",
                flush=True,
            )
    groups = {condition: [row for row in rows if row["condition"] == condition] for condition in CONDITIONS}
    angle_reference = [row["azimuth_rad"] for row in groups["real"]]
    same_angles = all([row["azimuth_rad"] for row in groups[condition]] == angle_reference for condition in CONDITIONS)
    protocols = [row["protocol"] for row in rows]
    same_pipeline = all(protocol == protocols[0] for protocol in protocols)
    finite_metrics = all(np.isfinite(row[key]) for row in rows for key in (
        "mean_turn_command", "mean_forward_command", "yaw_change_rad", "toward_turn_score", "toward_yaw_score", "path_m", "min_height_m",
    ))
    graph = graph_record(args.data)
    if not same_angles or not same_pipeline or not finite_metrics or not graph["hash_equal"] or not brain_record["hash_equal"] or len(video_frames) == 0 or any(len(group) != len(angles) for group in groups.values()):
        raise AssertionError("VISION-EVAL integrity failed")
    rng = np.random.default_rng(20260829)
    correlations = {}
    for condition in CONDITIONS:
        azimuths = np.asarray([row["azimuth_rad"] for row in groups[condition]], dtype=np.float64)
        turns = np.asarray([row["mean_turn_command"] for row in groups[condition]], dtype=np.float64)
        low, high = correlation_ci(azimuths, turns, rng)
        correlations[condition] = {
            "n_angles": len(azimuths),
            "mean_turn_command": float(turns.mean()),
            "azimuth_vs_turn_pearson_r": _correlation(azimuths, turns),
            "ci_low": low,
            "ci_high": high,
            "ci": "95% percentile bootstrap over fixed azimuth trials",
        }
    comparison_rows = vision_comparisons(groups, rng)
    real_corr = correlations["real"]
    beats_controls = real_corr["azimuth_vs_turn_pearson_r"] > max(
        correlations["shuffled"]["azimuth_vs_turn_pearson_r"], correlations["random"]["azimuth_vs_turn_pearson_r"]
    ) and real_corr["ci_low"] > max(correlations["shuffled"]["ci_high"], correlations["random"]["ci_high"])
    verdict = "VISION-EVAL-PRESENT" if beats_controls else "VISION-EVAL-NULL"
    result = {
        "method": "VISION-EVAL frozen optic-lobe visual-to-DNg100/DNb08 steering sweep",
        "protocol": {
            "angles_deg": list(VISION_ANGLES_DEG),
            "trial_secs": VISION_TRIAL_SECS,
            "prey_distance_m": VISION_PREY_DISTANCE_M,
            "same_retina": True,
            "same_vnc": True,
            "same_body": True,
            "only_condition_difference": "direct visual-to-DNg100/DNb08 edge values; CSR structure, all other brain edges, VNC, and body fixed",
            "brain_readout": "exact DNg100/DNb08 left/right means; turn=-6*(dR-dL)",
            "vnc_to_joint": "mn_to_joint.translate, fixed gains (-0.55, 0.65, 0.65)",
            "h_min_m": H_MIN,
            "learning": False,
        },
        "graph": graph,
        "brain": brain_record,
        "smoke": smoke,
        "correlations": correlations,
        "comparisons": comparison_rows,
        "real_video": {
            "path": str(args.video),
            "angles_deg": list(VISION_VIDEO_ANGLES_DEG),
            "frames": len(video_frames),
        },
        "pre_registered_null": null_text,
        "verdict": verdict,
        "gate_vision_eval": {
            "same_angles": same_angles,
            "same_fixed_pipeline": same_pipeline,
            "finite_metrics": finite_metrics,
            "vnc_graph_hash_equal": graph["hash_equal"],
            "brain_graph_hash_equal": brain_record["hash_equal"],
            "direct_visual_to_dn_edges": brain_record["direct_visual_to_descending_edge_count"],
            "rows": len(rows),
            "real_left_right_video": len(video_frames) > 0,
            "verdict": "GATE VISION-EVAL: PASS",
        },
        "zero_learning": True,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        columns = (
            "condition", "azimuth_deg", "azimuth_rad", "mean_turn_command", "initial_turn_command", "turn_command_std",
            "mean_forward_command", "toward_turn_score", "yaw_change_rad", "toward_yaw_score", "mean_dng100_left",
            "mean_dng100_right", "mean_dnb08_left", "mean_dnb08_right", "path_m", "forward_distance_m", "displacement_m",
            "mean_feet_planted", "min_height_m", "final_height_m", "fell",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    import imageio.v2 as imageio
    imageio.mimwrite(args.video, video_frames, fps=25)
    print(json.dumps({"correlations": correlations, "comparisons": comparison_rows}, sort_keys=True))
    print("GATE VISION-EVAL: PASS (fixed azimuth sweep; frozen visual wiring controls; left/right video)")
    print(f"RESULT: {verdict}")
    print(f"json -> {args.out.with_suffix('.json')}")
    print(f"csv -> {args.out.with_suffix('.csv')}")
    print(f"video -> {args.video}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("COORD-212", "COORD-INHIB", "WAVE-ANTAG", "VISION-EVAL"), default="COORD-212")
    parser.add_argument("--data", type=Path, default=Path("eon-fly-brain/results/path2"))
    parser.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    parser.add_argument("--index_name", default="all_legs_io_index.npz")
    parser.add_argument("--n", type=int, default=PLANNED_N)
    parser.add_argument("--secs", type=float, default=None)
    parser.add_argument("--seed", type=int, default=EPISODE_SEED)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()
    frozen_secs = VISION_TRIAL_SECS if args.phase == "VISION-EVAL" else PLANNED_SECS
    args.secs = frozen_secs if args.secs is None else args.secs
    if args.n < PLANNED_N or args.secs != frozen_secs:
        raise ValueError(f"{args.phase} is frozen at n=20 and secs={frozen_secs}")
    stem = {"COORD-212": "untried_coord212", "COORD-INHIB": "untried_coord_inhib", "WAVE-ANTAG": "untried_wave_antag", "VISION-EVAL": "untried_vision_eval"}[args.phase]
    args.out = args.out or Path(f"eon-fly-brain/results/path2/{stem}")
    args.video = args.video or Path(f"eon-fly-brain/results/path2/{stem}_real_multiview.mp4")
    if args.phase == "COORD-212":
        return run_coord212(args)
    if args.phase == "COORD-INHIB":
        return run_coord_inhib(args)
    if args.phase == "WAVE-ANTAG":
        return run_wave_antag(args)
    return run_vision_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
