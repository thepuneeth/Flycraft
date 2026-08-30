"""Untried zero-learning transfer methods, run one preregistered phase at a time."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

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
    SUMMARY_METRICS,
    calibrate_gain,
    comparisons,
    coupled_phases,
    direct_sources,
    endpoint_legs,
    graph_record,
    run_engine,
    run_direct_episode,
    summarize,
    vnc_phase_source,
)
from mn_to_joint import translate_antagonist  # noqa: E402
from preflight import EXPECTED_SHA256  # noqa: E402
from stepwalk_controls import episode_specs  # noqa: E402
from hexapod_gait import home_stance  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("COORD-212", "COORD-INHIB", "WAVE-ANTAG"), default="COORD-212")
    parser.add_argument("--data", type=Path, default=Path("eon-fly-brain/results/path2"))
    parser.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    parser.add_argument("--index_name", default="all_legs_io_index.npz")
    parser.add_argument("--n", type=int, default=PLANNED_N)
    parser.add_argument("--secs", type=float, default=PLANNED_SECS)
    parser.add_argument("--seed", type=int, default=EPISODE_SEED)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()
    if args.n < PLANNED_N or args.secs != PLANNED_SECS:
        raise ValueError("untried phases are frozen at n=20 and secs=8")
    stem = {"COORD-212": "untried_coord212", "COORD-INHIB": "untried_coord_inhib", "WAVE-ANTAG": "untried_wave_antag"}[args.phase]
    args.out = args.out or Path(f"eon-fly-brain/results/path2/{stem}")
    args.video = args.video or Path(f"eon-fly-brain/results/path2/{stem}_real_multiview.mp4")
    if args.phase == "COORD-212":
        return run_coord212(args)
    if args.phase == "COORD-INHIB":
        return run_coord_inhib(args)
    return run_wave_antag(args)


if __name__ == "__main__":
    raise SystemExit(main())
