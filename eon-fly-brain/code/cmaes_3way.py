"""C2: identical frozen CMA-ES search across real, random, and rewired VNC graphs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cmaes_walk import (
    DATA,
    EVALUATION_SEEDS,
    EVALUATION_SECS,
    MAX_GENERATIONS,
    MAX_HOURS,
    POPSIZE,
    RESULTS,
    ROLLOUT_SECS,
    RolloutContext,
    evaluate_best,
    run_search,
)


CONDITIONS = ("connectome", "random", "rewired")
FIRST_WALK_FITNESS = 0.0


def _first_walk_generation(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if float(row["best_fitness"]) > FIRST_WALK_FITNESS:
                return int(row["generation"])
    return None


def _summary(condition: str, search: dict, evaluation: dict, fitness_csv: Path) -> dict:
    metrics = evaluation["metrics"]
    return {
        "condition": condition,
        "best_fitness": float(search["best_fitness"]),
        "generations": int(search["generation"]),
        "first_walk_fitness_threshold": FIRST_WALK_FITNESS,
        "generations_to_first_walk": _first_walk_generation(fitness_csv),
        "best_gene": search["best_gene"],
        "evaluation_fitness": float(evaluation["fitness"]),
        "best_forward_distance_m": float(max(row["forward_distance_m"] for row in metrics)),
        "mean_forward_distance_m": float(np.mean([row["forward_distance_m"] for row in metrics])),
        "mean_measured_e0_coherence": float(np.mean([row["phase"]["tripod_coherence_scalar"] for row in metrics])),
        "max_measured_e0_coherence": float(max(row["phase"]["tripod_coherence_scalar"] for row in metrics)),
        "fitness_csv": str(fitness_csv),
        "evaluation_json": str(RESULTS / f"cmaes_walk_metrics_{condition}.json"),
        "video": evaluation["video"],
    }


def _existing_connectome_search() -> dict | None:
    path = RESULTS / "cmaes_best_gene.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_curve(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ("condition", "generation", "best_fitness", "mean_fitness", "wall_clock_s")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for condition in CONDITIONS:
        curve = [row for row in rows if row["condition"] == condition]
        axis.plot([int(row["generation"]) for row in curve], [float(row["best_fitness"]) for row in curve], label=condition)
    axis.axhline(FIRST_WALK_FITNESS, color="black", linewidth=0.8, linestyle="--")
    axis.set(xlabel="generation", ylabel="best fitness (m)", title="Frozen CMA-ES three-way search")
    axis.legend()
    fig.savefig(output.with_suffix(".png"), dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vnc_data", type=Path, default=RESULTS)
    parser.add_argument("--brain_data", type=Path, default=DATA)
    parser.add_argument("--weights_name", default="e4_sensory_weight_csr.pkl")
    parser.add_argument("--index_name", default="e4_sensory_io_index.npz")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_generations", type=int, default=None)
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--popsize", type=int, default=POPSIZE)
    parser.add_argument("--fresh", action="store_true", help="restart every condition, including existing connectome search")
    args = parser.parse_args()

    summaries = []
    curves = []
    for condition in CONDITIONS:
        search = None if args.fresh or condition != "connectome" else _existing_connectome_search()
        if search is None:
            search = run_search(
                condition,
                args.vnc_data,
                args.brain_data,
                args.weights_name,
                args.index_name,
                args.workers,
                max_generations=args.max_generations if args.max_generations is not None else MAX_GENERATIONS,
                max_hours=args.max_hours if args.max_hours is not None else MAX_HOURS,
                popsize=args.popsize,
                fresh=args.fresh,
            )
        fitness_csv = Path(search["fitness_csv"])
        with fitness_csv.open(newline="", encoding="utf-8") as stream:
            curves.extend({"condition": condition, **row} for row in csv.DictReader(stream))
        gene = np.asarray(search["best_gene"], dtype=np.float64)
        video = RESULTS / f"cmaes_walk_multiview_{condition}.mp4"
        evaluation = evaluate_best(
            condition,
            gene,
            RolloutContext(args.vnc_data, args.brain_data, condition, args.weights_name, args.index_name),
            RESULTS / f"cmaes_walk_metrics_{condition}.json",
            video,
        )
        summaries.append(_summary(condition, search, evaluation, fitness_csv))
        print(json.dumps(summaries[-1], sort_keys=True), flush=True)

    report = {
        "gate": "C2",
        "conditions": CONDITIONS,
        "identical_search_config": {
            "genes": 18,
            "population": args.popsize,
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "rollout_secs": ROLLOUT_SECS,
            "evaluation_secs": EVALUATION_SECS,
            "first_walk_fitness_threshold": FIRST_WALK_FITNESS,
            "frozen_constants": True,
        },
        "summaries": summaries,
        "curve_csv": str(RESULTS / "cmaes_3way.csv"),
        "curve_png": str(RESULTS / "cmaes_3way.png"),
        "stop_before_final_claim": True,
    }
    output = RESULTS / "cmaes_3way.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_curve(curves, RESULTS / "cmaes_3way")
    print("RESULT_JSON " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
