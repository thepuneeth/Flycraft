"""Run paired zero-learning Path-2 episodes and fixed controls."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
CODE = HERE.parent
DATA = CODE.parent / "data"
RESULTS = CODE.parent / "results" / "path2"
VNC_DATA = RESULTS

CONDITIONS = ("vnc_open_loop", "vnc_brain", "shuffled_vnc", "random_descending", "path1_hand_cpg")


def episode_specs(n: int, seed: int) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    specs = []
    for episode in range(n):
        yaw = float(rng.uniform(-np.pi, np.pi))
        distance = float(rng.uniform(1.0, 1.8))
        angle = float(rng.uniform(-np.pi, np.pi))
        specs.append({
            "episode": episode,
            "initial_yaw_rad": yaw,
            "prey_x_m": distance * np.cos(angle),
            "prey_y_m": distance * np.sin(angle),
            "episode_seed": int(seed + episode),
        })
    return specs


def command(condition: str, spec: dict[str, float | int], args) -> list[str]:
    prey = [str(spec["prey_x_m"]), str(spec["prey_y_m"])]
    common = [
        "--secs", str(args.secs),
        "--prey", *prey,
        "--initial_yaw", str(spec["initial_yaw_rad"]),
    ]
    if condition == "path1_hand_cpg":
        return [sys.executable, str(CODE / "run_hexapod_chase.py"), *common]
    result_args = [
        sys.executable, str(CODE / "run_vnc_chase.py"),
        "--data", str(DATA),
        "--vnc_data", str(VNC_DATA),
        "--vnc_weights_name", args.weights_name,
        "--vnc_index_name", args.index_name,
        "--vnc_model", "paper",
        "--vnc_drive", "100",
        "--vnc_gW", "0.03",
        "--vnc_substeps", "4",
        "--vnc_heterogeneity",
        "--vnc_seed", "1",
        "--seed", str(spec["episode_seed"]),
        *common,
    ]
    if condition in {"vnc_open_loop", "shuffled_vnc", "random_descending"}:
        result_args += ["--open_loop", "--open_loop_drive", "1"]
    if condition == "shuffled_vnc":
        result_args.append("--shuffle_vnc")
    if condition == "random_descending":
        result_args.append("--random_descending")
    return result_args


def run_one(condition: str, spec: dict[str, float | int], args) -> dict:
    result = subprocess.run(
        command(condition, spec, args),
        cwd=PROJECT,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )
    parsed = next((json.loads(line[len("RESULT_JSON "):]) for line in reversed(result.stdout.splitlines()) if line.startswith("RESULT_JSON ")), None)
    if parsed is None:
        return {
            **spec,
            "condition": condition,
            "returncode": result.returncode,
            "error": (result.stderr + result.stdout)[-2000:],
        }
    return {**spec, "condition": condition, "returncode": result.returncode, **parsed}


def summary(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    result = {}
    for condition in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition and "caught" in row]
        if not group:
            result[condition] = {"n": 0}
            continue
        result[condition] = {
            "n": len(group),
            "caught_rate": float(np.mean([row["caught"] for row in group])),
            "fell_rate": float(np.mean([row["fell"] for row in group])),
            "mean_path_m": float(np.mean([row["path_m"] for row in group])),
            "mean_forward_displacement_m": float(np.mean([row.get("forward_displacement_m", 0.0) for row in group])),
            "mean_displacement_m": float(np.mean([row.get("displacement_m", 0.0) for row in group])),
            "mean_min_distance_m": float(np.mean([row["min_distance_m"] for row in group])),
            "mean_motor_temporal_std": float(np.mean([row.get("motor_temporal_std_mean", 0.0) for row in group])),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    ap.add_argument("--index_name", default="all_legs_io_index.npz")
    ap.add_argument("--out", type=Path, default=RESULTS / "bodyfix_vnc_control_study")
    a = ap.parse_args()
    if a.n < 20:
        raise ValueError("--n must be >=20 for the planned study")
    specs = episode_specs(a.n, a.seed)
    rows = []
    for condition in CONDITIONS:
        for index, spec in enumerate(specs, 1):
            row = run_one(condition, spec, a)
            rows.append(row)
            print(f"{condition} {index}/{a.n} returncode={row['returncode']} caught={row.get('caught', 'ERROR')} fell={row.get('fell', 'ERROR')}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with a.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "n_per_condition": a.n,
        "conditions": list(CONDITIONS),
        "secs": a.secs,
        "seed": a.seed,
        "weights_name": a.weights_name,
        "index_name": a.index_name,
        "fixed_vnc_model": "paper",
        "fixed_vnc_drive": 100.0,
        "fixed_vnc_gW": 0.03,
        "fixed_vnc_substeps": 4,
        "fixed_motor_map_gains": [-0.55, 0.65, 0.65],
        "zero_learning": True,
        "summary": summary(rows),
    }
    a.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], sort_keys=True))
    print(f"csv -> {a.out.with_suffix('.csv')}")
    print(f"json -> {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
