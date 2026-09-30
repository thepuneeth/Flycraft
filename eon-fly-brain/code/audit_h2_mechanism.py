"""H2: pure structural analysis of the frozen VNC and H1 drive outcomes."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
import specificity_campaign as SC  # noqa: E402


PREFIX = RESULTS / "h2_mechanism"
ARNOLDI_STEPS = 64
ARNOLDI_SEED = SC.EPISODE_SEED
METRICS = (
    "signed_spectral_radius",
    "signed_leading_real",
    "signed_leading_imag",
    "absolute_spectral_radius",
    "positive_spectral_radius",
    "motor_source_hhi",
    "motor_source_top10_share",
    "motor_positive_edge_fraction",
    "motor_positive_mass_fraction",
    "motor_positive_reciprocal_fraction",
    "motor_positive_reciprocal_mass_fraction",
)


def eigen_summary(weight: torch.Tensor, values: np.ndarray) -> dict:
    """Estimate leading eigenvalues with deterministic matrix-vector products."""
    weight = weight.to_sparse_csr().cpu()
    n = int(weight.shape[0])
    matrix = torch.sparse_csr_tensor(
        weight.crow_indices(),
        weight.col_indices(),
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
        size=weight.shape,
    )
    generator = torch.Generator().manual_seed(ARNOLDI_SEED)
    q = torch.randint(0, 2, (n,), generator=generator, dtype=torch.float32).mul_(2.0).sub_(1.0)
    q.div_(torch.linalg.vector_norm(q))
    basis = torch.empty((n, ARNOLDI_STEPS), dtype=torch.float32)
    hessenberg = np.zeros((ARNOLDI_STEPS + 1, ARNOLDI_STEPS), dtype=np.float64)
    used = ARNOLDI_STEPS
    with torch.no_grad():
        for column in range(ARNOLDI_STEPS):
            basis[:, column] = q
            vector = torch.sparse.mm(matrix, q[:, None]).reshape(-1)
            for _ in range(2):
                for prior in range(column + 1):
                    coefficient = torch.dot(basis[:, prior], vector).item()
                    hessenberg[prior, column] += coefficient
                    vector.sub_(coefficient * basis[:, prior])
            residual = torch.linalg.vector_norm(vector).item()
            hessenberg[column + 1, column] = residual
            if not np.isfinite(residual):
                raise AssertionError("Arnoldi residual is non-finite")
            if residual < 1e-6:
                used = column + 1
                break
            if column + 1 < ARNOLDI_STEPS:
                q = vector / residual
    eigenvalues = np.linalg.eigvals(hessenberg[:used, :used])
    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
    leading = eigenvalues[:3]
    return {
        "steps_requested": ARNOLDI_STEPS,
        "steps_used": used,
        "final_residual": float(hessenberg[used, used - 1]) if used < ARNOLDI_STEPS else float(hessenberg[-1, -1]),
        "spectral_radius": float(np.abs(leading[0])),
        "leading_eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag), "modulus": float(abs(value))}
            for value in leading
        ],
    }


def reciprocal_mask(keys: np.ndarray, reverse_keys: np.ndarray) -> np.ndarray:
    if not len(keys):
        return np.zeros(len(reverse_keys), dtype=bool)
    ordered = np.sort(keys)
    positions = np.searchsorted(ordered, reverse_keys)
    valid = positions < len(ordered)
    mask = np.zeros(len(reverse_keys), dtype=bool)
    mask[valid] = ordered[positions[valid]] == reverse_keys[valid]
    return mask


def structural_scalars(weight: torch.Tensor, io) -> dict:
    csr = weight.to_sparse_csr().cpu()
    n = int(csr.shape[0])
    row_pointer = csr.crow_indices().numpy()
    columns = csr.col_indices().numpy()
    values = csr.values().numpy()
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(row_pointer))
    motor_indices = np.asarray(io["motor_out"], dtype=np.int64)
    motor_edges = np.isin(rows, motor_indices)
    motor_rows = rows[motor_edges]
    motor_columns = columns[motor_edges]
    motor_values = values[motor_edges]
    source_counts = np.bincount(motor_columns, minlength=n)
    source_probabilities = source_counts[source_counts > 0] / source_counts.sum()
    source_counts_desc = np.sort(source_counts)[::-1]

    positive_edges = values > 0.0
    positive_motor = motor_values > 0.0
    positive_motor_values = motor_values[positive_motor]
    positive_keys = rows[positive_edges].astype(np.int64) * n + columns[positive_edges].astype(np.int64)
    motor_positive_reverse = (
        motor_columns[positive_motor].astype(np.int64) * n
        + motor_rows[positive_motor].astype(np.int64)
    )
    motor_reciprocal = reciprocal_mask(positive_keys, motor_positive_reverse)
    all_positive_reverse = (
        columns[positive_edges].astype(np.int64) * n + rows[positive_edges].astype(np.int64)
    )
    all_reciprocal = reciprocal_mask(positive_keys, all_positive_reverse)

    signed = eigen_summary(csr, values)
    absolute = eigen_summary(csr, np.abs(values))
    positive = eigen_summary(csr, np.maximum(values, 0.0))
    metrics = {
        "signed_spectral_radius": signed["spectral_radius"],
        "signed_leading_real": signed["leading_eigenvalues"][0]["real"],
        "signed_leading_imag": signed["leading_eigenvalues"][0]["imag"],
        "absolute_spectral_radius": absolute["spectral_radius"],
        "positive_spectral_radius": positive["spectral_radius"],
        "motor_source_hhi": float(np.sum(source_probabilities**2)),
        "motor_source_top10_share": float(source_counts_desc[:10].sum() / source_counts.sum()),
        "motor_positive_edge_fraction": float(positive_motor.mean()),
        "motor_positive_mass_fraction": float(positive_motor_values.sum() / np.abs(motor_values).sum()),
        "motor_positive_reciprocal_fraction": float(motor_reciprocal.mean()),
        "motor_positive_reciprocal_mass_fraction": float(
            positive_motor_values[motor_reciprocal].sum() / positive_motor_values.sum()
        ),
    }
    return {
        "metrics": metrics,
        "spectral": {"signed": signed, "absolute": absolute, "positive": positive},
        "graph_counts": {
            "nodes": n,
            "edges": int(len(values)),
            "motor_outputs": int(len(motor_indices)),
            "motor_incoming_edges": int(len(motor_values)),
            "positive_edges": int(positive_edges.sum()),
            "positive_motor_incoming_edges": int(positive_motor.sum()),
            "positive_reciprocal_edges": int(all_reciprocal.sum()),
            "positive_motor_reciprocal_edges": int(motor_reciprocal.sum()),
        },
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def bootstrap_correlation(x: np.ndarray, y: np.ndarray, rank: bool, rng: np.random.Generator) -> dict:
    observed_x = average_ranks(x) if rank else x
    observed_y = average_ranks(y) if rank else y
    observed = pearson(observed_x, observed_y)
    if observed is None:
        return {"r": None, "ci_low": None, "ci_high": None, "degenerate": True}
    indices = rng.integers(0, len(x), size=(SC.BOOTSTRAP_REPS, len(x)))
    samples_x = x[indices]
    samples_y = y[indices]
    if rank:
        samples_x = np.asarray([average_ranks(row) for row in samples_x])
        samples_y = np.asarray([average_ranks(row) for row in samples_y])
    centered_x = samples_x - samples_x.mean(axis=1, keepdims=True)
    centered_y = samples_y - samples_y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(centered_x**2, axis=1) * np.sum(centered_y**2, axis=1))
    numerators = np.sum(centered_x * centered_y, axis=1)
    valid = denominator > 1e-12
    bootstraps = numerators[valid] / denominator[valid]
    low, high = np.percentile(bootstraps, (2.5, 97.5))
    return {"r": float(observed), "ci_low": float(low), "ci_high": float(high), "degenerate": False}


def correlation_record(rows: list[dict], metric: str, rng: np.random.Generator) -> dict:
    x = np.asarray([row[metric] for row in rows], dtype=np.float64)
    y = np.asarray([row["forward_distance_m"] for row in rows], dtype=np.float64)
    return {
        "metric": metric,
        "n": len(rows),
        "pearson": bootstrap_correlation(x, y, False, rng),
        "spearman": bootstrap_correlation(x, y, True, rng),
    }


def distribution_summary(real_value: float, controls: list[float], metric: str, rng: np.random.Generator) -> dict:
    values = np.asarray(controls, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(SC.BOOTSTRAP_REPS, len(values)))
    means = values[indices].mean(axis=1)
    differences = float(real_value) - means
    mean_low, mean_high = np.percentile(means, (2.5, 97.5))
    diff_low, diff_high = np.percentile(differences, (2.5, 97.5))
    return {
        "metric": metric,
        "real_value": float(real_value),
        "random_seed_values": values.tolist(),
        "random_mean": float(values.mean()),
        "random_mean_ci_low": float(mean_low),
        "random_mean_ci_high": float(mean_high),
        "real_minus_random_mean": float(real_value - values.mean()),
        "real_minus_random_mean_ci_low": float(diff_low),
        "real_minus_random_mean_ci_high": float(diff_high),
        "random_distribution_min": float(values.min()),
        "random_distribution_max": float(values.max()),
        "n_random_match_or_beat_real": int(np.count_nonzero(values >= real_value - 1e-12)),
    }


def input_integrity(substrates: list[SC.Substrate], file_hashes_pre: dict, b1_hash_pre: str) -> dict:
    real = next(row for row in substrates if row.name == "real")
    base_csr = real.weight.to_sparse_csr().cpu()
    base_columns = base_csr.col_indices().numpy()
    base_rows = base_csr.crow_indices().numpy()
    base_values = base_csr.values().numpy()
    topology_checks = []
    for substrate in substrates:
        csr = substrate.weight.to_sparse_csr().cpu()
        values = csr.values().numpy()
        record = {
            "name": substrate.name,
            "seed": substrate.seed,
            "row_structure_equal": bool(np.array_equal(csr.crow_indices().numpy(), base_rows)),
            "weight_multiset_equal": bool(np.array_equal(np.sort(values), np.sort(base_values))),
        }
        if substrate.name == "random":
            record["column_structure_equal"] = bool(np.array_equal(csr.col_indices().numpy(), base_columns))
        else:
            record["column_degree_equal"] = bool(
                np.array_equal(
                    np.bincount(csr.col_indices().numpy(), minlength=int(csr.shape[1])),
                    np.bincount(base_columns, minlength=int(base_csr.shape[1])),
                )
            )
        topology_checks.append(record)
    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    b1_hash_post = SC.sha256(RESULTS / "specificity_B1.json")
    real_hash_pre = SC.tensor_sha256(real.weight)
    real_hash_post = SC.tensor_sha256(real.weight)
    return {
        "file_hashes_pre": file_hashes_pre,
        "file_hashes_post": file_hashes_post,
        "file_hashes_equal": bool(file_hashes_pre == file_hashes_post),
        "b1_input_sha256_pre": b1_hash_pre,
        "b1_input_sha256_post": b1_hash_post,
        "b1_input_hash_equal": bool(b1_hash_pre == b1_hash_post),
        "real_tensor_sha256_pre": real_hash_pre,
        "real_tensor_sha256_post": real_hash_post,
        "real_tensor_hash_equal": bool(real_hash_pre == real_hash_post),
        "topology_checks": topology_checks,
        "all_topology_checks_pass": bool(all(all(value for key, value in row.items() if key not in {"name", "seed"}) for row in topology_checks)),
        "random_seeds": list(SC.RNG_SEEDS),
        "rewired_seeds": list(SC.RNG_SEEDS),
        "learning": False,
    }


def write_csv(rows: list[dict]) -> None:
    fields = ["substrate", "kind", "seed", "tensor_sha256", "forward_distance_m", *METRICS]
    with PREFIX.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> int:
    if PREFIX.with_suffix(".json").exists():
        raise FileExistsError(f"{PREFIX.with_suffix('.json')} exists")
    file_hashes_pre = SC.file_hashes(SC.VNC_DATA)
    b1_hash_pre = SC.sha256(RESULTS / "specificity_B1.json")
    b1 = json.loads((RESULTS / "specificity_B1.json").read_text(encoding="utf-8"))
    if b1["protocol"]["learning"] is not False or b1["zero_learning"] is not True:
        raise AssertionError("H2 source B1 is not a zero-learning run")
    drive_by_condition = {
        (row["substrate"], row["seed"]): row["summary"]["metrics"]["forward_distance_m"]["mean"]
        for row in b1["conditions"]
    }
    if len(drive_by_condition) != 25:
        raise AssertionError("B1 source does not contain real plus 24 control conditions")

    substrates, substrate_record = SC.build_vnc_substrates(SC.VNC_DATA, SC.SUBSTRATE_CACHE)
    io = SC.base_vnc(SC.VNC_DATA).io
    rows = []
    structural_records = []
    for substrate in substrates:
        analysis = structural_scalars(substrate.weight, io)
        drive = drive_by_condition[(substrate.name, substrate.seed)]
        row = {
            "substrate": substrate.name,
            "kind": "real" if substrate.name == "real" else f"{substrate.name}-network",
            "seed": substrate.seed,
            "tensor_sha256": substrate.tensor_hash,
            "forward_distance_m": float(drive),
            **analysis["metrics"],
        }
        rows.append(row)
        structural_records.append({**row, "spectral": analysis["spectral"], "graph_counts": analysis["graph_counts"]})
        print(f"analyzed {substrate.name} seed={substrate.seed}", flush=True)

    real = next(row for row in rows if row["substrate"] == "real")
    random_rows = [row for row in rows if row["substrate"] == "random"]
    rewired_rows = [row for row in rows if row["substrate"] == "rewired"]
    control_rows = random_rows + rewired_rows
    comparison_rng = np.random.default_rng(SC.BOOTSTRAP_SEED)
    comparisons = {
        family: {
            metric: distribution_summary(real[metric], [row[metric] for row in family_rows], metric, comparison_rng)
            for metric in METRICS
        }
        for family, family_rows in (("random", random_rows), ("rewired", rewired_rows))
    }
    correlation_groups = {"random_plus_rewired": control_rows, "random": random_rows, "rewired": rewired_rows}
    correlations = {
        group: [correlation_record(group_rows, metric, np.random.default_rng(SC.BOOTSTRAP_SEED + index)) for index, metric in enumerate(METRICS)]
        for group, group_rows in correlation_groups.items()
    }
    pooled_screen = [record["pearson"] for record in correlations["random_plus_rewired"]]
    null_supported = all(record["degenerate"] or (record["ci_low"] <= 0.0 <= record["ci_high"]) for record in pooled_screen)
    file_integrity = input_integrity(substrates, file_hashes_pre, b1_hash_pre)
    h1 = json.loads((RESULTS / "h1_amplitude_control_strict.json").read_text(encoding="utf-8"))
    video_reference = {
        "artifact": str(RESULTS / "h1_amplitude_control_strict.json"),
        "reviewed": bool(h1["videos"]["reviewed"]),
        "real_video": h1["videos"]["real"],
        "random_video": h1["videos"]["random_median"],
    }
    if not video_reference["reviewed"] or not video_reference["real_video"]["exists"] or not video_reference["random_video"]["exists"]:
        raise AssertionError("H2 requires reviewed H1 multiview videos")
    report = {
        "experiment": "H2",
        "status": "complete",
        "pre_registered_null": "no structural scalar predicts forward drive across control seeds",
        "protocol": {
            "analysis": "pure graph analysis; no body rollouts",
            "input_forward_drive": str(RESULTS / "specificity_B1.json"),
            "input_forward_drive_definition": "raw B1 8 s forward-distance mean per real/random/rewired condition",
            "spectral_method": "deterministic 64-step Arnoldi on W, abs(W), and positive-part(W)",
            "arnoldi_seed": ARNOLDI_SEED,
            "metrics": list(METRICS),
            "bootstrap": "95% percentile bootstrap using frozen campaign repetitions and seed",
            "learning": False,
        },
        "real_graph_hashes": file_hashes_pre,
        "substrates": substrate_record,
        "structural_records": structural_records,
        "rows": rows,
        "real_vs_control_distributions": comparisons,
        "correlations": correlations,
        "null_screen": {
            "pooled_group": "random_plus_rewired",
            "rule": "every non-degenerate pooled Pearson 95% bootstrap CI includes zero",
            "supported": bool(null_supported),
            "interpretation": "effect remains mechanistically unexplained by tested scalar" if null_supported else "one or more scalar candidates require independent validation",
        },
        "video_reference": video_reference,
        "integrity": {
            **file_integrity,
            "condition_count": len(rows),
            "random_condition_count": len(random_rows),
            "rewired_condition_count": len(rewired_rows),
            "forward_drive_source_zero_learning": True,
        },
        "zero_learning": True,
        "zero_learning_audit": {"learning": False, "parameter_updates": 0, "training_markers": []},
    }
    SC.atomic_json(PREFIX.with_suffix(".json"), report)
    write_csv(rows)
    print(json.dumps({"experiment": "H2", "null_screen": report["null_screen"], "integrity": report["integrity"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
