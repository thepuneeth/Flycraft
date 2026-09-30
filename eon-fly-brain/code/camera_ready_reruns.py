"""Camera-ready reruns for BIBM 2026 UGHS reviewer requests (plan: CAMERA_READY_PLAN.md).

One script, four answers, all on the LIF core, CPU, frozen measured weights:
  A    lesions T1/T2/T3/sensory over N_REAL Poisson realizations (R1-2, R1-3)
  NULL zero drive, static mean-posture hold, time-shuffled real signal (R2-1)
  B    rewiring dose response with realization-sampled measured graph (R1-3)
  C    body variations, measured graph over realizations vs shuffled weights (R1-3)
  MAP  pre-registered robustness sweep over the fixed motor map (R1-1, R2-2)

Why realizations: the paper's A-C replayed ONE network realization (noise seed 1)
into 20 start yaws, so their intervals saw body-start variance only. Here the unit
of inference is a network realization; start yaws are averaged inside it.

The network runs open loop, so a motor history depends on graph and noise seed
only, never on map or body. Histories are simulated once, cached, and every map
and body is applied afterwards.

Amplitude rule: every control and lesion arm uses
`amplitude_match(..., allow_unmatchable=True)` against the paired real
realization's joint-target SD -- the rule Experiments B and D already used.
A frozen channel is left alone and counted, never manufactured.

Usage:  SC_CORE is forced to lif here.
  python camera_ready_reruns.py [--workers 8]
  python camera_ready_reruns.py --mark-videos-reviewed
"""
from __future__ import annotations

import os

os.environ["SC_CORE"] = "lif"
os.environ["SC_DEVICE"] = "cpu"

import argparse
import hashlib
import json
import pickle
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import specificity_campaign as SC  # noqa: E402
import audit_h5_lesion_localization as H5  # noqa: E402
from audit_amplitude_control import amplitude_match  # noqa: E402
from audit_h4_dose_response import exact_spearman, partial_rewire  # noqa: E402
from mn_to_joint import translate  # noqa: E402
from vnc_lif_core import ShiuLIFStepper  # noqa: E402

OUT_DIR = SC.RESULTS / "camera_ready"
REPORT = OUT_DIR / "camera_ready_reruns.json"
HIST_CACHE = OUT_DIR / "histories.npz"
PREREG = OUT_DIR / "map_sweep_preregistration.json"
SECS = 8.0
N_REAL = 30                      # matches Experiment D's 30 realizations
N_REAL_SWEEP = 10                # per map; controls stay at 12
REAL_SEEDS = tuple(range(1, N_REAL + 1))
CONTROL_NOISE = {seed: index + 1 for index, seed in enumerate(SC.RNG_SEEDS)}  # control i pairs with realization i+1
LESIONS = (("segment", "somaNeuromere", "T1"), ("segment", "somaNeuromere", "T2"),
           ("segment", "somaNeuromere", "T3"), ("pathway", "superclass", "vnc_sensory"))
ALPHAS = (0.1, 0.25, 0.5, 0.75, 1.0)
BODIES = tuple(dict.fromkeys(SC.B6_PERTURBATIONS))  # 7 unique (three nominal rows are one body)
NOMINAL = ("leg_length", 1.0)
PERM_DRAWS = 200_000
SWEEP_SEED = 20260929

# Pre-registered map family (written to PREREG before any sweep rollout).
GAIN_EPS = (0.10, 0.25, 0.50)
SIGN_FLIP_FRACTIONS = (0.10, 0.25, 0.50)
DRAWS_PER_LEVEL = 4
PERMUTATION_DRAWS = 8


def body_label(body) -> str:
    return f"{body[0]}_x{body[1]:.2f}"


# ---------------------------------------------------------------- workers
_W = {}


def _init_worker() -> None:
    torch.set_num_threads(1)
    with (SC.VNC_DATA / "all_legs_weight_csr.pkl").open("rb") as stream:
        real = pickle.load(stream).to_sparse_csr().cpu()
    _W["real"] = real
    _W["io"] = SC.base_vnc(SC.VNC_DATA).io
    _W["ann"] = H5.annotations_for_nodes(_W["io"])
    _W["specs"] = SC.episode_specs(SC.N_EPISODES, SC.EPISODE_SEED)
    _W["models"] = {}


def _weight(kind: str, arg, graph_seed):
    if kind == "real":
        return _W["real"], None
    if kind == "lesion":
        field, label = arg
        mask = _W["ann"][field].fillna("NA").to_numpy(dtype=str) == label
        return H5.sparse_source_lesion(_W["real"], mask)
    if kind == "rewire":
        return partial_rewire(_W["real"], float(arg), int(graph_seed))
    if kind == "random":
        if "random" not in _W:
            subs, _ = SC.build_vnc_substrates(SC.VNC_DATA, SC.SUBSTRATE_CACHE)
            _W["random"] = {s.seed: s.weight for s in subs if s.name == "random"}
        return _W["random"][int(graph_seed)], None
    raise ValueError(kind)


def history_job(job):
    key, kind, arg, graph_seed, noise_seed = job
    weight, graph_record = _weight(kind, arg, graph_seed)
    vnc = ShiuLIFStepper(SC.VNC_DATA, weights_name="all_legs_weight_csr.pkl",
                         index_name="all_legs_io_index.npz", seed=SC.VNC_SEED,
                         noise_seed=int(noise_seed), device="cpu")
    if kind != "real":
        vnc.W = weight
    current = vnc.descending_current(SC.BASE_DRIVE)
    steps = int(round(SECS / SC.SIGNAL_SAMPLE_DT))
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)]).astype(np.float32)
    if not np.isfinite(history).all():
        raise AssertionError(f"non-finite history {key}")
    return key, history, graph_record


def rollout_job(job):
    key, body, targets, video = job
    body = tuple(body)
    if body not in _W["models"]:
        _W["models"][body] = SC.make_body_model(body)
    model, record = _W["models"][body]
    forward, fell = [], []
    for index, spec in enumerate(_W["specs"]):
        trace = SC.run_body_trace(targets, _W["io"], spec, SECS, perturbation=body, model=model,
                                  body_record=record, video=Path(video) if (video and index == 0) else None)
        forward.append(trace["horizons"]["8s"]["forward_distance_m"])
        fell.append(trace["horizons"]["8s"]["fell"])
    return key, np.asarray(forward), int(sum(fell))


# ---------------------------------------------------------------- maps
def io_view(io) -> dict:
    return {name: np.asarray(io[name]) for name in io.files}


def map_targets(history: np.ndarray, io: dict, gains=SC.MAP_GAINS) -> np.ndarray:
    return np.stack([translate(row, io, gains=gains) for row in history]).astype(np.float32)


def build_sweep_maps(io: dict) -> list[dict]:
    rng = np.random.default_rng(SWEEP_SEED)
    base = {"motor_leg": io["motor_leg"].copy(), "motor_joint": io["motor_joint"].copy(), "motor_sign": io["motor_sign"].copy()}
    maps = []
    for eps in GAIN_EPS:
        for draw in range(DRAWS_PER_LEVEL):
            direction = rng.choice((-1.0, 1.0), size=3)
            gains = tuple(float(g * (1.0 + eps * d)) for g, d in zip(SC.MAP_GAINS, direction))
            maps.append({"family": "gain_jitter", "level": eps, "draw": draw, "gains": gains, "labels": base})
    signed = np.flatnonzero(base["motor_sign"] != 0)
    for fraction in SIGN_FLIP_FRACTIONS:
        for draw in range(DRAWS_PER_LEVEL):
            flip = rng.choice(signed, size=int(round(fraction * len(signed))), replace=False)
            sign = base["motor_sign"].copy()
            sign[flip] = -sign[flip]
            maps.append({"family": "sign_flip", "level": fraction, "draw": draw, "gains": SC.MAP_GAINS, "labels": {**base, "motor_sign": sign}})
    for draw in range(PERMUTATION_DRAWS):
        joint = base["motor_joint"].copy()
        for leg in np.unique(base["motor_leg"]):
            members = np.flatnonzero(base["motor_leg"] == leg)
            joint[members] = joint[rng.permutation(members)]
        maps.append({"family": "within_leg_joint_permutation", "level": None, "draw": draw, "gains": SC.MAP_GAINS, "labels": {**base, "motor_joint": joint}})
    for draw in range(PERMUTATION_DRAWS):
        leg = base["motor_leg"][rng.permutation(len(base["motor_leg"]))]
        maps.append({"family": "cross_leg_permutation", "level": None, "draw": draw, "gains": SC.MAP_GAINS, "labels": {**base, "motor_leg": leg}})
    return maps


def map_digest(item: dict) -> str:
    digest = hashlib.sha256(json.dumps([item["family"], item["level"], item["draw"], list(item["gains"])]).encode())
    for name in ("motor_leg", "motor_joint", "motor_sign"):
        digest.update(np.ascontiguousarray(item["labels"][name]).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------- stats
def perm_greater(a: np.ndarray, b: np.ndarray, seed: int) -> float:
    """One-sided two-sample permutation p for mean(a) > mean(b), Monte Carlo."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate((a, b))
    rng = np.random.default_rng(seed)
    observed = a.mean() - b.mean()
    hits = 1
    for _ in range(PERM_DRAWS // 1000):
        idx = np.argsort(rng.random((1000, len(pooled))), axis=1)
        s = pooled[idx]
        hits += int(np.sum(s[:, :len(a)].mean(1) - s[:, len(a):].mean(1) >= observed - 1e-15))
    return hits / (PERM_DRAWS + 1)


def sign_flip_greater(d: np.ndarray, seed: int) -> float:
    """One-sided paired sign-flip p for mean(d) > 0, Monte Carlo (n=30 is too big to enumerate)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    hits = 1
    for _ in range(PERM_DRAWS // 1000):
        signs = rng.choice((-1.0, 1.0), size=(1000, len(d)))
        hits += int(np.sum((signs * d).mean(1) >= d.mean() - 1e-15))
    return hits / (PERM_DRAWS + 1)


def summary(values, seed: int = SC.BOOTSTRAP_SEED) -> dict:
    values = np.asarray(values, float)
    out = SC.metric_summary(values, np.random.default_rng(seed))
    return {**out, "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()), "n_positive": int((values > 0).sum())}


def variance_components(per_unit: list[np.ndarray]) -> dict:
    """Split episode (start yaw) variance from realization variance."""
    arr = np.stack(per_unit)
    return {"within_realization_episode_sd": float(np.sqrt(arr.var(axis=1, ddof=1).mean())),
            "across_realization_sd_of_means": float(arr.mean(axis=1).std(ddof=1)),
            "n_realizations": int(arr.shape[0]), "n_episodes": int(arr.shape[1])}


# ---------------------------------------------------------------- main
def mark_videos_reviewed() -> int:
    report = json.loads(REPORT.read_text())
    report["videos"]["reviewed"] = True
    report["videos"]["review_note"] = "owner watched real, T2-lesion and zero-drive four-view sequences"
    SC.atomic_json(REPORT, report)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--mark-videos-reviewed", action="store_true")
    args = parser.parse_args()
    if args.mark_videos_reviewed:
        return mark_videos_reviewed()
    if REPORT.exists():
        raise FileExistsError(f"{REPORT} exists")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    file_hashes_pre = SC.file_hashes(SC.VNC_DATA)
    with (SC.VNC_DATA / "all_legs_weight_csr.pkl").open("rb") as stream:
        real_hash_pre = SC.tensor_sha256(pickle.load(stream).to_sparse_csr().cpu())
    io = io_view(SC.base_vnc(SC.VNC_DATA).io)
    pool = get_context("spawn").Pool(args.workers, initializer=_init_worker)

    # ---- histories (cached so a crash resumes instead of restarting)
    jobs = [(f"real|{s}", "real", None, None, s) for s in REAL_SEEDS]
    jobs += [(f"lesion_{lab}|{s}", "lesion", (field, lab), None, s) for _, field, lab in LESIONS for s in REAL_SEEDS]
    jobs += [(f"random_{g}|{n}", "random", None, g, n) for g, n in CONTROL_NOISE.items()]
    jobs += [(f"rewire_{a:g}_{g}|{n}", "rewire", a, g, n) for a in ALPHAS for g, n in CONTROL_NOISE.items()]
    histories, graph_records = {}, {}
    if HIST_CACHE.exists():
        with np.load(HIST_CACHE) as cache:
            histories = {key: cache[key] for key in cache.files}
        graph_records = json.loads((OUT_DIR / "graph_records.json").read_text())
    todo = [job for job in jobs if job[0] not in histories]
    print(f"histories: {len(histories)} cached, {len(todo)} to run", flush=True)
    for done, (key, history, record) in enumerate(pool.imap_unordered(history_job, todo), 1):
        histories[key] = history
        if record is not None:
            graph_records[key.split("|")[0]] = record
        if done % 20 == 0 or done == len(todo):
            SC.atomic_npz(HIST_CACHE, histories)
            (OUT_DIR / "graph_records.json").write_text(json.dumps(graph_records, indent=1, default=SC.json_default))
            print(f"  history {done}/{len(todo)}  {time.time() - started:.0f}s", flush=True)

    # ---- nominal-map targets
    real_t = {s: map_targets(histories[f"real|{s}"], io) for s in REAL_SEEDS}
    real_sd = {s: real_t[s].std(axis=0) for s in REAL_SEEDS}
    stance = translate(np.zeros(len(io["motor_out"]), dtype=np.float32), io, gains=SC.MAP_GAINS)
    zero = np.tile(stance, (int(SECS * SC.CTRL_HZ), 1)).astype(np.float32)

    def matched(history, reference_sd, labels=io, gains=SC.MAP_GAINS):
        targets, record = amplitude_match(map_targets(history, labels, gains), reference_sd, allow_unmatchable=True)
        return targets, int(record["unmatchable_channels"])

    signals, frozen = {}, {}
    for s in REAL_SEEDS:
        signals[("real", s, NOMINAL)] = real_t[s]
        signals[("mean_hold", s, NOMINAL)] = np.tile(real_t[s].mean(axis=0), (len(real_t[s]), 1)).astype(np.float32)
        signals[("time_shuffled", s, NOMINAL)] = real_t[s][np.random.default_rng(s).permutation(len(real_t[s]))]
        for _, _, lab in LESIONS:
            t, n = matched(histories[f"lesion_{lab}|{s}"], real_sd[s])
            signals[(f"lesion_{lab}", s, NOMINAL)] = t
            frozen.setdefault(f"lesion_{lab}", []).append(n)
    for g, n in CONTROL_NOISE.items():
        t, k = matched(histories[f"random_{g}|{n}"], real_sd[n])
        frozen.setdefault("random", []).append(k)
        for a in ALPHAS:
            ta, ka = matched(histories[f"rewire_{a:g}_{g}|{n}"], real_sd[n])
            signals[(f"rewire_{a:g}", g, NOMINAL)] = ta
            frozen.setdefault(f"rewire_{a:g}", []).append(ka)
        for body in BODIES:
            signals[("random", g, body)] = t
    for body in BODIES:
        signals[("zero_drive", 0, body)] = zero
        if body != NOMINAL:
            for s in REAL_SEEDS:
                signals[("real", s, body)] = real_t[s]

    # ---- videos on the watched arms (first episode only)
    video_keys = {("real", 1, NOMINAL): OUT_DIR / "cr_real_seed1_multiview.mp4",
                  ("lesion_T2", 1, NOMINAL): OUT_DIR / "cr_lesion_T2_seed1_multiview.mp4",
                  ("zero_drive", 0, NOMINAL): OUT_DIR / "cr_zero_drive_multiview.mp4"}

    def run(sig: dict) -> dict:
        tasks = [(key, key[2], targets, str(video_keys[key]) if key in video_keys else None) for key, targets in sig.items()]
        out = {}
        for done, (key, forward, falls) in enumerate(pool.imap_unordered(rollout_job, tasks, chunksize=2), 1):
            out[key] = (forward, falls)
            if done % 100 == 0 or done == len(tasks):
                print(f"  rollouts {done}/{len(tasks)}  {time.time() - started:.0f}s", flush=True)
        return out

    print(f"main rollouts: {len(signals)} signals x {SC.N_EPISODES} episodes", flush=True)
    res = run(signals)
    mean = {key: float(v[0].mean()) for key, v in res.items()}
    total_falls = sum(v[1] for v in res.values())

    # ---- NULL (R2-1)
    zero_by_body = {body_label(b): summary(res[("zero_drive", 0, b)][0]) for b in BODIES}
    z0 = mean[("zero_drive", 0, NOMINAL)]
    real_nom = np.array([mean[("real", s, NOMINAL)] for s in REAL_SEEDS])
    null = {
        "zero_drive_by_body": zero_by_body,
        "real": summary(real_nom),
        "real_minus_zero_drive": summary(real_nom - z0),
        "mean_posture_hold": summary([mean[("mean_hold", s, NOMINAL)] for s in REAL_SEEDS]),
        "time_shuffled_real": summary([mean[("time_shuffled", s, NOMINAL)] for s in REAL_SEEDS]),
        "decomposition_mean_m": {
            "posture_component": float(np.mean([mean[("mean_hold", s, NOMINAL)] for s in REAL_SEEDS]) - z0),
            "fluctuation_component": float(real_nom.mean() - np.mean([mean[("mean_hold", s, NOMINAL)] for s in REAL_SEEDS])),
            "temporal_order_component": float(real_nom.mean() - np.mean([mean[("time_shuffled", s, NOMINAL)] for s in REAL_SEEDS])),
        },
        "p_real_gt_zero_drive": sign_flip_greater(real_nom - z0, 1),
        "p_real_gt_time_shuffled": sign_flip_greater(real_nom - np.array([mean[("time_shuffled", s, NOMINAL)] for s in REAL_SEEDS]), 2),
    }

    # ---- A (R1-2, R1-3)
    lesions = []
    for _, field, lab in LESIONS:
        les = np.array([mean[(f"lesion_{lab}", s, NOMINAL)] for s in REAL_SEEDS])
        loss = real_nom - les
        lesions.append({"lesion": lab, "annotation_field": field, "graph": graph_records.get(f"lesion_{lab}"),
                        "frozen_channels_per_realization": frozen[f"lesion_{lab}"],
                        "lesion_forward": summary(les), "loss": summary(loss),
                        "loss_variance": variance_components([res[("real", s, NOMINAL)][0] - res[(f"lesion_{lab}", s, NOMINAL)][0] for s in REAL_SEEDS]),
                        "p_raw": sign_flip_greater(loss, 10 + len(lesions))})
    for item, p in zip(lesions, SC.holm_adjust([x["p_raw"] for x in lesions])):
        item["p_holm"] = float(p)
    exp_a = {"real": summary(real_nom),
             "real_variance": variance_components([res[("real", s, NOMINAL)][0] for s in REAL_SEEDS]),
             "lesions": lesions,
             "test": "one-sided paired sign-flip over realizations (Monte Carlo), Holm over 4 lesions"}

    # ---- B (R1-3)
    dose = [{"alpha": 0.0, **summary(real_nom)}]
    for a in ALPHAS:
        vals = [mean[(f"rewire_{a:g}", g, NOMINAL)] for g in CONTROL_NOISE]
        dose.append({"alpha": a, **summary(vals), "frozen_channels_per_seed": frozen[f"rewire_{a:g}"],
                     "real_mean_above_ci_high": bool(real_nom.mean() > summary(vals)["ci_high"]),
                     "p_real_gt_fraction": perm_greater(real_nom, np.asarray(vals), 100 + int(a * 100))})
    for item, p in zip(dose[1:], SC.holm_adjust([x["p_real_gt_fraction"] for x in dose[1:]])):
        item["p_holm"] = float(p)
    exp_b = {"dose": dose, "spearman_rho": exact_spearman([d["mean"] for d in dose]), "zero_drive": z0}

    # ---- C (R1-3)
    bodies = []
    for b in BODIES:
        r = np.array([mean[("real", s, b)] for s in REAL_SEEDS])
        c = np.array([mean[("random", g, b)] for g in CONTROL_NOISE])
        zb = mean[("zero_drive", 0, b)]
        bodies.append({"body": body_label(b), "real": summary(r), "shuffled_weight": summary(c), "zero_drive": zb,
                       "real_minus_zero_drive": summary(r - zb),
                       "controls_at_or_above_real_mean": int((c >= r.mean()).sum()),
                       "p_raw": perm_greater(r, c, 200 + len(bodies))})
    for item, p in zip(bodies, SC.holm_adjust([x["p_raw"] for x in bodies])):
        item["p_holm"] = float(p)
    exp_c = {"bodies": bodies, "random_frozen_channels_per_seed": frozen["random"],
             "test": "one-sided two-sample permutation, 30 realizations vs 12 shuffled-weight graphs, Holm over 7 unique bodies"}

    # ---- MAP sweep (R1-1, R2-2): pre-register, then run
    maps = build_sweep_maps(io)
    prereg = {"written_at_unix": time.time(), "script_sha256": hashlib.sha256(HERE.read_bytes()).hexdigest(),
              "families": {"gain_jitter": {"eps": GAIN_EPS, "draws_per_eps": DRAWS_PER_LEVEL, "rule": "each gain x (1 + eps*u), u in {-1,+1} per gain"},
                           "sign_flip": {"fractions": SIGN_FLIP_FRACTIONS, "draws_per_fraction": DRAWS_PER_LEVEL, "rule": "flip motor_sign on that fraction of signed channels"},
                           "within_leg_joint_permutation": {"draws": PERMUTATION_DRAWS, "rule": "permute motor_joint among channels of the same leg"},
                           "cross_leg_permutation": {"draws": PERMUTATION_DRAWS, "rule": "permute motor_leg across all channels"}},
              "per_map": f"measured graph at realizations 1..{N_REAL_SWEEP} vs 12 shuffled-weight graphs, each amplitude-matched to its paired realization under the same map",
              "endpoints": {"primary": "fraction of maps per family with mean(real) - mean(shuffled) > 0",
                            "strict": "fraction of maps per family where mean(real) exceeds every shuffled-weight graph mean",
                            "zero_drive": "fraction of maps with mean(real) > zero-drive (zero input is home stance under every map)"},
              "sweep_seed": SWEEP_SEED, "maps": [{"index": i, "family": m["family"], "level": m["level"], "draw": m["draw"], "gains": m["gains"], "sha256": map_digest(m)} for i, m in enumerate(maps)]}
    SC.atomic_json(PREREG, prereg)
    print(f"preregistered {len(maps)} maps -> {PREREG}", flush=True)

    sweep_signals, sweep_frozen = {}, {}
    for i, m in enumerate(maps):
        labels = {**io, **m["labels"]}
        rt = {s: map_targets(histories[f"real|{s}"], labels, m["gains"]) for s in REAL_SEEDS[:N_REAL_SWEEP]}
        for s, t in rt.items():
            sweep_signals[(f"map{i}_real", s, NOMINAL)] = t
        for g, n in CONTROL_NOISE.items():
            ref = rt[n] if n in rt else map_targets(histories[f"real|{n}"], labels, m["gains"])
            t, k = matched(histories[f"random_{g}|{n}"], ref.std(axis=0), labels, m["gains"])
            sweep_signals[(f"map{i}_random", g, NOMINAL)] = t
            sweep_frozen.setdefault(i, []).append(k)
    print(f"sweep rollouts: {len(sweep_signals)} signals", flush=True)
    sres = run(sweep_signals)
    per_map = []
    for i, m in enumerate(maps):
        r = np.array([sres[(f"map{i}_real", s, NOMINAL)][0].mean() for s in REAL_SEEDS[:N_REAL_SWEEP]])
        c = np.array([sres[(f"map{i}_random", g, NOMINAL)][0].mean() for g in CONTROL_NOISE])
        per_map.append({"index": i, "family": m["family"], "level": m["level"], "draw": m["draw"],
                        "real_mean": float(r.mean()), "shuffled_mean": float(c.mean()), "difference": float(r.mean() - c.mean()),
                        "primary_pass": bool(r.mean() > c.mean()), "strict_pass": bool((r.mean() > c).all()),
                        "above_zero_drive": bool(r.mean() > z0), "frozen_channels": sweep_frozen[i]})
    families = {}
    for fam in prereg["families"]:
        rows = [p for p in per_map if p["family"] == fam]
        families[fam] = {"n_maps": len(rows),
                         "primary_fraction": float(np.mean([p["primary_pass"] for p in rows])),
                         "strict_fraction": float(np.mean([p["strict_pass"] for p in rows])),
                         "above_zero_drive_fraction": float(np.mean([p["above_zero_drive"] for p in rows])),
                         "difference_range": [float(min(p["difference"] for p in rows)), float(max(p["difference"] for p in rows))]}

    # ---- integrity
    file_hashes_post = SC.file_hashes(SC.VNC_DATA)
    with (SC.VNC_DATA / "all_legs_weight_csr.pkl").open("rb") as stream:
        real_hash_post = SC.tensor_sha256(pickle.load(stream).to_sparse_csr().cpu())
    pool.close()
    integrity = {"file_hashes_equal": file_hashes_pre == file_hashes_post, "real_tensor_sha256": real_hash_pre,
                 "real_tensor_hash_unchanged": real_hash_pre == real_hash_post, "zero_learning": True,
                 "history_cache_sha256": hashlib.sha256(HIST_CACHE.read_bytes()).hexdigest()}
    if not (integrity["file_hashes_equal"] and integrity["real_tensor_hash_unchanged"]):
        raise AssertionError("frozen measured graph changed during run")

    report = {
        "experiment": "camera_ready_reruns", "status": "PENDING-OWNER-REVIEW",
        "protocol": {"core": "Shiu LIF, CPU", "secs": SECS, "drive_hz": SC.BASE_DRIVE, "n_realizations": N_REAL,
                     "realization_seeds": list(REAL_SEEDS), "control_noise_pairing": {str(k): v for k, v in CONTROL_NOISE.items()},
                     "episodes_per_signal": SC.N_EPISODES, "episode_seed": SC.EPISODE_SEED, "map_gains": list(SC.MAP_GAINS),
                     "amplitude_rule": "amplitude_match(allow_unmatchable=True) to paired real realization SD; frozen channels counted, not scaled",
                     "unit_of_inference": "network realization (episode starts averaged within)", "learning": False},
        "null_arms": null, "experiment_A": exp_a, "experiment_B": exp_b, "experiment_C": exp_c,
        "map_sweep": {"preregistration": str(PREREG), "families": families, "per_map": per_map},
        "falls_total": int(total_falls), "rollouts_total": int((len(res) + len(sres)) * SC.N_EPISODES),
        "videos": {"paths": [str(p) for p in video_keys.values()], "reviewed": False, "review_note": "review required before any claim"},
        "integrity": integrity, "wallclock_s": round(time.time() - started, 1),
    }
    SC.atomic_json(REPORT, report)
    print(f"wrote {REPORT} in {report['wallclock_s']}s", flush=True)
    return 0


def _self_check() -> None:
    """Stat helpers behave: clear effect -> small p, no effect -> large p."""
    global PERM_DRAWS
    PERM_DRAWS = 20_000
    assert sign_flip_greater(np.full(10, 1.0), 0) < 0.01
    assert sign_flip_greater(np.array([1.0, -1.0] * 5), 0) > 0.4
    assert perm_greater(np.arange(10) + 100.0, np.arange(12), 0) < 0.01
    assert perm_greater(np.arange(10.0), np.arange(10.0), 0) > 0.4


if __name__ == "__main__":
    if os.environ.get("CR_SELF_CHECK"):
        _self_check()
        print("self-check ok")
        raise SystemExit(0)
    raise SystemExit(main())
