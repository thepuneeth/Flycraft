"""E0 specificity under [SH] dynamics, using the project's real controls.

My first attempt permuted columns, which scatters column degree and nearly
silences a threshold model -- the shuffles were dead, not uncoordinated, so the
comparison was void. `reservoir_3way.degree_preserving_rewire` preserves row AND
column degree plus the weight multiset, so every motor neuron keeps its
in-degree and the network stays alive; `randomize_weights` keeps the structure
and reassigns the signed weights. Those are the controls the rest of the project
already uses, and they are the ones this question needs.

Activity is reported for every arm, because a control is only a control while it
is as active as the thing it controls.

One rollout per arm is not a measurement
----------------------------------------
This sweep originally ran one rollout per arm, which was defensible while the
drive was a deterministic mean current. Restoring [SH]'s Poisson source made the
core stochastic, and `lif_realization_spread` measured what that costs: over 30
realizations of the SAME frozen real graph, tripod coherence has mean 1.0198 and
sd 0.3428, spanning 0.4461 to 1.6111. A 34% coefficient of variation from the
input noise alone.

At n=1 per arm the maximum over 200 controls is therefore an order statistic of
noise, and it beats a single real draw roughly whenever the real draw lands below
its own median. The earlier "1 of 200 rewires matched or beat real (max 1.2447
against real 0.9804)" is exactly that artifact: 1.2447 sits inside the real
graph's own realization distribution. So every arm is now averaged over TRIALS,
which is what [SH] do (`default_params` sets n_run = 30), and the comparison is
between per-arm means with the real arm's realization sd reported beside them.

Activity is not affected -- motor rate has sd 0.0005 across the same 30
realizations, and band share 0.0112. It is the phase metric that needs the
trials.

Throughput
----------
401 probes of 1500 steps each is the campaign's long pole, and it is
embarrassingly parallel: every (kind, seed) is independent. Three things were
leaving most of the box idle, and all three are fixed here rather than by asking
for a bigger pod:

  * every probe rebuilt a `ShiuLIFStepper` from the pickle on disk. The graph is
    the same 485k-edge matrix every time; only the control weights change. One
    stepper per worker now, with the weights swapped through `.W`.
  * the whole sweep ran on CPU while the 4090 sat at 0% and 1 MiB. The core is
    where all the time goes (3.07 s of rollout against 0.18 s of MuJoCo per
    episode) and the sparse matmul is ~4x faster on the GPU: 2.75 ms per control
    step against 10.80 on two CPU threads.
  * it ran serially. `--workers` fans the seeds out; one stream is nowhere near
    saturating the GPU, so processes scale until the host runs out of cores.

Results are keyed by (kind, seed) and sorted before they are written, so the
artifact does not depend on which worker finished first.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

# Resolve relative to this file, not to $HOME: the pod checks the tree out at
# /root/eon-fly-brain while the laptop has it under ~/Flycraft.
HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "path2"
sys.path.insert(0, str(HERE))

from audit_vnc_legs import leg_scores  # noqa: E402
from probe_vnc_phase import phase_metrics, band_power_share  # noqa: E402
from reservoir_3way import degree_preserving_rewire, randomize_weights  # noqa: E402
from vnc_lif_core import ShiuLIFStepper  # noqa: E402

STEPS, SDT, DRIVE = 1500, 0.02, 100.0
KW = dict(weights_name="all_legs_weight_csr.pkl", index_name="all_legs_io_index.npz")
SEEDS = tuple(range(11, 211))   # 200 controls per arm: p ~ 1/401 if none beat real

_WORKER: dict = {}


def _init_worker(device: str) -> None:
    stepper = ShiuLIFStepper(RESULTS, seed=1, device=device, **KW)
    # `.W` is signed synapse counts on both cores -- the mV scaling is the LIF
    # core's own business and happens inside the setter. Kept on the CPU here
    # because the rewire routines are CPU-side and run once per probe.
    _WORKER["stepper"] = stepper
    _WORKER["counts"] = stepper.W.cpu()
    _WORKER["current"] = stepper.descending_current(DRIVE)


def _weights_for(kind: str, seed: int):
    if kind == "real":
        return _WORKER["counts"]
    if kind == "rewired":
        return degree_preserving_rewire(_WORKER["counts"], seed)
    if kind == "random":
        return randomize_weights(_WORKER["counts"], seed)
    raise ValueError(f"unknown control arm {kind!r}")


def probe(task: tuple[str, int, int]) -> dict:
    """One arm, averaged over `trials` realizations of the Poisson drive."""
    kind, seed, trials = task
    stepper = _WORKER["stepper"]
    stepper.W = _weights_for(kind, seed)
    current = _WORKER["current"]

    coherence, band, rate, active = [], [], [], []
    for realization in range(trials):
        # A fresh realization, and the same set of them for every arm, so the
        # arms differ by their graph and by nothing else.
        stepper.core.generator.manual_seed(1000 + realization)
        stepper.reset()
        history = np.stack([stepper.step(current).cpu().numpy() for _ in range(STEPS)])
        scores = leg_scores(history, stepper.io)[..., 0]
        coherence.append(float(phase_metrics(scores, SDT)["tripod_coherence_scalar"]))
        band.append(float(band_power_share(scores, SDT)["band_power_share_mean"]))
        rate.append(float(history.mean()) * stepper.core.rate_ceiling_hz)
        active.append(float((history.mean(axis=0) > 1e-6).mean()))

    coherence = np.asarray(coherence)
    return dict(kind=kind, seed=seed, trials=trials,
                motor_hz=round(float(np.mean(rate)), 3),
                frac_active=round(float(np.mean(active)), 4),
                coherence=round(float(coherence.mean()), 4),
                coherence_sd=round(float(coherence.std(ddof=1)) if trials > 1 else 0.0, 4),
                band_share=round(float(np.mean(band)), 4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, nargs="?", default=RESULTS / "e0_lif_controls.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--trials", type=int, default=10,
                    help="Poisson realizations averaged per arm; 1 reproduces the old, invalid design")
    ap.add_argument("--real-trials", type=int, default=30,
                    help="realizations for the real arm ([SH] default_params n_run)")
    args = ap.parse_args()

    seeds = SEEDS[:args.seeds]
    tasks = [("real", 1, args.real_trials)]
    tasks += [(kind, seed, args.trials) for kind in ("rewired", "random") for seed in seeds]

    rows: list[dict] = []
    if args.workers > 1:
        # fork does not survive a CUDA context; every worker builds its own.
        context = mp.get_context("spawn")
        with context.Pool(args.workers, initializer=_init_worker, initargs=(args.device,)) as pool:
            for row in pool.imap_unordered(probe, tasks, chunksize=1):
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    else:
        _init_worker(args.device)
        for task in tasks:
            row = probe(task)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    rows.sort(key=lambda row: (row["kind"], row["seed"]))
    real = next(row for row in rows if row["kind"] == "real")
    summary = []
    for kind in ("rewired", "random"):
        values = [row["coherence"] for row in rows if row["kind"] == kind]
        summary.append({"kind": kind,
                        "real": real["coherence"], "real_realization_sd": real["coherence_sd"],
                        "mean": round(float(np.mean(values)), 4), "max": max(values),
                        "n_match_or_beat_real": sum(1 for v in values if v >= real["coherence"]),
                        "n": len(values), "trials_per_arm": args.trials,
                        # A control beating the real arm means nothing unless it
                        # beats it by more than the real arm moves on its own.
                        "n_beat_real_by_1sd": sum(
                            1 for v in values if v >= real["coherence"] + real["coherence_sd"])})
        print(json.dumps(summary[-1], sort_keys=True), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": args.device, "steps": STEPS, "drive": DRIVE,
                                    "trials_per_control": args.trials, "real_trials": args.real_trials,
                                    "summary": summary, "rows": rows}, indent=2) + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
