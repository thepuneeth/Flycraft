"""How much of the [SH] core's E0 number is the connectome, and how much is noise?

The corrected core restores [SH]'s Poisson drive, which is stochastic, and that
turns out to matter far more than a fidelity footnote. The same frozen graph at
the same drive gave tripod coherence 0.635 on CPU and 1.082 on CUDA -- not a
numerical discrepancy, but two different draws of the input process, because the
two devices' generators produce different streams from the same seed.

If one realization can move the headline metric by that much, then every E0 and
specificity number measured from a single run is a sample, not a measurement,
and a real-versus-control comparison at n=1 per arm is comparing noise. [SH]
already anticipate this -- `default_params` sets `n_run = 30` and their results
are averaged over trials. Our previous mean-current drive was deterministic, so
this was invisible; it is not a regression, it is a property of the published
model that the shortcut hid.

This measures the spread directly, so the campaign can be told how many trials
an arm needs rather than guessing. Zero training: fixed-map rollouts of a frozen
graph.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "path2"
sys.path.insert(0, str(HERE))

STEPS, SDT, DRIVE = 1500, 0.02, 100.0
KW = dict(weights_name="all_legs_weight_csr.pkl", index_name="all_legs_io_index.npz")

_CFG: dict = {}


def _init(device: str, steps: int, drive: float) -> None:
    _CFG.update(device=device, steps=steps, drive=drive)


def trial(noise_seed: int) -> dict:
    from audit_vnc_legs import leg_scores
    from probe_vnc_phase import phase_metrics, band_power_share
    from vnc_lif_core import ShiuLIFStepper

    stepper = ShiuLIFStepper(RESULTS, seed=1, noise_seed=noise_seed,
                             device=_CFG["device"], **KW)
    current = stepper.descending_current(_CFG["drive"])
    history = np.stack([stepper.step(current).cpu().numpy() for _ in range(_CFG["steps"])])
    scores = leg_scores(history, stepper.io)[..., 0]
    return {"noise_seed": noise_seed,
            "coherence": float(phase_metrics(scores, SDT)["tripod_coherence_scalar"]),
            "band_share": float(band_power_share(scores, SDT)["band_power_share_mean"]),
            "motor_mean": float(history.mean())}


def summarise(values: np.ndarray) -> dict:
    return {"mean": round(float(values.mean()), 4), "sd": round(float(values.std(ddof=1)), 4),
            "min": round(float(values.min()), 4), "max": round(float(values.max()), 4),
            # The quantity that decides whether an n=1 comparison means anything:
            # the standard error a single trial carries.
            "sem": round(float(values.std(ddof=1) / np.sqrt(len(values))), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30, help="[SH] default_params n_run")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--drive", type=float, default=DRIVE)
    ap.add_argument("--out", type=Path, default=RESULTS / "lif_realization_spread.json")
    args = ap.parse_args()

    seeds = list(range(1, args.trials + 1))
    if args.workers > 1:
        context = mp.get_context("spawn")
        with context.Pool(args.workers, initializer=_init,
                          initargs=(args.device, args.steps, args.drive)) as pool:
            rows = list(pool.imap_unordered(trial, seeds))
    else:
        _init(args.device, args.steps, args.drive)
        rows = [trial(seed) for seed in seeds]
    rows.sort(key=lambda row: row["noise_seed"])

    payload = {"device": args.device, "trials": args.trials, "steps": args.steps,
               "drive": args.drive, "rows": rows,
               "coherence": summarise(np.array([r["coherence"] for r in rows])),
               "band_share": summarise(np.array([r["band_share"] for r in rows])),
               "motor_mean": summarise(np.array([r["motor_mean"] for r in rows]))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    for key in ("coherence", "band_share", "motor_mean"):
        print(f"{key:<11} {json.dumps(payload[key], sort_keys=True)}", flush=True)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
