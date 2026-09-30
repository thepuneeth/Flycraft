"""Forward locomotion, zero training, across the dynamics we are exploring.

Same fixed anatomical map, same body, same 8 s protocol and the same 20 episode
specs the registered audits use, so every number here is directly comparable to
H1's recorded 0.224 m and to the 0.004 m random baseline.

Nothing is trained. The only thing that varies is which dynamics wrap the frozen
measured graph -- our rate model (SC_CORE=rate) or Shiu et al. 2024 as published
(SC_CORE=lif) -- and how hard the descending neurons are driven.

The [NMF] CPG reference (nmf_cpg.py) reaches 2.208 m on this body at 1.2 Hz, so
that is the ceiling a prescribed tripod buys here. It is a reference, not an arm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import specificity_campaign as SC

SECS = 8.0
METRIC_KEYS = ("forward_distance_m", "displacement_m", "path_m", "min_height_m")


def measure(drive: float, episodes: int, secs: float = SECS) -> dict:
    vnc = SC.base_vnc(SC.VNC_DATA)
    io = vnc.io
    steps = int(round(secs / SC.SIGNAL_SAMPLE_DT))
    current = vnc.descending_current(drive)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)]).astype(np.float32)
    targets = SC.target_history(history, io, {})
    specs = SC.episode_specs(episodes, SC.EPISODE_SEED)
    traces = [SC.run_body_trace(targets, io, spec, secs) for spec in specs]
    # run_body_trace reports its metrics per horizon, not at the top level; take
    # the horizon that matches the episode length so this is the same quantity
    # H1 recorded.
    rows = []
    for trace in traces:
        horizon = min(trace["horizons"].values(),
                      key=lambda h: abs(float(h["horizon_s"]) - secs))
        rows.append(horizon)
    out = {"core": SC.CORE_KIND, "drive": drive, "episodes": len(rows), "secs": secs,
           "horizon_s": float(rows[0]["horizon_s"]),
           "motor_mean": round(float(history.mean()), 5)}
    for key in METRIC_KEYS:
        values = np.array([row[key] for row in rows], dtype=float)
        out[f"{key}_mean"] = round(float(values.mean()), 4)
        out[f"{key}_sd"] = round(float(values.std()), 4)
    out["fell_fraction"] = round(float(np.mean([row.get("fell", False) for row in rows])), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drives", type=float, nargs="+",
                    default=[25.0, 50.0, 100.0, 200.0, 400.0])
    ap.add_argument("--episodes", type=int, default=SC.N_EPISODES)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = []
    for drive in a.drives:
        row = measure(drive, a.episodes)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    best = max(rows, key=lambda r: r["forward_distance_m_mean"])
    print(json.dumps({"core": SC.CORE_KIND, "best_drive": best["drive"],
                      "best_forward_m": best["forward_distance_m_mean"],
                      "recorded_h1_rate_core_m": 0.2240,
                      "random_baseline_m": 0.004,
                      "nmf_cpg_reference_m": 2.208}, sort_keys=True), flush=True)
    if a.out:
        a.out.write_text(json.dumps({"rows": rows, "core": SC.CORE_KIND}, indent=2) + "\n")


if __name__ == "__main__":
    main()
