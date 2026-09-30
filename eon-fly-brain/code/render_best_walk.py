"""Render the best zero-training rollout, and the reference ceiling beside it.

`walk_gain_envelope` reports numbers; this shows what those numbers look like,
which for a result this weak is the honest thing to publish. The [SH] core's best
arm nets 0.0764 m of forward travel in 8 s while covering 0.7763 m of path, and a
10:1 path-to-forward ratio is not something a table communicates.

The NMF CPG arm is rendered alongside on the SAME body and the SAME camera as a
labelled ceiling -- it prescribes the rhythm and the inter-leg coordination
outright, so it is not an arm of the same experiment and must never be mixed into
a connectome claim (see `nmf_cpg`). It is here so a viewer can see what walking
on this hexapod looks like when something actually walks.

Zero training throughout: fixed-map rollouts of a frozen graph.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "path2"
sys.path.insert(0, str(HERE))

from hexapod_gait import resolve_stance  # noqa: E402

SECS = 8.0
DRIVE = 100.0


def core_targets(core: str, secs: float, drive: float, gain: float):
    os.environ["SC_CORE"] = core
    import specificity_campaign as SC
    SC = importlib.reload(SC)
    vnc = SC.base_vnc(SC.VNC_DATA)
    steps = int(round(secs / SC.SIGNAL_SAMPLE_DT))
    current = vnc.descending_current(drive)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)]).astype(np.float32)
    targets = SC.target_history(history, vnc.io, {})
    if gain != 1.0:
        stance = resolve_stance(None).reshape(18)[None, :]
        targets = (stance + (targets - stance) * gain).astype(np.float32)
    return targets, vnc.io, SC


def cpg_targets(secs: float, SC, frequency: float):
    """The [NMF] reference arm, sampled onto the same control grid."""
    from nmf_cpg import NMFCPGGait

    gait = NMFCPGGait(timestep=SC.SIGNAL_SAMPLE_DT, intrinsic_frequency=frequency, seed=0)
    steps = int(round(secs / SC.SIGNAL_SAMPLE_DT))
    return np.stack([gait.targets() for _ in range(steps)]).astype(np.float32)


def render(SC, targets, io, label, out: Path, secs: float, episode: int) -> dict:
    spec = SC.episode_specs(max(episode + 1, 1), SC.EPISODE_SEED)[episode]
    trace = SC.run_body_trace(targets, io, spec, secs, video=out)
    row = min(trace["horizons"].values(), key=lambda r: abs(r["horizon_s"] - secs))
    record = {"label": label, "video": str(out), "episode": episode,
              "forward_m": round(float(row["forward_distance_m"]), 4),
              "displacement_m": round(float(row["displacement_m"]), 4),
              "path_m": round(float(row["path_m"]), 4)}
    record["path_per_forward"] = (round(record["path_m"] / record["forward_m"], 1)
                                  if record["forward_m"] > 0 else None)
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=SECS)
    ap.add_argument("--drive", type=float, default=DRIVE)
    ap.add_argument("--episode", type=int, default=0)
    # Each core's best scalar readout gain, from walk_gain_envelope.
    ap.add_argument("--lif-gain", type=float, default=5.0)
    ap.add_argument("--rate-gain", type=float, default=1.0)
    # 1.2 Hz, the best of the frequencies swept for the reference arm (2.208 m).
    ap.add_argument("--cpg-hz", type=float, default=1.2)
    ap.add_argument("--arms", nargs="+", default=["lif", "rate", "cpg"])
    ap.add_argument("--out-dir", type=Path, default=RESULTS / "best_walk_videos")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records, SC = [], None

    for arm in args.arms:
        if arm in ("lif", "rate"):
            gain = args.lif_gain if arm == "lif" else args.rate_gain
            targets, io, SC = core_targets(arm, args.secs, args.drive, gain)
            label = f"{arm} core, readout gain {gain:g}"
            out = args.out_dir / f"best_walk_{arm}_gain{gain:g}.mp4"
        elif arm == "cpg":
            if SC is None:
                _, io, SC = core_targets("lif", args.secs, args.drive, 1.0)
            targets = cpg_targets(args.secs, SC, args.cpg_hz)
            label = f"NMF CPG reference ceiling, {args.cpg_hz:g} Hz (NOT a connectome arm)"
            out = args.out_dir / f"reference_ceiling_nmf_cpg_{args.cpg_hz:g}hz.mp4"
        else:
            raise ValueError(f"unknown arm {arm!r}")
        records.append(render(SC, targets, io, label, out, args.secs, args.episode))

    manifest = args.out_dir / "best_walk_videos.json"
    manifest.write_text(json.dumps({"secs": args.secs, "drive": args.drive,
                                    "episode": args.episode, "arms": records}, indent=2) + "\n")
    print(f"-> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
