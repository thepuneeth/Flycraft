"""How far does each core walk when the readout gain is swept, not chosen?

The joint map converts a normalised motor score to joint angles through
`specificity_campaign.MAP_GAINS`, and those gains were calibrated against the
rate core. Comparing cores at a fixed gain therefore confounds "different
dynamics" with "different units", and every rule tried for removing that
confound smuggled a choice back in:

  * matching the [SH] core's per-channel SD to the rate core's borrows the rate
    core's units, which are arbitrary -- that is the whole reason the published
    core exists;
  * matching peak joint excursion to the body's own gait amplitude
    (`hexapod_gait.HexapodGait.lift_amp`) is set by outliers, and cost the rate
    core half its distance (0.2241 -> 0.1197 m) without telling us anything.

So the gain is not chosen here. It is ONE scalar on the deviation from stance,
it belongs to the interface rather than to either core, and it is swept over the
same grid for both. What each core reaches at its own best gain is then a
statement about the dynamics with the interface parameter maximised out. The
shape of each core's curve is reported alongside the peak, because a peak that
stands on one grid point is a knife edge and a peak with neighbours is an
optimum -- and that distinction turned out to separate the two cores.

Zero training: every arm is a fixed-map rollout of the frozen graph.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from hexapod_gait import resolve_stance  # noqa: E402

SECS = 8.0
DRIVE = 100.0
SWEEP_EPISODES = 8
CONFIRM_EPISODES = 20
GAINS = (0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0)


def targets_for(core: str, secs: float, drive: float):
    """Fixed-map joint targets from one core, driven at a constant rate."""
    os.environ["SC_CORE"] = core
    import specificity_campaign as SC
    SC = importlib.reload(SC)
    vnc = SC.base_vnc(SC.VNC_DATA)
    steps = int(round(secs / SC.SIGNAL_SAMPLE_DT))
    current = vnc.descending_current(drive)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)]).astype(np.float32)
    return SC.target_history(history, vnc.io, {}), vnc.io, SC


def walk(SC, targets, io, episodes, secs):
    specs = SC.episode_specs(episodes, SC.EPISODE_SEED)
    rows = [min(SC.run_body_trace(targets, io, spec, secs)["horizons"].values(),
                key=lambda row: abs(row["horizon_s"] - secs)) for spec in specs]
    forward = np.array([row["forward_distance_m"] for row in rows])
    path = np.array([row["path_m"] for row in rows])
    return float(forward.mean()), float(forward.std()), float(path.mean())


def sweep(core: str, secs: float, drive: float, gains, sweep_episodes, confirm_episodes):
    base, io, SC = targets_for(core, secs, drive)
    stance = resolve_stance(None).reshape(18)[None, :]
    curve = []
    for gain in gains:
        scaled = (stance + (base - stance) * gain).astype(np.float32)
        mean, sd, path = walk(SC, scaled, io, sweep_episodes, secs)
        row = {"core": core, "gain": gain, "forward_m": round(mean, 4),
               "forward_sd": round(sd, 4), "path_m": round(path, 4),
               "target_sd": round(float(scaled.std(axis=0).mean()), 5),
               "episodes": sweep_episodes}
        curve.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    top = max(curve, key=lambda row: row["forward_m"])
    scaled = (stance + (base - stance) * top["gain"]).astype(np.float32)
    mean, sd, path = walk(SC, scaled, io, confirm_episodes, secs)
    # A peak whose neighbours collapse is an operating point someone tuned; a
    # peak its neighbours approach is a property of the dynamics. Reported, not
    # judged, so the distinction survives into the artifact.
    order = [row["forward_m"] for row in curve]
    rank = order.index(top["forward_m"])
    neighbours = [order[i] for i in (rank - 1, rank + 1) if 0 <= i < len(order)]
    best = {"core": core, "gain": top["gain"], "forward_m": round(mean, 4),
            "forward_sd": round(sd, 4), "path_m": round(path, 4),
            "episodes": confirm_episodes,
            "neighbour_forward_m": [round(v, 4) for v in neighbours],
            "neighbour_ratio": round(max(neighbours) / mean, 3) if neighbours and mean > 0 else None}
    print(json.dumps({"BEST": best}, sort_keys=True), flush=True)
    return curve, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", nargs="+", default=["rate", "lif"], choices=("rate", "lif"))
    ap.add_argument("--secs", type=float, default=SECS)
    ap.add_argument("--drive", type=float, default=DRIVE)
    ap.add_argument("--sweep-episodes", type=int, default=SWEEP_EPISODES)
    ap.add_argument("--confirm-episodes", type=int, default=CONFIRM_EPISODES)
    ap.add_argument("--out", type=Path,
                    default=CODE.parent / "results" / "path2" / "walk_gain_envelope.json")
    args = ap.parse_args()

    payload = {"secs": args.secs, "drive": args.drive, "gains": list(GAINS), "arms": {}}
    for core in args.cores:
        curve, best = sweep(core, args.secs, args.drive, GAINS,
                            args.sweep_episodes, args.confirm_episodes)
        payload["arms"][core] = {"curve": curve, "best": best}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
