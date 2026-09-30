"""Fixed intra-leg posture map for the reduced-agency interface.

The brain keeps the part that makes walking walking -- WHEN each leg moves. What
it stops emitting is the redundant detail of how one leg's three joints are
shaped at a given extension, which is a static fact about this body, not a motor
decision. So the decoder shrinks from 260->18 to 260->6 (one scalar per leg) and
this map expands each scalar back to three joint targets.

The line this stays on the right side of: the map is STATIC. It contains no
clock, no phase, no oscillator. Feed it a constant and the legs hold a pose --
all timing still comes from the connectome. An external CPG, by contrast, would
supply the rhythm and reduce the connectome to a speed knob; that is the
forbidden version (EMBODY_V5_PLAN Tier 3) and this is not it.

Direction comes from the teacher's own joint covariance -- the first principal
component of each leg's 3 joints over the demonstrations -- so it is measured off
the body's gait, not hand-drawn by me.
"""
import json
from pathlib import Path

import numpy as np

import teacher_v4 as T

RESULTS = Path(__file__).resolve().parent.parent / "results" / "path2"
OUT = RESULTS / "leg_posture_map.npz"


def build() -> dict:
    _, actions, _ = T.build_teacher()          # (commands, steps, 18)
    flat = actions.reshape(-1, 18)
    home = flat.mean(axis=0)                   # (18,) neutral pose
    legs = (flat - home).reshape(-1, 6, 3)     # (samples, leg, joint)

    directions, explained, spans = [], [], []
    for leg in range(6):
        x = legs[:, leg, :]                    # (samples, 3)
        # First PC of this leg's joint motion.
        u, s, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
        direction = vt[0]
        # Sign convention: positive scalar extends the leg the way joint_1 moves,
        # so the scalar has a consistent meaning across legs.
        if direction[1] < 0:
            direction = -direction
        projection = x @ direction
        directions.append(direction)
        explained.append(float(s[0] ** 2 / (s ** 2).sum()))
        spans.append(float(np.abs(projection).max()))

    record = {
        "home": home,
        "direction": np.asarray(directions, dtype=np.float32),   # (6, 3)
        "span": np.asarray(spans, dtype=np.float32),             # (6,)
        "explained_variance_ratio": np.asarray(explained, dtype=np.float32),
    }
    np.savez(OUT, **record)
    print(json.dumps({
        "out": str(OUT),
        "explained_variance_ratio": [round(v, 3) for v in explained],
        "span": [round(v, 3) for v in spans],
        "note": "explained ratio is how much of each leg's joint motion one scalar carries",
    }, indent=2), flush=True)
    return record


if __name__ == "__main__":
    build()
