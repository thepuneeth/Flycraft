"""Phase A: fail-fast body, fall-floor, stance, and frozen-graph preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from gait_engine import clean_tripod_states  # noqa: E402
from hexapod_gait import H_MIN, HexapodGait, XML, pd_torque  # noqa: E402
from run_stepwalk import run_clean_engine  # noqa: E402


EXPECTED_SHA256 = {
    "all_legs_weight_csr.pkl": "018a4890e01f3dd608693e8a040103829e8523d134290ae918e6e5dd4d815506",
    "all_legs_io_index.npz": "3f65f5b3ebed0d63161dca416bffa1a55afc0be4906113103426b16492d09449",
}
FALL_FLOOR_FILES = (
    "hexapod_gait.py",
    "run_hexapod.py",
    "run_hexapod_chase.py",
    "run_vnc_chase.py",
    "run_stepwalk.py",
    "run_sensory_feedback.py",
    "run_reservoir_walk.py",
    "audit_transfer_layers.py",
    "cmaes_walk.py",
    "outcomeD_train.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stand_check() -> dict:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    gait = HexapodGait(stance=None)
    home = np.asarray(model.key("home").qpos[7:25], dtype=np.float64).reshape(6, 3)
    stance_match = bool(np.array_equal(gait.stance, home))
    target = gait.targets(0.0, 0.0, 0.0).astype(np.float32)
    data.qpos[2] = 0.15
    heights = []
    for _ in range(int(round(1.5 / model.opt.timestep))):
        data.ctrl[:] = pd_torque(model, data, target, 3.0, 0.05)
        mujoco.mj_step(model, data)
        heights.append(float(data.qpos[2]))
    return {
        "static_settle_height_m": float(data.qpos[2]),
        "static_min_height_m": float(np.min(heights)),
        "stance_matches_home": stance_match,
        "home_stance": home.tolist(),
    }


def fall_floor_check() -> dict:
    if not np.isclose(H_MIN, 0.08):
        raise AssertionError(f"H_MIN={H_MIN}; expected 0.08")
    bad = []
    for name in FALL_FLOOR_FILES:
        path = HERE.parent / name
        text = path.read_text(encoding="utf-8")
        if "H_MIN" not in text or re.search(r"<\s*0\.02", text):
            bad.append(name)
    if bad:
        raise AssertionError(f"fall-floor audit failed: {bad}")
    return {"h_min_m": float(H_MIN), "files_checked": list(FALL_FLOOR_FILES)}


def graph_check(data: Path) -> dict:
    hashes = {name: sha256(data / name) for name in EXPECTED_SHA256}
    if hashes != EXPECTED_SHA256:
        raise AssertionError(f"frozen graph hash mismatch: {hashes}")
    return {"sha256": hashes, "hash_equal": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RESULTS)
    parser.add_argument("--video", type=Path, default=RESULTS / "preflight_engine_walk_4s.mp4")
    parser.add_argument("--out", type=Path, default=RESULTS / "preflight.json")
    args = parser.parse_args()

    try:
        stand = stand_check()
        floor = fall_floor_check()
        graph = graph_check(args.data)
        clean = run_clean_engine(secs=8.0)
        phase = 0.0
        states = []
        frequency = 0.8 + 0.5 * 0.8
        for _ in range(8 * 50):
            phase = (phase + 2.0 * np.pi * frequency / 50.0) % (2.0 * np.pi)
            states.append(clean_tripod_states(phase))
        duty = float(np.asarray(states, dtype=bool).mean())
        checks = {
            "T1": stand["static_settle_height_m"] >= 0.12,
            "T2": np.isclose(floor["h_min_m"], 0.08),
            "T3": stand["stance_matches_home"],
            "T4": clean["path_m"] >= 1.0 and not clean["fell"],
            "T5": 0.35 <= duty <= 0.6,
            "T6": graph["hash_equal"],
        }
        checks = {name: bool(passed) for name, passed in checks.items()}
        failed = [name for name, passed in checks.items() if not passed]
        result = {
            "protocol": {
                "stand_s": 1.5,
                "clean_engine_s": 8.0,
                "clean_engine_clock": "clean_tripod_states",
                "clean_swing_duty_definition": "mean clean tripod swing state across 50 Hz control steps",
                "fall_floor_m": H_MIN,
            },
            "stand": stand,
            "fall_floor": floor,
            "frozen_graph": graph,
            "clean_engine": clean,
            "clean_mean_swing_duty": duty,
            "checks": checks,
            "verdict": "GATE A: PASS (T1-T6)" if not failed else f"GATE A: FAIL {','.join(failed)}",
        }
        if failed:
            print(result["verdict"])
            return 1
        clip = run_clean_engine(secs=4.0, video=args.video)
        result["engine_video"] = {"path": str(args.video), "secs": 4.0, "metrics": clip}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            "GATE A: PASS (T1={:.3f}m T2=H_MIN:{:.2f}m T3={} T4=path:{:.3f}m/min:{:.3f}m/fell:{} T5=duty:{:.3f} T6={})".format(
                stand["static_settle_height_m"], H_MIN, stand["stance_matches_home"], clean["path_m"], clean["min_height_m"], clean["fell"], duty, graph["hash_equal"]
            )
        )
        print(f"preflight -> {args.out}")
        print(f"engine video -> {args.video}")
        return 0
    except Exception as exc:
        print(f"GATE A: FAIL {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
