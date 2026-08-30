"""Path 3: fixed antagonist-sign decoder for per-leg step timing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parent.parent / "data"
RESULTS = HERE.parent.parent / "results" / "path2"
sys.path.insert(0, str(HERE))
from probe_vnc_phase import phase_metrics  # noqa: E402
from vnc_stepper import VNCStepper  # noqa: E402


LEG_NAMES = ("T1L", "T1R", "T2L", "T2R", "T3L", "T3R")
SAMPLE_DT_S = 0.005 * 4
STEPS = 1500
VNC_SEED = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def motor_pools(io: np.lib.npyio.NpzFile) -> list[dict[str, object]]:
    """Use the same annotated trochanter antagonist pools for all six legs."""
    labels = np.char.lower(io["motor_labels"].astype(str))
    leg_ids = io["motor_leg"].astype(np.int64)
    joint_ids = io["motor_joint"].astype(np.int64)
    signs = io["motor_sign"].astype(np.float64)
    trochanter = joint_ids == 1
    swing_mask = trochanter & (signs > 0) & (np.char.find(labels, "flexor") >= 0)
    stance_mask = trochanter & (signs < 0) & (np.char.find(labels, "extensor") >= 0)
    pools = []
    for leg, name in enumerate(LEG_NAMES):
        swing = np.flatnonzero(swing_mask & (leg_ids == leg))
        stance = np.flatnonzero(stance_mask & (leg_ids == leg))
        if not len(swing) or not len(stance):
            raise ValueError(f"missing antagonist pool for {name}: {len(swing)} / {len(stance)}")
        pools.append({
            "leg": name,
            "swing_indices": swing,
            "stance_indices": stance,
            "swing_labels": sorted(set(labels[swing].tolist())),
            "stance_labels": sorted(set(labels[stance].tolist())),
        })
    return pools


def decode_activity(activity: np.ndarray, io: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return swing states and the two mean antagonist activities."""
    values = np.asarray(activity, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != len(io["motor_out"]):
        raise ValueError(f"activity shape {values.shape}; expected (time, {len(io['motor_out'])})")
    pools = motor_pools(io)
    swing = np.column_stack([values[:, pool["swing_indices"]].mean(axis=1) for pool in pools])
    stance = np.column_stack([values[:, pool["stance_indices"]].mean(axis=1) for pool in pools])
    return swing > stance, swing, stance


def decode_vector(activity: np.ndarray, io: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, swing, stance = decode_activity(np.asarray(activity), io)
    return states[0], swing[0], stance[0]


def step_statistics(states: np.ndarray, sample_dt: float = SAMPLE_DT_S, burn_fraction: float = 0.5) -> dict:
    values = np.asarray(states, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"state shape {values.shape}; expected (time, 6)")
    tail = values[int(len(values) * burn_fraction):]
    duration = max(float(len(tail)) * sample_dt, sample_dt)
    rows = []
    for leg, name in enumerate(LEG_NAMES):
        onsets = int(np.count_nonzero(tail[1:, leg] & ~tail[:-1, leg])) if len(tail) > 1 else 0
        rows.append({
            "leg": name,
            "step_onsets": onsets,
            "step_frequency_hz": float(onsets / duration),
            "duty_factor_swing": float(tail[:, leg].mean()),
            "swing_fraction": float(tail[:, leg].mean()),
        })
    return {
        "burn_fraction": burn_fraction,
        "sample_dt_s": sample_dt,
        "per_leg": rows,
        "mean_step_frequency_hz": float(np.mean([row["step_frequency_hz"] for row in rows])),
        "mean_duty_factor_swing": float(np.mean([row["duty_factor_swing"] for row in rows])),
    }


class MatchedRandomTiming:
    """Independent alternating-renewal timing with fixed P1 mean rate/duty."""

    def __init__(self, frequency_hz: np.ndarray, duty_factor: np.ndarray, seed: int):
        self.frequency_hz = np.asarray(frequency_hz, dtype=np.float64).reshape(6)
        self.duty_factor = np.asarray(duty_factor, dtype=np.float64).reshape(6)
        if np.any(self.frequency_hz <= 0.0) or np.any((self.duty_factor <= 0.0) | (self.duty_factor >= 1.0)):
            raise ValueError("random timing requires positive frequencies and duty factors in (0, 1)")
        self.rng = np.random.default_rng(seed)
        self.time_s = 0.0
        self.states = self.rng.random(6) < self.duty_factor
        self.next_transition = np.asarray([self._draw_duration(leg) for leg in range(6)], dtype=np.float64)

    def _draw_duration(self, leg: int) -> float:
        mean = (self.duty_factor[leg] if self.states[leg] else 1.0 - self.duty_factor[leg]) / self.frequency_hz[leg]
        return max(float(self.rng.exponential(mean)), np.finfo(float).eps)

    def step(self, dt: float) -> np.ndarray:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        self.time_s += float(dt)
        for leg in range(6):
            while self.time_s >= self.next_transition[leg]:
                self.states[leg] = ~self.states[leg]
                self.next_transition[leg] += self._draw_duration(leg)
        return self.states.copy()


def run_preview(data: Path = RESULTS, weights_name: str = "all_legs_weight_csr.pkl", index_name: str = "all_legs_io_index.npz", shuffle: bool = False, steps: int = STEPS) -> tuple[dict, np.ndarray]:
    vnc = VNCStepper(
        data,
        substeps=4,
        tau=0.02,
        dt=0.005,
        gW=0.03,
        weights_name=weights_name,
        index_name=index_name,
        model="paper",
        threshold=7.5,
        activation_gain=1.0,
        fr_cap=200.0,
        heterogeneity=True,
        seed=VNC_SEED,
        shuffle=shuffle,
    )
    current = vnc.descending_current(100.0)
    history = np.stack([vnc.step(current).cpu().numpy() for _ in range(steps)])
    states, swing, stance = decode_activity(history, vnc.io)
    stats = step_statistics(states)
    phase = phase_metrics(states.astype(np.float64), SAMPLE_DT_S, "P1-PRESENT", "P1-ABSENT")
    pools = motor_pools(vnc.io)
    result = {
        "timing_source": "column_shuffled_vnc" if shuffle else "real_vnc",
        "protocol": {
            "weights_name": weights_name,
            "index_name": index_name,
            "model": "paper",
            "drive": 100.0,
            "gW": 0.03,
            "tau": 0.02,
            "dt": 0.005,
            "substeps": 4,
            "sample_dt_s": SAMPLE_DT_S,
            "steps": steps,
            "burn_fraction": 0.5,
            "heterogeneity": True,
            "heterogeneity_seed": VNC_SEED,
            "shuffle": shuffle,
            "shuffle_seed": VNC_SEED,
            "decode": "mean trochanter flexor/Acc. tr flexor > mean trochanter extensor; ties stance",
        },
        "leg_names": list(LEG_NAMES),
        "motor_pools": [
            {
                "leg": pool["leg"],
                "swing_labels": pool["swing_labels"],
                "stance_labels": pool["stance_labels"],
                "swing_count": len(pool["swing_indices"]),
                "stance_count": len(pool["stance_indices"]),
            }
            for pool in pools
        ],
        "weights_sha256": file_sha256(Path(data) / weights_name),
        "step_statistics": stats,
        "decoded_state_phase": phase,
        "mean_swing_pool_activity": swing[len(swing) // 2:].mean(axis=0).tolist(),
        "mean_stance_pool_activity": stance[len(stance) // 2:].mean(axis=0).tolist(),
    }
    return result, states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=RESULTS)
    parser.add_argument("--weights_name", default="all_legs_weight_csr.pkl")
    parser.add_argument("--index_name", default="all_legs_io_index.npz")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--out", type=Path, default=RESULTS / "step_decode_p1")
    args = parser.parse_args()
    real, real_states = run_preview(args.data, args.weights_name, args.index_name, False, args.steps)
    shuffled, shuffled_states = run_preview(args.data, args.weights_name, args.index_name, True, args.steps)
    result = {
        "protocol": real["protocol"],
        "real": real,
        "shuffled": shuffled,
        "assignment_rule": "trochanter flexor/Acc. tr flexor versus trochanter extensor, selected from the existing mn_to_joint annotation map",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.out.with_name(args.out.name + "_traces.npz"), real_states=real_states, shuffled_states=shuffled_states)
    print(json.dumps({key: result[key]["step_statistics"] for key in ("real", "shuffled")}, sort_keys=True))
    print(f"decode -> {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
