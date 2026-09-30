import json, numpy as np, torch
import audit_embody_3way as A, run_embody_3way as R

cal = A.calibration("cpu")
out = {}
for cond in ("connectome", "random"):
    agent = R.PPOAgent(cond, cal, R.STAGE1_CHECKPOINTS[cond])
    saved = agent.load_checkpoint(R.CHECKPOINTS[cond])
    assert int(saved["steps"]) == R.ARM_STEP_CAP, (cond, saved["steps"])
    rows = []
    for seed in R.EVALUATION_SEEDS:
        row = R.evaluate(agent, seed)
        rows.append(row)
        print(f"{cond} seed={seed} fwd={row['forward_distance_m']:+.4f} up={row['upright_s']:.2f}s "
              f"mps={row['mean_forward_speed_mps']:+.4f} fell={row['fell']} ret={row['episode_return']:.1f}", flush=True)
    R.assert_frozen(agent)
    out[cond] = rows
json.dump(out, open("../results/path2/v3_partial_eval.json", "w"), indent=1)
print("EVAL DONE", flush=True)
