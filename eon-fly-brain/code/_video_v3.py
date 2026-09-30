import json, numpy as np, torch
import audit_embody_3way as A, run_embody_3way as R
from pathlib import Path

cal = A.calibration("cpu")
RES = Path("../results/path2")
seed = R.VIDEO_SEED
for cond in ("connectome", "random"):
    agent = R.PPOAgent(cond, cal, R.STAGE1_CHECKPOINTS[cond])
    saved = agent.load_checkpoint(R.CHECKPOINTS[cond])
    assert int(saved["steps"]) == R.ARM_STEP_CAP
    out = RES / f"embody_3way_v3_{cond}_final_multiview.mp4"
    row = R.evaluate(agent, seed, video_path=out)
    R.assert_frozen(agent)
    print(f"{cond}: {out.name} fwd={row['forward_distance_m']:+.4f} path={row['path_m']:.4f} "
          f"disp={row['displacement_m']:.4f} up={row['upright_s']:.2f}s fell={row['fell']} ret={row['episode_return']:.1f}", flush=True)
print("VIDEO DONE", flush=True)
