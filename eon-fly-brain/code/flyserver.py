"""Real-time FlyWire connectome stepper for the Flycraft control loop.

Holds the full signed 138,639-neuron connectome as a torch sparse matrix and a
membrane-potential state vector. Speaks line-delimited JSON over stdio:

  in : {"az":<rad>, "dist":<blocks>, "loom":<f>, "present":<bool>}
  out: {"turn":<rad/s>, "forward":<0..1>, "dL":<f>, "dR":<f>}

Dynamics = connectome-constrained rate model (the standard shape: connectome IS
the weight matrix, neurotransmitter sign already baked into the weights):

    v <- decay*v + dt_tau*( gW * (W @ r) + I_ext )
    r  = relu(tanh(v))            # bounded firing rate

Visual azimuth injects current into left/right optic+visual_projection neurons;
left/right descending population rates are read out as steer + drive.
All gains are CLI-overridable so the loop can be calibrated against the body.
"""
import sys, json, pickle, argparse
import numpy as np
import torch

def log(*a):
    print(*a, file=sys.stderr, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--substeps", type=int, default=12)
    ap.add_argument("--tau", type=float, default=0.02)      # s, membrane time const
    ap.add_argument("--dt", type=float, default=0.005)      # s, integrator step
    ap.add_argument("--gW", type=float, default=6e-3)       # recurrent gain
    ap.add_argument("--drive_in", type=float, default=1.2)  # visual injection strength
    ap.add_argument("--turn_gain", type=float, default=6.0)
    ap.add_argument("--turn_sign", type=float, default=-1.0)  # prey in left field -> yaw increases (steer left)
    ap.add_argument("--fwd_gain", type=float, default=2.5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from pathlib import Path
    D = Path(a.data) if a.data else Path(__file__).resolve().parent.parent / "data"

    dev = torch.device(a.device)
    W = pickle.load(open(D / "weight_csr.pkl", "rb"))
    if W.layout != torch.sparse_csr:
        W = W.to_sparse_csr()
    W = W.float().to(dev)
    N = W.shape[0]

    io = np.load(D / "io_index.npz")
    vis_left  = torch.from_numpy(io["vis_left"]).long().to(dev)
    vis_right = torch.from_numpy(io["vis_right"]).long().to(dev)
    desc_left  = torch.from_numpy(io["desc_left"]).long().to(dev)
    desc_right = torch.from_numpy(io["desc_right"]).long().to(dev)

    v = torch.zeros(N, dtype=torch.float32, device=dev)
    decay = 1.0 - a.dt / a.tau
    dt_tau = a.dt / a.tau
    log(f"[flyserver] ready N={N} substeps={a.substeps} device={a.device}")
    print(json.dumps({"ready": True, "n": N}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obs = json.loads(line)
        except Exception:
            continue
        if obs.get("cmd") == "reset":
            v.zero_()
            print(json.dumps({"reset": True}), flush=True)
            continue

        present = bool(obs.get("present", False))
        az = float(obs.get("az", 0.0))          # retina convention: 0=ahead, + = LEFT field
        loom = float(obs.get("loom", 0.0))
        # object in the left visual field excites the LEFT optic lobe (az>0 -> left)
        mag = a.drive_in * (0.5 + min(1.0, abs(loom)))
        left_exc  = mag * (max(0.0,  az) + 0.15) if present else 0.0
        right_exc = mag * (max(0.0, -az) + 0.15) if present else 0.0

        I = torch.zeros(N, dtype=torch.float32, device=dev)
        if present:
            I[vis_left]  = left_exc
            I[vis_right] = right_exc

        for _ in range(a.substeps):
            r = torch.relu(torch.tanh(v))
            v.mul_(decay).add_(dt_tau * (a.gW * (W @ r) + I))
            v.clamp_(-5.0, 5.0)

        r = torch.relu(torch.tanh(v))
        dL = float(r[desc_left].mean())
        dR = float(r[desc_right].mean())
        turn = a.turn_sign * a.turn_gain * (dR - dL)
        forward = float(max(0.0, min(1.0, a.fwd_gain * (dL + dR))))
        print(json.dumps({"turn": turn, "forward": forward, "dL": dL, "dR": dR}), flush=True)

if __name__ == "__main__":
    main()
