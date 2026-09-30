# Field code review — how the others train, and the estimable build for us

**Purpose.** You have GPUs but want confidence before spending them: understand
how the published projects actually train a connectome/body controller, then
have our own stack specced, estimated, and testable first. This reviews the
three reference repos under `~/Documents/fly_field_refs/` and turns them into a
concrete, cost-estimated build plan for the embodiment goal ("the fly lives in
the hexapod, trained, no hand-scripting").

Repos reviewed:
- `flybody/` — Janelia biomechanical fly body + distributed RL (DMPO/Ray). The **environment + heavy RL baseline**.
- `flygym/` — NeuroMechFly MuJoCo fly, CPG/muscle imitation, GPU replay. The **sim/embodiment tooling**.
- `flygm/` — **whole-connectome GNN controller, imitation-pretrain + PPO, runs inside flybody. Our exact cousin.** Most of this review is FlyGM.

---

## 1. FlyGM: the architecture, read from the source

FlyGM (`src/flygm/flygm.py`, `flygm_message_passing.py`) is a three-layer stack:

`LightningAgent` (training wrapper) → `FlyGMAgent` (policy: obs-norm + mu/std heads) → `FlyGM` (the connectome core).

The core forward pass (`FlyGM.forward`) is, in plain terms:
1. **Sensory encode:** `encoder: Linear(obs_dim → 32)`; broadcast onto the afferent (sensory) neurons only.
2. **Input gate:** MLP mixes encoded obs into afferent node states.
3. **Message passing** over the connectome graph (`edge_index`), edge messages scaled by `edge_weight` (`x_j * edge_weight`), through a learned GNN (GraphSAGE default; GCN/GAT/GIN/PNA/EdgeCNN/custom FlyGM all selectable).
4. **Motor decode:** read the efferent (motor) neurons' states → `decoder: Linear(node_channels·num_efferent → latent 128)` → `mu_head`/`std_head` → action.

Each neuron is **not a scalar** — it's a learned `node_channels`-dim state vector (16 in `train_pre.py`). Recurrence is explicit: node states carry across steps (`node_state`), reset via a learned `node_init`.

### What is FIXED vs LEARNED (the novelty-critical seam)

Registered as **buffers (frozen)**:
- `edge_index` — connectome topology (FlyWire FAFB v783).
- `edge_weight` — measured synapse weights, used **only as message-scaling scalars**, optionally log10-compressed. Not an `nn.Parameter` — never trained.
- afferent / efferent / intrinsic masks.

Registered as **`nn.Parameter` (learned by imitation + PPO)**:
- `encoder`, `input_gate`, `decoder` — the body interface.
- `node_init` — learned initial recurrent state, one 16-vector per neuron.
- `intrinsic_features` — learned per-neuron descriptor (FlyGM-mode).
- the whole **GNN message-passing MLP** — the actual computation.
- `mu_head`, `std_head`, `input_norm`, and the PPO `ValueNet`.

**Verdict:** FlyGM keeps the connectome's *topology and weight scalars* fixed, but the thing that **computes** is a learned GNN over 16-dim learned node vectors with learned per-neuron features. The measured synaptic weights only *gate/scale* a learned message function; they are **not themselves the computer.** This is the field pattern we already named: *topology fixed, function learned.* It is exactly the line our project refuses to cross.

---

## 2. The training pipeline, with real numbers

### Stage 1 — imitation pretrain (`train_pre.py`)
- Behavior-clone expert trajectories. Loss = **KL(diag-normals) + weighted MSE** on (mu, std); MSE weight decays over the first 100 steps, KL carries the rest.
- Config: `node_channels=16, hidden_channels=16, num_layers=4`, GraphSAGE, `lr=1e-4`, AdamW, `ReduceLROnPlateau`.
- `obs_dim=1253`, `act_dim=59` (flybody walking actuators).
- `max_epochs=100`, **1 GPU** (`devices=[0]`), `accumulate_grad_batches=4`, grad-clip 2.0.
- Optional **TBPTT** (`tbptt_chunk_size` 8–32) to bound memory on long sequences.

### Stage 2 — PPO fine-tune (`train_post.py`)
- Loads the pretrained policy, adds a `ValueNet` MLP(256), trains **in the flybody `walk_imitation` env** (imitation-reward, ref-trajectory `.hdf5`).
- **Single environment**, on-policy, TD(0) advantage (labelled PPO; γ=.99, clip=.2, ent=.01, vf=.5, `lr=1e-4`, cosine LR).
- Warmup: value net trains alone for `FREEZE_POLICY_STEPS = 2048·50 = 102,400` steps, then the connectome policy unfreezes.
- Budget: `MAX_UPDATES=50,000` outer × `UPDATES_PER_STEP=2048` inner ≈ **~100M env steps**, **1 GPU** (`devices=1`), MuJoCo EGL headless, tuned for A100 (`float32_matmul_precision("high")`).

### flybody's own baseline (the heavy end)
flybody trains its ANN policy with **distributed DMPO on Ray** (`train_dmpo_ray.py`) — many parallel actors, much larger compute. FlyGM deliberately does **not** do this; it shows connectome-embodiment is reachable on **one GPU** with imitation + single-env PPO. That's the important precedent for your compute question.

---

## 3. Compute estimate for OUR run

Anchoring on FlyGM's actual single-GPU recipe, adjusted for our lighter setup:

| Factor | FlyGM | Ours | Effect on cost |
|---|---|---|---|
| Graph size | full FAFB (~130k nodes, M edges) | our all-leg VNC (11k nodes, 485k edges) | **~10× smaller → cheaper/step** |
| Node state | 16-dim learned vector | 1 scalar rate (frozen rate model) | cheaper core |
| Recurrent core | learned GNN | **frozen measured W (fixed sparse matmul)** | cheaper, no core gradient |
| Learned params | encoder+gate+GNN+intrinsic+node_init+decoder+heads | **only encoder + decoder interface** | far fewer grads |
| Body | flybody, 59 muscle actuators | hexapod, 18 joints | cheaper physics/step |

**Estimates on one modern GPU (A100/4090-class):**
- **Stage 1 imitation:** ~100 epochs, tiny interface to fit → **order of a few hours** (dominated by dataset/trajectory count, not model size).
- **Stage 2 single-env PPO:** the wall-clock bottleneck is single-env MuJoCo rollout throughput, not the GPU. ~100M steps is the FlyGM *upper bound*; walking behavior typically emerges far earlier and is checkpointed. Budget **overnight to ~2 days**, checkpointed, with early-stop on EMA return.
- **3-way control (real / random / rewired cores):** 3× the above, but embarrassingly parallel across GPUs → same wall-clock if you have 3 GPUs.

Cheapest confidence-builder: run Stage-2 for a **fixed small step budget** (e.g. 5M steps) on the real core first, watch EMA return climb, and extrapolate before committing the full run.

---

## 4. The honest build for us (adopt their harness, keep our substrate)

The move: **take FlyGM's training harness, refuse FlyGM's learned core.**

- **Environment:** reuse flybody's env machinery, but retarget to our hexapod MuJoCo model (we already have `hexapod_gait.py`, PD servo, 18-joint targets). This is the main porting work — a flybody-style task wrapper around our body.
- **Core:** our **frozen** measured-VNC rate model (`vnc_stepper.py`) as the recurrent operator. The synaptic weights ARE the computer. Nothing in the core is trained.
- **Learn only the interface:** a sensory encoder (hexapod state → descending drive into the VNC) and a motor decoder (VNC motor output → 18 joints, generalizing our fixed anatomical map). This is the literal "we build the fly a body and let it adapt to it" — the brain is untouched.
- **Training:** Stage 1 imitation (clone a forward-gait reference) + Stage 2 single-env PPO for robustness — FlyGM's exact two-stage recipe.
- **The control that decides novelty:** run the **identical** training with the core swapped to **amplitude-matched random** and **degree-preserving rewired**. If the real VNC core embodies **faster / more robustly / with a smaller interface** than the scrambled cores → the measured connectome earned it. If it ties → honest null, the optimizer did the work (same result we've gotten every time so far).

Why this is novel and not a copy: FlyGM/flybody **learn the functional weights**; we hold the measured weights fixed as the computer and learn only the body-interface. Nobody has embodied the raw measured connectome as the controller in a **non-fly** body. That is the one seat left at the table.

---

## 5. Readiness / testing checklist (before any GPU spend)

Do-first, all CPU, all cheap — this is the "confidence" list:
1. **Deps:** FlyGM needs `flybody` + `torch`, `pytorch-lightning`, `torch-geometric`, `acme`. Stand up the env; confirm `import flybody.fly_envs` and a torch-geometric message-pass run on CPU.
2. **Env port smoke test:** wrap our hexapod as a flybody-style task returning `(obs, reward, done)`; one random-action episode to confirm the loop closes.
3. **Core-as-policy shim:** wire `vnc_stepper` frozen core + a randomly-initialized interface into the `step(obs, node_state) → (mu, std)` signature. Confirm shapes, one forward pass.
4. **Imitation dataset:** produce the reference-gait `(input.npy, output.npy)` for behavior cloning (forward-drive target, not tripod — the axis where our structure actually helps).
5. **Overfit test:** Stage-1 on a tiny slice, confirm loss → ~0 (proves the interface can express the target at all).
6. **Control harness:** confirm the amplitude-matched-random and rewired cores load and run through the identical pipeline; hash-check the real graph unchanged.
7. **Budgeted PPO dry run:** 5M-step Stage-2 on the real core; watch EMA return. Extrapolate cost, then decide the full 3-way spend.

Only after 1–7 pass do you spend the full GPU budget. That's the confidence gate.

---

## 6. Honest bottom line

- The training the field uses is **real and reachable on one GPU** (FlyGM proves it) — your GPU access is sufficient.
- But their result is **"topology fixed, function learned."** If we copy that, we're them, and our own repo already shows a *random* core trained the same way keeps up.
- Our only honest, novel version: **frozen measured weights as the computer, learn only the body-interface, and beat the amplitude-matched-random core.** Same harness, stricter substrate, one decisive control.
- Best case earned by this build: *"the measured fly connectome, embodied in a non-fly hexapod, controls it better than a scrambled connectome under identical training."* That is the moat, stated honestly. Worst case: a null, plus a working connectome-driven hexapod demo. Either way the deciding experiment is small, specced above, and cheap to gate before the big spend.
