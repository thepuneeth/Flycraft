# BUILD SPEC — Path 2: Brain → VNC(MANC) → Hexapod, zero learning

> **Goal in one line:** Drive the hexapod's legs from the **real fly VNC connectome (MANC)** — not a hand-written CPG — so the *entire* control chain, brain steering **and** leg rhythm, comes from actual fly wiring with **zero learning**. On a **non-fly body**. That combination is unclaimed.

This document is a complete, standalone build brief for an agent taking over Path 2. Read `HANDOFF.md` first for the existing Path-1 system; this builds on it. Everything here reflects state as of 2026-08-29.

---

## 0. Why this path (the stakes)

Path 1 (already built, see HANDOFF.md) has the fly **brain** connectome steering a hexapod, but the walking rhythm is a **hand-written tripod CPG** — faked. Reviewers will hit that seam: "the CPG walks no matter what; the brain only contributes 2 numbers."

Path 2 removes the fake. The rhythm comes from the **MANC connectome** (the fly's ventral nerve cord — its "spinal cord"). Then the only hand-built piece left is the unavoidable **translation** from fly leg anatomy to hexapod joints — and that translation *is* the scientific contribution, not a cheat.

**The field context (verified 2026-08, see §9):** four strong 2025 papers already do connectome/learned locomotion **on the fly body**. Our moat is the **non-fly body + zero-learning** angle. A Sep-2025 paper proved the MANC connectome *does* generate a real walking rhythm (de-risks us). Do **not** drift toward "connectome walks a fly" — that's scooped four ways. Stay on the hexapod.

---

## 1. The honest chain (what Path 2 must implement)

```
prey (world)
   │
   ▼
retina()                       [REUSE from Path 1 — rendered vision is a later upgrade]
   │  present, az, dist, size
   ▼
BRAIN connectome               [REUSE from Path 1: flyserver.py / run_hexapod_chase.py brain()]
   │  FlyWire rate model, zero learning
   │  OUTPUT: descending command signals — specifically the walk-command
   │  descending neurons (e.g. DNg100-type "go", plus left/right turn bias)
   ▼
VNC connectome  ◄── NEW, THE CORE OF PATH 2
   │  MANC front-leg (T1) network, run as the SAME rate model as the brain.
   │  Descending inputs drive an intrinsic rhythm-generating circuit
   │  (the connectome's own CPG). OUTPUT: leg MOTOR-NEURON activity, per leg,
   │  time-varying (this is the rhythm the brain could not make on its own).
   ▼
MOTOR-NEURON → JOINT translation  ◄── NEW, THE UNAVOIDABLE HAND-BUILT PIECE
   │  Map fly leg motor-neuron pools (per joint, per leg) → hexapod joint targets.
   │  Fly leg ≈ 5 segments/joints; hexapod leg = 3 (coxa/femur/tibia).
   │  This is a documented, principled reduction — NOT learning.
   ▼
PD torque → MuJoCo hexapod     [REUSE from Path 1: pd_torque(), hexapod.xml]
   │
   ▼
new pose ──► loop
```

**What is reused (do not rebuild):** `retina()`, the brain rate-model + readout, `pd_torque()`, `hexapod.xml`, the physics loop, the metrics/logging. All in `eon-fly-brain/code/` (see HANDOFF.md §3).

**What is new:** (a) MANC data ingestion, (b) the VNC network stepper, (c) the brain→VNC coupling (which descending neurons inject where), (d) the motor-neuron→joint translation, (e) new controls for the study.

---

## 2. Data to acquire — the MANC / VNC connectome

The brain data we already have (`weight_csr.pkl`, `io_index.npz`) is FlyWire = **brain only**. Path 2 needs the **VNC**.

**Dataset:** MANC (Male Adult Nerve Cord), Janelia FlyEM. ~23,000 neurons, ~74M synapses. (FANC — Female Adult Nerve Cord — is the complementary set; the Sep-2025 CPG paper used **both** MANC + FANC front-leg neuropil. MANC alone is enough to start.)

**Where:**
- Browse / query: <https://neuprint.janelia.org>, dataset **MANC** (Google login).
- Bulk flat files: public Google bucket linked from <https://www.janelia.org/project-team/flyem/manc-connectome>.
- Programmatic: `neuprint-python` (Python) or `neuprintr` (R). MANC helpers: `natverse/malevnc` (<https://github.com/natverse/malevnc>).

**What to extract (mirror the brain format so the existing stepper code works unchanged):**
1. **Signed weight matrix** `vnc_weight_csr.pkl` — neuron×neuron, synapse counts, sign baked in by neurotransmitter (excitatory +, inhibitory −). Same torch sparse CSR format as `weight_csr.pkl`.
2. **Index file** `vnc_io_index.npz` with:
   - `descending_in` — indices of descending-neuron **targets** in the VNC (where brain output arrives). Key IDs to locate: **DNg100** (walk command), **DNb08** (parallel rhythm driver), plus left/right turn descendings.
   - `motor_out` — leg **motor-neuron** indices, grouped **per leg (T1L/T1R/T2L/T2R/T3L/T3R) and per joint** (coxa/trochanter/femur/tibia/tarsus). This grouping is what §5 translates from.
   - `rhythm_core` (optional) — the minimal rhythm circuit the paper identified: **1 inhibitory + 2 excitatory interneurons** in T1. Pull exact IDs from the paper / neuPrint (see §9). Useful for a reduced-model sanity check.

**neuPrint query sketch (adjust to live schema):**
```python
from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC
c = Client('neuprint.janelia.org', dataset='manc:v1.0', token='<your token>')
# leg motor neurons (front leg neuropil T1)
mn = fetch_neurons(NC(type='.*MN.*', rois=['LegNp(T1)(L)','LegNp(T1)(R)']))
# descending neurons of interest
dn = fetch_neurons(NC(type=['DNg100','DNb08']))
# full weighted adjacency for the subnetwork you keep
conn = fetch_adjacencies(sources=..., targets=...)   # -> build signed CSR
```
Confirm exact type strings / ROI names against the live `manc` schema — they drift between versions.

**Scope decision (start small):** begin with the **T1 (front-leg) neuropil only**, both sides — that's what the CPG paper used and it's where the rhythm circuit lives. Get one leg-pair oscillating before scaling to all six.

---

## 3. The VNC network stepper (new code)

Reuse the brain's rate-model dynamics **verbatim** — it's the standard connectome-constrained rate model, already in `flyserver.py`:

```
v ← decay·v + dt_tau·( gW · (W_vnc @ r) + I_descending )
r  = relu(tanh(v))
decay = 1 − dt/tau,  dt_tau = dt/tau
# start from brain defaults: substeps=12, tau=0.02, dt=0.005, gW=6e-3
```

**Critical difference from the brain:** the brain settled to a fixed point (no rhythm — that's HANDOFF.md §4's null). The VNC network **must oscillate**. The Sep-2025 paper shows it does, *given descending drive*. So:
- Inject the brain's descending "go" signal into `descending_in` (DNg100/DNb08 targets) as constant current `I_descending`.
- Step the VNC network and **watch `motor_out` activity over time**. Success = the motor-neuron pools **oscillate** (nonzero temporal std, anti-phase between antagonist pools / between legs). If it settles flat like the brain did, the gain/coupling is wrong — sweep `gW`, `tau`, drive magnitude before concluding failure.
- **Reduced sanity check:** isolate the `rhythm_core` (3 neurons) and confirm it oscillates alone. If the 3-neuron circuit rhythms but the full net doesn't, it's a scaling/inhibition-balance issue, not a data issue.

**File:** `eon-fly-brain/code/vnc_stepper.py` — mirror `flyserver.py`'s structure. Expose `step(I_descending) -> motor_activity_vector`.

---

## 4. Brain → VNC coupling (new)

The brain connectome output must become the VNC's descending input.
- Brain readout already gives `forward` (go) and `turn` (left/right bias) — see Path 1 `brain()`.
- Map: `forward` → magnitude of current injected into the **walk-command descending targets** (DNg100/DNb08). `turn` → **asymmetric** injection into left vs right descending populations (more drive to one side biases that tripod's stride → differential steering, exactly as the fly turns).
- Keep the **alignment gate** from Path 1 (`gate = max(0.1, cos(az))`) on the `forward` drive — still biologically a heading gate, still not learning. It's what prevents orbiting.

**Zero-learning discipline:** every gain here (injection scale, turn asymmetry) is a **fixed hyperparameter**, calibrated once by hand like the brain's `fwd_gain`/`turn_gain` — never fit to a reward or trained. Log them. If you ever find yourself optimizing a gain against "did it catch the prey," STOP — that's learning and it breaks the thesis.

---

## 5. Motor-neuron → hexapod-joint translation (new — the contribution)

This is the honest, unavoidable hand-built piece. Do it **principled and documented**, not tuned-to-work.

**The mismatch:** fly leg = ~5 joints (coxa, trochanter, femur, tibia, tarsus), driven by many motor-neuron pools. Hexapod leg = **3 joints** (coxa, femur, tibia), 18 total. No one-to-one exists.

**Proposed reduction (state it explicitly in the paper as a modeling choice):**
1. Group VNC `motor_out` neurons by **(leg, joint)** using neuPrint annotations.
2. For each hexapod joint, define its target angle as a **fixed linear readout** of the corresponding fly motor-neuron pool's mean activity: `joint_target = rest_angle + k · (flexor_pool − extensor_pool)`. `k`, `rest_angle` fixed per joint type, not trained.
3. Collapse the fly's 5 joints → hexapod's 3 by an anatomical mapping (e.g. fold trochanter into femur, tarsus into tibia). **Document the exact mapping table.**
4. Legs: fly T1/T2/T3 × L/R → hexapod legs. HANDOFF.md §5 gives the hexapod leg indexing (LEFT={0,2,4}, RIGHT={1,3,5}, tripod A={0,3,4}). Pick a fly-leg → hexapod-leg assignment that preserves left/right and front/mid/back, and preserves the tripod phase relationships the VNC produces.

**Why this is still "no faking":** the *rhythm, phase, and left/right coordination* all come from the VNC connectome's dynamics. The translation only re-expresses fly motor output in hexapod coordinates — a fixed coordinate change, declared up front. Contrast with Path 1 where the rhythm itself was invented.

Feed the resulting `joint_target` vector into the existing `pd_torque()` (HANDOFF.md §5). Nothing downstream changes.

---

## 6. File plan

New files under `eon-fly-brain/code/`:
| File | Purpose |
|---|---|
| `fetch_manc.py` | Pull MANC front-leg subnetwork from neuPrint → build `vnc_weight_csr.pkl` + `vnc_io_index.npz`. Run once. |
| `vnc_stepper.py` | VNC rate-model stepper (mirror `flyserver.py`). `step(I_descending)->motor_activity`. |
| `mn_to_joint.py` | The §5 translation. Fixed map, documented. `translate(motor_activity)->18 joint targets`. |
| `run_vnc_chase.py` | The Path-2 main experiment: brain → VNC → translation → hexapod → physics. Fork `run_hexapod_chase.py`; swap the hand-CPG (`HexapodGait.targets`) for `vnc_stepper` + `mn_to_joint`. Keep retina, pd_torque, metrics, video. |

New data under `eon-fly-brain/data/`: `vnc_weight_csr.pkl`, `vnc_io_index.npz` (gitignored like the brain data — see §8).

---

## 7. Milestones (build in this order — each is a checkpoint)

1. **Data in hand.** `fetch_manc.py` produces the VNC CSR + index. Sanity: shapes, sign distribution, motor-neuron count per leg looks right (~tens per leg).
2. **VNC oscillates.** `vnc_stepper.py` under constant descending drive → motor pools show rhythmic, anti-phase activity (nonzero temporal std, tripod-like phase). **This is the make-or-break checkpoint.** If it won't oscillate after a real gain sweep + the 3-neuron sanity check, that's a publishable *negative* ("the isolated front-leg MANC subnetwork needs X to rhythm") — report it, don't force it.
3. **VNC walks the hexapod (open loop).** Constant "go" drive → `mn_to_joint` → hexapod walks forward without falling, no brain, no prey. Calibrate `k`/rest angles here.
4. **Turning.** Asymmetric descending drive → hexapod turns, correct sign.
5. **Close the loop.** Plug in the brain (Path 1 `brain()`) → `run_vnc_chase.py` → connectome-brain steers a **connectome-VNC-walked** hexapod to prey. Render an episode (parallels `episode_hexapod_chase.mp4`).
6. **The study.** Same controlled protocol as HANDOFF.md §6-A: n≥20, random starts, + controls (§below). Report caught-rate, time-to-catch, path efficiency vs controls.

---

## 8. Controls & success criteria (what makes it science, not a demo)

**Controls (each isolates one claim):**
- **Shuffled-VNC:** permute VNC weights, preserve degree/sparsity. If it still walks → the specific VNC wiring didn't matter.
- **Path-1 hand-CPG baseline:** same body, same brain, hand-CPG instead of VNC. Shows whether the real VNC does *as well or better* than the fake one (either result is interesting).
- **No-brain / random descending:** VNC driven by matched-statistics random input → shows the brain's steering contributes.

**Success = any of these, all publishable:**
- **Strong positive:** VNC-driven hexapod walks and the brain steers it to prey, ≥ hand-CPG baseline, beats shuffled-VNC. → "fly VNC connectome transfers to a non-fly body, zero learning."
- **Partial:** VNC oscillates and walks open-loop but closed-loop pursuit is worse than Path 1. → "rhythm transfers, fine motor control doesn't" — still a real finding about connectome portability.
- **Informative null:** front-leg MANC subnetwork won't rhythm in isolation / won't drive a stable hexapod gait. → "the VNC CPG is body-tuned; it doesn't port to foreign morphology without X."

**Zero-learning audit (run before every claim):** grep the whole pipeline for anything fit to a reward, a target trajectory, or catch-success. There must be none. Every gain is a declared, logged constant.

---

## 9. Verified prior work (cite these; know what's already taken)

- **Connectome simulations identify a CPG circuit for fly walking** (bioRxiv, Sep 2025, 10.1101/2025.09.12.675944) — dynamic sims of **MANC+FANC front-leg (T1)** neuropil; found command neuron **DNg100** + parallel **DNb08**, and a **minimal 3-neuron rhythm circuit (1 inhibitory + 2 excitatory interneurons)**; optogenetically confirmed in live flies. **This proves the VNC connectome makes a real gait rhythm** — the feasibility backbone of Path 2. It is a *circuit-identification in a network model*, **not** a non-fly physics body. Pull exact interneuron IDs from this paper. (biorxiv blocks automated fetch — open manually or via neuPrint.)
- **Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion** (arXiv 2602.17997) — connectome GNN, **supervised learning**, **fly body**. Not zero-learning, not non-fly.
- **NeuroMechFly v2** (Nature Methods 2024, 10.1038/s41592-024-02497-y) — fly biomech body; legs driven by **hand-designed hybrid CPG + sensory feedback**; connectome only constrains the *visual* net.
- **Whole-body physics simulation of fruit fly locomotion** (Nature 2025, 10.1038/s41586-025-09029-4) — MuJoCo fly, **RL-trained** controller.
- **Eon** — fly brain connectome → fly body, zero learning (fly body case scooped).

**Our unoccupied intersection:** fly connectome (brain **and** VNC) + **zero learning** + **non-fly body (hexapod)** + quantified transfer. None of the above occupy it.

---

## 10. Environment (from HANDOFF.md §7 — repeated so this doc stands alone)

- Venv is `uv`-managed at `eon-fly-brain/.venv`. Install: `VIRTUAL_ENV=eon-fly-brain/.venv uv pip install <pkg>` (you'll need `neuprint-python`).
- Run from project root with `./eon-fly-brain/.venv/bin/python` (relative venv paths break after a `cd`; Bash cwd persists).
- Present: torch 2.13.0, mujoco 3.12.0, numpy, imageio(+ffmpeg). Pyright import warnings on torch/mujoco are false positives.
- Hexapod body, qpos/qvel layout, leg/tripod indexing, all the gotchas: **HANDOFF.md §5** — read it, they'll bite you otherwise (dtype-cast-last, torque-motors-need-PD, global-sign-not-per-side-mirror).

---

## 11. First three commands for the new agent

```bash
# 1. read the two briefs
#    HANDOFF.md (Path-1 system) then this file (BUILD_PATH2.md)

# 2. get a neuPrint token from neuprint.janelia.org, then install the client
VIRTUAL_ENV=eon-fly-brain/.venv uv pip install neuprint-python

# 3. write & run fetch_manc.py -> vnc_weight_csr.pkl + vnc_io_index.npz  (milestone 1)
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/fetch_manc.py
```

Then march the milestones in §7. **Checkpoint 2 (does the VNC oscillate) is the whole ballgame — get there fast, before building anything downstream.**
