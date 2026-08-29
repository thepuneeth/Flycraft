# Flycraft — Research Handoff

> **One line:** Can the *whole-brain fly connectome* (FlyWire, zero learning) steer a body it never evolved for — a walking hexapod — through real physics, and where does it break?

This document is the single source of truth for anyone (human or agent) picking up this project. Read it top to bottom before touching code. Everything below reflects the current, honest state of the work as of 2026-08-29.

---

## 1. The research question (what we are actually testing)

**Claim under test:** The FlyWire connectome, run as a rate model with **no task training of any kind**, contains enough structure to produce *useful goal-directed steering* when wired into a non-fly body and closed through physics.

**The honest scientific framing — read this carefully, it is the core of the project:**

The fly's nervous system splits labour across two structures:
- **Brain** (FlyWire connectome, ~139k neurons) — sensing and *decisions*: where is the target, how much to turn, whether to go.
- **VNC** (ventral nerve cord, MANC connectome — a *separate* dataset we do NOT have) — the *walking rhythm*: the central pattern generators (CPGs) that actually oscillate the legs.

**FlyWire is brain only. It has no CPG.** We proved this empirically (see §4, the null result): under steady visual input the rate model settles to a near-fixed-point (descending-population activity std over time ≈ 0.013). A fixed point carries no rhythm, so **the brain connectome cannot self-generate a gait**. It can only steer.

So the defensible, novel claim is:

> **The brain connectome STEERS; a CPG (standing in for the missing VNC) WALKS.**

We give the connectome exactly the two signals a real fly brain hands down its descending neurons — *how much to go forward* and *which way to turn* — and let a hand-calibrated tripod CPG turn those into leg motion. The body then physically pursues a prey target in MuJoCo.

**Why this is publishable / novel (the gap, verified via web search):**
- **Eon** already did fly-connectome → fly-body → zero-learning. (Scooped for the fly-body case.)
- **FlyGM** (arXiv 2602.17997) uses deep RL + a fly body. (Not zero-learning.)
- **NeuroMechFly v2** uses a connectome-*constrained* visual net + hand-designed controllers, fly body. (Not whole-brain-drives-body.)
- **Nobody** crosses **fly whole-brain connectome + zero learning + a non-fly body** with **quantified competence and failure**. That intersection is the unoccupied gap. That is us.

**What counts as a result here** — both directions are publishable:
1. **Positive:** the connectome reliably steers the hexapod to intercept, beating a fair control. → "wiring alone transfers across bodies."
2. **Null:** it can't steer better than chance / a shuffled-weight control. → "the brain's steering is body-specific; structure doesn't transfer." Still a real finding.

The trap to avoid: **dressing up CPG behaviour as brain behaviour.** The CPG walks no matter what. The *only* thing the connectome contributes is `{forward, turn}`. All scientific claims must isolate that contribution (see §6 experiments).

---

## 2. Architecture (the closed loop)

```
prey position (world)
      │
      ▼
  retina()            body-frame encode: present?, azimuth az (+=left), distance, angular size
      │
      ▼
  CONNECTOME          FlyWire rate model, ZERO learning.
  (brain)             visual current in → 12 substeps → read descending populations
      │               dL, dR = mean firing of left/right descending neurons
      │               turn    = turn_sign · turn_gain · (dR − dL)
      │               forward = clip(fwd_gain · (dL+dR), 0, 1) · alignment_gate
      ▼
  CPG gait            tripod pattern generator (VNC stand-in). Turns {forward,turn}
  (hexapod_gait)      into 18 joint-angle targets. NO connectome inside it.
      │
      ▼
  PD torque           tau = kp·(target − q) − kd·qdot, clamped to actuator range.
      │               (PhantomX uses TORQUE motors — zero ctrl = legs go limp,
      ▼                a servo is mandatory.)
  MuJoCo physics      6 legs × 3 joints, real ground contact, freejoint trunk.
      │
      ▼
  new body pose  ──► loop back to retina()
```

**Key design decision — the alignment gate.** `forward` is multiplied by `gate = max(0.1, cos(az))` when prey is visible, else 0. Meaning: *only charge when roughly facing the prey; pivot in place when it's off to the side.* This is a sensory heading gate (biologically plausible: center the target, then charge), **not** learning. It is what converted endless orbiting into clean interception. Do not remove it without re-solving the orbit problem.

---

## 3. Files (what each one is)

All paths relative to project root `/Users/puneeth/Documents/Flycraft/`.

### Code — `eon-fly-brain/code/`

| File | Role |
|---|---|
| `flyserver.py` | **Reference connectome stepper.** Line-delimited JSON over stdio: in `{az,dist,loom,present}` → out `{turn,forward,dL,dR}`. This is the canonical rate-model dynamics and readout; everything else copies it. Was the bridge to the old Minecraft loop; kept as the clean reference implementation. |
| `run_hexapod.py` | **The NULL experiment (rich readout).** Full descending population (1291 neurons) → 18 joints via a *fixed random* linear map (seeded, never trained). No CPG. Tests whether the wiring alone produces any gait. Result: it does not (fixed point, no rhythm). This file *is* the evidence the brain has no CPG. |
| `hexapod_gait.py` | **The CPG + PD layer (VNC stand-in).** Pure gait engine, NO connectome. Tripod CPG → 18 joint targets from a `{forward,turn}` drive; `pd_torque()` servo. Run directly to test walk/turn/stand. This is the calibrated body controller. |
| `run_hexapod_chase.py` | **THE main experiment.** Connectome steers the walking hexapod toward prey. Imports `HexapodGait`, `pd_torque`, `XML` from `hexapod_gait`. Contains `retina()`, `brain()` (readout + alignment gate), the physics loop, metrics, and optional video render. **Start here.** |

### Assets — `assets/hexapod/`

| File | Role |
|---|---|
| `hexapod.xml` | PhantomX-class hexapod MJCF (18 DoF, torque motors). **Downloaded, not hand-built**, from `nico-bohlinger/one_policy_to_run_them_all`. Edited to add a visible red prey sphere geom at `(1.2, 0.8, 0.07)` (contype/conaffinity=0 so it's a marker, no collision). |
| `meshes/*.stl` | 7 STL meshes (base, motor, and mirrored L/R legs) referenced by the XML. |

Only `assets/hexapod/` is retained; the earlier Unitree quadruped experiment was removed.

### Data — `eon-fly-brain/data/`

| File | What it is |
|---|---|
| `weight_csr.pkl` | Signed 138,639 × 138,639 torch **sparse CSR** weight matrix. ~15M synapses. Neurotransmitter sign already baked in (excitatory +, inhibitory −). This IS the connectome. |
| `io_index.npz` | Neuron index arrays: `vis_left`, `vis_right` (visual/optic input neurons), `desc_left`, `desc_right` (descending output neurons), plus command-neuron indices. These are how you inject sensing and read out decisions. |

### Deliverable

`episode_hexapod_chase.mp4` — rendered preview episode (connectome steers hexapod, CAUGHT at 16.9 s). Regenerate with the video flag on `run_hexapod_chase.py`.

---

## 4. Results so far (what is established)

**Null result (run_hexapod.py) — the brain has no gait.**
- Fixed random readout, full descending population → 18 joints, no CPG.
- Descending activity std over time ≈ **0.013** (≈ frozen fixed point).
- Trunk displacement ≈ 0.065 m over the whole run. VERDICT: **no locomotion.**
- Interpretation: the rate model settles to a fixed point under steady input; it carries no rhythm; therefore the brain connectome cannot walk a body on its own. This is *expected and publishable* — it's the evidence that walking rhythm lives in the VNC, not the brain.

**Gait layer (hexapod_gait.py) — the CPG works.**
- Stands stably.
- drive 0.8 → forward **+0.185 m / 5 s**; faster gait (swing_amp 0.9) → ~0.094 m/s, no falls.
- turn +2 → yaw +45°; turn −2 → yaw −99° (correct signs, differential-drive steering).
- Calibrated defaults: `stance=(0.0, 0.6, -1.2), swing_amp=0.8, lift_amp=0.65, base_freq=1.5, drive_freq=2.5, turn_scale=1.0`, `kp=3.0, kd=0.05`.

**Main experiment (run_hexapod_chase.py) — connectome steers, single episode.**
- CLI: `--secs 20 --prey 1.2 0.8 --fwd_gain 12 --turn_gain 6` (turn_sign=−1).
- Azimuth nulled from +35° → centered; **CAUGHT at 16.9 s**; min dist 0.25 m; path 1.90 m; mean|az| 53°; no fall.
- **This is n=1, one seed, static prey, hand-tuned gains.** It is a demo, not a study. See §6.

---

## 5. Key facts, conventions & gotchas (do not relearn these the hard way)

**Connectome rate model (identical across all three run files):**
```
v ← decay·v + dt_tau·( gW · (W @ r) + I_ext )
r  = relu(tanh(v))          # bounded firing rate
decay = 1 − dt/tau,  dt_tau = dt/tau
```
Defaults: `substeps=12, tau=0.02, dt=0.005, gW=6e-3, drive_in=1.2`. `v` clamped to [−5, 5]. 12 substeps per control tick.

**Visual injection:** prey in LEFT field (az>0) excites `vis_left`; right field excites `vis_right`. `mag = drive_in·(0.5 + min(1,size))`, injected as `mag·(max(0,±az)+0.15)`.

**Readout:** `dL,dR = mean firing of desc_left/desc_right`. `turn = turn_sign·turn_gain·(dR−dL)`, `turn_sign=−1`. `forward = clip(fwd_gain·(dL+dR),0,1)`, then × alignment gate.

**Hexapod body:**
- 18 DoF = 6 legs × 3 joints (coxa, femur, tibia).
- Leg mounts: `LEFT={0,2,4}`, `RIGHT={1,3,5}`.
- Tripod groups: `TRIPOD_A={0,3,4}` (half-cycle out of phase with `{1,2,5}`).
- Leg `i` → ctrl indices `[3i, 3i+1, 3i+2]`.
- **TORQUE motors:** ctrl = torque (not target angle), ctrlrange ±1.96, gaintype/biastype = 0. **Zero ctrl → legs go limp.** A PD servo is mandatory.
- MuJoCo qpos/qvel layout: freejoint trunk occupies `qpos[0:7]` / `qvel[0:6]`; the 18 hinges are `qpos[7:25]` / `qvel[6:24]`.
- Yaw from quat: `yaw = 2·atan2(qpos[6], qpos[3])`.

**Gotchas that already bit us (don't repeat):**
1. **dtype upcast:** dividing a float32 numpy array by a Python float upcasts to float64 → torch `addmv` Float/Double mismatch. Cast to float32 **last**: `(rng.standard_normal(...)/np.sqrt(n)).astype(np.float32)`. Also cast `actuator_ctrlrange` to float32.
2. **Steering direction:** a per-side coxa *mirror* makes it SPIN on a straight walk. Correct approach is a **global sign negate** on the coxa swing (see `hexapod_gait.targets`), not per-side mirroring.
3. **Forward fade:** brain `forward = fwd_gain·(dL+dR)` shrinks as az nulls, so it stalls once centered. Fix: raise `fwd_gain` (12) so forward saturates while prey is visible.
4. **Orbiting:** with forward pinned high it spirals around the prey. Lowering turn_gain does NOT fix it. The **alignment gate** does.
5. **Invisible prey in video:** prey was only a coordinate; the red sphere geom in the XML is purely for rendering.
6. **argparse negative nargs:** `--stance 0.0 -0.6 1.2` parses `-0.6` as a flag. Drive stance sweeps via `python -c "from hexapod_gait import run; ..."`, not CLI.

---

## 6. What to do next (open work, in priority order)

**A. Turn n=1 into a real study (highest priority).**
The single CAUGHT run proves nothing statistically. Run a **paired, quantified comparison**:
- ≥20 episodes, random prey positions + random start headings (seed the RNG per episode, log it).
- **Control condition(s)** — this is what isolates the connectome's contribution. Options:
  - *Shuffled-weight control:* permute the connectome weights, keep sparsity/degree. If steering survives, the specific wiring didn't matter.
  - *Random-descending control:* replace the descending readout with matched-statistics random signals.
  - *Straight-to-prey oracle:* upper bound on the gait's ability.
- **Metrics per episode:** caught (bool), time-to-catch, min distance, path length / straight-line ratio (path efficiency), mean|az|, fell (bool).
- Report mean ± CI and a significance test connectome vs control. **That table is the paper.**

**B. Nail the null rigorously.**
`run_hexapod.py` shows one seed/scale gives no gait. Sweep seeds × readout scales (grid) and show *no* fixed random readout walks. Strengthens "the brain carries no rhythm" from anecdote to claim.

**C. Robustness / honesty checks.**
- Moving prey (currently static).
- Sensitivity to gains (fwd_gain, turn_gain, kp/kd) — show the result isn't a knife-edge tuning artifact.
- Rough terrain / perturbations — does connectome steering degrade gracefully?

**D. Polish (low priority, cosmetic).**
Nicer camera/lighting, longer clip, smoother chase. Demo only — does not change any claim.

---

## 7. Environment & how to run

**Venv is `uv`-managed** (not plain pip), at `eon-fly-brain/.venv`.
- Install a package: `VIRTUAL_ENV=eon-fly-brain/.venv uv pip install <pkg>`
- Present: torch 2.13.0, mujoco 3.12.0, imageio + imageio-ffmpeg, numpy.
- **Run from project root** with `./eon-fly-brain/.venv/bin/python` (relative venv paths break if you `cd` — Bash cwd persists between calls).
- Pyright import warnings on torch/mujoco are false positives.

**Run the main experiment:**
```bash
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_hexapod_chase.py \
  --secs 20 --prey 1.2 0.8 --fwd_gain 12 --turn_gain 6
```
Add `--video out.mp4` to render.

**Test the gait alone:**
```bash
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/hexapod_gait.py --mode walk --drive 0.8 --turn 0 --secs 5
```

**Reproduce the null:**
```bash
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_hexapod.py --secs 8 --scale 10 --seed 0
```

---

## 8. History (so a new agent doesn't re-litigate settled decisions)

- **Minecraft is retired.** Early work drove a Minecraft mob; the "walking" there was our CPG code, not the fly brain. Pivoted to MuJoCo for real physics and an honest brain/body split.
- **Body choice:** quadruped (Unitree Go1) → rejected; **hexapod** chosen because 6 legs match the fly. Reuse an existing MJCF, don't hand-build (user directive).
- **Readout choice:** "rich" (full population → random map) was tried and produced the null; the project now uses the **steering readout** (dL/dR descending means) + CPG.
- **Standing user preferences:** explain in simple plain language; prefer reuse over building; be honest about what's real science vs dressing; ask when not 100% sure.

---

## 9. The one-paragraph pitch (for a fresh collaborator)

We take the entire fruit-fly brain connectome (FlyWire, ~139k neurons, real synaptic wiring), run it as an untrained rate model, feed it what it sees of a prey, and read out the fly's descending "go forward / turn" commands. We proved the brain alone can't walk — it settles to a fixed point, because the walking rhythm lives in the fly's ventral nerve cord, which we don't have — so we add a hand-built tripod CPG as a stand-in and put the whole thing on a 6-legged robot in a physics simulator. The question: does 139k neurons of real fly wiring, with zero learning, steer a body it never evolved for well enough to catch prey — and if not, where and why does it fail? Nobody has tested fly-whole-brain + zero-learning + non-fly-body with quantified results. That's the gap.
