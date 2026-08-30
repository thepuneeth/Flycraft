# Path 3 — zero-learning step-timing decode + fixed IK gait engine

You are continuing the Flycraft project. Read this entire brief, then
`HANDOFF.md` and `PATH2_METHODS.md`, before writing any code. You have web
access; use it in Phase 0 to study the source method thoroughly. Work the phases
in order. Every phase has a GATE — post the required numbers, then continue.
Never write a final headline claim; stop for user approval at the end.

This is a **zero-learning** experiment. It returns to the project's original
discipline: nothing is trained, nothing is optimized, no reward exists. If at any
point you find yourself tuning a number to make the body walk, you have broken
the experiment — stop and report instead.

---

## 1. Why this experiment exists (read this or you will build the wrong thing)

Four rungs have already failed to make a fly connectome walk a PhantomX-class
hexapod, all committed:

- **E0–E4** (zero-learn, direct motor map): oscillation, no tripod. ABSENT.
- **Reservoir readout** (learn one linear/nonlinear map): READOUT TOO WEAK.
- **Outcome C** (CMA-ES tune ~18 global gains, weights frozen): connectome
  0.067 m *worse* than random 0.158 / rewired 0.167. TUNED NEGATIVE, moat NULL.
- **Outcome D** (learn the weights, FlyGM-style imitation + PPO): imitation stays
  upright, locomotion fine-tune falls 3–4/4 every condition, connectome no better
  than random/rewired. NO STABLE WALK, topology NULL.

One positive survives all of it: **real VNC wiring drives far more open-loop
movement than column-shuffled VNC** (+0.1425 m path, exact p = 1.91e−6, Holm
p = 5.72e−6). The wiring carries real motor structure. It just does not carry a
*tripod trajectory* directly to a body it never evolved for.

**The diagnosis (from a literature sweep — verify it yourself in Phase 0):**
every project that successfully moves a body from a neural model inserts a
**body-appropriate gait layer** and uses the neural model as *command / steering
/ prior* — never as the raw joint-angle generator. The single project that
solved our exact problem — a real *C. elegans* connectome driving a **multi-
legged robot** with a different body plan (Sun et al., "Innate Orientating
Behavior of a Multi-Legged Robot Driven by the Neural Circuits of C. elegans,"
PMC11201571) — did it with **no training**, via a two-tier decode:

1. neural activity → **discrete commands** (per leg-region: extend / contract;
   plus a hold-course / turn decision), **not** joint angles;
2. a fixed **inverse-kinematics / gait primitive** turns each command into the
   actual joint motion.

We did the opposite: we mapped VNC motor rates straight onto 18 joint targets.
No successful cross-body project does that. **This experiment tests the interface
the field's only cross-body success actually used, at zero learning:** the frozen
fly connectome supplies *step timing*, a fixed gait engine supplies *the step*.

If the connectome's rhythm carries usable timing, the body walks and real wiring
beats shuffled. If it does not, we get the strongest possible negative. Either
outcome is honest and publishable. A walk that only appears after tuning is not.

---

## 2. The honesty spine (hard rules — violating any voids the result)

- **ZERO LEARNING.** No reward, loss, optimizer, backprop, ridge fit, evolution,
  or performance-based tuning anywhere. The brain and VNC stay byte-identical to
  the committed frozen chain. Confirm by hash.
- **The gait engine is a *fixed declared solver*, not learning.** It is the same
  category as the already-accepted fixed motor map: a kinematic function chosen
  a priori and never edited to improve a result. Prefer reusing the **Path-1
  hand-CPG per-leg swing/stance primitive** unchanged (it already walks this
  exact hexapod). You are not allowed to hand-tune its shape, amplitude, or
  timing to the connectome.
- **The decode rule must be declared a priori and, ideally, parameter-free.**
  The preferred rule is an **antagonist comparison**: a leg is in swing when its
  levator/protractor motor-pool activity exceeds its depressor/retractor pool,
  else stance — i.e. the *sign* of a fixed antagonist difference read from the
  frozen VNC. Parameter-free = nothing to tune = honest. If you must use a
  threshold, set it by a declared rule computed on a **burn-in window only**
  (e.g. each pool's own median), never by walking distance, and record it once.
- **No tripod, gait, coherence, catch, prey, or distance term may influence any
  choice.** Tripod coherence is an *evaluation metric only*, computed after the
  fact with the existing E0 probe. You never select anything to increase it.
- **Steering stays brain-driven and unlearned**, exactly as in the committed
  chain (`run_vnc_chase.py`).
- **Controls are mandatory** (Phase 4). Real VNC timing vs column-shuffled VNC
  timing vs random timing, all through the *identical* gait engine. The whole
  point is to separate "the connectome timed the gait" from "the gait engine
  walks no matter what." If shuffled/random time the legs just as well, the
  honest reading is that the gait engine is the walker — report it plainly.
- **Report nulls and failures verbatim.** If it does not walk, say so. If real
  ties shuffled, say so. Do not manufacture a walk.

When in doubt, stop and report rather than tune.

---

## 3. Phase 0 — research the method on the web (≤ 90 min) [GATE 0]

Study these yourself, extract *method only*, and write a "Path 3 method sources"
section in `PATH2_METHODS.md` (3–8 lines each: what to reuse, what to avoid, and
confirm it needs no learning):

- **Sun et al., C. elegans multi-legged robot** — PMC11201571. This is the
  template. Extract exactly: how neural activity becomes discrete per-leg
  commands (they use motor-neuron voltage thresholds and OLQ-RMD pairs mapped to
  four leg *regions*), how inverse kinematics converts commands to joint motion,
  and their explicit statement that the robot's kinematics differ from the worm's
  so a direct whole-brain mapping is impractical. Confirm: no training.
- **NeuroMechFly v2 hybrid controller** — neuromechfly.org hybrid-controller
  tutorial + Nature Methods 2024. Extract the *shape* of a correct per-leg
  swing/stance primitive and the sensory-feedback rescue rules (lift a stuck
  leg). Use only as a reference for what a clean step looks like; do **not**
  import its designed CPG as our timing source — our timing must come from the
  connectome.
- **NCAP** — arXiv 2410.07174. Extract how a circuit *motif* is used as a fixed
  structural prior with the body doing the mechanics; context for why a
  body-appropriate layer plus a neural command layer generalizes across bodies.
- **Connectome-CPG** — bioRxiv 2025.09.12.675944. Confirms rhythm comes from the
  network, tripod needs embodiment — the justification for letting the body's
  gait engine own the step while the connectome owns timing.
- Optionally skim `flygym` and `flybody` source for how a swing/stance phase
  variable drives leg joints, to implement our primitive cleanly.

**GATE 0:** post the method-sources section. State, in one line, that the chosen
decode + gait engine require zero training and name exactly which rule you will
use for the swing/stance decode and which primitive for the step.

---

## 4. The method (build to this exact data flow)

```text
prey -> FlyWire brain (frozen) -> DNg100/DNb08 -> real MANC VNC (frozen)
     -> per-leg motor-pool activity r_leg(t)
     -> DECODE: swing/stance state s_leg(t)  [fixed antagonist-sign rule]
     -> FIXED GAIT ENGINE: s_leg drives Path-1 per-leg swing/stance primitive
     -> 18 joint targets q(t) -> PD torque -> MuJoCo hexapod
brain also supplies steering, exactly as in the committed chain
```

The connectome's *only* job is `s_leg(t)` — when each of the six legs steps. The
gait engine owns the actual joint trajectory of every step, identically for all
conditions. This is the worm-robot interface, adapted to six legs and zero
learning.

---

## 5. Build phases

Run from `/Users/puneeth/Documents/Flycraft` with `eon-fly-brain/.venv/bin/python`.
Reuse existing modules — do not reimplement:
`vnc_stepper.py` (frozen VNC dynamics + shuffle control), `mn_to_joint.py`
(locate the per-leg motor pools and the 18 joint channels), `run_vnc_chase.py`
(MuJoCo setup, PD servo, camera rig, brain steering), `probe_vnc_phase.py` (E0
tripod-coherence metric — reuse verbatim), the Path-1 hand-CPG code (source of
the fixed step primitive), and `cmaes_walk.py` (reuse its rollout + distance/
fall/upright metrics + fixed-seed handling so Path 3 is measured identically).

### P1 — decode module → `code/step_decode.py` [GATE P1]
1. From the frozen VNC, for each leg identify its swing (levator/protractor) and
   stance (depressor/retractor) motor pools using `mn_to_joint.py`'s channel
   map. Document which motor-neuron groups you assigned and why (anatomy, not
   performance). This is a data-selection choice — allowed — declared once.
2. Implement the a-priori decode: `s_leg(t) = swing if pool_swing(t) >
   pool_stance(t) else stance` (parameter-free antagonist sign). If you instead
   use a declared burn-in threshold, record it and justify the rule.
3. Run the frozen VNC open-loop and log `s_leg(t)` for all six legs. Report:
   per-leg step frequency, duty factor (swing fraction), and the E0 tripod-
   coherence of the *decoded step states* (this is measurement, not a target).
4. **GATE P1:** post per-leg step frequency + duty factor + decoded-state tripod
   coherence, for real VNC and (as a preview) column-shuffled VNC. If real and
   shuffled step states are already indistinguishable here, note it — it
   foreshadows a null.

### P2 — fixed gait engine → `code/gait_engine.py` [GATE P2]
1. Wrap the Path-1 per-leg swing/stance primitive as a pure function:
   given a leg's current phase and its commanded state `s_leg`, advance one step
   of the fixed lift-swing-place (swing) or push-back (stance) trajectory and
   return that leg's 3 joint targets. No amplitude/shape tuning.
2. Sanity check in isolation: drive the engine with a *clean external tripod
   clock* (not the connectome) and confirm it walks the hexapod with a visible
   tripod. This proves the engine is correct before the connectome drives it, and
   gives the reference ceiling. Record its forward distance + coherence.
3. **GATE P2:** post the engine-only (clean-clock) forward distance and tripod
   coherence — the ceiling this experiment could reach if timing were perfect.

### P3 — closed loop: connectome times the engine → `code/run_stepwalk.py` [GATE P3]
1. Each control step: read live per-leg pool activity from the frozen VNC →
   `s_leg(t)` via `step_decode` → `gait_engine` → 18 joint targets → PD → MuJoCo.
   Brain supplies steering as in `run_vnc_chase.py`.
2. Measure with the reused metrics: forward distance, path length, displacement,
   falls, upright height, duty factor, and E0 tripod coherence on the realized
   leg phases. Render the multiview video.
3. **GATE P3:** post distance + coherence + falls + the video verdict (visible
   coordinated walk vs sprawl vs lurch). No headline language.

### P4 — the moat controls (mandatory) → `code/stepwalk_controls.py` [GATE P4]
Repeat P3 under the **identical** gait engine, decode rule, seeds, and steering,
changing only the timing source:
- **real:** real MANC VNC step timing.
- **shuffled:** column-shuffled VNC (existing `vnc_stepper.py` machinery), timing
  from the shuffled dynamics. Declare and seed the shuffle.
- **random:** step onsets drawn from a matched-rate random process (same mean
  step frequency and duty factor as real, phases random). Seed it. This is the
  floor: if random timing walks as well as real, the gait engine is the walker.
- Use the paired study design from the committed control study (N paired episode
  starts, shared initial yaw/prey, static prey) and the same exact paired
  sign-flip test with Holm adjustment, so Path 3's real-vs-shuffled test is
  directly comparable to the committed one.

Report the table — real vs shuffled vs random:
- forward distance [95% bootstrap CI], path, falls, upright;
- tripod coherence of realized leg phases;
- paired real−shuffled and real−random distance differences, exact p, Holm p.

Verdicts:
- **CONNECTOME TIMES THE GAIT:** real walks and beats shuffled *and* random on
  distance/coherence. The zero-learning positive — headline candidate.
- **ENGINE-DRIVEN NULL:** shuffled or random walk as well as real → the gait
  engine walks regardless of timing source; the connectome is not doing the
  timing work. Honest null, still publishable.
- **NO WALK:** nothing produces coordinated forward walking even with the engine
  → strongest negative; timing from the frozen VNC is too incoherent to drive a
  gait, confirming the transfer failure from a fifth independent angle.

### P5 — evaluation + report
Best of each condition: four fixed 8 s seeds, multiview video, coherence via
`probe_vnc_phase.py`, distance/falls/upright. Write the full Path 3 section in
`PATH2_METHODS.md`: method-sources, motor-pool assignment, decode rule (show it
is parameter-free or a-priori), gait-engine declaration, per-condition table,
verdict. Update `HANDOFF.md` with a Path 3 result block.

### P6 — STOP for user approval
Do **not** write any final headline/claim wording. Post the assembled evidence
and proposed verdict, then stop.

---

## 6. Anti-cheat guardrails (the reviewer will look for these)
- The gait-engine primitive is Path-1's, unmodified. Diff it against the Path-1
  source and confirm byte/shape equality where intended. If you changed its
  amplitude, timing, or shape, the result is void.
- The decode rule is fixed before you ever run P3. You may not revisit it after
  seeing P3/P4 distances. If you change it, you must rerun P1 and declare the
  change and the reason (anatomical, never performance).
- No parameter anywhere is selected to increase distance, walking, or tripod.
  Grep the code before every commit for any distance/tripod/catch/reward term
  feeding a choice — there must be none.
- The three timing conditions differ *only* in the timing source. Same engine,
  same seeds, same steering, same everything else. Confirm by construction.
- If the walk only appears with a hand-picked motor-pool assignment, report the
  full set of anatomically defensible assignments you tried and whether the
  result survives all of them. One lucky assignment is not a result.

## 7. Checkpointing, integrity, deliverables
- Checkpoint per phase; a crash loses < 15 min. Commit after each completed phase
  with a message naming the phase + verdict. Gitignored large blobs stay
  gitignored; commit code + small metrics + report + video.
- Before each commit: python-compile all `code/*.py`; JSON/CSV parse; confirm VNC
  and brain weights hash-equal to the frozen graph; confirm the gait primitive
  unchanged; zero-learning scan (no optimizer/backprop/reward/loss/target-fit);
  confirm the three timing conditions share the engine; `git diff --check`.
- Deliverables: `code/step_decode.py`, `code/gait_engine.py`,
  `code/run_stepwalk.py`, `code/stepwalk_controls.py`;
  `results/path2/stepwalk_metrics_{real,shuffled,random}.json`,
  `results/path2/stepwalk_controls.{json,csv,png}`,
  `results/path2/stepwalk_multiview.mp4`; the `PATH2_METHODS.md` Path 3 section;
  the `HANDOFF.md` Path 3 block. No final claim wording until the user approves.

## 8. Hardware
CPU only. VNC dynamics, decode, and gait engine are cheap; MuJoCo rollouts are
CPU. No GPU/RunPod needed. RunPod needs auth this agent does not have — stay
local.

---

## 9. One-line honest summary of what Path 3 claims to test
Whether a **frozen** fly connectome, decoded only as **step timing** into a
**fixed** hexapod gait engine, produces coordinated forward walking — and whether
**real wiring times it better than shuffled or random** — with **zero learning**
and no gait ever rewarded. A walk that needs tuning to appear is not a result;
report the honest outcome whichever way it falls.
