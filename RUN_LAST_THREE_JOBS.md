# Test the last three jobs — coordination, waveform, pose — honestly

**Read this whole file, then `HANDOFF.md`, `PATH2_METHODS.md`, and
`docs/archive/RUN_CONNECTOME_DRIVE.md` (the Methods 1–2 run) before writing any code.**
You are continuing the Flycraft project. This is the hardest run and the one most
likely to produce **negative results — which is exactly the point.** Do not treat a
negative as failure and do not tune to escape it. Work the phases in order. Every phase
has a GATE that must print PASS (or a declared NULL) before the next one starts.

---

## 0. The number, and what "success" means here

The moat: a **frozen, untrained** fly connectome (FlyWire brain + MANC VNC) driving a
**non-fly hexapod**, zero learning. The question is **how much of the walk the raw
wiring generates.** Honest job table and where we stand after Methods 1–2:

| Job | Owner now | This run tests |
|---|---|---|
| Steering | connectome ✓ | — |
| Step timing | connectome ✓ | — |
| Rhythm / duty | connectome ✓ (M2, earned) | — |
| **Coordination** | engine (M1 FAILED to take it) | **Phase B — dynamic coupling** |
| **Waveform** | fixed engine primitive | **Phase C — direct MN→joint map** |
| **Pose** | model `home` keyframe | **Phase D — connectome tonic stance** |

- **We are at ~30–35%.** These three jobs are the remaining ~65%.
- **These are the jobs that likely DO NOT transfer without training.** Method 1 already
  showed coordination does not transfer from tonic phase. The E0–E4 series already
  showed the direct motor map was absent — but on a *broken* body. This run re-tests the
  strongest honest version of each, **on the corrected standing body.**
- **Expected outcome: mostly negative.** That is a publishable boundary result:
  *"the frozen wiring earns steering + timing + rhythm, but coordination, waveform, and
  pose it cannot supply un-trained — and here is the exact evidence."* A clean negative
  here **finishes the paper.** A faked positive destroys it.
- **~30–35% may already be near the honest zero-training ceiling.** If these phases
  confirm that, you have found the ceiling. Do not chase 100%. Do not add a trained
  layer to force a higher number — that crosses into everyone else's territory and
  kills the novelty.

**Pre-register the null for each phase in writing before you run it.** State the exact
number that would count as "the connectome earned this job" and the number that would
count as NULL, *before* you see the result. Then report whichever happened.

---

## 1. Honesty spine (carry forward — plus rules specific to these harder jobs)

1. **Zero learning. Zero.** No optimizer, backprop, reward, loss, target-fit, or any
   quantity derived from distance/coherence/catch feeding any choice. VNC weights + brain
   **frozen and hash-equal** to the committed graph.
2. **The fixed maps stay fixed.** `mn_to_joint.translate()` uses per-joint gains
   `(-0.55, 0.65, 0.65)` — these are the **declared anatomical map, not tunable
   parameters.** Do not change them to make the body walk. If the raw map produces bad
   joint angles, report bad joint angles. No "correction" layer, no rescale-to-fit.
3. **The coupling weights ARE the connectome edges** (from `audit_intersegmental`), not
   tuned coupling strengths. You may calibrate **one global coupling gain** — and only
   against clean-clock stability (connectome-blind), shared identically across conditions.
4. **Pose is not searched.** Either the connectome's tonic drive implies a stance that
   stands, or it does not. Do not hunt for a standing stance. If it collapses, pose stays
   with the `home` keyframe (declared, correct) and you report "connectome did not earn pose."
5. **One engine / one pipeline, shared across conditions.** real / shuffled / random differ
   only in the *source* (edges, or weights). Same everything else.
6. **Every phase gets the three controls** (`real`, `shuffled`, `random`) with paired
   real−shuffled and real−random, bootstrap CIs, Holm-corrected p — and a rendered video
   of `real` that you eyeball.
7. **When in doubt, STOP and report. Never tune to rescue a result. A negative is the deliverable.**

---

## 2. Phase A — pre-flight (MANDATORY; reuse `code/preflight.py`; ~2 min)

The last-good run added `preflight.py` and it caught nothing because the body was already
fixed — keep it that way. Re-run the six-trap gate **before touching B/C/D:**

| # | Trap | Pass |
|---|---|---|
| T1 | Squat stance | static settle ≥ 0.12 m |
| T2 | Fall floor hides squat | `H_MIN = 0.08` everywhere |
| T3 | No L/R mirroring | stance == `m.key("home").qpos[7:25]` |
| T4 | Engine can't walk | clean-clock engine ≥ 1.0 m / 8 s, no fall |
| T5 | Broken duty | 0.35 ≤ clean-clock duty ≤ 0.6 |
| T6 | Weights changed | SHA-256(weight+index) == committed |

**GATE A:** print `GATE A: PASS (T1..T6 = numbers)` or `GATE A: FAIL <trap>` and stop.
Do not proceed on FAIL. This is the wall that stops another 10-hour waste.

---

## 3. Phase B — Coordination, the honest strong version: dynamic inter-leg coupling

Method 1 imposed a **static grouping** read from tonic phase → failed (real fell 16/20,
worse than shuffled). The stronger, more faithful test: let coordination **emerge from
running dynamics** by coupling the six per-leg phase oscillators (Method 2's continuous
VNC phases) **through the connectome's real intersegmental edges.**

**How:**
1. Get the anatomical cross-segment coupling from `audit_intersegmental` (`cross_edges` →
   pre/post/weight between T1/T2/T3 leg pools). Reduce to a signed 6×6 inter-leg coupling
   matrix `K` (leg→leg), sign and magnitude from the real edges. **`K` is fixed anatomy.**
2. Run six phase oscillators; each leg's phase is advanced by Method-2's own VNC phase
   **plus** a coupling term `g · Σ_j K[i,j]·sin(θ_j − θ_i)` (standard Kuramoto coupling,
   coupling matrix = connectome). Coordination is now emergent, not labelled.
3. **`g` = one global coupling gain, calibrated only against clean-clock stability**
   (does the coupled system settle to a stable limit cycle on a neutral input), blind to
   distance, shared across conditions.
4. Feed the coupled phases to the same Phase-A engine waveform. Nothing else changes.

**Controls:** `K` from real edges / shuffled edges (permute) / random matrix (matched
sparsity+magnitude). Same `g`, same engine.

**Pre-registered null (write before running):** real-edge coupling produces coordination
(tripod coherence, feet-planted, path) no better than shuffled/random edges → coordination
does not transfer even as emergent coupling.

**GATE B:** real/shuffled/random table (path CI, coherence, feet-planted, falls) + paired
tests + `real` video. Commit `lastthree: B dynamic coupling`.

---

## 4. Phase C — Waveform, the maximal test: connectome drives all joints directly

Currently the intra-step joint trajectory is the engine's fixed sinusoid. The honest test
of "connectome generates the waveform" is to **remove the engine waveform entirely** and
let the VNC's real motor output set all 18 joint targets each control step, via the
**fixed** map `mn_to_joint.translate(motor_activity, io)`. This is the E0–E4 direct motor
map — **re-tested on the corrected standing body it never had.**

**How:**
1. Each control step: `motor = vnc.step(descending_current(forward*100, turn))`;
   `targets = mn_to_joint.translate(motor, io)` (gains frozen at `(-0.55,0.65,0.65)`);
   PD-servo to `targets`. No engine, no decode, no phase machine — the connectome's motor
   output *is* the command. (Optional C2: keep Method-2 timing/coordination but take only
   the intra-step *shape* from the MN map — run only if C1 is not flatly dead.)
2. Same frozen VNC protocol, standing body, H_MIN=0.08.

**Controls:** real / shuffled / random motor sources through the identical fixed map.

**Pre-registered null:** raw MN output does not produce a walkable joint waveform (path
near zero or falls), reproducing the E-series ABSENT verdict now that the body can stand →
waveform does not transfer un-trained. (This is the expected result; report it cleanly.)

**GATE C:** real/shuffled/random table + `real` video + explicit statement of whether the
E-series ABSENT verdict reproduces on the standing body. Commit `lastthree: C direct MN waveform`.

---

## 5. Phase D — Pose: can the connectome's tonic drive stand the body up?

Pose currently comes from the `home` keyframe (correct). Honest test: does the VNC's
**tonic** output (baseline descending drive, rhythm suppressed) imply a stance that stands?
1. Run the VNC at a constant low drive to a steady state; take the mean motor vector;
   `stance = mn_to_joint.translate(mean_motor, io)`.
2. Hold that stance with the PD servo for 1.5 s (no gait). Measure settle height.
3. **If settle ≥ 0.12 m** → the connectome earns pose. **If it collapses (< 0.12)** →
   pose stays with the `home` keyframe, declared, and you report "connectome did not earn
   pose." Do **not** search for a standing stance.

**GATE D:** settle height for real / shuffled / random tonic stance + verdict. Commit
`lastthree: D connectome tonic pose`.

---

## 6. Phase E — reconcile and score (NO final claim)

Update `PATH2_METHODS.md`:
- Section "Last three jobs (coordination / waveform / pose)" with the B/C/D tables and the
  pre-registered nulls next to the outcomes.
- **Re-score the job table (§0) and the connectome %.** State plainly which of the three
  jobs (if any) transferred, and the new estimate. If none transferred, say the honest
  ceiling is ~30–35% and this run establishes it — the boundary result.
- Cross-reference: does Phase C reproduce the E-series ABSENT verdict on the standing body?
  Does Phase B confirm Method 1's coordination negative by a second, stronger method?

**STOP. Print `READY FOR CLAIM REVIEW` and wait.** The user writes/approves the claim.

---

## 7. Per-commit integrity gate (before EVERY commit)

1. `python -m compileall -q eon-fly-brain/code` clean.
2. New JSON/CSV parse.
3. VNC weight+index hash-equal to committed (T6).
4. `mn_to_joint` gains still `(-0.55, 0.65, 0.65)`; coupling matrix built from real edges,
   not hand-set numbers; stance provenance intact.
5. **Zero-learning scan** on the diff: `optim|backward|loss|reward|fit(|train|\.grad|
   target_fit|argmax.*dist|rescale.*(path|dist|walk)` — any hit justified or removed.
6. Shared-pipeline assert: real/shuffled/random construct identical objects except source.
7. Fall floor 0.08 everywhere. `git diff --check` clean. Message names phase + result.

---

## 8. Anti-mistake / fail-fast contract (the no-wasted-time rules that worked last run)

- **Every phase runs a <2-minute smoke test** (build source, few steps, eyeball) before
  the 20-episode ladder. If the smoke looks wrong, stop — never launch the long run "to see."
- **Never trust a metric without watching one `real` video.** A number can pass while the
  body drags or flails (Method 1 "flew" while pathing 1.5 m).
- **One thing at a time:** B, then C, then D. No stacked untested changes.
- **If a gate fails or returns NULL, that is a result — record it and move on.** Do not
  loop trying to turn a null into a positive.
- **No new dependencies.** Reuse `hexapod_gait.py`, `gait_engine.py`, `mn_to_joint.py`,
  `audit_intersegmental.py`, `vnc_stepper.py`, `probe_vnc_phase.py`, `preflight.py`.
- **Checkpoint + commit after every gate.** CPU only; each gate is minutes. Hours = a bug;
  stop and diagnose, do not wait it out.

## 9. One-line summary
On the corrected standing body, test the three jobs the connectome does not yet own —
coordination (emergent coupling through real intersegmental edges), waveform (direct fixed
MN→joint map, the E-series re-tested), and pose (connectome tonic stance) — each with a
pre-registered null, each against shuffled and random, never tuned, fixed maps frozen.
Report which transfer and re-score the connectome's honest share (~30–35% now). Most are
expected NULL; a clean negative is the paper's boundary result. No final claim.
