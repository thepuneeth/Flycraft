# Body-setup fix + full re-run on a body that can actually stand

You are continuing the Flycraft project. Read this entire brief, then `HANDOFF.md`
and `PATH2_METHODS.md`. A setup bug has been found that **confounds every
committed result**. Your job: fix the shared gait engine so the hexapod actually
stands at its designed ride height, verify it walks, then **re-run the whole
experiment ladder on the corrected body** and report which prior verdicts
survive. Work the phases in order; every phase has a GATE. Never write a final
headline claim — stop for user approval at the end.

---

## 1. The bug (confirmed, with evidence)

`code/hexapod_gait.py` defines the tripod-CPG stance as `(coxa, femur, tibia) =
(0.0, 0.6, -1.2)` applied **identically to all six legs**. Two things are wrong:

1. **The stance squats.** Holding that pose with the PD servo (no gait at all)
   settles the trunk at **z = 0.073 m**. The model's own `home` keyframe stands
   at **z = 0.161 m** (static) / **0.139 m** (settled). At 7 cm the body drags
   its belly — legs cannot clear the ground or get propulsion. Every VNC study
   (E0–E4, reservoir, Outcome C, Outcome D, Path 3) ran on this squat.
2. **No left/right mirroring.** The PhantomX right legs `{1,3,5}` are mounted
   mirrored (body quat rotated −90° about z). The model's `home` keyframe
   reflects this: left legs `{0,2,4}` use `femur +0.6, tibia -2.0`; right legs
   `{1,3,5}` use `femur -0.6, tibia +2.0`. The gait applied the same sign to all
   six, so the right legs were driven against their geometry.

Diagnostic evidence (already reproduced; you should reproduce it too):
- settle at old stance → z ≈ 0.073; a real Path-3 episode holds z ≈ 0.07–0.08 the
  whole run while the brain drive is healthy (`forward` mean ≈ 0.78). The
  "0/20 falls, upright" claim is an artifact of the `H_MIN = 0.02` fall floor:
  2 cm is far below the 15 cm ride height, so a belly-drag squat passes as
  "upright."
- **Fix, verified:** using the model's `home` pose as the per-leg stance (coxa
  spread ∓0.7 on front/back, femur ±0.6 by side, tibia ∓2.0 by side) plus a
  mirrored coxa sweep and mirrored swing lift, the pure engine (clean external
  tripod clock) stands at z ≈ 0.13 and walks **2.03 m in 8 s at freq 1.2**
  (25 cm/s), **3.0/6 feet planted** (clean alternating tripod), **no fall** —
  versus **0.09 m** belly-drag before.

**Consequence:** the connectome-vs-controls verdicts (`real < random`, `no
tripod`, `topology NULL`, `moat NULL`) were all measured on a body that could not
walk on *any* input. They are **confounded** and must be re-measured. They may
still hold — but until re-run on a standing body, no negative is publishable.

---

## 2. Honesty spine (unchanged; the fix must not become cheating)

- **The stance fix is correctness, not tuning.** Take the standing pose from the
  model's own `home` keyframe (author-declared, connectome-blind). Do not search
  the stance to improve walking distance; use the model's pose. Record that this
  is where it came from.
- **The gait engine is the *shared* gait layer.** You may calibrate the engine's
  own parameters (stance from home, sweep amplitude, lift, frequency mapping,
  duty factor) **only against a clean external tripod clock** — never against
  connectome performance, distance-with-the-connectome, or any control's score.
  The identical calibrated engine is then used for real, shuffled, and random.
  This preserves the moat logic (differences come only from the timing source).
- **Zero performance-based tuning of anything connectome-facing.** The VNC weights
  and brain stay frozen and hash-equal. No reward, loss, optimizer, or target
  fitting is added anywhere. The decode rule stays the declared a-priori rule.
- **No gait/tripod/catch/distance term** may influence any choice. Tripod
  coherence and distance are measured, never optimized.
- **Report confounding honestly.** State plainly in the report that prior results
  were confounded by the squat and are superseded by the re-run. If a prior
  negative survives on the standing body, say so; if it flips, say so.
- **Direction is a declared choice.** The gait has an inherent fore/aft asymmetry
  (one sweep direction gives a strong stroke ~0.5–2 m, the other ~0.17 m). Pick
  the strong direction as "forward," orient the prey/heading to it, and **declare
  this once**. Do not tune the asymmetry per condition. (Optionally investigate
  why it is asymmetric and symmetrize it — still connectome-blind.)

When in doubt, stop and report rather than tune.

---

## 3. Phase A — implement + verify the fix (the gate everything depends on)

Edit `code/hexapod_gait.py` (and `code/gait_engine.py`'s `STANCE`) so the engine:

1. **Standing stance from `home`.** Per-leg targets taken from the model's `home`
   keyframe: coxa spread (front `-0.7` / mid `0.0` / back `+0.7` pattern as in the
   keyframe), `femur = +0.6*side`, `tibia = -2.0*side`, where `side = +1` for
   legs `{0,2,4}` and `-1` for `{1,3,5}`. Read these from `m.key("home").qpos`
   rather than hard-coding, so they track the asset.
2. **Mirrored sweep and lift.** Coxa fore/aft sweep and the swing-phase lift are
   multiplied by `side` so both leg banks move in the same world direction. Lift
   raises the leg on the swing half only; the foot is planted through the stance
   (power) half. Keep the alternating tripod `TRIPOD_A = {0,3,4}` vs `{1,2,5}`.
3. **Keep the PD servo and control rates unchanged** (`kp=3.0, kd=0.05`,
   `CTRL_HZ` as used, torque clamp to `ctrlrange`).

Reference parameters that already walk (calibrate only against the clean clock,
adjust only if needed for a clean tripod): `coxa_amp ≈ 0.35`, `lift ≈ 0.7`,
`base/drive freq` giving ~1.2–1.8 Hz step rate. Target a **duty factor near 3/6
feet planted** under the clean tripod clock.

**GATE A (hard):** post, for the pure engine under a clean external tripod clock:
static settle height, walking distance over 8 s, mean ride height, min height,
fell flag, and mean feet-planted. It must **stand (ride height ≳ 0.12), walk
(≳ 1 m / 8 s), not fall, and hold ~3/6 stance**. If it does not, fix the engine
before touching any VNC condition. Do not proceed on a body that cannot walk.

Also confirm the old bug is gone: settle at the new stance ≫ 0.073.

---

## 4. Phase B — re-run the ladder on the corrected body

Re-run each prior study **unchanged except for the fixed engine/stance**, using
the same frozen VNC protocol, same seeds, same paired study design, and the same
metrics/tests as the committed versions. For each, report the corrected-body
result next to the old confounded number and state whether the verdict survives.

Order (cheap → expensive; checkpoint and commit after each):

1. **B1 — Path 3 step-decode (real / shuffled / random).** Rerun
   `run_stepwalk.py` + `stepwalk_controls.py` on the fixed engine. This is the
   primary test: on a body that can walk, does connectome-derived step timing
   drive coordinated forward walking, and does real beat shuffled and random?
   Also re-check the **decode duty factor** — the prior real decode gave ~0.71
   swing fraction (legs lifted 71% of the time), a broken duty cycle independent
   of the stance bug; report whether the antagonist-sign decode yields a sane
   (~0.5 or lower) swing fraction on the corrected body, and if not, note it as a
   decode limitation (do not tune it to walk).
   **GATE B1:** post the real/shuffled/random table (distance CIs, coherence,
   feet-planted, paired real−shuffled and real−random tests) + video verdict.

2. **B2 — control study (real VNC / brain-coupled / shuffled / random descending
   / Path-1).** Rerun `study_vnc.py` + `analyze_control_study.py`. Does the
   +0.1425 m real−shuffled positive hold on the standing body? Does brain-coupled
   now separate from random descending?
   **GATE B2:** post the 5-condition table + the three paired tests.

3. **B3 — E0/E-series phase probes.** Rerun `probe_vnc_phase.py` and the E0
   tripod-coherence on the realized joint phases, now that the body actually
   steps. Report whether tripod coherence is still absent.
   **GATE B3:** post the coherence matrix + verdict.

4. **B4 — Outcome C (CMA-ES gains) and Outcome D (learned weights)** — rerun
   **only if** B1–B3 suggest the corrected body changes the picture (e.g. real
   now beats controls, or a tripod appears). These are the expensive rungs; do
   not spend hours on them if B1–B3 reproduce the same clean negatives. If you do
   rerun, keep the identical honesty rules from the archived Outcome C/D briefs
   (`docs/archive/`), and the shared fixed engine.
   **GATE B4:** post the 3-way tables + verdicts, or a one-line justification for
   skipping.

---

## 5. Phase C — reconcile and report
Update `PATH2_METHODS.md` and `HANDOFF.md`: add a "Body-setup correction" section
documenting the bug, the fix (from the `home` keyframe), Gate A evidence, and a
before/after table of every re-run verdict. Mark all pre-fix results as
**confounded/superseded**. Do not delete the old numbers — show old vs corrected
side by side. **STOP for user approval before any final headline claim.**

---

## 6. Guardrails, integrity, deliverables
- Checkpoint and commit after each gate with a message naming the phase + result.
- Before each commit: python-compile all `code/*.py`; JSON/CSV parse; confirm VNC
  and brain weights hash-equal to the frozen graph; confirm the stance came from
  `m.key("home")` not a tuned search; zero-learning scan (no optimizer / backprop
  / reward / loss / target-fit / distance-or-tripod term feeding any choice);
  confirm real/shuffled/random share one engine; `git diff --check`.
- Videos: render the corrected multiview for at least Path-3 real, and a plain
  engine-walk clip for Gate A. Verify readable MP4 (H264, 8 s).
- Deliverables: fixed `code/hexapod_gait.py` + `code/gait_engine.py`; refreshed
  `results/path2/*` metrics/tables/videos for each re-run gate; the
  `PATH2_METHODS.md` "Body-setup correction" section with old-vs-corrected
  verdicts; the `HANDOFF.md` update. No final claim wording until approved.

## 7. Hardware
CPU only. Engine calibration and Path-3/control-study re-runs are minutes each;
Outcome C/D (only if warranted) are the hours-scale rungs. No GPU/RunPod needed.

## 8. One-line summary
The hexapod was squatting at 7 cm and could not walk on any input; fixing the
stance to the model's own `home` pose (with correct left/right mirroring) makes it
stand and walk. Re-run every prior verdict on the standing body, keep the engine
shared and connectome-blind, and report honestly which negatives survive — the
old results are confounded until this re-run replaces them.
