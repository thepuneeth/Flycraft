# Untried zero-training methods — coordination, waveform, vision — honestly

**Read this whole file, then `HANDOFF.md`, `PATH2_METHODS.md`, and
`docs/archive/RUN_CONNECTOME_DRIVE.md` before writing any code.** You are continuing the
Flycraft project. Everything here is **zero-learning.** No phase adds an optimizer, reward,
loss, target-fit, or any quantity derived from distance/coherence/catch feeding any choice.
This run tests only the methods that are **genuinely untried** — several obvious ideas are
already answered and must **not** be re-run (see §0.2). Work phases in order. Every phase has
a GATE that must print PASS (or a declared NULL) before the next starts. **Most phases are
expected to come back NULL — that is the point, not a failure.**

---

## 0. Where we are, and what "success" means

### 0.1 The honest scoreboard (do not re-litigate)
Frozen FlyWire brain + MANC VNC driving a non-fly hexapod, zero learning. Job table:

| Job | Owner | Evidence |
|---|---|---|
| Steering | connectome ✓ | real ≫ shuffled (command-translation, not vision) |
| Step timing | connectome ✓ | real−shuffled +0.331, p=3.05e-5 |
| Rhythm / duty | connectome ✓ | M2 duty 0.495, 0 falls; VNC CPG literature |
| Waveform | partial | C direct-MN: forward +0.224 m, beats shuffle & random p=3.8e-6, but a creep not a walk |
| Coordination | **NOT owned** | M1 NULL; Phase B bulk coupling fell 20/20; E4 feedback ABSENT |
| Pose | body, not wiring | D non-discriminating (random stands too) |

**We are at ~35–40%.** The remaining jobs (coordination, full waveform) are the ones raw
wiring has not given up. **There is no zero-training path to 100%** — every walker in the
field (flybody, flyGNN, NeuroMechFly) trains, imitates, or hand-scripts the gait, and our own
reservoir/CMA-ES attempts were "moat NULL." **Do not chase 100%. Do not add a trained layer.**
A clean negative here finishes the boundary paper; a faked positive destroys it.

### 0.2 ALREADY ANSWERED — DO NOT RE-RUN (cite these, do not redo)
- **E3 structured descending steering** (`probe_structured_descending.py`, FlyWire
  DNg100/DNb08 → VNC): `E3-ABSENT` on the corrected body. Identified steering DNs are done.
- **E4 sensory-feedback loop** (`run_sensory_feedback.py`, 622 proprioceptors closed through
  MANC): `E4-ABSENT` on the corrected body (coordination scalar −0.210). The closed
  proprioceptive loop is done — do **not** rebuild it.
- **Rhythm core** (`IN17A001`, `INXXX466`, `IN16B036`) is **already** in the VNC drive
  (`vnc_stepper.py`: `rhythm_core` is always in the active set). Not a separable new experiment.
- **Phase B bulk coordination coupling** (154k cross edges): fell 20/20, NULL. Do not repeat
  with the bulk edge set.
If a phase below overlaps one of these, its job is to **cite the prior verdict**, not re-run it.

### 0.3 Pre-register every null in writing before you run it
State the exact number that counts as "connectome earned this job" and the number that counts
as NULL, **before** you see the result. Then report whichever happened.

---

## 1. Honesty spine
1. **Zero learning. Zero.** VNC weights + brain frozen and hash-equal to the committed graph.
2. **Fixed maps stay fixed.** `mn_to_joint.translate()` gains `(-0.55, 0.65, 0.65)` are the
   declared anatomical map. A *fuller antagonist map* (Phase WAVE-ANTAG) may add **new declared
   per-joint anatomical structure**, but no gain is ever tuned to make the body walk.
3. **Coupling weights ARE the connectome edges**, not tuned strengths. You may calibrate **one
   global coupling gain**, and only against clean-clock stability (connectome-blind), shared
   identically across conditions.
4. **One pipeline, shared across conditions.** real / shuffled / random differ only in the
   *source* (which edges / which weights). Same engine, same body, same everything else.
5. **Every phase gets three controls** (`real`, `shuffled`, `random`) with paired real−shuffled
   and real−random, bootstrap CIs, Holm-corrected p, and a rendered `real` video you eyeball.
6. **When in doubt, STOP and report. Never tune to rescue a result. A negative is the deliverable.**

---

## 2. Phase A — pre-flight (MANDATORY; reuse `code/preflight.py`; ~2 min)
Six-trap gate before anything else:

| # | Trap | Pass |
|---|---|---|
| T1 | Squat stance | static settle ≥ 0.12 m |
| T2 | Fall floor hides squat | `H_MIN = 0.08` everywhere |
| T3 | No L/R mirroring | stance == `m.key("home").qpos[7:25]` |
| T4 | Engine can't walk | clean-clock engine ≥ 1.0 m / 8 s, no fall |
| T5 | Broken duty | 0.35 ≤ clean-clock duty ≤ 0.6 |
| T6 | Weights changed | SHA-256(weight+index) == committed |

**GATE A:** print `GATE A: PASS (T1..T6 = numbers)` or `GATE A: FAIL <trap>` and stop. This is
the wall that stops another 10-hour waste.

---

## 3. Phase COORD-212 — coordination from the identified circuit only
**Prior-art check first:** confirm Phase B used the **bulk** edge set (166,088 raw / 154,365
signed rows). This phase is distinct: build the coupling from **only the 212 published
coordination cells' cross-edges.**

**How:**
1. From `audit_intersegmental.cross_edges`, keep only edges where **both** endpoints are tagged
   `published_coordination_cell` (the 212-cell set; `io["published_coordination_body_ids"]`).
   Reduce to the signed 6×6 inter-leg matrix `K_212`. **`K_212` is fixed anatomy.**
2. Reuse the exact Phase B coupling machinery (`last_three_jobs.py` Kuramoto term
   `g·Σ K[i,j]·sin(θj−θi)`), swapping only the edge source to the 212-cell subset.
3. **`g` = one global gain, calibrated only against clean-clock stability**, blind to distance,
   shared across conditions.

**Controls:** `K_212` from real 212-cell edges / shuffled (permute within the 212 mask) /
random (matched sparsity+magnitude).

**Pre-registered null (write before running):** the 212-cell coupling is still net-synchronizing
(row sums dominated by one sign) and falls / does not beat shuffled+random on coherence and
feet-planted → coordination does not transfer even from the identified circuit. **First, print
the sign composition of `K_212` (positive vs negative cell count, row sums).** If it is as
net-positive as the bulk `K` (Phase B was ~83% positive), you may predict the fall and confirm
with a single smoke episode before the full ladder — do not burn 20 episodes to reproduce a
foregone synchrony.

**GATE COORD-212:** real/shuffled/random table (path CI, coherence, feet-planted, falls) +
paired tests + `real` video + the `K_212` sign table. Commit `untried: COORD-212`.

---

## 4. Phase COORD-INHIB — coordination from inhibitory edges only
Tripod needs anti-phase; anti-phase comes from **inhibition.** Phase B collapsed signs into a
net that was ~83% positive (excitatory → synchrony). This phase isolates the inhibitory anatomy.

**How:**
1. From the same cross-edges, keep only edges whose presynaptic sign is **inhibitory**
   (`fetch_manc.nt_signs`: GABA/glutamate = −1; drop acetylcholine/unknown). Build signed 6×6
   `K_inh`. Fixed anatomy.
2. Same coupling machinery, same single global `g` calibrated clean-clock only.

**Controls:** `K_inh` real / shuffled / random.

**Pre-registered null:** inhibitory-only coupling does not produce a stable anti-phase (tripod)
limit cycle that beats shuffled+random → the connectome's inhibitory inter-leg structure is not
sufficient for coordination un-trained.

**GATE COORD-INHIB:** table + paired tests + `real` video + `K_inh` sign/row-sum print.
Commit `untried: COORD-INHIB`.

---

## 5. Phase WAVE-ANTAG — fuller antagonist MN→joint waveform
Phase C used `mn_to_joint.translate()` (coarse 3-gain `_pool_score`) and got a directed creep
(forward +0.224 m). This phase tests whether a **fuller declared anatomical map** — per-joint
flexor vs extensor antagonist balance (as in `step_decode.motor_pools`, trochanter
flexor sign>0 / extensor sign<0) — improves the waveform, **without tuning any gain.**

**How:**
1. Extend the MN→joint readout so each of the 18 joints is driven by its **signed
   flexor−extensor antagonist difference** from the real motor pools, using the existing
   declared per-joint sign convention. The overall scale stays the frozen anatomical constant —
   **no per-joint gain is fit to distance or uprightness.**
2. Drive all joints directly from the frozen VNC motor output each control step (no engine,
   no phase machine), exactly as Phase C, only the readout richer.

**Controls:** real / shuffled / random motor source through the identical fuller map.

**Pre-registered null:** the fuller antagonist map does not beat Phase C's forward distance
beyond CI overlap, or does not beat shuffled+random → the extra anatomical structure does not
add walkable waveform. (Report even a small honest gain as partial, as with Phase C.)

**GATE WAVE-ANTAG:** table vs Phase C baseline + paired tests + `real` video. State plainly
whether it beats Phase C. Commit `untried: WAVE-ANTAG`.

---

## 6. Phase VISION-EVAL — does the frozen optic lobe actually steer?
**Prior-art check first:** the visual path already exists (`run_hexapod.py retina()` →
`flyserver.py` optic/visual_projection injection → connectome step; visual-input index in
`build_io_index.py`). It has only ever been used *inside* the chase or *with* CMA-ES training.
This phase is the **clean zero-training controlled evaluation** — never done — not a rebuild.

**How:**
1. Present a prey at a known azimuth. Run the **frozen** forward pass: `retina(az,dist,loom)` →
   inject into left/right optic + visual_projection neurons → connectome rate step → read the
   steering descending output (DNg100/DNb08) → turn command → VNC → hexapod. **No fit anywhere.**
2. Sweep prey azimuth left/right across a fixed set of angles; measure whether the emitted turn
   command sign/magnitude tracks the true azimuth (steering accuracy), and whether the body
   turns toward the prey.

**Controls:** real optic-lobe wiring / shuffled visual→DN weights / random. Same retina encode,
same VNC, same body.

**Pre-registered null:** the frozen optic lobe's steering output does not correlate with true
prey azimuth better than shuffled/random wiring → un-fit vision does not steer, and the earlier
"steering" credit stays attributed to command-translation, not visual decision. (A positive
here would be a genuine new result: the *brain*, not just the VNC, does real work. Do not force
it — report the correlation and its CI honestly.)

**GATE VISION-EVAL:** azimuth-vs-turn correlation with CI for real/shuffled/random + paired
tests + one `real` video of a left and a right trial. Commit `untried: VISION-EVAL`.

---

## 7. Phase RECONCILE — re-score (NO final claim)
Update `PATH2_METHODS.md`:
- New section "Untried zero-training methods (COORD-212 / COORD-INHIB / WAVE-ANTAG / VISION-EVAL)"
  with the four tables and each pre-registered null next to its outcome.
- **Re-score the job table (§0.1) and the connectome %.** State plainly which of the four (if
  any) transferred and the new estimate. If none, say the honest ceiling is ~35–40% and this run
  — together with E3/E4 — establishes it as the boundary. Cross-reference: COORD-212 and
  COORD-INHIB vs the Phase B / M1 coordination negatives; VISION-EVAL vs the E3 steering-DN
  negative.

**STOP. Print `READY FOR CLAIM REVIEW` and wait.** The user writes/approves the claim.

---

## 8. Per-commit integrity gate (before EVERY commit)
1. `python -m compileall -q eon-fly-brain/code` clean.
2. New JSON/CSV parse.
3. VNC weight+index hash-equal to committed (T6).
4. `mn_to_joint` base gains still `(-0.55, 0.65, 0.65)`; any new antagonist structure is a
   declared sign convention, **not** a fit gain; coupling matrices built from real edges, not
   hand-set numbers; stance provenance intact.
5. **Zero-learning scan** on the diff:
   `optim|backward|loss|reward|fit(|train|\.grad|target_fit|argmax.*dist|rescale.*(path|dist|walk)`
   — any hit justified or removed.
6. Shared-pipeline assert: real/shuffled/random construct identical objects except source.
7. Fall floor 0.08 everywhere. `git diff --check` clean. Message names phase + result.

---

## 9. Anti-mistake / fail-fast contract
- **Every phase runs a <2-minute smoke test** (build source, print the K sign table or the
  azimuth correlation on a few points, eyeball) before the 20-episode ladder. If the smoke looks
  wrong, stop — never launch the long run "to see."
- **Never trust a metric without watching one `real` video.**
- **One thing at a time:** COORD-212 → COORD-INHIB → WAVE-ANTAG → VISION-EVAL. No stacked changes.
- **If a gate returns NULL, that is a result — record it and move on.** Do not loop to turn a
  null into a positive.
- **Do not re-run E3, E4, the rhythm core, or Phase B bulk coupling** — cite their verdicts.
- **No new dependencies.** Reuse `preflight.py`, `last_three_jobs.py`, `audit_intersegmental.py`,
  `fetch_manc.py`, `mn_to_joint.py`, `step_decode.py`, `run_hexapod.py`, `flyserver.py`,
  `build_io_index.py`, `probe_vnc_phase.py`, `vnc_stepper.py`.
- **Checkpoint + commit after every gate.** CPU only; each gate is minutes. Hours = a bug; stop
  and diagnose, do not wait it out.

## 10. One-line summary
On the corrected standing body, test the four genuinely-untried zero-training methods —
coordination from the 212 identified cells only (COORD-212), coordination from inhibitory edges
only (COORD-INHIB), a fuller antagonist MN→joint waveform (WAVE-ANTAG), and a clean controlled
evaluation of the existing frozen optic-lobe steering path (VISION-EVAL) — each with a
pre-registered null, each against shuffled and random, never tuned, fixed maps frozen. Cite
E3/E4/rhythm-core/Phase-B rather than re-running them. Re-score the connectome's honest share
(~35–40% now). Most are expected NULL; a clean negative is the paper's boundary result. No final claim.
