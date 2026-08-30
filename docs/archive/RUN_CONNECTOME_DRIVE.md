# Give the connectome more of the walk — connectome-set coordination + continuous phase drive

**Read this whole file, then `HANDOFF.md` and `PATH2_METHODS.md`, before writing any
code.** You are continuing the Flycraft project. The last run wasted ~10 hours
because the *body model was mis-set-up* (a squat stance the code could not stand on)
and nobody checked the body could stand before running the ladder. **That class of
mistake is now forbidden.** This brief is built so a setup error is caught in the
first two minutes, not after ten hours. Work the phases in order. Every phase has a
GATE that must print PASS before you spend a single minute on the next one.

---

## 0. What we are actually building, and the number we are chasing

The moat: a **frozen, untrained** fly connectome (FlyWire brain + MANC VNC) driving a
**non-fly hexapod**. Nobody else does this — everyone else trains a controller or
replays recorded fly motion. Our question is not "can we make it walk" (training
does that). It is **"how much of the walking does the raw wiring generate, with zero
learning?"**

Score the honest way — split the walk into the jobs that must happen:

| Job | Who does it NOW | Target after this run |
|---|---|---|
| Steering (turn toward goal) | **connectome (brain)** ✓ | connectome ✓ |
| Per-leg step timing (when a leg lifts) | **connectome (VNC decode)** ✓ | connectome ✓ |
| Inter-leg coordination (which legs move together) | **hardcoded `TRIPOD_A`** ✗ | **connectome (Method 1)** |
| Step rhythm / duty (how long swing vs stance) | **binary threshold** ✗ | **connectome (Method 2)** |
| Intra-step joint waveform | fixed gait primitive ✗ | fixed (out of scope) |
| Standing pose / ride height | model `home` keyframe ✗ | fixed (correct as-is) |

- **Where we are: ~25%** of the walk is connectome-driven (timing + steering only).
- **This run targets ~50–60%** by handing **coordination** and **rhythm** to the
  wiring (Methods 1 and 2 below).
- **Honest ceiling ~60–70%.** Getting the connectome to also generate the joint
  *waveform* with zero learning is the wall the entire field hits. **100% zero-
  training is almost certainly not reachable** — do not chase it, do not fake it. If
  you find yourself adding a trained layer to hit a higher number, STOP: that crosses
  into everyone else's territory and kills the novelty.

**The deliverable of this project is a measurement, not a walking robot.** A clean
number ("the raw wiring buys you X% of coordinated walking, and here is exactly where
it runs out") is the publishable result. A tuned/faked walk is worth nothing.

---

## 1. Honesty spine (unchanged — violating any line invalidates the run)

1. **Zero learning. Zero.** No optimizer, backprop, reward, loss, target-fit, or
   any quantity derived from distance/coherence/catch feeding any choice. The VNC
   weights and brain stay **frozen and hash-equal** to the committed graph.
2. **Connectome-blind engine.** The gait engine (waveform, PD gains, frequency
   mapping, lift/sweep amplitude) may be calibrated **only against a clean external
   tripod clock** — never against connectome performance or any control's score.
3. **The coordination and rhythm you extract must come from the connectome's own
   anatomy/activity, fixed a-priori** — the measured VNC phase structure, or its
   dominant oscillatory mode. **You may not pick a coordination pattern because it
   walks better.** Derive it, apply it, report whatever it gives — even if worse.
4. **One engine, shared across conditions.** Real / shuffled / random all run the
   identical calibrated engine. The only difference between conditions is the
   timing/coordination *source*. That is what makes any gap attributable to wiring.
5. **Every condition gets the three controls:** `real`, `shuffled` (same weights,
   permuted), `random` (matched duty/frequency, no structure). Report paired
   real−shuffled and real−random with CIs and Holm-corrected p.
6. **Declared choices are allowed; tuned choices are not.** The fore/aft direction
   asymmetry is a declared choice (pick the strong direction once, orient the goal to
   it, state it). Threshold values, pool definitions, band-pass ranges are declared
   a-priori and frozen before you look at any distance.
7. **When in doubt, STOP and report. Never tune to rescue a result.**

---

## 2. The mistake that cost 10 hours, and the six traps that must be checked EVERY run

Last time: the gait stance was `(0,0.6,-1.2)` on all six legs → body settled at
**z=0.073 m** (belly-dragging) and could not walk on *any* input, but the fall check
`H_MIN=0.02` was so low the squat passed as "upright." Ten hours of ladder ran on a
body that physically could not walk. **Root cause: no pre-flight check that the body
stands and walks before the experiment.**

**Trap checklist — Phase A verifies ALL SIX before any experiment runs:**

| # | Trap | Detection (Phase A prints the number) | Pass condition |
|---|---|---|---|
| T1 | Squat stance | static settle height holding stance, no gait | **≥ 0.12 m** |
| T2 | Fall floor too low hides a squat | recompute `H_MIN` as a *ride-height* floor | **H_MIN = 0.08**, not 0.02 |
| T3 | No L/R mirroring | stance comes from `home_stance()` (per-leg mirrored) | stance == `m.key("home")` slice |
| T4 | Engine can't actually walk | pure engine on clean tripod clock, 8 s | **≥ 1.0 m path, no fall** |
| T5 | Broken duty (legs lifted most of the time) | mean swing fraction of whatever drives the legs | **0.35 ≤ duty ≤ 0.6** |
| T6 | Weights silently changed | SHA-256 of VNC weight+index vs committed | **hash-equal** |

If **any** of T1–T6 fails, **STOP, print which one, fix only that, re-run Phase A.**
Do not touch Methods 1/2 until Phase A prints `GATE A: PASS (T1-T6)`.

---

## 3. Phase A — pre-flight (must PASS before anything else; ~2 minutes)

Write `code/preflight.py`. It must, using the **unchanged** `hexapod_gait.py`:

1. **T1/T3 — stand:** build `HexapodGait(stance=None)` (defaults to `home_stance()`,
   the per-leg mirrored pose). Hold `gait.targets(0,0,0)` with the PD servo for 1.5 s.
   Print `static_settle_height`. Assert ≥ 0.12. Assert the stance equals
   `m.key("home").qpos[7:25]` (provenance: it came from the asset, not a tuned search).
2. **T4/T5 — walk on a clean clock:** run the pure engine (`run(mode="walk",
   drive=0.8, secs=8)` or the `StepGaitEngine` under `clean_tripod_states`). Print
   `path_m`, `min_height_m`, `fell`, and the **mean swing duty** of the clean clock.
   Assert path ≥ 1.0, `fell` False (with the corrected H_MIN=0.08), 0.35 ≤ duty ≤ 0.6.
3. **T2 — fix the fall floor everywhere:** grep every runner for `H_MIN`. Set it to
   **0.08** (a real ride-height floor). Commit that as a correctness fix. A run that
   dips below 8 cm is dragging and must count as fallen.
4. **T6 — frozen graph:** SHA-256 `all_legs_weight_csr.pkl` and `all_legs_io_index.npz`;
   compare to the hashes recorded in `PATH2_METHODS.md` / the committed files. Assert equal.

**GATE A:** print one line `GATE A: PASS (T1=.. T2=.. T3=.. T4=.. T5=.. T6=..)` with the
numbers, or `GATE A: FAIL <trap>` and stop. Render a 4-second engine-walk clip and
eyeball it: the body stands tall and strides. **Do not proceed on FAIL.**

---

## 4. Phase B — Method 1: connectome-set inter-leg coordination

**Goal:** replace the hardcoded `TRIPOD_A = {0,3,4}` with the coordination pattern the
**VNC's own activity** implies. The engine still supplies the waveform; the connectome
now decides *which legs are in phase with which*.

**How (a-priori, connectome-blind to performance):**
1. Run the frozen all-leg VNC under tonic drive (same protocol as `probe_vnc_phase.py`:
   drive 100, gW 0.03, tau 0.02, dt 0.005, substeps 4, seed 1). You already have
   `phase_metrics()` — it returns `phase_difference_matrix_cycles` (6×6, the mean
   phase lag between every pair of legs) and `phase_concentration_matrix` (how reliable
   each lag is).
2. **Derive coordination from that matrix, by a fixed rule declared now:** two legs are
   "same group" if their measured phase difference wraps to |Δ| < 0.25 cycles, "anti"
   if within 0.25 of 0.5. Cluster the six legs into phase groups from the matrix (e.g.
   sign of `cos(2π·Δ)` vs leg 0, or a 2-cluster split on the phase angles). This yields
   a **connectome-derived grouping** to hand the engine in place of `TRIPOD_A`.
3. Feed that grouping into the engine's phase offset (the `(np.pi if leg in GROUP ...)`
   term). **Everything else identical.** The waveform, amplitudes, and PD gains are the
   Phase-A-calibrated ones, unchanged.

**Controls:** derive the grouping three ways from the **same pipeline** — `real` (true
weights), `shuffled` (permuted weights → its own implied grouping), `random` (random
grouping matched to the real group sizes). Same engine, same everything else.

**Report honestly:** the connectome may imply a clean tripod, a wave gait, or noise.
Whatever it implies, apply it and measure. If it implies no usable structure, that is a
finding ("the raw VNC phase relationships do not encode a walkable coordination
pattern"), not a failure to fix.

**GATE B:** table of real/shuffled/random — path (bootstrap CI), coherence, feet-planted,
paired real−shuffled and real−random (Holm p) — plus the derived grouping for each, plus
a video of `real`. Commit `bodycoord: method-1 connectome coordination`.

---

## 5. Phase C — Method 2: continuous phase drive (kills the 0.69-duty problem honestly)

**The current decode** (`step_decode.py`) is a binary `swing > stance` threshold per
leg. It produces a broken **0.686 swing duty** (legs airborne 69% of the time) → almost
no propulsion → the "moves little" video. **Do not tune the threshold to fix this** —
that is optimizing toward walking.

**Instead, read a *continuous* phase from the VNC and drive the engine phase directly:**
1. Project each leg's antagonist motor-pool activity (the same trochanter flexor/extensor
   pools `step_decode.py` already defines — reuse `motor_pools()`) onto its **dominant
   oscillatory mode**: band-pass the flexor−extensor difference in the VNC's rhythm band
   (declare it now: 1.5–4.0 Hz, the same band `probe_vnc_phase` uses), take the analytic
   signal, read the instantaneous phase per leg per timestep.
2. **Drive the engine's per-leg phase with that continuous VNC phase**, instead of the
   binary state machine in `gait_engine.py`. Swing/stance now falls out of *where the
   VNC oscillation is*, so the duty is whatever the connectome's own rhythm produces —
   **not a number you chose.** Report the resulting duty; if it is sane (~0.5) that is
   the connectome earning it, if it is still broken that is an honest limitation.
3. Combine with Method 1's grouping if Phase B produced a usable one (continuous phase
   already encodes coordination — Method 2 may subsume Method 1; note which).

**Controls:** real / shuffled / random continuous-phase sources, same engine.

**GATE C:** real/shuffled/random table (as B) **plus the measured swing duty for each
condition** and a video of `real`. Commit `bodycoord: method-2 continuous phase drive`.

---

## 6. Phase D — reconcile and score (NO final claim)

Update `PATH2_METHODS.md`:
- New section "Connectome coordination + continuous phase (Methods 1–2)" with the B/C
  tables next to the Path-3 baseline (binary decode, real−random −0.097).
- **Re-estimate the connectome share** using the honest job table in §0: state which
  jobs the connectome now owns and the rough % (expect ~50–60% if a method worked, and
  say so plainly if it did not move off ~25%).
- Mark clearly whether Method 1/2 **beat the Path-3 binary baseline**, and whether real
  still separates from shuffled (the surviving positive) and still ties/loses to random.

**STOP. Do not write a final headline claim.** Print `READY FOR CLAIM REVIEW` and wait
for the user. The user writes/approves the claim.

---

## 7. Per-commit integrity gate (run before EVERY commit, not just at the end)

1. `python -m compileall -q eon-fly-brain/code` → clean.
2. All new JSON/CSV parse.
3. VNC weight + index **hash-equal** to committed (T6).
4. Stance provenance: engine stance == `m.key("home")` slice (not a tuned array).
5. **Zero-learning scan:** grep the diff for `optim|backward|loss|reward|fit(|train|
   \.grad|target_fit|argmax.*dist` — any hit must be justified or removed.
6. Shared-engine check: real/shuffled/random construct the **same** engine object /
   parameters; assert in code.
7. Fall floor is 0.08 everywhere (T2).
8. `git diff --check` clean. Commit message names the phase + the result.

---

## 8. Anti-mistake rules (the "no more wasted time" contract)

- **Fail-fast, always.** Every phase runs a <2-minute smoke test (stand + few steps)
  before the full 8-second × 20-episode ladder. If the smoke test looks wrong, stop —
  never launch the long run "to see."
- **Never trust a metric without eyeballing one video.** `H_MIN` fooled us once; a
  number can pass while the body drags. Render `real` at every gate and look.
- **Change one thing at a time.** Method 1, then Method 2. Do not stack untested changes.
- **If a gate fails, fix only the failing thing.** No opportunistic edits mid-run.
- **No new dependencies, no new frameworks.** Reuse `hexapod_gait.py`, `gait_engine.py`,
  `probe_vnc_phase.py`, `step_decode.py`, `vnc_stepper.py`. New files only for the two
  methods + `preflight.py`.
- **Checkpoint + commit after every gate.** A crash must never lose more than one phase.
- **Hardware: CPU only.** Each gate is minutes. If something is taking hours, something
  is wrong — stop and diagnose, do not wait it out.

## 9. One-line summary
Verify the body stands and walks (Phase A, six-trap pre-flight) BEFORE anything else,
then hand the connectome two jobs it does not yet own — inter-leg coordination (Method 1)
and step rhythm (Method 2) — derived from its own frozen activity, never tuned to walk,
measured against shuffled and random. Report how far that pushes the connectome's honest
share of the walk (~25% now, ~50–60% target, ~60–70% ceiling). No final claim.
