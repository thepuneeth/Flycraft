# Delay-coupled inter-leg oscillators — the last honest zero-training test

**Read this whole file, then `HANDOFF.md`, `PATH2_METHODS.md`, and the coordination
sections of `RUN_UNTRIED_ZEROTRAIN.md` and `RUN_LAST_THREE_JOBS.md` before writing any
code.** You are continuing Flycraft. This run tests the **one genuinely-untested
zero-training lever** for coordination: **delayed** inter-leg phase coupling. It is the
final experiment on the zero-training side of the project. Its **expected result is NULL**
— and a clean null makes the boundary airtight. A faked positive destroys the paper. Work
the phases in order; every phase has a GATE that must print PASS (or a declared NULL)
before the next starts.

---

## 0. The question, the theory, and what "honest success" means here

The moat: a **frozen, untrained** fly connectome (FlyWire brain + MANC VNC) driving a
**non-fly hexapod**, zero learning. Coordination is currently **NULL across five
independent methods** (M1 static grouping, Phase B bulk coupling, E4 proprioceptive
feedback, COORD-212 identified circuit, COORD-INHIB inhibitory-only). All five used
**instantaneous** coupling `g·Σ_j K[i,j]·sin(θ_j(t) − θ_i(t))`. The mechanism of failure
is understood: the real inter-leg edges are net **positive/excitatory**, and instantaneous
positive coupling drives oscillators toward **synchrony** (zero phase difference) — all six
legs lift together, the body tips. Every edge-subset re-slice (identified cells, inhibitory
only) failed the same way.

**The untested physics.** In coupled-oscillator theory, adding a transmission **delay** τ
to the coupling shifts the stable phase-locked fixed point. With an appropriate delay,
*the same excitatory edges* can stabilize an **anti-phase** relationship instead of
synchrony (delayed Kuramoto; Schuster & Wagner 1989; Yeung & Strogatz 1999). Delay is a
real biophysical property of axonal conduction that the instantaneous model omitted. This
run adds it — and **only** it.

**What this run can and cannot claim (read carefully — this bounds the honesty).**
- The connectome datasets we use give edge **existence, sign, and weight**. They do **not**
  measure per-edge conduction **delay**. Therefore any delay is a **declared assumption**,
  not a connectome measurement.
- So even a positive result is **NOT** "the connectome measures coordination." It is at
  most: **"the connectome edge *structure*, run as delayed oscillators under a single
  declared conduction delay, generates coordination — and the *structure* is what does it,
  because real edges beat shuffled/random edges under the identical delay."**
- The **real-vs-shuffled-under-identical-(g,τ)** comparison is the entire honesty backstop.
  If real does not beat shuffled under the shared constants, the delay (not the wiring)
  would be doing any work, and the result is **NULL**.

**Honest success (pre-register before running):** real edges + the declared delay produce
a **stable anti-phase tripod that walks** — feet-planted comparable to the clean-clock
positive control, low falls, positive forward path — **and** beat *both* shuffled and
random on coherence and feet-planted with Holm-corrected `p < 0.05` and non-overlapping
bootstrap CIs, **and** the watched `real` video shows a genuine tripod walk (not a drag or
flail). Anything short of all of that is **NULL**.

**Honest null:** real edges + declared delay do not generate a walking anti-phase tripod
beating the controls → coordination does not transfer even as delayed coupling → the
zero-training ceiling is airtight at **~35–40%** (steering + step timing + rhythm/duty).

---

## 1. Honesty spine (carry forward verbatim — plus delay-specific rules)

1. **Zero learning. Zero.** No optimizer, backprop, reward, loss, target-fit, or any
   quantity derived from distance / coherence / feet-planted / tripod-ness feeding any
   choice. VNC weights + brain **frozen and hash-equal** to the committed graph (T6).
2. **The coupling weights ARE the connectome edges.** `K` is built exactly as in
   `last_three_jobs.build_coupling_matrices` (line 104) — signed real edges from
   `audit_intersegmental.cross_edges` + `fetch_manc.nt_signs`, row-normalized anatomy only.
   Do **not** hand-set or re-tune any `K` entry.
3. **The fixed maps stay fixed.** `mn_to_joint.translate()` gains stay `(-0.55, 0.65, 0.65)`
   (`MAP_GAINS`, line 79). The engine waveform, stance, H_MIN=0.08, VNC protocol — all
   unchanged from the last-three-jobs Phase B deployment.
4. **The delay is a declared constant, calibrated connectome-BLIND, shared across
   conditions.** See §4. It is selected only against a neutral convergence criterion, never
   against tripod-ness or body outcome. Two free constants now exist — gain `g` and the
   delay constant — and **both** are pre-registered and identical for real/shuffled/random.
5. **One engine / one pipeline, shared across conditions.** real / shuffled / random differ
   only in the *source matrix*. Same `g`, same delay, same everything else.
6. **Every condition gets the three controls** (`real`, `shuffled`, `random`) with paired
   real−shuffled and real−random, bootstrap CIs (`BOOTSTRAP_REPS=100_000`,
   `BOOTSTRAP_SEED=20260829`), Holm-corrected `p` — and a rendered `real` video you eyeball.
7. **When in doubt, STOP and report. Never tune to rescue a result. A negative is the
   deliverable.** If no connectome-blind (g, delay) regime exists, that is itself a NULL —
   report it; do not relax thresholds to manufacture convergence.

---

## 2. Phase A — pre-flight (MANDATORY; reuse `code/preflight.py`; ~2 min)

Re-run the six-trap gate **before touching anything**. Do not modify `preflight.py`.

| # | Trap | Pass |
|---|---|---|
| T1 | Squat stance | static settle ≥ 0.12 m |
| T2 | Fall floor hides squat | `H_MIN = 0.08` everywhere |
| T3 | No L/R mirroring | stance == `m.key("home").qpos[7:25]` |
| T4 | Engine can't walk | clean-clock engine ≥ 1.0 m / 8 s, no fall |
| T5 | Broken duty | 0.35 ≤ clean-clock duty ≤ 0.6 |
| T6 | Weights changed | SHA-256(weight+index) == committed |

**GATE A:** print `GATE A: PASS (T1..T6 = numbers)` or `GATE A: FAIL <trap>` and stop.
Do not proceed on FAIL.

---

## 3. Phase DC-1 — implement delayed coupling (surgical; unit-checked)

Add **one** new function next to `coupled_phases` (`last_three_jobs.py:159`). Do not edit
`coupled_phases` itself — it stays as the instantaneous reference for the regression check.

**Anatomical delay matrix (primary, cleanest — delay structure from anatomy):**
- Segment index of a leg `i` is `seg(i) = i // 2` (T1→0, T2→1, T3→2), matching
  `LEG_NAMES = (T1L,T1R,T2L,T2R,T3L,T3R)`.
- Per-pair delay in control steps: `D[i,j] = round(per_hop_steps * abs(seg(i) - seg(j)))`.
  Same-segment left/right pairs get delay 0; adjacent segments 1 hop; T1↔T3 2 hops.
- `per_hop_steps` is the **single declared constant** (an integer number of 50 Hz control
  steps), selected in §4. This is the only new scalar. The *structure* of the delay comes
  from segment distance — anatomy, not tuning.

**Delayed coupling update** (discrete delayed Kuramoto; `θ_j(t−τ) − θ_i(t)`):

```
def delayed_coupled_phases(source, coupling, gain, delay_steps):
    # delay_steps: 6x6 int matrix (anatomical) OR a scalar broadcast to 6x6
    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 6 or len(source) < 2:
        raise ValueError(...)
    D = np.broadcast_to(delay_steps, (6, 6)).astype(int) if np.ndim(delay_steps) == 0 \
        else np.asarray(delay_steps, dtype=int)
    result = np.empty_like(source)
    result[0] = source[0]
    for step in range(1, len(source)):
        previous = result[step - 1]                      # theta_i(t), and self-drive base
        # neighbor phase delayed per pair; clamp to result[0] during warmup (declared)
        idx = np.maximum(0, (step - 1) - D)              # (6,6) source-time index per (i,j)
        neighbor = result[idx, np.arange(6)[None, :]]    # neighbor[i,j] = theta_j(t-1-D[i,j])
        phase_difference = neighbor - previous[:, None]  # theta_j(t-tau) - theta_i(t)
        coupling_term = np.sum(coupling * np.sin(phase_difference), axis=1)
        result[step] = previous + source[step] - source[step - 1] + gain * coupling_term
    if not np.isfinite(result).all():
        raise AssertionError("delayed coupled phase contains non-finite values")
    return result
```

**DC-1 self-checks (must all pass before DC-2):**
- **Regression:** `delayed_coupled_phases(src, K, g, delay_steps=0)` is element-wise
  `allclose` to `coupled_phases(src, K, g)` for a random `src` and the real `K`. If not,
  the indexing is wrong — fix before proceeding. (This proves delay=0 reproduces the
  committed instantaneous behavior exactly.)
- **Warmup boundedness:** with `per_hop_steps` up to the max candidate, output is finite and
  phase steps stay bounded on the neutral clock.
- **Indexing sanity:** print `D` for `per_hop_steps=1` and confirm it equals
  `[[0,0,1,1,2,2],[0,0,1,1,2,2],[1,1,0,0,1,1],...]` structure (|seg_i−seg_j|).

Commit `delay: DC-1 delayed coupling + regression self-check`.

---

## 4. Phase DC-2 — connectome-blind (g, delay) calibration (pre-registered, generation test)

**This is the honesty-critical phase.** The existing `calibrate_gain` (line 204) starts the
system from `clean_clock()` — the ideal anti-phase tripod — and only checks coupling does
not *disrupt* it. That is a **preservation** test and cannot be reused here, because it
starts from the answer. Replace it, for this run only, with a **generation** test that
starts from **random** phases and never looks at tripod-ness.

**Procedure:**
1. Build the neutral drive: the same clock **frequency** as `clean_clock` (line 176), but
   **do not** add the tripod offsets. Instead initialize each leg's phase from a **random**
   offset (fixed RNG, declared seed, e.g. `20260830`). This is a non-tripod start.
2. Candidate grid: `gain ∈ GAIN_CANDIDATES` (line 56, i.e. 0.00–0.20 step 0.01) ×
   `per_hop_steps ∈ {0,1,2,...,K_MAX}` where `K_MAX` corresponds to ≈ one half-period at the
   clock frequency (declare the number; a half-period is the physically-motivated ceiling
   for an anti-phase-inducing delay — state it, do not search beyond it).
3. For each (gain, per_hop_steps), run `delayed_coupled_phases` on the random-start neutral
   drive for `BODY_STEPS`, for **all three** conditions (real, shuffled, random `K`).
4. **Connectome-blind convergence criterion** (never references anti-phase or body):
   - **Phase-locked:** over the last third of the run, the variance of every pair's phase
     difference is below a declared threshold (the system settled to *a* limit cycle — any
     one, tripod or synchrony or other).
   - **Bounded:** `max |phase step| ≤ CLEAN_MAX_PHASE_STEP_RAD` (1.0) and
     `max |step change| ≤ CLEAN_MAX_STEP_CHANGE_RAD` (0.5), reusing the existing thresholds.
   - A (gain, per_hop_steps) is **admissible** only if **all three** conditions converge
     and stay bounded. Requiring all three prevents cherry-picking a constant that suits
     real. Record the full admissibility table.
5. **Declared selection rule (pre-register the exact words before running):** among
   admissible cells, select the one with the **smallest `per_hop_steps`**, breaking ties by
   the **largest `gain`**. Rationale: smallest delay = least assumption injected; largest
   stable gain = strongest expression of the (blind) coupling. **Selection never inspects
   coherence, anti-phase structure, feet-planted, or distance.** If you are tempted to pick
   by "which one looks like a tripod," STOP — that is the fake.
6. If **no** cell is admissible for all three conditions → print
   `DC-2 NULL: no connectome-blind delayed regime`, record the table, and **stop the run**.
   That is a legitimate negative (no honest delayed regime exists). Do not relax thresholds.

**GATE DC-2:** print the selected `(gain, per_hop_steps)`, the admissibility table, and the
pre-registered selection rule text. Commit `delay: DC-2 blind (g,tau) calibration`.

---

## 5. Phase DC-3 — generation run: real / shuffled / random, decisive control

Deploy exactly the last-three-jobs Phase B body pipeline, changing **only** the phase
generator: replace the instantaneous `coupled_phases` call with
`delayed_coupled_phases(source, K, g_selected, D_selected)`. The phase **source** per leg is
the VNC's own continuous phase (`vnc_phase_source`, `continuous_phase.py:63`) exactly as
Phase B used it — nothing about the source changes. Same engine waveform fed by the coupled
phases, same PD servo, same `MAP_GAINS`, same H_MIN, same episode count as Phase B.

**Conditions:** real / shuffled / random `K` (from `build_coupling_matrices`), identical
`g_selected` and `D_selected` for all three.

**Metrics (reuse Phase B machinery verbatim):** path m + 95% CI, forward m + CI, tripod
coherence scalar + CI, feet-planted + CI, falls (Wilson 95% CI), per-run `phase_metrics`
tripod verdict. Paired **real−shuffled** and **real−random** with exact + Holm `p` and
bootstrap CIs. Render and **watch** `results/path2/delay_coupling_real_multiview.mp4`.

**Pre-registered null (write into the results file BEFORE running):** real delayed coupling
does not produce a walking anti-phase tripod beating shuffled/random on coherence and
feet-planted → `DELAY-COUPLING-NULL`; the zero-training coordination ceiling holds.

**Pre-registered positive (write it too):** the full conjunction in §0 "Honest success" —
walking tripod, low falls, beats both controls Holm `p<0.05`, non-overlapping CIs, clean
video. Only that earns the narrowed claim "connectome edge structure + declared delay
generates coordination."

**GATE DC-3:** print the real/shuffled/random table + paired tests + the watched-video
verdict + the explicit `DELAY-COUPLING-NULL` or `DELAY-COUPLING-EARNED` line. Commit
`delay: DC-3 generation run real/shuffled/random`.

---

## 6. Phase DC-E — reconcile and score (NO final claim)

Update `PATH2_METHODS.md`:
- New section "Delayed inter-leg coupling (DC-1..DC-3)" with the DC-2 calibration table, the
  DC-3 result table, and both pre-registered nulls next to the outcomes.
- State plainly whether coordination is now a **sixth** null or the first earned method.
  If null: coordination is NULL across **six** independent zero-training methods; the
  ~35–40% ceiling is established and airtight; this closes the zero-training program.
- If earned: state the **narrowed** claim honestly (edge structure + declared conduction
  delay; delay is an assumption, not a connectome measurement; real beats shuffled/random
  under identical constants is the evidence) and flag it for user claim review.
- Update `docs/archive/README.md` to mark `RUN_DELAY_COUPLING.md` executed with the verdict.

**STOP. Print `READY FOR CLAIM REVIEW` and wait.** The user writes/approves the claim.

---

## 7. Per-commit integrity gate (before EVERY commit)

1. `python -m compileall -q eon-fly-brain/code` clean.
2. New JSON/CSV parse.
3. VNC weight+index hash-equal to committed (T6).
4. `mn_to_joint` gains still `(-0.55, 0.65, 0.65)`; `K` built from real edges via
   `build_coupling_matrices` (unmodified); delay from segment distance, not hand-set; the
   single `per_hop_steps` constant traces to the pre-registered §4 selection rule.
5. **Zero-learning scan** on the diff:
   `optim|backward|loss|reward|fit(|train|\.grad|target_fit|argmax.*dist|rescale.*(path|dist|walk)`
   — any hit justified or removed.
6. **Delay-honesty assert:** the (g, per_hop_steps) selection code references only the
   neutral convergence criterion — grep the selection function for any use of `coherence`,
   `tripod`, `feet`, `planted`, `path`, `forward`, `distance`; if present, it is faking —
   remove it.
7. Shared-pipeline assert: real/shuffled/random construct identical objects except the
   source matrix; identical `g` and `D`.
8. Fall floor 0.08 everywhere. `git diff --check` clean. Message names phase + result.

---

## 8. DO-NOT-RERUN (cite the committed verdicts; do not re-execute)

- **E3** structured descending steering → `E3-ABSENT`.
- **E4** sensory-feedback loop → `E4-ABSENT` (coordination scalar −0.210).
- **rhythm-core** cells already embedded in `vnc_stepper.py` drive set.
- **Phase B** bulk instantaneous coupling → falls 20/20.
- **COORD-212** identified-circuit instantaneous coupling → `COORD-212-NULL`.
- **COORD-INHIB** inhibitory-only instantaneous coupling → `COORD-INHIB-NULL`.
- **Reservoir** linear + random-feature readout → `READOUT TOO WEAK` (random ≥ connectome).
- **CMA-ES / Outcome D** → training; out of scope for this zero-training run.

This run's **only** new physics is the transmission **delay**. If you find yourself
re-touching any of the above, stop — you have drifted.

---

## 9. Anti-mistake / fail-fast contract

- **DC-1 regression check first.** If delay=0 does not reproduce `coupled_phases`, nothing
  downstream is trustworthy — fix the indexing before anything else.
- **Every phase runs a <2-minute smoke test** (few steps, eyeball) before any long run.
- **Never trust a metric without watching one `real` video.** A coherence number can pass
  while the body drags (Method 1 "flew" while pathing 1.5 m).
- **One thing at a time:** DC-1, then DC-2, then DC-3. No stacked untested changes.
- **A NULL is a result — record it and stop.** Do not loop over delays/gains hunting for a
  positive; the grid and selection rule are fixed in §4 and pre-registered.
- **No new dependencies.** Reuse `last_three_jobs.py`, `continuous_phase.py`,
  `probe_vnc_phase.py`, `audit_intersegmental.py`, `fetch_manc.py`, `mn_to_joint.py`,
  `vnc_stepper.py`, `preflight.py`.
- **Checkpoint + commit after every gate.** CPU only; each gate is minutes. Hours = a bug;
  stop and diagnose.

## 10. One-line summary
Add the one physics the zero-training coordination tests never had — a transmission
**delay** on the inter-leg coupling — as a segment-distance-derived, connectome-blind,
pre-registered constant; test whether the real edges (vs shuffled/random under the identical
delay) now **generate** a walking anti-phase tripod from the VNC's own phases. Expected
NULL (a sixth confirmation, closing the zero-training program at ~35–40%); a rigorous
positive would earn only the narrowed "edge structure + declared delay" claim. No final claim.
