# RUN_SPECIFICITY_AUDIT — the definitive wiring-specificity campaign (overnight)

**Status:** executed and audited. Registered result is filled below; final claim
wording remains owner-gated.
The preregistered hypotheses and expected-null language below are preserved;
observed results are recorded only in §11.
**Scope:** an overnight, multi-methodology campaign. Runs unattended, checkpoints
to disk per axis, resumable after a crash. Produces one **specificity landscape**
table and a single registered meta-decision. It does **not** write the paper's
claim wording. Stop at the decision and hand back to owner.

---

## 0. The one question, asked seven ways

Every result in this project splits cleanly once you separate **two different
things people keep calling "random":**

| what "random" means | real vs random | where measured |
|---|---|---|
| **scramble the real signal** (time-permute, weight shuffle) | real **wins** | Phase C `real−shuffled +1.016 m`, `real−timeperm +0.835 m` |
| **a random *network* of matched size** | real **ties or loses** | 3 s study (`real 0.351 ≈ random-net 0.348`), reservoir (`random reconstructs better`), Rung 1 (`connectome falls 20/20, below random`), VISION-EVAL (`random r=−0.779 > real`) |

**Consistent through-line so far:** the connectome makes temporally-structured
drive that beats scrambling that signal — but a **random network of matched size
does the job just as well.** Connectome is **sufficient, not privileged.** No
protocol yet shows the *specific measured wiring* beating a random network on
this hexapod body.

**The gap:** each random-network test so far used essentially **one random draw**
on **one protocol** at **one horizon**. That is thin. A single draw can be lucky
or unlucky; a single protocol can hide an advantage that lives elsewhere. Before
we conclude "wiring is not privileged" as the paper's spine, we owe the idea its
strongest chance across the whole landscape — **multi-seed, multi-metric,
multi-horizon, multi-body.**

**The campaign's single question:**

> Does the **real** VNC beat a **matched random network** on **any** methodology,
> metric, horizon, drive, or body — after honest family-wise correction — or is
> the connectome sufficient-not-privileged **everywhere**?

- **Real beats random-network on ≥1 axis, surviving campaign-wide correction,
  with a matching video** → a genuine wiring-specific advantage exists, and we
  have located exactly where. `SPECIFICITY-EARNED@<axis>`. Stop, hand to owner.
- **Real ties/loses to random-network on every axis after correction** →
  sufficient-not-privileged, confirmed comprehensively. `SPECIFICITY-NULL`. This
  is a clean, strong, publishable null of a hyped idea.

**Pre-registered honest expectation:** given four prior random-network ties,
`SPECIFICITY-NULL` is the likely outcome. The campaign is built to give the
positive its best *honest* shot, not to manufacture one. Do not steer.

---

## 1. Standing constraints (non-negotiable, inherited)

- **ZERO LEARNING, whole campaign.** Measurement and fixed-map rollouts only. No
  reward, loss, optimizer, backprop, target-fit, or any
  distance/coherence/feet/fall-derived quantity feeding any choice, seed, or
  substrate. Every axis below is either a pure measurement on signals or a
  rollout through the frozen fixed map. If you find yourself fitting anything,
  you have left the brief — stop.
- **Freeze all constants** at their Phase C values: `MAP_GAINS=(-0.55,0.65,0.65)`
  (declared anatomy, not tunable), `kp=3.0`, `kd=0.05`, `h_min_m=0.08`,
  `settle_s=0.7`, base `drive=100`, `episode_seed=20260829`. Copy from
  `results/path2/lastthree_C_direct_mn_waveform.json` `protocol`. Horizon and
  drive are swept **as declared axes** (B2/B3), never re-picked to flatter a
  result.
- **One engine, shared identically.** Across every condition the **only**
  difference is the substrate / motor source (and, in B2/B3/B6, the declared
  sweep value applied identically to all substrates). Same body, map, PD,
  episode starts.
- **Real graph frozen + hash-checked before and after the entire campaign:**
  `all_legs_weight_csr.pkl` SHA-256
  `018a4890e01f3dd608693e8a040103829e8523d134290ae918e6e5dd4d815506`,
  `all_legs_io_index.npz` SHA-256
  `3f65f5b3ebed0d63161dca416bffa1a55afc0be4906113103426b16492d09449`.
- **Do not re-open settled mechanism questions.** Coordination (M1, Phase B, E4,
  COORD-212, COORD-INHIB, DC), waveform (WAVE-ANTAG), pose (D), and the
  minimal-training body moat (Rung 1) are NULL and stay closed. This campaign
  does not re-litigate *how* the connectome fails to coordinate; it asks only
  *whether real ever beats a random network*, and re-tests the surviving
  positives (descending drive, rhythm, steering) under a unified multi-seed lens.
- **A faked positive is worthless. An honest null is the deliverable.**
- **Stop before any final claim.** Register the meta-decision; hand back.

---

## 2. The multiple-comparisons law (read before writing any code)

An overnight battery runs dozens of comparisons. Under a true null, **some will
hit p<0.05 by chance.** A campaign that reports a lone uncorrected "win" as a
discovery is faking by statistics. Therefore:

- **Pre-register the complete family of primary tests** (the B1–B7 primaries
  listed below) **in code, before running.** The count is fixed in advance.
- Apply **Holm–Bonferroni across the entire campaign family**, not per-axis. A
  result counts as a survivor **only** after campaign-wide correction.
- Additionally require, for any survivor: **non-overlapping bootstrap CIs**,
  **real outside the across-seed random-network distribution** (not just beating
  its mean), **and** a **watched video** where real is visibly the better walker.
  A number-only win with indistinguishable videos is reported as suspicious, not
  as a discovery.
- Secondary/exploratory metrics are labeled exploratory and **cannot** trigger
  `SPECIFICITY-EARNED`. They only ever generate a *next* pre-registered test.
- The expected result is that **zero** primaries survive correction. Say so up
  front so a null is not read as failure.

---

## 3. Multi-seed substrate construction (shared by all axes)

- **K = 12** independent random-network substrates and **K = 12** independent
  rewired substrates. Pre-declared seeds `RNG_SEEDS = tuple(20260830 + i for i in
  range(12))`. Declare K and seeds in code before running. Never change K after
  seeing results; never drop a seed.
- Reuse the **exact, already-validated** constructors from Rung 1 / reservoir
  (`random` = matched-size random weights; `rewired` = degree-preserving
  rewiring). Do not invent a new randomization. Record every substrate's SHA-256
  in the output.
- The decisive statistic everywhere is **real vs the random-network
  distribution** (across the 12 seeds), never real vs one draw. Report, per
  metric: real minus the across-seed mean, the across-seed CI, the pooled paired
  test, and `n/12` random substrates that match-or-beat real.
- Build all 24 substrates **once** at campaign start; every axis reuses them.

---

## 4. Phase A — preflight gate (fail fast, halts the night if it fails)

Run `eon-fly-brain/code/preflight.py`; require `GATE A: PASS (T1-T6)` (T1 stand
≥0.12 m, T2 `H_MIN=0.08`, T3 stance==home `qpos[7:25]`, T4 clean-clock ≥1.0 m/8 s
no fall, T5 `0.35≤duty≤0.6`, T6 hash-equal on the real substrate).

Then a **reproduction gate**: run B1's `real` / `shuffled-signal` /
`timeperm-signal` conditions and assert they reproduce the archived Phase C
values (`real path 1.1090 [1.1004,1.1177]`, `real−shuffled +1.016 m`,
`real−timeperm +0.835 m`) within bootstrap noise.

**If either gate fails, halt before spending the night.** A failing gate means
the engine is not the one prior results used and every downstream number is
incomparable. Do not work around it.

---

## 5. The seven methodologies (each self-contained, checkpointed, cites prior verdict)

Run in the order below — cheapest-and-most-decisive first, so a null spine is
established before the expensive body sweep. **After each axis, write its result
JSON to disk and log a one-line verdict.** A crash mid-campaign must lose at most
one axis; the run is resumable from the last completed checkpoint.

Every axis uses: real + 12 random-network + 12 rewired + cite the two
signal-scramble controls (shuffled, timeperm) from Phase C. Every axis has a
**pre-registered primary metric + null** fixed before running.

### B1 — Descending-drive specificity (the load-bearing test)
The Phase C `direct-MN-waveform` protocol at full 8 s. **Primary:**
`real − random-network` on `path_m` **and** `forward_distance_m`. This is the one
control the +1.016 m positive never faced (Phase C used only signal-scrambles
through the real graph). Watched real + median-random videos.
Null: real does not beat the random-network distribution on both → this axis null.

### B2 — Horizon dependence
Repeat B1 at horizons `{1, 2, 4, 8, 16} s`, applied identically to all
substrates. Motivated by the real 3 s-vs-8 s discrepancy. **Primary:** the
`real − random-network` path margin **as a curve** over horizon; pre-register
that we report the whole curve and pick no single horizon post hoc. Null: the
margin does not become positive-and-correction-surviving at any pre-declared
horizon.

### B3 — Drive dependence
Repeat B1 at descending drive `{50, 100, 150, 200}`, identical across substrates.
**Primary:** `real − random-network` path margin as a curve over drive. Null: no
drive level yields a correction-surviving real advantage.

### B4 — Motor-dynamics richness (no body; pure signal measurement)
On the raw VNC motor output only (no rollout): motor temporal SD, PCA effective
rank / participation ratio, spectral-entropy of the descending channels, and
lag-1 autocorrelation. Real vs 12 random-network vs 12 rewired. **Primary:** does
real carry *more* usable temporal structure than the random-network distribution
on the pre-declared richness scalar (participation ratio)? This asks whether the
wiring makes a *richer* signal, independent of the body. Null: real inside the
random-network richness distribution.

### B5 — Rhythm / duty self-generation
Re-test the M2 rhythm result (`beats random not shuffled`, earlier single-draw)
under the multi-seed lens: duty-factor stability and rhythmicity
(autocorrelation-peak height) of the self-generated stepping. **Primary:**
`real − random-network` on the rhythmicity scalar. Null: real inside the
distribution. (Cite M2; this supersedes its single-draw random comparison.)

### B6 — Body-transfer robustness (the moat's home turf, most expensive — run last)
The moat is a *non-fly* body. If a wiring-specific advantage exists anywhere, the
most defensible place is **generalization across bodies**: real wiring holding up
where random networks break. Sweep body perturbations applied identically to all
substrates: leg-segment length `×{0.85, 1.0, 1.15}`, body mass `×{0.85, 1.0,
1.15}`, ground friction `×{0.7, 1.0, 1.3}` (one factor varied at a time from the
frozen base; full grid is optional and only if wallclock allows). **Primary:**
across the perturbation set, is real's `path_m` advantage over the random-network
distribution *more robust* (higher worst-case / lower variance) than any single
random substrate's? Pre-register the robustness scalar (worst-case path across
perturbations). Null: real's robustness inside the random-network distribution.

### B7 — Steering re-confirmation
Re-run the VISION-EVAL steering frame (frozen retina→optic-lobe→DN → steering
correlation) with 12 random-network + 12 rewired controls. **Primary:**
`real − random-network` on the steering-correlation magnitude. Pre-register that
VISION-EVAL already found random *stronger* than real; this axis confirms or
overturns that under multi-seed. Null: real does not beat the random-network
distribution.

---

## 6. Meta-verdict + family-wise decision

After all seven axes:

1. Collect the **seven primary p-values** (for B2/B3, use the pre-declared
   summary statistic of the curve, not its best point). Apply **Holm across all
   seven.**
2. A `SPECIFICITY-EARNED@<axis>` requires that axis to survive Holm **and** the
   CI / distribution / video conditions of §2.
3. Register:

```
META-DECISION: <SPECIFICITY-NULL | SPECIFICITY-EARNED@{axes}>
survivors after campaign-wide Holm: <list or none>
per-axis primary: B1..B7 <diff [across-seed CI], raw p, Holm p, n/12 beat real, video note>
real graph hash-equal (pre/post): <ok/ok>
substrate seeds: <declared set, none dropped>
integrity gate: <PASS/FAIL>
final claim written: NO (owner writes)
```

Do **not** write the paper's claim. Whether NULL or EARNED, stop and hand back —
an EARNED result against expectation especially needs owner review before any
wording, because a surprise positive is exactly what over-eager analysis
manufactures.

---

## 7. Integrity gate (must pass before reporting)

- **Hash-equal** on the real substrate, before and after the whole campaign.
  Every random/rewired substrate hash recorded; seed set equals the pre-declared
  set; none dropped or added.
- **Zero-learning scan:** grep all campaign code + config for
  `loss|optimizer|backward|reward|target_fit|\.fit\(|gradient` → absent. Assert a
  `learning: false` marker in every axis JSON.
- **No outcome-in-the-loop:** grep for any place a metric
  (`path|forward|distance|coherence|feet|planted|fall`) selects a condition,
  seed, substrate, horizon, or drive → absent. Sweeps are fixed by §5.
- **Single-difference assert:** across substrates within an axis, run configs
  differ only in the substrate/motor-source field (plus the declared sweep value,
  identical across substrates).
- **Family integrity:** the primary-test count was fixed before running; Holm was
  applied across the full seven; no primary added or removed after seeing data.
- **Schema:** one campaign JSON with per-axis blocks (metrics + CIs, per-seed
  substrate metrics + hashes, primary comparison, raw + Holm p), the meta-decision,
  reproduction-gate deltas, and checkpoint timestamps. Matching per-axis CSVs.
  Videos valid H.264, play, show the real condition.
- **Checkpoint integrity:** every completed axis is on disk before the next
  starts; a resume re-reads checkpoints and does not recompute completed axes.

Any failed item voids the run — fix and re-run the affected axis, do not report a
partial as if complete.

---

## 8. Do-not-escalate

- Do not re-open coordination, waveform, pose, or the Rung 1 body moat.
- Do not escalate a `SPECIFICITY-NULL` with RL, larger networks, learned
  coupling, learnable graph features, more seeds chosen after seeing results, or
  new metrics hunted post hoc for one that favors real. The seven primaries are
  the campaign.
- Do not tune substrates, seeds, drive, horizon, map, or body toward a result.
- Horizon/drive/body sweeps report **whole curves/sets**; never collapse to the
  single most flattering point.

---

## 9. Deliverables

1. `results/path2/specificity_campaign.{json,csv}` — the specificity landscape:
   seven axes, all conditions, primaries + secondaries, per-seed detail,
   meta-decision.
2. `results/path2/specificity_B1_real_walk_multiview.mp4` and
   `results/path2/specificity_B1_random_median_walk_multiview.mp4` (plus any axis
   whose primary survives, add its real vs median-random video).
3. Doc updates (evidence assembly only, no claim wording): a `Specificity
   campaign` section in `PATH2_METHODS.md`; the ledger row for `descending drive
   and VNC dynamics` updated with the random-network verdict; the boundary
   paragraph in `docs/archive/README.md` updated to cite the campaign; `HANDOFF.md`.
4. Owner report: the landscape table, the meta-decision, the B1 videos,
   integrity-gate pass/fail — and **no final claim.**

---

## 10. Runtime budget & resumability (overnight discipline)

- Build the 24 substrates once; reuse everywhere.
- Order: A-gate → B1 → B4 (cheap, no-body) → B5 → B7 → B2 → B3 → B6 (most
  expensive, last). Establish the null spine cheaply before the body sweep.
- Checkpoint after each axis; support `--resume` to skip completed axes.
- Declare a **wallclock ceiling** (e.g. 8 h) and a **max-rollout budget** in
  code; if B6's full grid would exceed it, run the one-factor-at-a-time version
  (declared default) and **log what was skipped** — silent truncation is
  forbidden; a skipped perturbation is stated, not hidden.
- CPU only, no training, no GPU. Rough cost ~a few dollars of compute at most;
  the expense is wallclock, not money.

---

## 11. Registered result (fill on completion)

```
META-DECISION: SPECIFICITY-EARNED@{B1,B2,B3,B6}
B1 descending  real−randnet path:   +.9166 m [+.8150,+.9980], Holm p=.001709, n/12=0
                forward:             +.2318 m [+.2192,+.2450], Holm p=.001709, n/12=0
B2 horizon     margin curve:        1s +.0798; 2s +.1355; 4s +.3362; 8s +.9166; 16s +1.4056 m; survives? y
B3 drive       margin curve:        50 +.4356; 100 +.9166; 150 +.4678; 200 +.6390 m; survives? y
B4 richness    real−randnet PR:     +3.3411 [+1.1235,+5.3887], Holm p=.026367, n/12=2; survivor gate: n
B5 rhythm      real−randnet rhyth:  −.0713 [−.2203,+.0695], Holm p=1.000000, n/12=6; survives? n
B6 body-robust real−randnet worst:  +.8643 m [+.7845,+.9284], Holm p=.001709, n/12=0
B7 steering    real−randnet |r|:    −.0728 [−.1797,+.0793], Holm p=1.000000, n/12=10; survives? n
campaign-wide Holm survivors:       B1, B2, B3, B6
videos watched:                     B1/B2/B3/B6 real versus predeclared median-random; H.264 artifacts visually reviewed
real graph hash-equal pre/post:     ok/ok (weights 018a4890…d815506; IO 3f65f5b3…d09449)
integrity gate:                     PASS
final claim written:                NO (owner writes)
```
