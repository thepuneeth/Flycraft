# Camera-ready plan — IEEE BIBM 2026 UGHS (S59306)

**Accepted** (R1 and R2 both 4/5 Accept). **Deadline 2026-10-25.**
**Page limit: 5 pages including figures, tables and references** (confirmed on
the symposium site 2026-09-05, `paper/project_context.md`). The submitted PDF
is already exactly 5 pages, so every addition needs an equal cut.

Replanned 2026-09-29 after reading the code paths behind every paper number and
running a local pilot (scratchpad only, nothing written to `results/`).

---

## 0. What the paper actually rests on

| Paper section | Result file | Neural realizations sampled |
|---|---|---|
| A: lesions | `h5_lesion_localization_lif.json` | **1** (noise seed 1, CUDA) |
| B: rewiring dose | `h4_dose_response_lif.json` | 1 per control graph |
| C: body variations | `e3_body_generalization_amplitude_matched_lif.json` | **1** for the measured graph |
| D: named inputs | `labeled_line_*_lif_v3.json` | 30 |
| E: CPG walking | `cpg_specificity_variants.json` | across-realization SD reported |

How A–C run: `collect_history()` simulates the LIF network **once** for 8 s,
maps it to joint targets, then replays that *same* target signal into the body
from 20 different start yaws. The 20 "episodes" vary only the body's start
orientation. `vnc_lif_core.py:427` documents that the Poisson seed *"was not
forwarded, which pinned every stepper to generator seed 1 and made the noise
invisible."* A `noise_seed` argument now exists but A–C were never rerun with it.

The embodiment/interface-training track (EMBODY V3–V5, Rung 1, reservoir) is
out of scope for the camera-ready. The paper's premise is "no training";
importing any of it would contradict the framing and cost pages.

---

## 1. Pilot results (2026-09-29, local CPU, not paper results)

### 1a. Zero-drive / null arms (R2-1), 20 shared starts, one CPU realization

| Arm | Forward drive (m), mean over 20 starts |
|---|---|
| **Zero drive** (hold home stance, no network) | **−0.00327** (SD 1.3e-4) |
| Hold real signal's temporal mean (static posture) | −0.00102 |
| Real signal, time-shuffled | −0.00044 |
| Real signal, CPU seed 1 | +0.00044 |

**The body drifts backward on its own.** Standing still settles it about 3.3 mm
backward over 8 s. The controls in the paper (−0.003 to −0.005 m) sit at or just
below this standing baseline. So the right reference is not zero but −0.0033 m,
and against that reference the measured graph's advantage is larger than the
paper states, not smaller. R2's worry is answered in the paper's favour.

It also shows *where* the effect comes from, for this one realization:
about 2.3 mm of the gain over standing comes from the static posture the real
wiring holds (mean-hold arm) and about 1.5 mm from its time-varying output, of
which temporal order contributes roughly 0.9 mm. "Forward drive" is partly a
posture effect. That is still a wiring effect, but the text should say so.

### 1b. Poisson realization spread (R1-3), real vs T2 lesion, 10 CPU seeds

| | mean | SD across realizations | range | sign |
|---|---|---|---|---|
| Real forward drive (m) | +0.00057 | 0.00111 | −0.00117 to +0.00229 | **3 of 10 negative** |
| T2 lesion loss (m) | 0.00771 | 0.00475 | 0.00158 to 0.01863 | **10 of 10 positive** |
| Real minus zero drive (m) | ≥ +0.00210 in every realization | | | 10 of 10 positive |

Three consequences:

1. **The T2 lesion finding survives.** Loss is positive in every realization
   (sign-flip over 10 realizations, p = 1/1024). But its size is 0.0077 m on
   average, not 0.0112 m, and the paper's CI `[0.01100, 0.01140]` is roughly ten
   times too narrow because it only sees start-yaw variance.
2. **"Measured wiring produces positive forward drive" does not survive as
   worded.** The paper's +0.00208 m is one high draw; the realization mean is
   +0.00057 m and 3 of 10 draws are negative. The claim that does survive is
   "measured wiring moves the body forward relative to standing still, in every
   realization."
3. **The paper's headline number is not reproducible on CPU.** Same code, same
   seed, CPU gives +0.00044 m where the CUDA run gave +0.00208 m, because the
   CPU and CUDA random streams differ. Averaging over realizations removes this
   problem; reporting one draw cannot.

Claims in B and C that compare the measured graph's single value to control CIs
will need rechecking. In particular B says the measured mean is above the upper
CI at every nonzero fraction; α = 0.25 has an upper limit of +0.00188 m, above
the realization mean of +0.00057 m. That sentence will likely have to change.

---

## 1c. Full rerun results (2026-09-29) — PENDING OWNER REVIEW

Source: `eon-fly-brain/results/path2/camera_ready/camera_ready_reruns.json`
(script `eon-fly-brain/code/camera_ready_reruns.py`). LIF core on CPU, 30 Poisson
realizations × 20 start yaws, 30,140 rollouts, 0 falls, 35 min wall time.
Frozen-graph hash `128b5c…` unchanged before and after, matching the paper's run.
Videos (`cr_real_seed1`, `cr_lesion_T2_seed1`, `cr_zero_drive`) not yet reviewed.
p-values are Monte Carlo (200k draws), so ~5e-6 is the resolution floor.

**Null arms (R2-1).** Standing still drifts −0.00327 m. Measured graph
+0.00085 m [0.00051, 0.00118], positive in 24 of 30 realizations; **measured minus
zero drive +0.00412 m, positive in 30 of 30** (p ≈ 5e-6). Decomposition over
standing still: static posture +0.0023 m, time-varying output +0.0018 m,
**temporal order +0.00013 m (time-shuffled vs real p = 0.29, not detectable)**.
The effect lives in which channels are active and how much, not in the timing
of the output.

**Lesions (R1-2), loss = real − lesion, realization-level:**

| Lesion | edges removed | loss (m), 95% CI | positive | p_holm | realizations with a frozen channel |
|---|---|---|---|---|---|
| T1 | 102,353 | +0.00259 [0.00200, 0.00319] | 28/30 | 2e-5 | 11/30 |
| **T2** | 134,858 | **+0.00685 [0.00537, 0.00862]** | **30/30** | 2e-5 | 0/30 |
| T3 | 111,131 | **−0.00131** [−0.00195, −0.00066] | 10/30 | 1.0 | 24/30 |
| sensory | 26,187 | 0 (< 1e-10) | — | 1.0 | 0/30 |

Size does not explain the ordering: T3 is larger than T1 yet *raises* forward
drive. T2 > T1 > 0 > T3 is a segment-specific pattern. Caveat: T1/T3 arms
carry frozen joint channels in some realizations.

**Uncertainty (R1-3).** Measured graph: episode SD 0.00135 m within a
realization, realization SD of means 0.00096 m. T2 loss: episode SD 0.00085 m,
**realization SD 0.00462 m — five times larger**. The paper's CI
`[0.01100, 0.01140]` reflected episode variance only; the realization-level CI
is `[0.00537, 0.00862]`.

**Rewiring dose (B).** Spearman ρ = −0.771 (paper −0.886). Measured mean is
above the control upper CI at α = 0.1, 0.5, 0.75, 1.0 but **not at α = 0.25**
(upper +0.00199). Permutation p_holm < 0.05 at every α (0.045 at α = 0.25).

**Body variations (C), 7 unique bodies.** Measured beats shuffled weights at all
7 (p_holm ≤ 0.012) and beats zero drive in every realization at every body.
But **measured forward drive is not positive at all bodies**: leg ×1.15
−0.00043, mass ×1.15 −0.00052, friction ×1.30 +0.00004 (CI spans 0). And 0–3 of
12 control graphs reach the measured mean per body, so "none of the 12 control
seeds matches or exceeds it" no longer holds.

**Map sweep (R1-1, R2-2), pre-registered, 40 maps.** Fraction where measured
beats shuffled weights:

| Family | maps | measured > shuffled | measured > zero drive |
|---|---|---|---|
| gain jitter (±10/25/50%) | 12 | **12/12** | 12/12 |
| sign flips (10/25/50%) | 12 | 10/12 | 9/12 |
| within-leg joint permutation | 8 | **2/8** | 6/8 |
| cross-leg permutation | 8 | **3/8** | 3/8 |

The result is robust to the hand-chosen gains and mostly to sign noise, but it
does not survive scrambling which motor pool drives which joint. So the effect
depends on the anatomical pool-to-joint assignment, not only on the graph's
output statistics. That is a sharper, more honest version of the paper's
"this map" caveat.

### Claims that change (owner decision)
- Keep: T2 lesion effect (smaller, all 30 positive); measured > shuffled at
  every body; measured > standing still everywhere; dose trend negative.
- Weaken: "positive forward drive" → "forward relative to standing still";
  T2 loss 0.01120 → 0.00685 m with realization CI; ρ −0.886 → −0.771.
- Drop or rewrite: "above the upper confidence limit at every nonzero fraction";
  "positive forward drive in all nine tests"; "none of the 12 control seeds
  matches or exceeds it".
- New: T1/T3 comparison; posture vs timing decomposition; map-sweep fractions.

## 2. The reviewer asks, and how each gets answered

### R2-3 — abstract must state the failed antagonist-phase criterion (writing)
One sentence: the pre-registered antagonist-phase criterion was not met
(4.00 of 20 cells against a 12-cell threshold). Cost: minutes.

### R2-1 — zero-drive / null baseline (experiment, cheap)
Add three null arms to A–C, averaged over realizations where they have noise:
zero drive (stance hold), static mean-posture hold, time-shuffled real signal.
Report every result relative to zero drive as well as in absolute metres.
Zero drive has no neural noise, so it is one deterministic arm per body setting
(Experiment C needs it at each of the 7 unique bodies, since settling drift
depends on the body).

### R1-3 — separate simulator stochasticity from control randomization (rerun A–C)
There are three sources of variation, and the paper currently samples only one
of them for the measured graph:

| Level | What varies | Currently sampled in A–C |
|---|---|---|
| Episode | body start yaw | yes (20) |
| Realization | Poisson input noise to the network | **no** (1) |
| Control graph | rewire / shuffle randomization | yes (12, B and C) |

Fix: rerun A–C with the measured graph at N = 30 realizations (matching D's 30),
pair each control graph with its own realization seed, and report variance
components per level. Base all tests on realization-level means, so the unit of
inference is a network realization, not a start yaw. State plainly what each
p-value covers.

### R1-2 — T1 and T3 lesions (experiment, cheap)
These were dropped because Experiment A used strict all-channel amplitude
matching and each lesion freezes one of the 18 joint-target channels. But the
paper already uses the relaxed rule elsewhere:
`audit_h4_dose_response.py:227` (Experiment B) and
`labeled_line_campaign.py:1110,1413` (Experiment D) both call
`amplitude_match(..., allow_unmatchable=True)`, which leaves a frozen channel
alone and records the count. Experiment A was the odd one out.

Fix: apply Experiment B's rule to T1, T2 and T3 alike, report the frozen-channel
count per lesion (T1: 1, T2: 0, T3: 1), and run all three across the same 30
realizations. With lesion sizes 102,353 / 134,858 / 111,131 edges, the result is
a three-point size-versus-loss comparison instead of a one-lesion disclaimer.

### R1-1 + R2-2 — map robustness sweep (both reviewers; the main new experiment)
The map is `mn_to_joint.translate()`: three scalar gains
`MAP_GAINS = (-0.55, 0.65, 0.65)` plus three label arrays per motor channel
(`motor_leg`, `motor_joint`, `motor_sign`). Every perturbation is a change to
those inputs; no new map code is needed.

Pre-register before running, in increasing severity:
1. Gain jitter: each gain × (1 + ε), ε ∈ {±10%, ±25%, ±50%}.
2. Sign flips: flip 10% / 25% / 50% of `motor_sign`.
3. Within-leg joint permutation of `motor_joint`.
4. Cross-leg permutation of `motor_leg` (destroys the anatomical assignment).

Endpoint: for each family, the fraction of maps where
measured-minus-shuffled forward drive stays positive, reported including
failures. Families 1–2 test "is the result fragile to the hand-chosen numbers";
3–4 test "does the anatomical assignment matter". If 4 still shows the
ordering, the effect lives in the graph's output statistics rather than in the
map's anatomy, which is itself worth saying.

**Key saving:** the network runs open loop, so its motor history does not depend
on the map or the body. Simulate each realization and control graph once, cache
the 260-channel histories, then apply every map and body afterwards. Only
MuJoCo rollouts scale with the number of maps.

---

## 3. Compute and resources

Measured locally on this Mac (CPU): one 8 s LIF history ≈ 8–10 s; one 8 s
MuJoCo rollout ≈ 0.2 s. No GPU is needed, and running on CPU with realization
averaging sidesteps the CPU/CUDA reproducibility gap.

| Job | Histories | Rollouts | Est. wall time |
|---|---|---|---|
| A: real + T1/T2/T3/sensory × 30 realizations | 150 | 3,000 | ~35 min |
| Null arms (3 arms, 7 bodies) | 30 (reuse) | ~700 | ~5 min |
| B: 5 fractions × 12 seeds + real × 30 | 90 | 1,800 | ~20 min |
| C: 7 bodies × (30 real + 12 shuffled) | reuse | 5,900 | ~25 min |
| Map sweep: ~40 maps × (10 real + 12 shuffled) × 20 starts | reuse | 17,600 | ~60 min |
| **Total** | ~270 | ~29,000 | **~2.5–3 h** |

`MAX_ROLLOUTS = 12_000` in `specificity_campaign.py` is a per-campaign cap; the
map sweep needs its own run or a declared higher cap. If you prefer not to run
locally, a Runpod community pod at ~$0.17–0.22/hr covers this for under $1
(balance $2.89), but CPU-only is fine and cheaper.

Code changes, all small:
- A `noise_seed` pass-through in `collect_history` / `base_vnc` (the stepper
  already accepts it).
- Null arms: ~15 lines using the existing `translate(zeros, ...)` stance.
- `audit_h5_lesion_localization.py`: switch to `allow_unmatchable=True`, loop
  over realizations.
- One map-sweep script: cache histories, perturb `MAP_GAINS` / `io` labels,
  replay.
- A variance-components summary over the new result files.

Each new run keeps the standing protocol: frozen-weight hash check, watched
`real` video, amplitude-matched controls, Holm across the declared family.

---

## 4. Paper changes and the page budget

Text that must change once the reruns land (owner approval needed on every
claim):
- Abstract: T2 loss value and CI; "measured wiring produces positive forward
  drive" becomes "moves the body forward relative to a zero-drive baseline";
  add the failed phase criterion; add one sentence on the map sweep outcome.
- Methods: realization sampling and the three variance levels; the relaxed
  matching rule stated once for A and B.
- Results A: T1/T2/T3 comparison; realization-level test.
- Results A–C: zero-drive reference line in tables.
- Limitations: remove "We did not run these comparisons"; add the posture
  component of forward drive; keep the map caveat but cite the sweep.
- Close the open gate: `h5_lesion_localization_lif.json` still says
  `LESION-LOCALIZATION-PENDING-VIDEO`, `videos.reviewed: false`. The rerun
  replaces this file; watch the new `real` and `largest_loss` videos.

Room has to come from somewhere, since the draft is already at 5 pages. Options,
largest saving first — **owner decision**:
1. Shrink the full-width motor-map figure (Fig. 1) to one column.
2. Collapse Table III (9 body rows) into two sentences plus min/max, or fold it
   into a small combined figure with the map-sweep result.
3. Shorten Experiment E to one paragraph; it is descriptive and no reviewer
   asked about it beyond the abstract.
4. Tighten the model-constants paragraph in Methods.

Mechanics: IEEE CS format check, affiliation (currently "High School Student"),
figure sizes at final print, reference check, and **in-person attendance by at
least one student author** — a condition for inclusion in the proceedings.

---

## 5. Order and schedule

| When | Item |
|---|---|
| Day 1 | Abstract phase sentence; `noise_seed` plumbing; null arms |
| Days 2–3 | Rerun A (with T1/T3) and C over 30 realizations; variance components |
| Days 3–4 | Rerun B; pre-register map family in a short spec; run sweep |
| Days 5–7 | Owner review of numbers and claims; watch videos |
| Days 8–14 | Rewrite, cut to 5 pages, compile, visual audit |
| By Oct 18 | Submit (one week of margin) |

## 6. What to expect
- The T2 lesion result stays, smaller and with honest uncertainty.
- The zero-drive baseline turns R2's main concern into a point in the paper's
  favour.
- Some sentences get weaker: "positive forward drive" and at least one dose
  comparison. Both reviewers rewarded honest reporting; say what changed.
- The map sweep could still cut either way. Report the fraction, whatever it is.
