# Labeled-line loop v3

## Scope and pre-run decisions

Next run if this one needs follow-up: repeat the phase comparison with a larger
realization count for any unresolved mode-sensitive condition, while auditing
`active_sample_fraction` as a temporal-sparsity phenotype.

Biggest threat: the graph may change temporal sparsity or switch between LIF
regimes, while Strategy A can match total rate and active-channel count but
cannot recreate the source burst timing. A second threat is residual body/CPG
contamination; the phase readout therefore stays open-loop and CPG-free.

Hypothesis: frozen VNC wiring creates labeled-line-specific antagonist and
inter-leg structure that survives activity-matched rewiring and weight-random
controls.

Prediction: if wiring identity matters, real phase concentration and selectivity
separation exceed both controls by the declared realization-SD margins; if not,
the full grid returns a null or structural-control result.

Decision before new runs: apply Strategy A only. The mandatory matcher bug fix
is sufficient for the declared scale-invariant phase/selectivity readouts;
Strategy B is not used because `dnb08` is non-monotone and calibration would
consume materially more budget. Strategy C is reporting-only, not a gate:
`active_sample_fraction` is recorded for every condition and discussed as a
temporal-sparsity limitation. The band remains locked at 1.5--4.0 Hz and the
30 s signal remains locked.

Run order: carry forward run-2's spectrum result, extend it to one random graph
and two additional rewired seeds, investigate `dnb08` bimodality and possible
runaway before the full grid, then execute every declared condition with an
explicit terminal state. A single unmatchable condition must not veto other
conditions.

Run-2 root cause: `strict_activity_match` copied an unmatchable source row and
continued before total-rate scaling. V3 treats channel deficit and rate
matching as separate facts, records both, and always applies the scalar rate
step.

## Iteration 1 — spectrum extension

- Hypothesis: run-2's real-versus-rewired band-power difference should persist
  across additional graph controls.
- Change: reused run-2's real and first-rewire JSON without rerunning those
  rollouts; measured two additional degree-preserving rewires and one random
  graph at `dng100 @ 150 Hz`, using the locked 30 s window and same 30 noise
  seeds.
- Prediction: stable rewired spread with real above controls supports carrying
  the band into the grid; a random control at or above real weakens any
  connectome-specific phase interpretation.
- Verification: A4500 pod at `$0.190/hr`; CUDA, freeze inputs, and CPG-free
  readout checks passed. Raw log, JSON, and plot pulled before pod cleanup.
- Result: carried-forward real mean per-leg band share `0.121685`; carried-forward
  rewired `0.059139`; extra rewired seeds `0.062513` and `0.061658`; random
  control `0.177961`. Real per-leg peaks remained mixed (`1.33--23.53 Hz`),
  rewired peaks stayed high (`18.27--25.00 Hz`), and random peaks were low
  (`0.13--0.60 Hz`).
- Status: diagnostic complete. The rewired separation is repeatable, but the
  random graph has more locked-band power than real. Keep the band fixed for
  readout completeness, but do not treat band power alone as evidence for a
  connectome-specific phase claim.
- Artifacts: `spectrum_iteration1_v3.json`, `spectrum_iteration1_v3.png`, and
  `spectrum_iteration1_v3.log`.

## Iteration 2 — `dnb08` bimodality and runaway check

- Hypothesis: the wide `dnb08` spread is a reproducible two-regime LIF
  transition rather than ordinary measurement noise, and its high regime may
  still be saturating at the end of the window.
- Change: before the grid, measured all four real `dnb08` rates, all four real
  `dng100` negative-control rates, and rewired/random `dnb08 @100`; recorded
  per-realization motor Hz, first/last 5 s, sorted-gap mode split, and motor-pool
  rates. The split rule was fixed before inspection: largest sorted gap above
  `max(0.5 Hz, 3x median sorted gap)`.
- Verification: A4500 pod at `$0.190/hr`; CUDA, freeze inputs, and open-loop
  CPG-free readout checks passed. Raw log, JSON, and plot pulled before cleanup.
- Result: real `dnb08 @100` split at `7.927 Hz` into 18 low realizations
  (`4.287 Hz`, late-minus-early `+3.967 Hz`) and 12 high realizations
  (`10.487 Hz`, late-minus-early `+7.703 Hz`). The high mode is still climbing
  at 30 s, so it is runaway saturation, not a stable labeled-line regime.
  `dnb08 @150` also split and climbed. `dng100 @50/100/200` was unimodal and
  tight; `dng100 @150` had one climbing outlier, but not the dnb08-wide split.
  Rewired `dnb08 @100` was stable at `15.391 ± 0.023 Hz`; random was stable at
  `0.659 ± 0.023 Hz`; neither reproduced the real split.
- Decision before grid: **exclude `dnb08` from the headline comparison for
  runaway saturation**, retain all its measured rate/phase rows as explicit
  `excluded-with-reason` limitations, and run the headline interpretation on
  the other three input sets. Do not average the two dnb08 modes or use them
  to support S1--S4.
- Status: diagnostic complete; primary grid ready.
- Artifact: `dnb08_bimodality_v3.json`, `dnb08_bimodality_v3.png`, and
  `dnb08_bimodality_v3.log`.

## Full V3 run — grid, matching, ablations, and silencing

- Verification: the frozen CUDA LIF core completed all 400 declared grid
  conditions: 30 real realizations plus 12 rewired and 12 random graph seeds,
  with 10 noise realizations per control cell. There were 272 evaluated cells,
  100 explicit `dnb08` exclusions, and 28 grid cells ending
  `unmatchable-after-strategy-A`; no condition was absent. The silencing ladder
  completed 18 candidates across real and 12 rewired seeds (234 rows). All
  four V3 JSON artifacts are non-empty and have distinct local MD5 digests.
- Matcher result: the Strategy A deficit unit check passed. Total motor rate
  and active-channel-count ratios are recorded per control condition; raw and
  unscaled control rate tables are retained. `active_sample_fraction` is
  report-only. Unmatchable cases remain terminal audit states rather than being
  silently averaged into valid comparisons.
- Primary result at declared `dng100 @ 150 Hz`: real antagonist phase pass
  count `4.00 +/- 0.91` of 20 cells; rewired `0.040`; random `0.056`. Only 4/20
  real cells reached concentration `>= 0.3`, so S1 failed. S2, S3, S4, S5,
  S7, and S8 passed; S6 is a failed all-conditions gate because some controls
  remain unmatchable, although its audit is complete. Six non-dnb08 selectivity
  pairs exceeded both controls by the real-arm SD criterion.
- Headline selectivity pair (`dng100` versus `dng100_left_only`) separated at
  150 Hz: real distance `0.10943`, rewired `0.01264`, random `0.09339`.
  Random remains close, so this is not a blanket connectome-specific claim.
  The locked-band share was real `0.11874`, rewired `0.06809`, random
  `0.14438`; random exceeding real keeps band power from serving as a specific
  mechanism by itself.
- Silencing result: no named candidate produced a strong real-arm phase loss.
  Largest positive real pass-count drop was only `0.20` for `INXXX466`; most
  named candidates were zero or negative. The A/B/C/D ablation ladder was
  measured; constant-connectome and drive-only arms are correctly labelled
  phase-degenerate.
- Body result: corrected NMF CPG reference was descriptive only and travelled
  `0.8535 m` forward over 8 s with no fall; fixed real motor-mean mapping
  travelled `-0.0023 m`. This confirms the CPG interface can walk, but does not
  credit the connectome with body distance.
- Final verdict: **`LABELED-LINE-STRUCTURAL-DNB08-RUNAWAY`**; artifact status
  `complete`, decision `OWNER_APPROVAL_REQUIRED`. Biggest threat remains
  graph-dependent temporal sparsity and regime switching under rate matching.
  If another run is approved, increase realization count only for the resolved
  non-dnb08 phase mode and keep active-sample fraction as a reported phenotype.
- Post-pull metadata audit found ledger item 6 incorrectly conflated failed
  global matching success with completed matching audit. Generator and four
  local artifacts were corrected to mark the audit complete with the
  report-only/unmatchable caveat; numeric rows and verdict were unchanged.
