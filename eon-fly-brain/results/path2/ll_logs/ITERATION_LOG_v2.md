# Labeled-line loop v2

## Iteration 1 — resolve the invalid phase window

- Hypothesis: the frozen LIF graph may contain a measurable motor rhythm, but
  the previous 1 s rollout could not resolve a 1.5 Hz phase band.
- Prediction: with a 30 s signal and 15 s analysis tail, the spectrum will show
  whether the real graph has a distinct peak and whether `(1.5, 4.0) Hz` holds
  enough power for phase analysis. A rewired graph may differ if wiring identity
  creates the rhythm.
- Change: set the declared core window to 30 s; add a spectrum-only rollout for
  real `dng100 @ 150 Hz` and one degree-preserving rewire; add explicit window
  assertions and full 0–25 Hz per-leg/per-pool spectrum output.
- Verification: pending pod run. Laptop will run syntax and pure readout checks
  only; no experiment physics runs locally.
- Status: implementation complete; spectrum pod pending.

## Iteration 2 — enforce activity matching after amplitude matching

- Hypothesis: per-channel temporal-SD matching alone does not control the two
  reported activity metrics, so controls can remain louder or broader-active.
- Prediction: deterministic active-channel-count plus total-rate matching will
  put every rewired/random condition inside the preregistered `0.8–1.25`
  motor-Hz and fraction-active ratios; an unmatchable arm must fail loudly.
- Change: call existing `amplitude_match`, then re-measure and assert both
  metrics from the exact history used by the readouts.
- Verification: pending full pod run.
- Status: implementation complete; held until spectrum answers band question.

### Iteration 1 result — spectrum diagnostic

- Verification: A4500 pod at `$0.190/hr`; 30 s / 1500-step LIF rollouts,
  30 real and 30 single-rewire realizations; OMP/MKL limited to 2. Window
  assertions passed: 15 s analysed, 22.5 low-edge cycles, 37.5 band bins.
  Freeze hashes stayed equal and no learning occurred. Raw stdout and plot were
  pulled before pod deletion.
- Result: real per-leg `(1.5, 4.0) Hz` power share mean `0.1217`; rewired
  `0.0591`. Real per-pool mean `0.0907`; rewired `0.0528`. Real peaks are not
  one common rhythm: per-leg peaks are `1.33, 4.80, 5.00, 19.13, 23.33,
  23.53 Hz`; rewired peaks are `19.60–24.33 Hz`. Real motor mean was `7.21 Hz`
  in this 30 s run; its strongest peaks are not uniformly near 7 Hz.
- Interpretation: the declared `(1.5, 4.0) Hz` band has measurable real power,
  and is not simply absent, but power is mixed across legs/pools. Keep the band
  declared for the full test; do not retune it from outcome. Spectrum artifact:
  `spectrum_iteration1.json` plus `spectrum_iteration1.png`.
- Status: diagnostic resolved; proceed to the preregistered full grid with the
  30 s window and strict post-match activity assertions.

## Full-grid gate result — stop before phase comparison

- Verification: reproducible A4000 pod at `$0.170/hr`; fixed 30 s LIF window,
  30 real realizations, declared graph seeds, OMP/MKL=2. Freeze audit stayed
  clean: file hashes equal, real tensor hash equal to
  `128b5ceddec4d8e14f3a04fbb66a08a2e83564c674624741eb03fb2bf01517fc`,
  zero learning, parameter updates `0`. Two raw reproduction logs were pulled.
- Result: at rewired seed `20260830`, `dnb08 @ 100 Hz`, post-match motor rate
  was `9.3718 Hz` versus real `6.9340 Hz`, ratio `1.352` (> `1.25`). Four
  channels were unmatchable; fraction-active ratio was within tolerance. The
  exact failure reproduced on a second pod run.
- Decision: `LABELED-LINE-CONTROL-UNMATCHABLE`. Phase and silencing comparisons
  were not run against this unmatched control. Fresh `_lif_v2` artifacts record
  the stop, with distinct content and empty readout rows rather than a partial
  claim. No S1–S4 headline is valid.
- Interpretation: this VNC rewire changes activity structure enough that the
  declared Shiu-style graph comparison cannot be made fair under the fixed
  `0.8–1.25` activity rule. This is a control failure, not evidence that the
  real graph carries labeled-line specificity.

## Iteration 3 — repair S4 and artifact separation

- Hypothesis: S4 must compare real selectivity separation against both graph
  controls, and rate/selectivity, phase, and silencing artifacts must contain
  different readout objects.
- Prediction: S4 will be reported over the declared input-set pairs and rates;
  no control can be mistaken for a real effect, and output MD5 digests differ.
- Change: compare each real distance against rewired and random by more than
  real realization SD; trim each JSON to its named readout and assert distinct
  digests; write fresh `_lif_v2` artifacts.
- Verification: pending full pod run.
- Status: implementation complete; held until spectrum answers band question.
