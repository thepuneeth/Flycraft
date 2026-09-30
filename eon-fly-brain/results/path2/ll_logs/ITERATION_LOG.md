# Labeled-line loop

## Iteration 1 — readout and preflight

- Hypothesis: named descending input reaches named motor-neuron pools, so the
  frozen real VNC should show structured antagonist phase and different
  selectivity rows for DNg100 versus DNb08.
- Prediction: real antagonist cells should cluster near `|phase_cycles|=0.5`
  with concentration at least `0.3`; controls should pass fewer cells. DNg100
  and DNb08 rows should separate after within-row z-scoring. If phase is noise,
  concentration will sit near zero and the graph controls will match real.
- Change: add the minimal labeled-line readout and launch preflight; no core,
  body, CPG, training, or experiment rollout runs locally.
- Verification: local `py_compile` and pure synthetic checks passed; anti-phase
  returned `0.5`, noise had low concentration, constant phase was undefined;
  the frozen index reproduced 14 types and 20 testable cells. Pod preflight
  passed on `cuda:0` with expected tensor hash; no physics ran on laptop.
- Status: first rollout stopped before artifacts after a readout indexing bug;
  fixed and smoke-tested on pod. Full run pending.

## Iteration 2 — reference-duration correction

- Hypothesis: Shiu's 1 s labeled-line trial is sufficient for the named rate and
  selectivity readouts; extending every graph-control trial to 8 s only multiplies
  cost and changes the reference protocol.
- Prediction: the full 1 s grid will finish within practical pod time; phase may
  be low-resolution and must be marked unsupported if band power is weak.
- Change: core sweep `SIGNAL_SECS` changed from 8 s to the reference 1 s;
  separate 8 s body trace remains descriptive.
- Verification: local `py_compile` passed; no result was interpreted from the
  stopped run.
- Status: superseded by Iteration 3 execution optimization.

## Iteration 3 — batched fixed realizations

- Hypothesis: batching independent frozen LIF realizations changes throughput,
  not dynamics or readout values.
- Prediction: output shape remains `(realization, time, 260)` and summaries keep
  30 real / 10 control realizations.
- Change: run each declared seed stream as one batched call through the existing
  `ShiuLIFVNC` core; no weights, drive, or readout formula changed.
- Verification: pod smoke returned `(3, 50, 260)`, 14 types, finite summaries.
- Status: complete.

## Iteration 4 — antagonist slice indexing

- Hypothesis: the first full run's unequal agonist/antagonist vector lengths came
  from NumPy advanced-index ordering, not the VNC output.
- Change: use explicit `(time, motor_mask)` slicing before pool means.
- Verification: pod 400-step smoke passed; no artifact from failed run retained.
- Status: complete.

## Iteration 5 — rhythm-type annotation alignment

- Hypothesis: `rhythm_core_labels` stores three unique labels, while 18 rhythm
  nodes need annotation-derived per-node labels for type silencing.
- Change: resolve rhythm node types from the frozen body annotation table.
- Verification: exactly 18 candidates built: 3 rhythm types (6 nodes each) plus
  top 15 published-coordination types by motor-target out-degree.
- Status: complete; final rollout finished.

## Final rollout result

- Full grid: 16 input-set/rate rows for real plus 12 rewired and 12 random graph
  seeds, all with declared Poisson realizations; 400 readout rows and 234
  silencing rows pulled from pod.
- Primary `dng100 @ 150 Hz`: real antagonist pass count `4.33 ± 1.37` of 20;
  rewired `2.03`; random `1.63`. S1 fails; S2/S3 pass numerically.
- Named selectivity distance `dng100` vs `dnb08`: real `0.09697 ± 0.01886`,
  rewired `0.01620`, random `0.20705`; S4 passes against declared real SD.
- Band power: real `0.1489`, rewired `0.1344`, random `0.2160`. Phase is not
  supported because real in-band power is low and not above random; numeric
  status `LABELED-LINE-PHASE-NOT-SUPPORTED-BAND-POWER`.
- Freeze audit: expected file hashes and real tensor hash equal pre/post;
  `zero_learning=true`, `parameter_updates=0`. Aggregate activity ratios remain
  below the 2× gate, but individual graph/input rows are visibly heterogeneous
  and are retained in the artifacts.
- Interpretation: no approved headline. The named selectivity contrast is
  descriptive until the low-band-power phase issue is resolved; body distances
  are descriptive only. `OWNER_APPROVAL_REQUIRED`.
- Lifecycle: artifacts and final raw stdout pulled; pod `r64crbclu36zxk`
  terminated; `runpodctl pod list` is empty.
