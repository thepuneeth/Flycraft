# Path 2 pivot — connectome-as-reservoir readout (cheapest learning)

The no-learn chain is exhausted (E0–E4 all ABSENT). This is the learning pivot,
chosen because it is the smallest possible learned surface: **one linear readout
matrix, fit in closed form.** The real MANC VNC weights stay frozen. Nothing in
the connectome is trained.

## Claim under test

The frozen fly VNC population, treated as a reservoir, linearly encodes a
tripod walking gait for a PhantomX-class hexapod — even though the pooled motor
outputs do not (E0). And: a real-connectome reservoir does this better than a
random or rewired reservoir of matched size (the prior transfers to a non-fly
body).

## Honesty rules (unchanged spine)

- VNC weights **frozen**. Only `W_out` (population state → 18 joint targets) is
  learned, by ridge regression. No backprop, RL, reward, or catch-success term.
- The regression target is a **fixed kinematic tripod template** (Path 1's
  hand-CPG joint trajectory on this exact hexapod), NOT prey capture, NOT
  distance, NOT any performance signal. Steering stays brain-driven, unlearned.
- The template is declared once and never edited to improve a result.
- Report the null honestly if random ties connectome. Report failure honestly
  if nothing walks. Do not tune the reservoir drive or ridge λ to manufacture a
  walk — λ is set by generalization on held-out time, not by walking distance.

## Data flow

```text
constant tonic drive -> frozen MANC VNC (reservoir) -> population state r(t)
r(t)  --ridge-->  W_out  ->  18 joint targets q(t)  -> PD torque -> MuJoCo
target for the fit: fixed Path-1 tripod q*(t)   (kinematic template only)
```

## Build steps

### R0 — collect reservoir states
1. Drive the frozen VNC with the existing constant tonic drive (same protocol
   as the rhythm validation: `tau=.02`, `dt=.005`, substeps 4, `gW=.03`,
   drive 100). No rhythmic/clock input — the reservoir must supply its own
   dynamics.
2. Record the **full population activation** `r(t)` (all VNC neurons, not just
   the 260 MN outputs) for a long window (≥ 3000 steps after burn-in), sampled
   at the sim rate. Save as a dense matrix `[T × N]`.
3. Standardize columns (zero-mean, unit-var) using train-window stats only.

### R1 — fit the linear readout (the only learning)
1. Target `q*(t)`: the Path-1 hand-CPG 18-joint trajectory, resampled to the
   same timebase. This is the fixed template.
2. Split time into train / held-out (e.g. first 70% / last 30%). Fit
   `W_out = argmin ||q* − r·W||² + λ||W||²` by closed-form ridge on the train
   window only.
3. Choose λ by reconstruction error on the **held-out time window** — never by
   walking distance. Record the λ path and the chosen value.
4. Report train and held-out reconstruction R² per joint.

### R2 — deploy and measure walking
1. Run closed-loop: at each control step feed live VNC population state through
   the frozen `W_out` to get joint targets, PD to torque, into MuJoCo. Brain
   supplies steering exactly as now.
2. Measure: tripod-coherence scalar (same E0 metric, now on the *readout* joint
   targets), forward distance, path length, falls, min prey distance.
3. Render the multiview video. Confirm visible tripod vs sprawl.

### R3 — the 3-way prior test (the moat)
Repeat R0–R2 with two matched control reservoirs, identical drive, identical
readout procedure, identical template, identical λ-selection rule:
- **connectome:** real MANC weights.
- **random:** random weights matched to the connectome's size and degree
  distribution.
- **rewired:** degree-preserving edge shuffle of the real graph.

Primary comparison metrics, connectome vs random vs rewired:
- held-out reconstruction R² of the tripod template;
- readout rank needed to reach a fixed R² (effective dimensionality);
- actual closed-loop forward distance and tripod coherence;
- generalization to a held-out drive level or phase offset.

Verdicts:
- **PRIOR CONFIRMED:** connectome reservoir reaches the template with lower
  error / fewer dimensions / better walking than random and rewired.
- **NULL:** random ties connectome → linear readout suffices, wiring not
  special for this task. Honest, still publishable.
- **READOUT TOO WEAK:** nothing walks under a linear readout → escalate to a
  single small nonlinear readout (one hidden layer, still tiny), rerun R1–R3,
  and report that linear capacity was insufficient.

## Hardware

Trivial. Reservoir-state collection and closed-form ridge run on CPU in
minutes; MuJoCo rollouts are CPU. No GPU, no RunPod needed for this pivot. If
R2 escalates to a nonlinear readout, still CPU-scale.

## Deliverables
- `code/collect_reservoir_states.py` — R0.
- `code/fit_readout.py` — R1 ridge fit, λ path, R² report.
- `code/run_reservoir_walk.py` — R2 closed-loop + video.
- `code/reservoir_3way.py` — R3 connectome/random/rewired comparison + figure.
- `results/path2/reservoir_*` — states, W_out, R² tables, 3-way comparison,
  video.
- Report section: method, template declaration, λ-selection rule, per-condition
  metrics, verdict. No final claim wording until user approval.

## Gates
- GATE R1: post held-out R² per joint before deploying. If R² is near zero even
  in-sample, stop — the template isn't in the reservoir; report and escalate.
- GATE R2: post tripod coherence + distance + video before the 3-way.
- GATE R3: post the connectome-vs-random-vs-rewired table. STOP for approval
  before any headline.

## Discipline
Same as before: freeze all VNC constants, declare every choice, select λ by
generalization not performance, report nulls and failures plainly, no
catch-fitting anywhere. The learned surface is one matrix and nothing else.
