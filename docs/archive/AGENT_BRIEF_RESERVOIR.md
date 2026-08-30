# Agent brief — Path 2 reservoir pivot (self-contained)

You are continuing the Flycraft project. Read this whole brief before acting.
Then read `HANDOFF.md`, `PATH2_METHODS.md`, and `RUN_PATH2_RESERVOIR.md` for
full context. Do not restart finished work.

## Where the project stands

An untrained FlyWire brain + the real Male-CNS MANC VNC were wired to a
PhantomX-class hexapod in MuJoCo with **zero learning**. That no-learn chain is
finished and committed (`6481a2d`). It is a clean negative: the VNC oscillates
and real wiring beats shuffled (+0.1425 m, p=5.7e-6), but the six legs never
organize into an alternating tripod. Four anatomical interventions (E0 baseline,
E2 completeness, E3 real descending drive, E4 anatomical proprioceptors) all
failed to produce tripod. Failure is localized to VNC phase organization
(layer i), not the motor map or physics.

**Your job is the learning pivot** — and specifically the *cheapest* learning:
treat the frozen VNC as a reservoir and learn exactly one linear readout matrix.

## The one hard rule (this project's spine)

We publish an honest result, never a manufactured one. Therefore:

- The **VNC weights stay frozen.** You train nothing inside the connectome.
- The **only** learned object is `W_out`, a linear map from VNC population state
  to 18 hexapod joint targets, fit by closed-form ridge regression. No backprop,
  no RL, no optimizer loop, no reward, no loss on task success.
- The regression **target is a fixed kinematic tripod template** — the Path-1
  hand-CPG joint trajectory that already walks this exact hexapod. It is NOT
  prey capture, NOT distance, NOT any performance signal. Declare it once; never
  edit it to improve a result.
- Ridge `λ` is selected by reconstruction error on a **held-out time window**,
  never by walking distance or catch. Record the full λ path.
- Steering stays brain-driven and unlearned, exactly as in the current chain.
- Report nulls and failures plainly. If random ties connectome, say so. If
  nothing walks, say so. Do not tune drive, λ, or the template to force a walk.

Violating any of these voids the result. When in doubt, stop and report.

## Repo facts you need

- Run everything from `/Users/puneeth/Documents/Flycraft` with the frozen venv:
  `eon-fly-brain/.venv/bin/python`.
- Frozen VNC protocol (do not change): rate model, `tau=.02`, `dt=.005`,
  substeps 4, `gW=.03`, tonic drive 100, DNb08 gain 1, threshold 7.5,
  activation gain 1, firing-rate cap 200, heterogeneity seed 1.
- Reuse existing code — do not reimplement:
  - `eon-fly-brain/code/vnc_stepper.py` — paper-rate VNC dynamics + shuffle
    control. This is your reservoir; call it, don't rewrite it.
  - `eon-fly-brain/code/mn_to_joint.py` — the fixed 5→3 motor-pool→joint map
    (used by the no-learn chain; the reservoir readout replaces its role but
    reuse it to locate the 18 joint channels and the Path-1 template).
  - `eon-fly-brain/code/run_vnc_chase.py` — brain/VNC/body episode runner +
    multiview video renderer. Reuse its MuJoCo setup, PD servo, and camera rig.
  - Path-1 hand-CPG gait (in the Path-1 hexapod code) — the source of the
    fixed tripod template `q*(t)`.
  - `eon-fly-brain/code/probe_vnc_phase.py` — the E0 tripod-coherence metric.
    Reuse it verbatim to score readout output; do not invent a new metric.
- Build VNC graph artifacts with `fetch_manc.py` (data under
  `eon-fly-brain/data/manc/`, gitignored, already present). Weight matrices in
  `results/` named `*weight_csr.pkl` are gitignored and rebuildable.

## What to build

### R0 — collect reservoir states  → `code/collect_reservoir_states.py`
1. Load the frozen all-leg VNC graph. Drive it with the constant tonic drive
   above. No rhythmic/clock input — the reservoir must supply its own dynamics.
2. Record the **full population activation** `r(t)` for all VNC neurons (not
   just the 260 motor outputs), ≥ 3000 steps after a half-run burn-in, at the
   sim sample rate. Save a dense `[T × N]` matrix + the neuron id order.
3. Standardize columns to zero-mean/unit-var using **train-window stats only**;
   save the scaler.

### R1 — fit the one readout  → `code/fit_readout.py`   [GATE R1]
1. Build target `q*(t)`: the Path-1 hand-CPG 18-joint trajectory, resampled to
   the reservoir timebase. Save it as the declared template artifact.
2. Split time: first 70% train, last 30% held-out.
3. Closed-form ridge: `W_out = (RᵀR + λI)⁻¹ Rᵀ Q*` on the train window only.
4. Sweep λ over a log grid; pick λ by **held-out** reconstruction R² (mean over
   18 joints). Record the whole λ→R² path and the chosen λ.
5. Report per-joint train and held-out R².
6. **GATE R1 — STOP and post** per-joint held-out R² + chosen λ before R2. If
   held-out R² is near zero even in-sample, the template is not in the
   reservoir: report it and go to the nonlinear-escalation note in R2, do not
   silently proceed.

### R2 — deploy and measure  → `code/run_reservoir_walk.py`   [GATE R2]
1. Closed loop: each control step, feed live VNC population state through the
   frozen scaler + `W_out` to get 18 joint targets → PD torque → MuJoCo. Brain
   supplies steering exactly as in `run_vnc_chase.py`.
2. Measure with the reused E0 metric on the readout joint targets:
   tripod-coherence scalar, within/cross phase, plus forward distance, path
   length, falls, min prey distance, upright height.
3. Render the multiview video.
4. **GATE R2 — STOP and post** tripod coherence + distance + video verdict
   (visible tripod vs sprawl) before R3.
5. If linear readout produces no walk anywhere, escalate ONCE to a single small
   nonlinear readout (one hidden layer, ≤ 64 units, fit by ridge on random
   features or a short least-squares — still no RL/reward), rerun R1–R2, and
   label the result "linear capacity insufficient." Do not escalate further.

### R3 — the 3-way prior test (the moat)  → `code/reservoir_3way.py`   [GATE R3]
Repeat R0–R2 with three reservoirs under identical drive, identical readout
procedure, identical template, identical λ-selection rule:
- **connectome:** real MANC weights.
- **random:** random weights matched to size + degree distribution.
- **rewired:** degree-preserving edge shuffle of the real graph.
Use the existing shuffle machinery in `vnc_stepper.py` where possible; declare
exactly how random/rewired are generated and seed them.

Report a table, connectome vs random vs rewired:
- held-out reconstruction R² of the template,
- readout rank/dimensionality needed to reach a fixed R² (effective dim),
- closed-loop forward distance and tripod coherence,
- generalization to a held-out drive level or phase offset.

Verdicts:
- **PRIOR CONFIRMED:** connectome beats random and rewired on error / dimension
  / walking.
- **NULL:** random ties connectome → linear readout suffices, wiring not
  special. Honest, publishable.
- **READOUT TOO WEAK:** nothing walks under linear readout → the nonlinear
  escalation result stands; report linear capacity was insufficient.

**GATE R3 — STOP for user approval** before writing any headline/claim.

## Reporting format each gate

Post: the command run, the numeric result (table or per-joint values), the
verdict label, and one line on whether the honesty rules held (frozen VNC,
template unchanged, λ by held-out). No headline language until GATE R3 approval.

## Deliverables
- `code/collect_reservoir_states.py`, `code/fit_readout.py`,
  `code/run_reservoir_walk.py`, `code/reservoir_3way.py`.
- `results/path2/reservoir_states.*`, `reservoir_template_q.*`,
  `reservoir_readout_W.*`, `reservoir_lambda_path.*`,
  `reservoir_walk_metrics.json`, `reservoir_walk_multiview.mp4`,
  `reservoir_3way.{json,csv,png}`.
- A `PATH2_METHODS.md` section: method, template declaration, λ-selection rule,
  per-condition metrics, verdict. No final claim wording until approved.
- Reservoir `*weight_csr.pkl` are gitignored/rebuildable; commit code + small
  artifacts + report only.

## Hardware
CPU only. Reservoir collection and closed-form ridge run in minutes; MuJoCo
rollouts are CPU. No GPU/RunPod for this pivot, including the nonlinear
escalation.

## Integrity before any commit
Python compile all `code/*.py`; JSON/CSV parse; confirm VNC weights unchanged
(hash-equal to the frozen graph); zero-learning-except-readout scan (the only
fit is the ridge/least-squares readout — no optimizer/backprop/reward/loss on
task success); `git diff --check`. Then commit code + small artifacts + report
with a message stating the verdict. STOP for user approval before the headline.
