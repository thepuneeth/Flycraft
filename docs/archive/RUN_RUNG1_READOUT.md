# Rung 1 — minimal-training trained readout, and the moat test that decides the project

**Read this whole file, then `HANDOFF.md`, `PATH2_METHODS.md`, and the
"Frozen-reservoir readout pivot (R0–R3)" section of `PATH2_METHODS.md` before writing any
code.** You are continuing Flycraft. **This is the first brief that permits training.** The
zero-training program is closed (coordination NULL across six methods; ceiling ~35–40%).
This run adds the **smallest possible** trained component — a tiny readout — and its entire
purpose is to answer one question: **does the frozen connectome earn its keep under
training, or is it decoration?** Work the phases in order; every phase has a GATE that must
print PASS (or a declared NULL) before the next starts.

---

## 0. Scope change, the question, and what "honest success" means now

**Scope change (read carefully).** Previous briefs forbade all learning. This one allows
learning **only** in a single small readout network. Everything that made the project novel
must stay frozen:
- The connectome (FlyWire brain + MANC VNC) stays **frozen and hash-equal** to the committed
  graph. Not one weight trains. (T6 still enforced.)
- The fixed body map `mn_to_joint` gains, the stance, H_MIN=0.08, the VNC rate-model protocol
  — all frozen.
- The **only** trained object is the readout `f_θ: (frozen VNC state) → (18 joint targets)`.

**The question.** The reservoir pivot (R0–R3) already tried a *linear* readout and a *fixed
random-feature* readout of the frozen states, both via closed-form ridge, and both were
`READOUT TOO WEAK`. Worse, a **random** reservoir reconstructed the target gait **better**
than the connectome (dynamic held-out R² −0.02 vs −0.33). That is the warning this run must
confront head-on. The new lever is a genuinely **backprop-trained, nonlinear, memoryless**
readout. The question is not merely "can a trained readout walk the body" — it is:

> **Under an identical trained readout, does the connectome substrate beat random and
> rewired substrates on the body?**

**The moat lives or dies on that comparison. Nothing else in this run matters as much.**

**Honest success (pre-register before running):** the connectome-fed readout produces a
**stable walking tripod** (feet-planted comparable to the clean-clock positive control, low
falls, positive forward path) **and beats both the random-substrate and rewired-substrate
readouts** on the body metrics (coherence and feet-planted) with Holm-corrected `p < 0.05`
and non-overlapping bootstrap CIs, **and** the watched `connectome` video shows a genuine
tripod walk. Only that earns: *"frozen connectome + a thin learned body-interface walks, and
the connectome is necessary."*

**Honest null:** the connectome-fed readout does not beat the random-substrate readout on the
body → the trained readout learns the gait regardless of substrate; the connectome is not a
privileged locomotor substrate even under minimal training → report it and **stop**. Do not
climb to RL/bigger nets to force a walk that would prove nothing about the wiring.

---

## 1. Honesty spine for the training era (new — read every line)

1. **Only the readout trains. The connectome is frozen and hash-equal (T6).** No optimizer
   ever touches a MANC/FlyWire weight. Assert the graph SHA-256 before and after training.
2. **The training target is the clean-clock gait — a declared, connectome-blind reference.**
   The hand-scripted clean-clock tripod engine (the Phase-A T4 positive control) generates
   the 18 joint targets. This is the same kind of behavioral-cloning target the reservoir
   declared. The target is fixed before fitting and never changed.
3. **No body-outcome quantity feeds training or model selection.** The **only** loss is
   readout MSE (or Huber) on the clean-clock joint targets. Distance, coherence, feet-planted,
   falls, reward, prey — none of them enter the loss, the optimizer, or hyperparameter/epoch
   selection. Model selection uses **validation BC error on held-out drives only**.
4. **The readout is memoryless and clock-free.** Feedforward MLP, **no recurrent state, no
   time index, no phase input** — its input at step t is the current frozen VNC state vector
   only. This is load-bearing: a memoryless map cannot count steps, so any gait it produces
   **must** come from the temporal structure of its input — i.e. from the connectome's
   dynamics. This is what makes the substrate comparison meaningful.
5. **The readout is capacity-limited and declared.** One hidden layer, small declared width
   (start ~64 units; declare it; do not grow it to rescue a walk). A big net can approximate a
   standalone clock from spurious cues — keep it small so the input must do the work.
6. **Identical training across all substrate conditions.** `connectome`, `random`, `rewired`,
   and the `static-input` control share **identical** architecture, optimizer, learning rate,
   epochs, batch order, seed, target, split, and standardization procedure. The **only**
   difference is the input substrate. Any divergence is a confound — assert equality.
7. **Every condition is scored on the body** with the standard controls + a watched video.
8. **When in doubt, STOP and report. A null moat is the deliverable.** Do not tune the net,
   the loss, or the target to make the connectome win. If connectome ties random, that is the
   finding.

---

## 2. Phase A — pre-flight (MANDATORY; reuse `code/preflight.py`; ~2 min)

Re-run the six-trap gate unchanged.

| # | Trap | Pass |
|---|---|---|
| T1 | Squat stance | static settle ≥ 0.12 m |
| T2 | Fall floor hides squat | `H_MIN = 0.08` everywhere |
| T3 | No L/R mirroring | stance == `m.key("home").qpos[7:25]` |
| T4 | Engine can't walk | clean-clock engine ≥ 1.0 m / 8 s, no fall |
| T5 | Broken duty | 0.35 ≤ clean-clock duty ≤ 0.6 |
| T6 | Weights changed | SHA-256(weight+index) == committed |

**GATE A:** print `GATE A: PASS (T1..T6 = numbers)` or `GATE A: FAIL <trap>` and stop.
Additionally confirm the clean-clock engine (the BC target generator) runs and produces the
18-joint trajectory you will clone. Do not proceed on FAIL.

---

## 3. Phase T-1 — build the behavioral-cloning dataset (declared, leak-free)

Reuse the reservoir R0 state-recording approach (`PATH2_METHODS.md` R0) — do not invent a new
one.

1. **Input X** — per control step, the frozen VNC population state under the driving protocol
   that produces forward walking (the same tonic/descending drive the clean-clock gait uses).
   Use the same varying-column selection and **train-window-only** standardization the
   reservoir declared (column means/scales from the training window only; never from val/test).
2. **Target Y** — per control step, the clean-clock engine's 18 joint targets, **time-aligned**
   to X step-for-step at 50 Hz (`CTRL_HZ`). The known fixed stance is a coordinate convention,
   exactly as in the reservoir pivot; the sole learned object is the state→joint map.
3. **Split (declared, pre-registered):** fit on the first fraction of trajectory / a declared
   set of drives; **select epochs/hyperparameters on a held-out validation split**; **deploy
   and score on drives never used in fit or selection.** Walking/distance/prey/reward never
   enter any split decision.

**Substrate conditions (reuse the committed R3 graph controls — same definitions, same
seeds):**
- `connectome` — the real frozen MANC graph.
- `random` — the exact sparse graph with signed edge weights permuted (matched degree and
  weight multiset), R3 seed.
- `rewired` — degree-preserving directed endpoint swaps (matched in/out degree and weight
  multiset), R3 seed.
- `static-input` **(leak detector / clock control)** — the connectome dynamics replaced by a
  time-constant input (e.g. the mean training-window state repeated) with matched marginal
  scale. A memoryless readout fed a constant input **must** produce a constant output and
  therefore **cannot** walk. If `static-input` walks, there is a leak (a time cue in the
  input, or a non-memoryless net) — **STOP and fix** before trusting any other condition.

**T-1 self-checks:** X/Y shapes and step-alignment; standardization stats derived from the
train window only (print them); no row of val/test appears in train; every substrate produces
an X of identical shape so the readout architecture is byte-identical across conditions.

Commit `rung1: T-1 BC dataset + substrate controls`.

---

## 4. Phase T-2 — train the readout (identical budget across substrates)

1. **Architecture (declare, freeze for all conditions):** feedforward MLP, input = VNC state
   dim, **one** hidden layer of declared width (~64), nonlinearity (declare, e.g. tanh/ReLU),
   output = 18. No recurrence, no time/phase input, no batchnorm that leaks sequence order.
2. **Loss:** MSE (or Huber) on the 18 joint targets. **Nothing else.**
3. **Optimizer/budget:** declare optimizer, learning rate, batch size, epoch cap, and a single
   fixed seed. **Identical** for `connectome`, `random`, `rewired`, `static-input`.
4. **Model selection:** pick the epoch with the best **validation BC error** (not body
   outcome). Report train/val BC error per condition.
5. **Assert identical setup:** hash or print the architecture/optimizer/seed/target/split for
   all four conditions and assert equality. The only legal difference is the input substrate.

**Read the BC errors honestly before deployment:** if `random`/`rewired` reach BC error
comparable to or better than `connectome` (as the reservoir found for the linear case), that
already foreshadows a null moat — but still deploy, because BC error and body walking can
diverge.

**GATE T-2:** print the per-condition train/val BC error table + the identical-setup assertion.
Commit `rung1: T-2 trained readout (identical budget)`.

---

## 5. Phase T-3 — closed-loop deployment and the decisive substrate comparison

Deploy each trained readout closed-loop in MuJoCo exactly as the reservoir R2 deployment
(`PATH2_METHODS.md` R2): the readout's 18 outputs are the PD-servo targets each control step;
the FlyWire forward pass supplies its existing unlearned steering exactly as in
`run_vnc_chase.py`; H_MIN=0.08; same episode count and protocol as the last-three-jobs body
runs.

**Conditions:** `connectome` / `random` / `rewired` / `static-input`, each its own trained
readout from T-2, identical deployment.

**Metrics (reuse the committed body-metric machinery):** path m + 95% CI, forward m + CI,
tripod coherence scalar + CI, feet-planted + CI, falls (Wilson 95% CI), per-run tripod verdict
(`phase_metrics`). **Decisive paired tests:** `connectome − random` and `connectome − rewired`
on coherence and feet-planted, with exact + Holm `p` and bootstrap CIs (`BOOTSTRAP_REPS=100_000`,
`BOOTSTRAP_SEED=20260829`). Render and **watch**
`results/path2/rung1_connectome_walk_multiview.mp4`.

**Pre-registered null (write into the results file BEFORE running):** `connectome` does not
beat `random` on the body walking metrics under identical training → `RUNG1-MOAT-NULL`: the
trained readout learns the gait regardless of substrate; the connectome is not a privileged
locomotor substrate even under minimal training.

**Pre-registered positive (write it too):** the full conjunction in §0 — `connectome` walks a
stable tripod (feet-planted ~ clean-clock, low falls, forward path), beats **both** `random`
and `rewired` with Holm `p<0.05` and non-overlapping CIs, `static-input` does **not** walk
(leak check clean), and the video shows a real tripod → `RUNG1-MOAT-EARNED`: frozen connectome
+ thin learned interface walks and the connectome is necessary.

**GATE T-3:** print the four-condition table + paired tests + leak-check result + watched-video
verdict + the explicit `RUNG1-MOAT-NULL` or `RUNG1-MOAT-EARNED` line. Commit
`rung1: T-3 deployment + substrate comparison`.

---

## 6. Phase T-E — reconcile, score, decide (NO final claim)

Update `PATH2_METHODS.md` and `HANDOFF.md`:
- New section "Rung 1 trained readout (T-1..T-3)" with the BC-error table, the body-metric
  table, both pre-registered outcomes, and the leak-check.
- **State the decision plainly:**
  - `RUNG1-MOAT-EARNED` → the minimal-training moat holds. This is the new positive result and
    the natural spine of the paper: *frozen connectome does the computation; a thin learned map
    only adapts it to the non-fly body, and a random substrate cannot substitute.* Flag for
    user claim review; recommend Rung-2 questions (how thin can the readout get, ablations).
  - `RUNG1-MOAT-NULL` → the connectome is not a privileged locomotor substrate under a trained
    readout. Do **not** escalate to RL to force a walk — a walk that random also achieves proves
    nothing about the wiring. The honest paper is then the six-null zero-training boundary plus
    this negative moat test. Recommend stopping the "connectome drives the body" line.
- Update `docs/archive/README.md` to mark `RUN_RUNG1_READOUT.md` executed with the verdict.

**STOP. Print `READY FOR CLAIM REVIEW` and wait.** The user writes/approves the claim.

---

## 7. Per-commit integrity gate (before EVERY commit)

1. `python -m compileall -q eon-fly-brain/code` clean.
2. New JSON/CSV parse; H.264 video plays.
3. **Connectome graph hash-equal before AND after training** (T6). The trained readout is the
   only new set of weights; save it separately.
4. `mn_to_joint` gains still `(-0.55, 0.65, 0.65)`; stance/H_MIN/VNC protocol intact.
5. **No-leakage scan:** the loss/optimizer/model-selection code references **only** the BC
   joint-target error — grep the training + selection functions for
   `path|forward|distance|coherence|feet|planted|fall|reward|prey`; any hit there is training
   to the outcome = faking → remove it.
6. **Memoryless assert:** the readout has no recurrent state and no time/phase input; the
   deployment feeds only the current VNC state. Grep the model for `time|step_index|phase|
   hidden_state|lstm|gru|rnn|recurrent` — none permitted in the readout path.
7. **Identical-setup assert:** architecture/optimizer/seed/target/split identical across
   `connectome`/`random`/`rewired`/`static-input`; only the input substrate differs.
8. Fall floor 0.08 everywhere. `git diff --check` clean. Message names phase + result.

---

## 8. DO-NOT-RERUN / DO-NOT-ESCALATE (cite verdicts; stay minimal)

- **Reservoir linear + fixed random-feature ridge** → `READOUT TOO WEAK` (random ≥ connectome).
  This run's only new lever is a **backprop-trained memoryless nonlinear MLP** — do not re-run
  the ridge variants.
- **All six zero-training coordination methods** (M1, Phase B, E4, COORD-212, COORD-INHIB, DC)
  → NULL. Do not re-touch them.
- **RL / PPO / DMPO, learnable node features, learned coupling, bigger nets** → **out of scope
  for Rung 1.** Rung 1 is the minimal-footprint moat test. Only if `RUNG1-MOAT-EARNED` do we
  consider a later rung — and never to rescue a null.

---

## 9. Anti-mistake / fail-fast contract

- **`static-input` leak check is a hard gate.** If a memoryless readout fed constant input
  walks, everything downstream is untrustworthy — stop and find the time leak.
- **Watch the `connectome` video before believing any number.** Coherence can pass while the
  body drags (Method 1 "flew" at 1.5 m path).
- **Read BC errors before deploying.** If random ties connectome in BC, expect a null moat and
  do not be surprised into tuning.
- **One thing at a time:** T-1, then T-2, then T-3. No stacked untested changes.
- **A null moat is a result — record it and stop.** Do not grow the net, add RL, or change the
  target to make the connectome win.
- **No new heavy dependencies.** `torch` is already in `.venv`. Reuse the reservoir state
  recorder, `mn_to_joint.py`, `vnc_stepper.py`, the body-metric + bootstrap machinery,
  `preflight.py`. CPU only; a small MLP trains in minutes. Hours = a bug; stop and diagnose.

## 10. One-line summary
Train the smallest possible component — a memoryless, capacity-limited, backprop MLP that maps
the **frozen** VNC state to 18 joint targets, behavior-cloned to the clean-clock gait, no
body-outcome in the loss — and deploy it closed-loop under **identical** training on
`connectome` vs `random` vs `rewired` substrates (plus a `static-input` leak check). The moat
holds only if `connectome` walks a tripod and **beats random/rewired** on the body; a tie is
`RUNG1-MOAT-NULL` and the honest end of the "connectome drives the body" line. No final claim.
