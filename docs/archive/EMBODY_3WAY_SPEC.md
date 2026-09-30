# Embodiment training — 3-way control spec (agent instructions)

**Read this whole file before writing code.** Context lives in
`FIELD_CODE_REVIEW.md` (§4 the honest build, §5 the checklist) and
`NEXT_EXPERIMENTS.md` (standing constraints). This document specifies the
*decisive* experiment only: whether the measured connectome earns its place
once a trainable interface is allowed.

---

## 0. The question this test answers

We are training a body-interface (sensory encoder + motor decoder) around a
**frozen** measured-VNC core so a hexapod walks forward. The interface is
trainable; the connectome is not. The scientific question is **not** "does the
hexapod walk" — a big enough trainable interface will make almost anything
walk. The question is:

> Under an **identical** training budget, architecture, seed protocol, and
> reward, does the **real** VNC core reach better forward locomotion than an
> **amplitude-matched random** core and a **degree-preserving rewired** core?

If real ≈ scrambled, the connectome is decorative and the honest verdict is a
null. That is a publishable result and must be reported as such. Four prior
training campaigns in this repo (Outcome C, R0–R3, Rung 1, Outcome D) all
returned nulls; expect the same and be pleasantly surprised, not disappointed.

**Do not run real-to-completion first.** Run all three arms concurrently at a
small budget. The comparison is the deliverable; a single well-trained real arm
is worth nothing on its own.

---

## 1. Hard constraints (violating any of these invalidates the run)

1. **The VNC core weights are frozen.** `vnc.W` is never an `nn.Parameter`,
   never appears in an optimizer's `param_groups`, never receives a gradient.
   Assert `requires_grad is False` on every core tensor at construction and
   again after training.
2. **Hash the graph before and after.** Use
   `specificity_campaign.tensor_sha256(vnc.W)` and
   `specificity_campaign.file_hashes(VNC_DATA)`. Record pre and post values in
   the result JSON. A mismatch is a hard failure, not a warning.
3. **All three arms are identical except the core.** Same interface
   architecture, same parameter count, same optimizer, same learning rate, same
   step budget, same reward, same seed list, same env, same episode length.
   Any per-arm tuning invalidates the comparison.
4. **The random arm is amplitude-matched.** A raw random core with different
   activity scale is not a control — the previous campaign's central mistake.
   See §3.
5. **No body outcome may leak into interface selection** beyond the declared
   reward. No hand-picking the best checkpoint by eye; the selection rule is
   fixed in advance (§5).
6. **Pre-register before running.** Write the prereg JSON first (§6), then
   train. Do not edit it afterward.
7. **Do not commit.** Leave artifacts in the working tree.

---

## 2. Existing pieces to reuse (do not reimplement)

| Need | Use |
|---|---|
| hexapod env, reward, fall gate | `eon-fly-brain/code/embody_env.py` |
| frozen core + encoder/decoder shim | `eon-fly-brain/code/embody_policy.py` |
| capacity check for the interface | `eon-fly-brain/code/embody_overfit.py` |
| real / random / rewired core factory | `rung1_readout.condition_stepper(data, condition)` |
| degree-preserving rewire, weight randomize | the helpers `condition_stepper` already calls |
| amplitude matching | `audit_amplitude_control.amplitude_match(history, target_sd)` |
| graph + file hashing, atomic JSON | `specificity_campaign.{tensor_sha256,file_hashes,atomic_json}` |
| paired sign-flip test, bootstrap CI | `audit_t2_forward_readout.{one_sided_sign_flip,paired_forward,metric_summary}` |
| forward gait reference | `hexapod_gait.HexapodGait.targets(0.8, 0.0, dt)` |

Verify each of these still exists and its signature matches before relying on
it. If `embody_env.py` / `embody_policy.py` / `embody_overfit.py` were written
by another agent, **run their `__main__` self-checks first** and report the
result; do not build on unverified scaffolding.

---

## 3. The three arms

Build all three with `condition_stepper(VNC_DATA, condition)`:

- **`connectome`** — the measured MANC VNC weights, untouched.
- **`random`** — weights randomized, then **amplitude-matched**: the core's
  motor-output history must be rescaled per channel so its temporal SD equals
  the real core's per-channel SD, mean and temporal pattern preserved
  (`amplitude_match`). Compute the real per-channel SD once, from the real
  core's rollout under the same drive protocol, before any training.
- **`rewired`** — degree-preserving rewire, amplitude-matched the same way.

Record the amplitude-match diagnostics (`max_sd_error_all_channels`,
`mean_preserved_max_error`) in the result JSON. If any channel cannot be
matched (zero-variance source), exclude it and say so explicitly in the
report — the earlier pilot's unmatchable-channel problem must not recur
silently.

Add a fourth diagnostic arm if cheap: **`static-input`**, the real core's
train-split mean state repeated. It should perform near chance. If it does
not, the pipeline has a leak and the whole run is void.

---

## 4. Training protocol

**Stage 1 — imitation (cheap, run first).**
- Target: forward locomotion reference. Two options; **use option B** and note
  option A as the fallback.
  - *Option A (easy):* clone `HexapodGait.targets(0.8, 0.0, dt)` — hands the
    gait to the interface directly.
  - *Option B (paper-faithful, preferred):* target **forward body
    displacement** only (a constant-speed forward path, in the spirit of
    flybody's `constant_speed_trajectory`); the interface and core must
    produce the joint pattern themselves.
- Behavior-clone / supervised fit of encoder+decoder only. Core frozen.
- Same epoch cap and LR for all arms.
- Record final train and held-out loss per arm.

**Stage 2 — PPO, budgeted.**
- **5M environment steps per arm. Hard cap. Do not extend one arm.**
- Single-env, on-policy, following FlyGM's Stage-2 shape (γ=.99, clip=.2,
  ent=.01, vf=.5, lr=1e-4). Value-warmup before unfreezing the policy.
- Checkpoint on a fixed interval; log EMA return.
- **Seeds: K=12 per arm** if compute allows; K=5 minimum. Seeds are the unit of
  analysis — a single seed per arm proves nothing.

If 5M steps per arm × 3 arms is too expensive, reduce the step budget, never
the number of arms or seeds.

---

## 5. Endpoints and statistics (fix these before running)

**Primary endpoint:** mean forward displacement (m) per evaluation episode, on
held-out seeds, at the fixed final checkpoint.

**Selection rule (declare in advance):** last checkpoint, or best-by-EMA-return
on a *training* signal — never by the evaluation forward distance.

**Tests:**
- Paired across seeds: `connectome − random`, and `connectome − rewired`.
- One-sided sign-flip exact test per contrast.
- **Holm correction across the two contrasts.**
- Bootstrap 95% CI on each paired difference.
- Report `n/K` — how many control seeds match or beat the real arm.

**Secondary (report, do not promote):** sample efficiency (steps to reach a
displacement threshold), final EMA return, interface parameter count needed.

**Pre-registered null:** *the connectome arm ties or loses to the
amplitude-matched random arm under identical training → the structural
advantage does not survive a trainable interface; the moat is
zero-training-only.* Write this string into the prereg file verbatim.

**Decision labels** (use exactly one in the result JSON):
- `EMBODY-MOAT-EARNED` — connectome beats **both** controls, Holm-corrected,
  CI excludes zero.
- `EMBODY-MOAT-NULL` — ties or loses on either contrast.
- `EMBODY-NO-WALK` — no arm achieves stable forward locomotion; the comparison
  is undefined and nothing may be claimed either way.

---

## 6. Artifacts to produce

Under `eon-fly-brain/results/path2/`, prefix `embody_3way`:

1. `embody_3way_preregistered.json` — written **before** training: conditions,
   architecture, parameter count, optimizer, LR, step budget, seed list,
   endpoints, selection rule, pre-registered null, decision labels.
2. `embody_3way.json` — results: per-arm per-seed displacement, the two paired
   contrasts with CI + Holm p, `n/K`, decision label, amplitude-match
   diagnostics, graph hashes pre/post, `vnc_weights_trained: false`, wall-time
   and step counts per arm.
3. `embody_3way.csv` — per-seed rows, one row per (arm, seed, episode).
4. One **watched** multi-view video of the `connectome` arm's final policy,
   plus one for `random`. Actually watch them; report what the bodies do,
   including failure modes (dragging, hopping, standing still while sliding).
5. A short `RUN_EMBODY_3WAY.md` brief: what ran, what the numbers are, the
   decision label, and the honest limitations.

---

## 7. Reporting rules

- Report the null plainly if it is a null. Do not soften it, do not lead with
  the walking demo, do not describe a tie as "comparable performance."
- If the connectome wins, state the effect size and CI before any
  interpretation, and explicitly name what the result does **not** show
  (it is forward drive under one training regime, not coordination, not
  general control).
- Quote exact error lines for anything that failed.
- **No final claim without owner approval.** Present the numbers and the
  decision label; stop there.

---

## 8. Order of operations

1. Run the three scaffold self-checks (`embody_env.py`, `embody_policy.py`,
   `embody_overfit.py`); report pass/fail with the exact commands. **Stop and
   report if the overfit test does not drive loss toward zero** — the interface
   lacks capacity and nothing downstream is meaningful.
2. Build and hash the three cores; compute the real per-channel SD; verify
   amplitude matching. Report the diagnostics.
3. Write the prereg JSON.
4. Stage 1 imitation, all three arms, identical config. Report per-arm loss.
5. **Checkpoint with the owner before spending Stage-2 GPU budget.**
6. Stage 2 PPO, 5M steps per arm, all seeds.
7. Evaluate, test, label, write artifacts, watch videos, write the brief.
