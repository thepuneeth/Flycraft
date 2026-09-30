# EMBODY v5 — minimum crutches

**Date:** 2026-09-05
**Status:** plan. Nothing here has been run. Supersedes the "just train longer"
posture of `EMBODY_V4_SPEC.md`; that document keeps the run-1 and run-2
postmortems, which are the evidence this plan is built on.

## Why the direction changed

v4 run 2 was the first run in this project where PPO demonstrably worked. It
still did not walk, and the way it failed is the reason for this document.

| | ride height | joints off stance | landings/s | upright | speed |
|---|---:|---:|---:|---:|---:|
| CEM teacher (walks 0.424 m) | 0.148 m | **0.103 rad** | 24.5 | — | **0.106 m/s** |
| Stage 1 only (behaviour cloned) | 0.144 m | 0.170 rad | 36.1 | 8.1 s | 0.034 m/s |
| run 1 (20.3M steps, optimiser broken) | 0.143 m | 0.175 rad | 35.5 | 5.8 s | — |
| run 2 (24.9M steps, optimiser working) | 0.165 m | **0.636 rad** | 28.4 | 12.0 s | 0.026 m/s |

Run 2 stopped falling entirely and survived every episode to the 12 s limit. It
did that by **bracing** — folding into a splayed posture whose joints sit 3.7x
further from the home stance than any other policy and 6x further than the
teacher's, at a ride height *above* nominal standing. It is statically stable
because it cannot fall, not because it learned to balance. Instantaneous speed
went *down* relative to the behaviour-cloned prior it started from, 0.034 to
0.026 m/s, because falling ends the episode and forfeits all future return, so
caution pays better than progress.

**This is the third reward hack in this project.** v2 dove, v3 rocked, v4 braces.
It survived because `upright` scores a flat 1.0 for any height above 0.13 m and
nothing anywhere in the reward penalises joint deviation from the home stance —
so a 0.636 rad contortion and a clean stance are worth exactly the same. The
per-iteration metrics all improved while the behaviour got worse, and the failure
was caught by watching the video, not by the instrumentation.

### The plateau, quantified

`track_lin` slope per 1,000 iterations, by window, over 2,025 iterations:

| iterations | speed slope | EMA slope | reach rate |
|---|---:|---:|---:|
| 6–400 | 0.0051 | 0.0350 | 5.3% |
| 400–800 | **0.0554** | 0.0623 | 10.2% |
| 800–1200 | 0.0217 | 0.0242 | 21.2% |
| 1200–1600 | 0.0180 | 0.0018 | 23.8% |
| 1600–2000 | 0.0091 | **−0.0016** | 36.7% |

The slope halves roughly every 400 iterations and the return has gone flat to
slightly negative. Extrapolating the trailing-600 slope, teacher-level
`track_lin` (0.398) needs **27,600 more iterations**. The run had ~1,000 left.
More PPO on this setup does not reach walking, and this is not close.

### What that does and does not license

It does **not** license "the frozen connectome cannot walk a hexapod". Two facts
block that claim:

1. **The body is not the limit.** The CEM teacher walks 0.106 m/s on the identical
   body, PD gains, action space and control rate.
2. **The interface can express the gait, but the shipped prior never learned it.**
   An earlier version of this section claimed Stage 1 fit the teacher to MSE
   2.8e-4. That number was wrong and load-bearing, so it is worth naming
   precisely: `2.8e-4` is `value_loss` from `embody_3way_v2.json` — the **PPO
   critic's** regression loss — not a behaviour-cloning fit. The real Stage-1
   numbers, from `embody_v4_stage1.json`, are train **0.0824** / held-out
   **0.0394** after **158 epochs**.

   Those are worse than doing nothing. Predicting one constant action for the
   whole gait scores **0.0151**, so the shipped prior was about 5x worse than a
   constant — it had not learned the average pose, let alone the gait. Every run
   so far (run 1, run 2, all of DAgger) started from it.

   The capacity claim still holds, but on different evidence: an overfit probe on
   a single command reached **0.00611** by epoch 400 against a **0.01109**
   constant floor for that demo, and was still descending. The 4-latent
   anatomical map *can* represent the gait. Stage 1 was simply stopped at 158
   gradient steps with held-out loss still falling — one epoch is one full-batch
   step, so the epoch count *is* the gradient-step count, and `min_delta` 1e-4
   read a real descent as a plateau.

So the honest reading of run 2 is that **the training procedure is the binding
constraint, not the connectome** — and every fix below is aimed at the
procedure. Stage 1 now runs a 3000-epoch budget at `min_delta` 1e-6, and a gate
refuses to start PPO unless held-out beats half the constant-action baseline.

## The principle for crutches

The moat is a **non-fly hexapod body driven by frozen, measured connectome
weights**. A crutch that makes learning the *interface* easier is fine. A crutch
that *replaces the connectome's computation* destroys the result, because the
walking would then come from the crutch and the connectome would be decoration.

Every assist below is therefore graded against one question: **after this
change, is the connectome still the thing generating the motor pattern?**

## What the field actually does

Read out of the reference implementations already vendored in `fly_field_refs/`,
not from memory.

### flybody — `flybody/tasks/walk_imitation.py` (Vaxenburg et al., Nature 643:1312, 2025)

| assist | code |
|---|---|
| **Reference state initialization** — episodes start *inside* a reference pose | `physics.bind(self._mocap_joints).qpos = self._ref_qpos[0, :]`, and reference `qvel` too when `_initialize_qvel` |
| **DeepMimic tracking reward** — reward is *matching a reference*, not "go fast" | `reward_factors_deep_mimic(..., weights=(20, 1, 1, 1))`, four Gaussian factors over COM, qvel, egocentric end-effector vectors, joint quaternions |
| **Reference lookahead in the observation** | `_future_steps`, plus `ref_displacement` and `ref_root_quat` observables |
| **Aggressive early termination** | terminal on `linvel`, `angvel`, and `com_dist` from the reference |
| **Multiplicative reward factors** | `get_reward_factors` returns factors that multiply |
| **Short snippets from a clip library** | `self._traj_generator.get_trajectory(...)` |
| **Tethered variant** — balance removed entirely | `walk_on_ball.py` |

### NeuroMechFly 2 — `flygym`, `nmf2-paper/leg_adhesion`

| assist | code |
|---|---|
| **Leg adhesion — sticky feet** | `enable_adhesion=True, adhesion_gain=20` |
| **Stabilization period before control** | `nmf.run_stabilisation(...)`, `stab_dur = 0.2` |
| **CPG generates the gait; the policy supplies 2 numbers** | `TurningController`, `action = [1.0, 1.0]` |
| **Posed initial state** | `stretched_pose` |

The NMF2 turning controller is the sharpest lesson here. It reduces the whole
control problem to a **two-dimensional descending command on top of a central
pattern generator** — structurally the same shape as our 4-latent descending
map, except their oscillator produces the rhythm and we are asking the
connectome to produce it. That is exactly the line we must not cross.

## The plan

### Tier 0 — the task was not asking for walking (done, 2026-09-05)

Found by the owner watching run 2's video: a leg extending to the ball rather
than the body walking to it. Measured, the geometry was worse than it looked.
The feet sit **0.259-0.284 m** from the body centre at stance while
`TARGET_REACH_M` was **0.15 m**, so the footprint was wider than the reach zone —
the sphere could sit between the legs with the body barely moved. A target at the
0.4 m inner edge required only 0.25 m of body travel, less than one footprint
diameter, and the prey is a ground-level sphere (radius 0.07 at z=0.07), so leg
extension beat locomotion for the last stretch.

| | before | after |
|---|---:|---:|
| spawn annulus | 0.4–1.0 m | **1.0–1.8 m** |
| reach radius | 0.15 m | 0.20 m |
| minimum body travel | 0.25 m | **0.80 m**, 2.8x the footprint |
| episode | 600 steps / 12 s | **1500 steps / 30 s** |

Episode length is free — PPO segments are `PPO_ROLLOUT_STEPS` regardless, so this
changes reset frequency, not iteration cost. 30 s covers 3.2 m at the teacher's
0.106 m/s, enough for the far edge plus a respawn. Locked by
`test_the_target_cannot_be_reached_without_walking`.

Confirmed by eye as well as by arithmetic, since the arithmetic is what missed
this the first time. The CEM teacher was driven straight at a fixed target at
each edge of the annulus and filmed (`results/path2/target_geometry_{near,far}.mp4`):

| spawn | body travel to reach | time | speed |
|---:|---:|---:|---:|
| 1.0 m (inner) | 0.918 m | 8.0 s | 0.106 m/s |
| 1.8 m (outer) | 1.768 m | 16.9 s | 0.106 m/s |

Both reach, so the task is still solvable by something that walks, with 13 s of
the 30 s episode to spare at the far edge. In frame one of the near case the ball
sits about two body widths beyond the outermost leg tip, so leg extension cannot
touch it from the spawn pose. The reach counter, not the observed distance, is
the honest reach signal here -- `step` respawns the target on contact, so a
post-respawn distance never reads as a reach.

**This invalidates run 2's reach counts.** Every `targets_reached` number in the
run-2 log was scored under a geometry that did not require walking.

### Tier 1 — safe. No capacity added, all standard practice.

1. **Fix the brace hack first.** This is a bug, not a crutch. Add a joint-deviation
   penalty so a 0.636 rad contortion cannot score like a clean stance, and make
   `upright` discriminate above 0.13 m instead of saturating. Re-run the
   real-rollout ordering test: the teacher must still beat everything.
2. **DAgger for Stage 1.** Roll out the current policy, relabel the states *it*
   visits with teacher actions, refit, repeat. Attacks the 0.106 → 0.034
   compounding-error gap directly, needs no pod, and is the single highest-value
   item on this list.
3. **Reference state initialization.** Start episodes from states sampled along
   the teacher trajectory rather than always from home stance. flybody's exact
   mechanism. Pure exploration aid; the policy still has to produce the actions.
4. **DeepMimic-style tracking reward against the teacher trajectory**, alongside
   or in place of the `[LG]` velocity terms. This is what flybody optimises and
   it is far denser than "track 0.1 m/s".
5. **Early termination on reference deviation**, as flybody terminates on
   `com_dist`.
6. **Target-distance curriculum.** Start the spawn annulus tight, widen it.
7. **Stabilization period** before control is handed over, as NMF2 does.

### Tier 2 — physical assist. Declare it, prefer to anneal it away.

8. **Leg adhesion.** NMF2 needs it for a fly and ships it on by default. Our body
   is a PhantomX, not a fly, so this is a deviation from our own body spec and
   must be declared in the methods. Anneal the gain to zero if the final claim is
   to be about unassisted walking.
9. **Annealed trunk stabilization** — a small assistive force early, annealed to
   zero. Standard in humanoid RL. Only acceptable if it reaches exactly zero
   before the reported result.

### Tier 3 — forbidden. These end the project's novelty.

10. **A CPG or pattern generator outside the connectome.** The rhythm is the thing
    we are asking the VNC to produce. Adding an oscillator answers our own
    research question for us.
11. **Residual policy on top of the teacher gait.** Same objection: the gait would
    come from CEM, not from the fly.
12. **Widening the descending interface** beyond the fixed 4-latent anatomical
    map, or making the decoder deep. Any connectivity result becomes an interface
    result.
13. **Unfreezing the core.** Never. This is the whole point.

## Order of work

1. Tier 1.1 (brace fix) and 1.2 (DAgger) — local, free. **Decision gate:** if
   DAgger lifts closed-loop speed materially toward 0.106 m/s, the interface was
   never the limit and v5 proceeds. If a good fit still leaves it at ~0.034 m/s,
   then the 4-latent anatomical map genuinely cannot drive this gait closed-loop,
   and *that* is a real connectome finding to write up rather than engineer around.
2. Tier 1.3–1.7 — implement, verify with the existing 21-test suite plus new
   tests per change.
3. One pod run under the locked recipe.
4. Tier 2 only if 1 and 3 leave it short, and only declared.
5. Controls (random weights, rewired) under whatever recipe is locked — the
   result is the three-arm comparison, never the connectome arm alone.

## Budget

Balance $8.74 after run 2 ($1.41 spent). Tier 1.1 and 1.2 cost nothing. A v5 pod
run costs ~$2 at the measured 3070 rate; the two control arms cost ~$2 each. The
measured bottleneck is **single-thread CPU**, not the GPU — run 2 sat at 100% of
one core with the GPU near-idle between kernels, because the vector env steps 512
MuJoCo instances in a serial Python loop. Do not buy a bigger GPU.

## Instrumentation debt

Run 2's failure was invisible to every logged scalar. Before the next run, the
per-iteration log must carry **joint deviation from stance** and **ride height**,
and evaluation must report them over the 12 fixed seeds. A metric set that cannot
distinguish a gait from a brace is not an evaluation.
