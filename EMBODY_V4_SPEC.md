# EMBODY v4 — goal-directed locomotion, copied from the field

**Date:** 2026-09-03
**Status:** first Stage-2 run completed, 2026-09-04: 7.0 h, 264 iterations,
20.3M steps on an RTX 3070. **It did not train, and the reason was the device.**
See *Run 1 postmortem*. The pipeline is otherwise audited against the source
implementations and covered by a 14-check pre-flight suite
(`eon-fly-brain/code/test_embody_v4.py`, 14/14 on CPU).
**Task:** a frozen, untrained FlyWire/MANC VNC (11,000 neurons, 485,074 edges)
drives a non-fly 18-DoF hexapod to walk to a target object in MuJoCo. Only the
body interface trains. `core_parameters: 0`.

## Why there was no walking connectome

Three separate defects, all of them mine, all upstream of the connectome. None of
the three prior nulls measured what they claimed to measure.

### 1. The teacher never walked (largest)

Stage 1 fits the interface by imitating a CEM-selected periodic gait. v2 and v3
ran that search at population 24 for 3 rounds and shipped whatever it found.
Measured on the straight command over 200 steps:

| CEM budget | teacher path MSE | distance in 4 s (reference 0.400 m) |
|---|---:|---:|
| population 24, 3 rounds (v2/v3, shipped) | 1.25e-2 | 0.166 m |
| population 64, 12 rounds (v4) | **2.77e-4** | **0.424 m** |

v3's own recorded artifact agrees: `embody_3way_v3_stage1.json` has
`best_path_mse_m2 = 0.0117`. The demonstration every arm spent Stage 1 imitating
was a shuffle. The interface was initialised to shuffle before PPO ever started,
identically in all three conditions — so the "connectome rocks in place" finding
was partly the connectome faithfully reproducing what it was taught.

### 2. The teacher only taught straight-line walking

One command, 0.1 m/s forward. The encoder's four latent channels are
`(go, steer, family, walkdn)`, and `steer` was never given a reason to mean
anything. Steering is the whole of reaching a target.

### 3. The reward was invented here, twice, and gamed twice

- **v2**: unbounded forward displacement. Monotone in distance with no fixed
  point, so the most ballistic behaviour wins — the dive, 0.588 m/s for 0.52 s,
  12/12 falls.
- **v3**: bounded `tolerance` on *instantaneous* speed × upright. Oscillation
  crosses the target window on nearly every step while net travel cancels — the
  rock, 0.0504 m in 8 s, path/displacement 7.6×, 0 falls.

v3 was itself an attempt to copy flybody, and it copied the wrong half: the
*combination rule* (bounded factors, multiplied) without the *thing being
tracked* (a reference trajectory). A product of two factors that oscillation
satisfies is still satisfied by oscillation.

### And there was never a target

The reward has only ever scored forward locomotion. No object, no bearing, no
distance-to-goal term has existed in this pipeline at any version.

## The audit, second pass

The first build of v4 was reviewed line by line against the source
implementations before any pod time was spent. It contained eleven defects. All
are fixed; each now has a check in `test_embody_v4.py` that fails if it returns.

**Would have produced a meaningless run:**

1. **`targets_reached` was structurally always zero.** The vector wrapper
   auto-resets a terminal environment and `reset()` clears the reach counter, and
   the rollout length equalled the episode length, so every environment
   terminated on the final step. The primary endpoint would have read 0.000 for
   all 50M steps. The counter is now banked in the wrapper before the reset.
2. **648 gradient updates for the whole budget.** The rollout was a complete
   600-step episode with no minibatching: 162 iterations × 4 epochs. legged_gym
   reaches locomotion in ~1500 iterations × 5 epochs × 4 minibatches. Now
   24-step segments over persistent environments, 4 minibatches over the
   environment axis (the time axis has to stay contiguous for BPTT):
   **4,069 iterations, 81,380 updates.** The segment was 150 when this was
   written; see *Run 1 postmortem* for why it is 24.
3. **PPO replayed the rollout from a zero recurrent state.** Segments are shorter
   than episodes, so from the second iteration onwards a segment begins
   mid-episode. The recomputed means then differed from the behaviour means for a
   reason unrelated to the parameter update, inflating the KL, corrupting the
   importance ratio, and driving the learning rate down in response to a
   discrepancy no optimiser could fix. The segment's starting state is now
   carried in the rollout and replayed from.
4. **The value warm-up could never fire.** Its gate was 102,400 steps, which one
   iteration exceeded on its own. It now counts iterations.

**Fidelity to the sources:**

5. **The observation was raw `qpos`+`qvel`.** That carries three things
   `LeggedRobot.compute_observations` deliberately excludes: world XY, which
   drifts without bound past a linear encoder; absolute world yaw, which makes
   the policy non-equivariant so the body must relearn walking once per heading;
   and world-frame base velocity, so "forward" depends on where north is. None
   carries task information the egocentric target block does not already carry.
   Replaced with `[LG]`'s observation in its own order and with its own
   `obs_scales`, the target block occupying the `commands` slot. 52 → **66**.
   `test_observation_is_egocentric` asserts the observation is invariant to world
   rotation and translation; the v2/v3 observation fails it.
6. **Two `[LG]` reward terms with non-zero defaults were missing** — `torques`
   (−1e-5) and `dof_acc` (−2.5e-7), the smoothness regularisers. Without them
   nothing costs a buzzing high-torque policy that satisfies tracking by
   vibrating. Added at the source weights.
7. **No adaptive learning-rate schedule.** `schedule = 'adaptive'` is a large part
   of why legged_gym trains in minutes: rsl_rl re-tunes the rate every minibatch
   to hold the policy step at `desired_kl = 0.01`. v4 had inherited a fixed 1e-4.
   Ported, along with `use_clipped_value_loss` and `value_loss_coef = 1.0`.
8. **Yaw was computed as if the body were level** — `atan2(2·qw·qz, 1−2·qz²)`
   drops `qx, qy`, so it was wrong exactly when the hexapod tilts. Worse, the
   teacher used its own copy of the approximation, so the bearing Stage 1 was
   fitted on was not the bearing that had been commanded. One `env.heading()`
   now serves both.
9. **Two Adam optimizers held the same value parameters** with independent moment
   estimates. Removed; the main optimizer already owns them.
10. **Rollout seeds strided by 1**, so consecutive iterations redrew 511 of the
    same 512 seeds. Now strided by batch size.
11. **The BPTT depth was set by argument, not measurement.** v4 raised it 25 → 50
    on the reasoning that 25 control steps is 0.5 s while the gait period is
    0.83 s, so no stride could be credited whole. Measured on this core at
    `PPO_ROLLOUT_STEPS = 150` (as it was then), one full update, same batch:

    | BPTT chunk | gradient norm | update wall time |
    |---:|---:|---:|
    | 25 | **1.4e1** | 600 s |
    | 50 | 1.9e2 | 436 s |
    | 75 | 9.8e4 | 466 s |

    The measurement overrules the argument. A gradient of 1e5 clipped to norm 1
    is a unit vector pointing wherever the exploding path points, which credits
    nothing at all. Credit still crosses chunk boundaries — through the value
    function and GAE, not through BPTT. That is how the source handles it:
    `[LG]` runs `num_steps_per_env = 24`, shorter than this chunk. Back to 25 —
    and then to **24**, one whole segment, once the postmortem set the segment
    length to `[LG]`'s 24 as well.

### Two measured deviations from rsl_rl

Both are documented at the constant, and both were found by measurement rather
than assumed.

- **Initial learning rate 1e-5, not rsl_rl's 1e-3.** One update at 1e-3 moves the
  policy to KL ≈ 1.0 — a hundred times the trust region, and enough to destroy
  the Stage-1 behaviour-cloned prior on the very first update, before the
  controller has seen anything to react to. The schedule raises the rate by 1.5×
  whenever KL comes in under half the target, so starting low costs a few
  iterations of ramp and starting high costs the prior. At 1e-5 the measured
  first update is **KL 0.0083** against a target of 0.01, `mean_ratio` 0.998,
  gradient norm 1.47.
- **Learning-rate floor 1e-8, not rsl_rl's 1e-5.** The encoder feeds a sigmoid
  scaled by 100 into an 11,000-node recurrent core, so a 1e-5 change in an
  interface weight is not a small change in the policy. Measured at the 24-step
  segment, the KL after one step is 1.6e-2 at 1e-5 and 1.8e-4 at both 1e-8 and
  1e-9: below 1e-8 the rate stops mattering. This floor was 1e-9, which is how
  run 1 was able to wind down forever instead of failing loudly; the three
  decades below rsl_rl are the deviation and they are now bounded.

- **An update is abandoned once KL exceeds 50× the target.** `[LG]` has no such
  guard. The adaptive schedule is reactive — it shrinks the next step and cannot
  undo the last one — so without this a single overshooting minibatch leaves the
  other 19 optimising against a policy the parameters no longer resemble.

## Run 1 postmortem — 2026-09-04

Pod `73wareeh9ar4jy`, RTX 3070, $0.130/hr, $1.00 spent. Pre-flight 14/14 on the
pod, Stage 1 and the smoke gate both passed, Stage 2 ran its full 7 h wall-clock
budget and checkpointed cleanly. Evaluation failed separately on a missing
OpenGL context (headless pod, no EGL libraries installed) — a packaging problem,
not a training one.

**The training did nothing.**

| iteration | segment return | targets reached | KL | learning rate | grad norm |
|---:|---:|---:|---:|---:|---:|
| 6 (first PPO) | 1.700 | 0.000 | 10.57 | 3.0e-9 | 77 |
| 55 | 1.700 | 0.000 | 7.90 | 1.0e-9 | 188 |
| 135 | 1.703 | 0.000 | 7.41 | 1.0e-9 | 705 |
| 264 (last) | 1.700 | 0.000 | 8.09 | 1.0e-9 | 75 |

The EMA return moved from 1.665 to 1.712 across 20.3M steps and the policy
entropy was bit-identical between iterations 200 and 264, i.e. `log_std` never
moved. Highest `targets_reached` in the whole run was 0.002.

The signature is the adaptive learning rate on its 1e-9 floor from the *first*
PPO iteration while KL sits near 8. At 1e-9 across 20 minibatches the parameters
move by ~2e-8 in total, so a KL of 8 cannot have been caused by the updates. The
replayed policy was simply not the behaviour policy — which makes both the
importance ratio and the KL controller measure a discrepancy rather than an
update, and pins the rate at the floor forever. Everything downstream was
decorative.

**Reproduced on CPU in five minutes.** `run_embody_v4.py smoke` on this laptop
returns `kl 10.97, learning_rate 6.8e-9, mean_ratio 0.71` — the pod's signature
exactly. Nothing about the failure was CUDA-specific, and the CUDA hypothesis
first recorded here was wrong.

The mistake behind that hypothesis was measuring the wrong thing. Replay under
*unchanged* parameters is bit-exact — 0.0 at 150 steps, under batch-width
slicing, with checkpointing on and off — which is why the first diagnosis
concluded the local machine was innocent. But PPO never replays under unchanged
parameters after the first minibatch. Tracing the KL per minibatch through a
single update:

| minibatch | KL | learning rate | mean ratio |
|---:|---:|---:|---:|
| 0 | 0.00018 | 1.5e-5 | 1.0000 |
| 1 | 11.89 | 1.0e-5 | 1.3194 |
| 2 | 11.96 | 6.7e-6 | 0.6303 |
| … | ~11 | winding down | ~0.65 |
| 19 | 10.53 | 6.8e-9 | 0.5858 |

**One Adam step at 1e-5 takes the KL from 0.0002 to 11.9.** The adaptive schedule
is reactive: it can shrink the *next* step but cannot undo the one that has
already left the trust region, so the remaining 19 minibatches optimise a ratio
computed against a policy the parameters no longer resemble, and the rate winds
to its floor and stays there. Every subsequent iteration repeats it.

The cause is the replay horizon, not the learning rate. Sweeping the rate at the
150-step horizon, the KL after one step is 10.6 at 1e-5 and still 6.87 at 1e-8 —
a floor no rate can get under, because the policy contains an 11,000-node
recurrent core with a gain of 100 on its input, and it amplifies any perturbation
exponentially along the segment. Sweeping the horizon instead, at a fixed 1e-5:

| replay horizon | 1 | 12 | 24 | 32 | 48 | 75 | 150 |
|---|---:|---:|---:|---:|---:|---:|---:|
| KL after one step | 1.8e-4 | 2.0e-4 | **1.5e-2** | 2.5e-1 | 1.35 | 4.62 | 10.3 |

The trust region is 0.01. `PPO_ROLLOUT_STEPS = 150` was roughly 700x outside it
before a single gradient was applied. 24 — which is [LG]'s `num_steps_per_env`,
the value the v4 comments had argued their way out of — lands at 0.0146.

**The fixes.**

- `PPO_ROLLOUT_STEPS` 150 → 24, `PPO_BPTT_CHUNK` 25 → 24. Budget arithmetic
  follows: 4,069 iterations and 81,380 gradient updates for the same 50M steps,
  against legged_gym's ~30,000. Measured after the change, the worst KL across
  all 20 minibatches of one update is 0.064, and `mean_ratio` is 0.984.
- `PPO_LEARNING_RATE_MIN` 1e-9 → 1e-8. Below 1e-8 the rate has no measurable
  effect on the policy step, so a lower floor only lets the controller hide a
  horizon bug by winding down forever.
- **Collection and replay now step the core at the same batch width.** Collection
  ran the whole batch at 512 and the update replayed each minibatch at 128. The
  core's columns are independent so this is exact in algebra, but a sparse matmul
  kernel picks its reduction order from the operand width, so it is not exact in
  floating point. Both paths now use one fixed contiguous partition — which is
  also what rsl_rl's *recurrent* minibatch generator does, so this moves toward
  the source rather than away from it. Not the cause of run 1; a second way to
  produce the same symptom, removed.
- **An update that leaves the trust region is abandoned rather than continued**
  (`PPO_KL_ABORT_FACTOR`). A declared deviation from [LG], which has no such
  guard. Run 1 spent roughly 13,000 updates in the state this stops at one.
- **GAE bootstraps through the time limit.** `_done` merged "fell over" and "the
  clock ran out"; only the first is a genuine termination, and every episode here
  ends on the clock. The env now reports `_timeout` separately and the vector
  wrapper banks the pre-reset observation to bootstrap from.

**Why the pre-flight missed it.** `test_replay_reproduces_the_behaviour_policy`
existed and passed on the pod. It ran under `torch.no_grad()`, on a six-step
rollout, on the full batch. Strengthening it to the real rollout length, real
slicing and real gradients — done immediately after run 1 — did not help either,
and would not have: it still compared a replay under unchanged parameters, which
is bit-exact at *any* horizon and therefore says nothing about this failure. A
check that passes on a machine where training is already broken is worse than no
check, because it buys confidence.

The check that does catch it is
`test_the_rollout_is_short_enough_that_a_step_stays_in_the_trust_region`, which
takes a real gradient step and asserts the resulting KL is inside the trust
region. `replay_fidelity()` is kept in the smoke gate as the cheaper necessary
condition, with its docstring corrected to say what it does not prove.

### The second blocker: the reward barely preferred walking

Fixing the optimiser exposed the next problem. With the optimiser working, the
per-term log showed the arm standing still rather than walking, so the reward was
scored directly against a policy that provably walks — the CEM teacher, replayed
open-loop, covering 0.424 m at 0.106 m/s against a 0.100 m/s command — and against
the same body holding its stance.

| | reward/step | ratio |
|---|---:|---:|
| teacher gait (0.424 m) | 0.02253 | |
| holding stance, motionless | 0.01825 | **1.23x** |

At that contrast PPO finds standing still, which is the v3 rocker again by another
route. Two specification errors, both found by measurement, neither of them a new
invention — each restores a rule the file had already stated and then not followed.

**1. The `[FB]` ride-height factor was an additive term inside an `[LG]` sum.**
This file's own header says `[FB]` multiplies bounded factors while `[LG]` sums
weighted terms, and that v4 uses the `[LG]` sum. `upright: 0.5` went into that sum
anyway. A multiplicative factor can only ever reduce a reward; an additive one pays
for existing, and it was contributing 55% of everything a motionless policy earned.
Restored to a gate on the summed reward, it keeps the property it was kept for — it
still collapses the reward as the body sinks, which is what stopped the v2 dive, and
the posed v2 dive now scores exactly 0.000 — while paying nothing for standing up.
Ratio 1.23 → **1.52**.

**2. The angular tracking sigma was never rescaled.** v4 rescaled
`tracking_sigma` for the linear term by the square of the command-magnitude ratio,
reasoning that `[LG]`'s 0.25 is tuned for ~1 m/s commands and ours is 0.1 m/s, and
that without the rescale "a policy standing still would score 0.96". It then wrote
*"Yaw-rate commands stay on `[LG]`'s own scale (order 1 rad/s), so sigma is
theirs."* That is wrong. The heading law emits `clip(0.5 * bearing, ±1)` and
`teacher_v4`'s widest command is 0.4 rad/s. Measured, this was the single largest
free lunch in the reward: a body holding its stance, never turning, scored **0.790**
on the angular tracking term — **96% of everything a motionless policy earned** —
because at sigma 0.25 a 0.24 rad/s command sits well inside the tolerance, so not
turning reads as tracking. Rescaled by the same rule as the linear term
(`0.25 * (0.4/1.0)^2 = 0.04`), the same body scores 0.229. Ratio 1.52 → **3.53**.

| | reward/step |
|---|---:|
| teacher gait | 0.00931 |
| holding stance | 0.00264 |
| **ratio** | **3.53x** |

Both are now locked by `test_a_real_gait_outscores_the_shuffle_it_has_to_beat`,
which replays the real CEM teacher rather than posing a body and injecting a
velocity. The posed-body checks in `smoke_test` could not see this: they set the
bearing to zero, where a motionless body genuinely *is* tracking a zero yaw
command, so standing scored legitimately and the free lunch stayed invisible.

**`FEET_AIR_TIME_THRESHOLD_S` was checked and left alone.** Its `ponytail:` comment
asked for a retune from realised duty factor. Measured on the teacher: 49 landings
over 200 steps, air time median 0.120 s, p90 0.400 s, mean 0.204 s. The 0.2 s
threshold sits at the mean, making the term a mild pressure toward longer strides —
which is exactly `[LG]`'s intent for it at 0.5 s on ANYmal. No change.

### What run 1 does and does not say about the connectome

**Nothing.** No claim about the frozen VNC is available from it. The core stayed
frozen throughout (hashes equal, `core_parameters: 0`), Stage 1 fit the interface
to the command-conditioned teacher as designed, and the 20.3M steps that followed
applied essentially no gradient. This is a null about the run, not a result.

### Also fixed after run 1

- **Evaluation needs a GL backend on a headless pod.** `MUJOCO_GL=egl` is not
  enough; the image needs `libegl1 libglvnd0 libgles2` installed, or evaluation
  should be run locally from the downloaded checkpoint.
- **Community-cloud SSH is unreliable for this workflow.** Two pods lost direct
  SSH within minutes; both shared a host IP whose sshd appears to rate-limit.
  The RunPod proxy (`<podHostId>@ssh.runpod.io`) works but forces a PTY, ignores
  command arguments, and corrupts long input lines — drive it from stdin with
  short lines. `runpodctl send` / `receive` moves artifacts without SSH and is
  the reliable retrieval path.

## Run 2 — launched 2026-09-05

Pod `mt8vuwhxrq6kgf`, RTX 3070, community cloud, $0.130/hr, 16 vCPU (13.6 by
cgroup quota), 31 GB. `V4_MAX_HOURS=15`, everything else at the checked-in
defaults. Balance at launch $10.15.

What is different from run 1, in the order the problems were found:

1. `PPO_ROLLOUT_STEPS` 150 → 24. Run 1's optimiser never applied a gradient; see
   *Run 1 postmortem*.
2. The reward now prefers a real gait 3.53x over a motionless body, against 1.23x
   in run 1; see *The second blocker*.
3. The log carries `target_distance_m` and the unweighted tracking, air-time and
   upright terms per iteration, so progress is legible before `targets_reached`
   leaves zero — which in run 1 it never did.
4. Stage 2 stops itself on run 1's signature (rate at its floor, KL outside the
   region, 50 consecutive iterations) and raises rather than going on to evaluate
   a policy that never trained.

Expected: ~15.3 s/iteration, so ~3,500 of 4,069 iterations in the budget, about
$2.00. The full step cap needs 17.3 h.

Reference points to judge the morning against, both measured on this machine:

| | value |
|---|---|
| teacher gait, replayed open loop | 0.424 m at 0.106 m/s, 0.00931 reward/step |
| holding stance, motionless | 0.00264 reward/step |
| run 1 checkpoint, 12 fixed seeds | 0/12 reached, closest 0.71 m, path/displacement 8.15, 7/12 falls |

## Run 2 postmortem — 2026-09-05

Pod `mt8vuwhxrq6kgf`, RTX 3070, 10.0 h, 2,025 iterations, 24,883,200 steps,
$1.41. Stopped deliberately, not by the wall clock: the direction changed once
the failure mode was understood. Artifacts in `results/path2/run2/`.

**The optimiser worked.** This is the first run in the project where it did — KL
held near the 0.01 target, the learning rate moved freely between 2.6e-7 and
1e-5 instead of pinning at its floor, the importance ratio stayed within 0.99 to
1.00, and `replay_kl` measured 1.6e-5 on the GPU. That also formally closes the
run-1 CUDA hypothesis: the GPU was never the problem.

**It still did not walk, and it failed in a new way.**

| | ride height | joints off stance | landings/s | upright | speed |
|---|---:|---:|---:|---:|---:|
| CEM teacher | 0.148 m | **0.103 rad** | 24.5 | — | **0.106 m/s** |
| Stage 1 only | 0.144 m | 0.170 rad | 36.1 | 8.1 s | 0.034 m/s |
| run 1 | 0.143 m | 0.175 rad | 35.5 | 5.8 s | — |
| run 2 | 0.165 m | **0.636 rad** | 28.4 | 12.0 s | 0.026 m/s |

Run 2 eliminated falls completely and held 12 s upright on every seed — by
**bracing** into a splayed posture 3.7x further from the home stance than any
other policy, at a ride height above nominal standing. Statically stable because
it cannot fall. Speed went *down* against the prior it started from: falling ends
the episode and forfeits future return, so caution outscores progress.

**The third reward hack.** v2 dove, v3 rocked, v4 braces. It survived because
`upright` scores a flat 1.0 anywhere above 0.13 m and nothing penalises joint
deviation from stance, so a 0.636 rad contortion and a clean stance score
identically. Every logged scalar improved while the behaviour degraded; the
failure was found by watching the video, not by the instrumentation. That is an
instrumentation defect and it is recorded as debt in `EMBODY_V5_PLAN.md`.

**The plateau.** `track_lin` slope per 1,000 iterations by window: 0.0051
(6–400), 0.0554 (400–800), 0.0217 (800–1200), 0.0180 (1200–1600), 0.0091
(1600–2000), with the EMA slope going negative in the last window. Halving every
~400 iterations. Teacher-level `track_lin` at the trailing slope needs **27,600
more iterations**; ~1,000 remained.

**What run 2 says about the connectome: still nothing.** Two facts block that
claim. The teacher walks 0.106 m/s on the identical body and action space, so the
body is not the limit. And Stage 1 fits the teacher's actions to MSE 2.8e-4
through the 4-latent map, reproducing a real gait almost exactly on the teacher's
own trajectory, then collapsing to 0.034 m/s closed loop. That is compounding
error in behaviour cloning, not a capacity ceiling. The binding constraint is the
training procedure.

The successor plan, including what the field's own code does to make this
tractable and which of those assists are compatible with the moat, is
`EMBODY_V5_PLAN.md`.

## What v4 copies, and from where

### [LG] `leggedrobotics/legged_gym` — the reward set

Rudin, Hoeller, Reist, Hutter, *Learning to Walk in Minutes Using Massively
Parallel Deep Reinforcement Learning*, CoRL 164:91-100 (2022).
Source: `legged_gym/envs/base/legged_robot.py`, `legged_robot_config.py`.

This is the recipe Kim et al. (*Actuators* 15(1):33, 2026) use to train an 18-DoF
hexapod — our exact morphology — with PPO. Transcribed into
`eon-fly-brain/code/embody_env_v4.py`, term by term, with legged_gym's own default
weights from `LeggedRobotCfg.rewards.scales`:

| term | source method | weight | note |
|---|---|---:|---|
| `tracking_lin_vel` | `_reward_tracking_lin_vel` | +1.0 | σ rescaled by (0.1/1.0)² for our 0.1 m/s command |
| `tracking_ang_vel` | `_reward_tracking_ang_vel` | +0.5 | σ unchanged; yaw commands are on their scale |
| `feet_air_time` | `_reward_feet_air_time` | +1.0 | threshold 0.2 s, not 0.5 s (hexapod stride) |
| `lin_vel_z` | `_reward_lin_vel_z` | −2.0 | verbatim |
| `ang_vel_xy` | `_reward_ang_vel_xy` | −0.05 | verbatim |
| `action_rate` | `_reward_action_rate` | −0.01 | verbatim |
| `collision` | `_reward_collision` | −1.0 | over the trunk geoms |
| `orientation` | `_reward_orientation` | −1.0 | legged_gym ships −0.0; turned on here |
| `upright` | *(not legged_gym)* | +0.5 | [FB], below |

Also copied: **heading-mode command generation** — hold target speed straight
ahead and steer with `ang_vel_cmd = clip(0.5 · wrap_to_pi(heading_error), −1, 1)`;
the `only_positive_rewards` clipping of the weighted sum at zero; and the
`scale *= dt` convention, so the weights above are literally legged_gym's numbers
rather than retuned lookalikes.

**`feet_air_time` is the term that forbids rocking.** It pays a foot only at the
moment it lands, for the time it spent airborne. A body that never unloads a foot
earns nothing from it; a body that shuffles with sub-threshold strides is
*penalised* by it. Neither v2 nor v3 had any term that referred to feet at all —
nothing in the pipeline had ever specified a gait.

### [FB] `google-deepmind/flybody` — the ride-height factor

Vaxenburg et al., *Whole-body physics simulation of fruit fly locomotion*,
*Nature* 643:1312-1320 (2025). Source: `flybody/tasks/vision_flight.py`.

The bounded `dm_control.utils.rewards.tolerance` height factor is kept from v3,
because it is the term that genuinely stopped the diving: v2 → v3 took the
connectome arm from 12/12 falls and 0.52 s upright to **0/12 falls and 8.00 s
upright**. That part of the flybody port worked and is retained.

What is *not* retained is flybody's multiplicative combination. v4 uses
legged_gym's weighted sum. The reasoning is in the module docstring: the
combination rule was never what protected against reward hacking.

### [K] Kim et al. — command-conditioned imitation prior

*Deep Reinforcement Learning Control of a Hexapod Robot*, *Actuators* 15(1):33
(2026). Their PPO run is bootstrapped from *command-conditioned kinematic
trajectories*, not a single reference gait.

`eon-fly-brain/code/teacher_v4.py` copies that: one independent CEM search per
yaw-rate command over `(-0.4, -0.2, 0.0, +0.2, +0.4)` rad/s, each scored against
the arc that command traces at 0.1 m/s. Measured, 200 steps per command:

| command (rad/s) | realised Δyaw (rad) | distance (m) | path MSE |
|---:|---:|---:|---:|
| −0.4 | −0.907 | 0.369 | 3.15e-4 |
| −0.2 | −0.683 | 0.367 | 1.30e-4 |
| 0.0 | −0.030 | 0.424 | 2.77e-4 |
| +0.2 | +0.504 | 0.364 | 1.28e-4 |
| +0.4 | +0.949 | 0.386 | 1.96e-4 |

Monotone in the command, and every arm covers the reference distance. Under the
old budget the same table read `+0.0 → −0.484 rad` — the straight command turned
half a radian — and distances of 0.08-0.17 m.

### Not copied, and why

- **DeepMimic / flybody walk-imitation** (per-timestep pose tracking against
  recorded fly motion). There is no hexapod mocap, and imitating fly limb angles
  on a non-fly body would import the morphology we are trying to keep foreign.
  The synthetic arc teacher stays a *task teacher*, not biological imitation.
- **legged_gym's terrain and command curricula.** Flat ground only for now.
- **Massively parallel Isaac Gym.** No CUDA path for the sparse core, and the
  measured bottleneck is not the core anyway (below).

## The target object

New here, copied from nobody. The hexapod XML already carried a decorative red
`prey` sphere; v4 makes it the task.

- Respawns at a uniform bearing, 0.4-1.0 m away. Reaching within 0.15 m pays a
  one-shot **+20** and respawns it, so one 12 s episode is several approaches.
- The observation gains three channels: `cos(bearing)`, `sin(bearing)`,
  `clip(distance/2, 0, 1)`. Observation dimension 49 → **52**.

**This is the seam, and it is anatomy rather than a shortcut.** A real fly's VNC
does not see the world; it receives descending commands from the brain, which is
where vision and olfaction live. Feeding a target bearing into the descending
channels *is* that pathway. The VNC still has to turn a steering command into
eighteen coordinated joints, and that part stays frozen.

## Compute — the earlier framing was wrong

An earlier note in this project said Stage 2 ran a single environment. It did
not: v2/v3 ran **128 parallel environments**. The real waste is different, and the
project's own `THROUGHPUT_REPORT.md` shows it:

- Core sparse matmul is **8.5 ms** of a **68.9 ms** batch step at B=128. The
  frozen VNC is **12%** of wall time; MuJoCo and Python on 4 threads are the
  other 88%. **A GPU buys almost nothing.** Cores do.
- 5M steps/arm is roughly **2%** of the 100-300M that Rudin et al. need for a
  quadruped with no recurrent core in the loop.

v4 therefore runs **B=512 on a high-core CPU pod**. A null after 5M could not
distinguish "this core cannot walk" from "this was never trained"; a null after a
real budget is worth something.

**The wall clock is the open question, and it is not settled.** Measured on this
laptop, one iteration at `PPO_ROLLOUT_STEPS = 24`, `PPO_BATCH_SIZE = 512`,
5 epochs × 4 minibatches:

| phase | wall time | share |
|---|---:|---:|
| collect (12,288 environment steps) | 8.1 s | 11% |
| update (20 minibatch passes) | 67.0 s | 89% |
| **one iteration** | **75.1 s** | |

That is 6.11 ms per environment step of total compute, so the full
4,069-iteration budget is **85 hours on this machine**. The update dominates
roughly eight to one, and it
scales directly with `PPO_UPDATE_EPOCHS × PPO_MINIBATCHES`.

The speedup on a 32-vCPU pod is **not measured**, and the earlier "roughly 2 h"
figure was arithmetic from a 4-thread benchmark, not a measurement. It should not
be trusted. The honest way to settle it costs about fifteen minutes of pod time:
start Stage 2, read `elapsed_s` off the first three logged iterations, and set the
budget from the pod's own number. Stage 2 checkpoints every 1M steps and resumes
from its checkpoint, and `V4_PPO_STEPS`, `V4_UPDATE_EPOCHS` and `V4_MINIBATCHES`
are read from the environment, so the budget can be retuned after that
calibration without losing work and without editing a script bash is still
reading.

The lever, if the pod number comes back too slow, is `PPO_UPDATE_EPOCHS ×
PPO_MINIBATCHES` — it buys wall clock directly at the cost of gradient updates:

| epochs × minibatches | updates over 4,069 iterations | iteration cost (this laptop) |
|---|---:|---:|
| 5 × 4 (source, current) | 81,380 | 75 s |
| 3 × 4 | 48,828 | ~48 s |
| 2 × 2 | 16,276 | ~22 s |

At the source setting the budget already buys 81,380 updates against legged_gym's
~30,000, so this lever now has slack it did not have at the 150-step segment.

## Pipeline

```
[LG] observation: base linear vel (3, body frame, x2.0)
                  base angular vel (3, body frame, x0.25)
                  projected gravity (3)
                  target: cos(bearing), sin(bearing), distance (3)   <- commands slot
                  joint pos - stance (18)
                  joint vel x0.05 (18)
                  previous action (18)                          =  66
                     |
       encoder  nn.Linear(66, 4)          <- trained
                     |
       fixed dn_command_map (4 x 1325)    <- frozen, predeclared
       drive = 100.0 * sigmoid(command)
                     |
       FROZEN VNC CORE  (11,000 neurons, 485,074 edges, 0 parameters)
                     |
       260 motor channels, amplitude-matched
                     |
       decoder  nn.Linear(motor, 18)      <- trained
                     |
              18 joint targets
```

**Stage 1** — command-conditioned CEM teacher, then behaviour cloning of the
interface. Held-out split is *within* each command, not across commands, and the
recurrent state resets between demonstrations. Both differ from v3's
`_stage1_fit`, which would otherwise have measured command extrapolation and
carried state across independent episodes.

**Stage 2** — PPO, connectome arm only, 50M steps, resumable from checkpoint.

**Evaluation** — 12 fixed seeds. Primary endpoint is **targets reached**, not
distance walked: v3 established that this body can score well and go nowhere.
Also logged: closest approach, path over displacement (a rocker's ratio is large,
a walker's is near 1), falls, upright seconds.

## Files

| path | what |
|---|---|
| `eon-fly-brain/code/embody_env_v4.py` | goal-directed env, legged_gym reward set, vector wrapper, smoke test |
| `eon-fly-brain/code/teacher_v4.py` | command-conditioned CEM teacher + self-check |
| `eon-fly-brain/code/run_embody_v4.py` | Stage 1, Stage 2, evaluation, smoke |
| `eon-fly-brain/code/test_embody_v4.py` | 14 pre-flight checks; the pod runner fails on these before spending |
| `eon-fly-brain/deploy/run_v4_pod.sh` | on-pod runner, resumable, no self-termination |

v2 and v3 artifacts are untouched; v4 writes to its own `embody_v4_*` names.
`audit_embody_3way._policy` gained an optional `obs_dim` argument (defaulting to
the v3 observation) so both versions share one policy factory.

## What is still open

- **The controls have not run.** Random and rewired arms go back in once the
  connectome recipe stops moving. No comparative claim is available until then,
  and none is being made.
- **The static-input diagnostic is still not run.** Clamp the descending drive
  constant and see whether the behaviour survives. This is the decisive test of
  whether the core contributes anything beyond being a filter, it was
  preregistered as deferred, and it has never been executed.
- **`FEET_AIR_TIME_THRESHOLD_S = 0.2` is a stride-rate guess**, not a
  measurement. Retune from the realised duty factor once an arm walks.
- **The pod's throughput is unmeasured**, and it is the only thing standing
  between here and a launch decision. See *Compute*.
- **`active_motor_channels` was 29 on one pod and 52 on another** — a
  reproducibility issue flagged earlier and still not chased.
- **The 208/260 excluded-motor-channel sensitivity analysis** remains outstanding
  from the research scout's ranked list.
