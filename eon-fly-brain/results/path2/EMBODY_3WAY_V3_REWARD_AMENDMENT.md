# EMBODY-3WAY v3 — reward amendment

**Status:** proposed. Not run. Requires owner approval before any pod launch.
**Date:** 2026-09-03
**Amends:** Stage-2 protocol addendum, `protocol_hash` `be73a28fd39397e14c8c2b8d327d2f58cb8af533aa129ae3a81e00caf3121e7f`
**New:** `protocol_hash` `2ec2357c43272dde5510ce1ef6bb401bdfa44183b44fc4c59ab8abe1b9d70d4f`

## Why v2 is being amended

The v2 run completed all three arms on identical 4,999,936-step budgets and
produced a statistically clean but scientifically empty result:

| arm | forward | 95% CI | upright | speed | falls |
|---|---:|---:|---:|---:|---:|
| connectome | 0.3078 m | [0.2986, 0.3171] | 0.52 s | 0.588 m/s | 12/12 |
| rewired | 0.1701 m | [0.1684, 0.1721] | 8.00 s | 0.021 m/s | 0/12 |
| random | 0.1500 m | [0.1480, 0.1523] | 8.00 s | 0.019 m/s | 0/12 |

Both contrasts reached `p_holm = 0.00048828125` with `0/12` control reversals,
and the decision label was correctly `EMBODY-PILOT-INCONCLUSIVE` because
`fall_rate = 1.0` for the connectome arm.

The connectome arm did not walk further. It fell forward faster, and the
episode terminated at the `H_MIN = 0.08` floor after 26 of 400 steps. The
reported distance compared 0.52 s of connectome against 8.00 s of controls.

This is a reward defect, not a training defect. Two independent checks confirm
the optimiser behaved correctly:

- GAE already zeroes the bootstrap at a fall
  (`run_embody_3way.py`: `nonterminal = 1.0 - dones_np[index]`), matching
  flybody's `Walking.get_discount` returning `0.0` on termination. The fall
  already forfeited all future value.
- Falling was nonetheless optimal: forfeiting 7.5 s of a 0.019 m/s shuffle
  costs about 0.14 m, less than the 0.308 m the dive banks immediately.

PPO found the true optimum of a misspecified objective.

## Root cause

`reward = dx*cos(heading) + dy*sin(heading)`, summed over the episode, is
unbounded and monotone in displacement. It has no fixed point, so the highest
scoring behaviour is always the most ballistic one the body can produce.

A second, compounding defect: Stage 1 fits a CEM teacher selected on
*constant-speed 0.1 m/s* forward body-path MSE, while Stage 2 rewarded
*unbounded maximum displacement*. The two stages optimised different
objectives, and Stage 2 won — the connectome arm converged to 0.588 m/s, about
6x the speed Stage 1 taught it.

## What the reference implementations do

Read from `fly_field_refs/flybody` (google-deepmind/flybody) and
`fly_field_refs/flygm`, which consumes flybody's task rewards through an
Acme-style env.

1. **Bounded target tracking, never maximisation.** `walk_on_ball.py:65-75`
   rewards `rewards.tolerance(ball_qvel - target_ball_qvel, bounds=(0,0),
   sigmoid='linear', margin=6, value_at_margin=0.0)` against
   `target_ball_qvel = [0., -5, 0]`. `rewards` is `dm_control.utils.rewards`;
   `tolerance` is bounded in [0,1] and peaks at zero error. Exceeding the target
   speed scores *worse*, so a lunge cannot outscore a walk.
2. **Multiplicative combination.** `base.py:212-215`:
   `get_reward` returns `np.prod(self.get_reward_factors(physics))`. No factor
   can be bought with another; one collapsed factor zeroes the step. A summed
   reward permits the stability-for-distance trade that v2 made.
3. **Termination is a physics-blowup guard, not the shaping signal.**
   `constants.py`: `_TERMINAL_QACC = 1e14`, `_TERMINAL_LINVEL = 50` cm/s,
   `_TERMINAL_ANGVEL = 200` rad/s. Stability is shaped continuously by the
   reward factors. v2 inverted this: a binary terminal height floor was the
   only stability signal, giving the policy a cliff to sprint at instead of a
   gradient to climb.
4. **Imitation is per-timestep pose tracking.** `walk_imitation.py:163-166`
   scores com, qvel, root2site, and joint quaternions against a reference clip
   at every step. A body-path-only teacher leaves limb phase, contacts, and
   joint kinematics undetermined — already flagged as Q5 in
   `RESEARCH_SCOUT_FINDINGS.md`.

## The amendment

`dm_control.utils.rewards.tolerance` and its `_sigmoids` helper are vendored
verbatim into `eon-fly-brain/code/dm_rewards.py` (Apache-2.0, header retained,
retrieved 2026-09-03 from google-deepmind/dm_control @ main). Vendored rather
than depended on because the project needs exactly one function, and using the
identical implementation is what makes this a port rather than a lookalike.

`HexapodEnv.reward_factors()` returns two factors, each in [0,1]:

```python
speed = tolerance(forward_speed,
                  bounds=(TARGET_SPEED_MPS, TARGET_SPEED_MPS),
                  margin=TARGET_SPEED_MPS,
                  sigmoid="linear", value_at_margin=0.0)
upright = tolerance(height,
                    bounds=(UPRIGHT_HEIGHT_M, inf),
                    margin=UPRIGHT_HEIGHT_M - H_MIN,
                    sigmoid="linear", value_at_margin=0.0)
reward = speed * upright
```

- `TARGET_SPEED_MPS = 0.1` — chosen to equal the Stage-1 CEM teacher's
  `speed_mps`, so both stages now optimise the same locomotion objective.
- `UPRIGHT_HEIGHT_M = 0.13` — full credit above this, decaying linearly to zero
  at `H_MIN = 0.08`. Holding the home stance for 400 control steps settles at
  0.154 m (measured on this model), so 0.13 leaves room for normal gait bob
  while making lost ride height cost reward continuously.
- `forward_speed` is read from `qvel` projected on the seeded initial heading,
  not finite-differenced from `qpos`.
- The `H_MIN` termination is retained unchanged, now as the blowup guard it
  should always have been rather than the shaping signal.

Under this reward the v2 dive scores exactly `0.0` per step on both factors,
and an on-target upright walker scores approximately `1.0`.

## Metric changes

Distance alone hid the unequal episode lengths. Each evaluation row now also
carries `upright_s` and `mean_forward_speed_mps`, each condition summary carries
`mean_upright_s`, `mean_forward_speed_mps`, and `mean_episode_return`, and the
run brief prints upright seconds and m/s beside the distance. `episode_return`
in evaluation now accumulates the actual reward instead of re-reporting
displacement.

## Explicitly unchanged

- `MIN_STABLE_FORWARD_M = 0.20` and the `stable_forward` rule — not retuned.
- The 12 evaluation seeds, `VIDEO_SEED`, and the training seed.
- `ARM_STEP_CAP = 4,999,936`, `batch_size = 128`, all PPO hyperparameters.
- The three arms and their frozen cores; the amplitude-matching rule.
- The paired sign-flip test, Holm correction, and bootstrap CIs.
- Every frozen-core integrity check.

One change to the objective, nothing else, so any difference from v2 is
attributable to the reward.

## Known limitations of v3

- The upright factor uses ride height only. It does not check body attitude, so
  a policy that stays high while tilted is not penalised for the tilt.
- Contact sequence, limb phase, and duty factor are still not scored or
  reported. Until they are, a v3 positive supports "moved forward at target
  speed without falling", not "produced a gait". This is the Q5 gap and it
  remains open.
- Speed tracking is scored against a single fixed target with no turning or
  speed-change demand.
- Single-body result; no morphology generalisation claim is available.

## Cost

Approximately $0.65 and roughly 50 minutes on the same RTX 3090 class pod,
three arms at the unchanged 5M-step budget.

## Run instructions

The Stage-2 addendum is immutable by design (`write_addendum` raises
`FileExistsError`). A v3 run needs the v2 addendum moved aside first, and
`embody_3way_v2.json` preserved, or the new run will refuse to start. Stage 1
does not need rerunning: the reward change affects Stage 2 only, and the
Stage-1 checkpoints and their recorded sha256 values are unchanged.
