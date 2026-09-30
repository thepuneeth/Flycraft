# CPG translation loop

## Iteration 1 — 2026-09-06

- Hypothesis: replacing the sine/lift output shape with recorded NMF
  `PreprogrammedSteps` kinematics will improve directed movement while V1
  amplitude modulation preserves the NMF tripod timing.
- Change: port the recorded fly step spline and add a fixed role-based
  fly-to-hexapod retargeter; no rollout, constants not selected by outcome.
- Prediction: V1 with the recorded shape will move closer to the recorded
  `2.3944 m` CPG-only forward ceiling and keep path/forward below the direct
  map's `11.4x` baseline; exact value unknown before rollout.
- Verification: `python3 -u nmf_cpg.py` passed. A lazy MuJoCo import was needed
  so this pure math check runs on the laptop; physics still raises clearly when
  MuJoCo is unavailable.
- Runner verification: `python3 -m py_compile nmf_cpg.py cpg_connectome_walk.py`
  and V1 affine boundary checks passed.
- Status: completed on RTX 4070 Ti at `$0.190/hr`; pod deleted after pull.
- Result: 8 matched episodes, 8 s each. Full V1 mean forward `-0.1897 m`,
  constant-connectome `-0.3599 m`, time-shuffled recorded signal reported,
  CPG-only `0.1002 m`; full beat constant but failed forward gate.
- Freeze audit: graph hashes and real tensor hash equal pre/post; artifact
  reports `zero_learning=true`, `parameter_updates=0`, empty training markers.
- Interpretation: recorded-step retargeting plus V1 did not recover old sine
  CPG ceiling; CPG-only also degraded, so frequency is the next isolated
  change before judging interface specificity.

## Iteration 2 — 2026-09-06

- Hypothesis: the recorded-step retargeter may have a lower hexapod operating
  frequency than the inherited `1.2 Hz`; a declared sweep may recover forward
  motion without changing geometry or interface mapping.
- Change: sweep `(0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0) Hz` once, using Iteration 1's
  highest-forward arm (`cpg_only`) as winner; run the full four-arm ablation
  ladder at every point, same body and episode specs.
- Prediction: one of `0.8–1.5 Hz` will beat Iteration 1's `0.1002 m` CPG-only
  mean forward, with neighbours reported to distinguish a broad optimum from
  a knife edge. No narrowed follow-up sweep will be allowed.
- Verification: local pycompile and import self-check passed; no constants
  selected from body distance.
- Status: completed on RTX 4070 Ti at `$0.190/hr`; pod deleted after pull.
- Result: all seven frequencies, four arms, eight matched episodes each.
  CPG-only curve was `0.1411, -0.2252, 0.0466, 0.1002, -0.2448, 0.0710,
  -0.3351 m` for `0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0 Hz`; selected `0.6 Hz`.
  Its sole neighbour (`0.8 Hz`) collapsed, so peak is knife-edge. Full V1
  peaked separately at `0.2738 m` at `0.8 Hz`, but was not selected post hoc.
- Freeze audit: graph hashes and real tensor hash equal pre/post; artifact
  reports `zero_learning=true`, `parameter_updates=0`, empty training markers.
- Interpretation: frequency is unstable under this retargeting and no arm
  meets walking gate. Proceed to all three variants at fixed preregistered
  `0.6 Hz`; no further frequency search.

## Iteration 3 — 2026-09-07

- Hypothesis: frequency fixed at `0.6 Hz`, stronger connectome interfaces may
  recover coordination: V1 amplitude is most likely to walk, V2 frequency may
  improve leg timing, and V3 measured phase may replace the hand-written tripod.
- Change: implement all three variants with affine rate-ceiling-normalized
  maps; evaluate scale grid `(0.25, 0.5, 1.0)` once, plus full, constant,
  time-shuffled, and CPG-only arms at identical eight-episode specs.
- Prediction: V1 remains most stable; V2 may reduce drift; V3 likely loses
  tripod lock because measured VNC phase is not a reliable gait signal. No
  variant or scale will be selected before the complete grid is reported.
- Verification: local pycompile and affine/phase self-check passed. Custom
  phase-bias coupling changes only apply to V3; default NMF tripod remains.
- Bug loop: detached rollout exposed CPG-only V1 at `-0.116 m` across scales;
  root cause was synthetic zero scores passing through V1, which set open-loop
  amplitude to its affine midpoint `0.5` instead of declared CPG-only `1.0`.
  PID was stopped before artifact/video completion.
- Correction: CPG-only now calls default NMF amplitude/frequency directly;
  local self-check and pycompile rerun passed. This is a rerun of Iteration 3,
  not a new scale or frequency choice.
- Status: completed on RTX 4070 Ti at `$0.190/hr`; pod deleted after pull.
- Result: all 3 variants × 3 scales × 4 arms, eight matched episodes each.
  Best full arm was V2 frequency at scale `0.25`: `0.4029 m` forward,
  `2.9023 m` path, ratio `10.09×`; below `0.5 m` and `2.5×` gates. V1 best
  full `0.2908 m`; V3 best full `0.1711 m`. Controls beat full in several
  cells, so no specificity claim is supported.
- Freeze audit: graph hashes and real tensor hash equal pre/post; artifact
  reports `zero_learning=true`, `parameter_updates=0`, empty training markers.
- Interpretation: V2 is the least-bad exploratory interface but still not
  walking. Do not run specificity controls yet; enter bug loop to localize
  retarget/target/physics failure before any 20-episode confirmation.

## Iteration 4 — 2026-09-07 (bug loop diagnostic)

- Hypothesis: the current source-peak-to-full-MuJoCo-range retarget saturates
  joints and explains the gap between old sine CPG (`2.3944 m`) and recorded
  CPG-only (`0.1411 m`).
- Change: no production change yet; compare old sine targets, current
  full-range recorded targets, and raw role-offset targets on one fixed body
  episode. Log target ranges, continuity, actuator limits, and physics output.
- Prediction: raw role offsets, whose excursions resemble old CPG scale, will
  improve forward/path ratio; if not, inspect swing phase/sign next.
- Result: fixed episode at `1.2 Hz`: old sine `2.3944 m` forward / `2.6918 m`
  path / no fall; full-range recorded `0.3605 m` / `4.1764 m` / fell, with
  max target jump `1.051 rad`; raw role offsets `-0.3544 m` / `1.9781 m` / no
  fall. Reversing phase improved raw offsets to `0.3004 m` / `1.8826 m`.
  Removing tibia motion from reversed raw offsets reached `0.7331 m` / `1.3399
  m` / no fall; old sine scale with tibia static reached `2.3710 m`.
- Localization: full-range target scaling is a physics failure; tibia waveform
  is the dominant remaining destabilizer; source phase direction is inverted
  relative to this body's productive coxa/femur direction. No production fix
  selected from distance; diagnostic alternatives are not reported arms.
- Status: diagnostic complete; pod pulled conceptually (stdout only) and ready
  for a deterministic retarget fix.

## Iteration 5 — 2026-09-07

- Hypothesis: retaining recorded angular excursions instead of expanding them
  to the entire hexapod joint range, plus the diagnosed source/body phase
  reversal, will remove target discontinuity and restore forward stepping.
- Change: production `PreprogrammedHexapodSteps` now uses source-sized offsets
  clipped by model range, reverses fixed translated phase direction, and updates
  `swing_period`; no locomotion-fit constant or new control map.
- Prediction: target jump stays below `0.35 rad`, CPG-only avoids falls and
  becomes positive-forward; whether it clears `0.5 m` remains an empirical gate.
- Verification: local NMF demo, pycompile, and diff checks passed.
- Result: corrected 8-episode probe at `1.2 Hz`: CPG-only `0.7715 m` forward,
  `1.9189 m` path, ratio `2.498×`, no falls; full V1 `0.0585 m`, constant
  `0.0591 m`, time-shuffled `0.0613 m`. Corrected prior clears raw walking
  gates, but connectome arm loses to controls, so specificity remains null.
- Freeze audit: graph hashes and real tensor hash equal pre/post; artifact
  reports `zero_learning=true`, `parameter_updates=0`, empty training markers.
- Interpretation: retarget/phase bug fixed. Re-run required frequency sweep
  under corrected mapping, then all three variants at its fixed winner; old
  Iterations 1–3 frequency/variant values remain superseded diagnostics.

## Iteration 6 — 2026-09-07

- Hypothesis: corrected source-sized retargeting and phase direction will make
  the frequency curve resemble the recovered `1.2 Hz` gait-only probe, while
  full V1 will remain near its ablation controls.
- Change: repeat the frozen seven-point frequency grid once under corrected
  retarget semantics; no new frequencies or narrowed search.
- Prediction: `1.0–1.5 Hz` will contain corrected CPG-only winner; full V1 will
  not beat CPG-only by the `0.5 m`/`2.5×` success gates.
- Verification: prior corrected probe and all freeze checks passed.
- Status: completed on RTX 4070 Ti at `$0.190/hr`; pod deleted after pull.
- Result: corrected CPG-only forward curve was `0.2833, 0.5929, 0.4896,
  0.7715, 0.4281, 0.2885, 0.0359 m` over the declared grid; selected `1.2
  Hz`, with neighbours `0.4896` and `0.4281 m`, not knife-edge. Full V1 was
  `-0.0340, 0.0288, -0.0012, 0.0585, 0.3138, 0.0905, -0.0060 m` and tracked
  constant/time-shuffled controls at every point.
- Freeze audit: graph hashes and real tensor hash equal pre/post; artifact
  reports `zero_learning=true`, `parameter_updates=0`, empty training markers.
- Interpretation: corrected prior walks at `1.2 Hz`; full connectome V1 still
  fails A-vs-B and remains below walking gate. Run all three variants at fixed
  corrected `1.2 Hz`; no more frequency search.

## Iteration 7 — 2026-09-07

- Hypothesis: corrected retarget plus V1/V2/V3 at valid `1.2 Hz` may reveal a
  connectome-specific interface even though V1 failed; V2 gets declared
  physical range `[0.6, 1.8] Hz` around the fixed base, V1/V3 keep prior maps.
- Change: rerun all variants and scales `(0.25, 0.5, 1.0)` at fixed corrected
  frequency, with full/constant/time-shuffled/CPG-only arms and no new tuning.
- Prediction: V1 remains near controls; V2 may be best interface but should
  stay below `0.5 m` full-arm gate; V3 likely loses tripod/forward motion.
- Verification: local pycompile, affine/phase self-check, and corrected sweep
  freeze audit passed.
- Status: ready for corrected all-variant rollout.

- Status: completed on RTX 4070 Ti at `$0.190/hr`; pod deleted after pull.
- Result: all three variants × scales `(0.25, 0.5, 1.0)` at corrected fixed
  `1.2 Hz`, four matched arms, eight episodes each. Best full V1 was `0.0703 m`
  at scale `0.25`; best V2 was `0.7954 m` at scale `1.0`, path `1.9196 m`,
  ratio `2.418×`; best V3 was `0.6711 m` at scale `0.5`, path `1.6206 m`,
  ratio `2.455×`. V2 full beat constant by `0.0210 m`; V3 full beat constant
  by `1.0587 m`; these are exploratory eight-episode results, not specificity
  evidence.
- Freeze audit: all rows reported equal graph/tensor hashes; artifacts report
  `zero_learning=true`, `parameter_updates=0`, empty training markers.
- Interpretation: V2 and V3 clear exploratory gait threshold, but real-vs-control
  specificity and the required `>=20`-episode confirmation remain unfinished.
  Freeze variants/scales now; run final confirmation with at least 12 rewired and
  12 random-network control seeds.

## Iteration 8 — 2026-09-07 (final specificity confirmation)

- Hypothesis: the corrected V2/V3 walking interfaces will retain a real-connectome
  advantage after amplitude-matched graph controls and stochastic realization
  averaging; V1 is included regardless of expected weakness.
- Change: freeze Iteration 7 scales (`V1=.25`, `V2=1.0`, `V3=.5`) and corrected
  `1.2 Hz`; run all three variants with the four-arm ladder, 20 shared body
  episodes, 30 real Poisson realizations, and 10 Poisson realizations for each
  of 12 random and 12 degree-preserving rewired graphs. Controls were matched at
  six per-leg mean-score inputs with `audit_amplitude_control.amplitude_match`.
- Prediction: V3 should be strongest but may fail real-vs-control specificity;
  no scale or frequency search is permitted after this run.
- Verification: [SH] LIF and [NMF] CPG preflight passed; graph hashes and real
  tensor hash matched before/after; 2,490 body rollouts stayed below harness
  cap; `zero_learning=true`, `parameter_updates=0`, empty training markers.
- Result: V1 real `0.0722 m` forward / `21.52×` path-forward; V2 real `0.7647
  m` / `2.5095×`; V3 real `0.8386 m` / `2.4624×`. V3 full beat its constant
  ablation by `0.7304 m`; V2 lost to constant by `0.0031 m`. V3 random-control
  mean was `0.8429 m` (real margin `-0.0043 m`); rewired-control mean was
  `0.4540 m` (margin `0.3845 m`), below the real realization SD `0.4083 m`.
  Thus no variant beat both controls by more than real realization noise.
- Decision: `CPG-INTERFACE-SPECIFICITY-NULL`. Walking was achieved by V3, but
  real connectome structure did not survive the CPG interface specificity test.
  Stop; do not tune further. Honest ceiling remains: the connectome modulates a
  hand-written biologically-derived gait; it does not generate the gait.
