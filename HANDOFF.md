# Flycraft — Path 2 handoff

Status: corrected body-setup rerun complete through E0–E4 and Path 3; current
`H_MIN=.08 m` gate, Methods 1–2, and corrected Path 3 controls are complete.
Pre-fix body-level results are confounded/superseded; the final claim remains
gated on user approval.

## Current result (claim review pending)

Corrected home-pose body setup makes the clean external tripod clock stand and
travel (`1.660 m` forward, `2.038 m` path in 8 s; minimum height `.121 m`; no
fall), but the expanded frozen VNC still does not produce the declared
alternating tripod. All 212 published T1/T2/T3 coordination candidates are
included, including both formerly omitted T1 `IN12B013` cells. E0, E3, E4,
Path 3, Method 1, and Method 2 remain phase-absent. Method 1's real 4/2
grouping falls in 16/20 episodes; Method 2 restores near-half duty but not
canonical inter-leg phase.

The five-condition study shows real VNC wiring beats column-shuffled VNC on
path (`+0.2663 m`, 95% CI `[+.2611,+.2714]`, exact `p=1.91×10⁻⁶`, Holm
`p=5.72×10⁻⁶` for the predeclared B2 path test), while brain-coupled VNC is
below matched random descending drive (`−.0621 m`, 95% CI
`[−.1192,−.0092]`, Holm `p=.0422`). This is a real VNC motor-output positive,
not evidence that connectome coordination transfers to a tripod.

This is corrected evidence assembly, not final claim wording. Outcome C/D were
not rerun; the fixed zero-learning routes still show no coordinated walk.

## Pre-fix working headline (confounded/superseded)

The earlier body-level headline and body-dependent result numbers used uniform
stance `(0.0, 0.6, -1.2)` for all legs. They remain below for provenance only;
the `bodyfix_*` artifacts supersede them for body-level inference. Graph
selection and direct VNC-only measurements are retained, not silently deleted.

## Body-setup correction

The shared bug applied `(0.0, 0.6, -1.2)` to every leg. That settles at about
`.073 m`, while the MJCF `home` keyframe encodes `z=.161 m` and mirrored right
leg angles. The corrected code reads `m.key("home").qpos[7:25]`, preserves the
fixed PD (`kp=3`, `kd=.05`), torque limits, engine state assignment, and
steering, and uses the declared mirrored coxa/femur signs. Clean-clock
constants are coxa `.35`, lift `.7`, base rate `.8`, drive rate `.5` (`1.2 Hz`
at forward `.8`); no post-rerun tuning was performed. Commanded coxa phase
uses fixed world-forward signs `[+1,-1,+1,-1,+1,-1]` to remove coordinate
mirroring, not to fit performance.

| result | pre-fix retained | corrected bodyfix |
|---|---:|---:|
| static settle | `.073 m` | `.145 m` |
| Gate A clean engine | `.530 m` forward; `1.080 m` path; `.074 m` min height | `1.660 m` forward; `2.038 m` path; `.121 m` min height |
| B1/P3 real episode | `.194 m` forward; `.461 m` path; scalar `+.112` | `.202 m` forward; `.542 m` path; scalar `−.306` |
| B1/P4 real − shuffled | `+.2127 m` `[+.1636,+.2563]` | `+.3314 m` `[+.2529,+.4070]` |
| B1/P4 real − random | `−.0946 m` `[−.1265,−.0625]` | `−.0966 m` `[−.1395,−.0562]` |
| B2 real − shuffled | `+.1425 m` `[+.1397,+.1449]` | `+.2663 m` `[+.2611,+.2714]` |
| B2 brain − random | `−.0383 m` `[−.0775,+.0012]` | `−.0621 m` `[−.1192,−.0092]` |
| E0 | `E0-ABSENT` | `E0-ABSENT`; `3/6` within, `3/9` cross, scalar `+.098` |
| E3 | `E3-ABSENT` | `E3-ABSENT`; `2/6` within, `3/9` cross, scalar `+.103` |
| E4 | `E4-ABSENT` | `E4-ABSENT`; `0/6` within, `4/9` cross, scalar `−.210` |

Corrected files use `eon-fly-brain/results/path2/bodyfix_*`; original files
remain untouched as superseded provenance.

Do not replace this with a catch headline. No condition caught prey.

## Research question

Can an untrained FlyWire brain plus the official Male CNS VNC drive a body it
did not evolve for — a PhantomX-class hexapod in MuJoCo — and where does
transfer fail?

```text
prey -> FlyWire brain -> DNg100/DNb08 -> real MANC VNC
     -> fixed motor map -> PD torque -> MuJoCo hexapod
```

The brain supplies visual steering. MANC VNC dynamics supply motor activity.
The fixed map reduces five fly joint groups to coxa/femur/tibia targets. The
body receives only fixed PD torque. No reward, optimizer, backpropagation,
target fitting, learned readout, or performance-based tuning exists.

## Task A: published coordination candidates

The fixed anatomical rule is:

```text
traced AND vnc_intrinsic AND type in the 66 labels from eLife 96084
Supplementary file 8 AND soma neuromere in requested T1/T2/T3 scope
```

Source: [Cheong et al., eLife 96084](https://elifesciences.org/articles/96084),
[Supplementary file 8](https://cdn.elifesciences.org/articles/96084/elife-96084-supp8-v1.csv).

Current v1.0 annotations contain 212 scoped cells matching this rule. The
previous graph omitted two `IN12B013` cells; the rebuilt graph includes:

| body ID | type | neuromere | side | added candidate edge |
|---:|---|---|---|---:|
| 802193 | IN12B013 | T1 | R | 6 synapses to INXXX008 |
| 802470 | IN12B013 | T1 | L | 5 synapses to IN10B014 |

That is two cells and two candidate-to-candidate cross-neuromere edges, 11
synapses total. The scoped graph has 212/212 candidate cells included. The
66-label set contains 62 labels represented in T1/T2/T3; `IN01A001` is absent
from the v1.0 annotation, while `INXXX025`, `INXXX058`, and `INXXX115` occur
only on A-segment annotations and are outside scope. This distinction is
recorded in `results/path2/all_legs_coordination.json`.

The missing tripod is therefore not explained by omission of the published
T1/T2/T3 candidate cells. Other non-published boundary cells remain outside
the declared subnetwork; do not broaden the claim beyond this resolution.

## Frozen graph and dynamics

All-leg artifact:

- 11,000 nodes;
- 1,704,763 induced raw edges;
- 492,722 edges at absolute synapse floor 5;
- 485,074 signed nonzero edges;
- 260 motor outputs.

T1 artifact:

- 4,418 nodes;
- 144,214 signed nonzero edges;
- 106 motor outputs.

Frozen VNC protocol: paper-rate model, `tau=.02`, `dt=.005`, four substeps,
`gW=.03`, bilateral DNg100/DNb08 drive 100, DNb08 gain 1, threshold 7.5,
activation gain 1, firing-rate cap 200, seeded heterogeneity seed 1. The T1
validation uses 1,500 steps sampled every .02 s, half-run burn-in, no smoothing.

T1 output under constant drive:

- dominant frequency: 2.467 Hz;
- autocorrelation frequency: 2.500 Hz;
- autocorrelation peak mean: .342;
- opposing-pool correlation: −.321;
- left/right correlation: +.031;
- active motor fraction: .368.

The old 3.07 Hz number is stale. The current frozen rerun produced 2.467 Hz;
report the current value.

All-leg coxa phase proxy:

| group | mean correlation |
|---|---:|
| tripod A within | −.057 |
| tripod B within | +.307 |
| cross-tripod | −.185 |

One within-tripod group is positive while the other is negative; the expected
coherent alternating tripod does not emerge. Raw five-synapse-floor coupling
has 166,088 cross-neuromere edges; 71,769 are retained, from 4,341 source and
7,048 target neurons. There are 4,860 omitted boundary candidates after the
two published candidates are included.

## Control study and inference

Corrected-body study design: five conditions, 20 paired episode starts per
condition, 3 s each, static prey, episode base seed `20260829`. Every
condition shares each episode's initial yaw and prey placement.

All 100 episodes returned normally: 0 catches and 0 falls in every condition;
Wilson 95% CI for each binary rate is `[0.000, 0.161]`.

| condition | path m [95% bootstrap CI] | forward m [95% bootstrap CI] | displacement m [95% bootstrap CI] | min distance m [95% bootstrap CI] |
|---|---:|---:|---:|---:|
| real VNC open-loop | .351 [.348, .353] | .042 [.039, .045] | .093 [.090, .096] | 1.352 [1.250, 1.453] |
| brain-coupled VNC | .286 [.224, .340] | .022 [.007, .038] | .046 [.034, .059] | 1.366 [1.264, 1.467] |
| shuffled VNC | .084 [.080, .090] | .016 [.012, .020] | .041 [.038, .045] | 1.395 [1.295, 1.493] |
| random descending | .348 [.317, .378] | .027 [.010, .045] | .056 [.045, .069] | 1.363 [1.264, 1.460] |
| Path 1 hand CPG | .179 [.129, .230] | .067 [.030, .107] | .077 [.041, .115] | 1.349 [1.232, 1.459] |

Mean motor temporal SD is .140 [.140, .140] real VNC, .116 [.092, .135]
brain-coupled, .067 [.067, .067] shuffled, and .133 [.128, .138] random;
Path 1 has no VNC motor-output metric.

Primary exact two-sided paired sign-flip tests on path, with Holm adjustment
across three predeclared comparisons:

| comparison | difference m [95% CI] | exact p | Holm p |
|---|---:|---:|---:|
| real VNC − shuffled VNC | +.2663 [.2611, .2714] | 1.91e−6 | 5.72e−6 |
| brain-coupled − random descending | −.0621 [−.1192, −.0092] | .0422 | .0422 |
| open-loop VNC − Path 1 hand CPG | +.1713 [.1193, +.2225] | 5.72e−6 | 1.14e−5 |

The real-versus-shuffled motor temporal-SD difference remains +.0730 with
exact paired `p=1.91e−6`. This is the positive wiring result. It does not turn
the brain-coupled controller into a successful pursuit controller.

Full metrics and tests: `results/path2/bodyfix_vnc_control_study_3s_stats.json`.

## Failure localization

`results/path2/bodyfix_transfer_layers.png` is corrected mechanistic figure;
`results/path2/bodyfix_transfer_layers.json` contains corrected measurements.
The old `transfer_layers.*` files remain as superseded provenance.

1. **VNC dynamics (layer i):** dynamic output exists — 260 motor channels,
   mean temporal SD .152 — and T1 oscillates. But all-leg inter-leg phase is
   inconsistent: A within −.057, B within +.307, cross −.185. Primary loss:
   oscillation without coherent tripod organization.
2. **Fixed motor map (layer ii):** all 18/18 target channels vary; mean target
   SD .133 rad and mean target range .588 rad. The map transfers activity but
   cannot repair phase.
3. **Hexapod physics (layer iii):** corrected mean joint-angle SD `.144` rad,
   target-to-joint correlation `.438`, and target tracking MAE `.110` rad.
   Body remains upright (minimum height `.129 m`, no fall) and travels `1.079 m`
   path / `.234 m` forward in the 8 s open-loop replay. Physics attenuates and
   distorts the signal, but does not erase it or create the missing tripod.
   Layer i is the primary bottleneck; ii/iii transmit the incoherence.

## E0–E4 branch completion

E0 after the completeness rebuild is `E0-ABSENT`; the corrected rerun is in
`results/path2/bodyfix_vnc_phase_probe_e0.json` and gives within cosine `+.047`,
cross cosine `−.052`, scalar `+.098`, with 3/6 within and 3/9 cross pairs
passing. Its phase matrix and verdict match the direct pre-E2 neural probe
(JSON field ordering differs). E2 found 212/212 published coordination
candidates selected, with no anatomically justified widening. The raw
inventory contains 12,745 cross-neuromere endpoint cells and 166,088
floor-qualified edge rows.

E3 drove the VNC with the full FlyWire forward pass at the exact DNg100/DNb08
type-and-side outputs. VNC and joint-target phase both remained absent:
within cosine −.181, cross cosine −.284, scalar +.103, with 2/6 within and
3/9 cross pairs passing tolerance. Corrected artifact:
`results/path2/bodyfix_e3_structured_descending.json`.

E4 added all 622 traced leg proprioceptors defined by the fixed
mechanosensory anatomy rule (campaniform, hair plate, chordotonal, or leg;
ProLN/MesoLN/MetaLN; root side L/R). On the corrected body, the 8 s closed-loop
replay remained phase-absent: within cosine `−.455`, cross cosine `−.245`,
scalar `−.210`, with 0/6 within and 4/9 cross pairs passing. The body traveled
`1.004 m` path and `.162 m` forward, remained upright at minimum height
`.137 m`, and did not produce a tripod. This is one declared fixed
contact/angle encoding, not an exhaustive sensory-model search. Corrected
artifact: `results/path2/bodyfix_e4_sensory_feedback.json`.

E4 still forces the pivot after body correction: E2 establishes published-set
completeness, E3 fails with computed descending drive, and the anatomically
mapped sensory path also fails to produce declared tripod phase. The body bug
was real, but correcting it does not close transfer within this fixed model/body
resolution.

Outcome B reservoir pivot is complete as graph/readout provenance. Its
body-deployment and locomotion numbers used the pre-fix stance and are
confounded/superseded; corrected B1–B3 did not trigger a rerun. R0 collected
3,000 post-burn full-VNC
states; R1 fit only closed-form ridge `W_out` to the fixed Path-1 template;
R2 deployed linear and one permitted 64-unit fixed-random-feature escalation;
R3 compared connectome, seeded random-weight, and exact degree-preserving
rewired reservoirs. No readout reached dynamic held-out
R² 0.8 or produced a stable tripod walk. The gate verdict is `READOUT TOO
WEAK`; final claim remains withheld pending approval.

See `PATH2_METHODS.md` and `results/path2/reservoir_3way.json` for the full
readout metrics and integrity record.

## Outcome C: CMA-ES gain-tuning (pre-fix; superseded)

> C0–C2 body-level values below used uniform squat stance and are retained as
> confounded provenance. Corrected B1–B3 did not trigger the expensive B4
> rerun, so no corrected C table exists.

The learning ladder to walking: zero-learn open-loop (E0–E4, absent) →
linear/nonlinear readout on frozen dynamics (reservoir R0–R3, `READOUT TOO
WEAK`) → tune global gains on the frozen VNC pattern (Outcome C, complete) →
learn VNC weights (Outcome D, complete). Each rung relaxes exactly one
constraint and is only taken because the rung below ceilinged.

Outcome C tunes ~18 global scalar gains only — descending drive, intersegmental
T1↔T2↔T3 coupling multipliers, and per-class proprioceptor feedback gain/sign.
The **VNC synaptic weight pattern stays frozen**; only these global multipliers
are searched. Fitness is `forward_distance_m − fall_penalty`, averaged over four
fixed seeds. There is **no tripod, catch, or target term** in the objective —
tripod is measured, never rewarded. Search is CMA-ES (pycma), popsize 14, 4 s
rollouts, wall-cap bounded. Harness `code/cmaes_walk.py`; C0 regression
reproduces E4 exactly (CSR pattern hash unchanged, dtype/scene/forward-axis
matched to E4). Fitness log: `results/path2/cmaes_fitness_connectome.csv`.

C1 result (complete): 22 generations, clean wall-cap stop at 7529 s. Best
forward distance 0.134644 m, produced by a step at generation 17 (0.097214 →
0.134644); flat generations 17–22, population mean noisy and low throughout
(0.000–0.031 m, no upward drift). That is ~13.5 cm in a 4 s rollout ≈ 3.4 cm/s —
better than E4 (0.095 m) and the reservoir, but still crawl-scale, not a
demonstrated coherent tripod. Best gene: `results/path2/cmaes_best_gene_connectome.json`.

Outcome C result (complete): C2 gave connectome, random-weight, and
degree-preserving-rewired conditions identical 18-gene, population-14,
four-second, 22-generation searches and four fixed eight-second evaluations.
Means with 95% percentile-bootstrap CIs (100,000 replicates, seed
`20260829`):

| condition | search best fitness | forward m | path m | tripod scalar | first positive generation | falls |
|---|---:|---:|---:|---:|---:|---:|
| connectome | .1346 | .0668 [.0435, .0962] | .5226 [.2744, .7708] | −.2526 [−.3772, −.0739] | 1 | 0/4 |
| random weights | .1760 | .1580 [.1176, .1784] | .1657 [.1257, .1878] | −.3533 [−.4658, −.2357] | 1 | 0/4 |
| rewired topology | .1633 | .1674 [.1642, .1713] | .2399 [.2260, .2539] | +.1777 [−.1443, .3909] | 2 | 0/4 |

All 12 evaluations report `E3-ABSENT`. Random minus connectome forward
difference is `+.0912 m` [+.0476, +.1349]; rewired minus connectome is
`+.1006 m` [+.0744, +.1217]. C2 verdict: **TUNED NEGATIVE; moat NULL**.
Movement appears after tuning, but no condition produces the declared tripod
and real wiring does not beat matched controls. Evidence:
`results/path2/cmaes_3way.json`, `results/path2/cmaes_3way_stats.json`,
`results/path2/cmaes_3way.png`, and the three `cmaes_walk_multiview_*.mp4`
files.

Do not write the final claim until the user approves the assembled evidence.

## Outcome D: learn VNC weights (pre-fix; superseded; final claim gated)

> Outcome D was not rerun after body correction because corrected B1–B3 still
> showed no tripod and no real-over-random advantage. Old learning outputs are
> retained for method provenance; body-level metrics are confounded/superseded.

Outcome C ceilings, so the remaining rung that could plausibly produce
walking is learning the VNC weights themselves — the FlyGM recipe (connectome
graph as architecture, imitation-learning init + PPO fine-tune on a
locomotion-only reward). This is where actual coordinated walking is expected to
live, at the explicit cost of the frozen-weight moat: the connectome would then
constrain *topology*, not values.

Reuse assessment (checked 2026-08-29): FlyGM
([arXiv 2602.17997](https://arxiv.org/abs/2602.17997),
[project](https://lnsgroup.cc/research/FlyGM)) publishes **source code, no
pretrained weights** — obtaining a model means running their two-stage pipeline
ourselves. Its decoder outputs **59 fly-body actuators** (leg joints + adhesion)
plus a separate 12-dim wing controller, jointly trained end-to-end with the
connectome for the `flybody` MuJoCo model. It does **not** map to the 18-DoF
hexapod and is not a liftable module. Therefore FlyGM is usable only as a
**reference recipe** for wiring connectome-graph → decoder → PPO, not as a
drop-in trained controller. Outcome D uses the available CPU-feasible fixed
low-rank/closed-form-decoder/ES implementation: learned VNC edge values are
the only relaxed graph object; objective is locomotion (distance + upright)
only, never prey capture; tripod measured, never rewarded. Target replay is
recorded at `results/path2/outcomeD_imitation_target.npz` and
`results/path2/outcomeD_target_replay.json`.

The first D2 exploratory run exposed a height-only upright-reward bug and was
discarded. The corrected run adds fixed root orientation `up_z >= 0.5` to the
existing height check, then reruns all three conditions fresh with unchanged
budgets, seeds, target, and graph controls. D1 imitation stays orientation-
valid in all 12 evaluations and produces `E3-PRESENT` command phase. D2
locomotion fine-tuning produces `E3-PRESENT` command phase in all 12, but
orientation-gated falls occur in connectome 3/4, random 4/4, and rewired 3/4.
The command phase is not treated as a body-level walking claim.

| condition | D1 held-out dynamic R² | D1 falls | D2 forward m [95% CI] | D2 return [95% CI] | D2 falls | max tilt deg [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| connectome | −.4633 | 0/4 | .5804 [.2849, .8848] | .2179 [−.1891, .6250] | 3/4 | 113.0 [73.1, 152.9] |
| random topology | −1.2354 | 0/4 | .8488 [.7307, .9188] | .3488 [.2307, .4188] | 4/4 | 160.3 [154.2, 163.6] |
| rewired topology | +.2219 | 0/4 | .5417 [.2215, .7690] | .1792 [−.2515, .5917] | 3/4 | 78.9 [53.5, 116.0] |

D2 verdict: **NO STABLE WALK; topology NULL**. The random forward mean is
not a success because all four episodes fail the physical upright audit. Full
records are in `results/path2/outcomeD_metrics_*.json`,
`results/path2/outcomeD_stats.json`, `results/path2/outcomeD_3way.*`, and
`results/path2/outcomeD_walk_multiview*.mp4`; stable D1 imitation videos are
`results/path2/outcomeD_imitation_multiview_*.mp4`.

Secondary checklist: saved models were replayed at held-out descending drive
`90` after training at drive `100` (`results/path2/outcomeD_heldout_drive_90.json`);
no retraining or weight change occurred. Four-seed forward CIs use 10,000
percentile-bootstrap replicates, seed `20260829`. D1 connectome/random/rewired
falls were 4/4, 0/4, 0/4; D2 falls were 4/4 in all three conditions. All
held-out command traces remain `E3-PRESENT`, which is not a body-level walking
claim. Outcome D has no prey input, so held-out prey-start generalization is
not applicable. Fixed-distance sample efficiency is not estimable: no
distance threshold or wall-time logger was declared before training. Equal
budget remains auditable from checkpoints: D1 12 × 8 candidates and D2 12 × 6
candidates evaluated on four fixed seeds per condition.

## Connectome coordination + continuous phase (Methods 1–2)

Both methods are zero-learning, use 20 paired 8-second episodes, share starts
and the unchanged engine, and enforce current `H_MIN=.08 m`.

Method 1 applies the fixed ±`.25`-cycle rule to the real VNC phase matrix.
It produces A = T1L/T1R/T2L/T2R and B = T3L/T3R, a 4/2 grouping rather than
a tripod. The shuffled control produces 3/3; seeded random control uses a
matched 4/2 split.

| source | grouping | path m [95% CI] | imposed command scalar | feet planted [95% CI] | falls |
|---|---|---:|---:|---:|---:|
| real VNC | 4 / 2 | `1.4982 [1.3503, 1.6385]` | `2.000` | `.2387 [.2001, .2757]` | 16/20 |
| shuffled VNC | 3 / 3 | `2.0729 [2.0658, 2.0802]` | `2.000` | `.4101 [.4083, .4120]` | 0/20 |
| random grouping | 4 / 2 | `1.9584 [1.9523, 1.9646]` | `2.000` | `.4158 [.4139, .4175]` | 0/20 |

Real minus shuffled path is `−.5747 m` (95% CI `[-.7172,-.4378]`, exact
`p=1.91×10⁻⁶`, Holm `p=3.81×10⁻⁶`); real minus random is `−.4603 m`
(95% CI `[-.6068,-.3202]`, exact `p=3.81×10⁻⁶`, Holm `p=3.81×10⁻⁶`).
The scalar is imposed engine-command coherence, not recovered VNC/body
coherence; it is 2.0 by construction in all three rows. The real source
phase audit remains absent (`3/6` within-tripod, `3/9` cross-tripod).

Method 2 applies analytic phase to 1.5–4.0 Hz band-passed antagonist motor
pools, with fixed `+π/2` mapping. Real per-leg frequencies are
`[2.153,2.750,2.266,2.550,2.490,2.692] Hz`; mean swing duty is `.4954`.
Continuous phase already contains inter-leg offsets, so Method 1's group
offset is not added.

| source | path m [95% CI] | phase scalar | mean duty | feet planted [95% CI] | falls | phase verdict |
|---|---:|---:|---:|---:|---:|---|
| real VNC | `1.3749 [1.3728, 1.3770]` | `.3174` | `.4954` | `.3984 [.3971, .3997]` | 0/20 | 20/20 `M2-PHASE-ABSENT` (`3/6`,`5/9`) |
| shuffled VNC | `1.5602 [1.5572, 1.5632]` | `−.3035` | `.5008` | `.4497 [.4485, .4510]` | 0/20 | 20/20 `M2-PHASE-ABSENT` (`2/6`,`3/9`) |
| random continuous phase | `1.2687 [1.2662, 1.2711]` | `.0724` | `.5000` | `.4224 [.4216, .4231]` | 0/20 | 20/20 `M2-PHASE-ABSENT` (`0/6`,`3/9`) |

Real minus shuffled path is `−.1852 m` (95% CI `[-.1884,-.1819]`, exact
`p=1.91×10⁻⁶`, Holm `p=3.81×10⁻⁶`); real minus random is `+.1063 m`
(95% CI `[+.1032,+.1097]`, exact `p=1.91×10⁻⁶`, Holm `p=3.81×10⁻⁶`).
Method 2 fixes duty but does not produce canonical phase.

The current-floor Path-3 binary control was rerun for fair validity accounting:
real path `.3784 [.2834,.4726]`, `3/20` falls, `20/20 P3-ABSENT`; random
path `.4750 [.3564,.5915]`, `11/20` falls. Method 1 does not beat this
baseline: it is unstable and loses to both controls. Method 2 has higher path
numerically, but does not beat the baseline's coordinated-walk endpoint:
continuous per-leg drive has `20/20 M2-PHASE-ABSENT`. Current artifacts are
`results/path2/connectome_coordination_method1.*`,
`results/path2/continuous_phase_method2.*`, and
`results/path2/hminfix_stepwalk_controls.*` plus their condition metrics and
four-view videos.

Layer i remains primary failure: VNC phase relations are absent before and
after decoding. Layer ii transmits usable oscillation and balanced duty but
does not repair coordination; Method 1 can impose scalar 2.0 and still falls
16/20, while Method 2 carries finite phase and target motion without a tripod.
Layer iii passes the clean engine gate (`2.038 m` path, `.121 m` minimum height,
no fall, scalar 2.0), so body/engine is not total-signal-loss bottleneck.
Effective connectome ownership remains about 25%: it owns descending drive,
VNC dynamics, motor activity, and steering, but not successful whole-body
coordination or rhythm. No 50–60% ownership upgrade is justified.

## Path 3: zero-learning step-timing decode (corrected; final claim gated)

Phase 0 verified the cross-body method sources in `PATH2_METHODS.md`. The
fixed decode uses the existing `mn_to_joint.py` map and corrected home-pose
body: per-leg mean trochanter
`Tr flexor`/`Acc. tr flexor` activity greater than mean `Tr extensor` activity
means swing; ties mean stance. `gait_engine.py` wraps the fixed Path-1
`HexapodGait` lift-swing-place/push-back primitive with mirrored joint signs. No
learning, reward, optimizer, phase selection, catch term, or distance term is
present. The brain continues to provide only the existing frozen steering
signals.
This is the single common direct antagonist pair across T1, T2, and T3; no
alternative pool assignment was tried or selected by outcome.

P1 decode gate: real VNC frequencies were `0.867, 0.867, 1.733, 0.933,
0.067, 0.133 Hz` for T1L/T1R/T2L/T2R/T3L/T3R, with swing duties `.500, .244,
.761, .635, .999, .979`; decoded state scalar `+.502`, only `2/6` within-
tripod and `4/9` cross-tripod pairs passing, `P1-ABSENT`. Shuffled VNC had no
step onsets and `P1-ABSENT`.

P2 clean-clock ceiling on corrected home pose: `1.6599 m` forward, `2.0380 m`
path, mean ride height `.1345 m`, minimum height `.1213 m`, no fall, canonical
tripod scalar `2.000`, `P2-ENGINE-PRESENT`. Commanded stance is exactly `3/6`;
physical mean foot contact is `.419`.

P3 representative corrected real closed loop: `.2022 m` forward, `.5420 m`
path, mean ride height `.1391 m`, minimum height `.0930 m`, no fall, realized
scalar `−.3058`, `P3-ABSENT`. The four-view video is
`eon-fly-brain/results/path2/bodyfix_stepwalk_multiview.mp4`; visual verdict is
irregular asymmetric stepping/lurching, not a coordinated tripod.

P4 used 20 shared static-prey starts, 8 s each, same engine/decode/steering/body
and fixed seeds. Means have 95% percentile-bootstrap CIs (100,000 replicates,
seed `20260829`). The authoritative rerun uses current `H_MIN=.08 m`; prior
`bodyfix_*` P4 fall labels used `.02 m` and are superseded provenance.
The predeclared four-seed evaluation set is `20260829`–`20260832`, present as
episodes 0–3 in every condition; P4 CIs use all 20, and videos are fixed-seed
exemplars rather than post-hoc best selections.

| timing source | forward m [95% CI] | path m [95% CI] | falls | upright | realized scalar [95% CI] |
|---|---:|---:|---:|---:|---:|
| real VNC | .1014 [.0659, .1391] | .3784 [.2834, .4726] | 3/20 | 17/20 | −.1317 [−.2070, −.0515] |
| shuffled VNC | −.0007 [−.0037, .0024] | .0470 [.0268, .0686] | 0/20 | 20/20 | ~0 [0, 0] |
| random matched-rate | .1329 [.0811, .1893] | .4750 [.3564, .5915] | 11/20 | 9/20 | −.0293 [−.1542, .1094] |

Primary path tests: real minus shuffled `+.3314 m` [+.2529, +.4070], exact
paired sign-flip `p=1.53e-5`, Holm `p=3.05e-5`; real minus random `−.0966 m`
[−.1395, −.0562], exact `p=4.58e-5`, Holm `p=4.58e-5`. All 60 realized
phase verdicts are `P3-ABSENT`. Real timing produces more movement than
shuffled but less than random matched-rate timing; it does not produce a
connectome-specific coordinated walk. Corrected condition videos and stats
are under `results/path2/hminfix_stepwalk_*`. This remains evidence assembly;
do not promote it to a final claim without user approval.

Path 3 localizes primary failure to layer i: real decoded states pass only
`2/6` within-tripod and `4/9` cross-tripod pairs; mean decoded swing fraction
is `.686` and imbalanced across legs. Layer ii carries variation (corrected
real episode motor temporal SD `.129`, target temporal SD `.086` rad) but does
not repair phase. Layer iii passes the corrected clean-clock ceiling (`1.660 m`
forward, scalar `2.000`) and remains upright in P3; decoded timing still
produces no tripod.

Corrected artifacts: `eon-fly-brain/results/path2/step_decode_p1.json`,
`bodyfix_gate_a.json`, `hminfix_stepwalk_metrics_{real,shuffled,random}.json`,
`hminfix_stepwalk_controls.{json,csv,png}`, and the three
`hminfix_stepwalk_multiview*.mp4` videos. Old `bodyfix_stepwalk_*`,
`gait_engine_sanity.json`,
`stepwalk_metrics_*`, and `stepwalk_multiview*.mp4` body outputs are retained as
superseded provenance.

## Reproduction commands

Run from `/Users/puneeth/Documents/Flycraft` with
`eon-fly-brain/.venv/bin/python`.

```bash
# Build final all-leg and T1 artifacts from official data.
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/fetch_manc.py \
  --data eon-fly-brain/data --neuromeres T1,T2,T3 --premotor_scope all \
  --synapse_floor 5 \
  --out eon-fly-brain/results/path2/all_legs_io_index.npz \
  --weights_out eon-fly-brain/results/path2/all_legs_weight_csr.pkl

# Run T1 rhythm, all-leg coordination, transfer-layer audit, and controls.
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/analyze_vnc_rhythm.py \
  --data eon-fly-brain/results/path2 \
  --weights_name t1_paper_weight_csr.pkl --index_name t1_paper_io_index.npz \
  --out eon-fly-brain/results/path2/t1_paper_rhythm_validation_heterogeneous \
  --model paper --drive 100 --gW .03 --substeps 4 --steps 1500 \
  --heterogeneity --seed 1
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/audit_vnc_legs.py \
  --data eon-fly-brain/results/path2 \
  --weights_name all_legs_weight_csr.pkl --index_name all_legs_io_index.npz \
  --out eon-fly-brain/results/path2/all_legs_coordination \
  --model paper --drive 100 --gW .03 --tau .02 --dt .005 --substeps 4 \
  --steps 1500 --heterogeneity --seed 1
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/audit_transfer_layers.py \
  --data eon-fly-brain/results/path2 --out eon-fly-brain/results/path2/transfer_layers
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/study_vnc.py \
  --n 20 --secs 3 --seed 20260829 \
  --out eon-fly-brain/results/path2/vnc_control_study_3s
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/analyze_control_study.py \
  --input eon-fly-brain/results/path2/vnc_control_study_3s.csv \
  --out eon-fly-brain/results/path2/vnc_control_study_3s_stats
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/audit_intersegmental.py \
  --data eon-fly-brain/results/path2 --index_name all_legs_io_index.npz \
  --out eon-fly-brain/results/path2/e2_intersegmental
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/probe_vnc_phase.py \
  --data eon-fly-brain/results/path2 \
  --out eon-fly-brain/results/path2/vnc_phase_probe_e0_after_e2
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/probe_structured_descending.py \
  --data eon-fly-brain/results/path2 --brain_data eon-fly-brain/data \
  --out eon-fly-brain/results/path2/e3_structured_descending
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/fetch_manc.py \
  --data eon-fly-brain/data --neuromeres T1,T2,T3 --premotor_scope all \
  --include_sensory_feedback --synapse_floor 5 \
  --out eon-fly-brain/results/path2/e4_sensory_io_index.npz \
  --weights_out eon-fly-brain/results/path2/e4_sensory_weight_csr.pkl
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_sensory_feedback.py \
  --data eon-fly-brain/results/path2 --brain_data eon-fly-brain/data \
  --out eon-fly-brain/results/path2/e4_sensory_feedback

# Reservoir pivot R0–R3 (CPU; R0 saves 56 MB states locally).
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/collect_reservoir_states.py
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/fit_readout.py
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/fit_readout.py \
  --nonlinear --out eon-fly-brain/results/path2/reservoir_readout_W_nonlinear.npz
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_reservoir_walk.py
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_reservoir_walk.py \
  --readout eon-fly-brain/results/path2/reservoir_readout_W.npz \
  --out eon-fly-brain/results/path2/reservoir_walk_metrics_linear.json \
  --video eon-fly-brain/results/path2/reservoir_walk_multiview_linear.mp4
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/reservoir_3way.py

# Pre-fix Outcome C C2 endpoint (retained; no corrected B4 rerun).
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/cmaes_3way.py \
  --workers 8 --max_generations 22 --max_hours 2

# Pre-fix Outcome D runs (retained; corrected B4 rerun skipped).
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/outcomeD_train.py \
  --condition connectome --fresh
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/outcomeD_train.py \
  --condition random --fresh
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/outcomeD_train.py \
  --condition rewired --fresh

# Re-evaluate saved models at held-out descending drive 90; no retraining.
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/outcomeD_train.py \
  --heldout-drive 90

# Corrected home-pose rerun before the final H_MIN correction; retained as
# bodyfix provenance and not overwritten.
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/step_decode.py \
  --data eon-fly-brain/results/path2 \
  --out eon-fly-brain/results/path2/step_decode_p1
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_stepwalk.py \
  --engine_only --secs 8 \
  --video eon-fly-brain/results/path2/bodyfix_engine_walk.mp4 \
  --out eon-fly-brain/results/path2/bodyfix_gate_a.json
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_stepwalk.py \
  --condition real --secs 8 \
  --video eon-fly-brain/results/path2/bodyfix_stepwalk_multiview.mp4 \
  --out eon-fly-brain/results/path2/bodyfix_stepwalk_metrics_real_episode.json
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/stepwalk_controls.py \
  --data eon-fly-brain/data --vnc_data eon-fly-brain/results/path2 \
  --n 20 --secs 8 \
  --timing_reference eon-fly-brain/results/path2/step_decode_p1.json \
  --out eon-fly-brain/results/path2/bodyfix_stepwalk_controls

# Corrected B2 five-condition control and statistics.
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/study_vnc.py \
  --n 20 --secs 3 --seed 20260829 \
  --out eon-fly-brain/results/path2/bodyfix_vnc_control_study_3s
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/analyze_control_study.py \
  --input eon-fly-brain/results/path2/bodyfix_vnc_control_study_3s.csv \
  --out eon-fly-brain/results/path2/bodyfix_vnc_control_study_3s_stats

# Corrected B3 E0/E3/E4 and transfer-layer audits.
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/probe_vnc_phase.py \
  --data eon-fly-brain/results/path2 \
  --out eon-fly-brain/results/path2/bodyfix_vnc_phase_probe_e0
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/probe_structured_descending.py \
  --data eon-fly-brain/results/path2 --brain_data eon-fly-brain/data \
  --out eon-fly-brain/results/path2/bodyfix_e3_structured_descending
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_sensory_feedback.py \
  --data eon-fly-brain/results/path2 --brain_data eon-fly-brain/data \
  --weights_name e4_sensory_weight_csr.pkl \
  --index_name e4_sensory_io_index.npz \
  --out eon-fly-brain/results/path2/bodyfix_e4_sensory_feedback
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/audit_transfer_layers.py \
  --data eon-fly-brain/results/path2 \
  --out eon-fly-brain/results/path2/bodyfix_transfer_layers

# Phase A and current-floor Path 3 control rerun.
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/preflight.py
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/stepwalk_controls.py \
  --data eon-fly-brain/data --vnc_data eon-fly-brain/results/path2 \
  --n 20 --secs 8 \
  --timing_reference eon-fly-brain/results/path2/step_decode_p1.json \
  --out eon-fly-brain/results/path2/hminfix_stepwalk_controls

# Connectome-set coordination and continuous phase methods.
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/connectome_coordination.py
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/continuous_phase.py
```

The current corrected synchronized four-camera Path-3 episode is
`results/path2/bodyfix_stepwalk_multiview.mp4`; the old
`results/path2/stepwalk_multiview.mp4` remains pre-fix provenance.

## Integrity checklist

Corrected rerun checks already recorded:

- Python compile for every `code/*.py` file;
- JSON/CSV parse and expected-row checks;
- IO index/body-ID equality, including 212 published candidates;
- equality of canonical and results graph inputs where intended;
- video duration and readable MP4 stream;
- zero catches/falls in frozen E0–E4/C2 outputs, plus explicit D2
  orientation-gated fall accounting;
- zero-learning scan scoped to frozen E0–E4 code/artifacts; D2 objective scan
  confirms no catch/prey/target/gait/coherence term;
- immutable target and exact row/column degree sequences for D controls;
- `git diff --check`.

All 65 JSON and 44 CSV files parse; all 212 published candidates are selected
with 0 omitted; VNC hash is
`018a4890e01f3dd608693e8a040103829e8523d134290ae918e6e5dd4d815506`; brain
hash is `f84f8dcd8397a43b8825ba45573cf14105f23d1846c82b58e4f07422f7586a88`.
The all-leg index hash is
`3f65f5b3ebed0d63161dca416bffa1a55afc0be4906113103426b16492d09449` and
matches the frozen graph record. The corrected engine, E3/E4, layer, Path 3,
Method-1, and Method-2 artifacts retain zero-learning or learning=false
protocol flags; source scan found no optimizer, backprop, reward, target-fit,
or performance-based choice in frozen rerun paths. New method/control videos
are readable H.264, 8.000 s; the preflight video is 4.000 s. Outcome C/D
remain separately labeled learning rungs and were not rerun.
The lexical zero-learning scan's only matches are the preflight inventory
filename `outcomeD_train.py` and Method-2's deterministic `np.polyfit` slope
used to estimate phase frequency; neither is training, optimization, or a
performance-derived choice.

Do not add constants, retune gains, delete superseded artifacts, or broaden
the claim before user approval.

## File map

| file | role |
|---|---|
| `eon-fly-brain/code/fetch_manc.py` | fixed official-MANC selection and signed CSR builder |
| `eon-fly-brain/code/vnc_stepper.py` | paper-rate VNC dynamics and shuffle control |
| `eon-fly-brain/code/mn_to_joint.py` | fixed motor-pool to 18-joint map |
| `eon-fly-brain/code/hexapod_gait.py` | MJCF-home stance and mirrored fixed gait primitive |
| `eon-fly-brain/code/run_vnc_chase.py` | brain/VNC/body episode runner and video renderer |
| `eon-fly-brain/code/audit_vnc_legs.py` | coupling, candidate, and phase audit |
| `eon-fly-brain/code/audit_transfer_layers.py` | VNC → target → physics audit and figure |
| `eon-fly-brain/code/analyze_control_study.py` | bootstrap CIs and paired tests |
| `eon-fly-brain/code/audit_intersegmental.py` | E2 intersegmental completeness inventory |
| `eon-fly-brain/code/probe_structured_descending.py` | E3 structured brain-drive phase probe |
| `eon-fly-brain/code/run_sensory_feedback.py` | E4 anatomical proprioceptive feedback replay |
| `eon-fly-brain/code/cmaes_walk.py` | Outcome C gain-tuning CMA-ES harness (frozen VNC pattern) |
| `eon-fly-brain/code/cmaes_3way.py` | matched Outcome C connectome/random/rewired endpoint |
| `eon-fly-brain/code/outcomeD_train.py` | Outcome D fixed-topology weight learning, controls, and audits |
| `eon-fly-brain/code/step_decode.py` | Path 3 fixed antagonist-sign step decoder and P1 gate |
| `eon-fly-brain/code/gait_engine.py` | Path 3 fixed per-leg wrapper over Path-1 `HexapodGait` |
| `eon-fly-brain/code/run_stepwalk.py` | Path 3 episode runner and P2/P3 metrics |
| `eon-fly-brain/code/stepwalk_controls.py` | Path 3 paired real/shuffled/random timing controls |
| `eon-fly-brain/code/preflight.py` | Gate-A home-pose, floor, graph-hash, and clean-engine checks |
| `eon-fly-brain/code/connectome_coordination.py` | Method-1 phase-derived grouping and controls |
| `eon-fly-brain/code/continuous_phase.py` | Method-2 continuous VNC phase source and controls |
| `PATH2_METHODS.md` | methods-complete working report |
