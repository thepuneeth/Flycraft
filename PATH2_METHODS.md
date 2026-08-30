# Path 2 — methods and current evidence

Status: corrected body-setup rerun complete through E0–E4, Path 3, and the
five-condition controls. Pre-fix body-level results are confounded/superseded;
the final claim remains pending approval.

## Corrected rerun summary (provisional; final claim withheld)

The body correction makes the declared clean-clock engine stand and travel
(`1.660 m` forward, `2.038 m` path in 8 s, minimum height `.121 m`, no fall),
but it does not make the frozen VNC produce an alternating tripod. All 212
published T1/T2/T3 coordination candidates are present, and corrected E0,
E3, E4, and Path 3 remain phase-absent. Real VNC wiring beats column-shuffled
VNC on open-loop path (`+0.2663 m`, 95% CI `[+.2611, +.2714]`, Holm
`p=1.91×10⁻⁶`), while brain-coupled VNC is below matched random descending
drive (`−.0621 m`, 95% CI `[−.1192, −.0092]`, Holm `p=.0422`).

This is a corrected evidence summary, not final claim wording. Outcome C/D
were not rerun: B1–B3 still show no tripod and no real-over-random advantage,
so B4 is skipped under the fixed protocol.

## Pre-fix working headline (confounded/superseded)

The earlier body-level headline and all body-dependent numbers below used the
uniform stance `(0.0, 0.6, -1.2)` for every leg. Retained for provenance only;
they are confounded by the body-setup bug and superseded by the `bodyfix_*`
artifacts. Graph-selection counts and direct VNC-only measurements are
retained as data provenance, not silently rewritten.

## Body-setup correction

The old shared setup applied `(0.0, 0.6, -1.2)` to all six legs. In the MJCF,
that put the body at about `.073 m` after settling instead of the model's
declared `home` pose (`z=.161 m` keyframe; `.144732 m` after the fixed `.7 s`
settle). Right legs are mirrored in `home`: coxa/femur/tibia are
`(+.7,-.6,+2.0)` for T1R/T2R/T3R, while left legs use
`(-.7,+.6,-2.0)` or the matching coxa value.

The fix reads `m.key("home").qpos[7:25]` at runtime, carries six per-leg
stance triples through the fixed motor map and PD servo, and applies the
declared mirrored coxa/femur swing signs. `kp=3`, `kd=.05`, torque limits,
tripod assignment, and steering remain fixed. The clean-clock body protocol
uses coxa amplitude `.35`, lift amplitude `.7`, base rate `.8`, drive rate `.5`
(`1.2 Hz` at forward `.8`); no value was changed after the corrected reruns.
Because mirrored coxa coordinates have opposite signs, commanded-joint phase
metrics use the fixed world-forward sign vector `[+1,-1,+1,-1,+1,-1]`; this is
a coordinate conversion, not a fitted parameter.

| gate/result | pre-fix retained value | corrected bodyfix value |
|---|---:|---:|
| 1 s static settle height | `.073 m` final/min | `.145 m` after `.7 s` settle |
| clean engine, 8 s forward | `.530 m` forward; `.074 m` minimum height | `1.660 m` forward; `.121 m` minimum height |
| clean engine path | `1.080 m` | `2.038 m` |
| clean engine phase | `P2-ENGINE-PRESENT`, scalar `2.000` under old coordinates | `P2-ENGINE-PRESENT`, scalar `2.000` after mirrored-coordinate normalization |
| Path 3 real episode | `.194 m` forward; `.461 m` path; scalar `+.112` | `.202 m` forward; `.542 m` path; scalar `−.306` |
| Path 3 real − random path | `−.0946 m` `[−.1265,−.0625]` | `−.0966 m` `[−.1395,−.0562]` |
| five-condition real − shuffled path | `+.1425 m` `[+.1397,+.1449]` | `+.2663 m` `[+.2611,+.2714]` |
| five-condition brain − random path | `−.0383 m` `[−.0775,+.0012]` | `−.0621 m` `[−.1192,−.0092]` |
| E0 VNC phase | `E0-ABSENT` | `E0-ABSENT`; within `3/6`, cross `3/9`, scalar `+.098` |
| E3 structured drive | `E3-ABSENT` | `E3-ABSENT`; within `2/6`, cross `3/9`, scalar `+.103` |
| E4 sensory feedback | `E4-ABSENT` | `E4-ABSENT`; within `0/6`, cross `4/9`, scalar `−.210` |

Corrected artifacts use the `eon-fly-brain/results/path2/bodyfix_*` prefix.
Original files remain in place for audit and are not evidence for a final
body-level conclusion.

## Question and pipeline

Path 2 tests whether an untrained FlyWire brain plus the official Male CNS
MANC VNC can drive a body it did not evolve for:

```text
prey -> FlyWire brain -> DNg100/DNb08 -> MANC VNC
     -> fixed motor-neuron map -> PD torque -> MuJoCo hexapod
```

The brain supplies descending drive and steering. The real MANC graph supplies
time-varying motor-neuron activity. `mn_to_joint.py` performs a fixed
five-fly-joint to three-hexapod-joint reduction; no learned readout or target
trajectory is present. MuJoCo receives torque commands from the fixed PD
servo.

## Data selection and Task A resolution

The graph builder consumes official Male CNS v1.0 Feather files in
`eon-fly-brain/data/manc/`. It keeps traced T1/T2/T3 leg motor neurons,
traced premotor predecessors, traced descending predecessors, the annotated
rhythm-core cells, and every traced T1/T2/T3 `vnc_intrinsic` cell whose type is
in the sequential, ipsilateral-all-leg, or bilateral coordination groups in
[Cheong et al., eLife 96084](https://elifesciences.org/articles/96084),
using the paper's [Supplementary file 8](https://cdn.elifesciences.org/articles/96084/elife-96084-supp8-v1.csv).

The rule is anatomical and fixed:

```text
traced AND vnc_intrinsic AND type in the 66 published coordination labels
AND soma neuromere in the requested T1/T2/T3 scope
```

The rebuilt all-leg artifact contains 212/212 current-v1.0 cells matching
that rule, including the two formerly omitted `IN12B013` T1 cells:

| body ID | type | neuromere | side | candidate-to-candidate cross-neuromere edge |
|---:|---|---|---|---:|
| 802193 | IN12B013 | T1 | R | 6 synapses to INXXX008 |
| 802470 | IN12B013 | T1 | L | 5 synapses to IN10B014 |

Thus the completeness change adds two cells and two cross-neuromere
candidate edges (11 synapses total). The graph now has 11,000 nodes, 492,722
edges at the five-synapse floor, 485,074 signed nonzero edges, and 260 motor
outputs. The T1 validation artifact has 4,418 nodes, 144,214 signed edges,
and 106 motor outputs.

The 66-label rule is broader than the requested neuromere scope. Sixty-two
labels occur in T1/T2/T3. `IN01A001` is absent from the v1.0 annotation;
`INXXX025`, `INXXX058`, and `INXXX115` occur only on A-segment annotations
and are outside this T1/T2/T3 experiment. These scope facts are recorded in
`all_legs_coordination.json`; the candidate-cell table is
`all_legs_coordination_published_coordination.csv`.

The E2 intersegmental inventory found 12,745 T1/T2/T3 cross-neuromere
endpoint cells and 166,088 floor-qualified raw edge rows. All 212 published
coordination candidates had cross-neuromere signal and were selected; 0 were
omitted. The omitted raw boundary pool is mostly non-published intrinsic and
ascending cells, so widening it would not be an anatomy-defined completion of
the published coordination set. Full cell/type inventories and decision are
in `eon-fly-brain/results/path2/e2_intersegmental.json`.

For E4, the separate feedback graph adds 622 traced
`vnc_sensory`/`mechanosensory_proprioceptive` cells whose subclass is
campaniform sensilla, hair plate, chordotonal organ, or leg, and whose entry
nerve maps to ProLN/MesoLN/MetaLN with root side L/R. It contains 11,264 nodes
and 488,158 signed nonzero edges. This is an anatomical branch artifact; the
canonical no-feedback graph remains the 11,000-node graph above.

## Frozen constants

Frozen baseline constants were not changed after the completeness rebuild.
E3/E4 add only explicitly declared, fixed anatomical input conventions; no
parameter sweep or performance-based adjustment was performed. Values below
were inherited from the frozen Path 2 protocol or are fixed coordinate
conventions.

| component | frozen values |
|---|---|
| VNC model | published paper-rate model; `tau=0.02`, `dt=0.005`, `substeps=4`, `gW=0.03` |
| VNC input | bilateral DNg100/DNb08 constant drive `100`; DNb08 gain `1`; threshold `7.5`; activation gain `1`; firing-rate cap `200` |
| VNC heterogeneity | enabled; seed `1`; fixed seeded cellwise tau/gain/threshold/cap perturbations |
| brain | `tau=0.02`, `dt=0.005`, `substeps=12`, `gW=0.006`, `drive_in=1.2`, `turn_gain=6`, `turn_sign=-1`, `fwd_gain=12`, fixed alignment gate |
| E3 descending coupling | exact brain DNg100/DNb08 type/side rates; VNC current scale `100` (existing frozen VNC drive scale) |
| E4 sensory coupling | `622` proprioceptors; campaniform = normalized foot load; hair plate/chordotonal/leg = mean absolute joint displacement; angle normalization `2.35619` rad; current scale `100` |
| motor map | stance from MJCF `home` keyframe; gains `(-0.55, 0.65, 0.65)` for coxa/femur/tibia |
| body servo | `ctrl_hz=50`, `kp=3.0`, `kd=0.05`; torque range comes from the supplied MJCF |
| graph | `premotor_scope=all`; absolute synapse floor `5`; rows postsynaptic, columns presynaptic |
| controls | 20 paired episodes/condition, 3 s, episode-seed base `20260829`; static prey placements and initial headings are shared across conditions |

The fixed map uses coxa, mean(trochanter/femur), and mean(tibia/tarsus);
positive-minus-negative motor pools use the annotation-derived signs. The
global coxa sign is a coordinate convention, not a post-hoc performance fix.

## VNC rhythm and coordination

The frozen T1 run uses 1,500 steps sampled every 0.02 s, discards the first
half for summary metrics, and applies no smoothing. It produces:

- dominant spectral peak: **2.467 Hz**;
- autocorrelation frequency: **2.500 Hz**, mean peak **0.342**;
- opposing-pool correlation: **−0.321**;
- left/right correlation: **+0.031**;
- active motor fraction: **0.368**;
- mean tail motor temporal SD: **0.0756**.

The column-shuffled T1 control has mean tail motor SD `0.00624`, active
fraction `0.104`, opposing-pool correlation `+0.161`, and left/right
correlation `−0.085`. Its isolated spectrum is not itself a success
criterion; the paired behavior and motor-output tests are the inferential
comparison against shuffle.

The all-leg run has nonzero edge mass between every T1/T2/T3 neuromere pair,
but its coxa phase proxy is not a coherent tripod:

| phase group | mean correlation |
|---|---:|
| tripod A within | −0.057 |
| tripod B within | +0.307 |
| cross-tripod | −0.185 |

The corrected mechanistic figure is
[`results/path2/bodyfix_transfer_layers.png`](eon-fly-brain/results/path2/bodyfix_transfer_layers.png);
the pre-fix figure remains at
[`results/path2/transfer_layers.png`](eon-fly-brain/results/path2/transfer_layers.png)
as superseded provenance;
the connectivity/phase figure is
[`results/path2/all_legs_coordination.png`](eon-fly-brain/results/path2/all_legs_coordination.png).

### E0–E4 phase branch

E0 after E2 is `E0-ABSENT`. Its full six-by-six phase matrix is byte-identical
to the first E0 probe: only 3/6 within-tripod pairs pass the ±0.15-cycle
in-phase tolerance, and only 3/9 cross-tripod pairs pass the 0.5±0.15-cycle
antiphase tolerance. E2 therefore stops without widening the published
coordination rule.

E3 replaces flat VNC input with the full FlyWire forward pass's exact
DNg100/DNb08 type-and-side outputs. Both VNC motor output and joint-target
coxa phase remain absent: within cosine `−0.181`, cross cosine `−0.284`,
tripod scalar `+0.103`, with 2/6 and 3/9 pairs passing. This corrected result
is unchanged because the E3 neural drive and fixed map do not depend on body
height.

E4 adds the 622-cell anatomical proprioceptor set and closes the loop through
the MANC graph. Both phase layers remain absent on the corrected body: within
cosine `−0.455`, cross cosine `−0.245`, tripod scalar `−0.210`, with 0/6 and
4/9 pairs passing. The 8 s replay travels `1.004 m` path and `0.162 m`
forward, remains upright (minimum height `.137 m`), and has mean motor,
target, and physical joint temporal SDs of `.148`, `.126` rad, and `.124` rad
respectively. Corrected artifacts are
`eon-fly-brain/results/path2/bodyfix_vnc_phase_probe_e0.json`,
`eon-fly-brain/results/path2/e2_intersegmental.json`,
`eon-fly-brain/results/path2/bodyfix_e3_structured_descending.json`, and
`eon-fly-brain/results/path2/bodyfix_e4_sensory_feedback.json`.
E4 is a declared fixed contact/angle encoding of the available proprioceptor
classes, not an exhaustive search over sensory models.

E4 still forces the same pivot after body correction: E2 establishes published-
set completeness, E3 fails with computed descending drive, and the anatomically
mapped sensory path also fails to produce declared tripod phase. The body bug
was real, but correcting it does not close transfer within this fixed
model/body resolution.

## Paired control study

Each condition uses the same 20 episode starts on the corrected home-pose
body. Means below have 95% percentile-bootstrap CIs (100,000 replicates, seed
`20260829`). Catch and fall rates are zero in every condition; each has Wilson
95% CI `[0.000, 0.161]`.

| condition | path (m) | forward (m) | displacement (m) | min prey distance (m) | motor temporal SD |
|---|---:|---:|---:|---:|---:|
| real VNC open-loop | 0.351 [0.348, 0.353] | 0.042 [0.039, 0.045] | 0.093 [0.090, 0.096] | 1.352 [1.250, 1.453] | 0.140 [0.140, 0.140] |
| brain-coupled VNC | 0.286 [0.224, 0.340] | 0.022 [0.007, 0.038] | 0.046 [0.034, 0.059] | 1.366 [1.264, 1.467] | 0.116 [0.093, 0.135] |
| shuffled VNC | 0.084 [0.080, 0.090] | 0.016 [0.012, 0.020] | 0.041 [0.038, 0.045] | 1.395 [1.295, 1.493] | 0.067 [0.067, 0.067] |
| random descending | 0.348 [0.317, 0.378] | 0.027 [0.010, 0.045] | 0.056 [0.045, 0.069] | 1.363 [1.264, 1.460] | 0.133 [0.128, 0.138] |
| Path 1 hand CPG | 0.179 [0.129, 0.230] | 0.067 [0.030, 0.107] | 0.077 [0.041, 0.115] | 1.349 [1.232, 1.459] | not applicable |

Primary paired tests are exact two-sided sign-flip tests on path, with Holm
adjustment across the three predeclared comparisons:

| comparison | mean difference (m) | 95% CI | exact p | Holm p |
|---|---:|---:|---:|---:|
| real VNC − shuffled VNC | +0.2663 | [+0.2611, +0.2714] | 1.91×10⁻⁶ | 5.72×10⁻⁶ |
| brain-coupled − random descending | −0.0621 | [−0.1192, −0.0092] | 0.0422 | 0.0422 |
| open-loop VNC − Path 1 hand CPG | +0.1713 | [+0.1193, +0.2225] | 5.72×10⁻⁶ | 1.14×10⁻⁵ |

The motor temporal-SD difference for real VNC versus shuffled VNC is
`+0.0730` with exact paired `p=1.91×10⁻⁶`; its path difference is the
load-bearing wiring positive. The brain-coupled path is lower than matched
random descending drive, with a corrected CI excluding zero. This is a
short-horizon path result, not a successful pursuit or walking result.

Full metric CIs and all predeclared pairwise tests are in
`eon-fly-brain/results/path2/bodyfix_vnc_control_study_3s_stats.json`.

## Transfer-layer localization

The corrected fixed open-loop VNC run was replayed for 8 s with frozen VNC/map
constants and the home-pose body:

1. **Layer i — VNC dynamics:** motor output is dynamic (`260` outputs; mean
   temporal SD `0.152`), and T1 has a reproducible oscillatory peak. The
   all-leg phase audit nevertheless gives opposite signs for the two
   within-tripod groups (`−0.057` versus `+0.307`) and only moderate negative
   cross-tripod correlation (`−0.185`). This is oscillation without coherent
   alternating tripod organization. It remains the primary failure layer.
2. **Layer ii — fixed motor map:** all `18/18` target channels vary; mean
   target temporal SD is `0.133` rad and mean target range is `0.588` rad.
   The map transmits VNC variation but cannot repair its phase structure.
3. **Layer iii — body physics:** mean joint-angle SD is `0.144` rad,
   target-to-joint correlation is `0.438`, and the corrected body remains
   upright (`min height .129 m`, no fall). It travels `1.079 m` of path and
   `.234 m` forward in this 8 s open-loop run. Physics attenuates/distorts the
   target (`.110` rad mean target-tracking MAE), but does not erase the signal
   or create the missing tripod.

The corrected mechanistic reading is therefore: the real VNC carries dynamic
activity and the fixed map/body transmit it, but the expanded VNC does not
organize that activity into the phase pattern required by this hexapod. The
body-setup confound is removed; the remaining negative is narrower than “the
fly VNC cannot produce rhythm” and narrower than “all foreign bodies fail.”

## Outcome C — frozen-pattern CMA-ES gain search (pre-fix; superseded)

> These search/evaluation numbers were produced with the uniform squat stance
> and are retained as confounded provenance. B4 was not rerun: corrected B1–B3
> still show no tripod and no real-over-random advantage.

C0 reproduced the E4 connectome endpoint exactly before search. C1 then
searched the same 18 global gains on the connectome for 22 generations under a
two-hour wall cap. C2 ran the identical population, seed set, four-second
fitness rollout, fixed fall penalty, and 22-generation endpoint for connectome,
random-weight, and degree-preserving-rewired VNC conditions. The VNC weight
pattern was not changed; fitness was only forward displacement minus the fixed
fall penalty. Tripod coherence was measured after search and never entered
fitness.

The C2 evaluation used four fixed eight-second seeds. Values below are means
with 95% percentile-bootstrap CIs (100,000 replicates, seed `20260829`); a
positive fitness threshold only means positive forward-minus-fall score, not a
declared gait.

| condition | search best fitness (m) | forward (m) | path (m) | tripod scalar | first positive generation | falls |
|---|---:|---:|---:|---:|---:|---:|
| connectome | .1346 | .0668 [.0435, .0962] | .5226 [.2744, .7708] | −.2526 [−.3772, −.0739] | 1 | 0/4 |
| random weights | .1760 | .1580 [.1176, .1784] | .1657 [.1257, .1878] | −.3533 [−.4658, −.2357] | 1 | 0/4 |
| rewired topology | .1633 | .1674 [.1642, .1713] | .2399 [.2260, .2539] | +.1777 [−.1443, .3909] | 2 | 0/4 |

Per-seed evaluation phase verdict is `E3-ABSENT` for all 12 episodes. The
rewired condition has a positive mean scalar only because its scalar is a
post-hoc summary; its CI crosses zero, and it does not satisfy the declared
within-tripod/cross-tripod phase pattern. Random and rewired controls are not
inferior to the connectome on forward distance: paired mean differences are
`+0.0912 m` [+.0476, +.1349] and `+0.1006 m` [+.0744, +.1217], respectively.

C2 verdict: **TUNED NEGATIVE; moat NULL**. Gain search finds movement in every
condition, but no condition produces the declared tripod and the connectome
does not beat matched controls. This closes the frozen-weight tuning rung
without turning a positive displacement score into a walking claim. Full
search curves, per-seed metrics, and CI calculations are in
`eon-fly-brain/results/path2/cmaes_3way.json`,
`eon-fly-brain/results/path2/cmaes_3way_stats.json`, and
`eon-fly-brain/results/path2/cmaes_3way.png`.

## Frozen-reservoir readout pivot (R0–R3; pre-fix body results superseded)

> Reservoir reconstruction metrics are graph/readout provenance. Its body
> deployments and locomotion numbers used the pre-fix stance and are
> confounded/superseded; no corrected deployment was warranted under B4.

The completed pivot kept every MANC weight and all Path 2 rate-model constants
frozen. R0 drove the 11,000-neuron all-leg graph with constant tonic drive 100,
discarded 3,000 steps, then recorded 3,000 full-population activation vectors
at 50 Hz. Of 11,000 columns, 5,243 varied. Column means and scales came only
from the first 70% training window. The source CSR SHA-256 was
`018a4890e01f3dd608693e8a040103829e8523d134290ae918e6e5dd4d815506`.

The target was declared before fitting: the existing Path-1 `HexapodGait`
defaults at straight `forward=0.8`, `turn=0`, 50 Hz, starting from phase zero.
The target was never changed. The known fixed stance was treated as a joint
coordinate convention; the sole fit object was the state-to-18-joint-deviation
matrix `W_out`. Ridge used the algebraically equivalent exact dual closed form.
The first 70% of time fitted `W_out`; the final 30% selected λ from the fixed
grid `10^-6 ... 10^6`. Walking, distance, prey, and reward never entered the
fit or λ selection.

### R1 reconstruction gate

The selected linear λ was `10^6`. Mean train/held-out R² over all 18 channels
was `0.1182/0.1112`. The six coxa channels each had held-out R² `0.0075`; the
six femur channels ranged from `-0.6744` to `-0.6736`; the six constant tibia
channels were reconstructed exactly and therefore report `1.0`. Across only
the 12 dynamic channels, held-out R² was `-0.3332`. Thus the all-channel mean
is inflated by constant tibia channels and is not evidence of useful template
reconstruction.

This triggered the one permitted capacity escalation: 64 fixed Gaussian
random features (`tanh`, seed `20260829`) followed by the same closed-form
ridge fit. The random projection was not learned. Its selected λ was `10^5`;
mean train/held-out all-channel R² was `0.1076/0.1065`, and dynamic-channel
held-out R² was `-0.3402`. No optimizer, backpropagation, RL, reward, or task
loss was introduced. The escalation did not improve reconstruction.

### R2 deployment gate

The live reservoir was burned in under the same tonic drive, then the frozen
readout generated 18 targets for the existing PD servo. The existing FlyWire
forward pass supplied its exact unlearned forward and left/right steering
outputs, as in `run_vnc_chase.py`; no steering signal was learned. The
reservoir burn-in used tonic drive 100. The exact E0 phase procedure scored
the readout targets.

| readout | tripod scalar | target SD (rad) | forward (m) | path (m) | fall/catch | video verdict |
|---|---:|---:|---:|---:|---|---|
| linear | 2.000 | 75.77780 | +0.12481 | 1.10284 | no/no | unstable out-of-distribution target explosion; no tripod walk |
| 64-unit random features | 2.000 | 0.00120 | -0.00203 | 0.00432 | no/no | near-static sprawl; no visible walk |

All 6/6 within-tripod and 9/9 cross-tripod phase pairs pass because the tiny
readout signals retain the template's relative signs. The nonlinear readout's
amplitude and body motion show why phase coherence alone is insufficient. The
linear readout instead leaves the training distribution and explodes; its
extra displacement is not a stable gait. Neither readout produced a tripod
walk. The synchronized 8 s, 1280×960, four-view escalation video is
`eon-fly-brain/results/path2/reservoir_walk_multiview.mp4`.

### R3 three-way prior gate

All conditions used identical drive, burn-in, template, split, λ grid,
heterogeneity, ridge procedure, body, and steering. `random` keeps the exact
sparse graph and permutes its signed edge weights with seed `20260829`, so its
size, degree structure, and empirical weight multiset match the connectome.
`rewired` performs one successful directed endpoint swap per edge (seed
`20260829`), preserving every node's in/out degree and the exact weight
multiset. Generalization used a declared held-out tonic drive of 90. Effective
rank was the number of descending kernel modes needed to reach fixed
dynamic-channel held-out R² `0.8`.

| reservoir | readout | held-out R², all | held-out R², dynamic | effective rank to .8 | drive-90 dynamic R² | forward (m) | path (m) | target SD (rad) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| connectome | linear | 0.1112 | -0.3332 | not reached | -75157.4582 | +0.12481 | 1.10284 | 75.77780 |
| random | linear | 0.3187 | -0.0219 | not reached | -0.0136 | +0.00260 | 0.01110 | 0.01350 |
| rewired | linear | 0.0876 | -0.3687 | not reached | -0.3312 | -0.00304 | 0.00550 | 0.01497 |
| connectome | random features | 0.1065 | -0.3402 | not reached | -0.3405 | -0.00203 | 0.00432 | 0.00120 |
| random | random features | 0.1064 | -0.3405 | not reached | -0.3404 | -0.00204 | 0.00430 | 0.00010 |
| rewired | random features | 0.1154 | -0.3269 | not reached | -0.3288 | -0.00147 | 0.00471 | 0.01534 |

R3 gate verdict: **READOUT TOO WEAK**. Random linear reconstruction is less
bad than connectome reconstruction, so these data do not support a special
connectome prior. No condition reaches the fixed reconstruction target or
produces a stable visible tripod walk; the connectome linear rollout's
displacement comes with 75.8 rad target explosion. This is a gate verdict, not
final headline wording; final claim remains withheld pending user approval.

## Outcome D method sources and results (pre-fix body results superseded)

> Outcome D was not rerun after the body correction because corrected B1–B3 did
> not meet the B4 trigger. Its learning methods and old numbers remain for
> provenance only; all pre-fix body-level values are confounded/superseded.

Method study performed 2026-08-29. These sources inform implementation choices
only; they do not change the frozen VNC constants, body, candidate rule, or
Outcome C objective.

Local training-stack check: PyTorch `2.13.0`, MuJoCo `3.12.0`, and `cma`
`4.4.4` import successfully; JAX, Optax, Stable-Baselines3, Ray, and
Gymnasium are unavailable in the frozen venv.

### FlyGM

[FlyGM](https://arxiv.org/html/2602.17997) uses directed connectome message
passing, separates afferent/intrinsic/efferent roles, decodes efferent states
to actions, and stages imitation learning before PPO. Its reported controls
include connectome, degree-preserving rewiring, Erdős–Rényi random graphs, and
an MLP under matched training. Reuse: fixed sparse topology, explicit
connectome/random/rewired controls, and a declared imitation-to-task-learning
sequence. Avoid: its flybody action space, task reward, and expert data; D is
an 18-joint hexapod transfer test. D1 target is only the declared Path-1 hand
CPG trajectory `q*(t)`. D2 objective is locomotion-only forward displacement
with upright/fall handling; catch, prey distance, target tracking, gait, and
phase/coherence terms are excluded.

### C. elegans topology-constrained learning

[Topology-constrained C. elegans study](https://www.biorxiv.org/content/10.1101/2025.07.21.665845v1.full)
learns synaptic weights while preserving anatomical topology and synapse-type
proportions, using an explicit behavior error. Reuse: learn edge values on a
fixed anatomical support and keep topology/initialization/control provenance
auditable. Avoid: organism-specific targets and backward/forward locomotion
tasks. D keeps the real signed MANC weights as the connectome initialization;
controls use declared matched-size random or degree-preserving supports and
the same optimizer budget, target, and seeds.

### NeuroMechFly v2 and flybody

[NeuroMechFly v2](https://www.nature.com/articles/s41592-024-02497-y) and
[flybody](https://github.com/TuragaLab/flybody) show useful imitation/RL
patterns for embodied fly models, but also use fly-specific bodies, action
spaces, and task definitions. Reuse: checkpointed staged training and an
explicit separation between imitation target and downstream task objective.
Avoid: importing their designed CPG, fly-specific action dimension, or
navigation/catch reward. The local stack has PyTorch, MuJoCo, and `cma`, but
not JAX, Optax, Stable-Baselines3, Ray, or Gymnasium; D therefore uses the
smallest locally reproducible PyTorch/ES path and records that method honestly.

### Connectome-CPG context

[Connectome-CPG](https://www.biorxiv.org/content/10.1101/2025.09.12.675944v1.full)
is used as context for testing whether connectome-derived dynamics can support
motor rhythm. It is not used as a controller or as a gait-label source. D
measures tripod structure only after training; no phase, coherence, or gait
quantity enters either D objective.

### Declared target and objectives

Phase 2 writes `results/path2/outcomeD_imitation_target.npz`: 18 joint targets
from the Path-1 hand CPG at the fixed 50 Hz control rate. D1 minimizes only
trajectory error to that frozen `q*(t)`. D2 starts from the D1 checkpoint and
optimizes only locomotion forward displacement while retaining upright/fall
handling. The same target, budget, seeds, checkpoint cadence, and audit rules
apply to connectome, random, and rewired conditions. Tripod metrics are
post-hoc diagnostics.

The declared 400-step (8 s) target replay is upright and yields `0.5356 m`
forward distance, `1.1036 m` path, and tripod coherence scalar `2.000`; this
is the reference kinematic ceiling, not a learned score or D reward. The
replay record is `results/path2/outcomeD_target_replay.json`.

Outcome D uses an 8-dimensional fixed random low-rank edge-value basis over the
11,000-node, 485,074-edge canonical all-leg graph, a closed-form ridge decoder
during D1, and CMA-ES weight fine-tuning during D2 because no PPO stack is
installed and full sparse-CSR autograd is impractical on this CPU.
All D phase/coherence diagnostics call the canonical `probe_vnc_phase.py`
procedure; E3 supplies only its own verdict labels.
The measured connectome weights initialize the connectome condition; controls
use seeded randomized values, an exact degree-sequence configuration-model
support (random), or degree-preserving rewiring (rewired). No graph
topology, body constant, PD constant, or objective constant is selected from
post-hoc performance.

The first D2 exploratory run exposed a reward-physics bug: height alone marked
inverted bodies as upright. Those outputs are discarded. Before the corrected
run, the fixed physical validity rule became `root up_z >= 0.5` and height
`>= 0.02 m`; a failed episode receives the already-declared `−0.5` penalty and
does not receive the `+0.05` upright bonus. All three conditions were then
rerun fresh through D1/D2 with the same seeds and budgets. No gait term was
added.

### Outcome D results

D1 imitation replay produces the target-like command phase in all 12
condition/seed evaluations (`E3-PRESENT`) and remains orientation-valid in all
12. This is decoder imitation, not evidence that the topology is special: the
rewired condition has the highest dynamic held-out R², while every condition
reproduces the same command-level phase pattern. D1 forward means are
`0.3698 m` [0.3609, 0.3768] for connectome, `0.4091 m` [0.3774, 0.4397] for
random, and `0.3629 m` [0.3445, 0.3807] for rewired.

D2 optimizes only forward displacement plus the fixed upright/fall terms. The
table reports four-seed means with 95% percentile-bootstrap CIs (100,000
replicates, seed `20260829`). Forward distance is shown for comparability, but
upright-gated locomotion return and failure count determine whether replay is
valid.

| condition | D1 held-out dynamic R² | D1 falls | D2 forward (m) | D2 return | D2 falls | D2 max tilt (deg) |
|---|---:|---:|---:|---:|---:|---:|
| connectome | −.4633 | 0/4 | .5804 [.2849, .8848] | .2179 [−.1891, .6250] | 3/4 | 113.0 [73.1, 152.9] |
| random topology | −1.2354 | 0/4 | .8488 [.7307, .9188] | .3488 [.2307, .4188] | 4/4 | 160.3 [154.2, 163.6] |
| rewired topology | +.2219 | 0/4 | .5417 [.2215, .7690] | .1792 [−.2515, .5917] | 3/4 | 78.9 [53.5, 116.0] |

All 12 D2 command traces also score `E3-PRESENT`, with scalar `2.000`, but
that metric is computed on the decoder's 18-joint command trace using the
existing phase procedure. It does not override the physical orientation
audit. Connectome and rewired retain only 1/4 orientation-valid D2 episodes;
random retains 0/4. The high random forward mean is therefore a failed-body
outcome, not a walking result. D2 verdict: **NO STABLE WALK; topology NULL**.
The learned decoder can express the declared kinematic pattern, but the
locomotion fine-tune does not produce a robust body-level transfer and no
condition separates as a topology-specific result. Full per-seed records,
checkpoints, videos, curves, and CIs are in
`eon-fly-brain/results/path2/outcomeD_metrics_{connectome,random,rewired}.json`,
`eon-fly-brain/results/path2/outcomeD_stats.json`,
`eon-fly-brain/results/path2/outcomeD_3way.{json,csv,png}`, and
`eon-fly-brain/results/path2/outcomeD_walk_multiview*.mp4`.

### Outcome D secondary checklist

The saved D1/D2 models were replayed once at held-out descending drive `90`
after training at drive `100` (`outcomeD_heldout_drive_90.json`). This was an
evaluation-only change: no model was retrained and no weight changed. Values
are four-seed means with 95% percentile-bootstrap CIs (10,000 replicates,
seed `20260829`); falls use the same orientation/height gate as D2.

| condition | D1 forward m [95% CI] | D1 falls | D2 forward m [95% CI] | D2 return [95% CI] | D2 falls |
|---|---:|---:|---:|---:|---:|
| connectome | .3136 [.1404, .4745] | 4/4 | .1984 [−.2748, .6721] | −.3016 [−.7748, .1721] | 4/4 |
| random topology | .4339 [.2905, .6245] | 0/4 | .4752 [.3602, .5907] | −.0248 [−.1398, .0907] | 4/4 |
| rewired topology | .4028 [.1955, .5856] | 0/4 | .0709 [−.0527, .2290] | −.4291 [−.5527, −.2710] | 4/4 |

All held-out-drive command traces still score `E3-PRESENT`; this remains a
decoder-command diagnostic, not a body-level walking result. Outcome D has no
prey state or prey input, so held-out prey-start generalization is not
applicable. A fixed-distance sample-efficiency endpoint and wall-time logger
were not declared before training; therefore updates/wall-time to threshold
are not estimable without adding a post-hoc criterion. The matched training
budget is nevertheless fixed and equal: D1 used 12 generations × 8 candidates
per condition, and D2 used 12 generations × 6 candidates evaluated on the four
fixed seeds.

## Path 3 method sources

Method study performed 2026-08-30. These sources set the interface only; they
do not change the frozen graph, VNC constants, steering, body, or evaluation.

### Sun et al., *C. elegans* multi-legged robot

The paper decodes motor-neuron voltage into discrete leg-region commands using
thresholded activation, including OLQ-RMD pairs mapped to four robot leg
regions. A fixed robot-specific kinematic/inverse-kinematic layer turns each
extend/contract command into leg motion. It explicitly rejects direct whole-
brain-to-joint transfer when robot and worm kinematics differ. Reuse: discrete
neural command plus body-specific solver. Avoid: its worm-specific regions and
stimulus thresholds. The paper states the circuit is used directly, with no
training: [PMC11201571](https://pmc.ncbi.nlm.nih.gov/articles/PMC11201571/).

### NeuroMechFly v2 hybrid controller

The hybrid controller combines a tripod stepping pattern with local sensory
corrections for overstretch and stumbling. Its clean step is a per-leg
swing/stance trajectory with lift during swing and ground-contact/adhesion
during stance. Reuse: shape of a body-appropriate step and the idea that stuck
legs need sensory rescue. Avoid: importing its designed CPG or corrections as
Path 3 timing. This is a reference controller, not a Path 3 learned component:
[hybrid tutorial](https://neuromechfly.org/tutorials/4c_hybrid_controller/),
[Nature Methods paper](https://www.nature.com/articles/s41592-024-02497-y).

### NCAP

NCAP uses a small circuit motif as a structural prior while the body and its
mechanics produce locomotion; the neural architecture is organized into
rhythm-generation and pattern-formation roles. Reuse: keep neural structure
and body-specific mechanics as separate layers. Avoid: its trained ANN and
quadruped task as evidence for this zero-learning experiment. Path 3 uses the
connectome only for step timing and leaves step shape fixed:
[arXiv:2410.07174](https://arxiv.org/abs/2410.07174).

### Connectome-CPG

The connectome-CPG study identifies a core fly walking rhythm circuit and tests
descending DNb08 influence on rhythmic leg output. Reuse: treat network rhythm
as an empirical timing signal, then evaluate embodiment separately. Avoid:
turning its fly-specific CPG result into a tripod label or a learned body
controller. Path 3 therefore measures decoded states post hoc and adds no
training, reward, or phase selection:
[bioRxiv 2025.09.12.675944](https://www.biorxiv.org/content/10.1101/2025.09.12.675944v1.full).

**Path 3 gate choice:** the antagonist-sign decode and the Path-1
`HexapodGait` swing/stance primitive require zero training. For each leg,
`swing` means mean `Tr flexor`/`Acc. tr flexor` activity exceeds mean `Tr
extensor` activity; otherwise `stance`. The fixed engine supplies the
lift-swing-place/push-back trajectory; the frozen brain supplies steering only.

## Path 3 — zero-learning step-timing decode

Implementation uses `code/step_decode.py`, `code/gait_engine.py`, and
`code/run_stepwalk.py`. The six motor pools are selected once from the existing
`mn_to_joint.py` map: positive trochanter `Tr flexor`/`Acc. tr flexor` channels
are the swing pool and negative trochanter `Tr extensor` channels are the
stance pool. The rule is `swing = pool_swing > pool_stance`, with ties stance.
This is the single common direct antagonist pair exposed across T1, T2, and
T3; no alternative pool assignment was tried or selected by outcome.
It uses no threshold, fit, optimizer, reward, gait score, catch score, or
distance score. The step engine calls the fixed Path-1 `HexapodGait` shape with
the corrected MJCF-home stance and mirrored left/right joint signs; state
transitions only choose phase 0 (swing) or π (stance), then the inherited fixed
rate advances to the half-cycle endpoint.

### P1 decode gate

The frozen all-leg paper-rate VNC used 1,500 samples at 50 Hz, heterogeneity
seed 1, constant drive 100, and the same column shuffle implementation used by
Path 2. Numbers below are measured on the second half; decoded-state coherence
uses the canonical `probe_vnc_phase.py` metric.

| timing source | T1L Hz / duty | T1R Hz / duty | T2L Hz / duty | T2R Hz / duty | T3L Hz / duty | T3R Hz / duty | scalar | within / cross | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| real VNC | .867 / .500 | .867 / .244 | 1.733 / .761 | .933 / .635 | .067 / .999 | .133 / .979 | .502 | 2/6, 4/9 | P1-ABSENT |
| shuffled VNC | 0 / .000 | 0 / .000 | 0 / .000 | 0 / 1.000 | 0 / .000 | 0 / 1.000 | .000 | 6/6, 0/9 | P1-ABSENT |

The real and shuffled state sequences are therefore not indistinguishable,
but the real sequence is not an alternating tripod. Full pools, traces, and
protocol are in `results/path2/step_decode_p1.json` and
`results/path2/step_decode_p1_traces.npz`.

### P2 fixed-engine gate

A clean external alternating-tripod clock drove the same `StepGaitEngine` for
8 s with forward input `.8`; no neural signal entered this ceiling run. On the
corrected home-pose body it produced `1.6599 m` forward distance, `2.0380 m`
path, mean ride height `.1345 m`, minimum height `.1213 m`, no fall, physical
mean foot contact `.419` (commanded stance exactly `3/6`), and canonical tripod
scalar `2.000` (`P2-ENGINE-PRESENT`). The corrected record and plain engine
video are `results/path2/bodyfix_gate_a.json` and
`results/path2/bodyfix_engine_walk.mp4`.

### P3 real closed loop

The frozen brain supplied only the existing forward/turn steering pass. Each
50 Hz step decoded live VNC antagonist activity, passed the six states through
the fixed engine, and applied the unchanged PD servo. The corrected
representative 8-second episode produced forward distance `0.2022 m`, path
`0.5420 m`, mean ride height `.1391 m`, minimum height `.0930 m`, no fall, and
realized tripod scalar `−.3058` (`P3-ABSENT`). The synchronized four-view
video shows irregular asymmetric stepping/lurching, not a coordinated
alternating tripod: `results/path2/bodyfix_stepwalk_multiview.mp4`.

### P4 timing-source controls

The corrected paired study used 20 shared static-prey starts, 8 s per episode,
and the same engine, decode rule, steering, body, PD constants, and seeds. The
random condition used independent alternating-renewal swing/stance durations
whose per-leg mean frequency and swing duty were fixed from the P1 real decode;
only the timing source changed. Means have 95% percentile-bootstrap CIs
(100,000 replicates, seed `20260829`). Falls use the unchanged height threshold
`.02 m`.

The predeclared four-seed evaluation set is `20260829`–`20260832`, present as
episodes 0–3 in every condition. P4 CIs use all 20 paired episodes; videos are
fixed-seed exemplars, not post-hoc best selections.

| timing source | forward m [95% CI] | path m [95% CI] | falls | upright | realized scalar [95% CI] | phase verdicts |
|---|---:|---:|---:|---:|---:|---|
| real VNC | .1014 [.0659, .1391] | .3784 [.2834, .4726] | 0/20 | 20/20 | −.1317 [−.2070, −.0515] | 20/20 P3-ABSENT |
| shuffled VNC | −.0007 [−.0037, .0024] | .0470 [.0268, .0686] | 0/20 | 20/20 | ~0 [0, 0] | 20/20 P3-ABSENT |
| random matched-rate | .1329 [.0811, .1893] | .4750 [.3564, .5915] | 0/20 | 20/20 | −.0293 [−.1542, .1094] | 20/20 P3-ABSENT |

For the predeclared primary path endpoint, real minus shuffled was `+0.3314
m` [+.2529, +.4070], exact paired sign-flip `p=1.53×10⁻⁵`, Holm
`p=3.05×10⁻⁵`; real minus random was `−.0966 m` [−.1395, −.0562], exact
`p=4.58×10⁻⁵`, Holm `p=4.58×10⁻⁵`. Forward-distance differences were
`+.1021 m` [+.0676, +.1378] and `−.0315 m` [−.0634, −.0024], respectively;
the real-minus-random forward test is exact `p=.0630`. Mean physical foot
contact was `.716 [.667,.774]`, `.982 [.973,.990]`, and `.668 [.612,.735]`
for real, shuffled, and random; commanded stance-leg means were `2.119
[1.456,2.893]`, `5.121 [4.727,5.513]`, and `1.885 [1.806,1.964]`.
Real timing beats shuffled on movement, but not random timing, and no
condition produces the declared coordinated walk. This is evidence assembly
only; final headline wording remains withheld pending user approval.

### Path 3 layer localization

Layer i remains the primary failure: the real VNC produces dynamic decoded
states, but only `2/6` within-tripod and `4/9` cross-tripod state pairs pass in
P1; mean decoded swing fraction remains `.686`, strongly imbalanced across
legs. Layer ii is not silent: the corrected real episode has motor temporal SD
`.129` and target temporal SD `.086` rad, so the fixed decode/interface carries
variation but cannot repair its phase. Layer iii passes the corrected clean-
clock gate (`1.660 m` forward, scalar `2.000`) and remains upright in P3; when
driven by decoded timing it produces no tripod. This localizes the remaining
transfer failure to layer-i timing organization, with layers ii/iii transmitting
or expressing the incoherence rather than causing total signal loss.

The corrected control manifest, raw episode table, per-condition metrics, and
figure are `results/path2/bodyfix_stepwalk_controls.json`,
`results/path2/bodyfix_stepwalk_controls.csv`,
`results/path2/bodyfix_stepwalk_metrics_{real,shuffled,random}.json`, and
`results/path2/bodyfix_stepwalk_controls.png`. Corrected condition videos are
`results/path2/bodyfix_stepwalk_multiview{,_shuffled,_random}.mp4`. The old
`stepwalk_*` body-level outputs remain as superseded provenance.

## Integrity and zero-learning checks

The current artifact set includes:

- rebuilt CSR/IO artifacts for all-leg, T1, and canonical VNC paths;
- published-candidate index/body-ID table and coupling tables;
- T1 sweep, real/shuffled rhythm traces, all-leg coordination figure, and
  transfer-layer figure;
- synchronized four-view brain-coupled video;
- 20×5 control CSV/manifest plus CI/significance outputs.
- E0–E4 phase matrices, intersegmental inventories, and sensory-feedback
  replay artifacts.
- C0/C1/C2 frozen-gain search logs, three-way curves, per-seed videos, and
  bootstrap CI summary.
- Outcome D target, D1/D2 checkpoints, three-way metrics, CI summary, and
  matched-condition multiview episodes.
- Path 3 method-source record, antagonist pool decode, fixed-engine ceiling,
  paired timing controls, CIs/sign-flip tests, and condition videos.
- Body-fix Gate A, corrected E0/E3/E4/layer audits, corrected five-condition
  control statistics, and corrected Path 3 real/shuffled/random videos under
  the `bodyfix_*` prefix.

The corrected E3/E4/layer artifacts carry `zero_learning=true`; the E0 and
Path 3 probes are fixed replay/measurement code with no optimizer, backprop,
reward, target fit, or performance-based choice. All 58 JSON and 41 CSV files
parse; all 212 published candidates are selected with 0 omitted; the frozen
all-leg VNC file hash is
`018a4890e01f3dd608693e8a040103829e8523d134290ae918e6e5dd4d815506`, and the
frozen FlyWire brain file hash is
`f84f8dcd8397a43b8825ba45573cf14105f23d1846c82b58e4f07422f7586a88`.
Python compilation, `git diff --check`, shared-condition checks, home-keyframe
provenance, and readable eight-second H.264 video checks all pass. Outcome C/D
remain separately labeled learning rungs and were not rerun.

## Limits and claim boundary

This is a fixed-rate-model and fixed-body transfer experiment, not a claim
about all MANC models, all hexapod morphologies, or biological behavior in
general. The 3 s controls use static prey and measure short-horizon motion;
zero catches are descriptive, not evidence of impossible pursuit. The
published-candidate completeness result rules out omission of the 212
current-v1.0 T1/T2/T3 candidate cells as the explanation for the missing
tripod, while other non-published boundary cells remain outside the declared
subnetwork. Outcome D relaxes weight values and learns a decoder; it is not a
zero-learning connectome claim, and its matched controls are part of the
pre-fix provenance only. The corrected body evidence is assembled, but the
exact final claim wording remains intentionally withheld until user approval.
