# Path 2 — methods and current evidence

Status: methods-complete working report; E0–E4 localization complete. Final
claim remains pending approval.

## Working headline

All 212 current MANC T1/T2/T3 cells matching the published leg-coordination
candidate rule are included. E0–E4 are exhausted: the expanded graph,
structured descending drive, and fixed anatomical proprioceptive feedback do
not produce a coherent alternating tripod. Real VNC wiring produces more
movement than its column-shuffled control, but the brain-coupled controller
does not outperform matched random descending drive in the short-horizon
hexapod study.

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
| motor map | stance `(0.0, 0.6, -1.2)`; gains `(-0.55, 0.65, 0.65)` for coxa/femur/tibia |
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

The mechanistic figure is
[`results/path2/transfer_layers.png`](eon-fly-brain/results/path2/transfer_layers.png);
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
tripod scalar `+0.103`, with 2/6 and 3/9 pairs passing.

E4 adds the 622-cell anatomical proprioceptor set and closes the loop through
the MANC graph. Both phase layers remain absent: within cosine `−0.071`, cross
cosine `+0.102`, tripod scalar `−0.173`, with 2/6 and 3/9 pairs passing. The
8 s replay travels `0.401 m` path and `0.095 m` forward, remains upright, and
has mean motor, target, and physical joint temporal SDs of `0.136`, `0.111`
rad, and `0.081` rad respectively. Artifacts:
`eon-fly-brain/results/path2/vnc_phase_probe_e0_after_e2.json`,
`eon-fly-brain/results/path2/e2_intersegmental.json`,
`eon-fly-brain/results/path2/e3_structured_descending.json`, and
`eon-fly-brain/results/path2/e4_sensory_feedback.json`.
E4 is a declared fixed contact/angle encoding of the available proprioceptor
classes, not an exhaustive search over sensory models.

E4 forced the pivot: after E2 established published-set completeness and E3
failed with computed descending drive, adding the anatomically mapped sensory
path still produced no declared tripod phase. Anatomy therefore did not close
the transfer within this fixed model/body resolution.

## Paired control study

Each condition uses the same 20 episode starts. Means below have 95%
percentile-bootstrap CIs (100,000 replicates, seed `20260829`). Catch and
fall rates are zero in every condition; each has Wilson 95% CI `[0.000,
0.161]`.

| condition | path (m) | forward (m) | displacement (m) | min prey distance (m) | motor temporal SD |
|---|---:|---:|---:|---:|---:|
| real VNC open-loop | 0.207 [0.205, 0.209] | 0.019 [0.017, 0.020] | 0.020 [0.018, 0.021] | 1.390 [1.287, 1.491] | 0.140 [0.140, 0.140] |
| brain-coupled VNC | 0.185 [0.140, 0.228] | 0.030 [0.013, 0.049] | 0.040 [0.025, 0.057] | 1.385 [1.278, 1.488] | 0.116 [0.092, 0.135] |
| shuffled VNC | 0.065 [0.063, 0.066] | −0.061 [−0.063, −0.059] | 0.061 [0.059, 0.063] | 1.386 [1.286, 1.484] | 0.067 [0.067, 0.067] |
| random descending | 0.223 [0.193, 0.256] | 0.020 [0.003, 0.037] | 0.036 [0.025, 0.049] | 1.383 [1.277, 1.487] | 0.133 [0.128, 0.138] |
| Path 1 hand CPG | 0.246 [0.189, 0.300] | 0.094 [0.039, 0.152] | 0.120 [0.071, 0.171] | 1.322 [1.203, 1.436] | not applicable |

Primary paired tests are exact two-sided sign-flip tests on path, with Holm
adjustment across the three predeclared comparisons:

| comparison | mean difference (m) | 95% CI | exact p | Holm p |
|---|---:|---:|---:|---:|
| real VNC − shuffled VNC | +0.1425 | [+0.1397, +0.1449] | 1.91×10⁻⁶ | 5.72×10⁻⁶ |
| brain-coupled − random descending | −0.0383 | [−0.0775, +0.0012] | 0.0789 | 0.1578 |
| open-loop VNC − Path 1 hand CPG | −0.0387 | [−0.0924, +0.0187] | 0.1984 | 0.1984 |

The motor temporal-SD difference for real VNC versus shuffled VNC is
`+0.0730` with exact paired `p=1.91×10⁻⁶`; its path difference is the
load-bearing behavioral positive. The brain-coupled path is lower than the
random-input mean, but its predeclared path CI crosses zero after adjustment;
the data do not support a claim that brain coupling improves pursuit.

Full metric CIs and all predeclared pairwise tests are in
`eon-fly-brain/results/path2/vnc_control_study_3s_stats.json`.

## Transfer-layer localization

The fixed open-loop VNC run was replayed for 8 s with the same frozen VNC,
map, and body constants:

1. **Layer i — VNC dynamics:** motor output is dynamic (`260` outputs; mean
   temporal SD `0.152`), and T1 has a reproducible oscillatory peak. The
   all-leg phase audit nevertheless gives opposite signs for the two
   within-tripod groups (`−0.057` versus `+0.307`) and only moderate negative
   cross-tripod correlation (`−0.185`). This is oscillation without coherent
   alternating tripod organization. It is the primary failure layer.
2. **Layer ii — fixed motor map:** all `18/18` target channels vary; mean
   target temporal SD is `0.133` rad and mean target range is `0.588` rad.
   The map transmits VNC variation but cannot repair its phase structure.
3. **Layer iii — body physics:** mean joint-angle SD remains `0.099` rad,
   target-to-joint correlation is `0.424`, and the body remains upright
   (`min height 0.062 m`, no fall). It travels `0.475 m` of path and `0.107 m`
   forward in this 8 s open-loop run. Physics attenuates/distorts the target
   (`0.112` rad mean target-tracking MAE), but does not erase the signal or
   produce the primary failure.

The current mechanistic reading is therefore: the real VNC carries dynamic
activity and the fixed map/body transmit it, but the expanded VNC does not
organize that activity into the phase pattern required by this hexapod. The
negative is narrower than “the fly VNC cannot produce rhythm” and narrower
than “all foreign bodies fail.”

## Outcome B: connectome-as-prior learning pivot

E0–E4 exhaust the no-learning chain at this model and body resolution. The
next experiment should keep the real MANC graph fixed and learn only a minimal
motor readout and sensory-feedback-gain layer. Use matched connectome-
initialized, random-initialized, and degree-preserving rewired initializations
on the same hexapod; compare walking performance and sample efficiency. Keep
the present open-loop no-learning result as the baseline. Cache the frozen
connectome forward pass so one RTX 4090 trains only the small learned layer.

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

The final audit must continue to require: Python compilation, JSON/CSV
manifest parsing, IO-index/body-ID equality, current artifact hashes,
video duration, zero catches/falls, and a source scan showing no optimizer,
backpropagation, reward, loss, or target-fitting path. No learning code is
part of this pipeline.

## Limits and claim boundary

This is a fixed-rate-model and fixed-body transfer experiment, not a claim
about all MANC models, all hexapod morphologies, or biological behavior in
general. The 3 s controls use static prey and measure short-horizon motion;
zero catches are descriptive, not evidence of impossible pursuit. The
published-candidate completeness result rules out omission of the 212
current-v1.0 T1/T2/T3 candidate cells as the explanation for the missing
tripod, while other non-published boundary cells remain outside the declared
subnetwork.

The exact wording of the final claim is intentionally withheld until user
approval.
