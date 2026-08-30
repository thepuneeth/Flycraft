# Flycraft — Path 2 handoff

Status: frozen rerun, E0–E4 localization, and reservoir R0–R3 complete;
documentation and audits updated. Final claim still requires user approval.

## Working headline

All 212 current Male CNS MANC T1/T2/T3 cells matching the published
leg-coordination candidate rule are included, including both `IN12B013` T1
cells. E0–E4 are exhausted: the expanded real VNC, structured brain drive,
and fixed anatomical proprioceptive feedback still do not form a coherent
alternating tripod. Real VNC wiring produces substantially more open-loop
movement than column-shuffled VNC, but brain coupling does not outperform
matched random descending drive in the short-horizon study.

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

Study design: five conditions, 20 paired episode starts per condition, 3 s
each, static prey, episode base seed `20260829`. Every condition shares each
episode's initial yaw and prey placement.

All 100 episodes returned normally: 0 catches and 0 falls in every condition;
Wilson 95% CI for each binary rate is `[0.000, 0.161]`.

| condition | path m [95% bootstrap CI] | forward m [95% bootstrap CI] | displacement m [95% bootstrap CI] | min distance m [95% bootstrap CI] |
|---|---:|---:|---:|---:|
| real VNC open-loop | .207 [.205, .209] | .019 [.017, .020] | .020 [.018, .021] | 1.390 [1.287, 1.491] |
| brain-coupled VNC | .185 [.140, .228] | .030 [.013, .049] | .040 [.025, .057] | 1.385 [1.278, 1.488] |
| shuffled VNC | .065 [.063, .066] | −.061 [−.063, −.059] | .061 [.059, .063] | 1.386 [1.286, 1.484] |
| random descending | .223 [.193, .256] | .020 [.003, .037] | .036 [.025, .049] | 1.383 [1.277, 1.487] |
| Path 1 hand CPG | .246 [.189, .300] | .094 [.039, .152] | .120 [.071, .171] | 1.322 [1.203, 1.436] |

Mean motor temporal SD is .140 [.140, .140] real VNC, .116 [.092, .135]
brain-coupled, .067 [.067, .067] shuffled, and .133 [.128, .138] random;
Path 1 has no VNC motor-output metric.

Primary exact two-sided paired sign-flip tests on path, with Holm adjustment
across three predeclared comparisons:

| comparison | difference m [95% CI] | exact p | Holm p |
|---|---:|---:|---:|
| real VNC − shuffled VNC | +.1425 [.1397, .1449] | 1.91e−6 | 5.72e−6 |
| brain-coupled − random descending | −.0383 [−.0775, +.0012] | .0789 | .1578 |
| open-loop VNC − Path 1 hand CPG | −.0387 [−.0924, +.0187] | .1984 | .1984 |

The real-versus-shuffled motor temporal-SD difference is +.0730 with exact
paired `p=1.91e−6`. This is the positive wiring result. It does not turn the
brain-coupled controller into a successful pursuit controller.

Full metrics and tests: `results/path2/vnc_control_study_3s_stats.json`.

## Failure localization

`results/path2/transfer_layers.png` is the mechanistic figure;
`results/path2/transfer_layers.json` contains raw measurements.

1. **VNC dynamics (layer i):** dynamic output exists — 260 motor channels,
   mean temporal SD .152 — and T1 oscillates. But all-leg inter-leg phase is
   inconsistent: A within −.057, B within +.307, cross −.185. Primary loss:
   oscillation without coherent tripod organization.
2. **Fixed motor map (layer ii):** all 18/18 target channels vary; mean target
   SD .133 rad and mean target range .588 rad. The map transfers activity but
   cannot repair phase.
3. **Hexapod physics (layer iii):** mean joint-angle SD .099 rad, target-to-
   joint correlation .424, target tracking MAE .112 rad. Body remains upright
   (minimum height .062 m, no fall) and travels .475 m path / .107 m forward
   in the 8 s open-loop replay. Physics attenuates and distorts the signal,
 but does not erase it. Layer i is the primary bottleneck; ii/iii add
 distortion rather than causing a total collapse.

## E0–E4 branch completion

E0 after the completeness rebuild is `E0-ABSENT`; its six-by-six matrix is in
`results/path2/vnc_phase_probe_e0_after_e2.json` and is byte-identical to the
pre-E2 probe. E2 found 212/212 published coordination candidates selected,
with no anatomically justified widening. The raw inventory contains 12,745
cross-neuromere endpoint cells and 166,088 floor-qualified edge rows.

E3 drove the VNC with the full FlyWire forward pass at the exact DNg100/DNb08
type-and-side outputs. VNC and joint-target phase both remained absent:
within cosine −.181, cross cosine −.284, scalar +.103, with 2/6 within and
3/9 cross pairs passing tolerance.

E4 added all 622 traced leg proprioceptors defined by the fixed
mechanosensory anatomy rule (campaniform, hair plate, chordotonal, or leg;
ProLN/MesoLN/MetaLN; root side L/R). The 8 s closed-loop replay remained
phase-absent: within cosine −.071, cross cosine +.102, scalar −.173, with 2/6
within and 3/9 cross pairs passing. The body traveled .401 m path and .095 m
forward, remained upright, and did not produce a tripod. This is one declared
fixed contact/angle encoding, not an exhaustive sensory-model search.

E4 forced the pivot: after E2 established published-set completeness and E3
failed with computed descending drive, the anatomically mapped sensory path
still produced no declared tripod phase. Anatomy did not close transfer within
this fixed model/body resolution.

Outcome B reservoir pivot is complete. R0 collected 3,000 post-burn full-VNC
states; R1 fit only closed-form ridge `W_out` to the fixed Path-1 template;
R2 deployed linear and one permitted 64-unit fixed-random-feature escalation;
R3 compared connectome, seeded column-permuted random, and exact
degree-preserving rewired reservoirs. No readout reached dynamic held-out
R² 0.8 or produced a stable tripod walk. The gate verdict is `READOUT TOO
WEAK`; final claim remains withheld pending approval.

See `PATH2_METHODS.md` and `results/path2/reservoir_3way.json` for the full
readout metrics and integrity record.

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
```

The synchronized four-camera episode is
`results/path2/vnc_brain_closed_loop_8s_multiview.mp4`.

## Integrity checklist

Before final claim approval, rerun and record:

- Python compile for every `code/*.py` file;
- JSON/CSV parse and expected-row checks;
- IO index/body-ID equality, including 212 published candidates;
- equality of canonical and results graph inputs where intended;
- video duration and readable MP4 stream;
- zero catches/falls in control outputs;
- zero-learning scan for optimizer/backpropagation/reward/loss/target fitting;
- `git diff --check`.

Do not add constants, retune gains, delete superseded artifacts, or broaden
the claim before user approval.

## File map

| file | role |
|---|---|
| `eon-fly-brain/code/fetch_manc.py` | fixed official-MANC selection and signed CSR builder |
| `eon-fly-brain/code/vnc_stepper.py` | paper-rate VNC dynamics and shuffle control |
| `eon-fly-brain/code/mn_to_joint.py` | fixed motor-pool to 18-joint map |
| `eon-fly-brain/code/run_vnc_chase.py` | brain/VNC/body episode runner and video renderer |
| `eon-fly-brain/code/audit_vnc_legs.py` | coupling, candidate, and phase audit |
| `eon-fly-brain/code/audit_transfer_layers.py` | VNC → target → physics audit and figure |
| `eon-fly-brain/code/analyze_control_study.py` | bootstrap CIs and paired tests |
| `eon-fly-brain/code/audit_intersegmental.py` | E2 intersegmental completeness inventory |
| `eon-fly-brain/code/probe_structured_descending.py` | E3 structured brain-drive phase probe |
| `eon-fly-brain/code/run_sensory_feedback.py` | E4 anatomical proprioceptive feedback replay |
| `PATH2_METHODS.md` | methods-complete working report |
