# Flycraft

## Path 2 result — working headline

All 212 current Male CNS MANC T1/T2/T3 cells matching the published
leg-coordination candidate rule are included, including both `IN12B013` T1
cells. E0–E4 are exhausted: the expanded real VNC, structured brain drive,
and fixed anatomical proprioceptive feedback still do not form a coherent
alternating tripod. Real VNC wiring beats its shuffled-topology control, but
the brain-coupled controller does not beat matched random descending drive in
the 3 s hexapod study. Final claim wording is pending review.

```text
prey -> FlyWire brain -> DNg100/DNb08 -> real MANC VNC
     -> fixed motor map -> PD torque -> MuJoCo PhantomX hexapod
```

No learning, reward, fitting, target trajectory, or performance-based tuning.

## Current numbers

- All-leg graph: 11,000 nodes, 492,722 edges at absolute weight floor 5,
  485,074 signed nonzero edges, 260 motor outputs.
- T1 graph: 4,418 nodes, 144,214 signed nonzero edges, 106 motor outputs.
- Added candidates: body `802193` (`IN12B013`, T1R) and body `802470`
  (`IN12B013`, T1L); candidate-to-candidate cross-neuromere edges have 6 and
  5 synapses.
- E0 after E2: phase remains absent; 3/6 within-tripod pairs and 3/9
  cross-tripod pairs pass the declared tolerance.
- E2 inventory: 212/212 published coordination candidates have
  cross-neuromere edges and are selected; no anatomical widening was
  justified. Raw inventory: 12,745 endpoint cells and 166,088 floor-qualified
  cross-edge rows.
- E3 structured DNg100/DNb08 drive: VNC and joint-target phase remain absent
  (within cosine −0.181, cross cosine −0.284, scalar +0.103; 2/6 and 3/9
  pairs pass).
- E4 feedback branch: 622 traced leg proprioceptors added under a fixed
  anatomical rule; phase remains absent (within cosine −0.071, cross cosine
  +0.102, scalar −0.173; 2/6 and 3/9 pairs pass), with 0.401 m path, 0.095 m
  forward displacement, and no fall.
- T1 frozen rhythm: dominant peak 2.467 Hz, autocorrelation frequency 2.500
  Hz, autocorrelation peak 0.342, opposing-pool correlation −0.321, left/right
  correlation +0.031.
- All-leg coxa phase: tripod A within −0.057, tripod B within +0.307,
  cross-tripod −0.185. No coherent tripod.
- 20 paired episodes per condition, 3 s: zero catches and zero falls in all
  100 episodes.

| condition | mean path (m), 95% bootstrap CI |
|---|---:|
| real VNC open-loop | 0.207 [0.205, 0.209] |
| brain-coupled VNC | 0.185 [0.140, 0.228] |
| shuffled VNC | 0.065 [0.063, 0.066] |
| random descending | 0.223 [0.193, 0.256] |
| Path 1 hand CPG | 0.246 [0.189, 0.300] |

Real VNC − shuffled path difference is +0.1425 m [0.1397, 0.1449], exact
paired sign-flip `p=1.91e-6`, Holm-adjusted `p=5.72e-6`. Brain-coupled − random
is −0.0383 m [−0.0775, +0.0012], adjusted `p=0.158`. Full metric CIs and
tests: `eon-fly-brain/results/path2/vnc_control_study_3s_stats.json`.

## Outcome B result

The reservoir pivot is complete. Only closed-form ridge `W_out` was fit to the
fixed Path-1 tripod template; MANC weights stayed frozen. Linear and one
permitted 64-unit fixed-random-feature escalation were compared across real,
column-permuted random, and degree-preserving rewired reservoirs. No condition
reached dynamic held-out R² `0.8` or produced a stable tripod walk. Gate verdict:
`READOUT TOO WEAK`; final claim wording remains pending review. See
`PATH2_METHODS.md` and `eon-fly-brain/results/path2/reservoir_3way.json`.

## Reproduce

Run from repository root:

```bash
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/fetch_manc.py \
  --data eon-fly-brain/data --neuromeres T1,T2,T3 --premotor_scope all \
  --synapse_floor 5 \
  --out eon-fly-brain/results/path2/all_legs_io_index.npz \
  --weights_out eon-fly-brain/results/path2/all_legs_weight_csr.pkl

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/sweep_vnc.py \
  --data eon-fly-brain/results/path2 \
  --weights_name t1_paper_weight_csr.pkl \
  --index_name t1_paper_io_index.npz \
  --out eon-fly-brain/results/path2/t1_paper_bifurcation_sweep \
  --model paper --steps 300 --substeps 4 --dt .005 \
  --heterogeneity --seed 1

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/analyze_vnc_rhythm.py \
  --data eon-fly-brain/results/path2 \
  --weights_name t1_paper_weight_csr.pkl \
  --index_name t1_paper_io_index.npz \
  --out eon-fly-brain/results/path2/t1_paper_rhythm_validation_heterogeneous \
  --model paper --drive 100 --gW .03 --substeps 4 --steps 1500 \
  --heterogeneity --seed 1

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/audit_vnc_legs.py \
  --data eon-fly-brain/results/path2 \
  --weights_name all_legs_weight_csr.pkl \
  --index_name all_legs_io_index.npz \
  --out eon-fly-brain/results/path2/all_legs_coordination \
  --model paper --drive 100 --gW .03 --tau .02 --dt .005 --substeps 4 \
  --steps 1500 --heterogeneity --seed 1

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/audit_transfer_layers.py \
  --data eon-fly-brain/results/path2 \
  --out eon-fly-brain/results/path2/transfer_layers

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/study_vnc.py \
  --n 20 --secs 3 --seed 20260829 \
  --out eon-fly-brain/results/path2/vnc_control_study_3s

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/analyze_control_study.py \
  --input eon-fly-brain/results/path2/vnc_control_study_3s.csv \
  --out eon-fly-brain/results/path2/vnc_control_study_3s_stats

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/audit_intersegmental.py \
  --data eon-fly-brain/results/path2 \
  --index_name all_legs_io_index.npz \
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
```

The canonical runner reads `eon-fly-brain/data/vnc_io_index.npz` and
`eon-fly-brain/data/vnc_weight_csr.pkl`; regenerate those from the same fetch
command when needed. Official MANC source files belong in
`eon-fly-brain/data/manc/`.

## Video

One frozen brain-coupled episode, synchronized in four views (oblique,
overhead, side, front):

`eon-fly-brain/results/path2/vnc_brain_closed_loop_8s_multiview.mp4`

## Key artifacts

- `PATH2_METHODS.md` — methods-complete working report, constants, controls,
  statistics, limitations, and layer localization.
- `eon-fly-brain/results/path2/all_legs_coordination.json` — graph, candidate,
  coupling, and phase audit.
- `eon-fly-brain/results/path2/transfer_layers.png` — VNC → target → physics
  mechanistic figure.
- `eon-fly-brain/results/path2/vnc_control_study_3s_stats.json` — CIs and
  paired tests.
- `eon-fly-brain/results/path2/vnc_phase_probe_e0_after_e2.json` — Gate 2
  phase matrix and verdict.
- `eon-fly-brain/results/path2/e2_intersegmental.json` — full intersegmental
  inventory and selection decision.
- `eon-fly-brain/results/path2/e3_structured_descending.json` — structured
  brain-drive phase matrices.
- `eon-fly-brain/results/path2/e4_sensory_feedback.json` — fixed sensory-loop
  replay, phase matrices, and body metrics.

Path 1 remains only as the hand-written CPG baseline. It is not the rhythm
source in Path 2.
