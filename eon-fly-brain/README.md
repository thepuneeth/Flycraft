# Flycraft experiments

## Path 2 current result

Whole-brain FlyWire → official Male CNS VNC → fixed motor map → MuJoCo
hexapod, with zero learning. All 212 current-v1.0 T1/T2/T3 cells matching
the published leg-coordination candidate rule are included, including the two
`IN12B013` T1 cells. E0–E4 are exhausted: the expanded graph, structured
descending drive, and fixed anatomical proprioceptive feedback still lack
coherent alternating tripod phase.

The real VNC is not equivalent to a shuffled VNC: in the paired 3 s study,
real open-loop path is `0.207 m [0.205, 0.209]` versus shuffled
`0.065 m [0.063, 0.066]`, exact paired sign-flip `p=1.91e-6`,
Holm-adjusted `p=5.72e-6`. Brain-coupled path is `0.185 m [0.140, 0.228]`
versus random descending `0.223 m [0.193, 0.256]`; adjusted `p=0.158`.
All 100 control episodes were catch-free and fall-free. E0 after E2 passes
3/6 within-tripod and 3/9 cross-tripod pairs; E3 and E4 remain absent at VNC
and joint-target layers. Final claim wording is pending review.

E2 inventory found 212/212 published candidates selected. E4 adds 622 traced
leg proprioceptors under a declared anatomical rule and yields 0.401 m path,
0.095 m forward displacement, no fall, but no tripod. See root-level
`PATH2_METHODS.md` for full matrices, CIs, controls, and Outcome B plan.

## Frozen commands

From repository root:

```bash
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/fetch_manc.py \
  --data eon-fly-brain/data --neuromeres T1,T2,T3 --premotor_scope all \
  --synapse_floor 5 \
  --out eon-fly-brain/results/path2/all_legs_io_index.npz \
  --weights_out eon-fly-brain/results/path2/all_legs_weight_csr.pkl

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/sweep_vnc.py \
  --data eon-fly-brain/results/path2 \
  --weights_name t1_paper_weight_csr.pkl --index_name t1_paper_io_index.npz \
  --out eon-fly-brain/results/path2/t1_paper_bifurcation_sweep \
  --model paper --steps 300 --substeps 4 --dt .005 --heterogeneity --seed 1

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
```

For one synchronized episode with four views:

```bash
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_vnc_chase.py \
  --data eon-fly-brain/data --vnc_data eon-fly-brain/results/path2 \
  --vnc_weights_name all_legs_weight_csr.pkl \
  --vnc_index_name all_legs_io_index.npz --secs 8 \
  --vnc_model paper --vnc_drive 100 --vnc_gW .03 --vnc_substeps 4 \
  --vnc_heterogeneity --vnc_seed 1 --seed 1 --multi_view \
  --video eon-fly-brain/results/path2/vnc_brain_closed_loop_8s_multiview.mp4
```

## Artifacts

- `code/fetch_manc.py` — official MANC selection and signed CSR builder.
- `code/vnc_stepper.py` — paper-rate VNC dynamics and shuffle control.
- `code/mn_to_joint.py` — fixed motor-pool to 18-joint map.
- `code/run_vnc_chase.py` — brain/VNC/body runner.
- `code/audit_vnc_legs.py` — candidate, coupling, and tripod audit.
- `code/audit_transfer_layers.py` — VNC → target → physics localization.
- `code/analyze_control_study.py` — CIs and paired significance tests.
- `code/audit_intersegmental.py` — E2 cross-neuromere inventory.
- `code/probe_structured_descending.py` — E3 structured brain-drive probe.
- `code/run_sensory_feedback.py` — E4 proprioceptive feedback replay.
- `results/path2/all_legs_coordination.json` — current graph/phase audit.
- `results/path2/transfer_layers.png` — mechanistic layer figure.
- `results/path2/vnc_control_study_3s_stats.json` — full control statistics.
- `results/path2/vnc_brain_closed_loop_8s_multiview.mp4` — four synchronized
  views of one brain-coupled episode.
- `results/path2/vnc_phase_probe_e0_after_e2.json` — post-completeness E0.
- `results/path2/e2_intersegmental.json` — E2 inventory and decision.
- `results/path2/e3_structured_descending.json` — E3 phase matrices.
- `results/path2/e4_sensory_feedback.json` — E4 phase and body metrics.

See repository-root `PATH2_METHODS.md` for constants, selection decisions,
results, controls, limitations, and the provisional claim boundary.
