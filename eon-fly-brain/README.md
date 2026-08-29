# Flycraft hexapod experiments

This directory contains the FlyWire connectome controller and MuJoCo body
experiments. The brain runs as a signed sparse rate model with no task
training.

## Experiments

- `code/run_hexapod_chase.py` — main experiment. Retina features enter the
  connectome; left/right descending activity becomes `{forward, turn}`; a
  tripod CPG and PD servo drive the hexapod.
- `code/run_hexapod.py` — null experiment. A fixed random readout maps the full
  descending population to 18 joints with no CPG.
- `code/hexapod_gait.py` — standalone gait calibration and MuJoCo PD control.
- `code/flyserver.py` — line-delimited JSON reference stepper for the same
  connectome dynamics.
- `code/build_io_index.py` — regenerates `data/io_index.npz` from FlyWire
  annotations.

## Run

From project root:

```bash
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_hexapod_chase.py \
  --secs 20 --prey 1.2 0.8 --fwd_gain 12 --turn_gain 6

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/hexapod_gait.py \
  --mode walk --drive 0.8 --turn 0 --secs 5

./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_hexapod.py \
  --secs 8 --scale 10 --seed 0
```

Add `--video output.mp4` to either MuJoCo experiment to render video.

## Data

- `data/weight_csr.pkl` — signed 138,639 × 138,639 sparse connectome.
- `data/io_index.npz` — visual input and descending output indices.
- `data/2025_Completeness_783.csv` — FlyWire neuron ordering and metadata.
- `data/2025_Connectivity_783.parquet` — source connectivity table.
- `data/flywire_annotations.tsv` — annotations used to build IO indices.

The local `.venv` contains the working NumPy, PyTorch, MuJoCo, and imageio
runtime. The cached CSR matrix is loaded directly by the experiments.
