# Flycraft

Hexapod research: test whether an untrained FlyWire brain connectome can steer
a body it never evolved for.

```text
prey position
  -> visual encoding
  -> 138,639-neuron FlyWire rate model
  -> descending left/right readout: {forward, turn}
  -> tripod CPG (VNC stand-in)
  -> PD torque control
  -> MuJoCo PhantomX hexapod
```

## Run

Use the existing environment:

```bash
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_hexapod_chase.py \
  --secs 20 --prey 1.2 0.8 --fwd_gain 12 --turn_gain 6
```

Other checks:

```bash
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/hexapod_gait.py \
  --mode walk --drive 0.8 --turn 0 --secs 5
./eon-fly-brain/.venv/bin/python eon-fly-brain/code/run_hexapod.py \
  --secs 8 --scale 10 --seed 0
```

## Layout

- [HANDOFF.md](HANDOFF.md): research question, conventions, results, and next work.
- `assets/hexapod/`: PhantomX MJCF and meshes.
- `eon-fly-brain/code/run_hexapod_chase.py`: main connectome-steering experiment.
- `eon-fly-brain/code/run_hexapod.py`: rich-readout null; no CPG.
- `eon-fly-brain/code/hexapod_gait.py`: calibrated tripod gait and PD servo.
- `eon-fly-brain/code/flyserver.py`: line-delimited JSON connectome stepper.
- `eon-fly-brain/code/build_io_index.py`: rebuilds sensory/output neuron indices.
- `eon-fly-brain/data/`: FlyWire v783 source data, sparse weights, and IO indices.
- `episode_hexapod_chase.mp4`: rendered preview.

The connectome contributes only `{forward, turn}`. The CPG supplies walking
rhythm because the FlyWire brain dataset does not include the fly VNC.
