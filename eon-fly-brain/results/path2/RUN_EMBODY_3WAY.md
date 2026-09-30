# EMBODY-3WAY status report

**Status:** Stage 1 complete. Stage 2 not run. No decision label assigned.

## Bottom line

The decisive comparison remains open: frozen measured ventral nerve cord (VNC)
versus amplitude-matched random and degree-preserving rewired VNC under identical
proximal policy optimization (PPO) training. Stage 1 tested interface fit only.
Its losses do not measure locomotion.

## Locked protocol

- Interface: 49-dimensional body observation, linear encoder, frozen 11,000-node
  VNC rate core, linear 18-joint decoder.
- Interface parameters: 67,204 in every arm. Core parameters: 0.
- Teacher: Option B generic periodic controller, selected by constant-speed body
  path error and fall penalty. Speed `0.10 m/s`, frequency `1.20 Hz`,
  cross-entropy method (CEM)
  population `24`, elites `6`, rounds `3`, seed `20260830`. Best path MSE:
  `0.0117189069 m^2`; displacement: `0.2415102869 m`; fall: `false`.
- Stage 1: Adam, learning rate `0.001`, 20 epochs, 140 train steps and 60
  held-out steps, seed `20260829`.
- Stage 2 remains fixed at PPO, `5,000,000` steps per arm, one environment,
  12 evaluation seeds (`20260830` to `20260841`), last fixed checkpoint.

## Stage 1 results

Loss is action MSE in `rad^2`.

| arm | initial train | final train | initial held-out | final held-out |
|-|-:|-:|-:|-:|
| connectome | 1.472574 | 1.081586 | 1.497829 | 1.048560 |
| random | 41,599,852 | 10,954,545 | 2.764222 | 2.666219 |
| rewired | 1,223,452.875 | 557,322.750 | 1.829012 | 1.827670 |

The random and rewired train losses reflect their fixed amplitude-matching
affines. No constant or arm-specific setting changed after registration.

## Amplitude and integrity checks

- Common varying motor channels: `52/260`; declared excluded channels: `208`.
- Exact-zero temporal-SD channels: connectome `93`, random `100`, rewired `143`.
- Nonzero target channels unmatchable before the common mask: random `60`, rewired
  `86`. These channels were excluded explicitly.
- Random match error: max SD error `5.55e-17`; mean-preservation max error
  `2.15e-11`.
- Rewired match error: max SD error `1.11e-16`; mean-preservation max error
  `5.88e-12`.
- VNC file hashes matched pre/post. Core tensor hashes matched pre/post in all
  three arms. `vnc_weights_trained=false`; core gradients stayed absent.
- Input SHA-256: weights
  `018a4890e01f3dd608693e8a040103829e8523d134290ae918e6e5dd4d815506`;
  I/O index
  `3f65f5b3ebed0d63161dca416bffa1a55afc0be4906113103426b16492d09449`.
- Core tensor SHA-256: connectome
  `128b5ceddec4d8e14f3a04fbb66a08a2e83564c674624741eb03fb2bf01517fc`;
  random `0b8f5c35efc0025ab585ac1cf816787b76b3c3b559a2d6a137dd447b263a6037`;
  rewired `772e31e44fcaf0f1d055a904883e2fe1567f7eb3d2910a23e23d8711a15cc650`.

Self-checks passed:

```text
ENV PORT SMOKE: PASS
FROZEN-CORE POLICY: PASS
OVERFIT TEST: PASS, loss 1.76755 -> 6.42917e-05
PPO smoke: PASS, 8 steps in each arm, W.grad is None
```

## Why Stage 2 is absent

The preregistered owner gate remained before the PPO budget. Hardware checks also
found no CUDA device. MPS is available, but its sparse CSR constructor fails:

```text
NotImplementedError: Could not run 'new_compressed_tensor' from the 'mps:0' device.)
```

Therefore this report contains no PPO displacement values, confidence intervals,
paired sign-flip tests, Holm correction, `n/K` control counts, final checkpoint
videos, or decision label. `EMBODY-NO-WALK` cannot be assigned because no arm was
evaluated. The absence of Stage 2 is an incomplete run, not a null result.

## Artifacts

- `embody_3way_preregistered.json`
- `embody_3way_stage1.json`, `embody_3way_stage1.csv`
- `embody_3way_stage1_connectome.pt`, `embody_3way_stage1_random.pt`,
  `embody_3way_stage1_rewired.pt`
- `embody_3way_stage2_addendum.json`
- Code: `../../code/audit_embody_3way.py`, `../../code/run_embody_3way.py`

No final scientific claim is written here. A claim requires the registered Stage 2
comparison and owner approval.
