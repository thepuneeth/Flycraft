# EMBODY-3WAY run

Stage 2 uses the preregistered B=128 vector environment per frozen core, with identical budget and final-checkpoint rule.

Decision label: `EMBODY-PILOT-INCONCLUSIVE`. Final claim remains owner-gated.

## Forward displacement

| arm | mean m | bootstrap 95% CI | falls | stable |
|---|---:|---:|---:|---|
| connectome | 0.307772 | [0.298630, 0.317056] | 12/12 | False |
| random | 0.149968 | [0.147965, 0.152262] | 0/12 | False |
| rewired | 0.170077 | [0.168360, 0.172123] | 0/12 | False |

## Paired contrasts

- `connectome_minus_random`: +0.157804 m, CI [+0.148704, +0.166960], one-sided exact p=0.000244141, Holm p=0.000488281, n/K=0/12
- `connectome_minus_rewired`: +0.137695 m, CI [+0.128657, +0.146904], one-sided exact p=0.000244141, Holm p=0.000488281, n/K=0/12

## Limitations

- Amplitude calibration retained 29/260 motor channels and explicitly excluded 231.
- This is forward locomotion under one body, interface, optimizer, and PPO budget; it is not a coordination proof.
- Final interpretation awaits owner approval.
