# EMBODY-3WAY v2 — revised spec (supersedes v1)

**Status:** v1 Stage 1 is complete and stays on record. v1's Stage 2 is
**withdrawn unrun**. This document is a new pre-registration, not an edit —
v1 registered a wide encoder and that constant cannot be changed inside a
live registration.

**What changed and why:** the research scout confirmed v1's encoder learned
**1,325 independent descending drives** (verified in code at
`embody_policy.py`, registered as `"encoder_output": "all descending input
channels"`). That is enough freedom to program the core directly, so a v1
Stage-2 result — win or null — would have been an interface result, not a
connectivity result. See `RESEARCH_SCOUT_FINDINGS.md` Q3 and its ranked
change 1.

---

## 1. The change: a fixed 4-command bottleneck

The encoder now emits **4 latent commands**, expanded into the 1,325
descending inputs through a **fixed, anatomy-derived, non-trainable** matrix.

Built by `eon-fly-brain/code/build_dn_command_map.py` into
`eon-fly-brain/results/path2/dn_command_map.npz`:

| channel | definition | support |
|---|---|---|
| `go` | uniform drive to every descending input | 1325 |
| `steer` | +1 left soma, −1 right soma, 0 midline/unknown | 1303 |
| `family` | +1 DNg/DNge, −1 DNp/DNpe, 0 otherwise | 1175 |
| `walkdn` | +1 on DNg100 and DNb08, the classes already declared in the io index | 6 |

Side and type labels come from
`data/manc/body-annotations-male-cns-v1.0-minconf-0.5.feather`, matched
**1325/1325** (655 L, 648 R, 10 M, 12 unknown). Matrix SHA-256:
`97e6d46d0605204716accc8fdc6a945b2e16d367d68feaf408f0d0c56fd11cb3`.
Rank is asserted equal to 4 at build and at load.

Forward path: `latent = encoder(obs)` → `command = latent @ command_map` →
`drive = drive_scale · sigmoid(command)`. The control manifold is
4-dimensional; no per-DN gain or bias is learned.

**Honest framing, required in every write-up:** 4 is an engineering
bottleneck chosen for distinct go / steer / family / declared-walking-class
axes. It is **not** a measured biological control rank. The descending-neuron
literature motivates *a* bottleneck and argues against both a single scalar
and 1,325 free drives, but does not establish 4 as a biological fact
(`RESEARCH_SCOUT_FINDINGS.md` Q3).

### Measured effect of the change
- Interface parameters: **70,948 → 4,898** (14.5× fewer).
- Overfit capacity retained: loss `1.7688 → 0.000295` with the bottleneck,
  versus `1.7676 → 6.43e-05` wide. The interface can still express the
  target; it simply can no longer address neurons individually.

---

## 2. What is unchanged from v1

Every constraint in `docs/archive/EMBODY_3WAY_SPEC.md` §1, §3, §5 still binds:
frozen core, hash-equal pre/post, three arms identical except the core,
amplitude-matched random plus degree-preserving rewired, paired sign-flip
with Holm across the two contrasts, bootstrap CIs, `n/K`, selection never by
evaluation displacement, no final claim without owner approval.

Verified after the change:
- `policy.core.W.grad is None`; `command_map.requires_grad is False`;
  neither is an `nn.Parameter`.
- Core SHA-256 identical before and after a backward pass.
- Self-checks pass: `ENV PORT SMOKE`, `FROZEN-CORE POLICY`
  (`device=cpu latent_dim=4 interface_params=4898`), `OVERFIT TEST`.

---

## 3. New requirements in v2

1. **All 260 motor channels reported.** v1 excluded `208/260` under amplitude
   matching. No field convention supports an exclusion at that scale
   (`RESEARCH_SCOUT_FINDINGS.md` Q4). Report the common-mask result **and** an
   all-channel analysis as a declared sensitivity pair. Do not retune anything
   to make them agree; if they disagree, that disagreement is the finding.

2. **Compute honesty gate.** Field-standard hexapod PPO uses thousands of
   parallel environments. One environment at 5M steps per arm is a **pilot**.
   Register it as such: a null under this budget may reflect throughput, not
   biology, and must be reported as `EMBODY-PILOT-INCONCLUSIVE` rather than a
   moat null unless at least one arm reaches stable forward locomotion. Only a
   run where some arm walks can license `EMBODY-MOAT-NULL`.

3. **Device policy.** CPU or CUDA only. MPS is refused at construction —
   PyTorch has no sparse CSR on MPS (pytorch/pytorch#129842). All three arms
   must run on the **same** device and the same budget; record the device in
   the results JSON. `--device auto` selects CUDA when present.

4. **Teacher naming.** Option B is a **synthetic body-path teacher**, never
   "imitation" or "biological gait." Before any gait language, report contact
   sequence, limb phase, joint-range, and sliding/drag diagnostics
   (`RESEARCH_SCOUT_FINDINGS.md` Q5).

5. **Claim ceiling.** One fitted body is embodiment, not morphology
   generalization. Cross-body claims require a held-out body evaluated with
   the encoder/decoder **unrefit** (`RESEARCH_SCOUT_FINDINGS.md` Q7).

6. **Novelty wording.** No "first connectome robot" — that is occupied by the
   *C. elegans* robot work, and connectome-reservoir-plus-linear-readout is
   also established. The defensible sentence is the narrow intersection:
   frozen measured MANC/VNC weights as the computer, a non-fly 18-DoF hexapod,
   a 4-command interface, and predeclared weight/degree nulls.

---

## 4. Decision labels (v2)

- `EMBODY-MOAT-EARNED` — connectome beats **both** controls, Holm-corrected,
  CI excludes zero, and at least one arm walks.
- `EMBODY-MOAT-NULL` — at least one arm walks, and connectome ties or loses.
- `EMBODY-PILOT-INCONCLUSIVE` — no arm reaches stable forward locomotion.
  Nothing may be concluded about connectivity; report as a compute-limited
  pilot.

---

## 5. Order of operations

1. Rebuild the command map; confirm the SHA-256 above. **Stop if it differs.**
2. Re-run the three self-checks on the target device.
3. Write `embody_3way_v2_preregistered.json` — including `latent_dim: 4`, the
   map hash, the device, the pilot clause, and the labels above.
4. Stage 1 imitation, three arms, identical config. Report per-arm loss.
5. **Owner checkpoint before Stage-2 spend.**
6. Stage 2 PPO, same device and budget for all arms, all seeds.
7. Evaluate, test, label, write artifacts, watch videos, write the brief —
   including the all-channel sensitivity analysis.

---

## 6. Files changed

- `eon-fly-brain/code/build_dn_command_map.py` — **new**, builds and hashes the map.
- `eon-fly-brain/results/path2/dn_command_map.npz` — **new** artifact.
- `eon-fly-brain/code/embody_policy.py` — `command_map` bottleneck, CUDA
  support, MPS refusal, `resolve_device`, extended frozen asserts.
- `eon-fly-brain/code/embody_overfit.py` — accepts a command map.

`audit_embody_3way.py` and `run_embody_3way.py` still construct the wide
encoder; they must pass `command_map=load_command_map()` before a v2 run.
That wiring is the remaining code task.
