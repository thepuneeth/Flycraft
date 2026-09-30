# Codex loop — port the [SH] labeled-line method into the VNC hexapod

You are continuing Flycraft. The previous loop
(`CODEX_CPG_TRANSLATION_LOOP.md`) is **finished**, and its verdict was
`CPG-INTERFACE-SPECIFICITY-NULL`. Read that file's iteration log before you start:
`eon-fly-brain/results/path2/cpg_logs/ITERATION_LOG.md`.

This is a different experiment. Do not redo the previous one.

---

## 0. The one-paragraph version

The previous loop made the hexapod walk (`0.8386 m` forward, `2.4624×`
path/forward, no falls) and then failed its specificity test: a **random** graph
walked `0.8429 m` — slightly further than the real connectome. The reason is not
that the connectome carries no information. The reason is that **we read the
connectome out as a scalar.** Forward distance in metres is one number; any graph
with matched activity statistics produces one number just as well. Shiu et al.
never used a scalar. They used a **labeled line**: a *named* input population
(sugar-sensing GRNs) driving a *named* output neuron (MN9) whose behavioural role
is known in advance, with other named input populations as the contrast and
single-neuron silencing as the causal ladder. Your job is to port that design.
Replace "how far did it walk" with "which named motor neurons fired, in what
phase relationship, under which named descending drive" — a structured readout a
rewired graph should not be able to fake.

---

## 0b. RUN 1 FAILED FOR AN INSTRUCTION BUG — read this before anything else

A first pass of this loop completed and returned
`LABELED-LINE-PHASE-NOT-SUPPORTED-BAND-POWER` with `4.33 ± 1.37` of 20
antagonist cells passing. **That null is void.** It was caused by a defect in an
earlier version of this file, not by the connectome. Four things went wrong.
Every one of them is now a hard rule below.

### Bug 1 — the phase window was shorter than one cycle (fatal)

The previous version of §2.1 described [SH]'s protocol as "30 trials, 1 s each",
and the agent reasonably concluded that 1 s was the reference trial length and
cut the sweep from 8 s to 1 s. Its own log records the reasoning:

> *"Shiu's 1 s labeled-line trial is sufficient for the named rate and
> selectivity readouts … phase may be low-resolution and must be marked
> unsupported if band power is weak."*

That is correct for [SH] and catastrophic for us, **because [SH] measure rates
and we measure phase.** A rate converges in 1 s. A phase in a 1.5–4.0 Hz band
does not. Do the arithmetic the previous run did not:

```
signal_secs      = 1.0
sample_dt        = 0.02 s          ->  50 samples
burn_fraction    = 0.5             ->  25 samples analysed  =  0.5 s
FFT resolution   = 1 / 0.5 s       =  2.0 Hz
BANDPASS_HZ      = (1.5, 4.0)      =  2.5 Hz wide  ->  ~1.25 usable bins
period at 1.5 Hz = 0.667 s         >  the entire 0.5 s window
```

**The analysis window did not contain a single cycle of the lowest frequency in
the band.** `band_power_share` measured on 25 samples is equally meaningless. The
band-power stop rule then fired and the agent correctly refused to claim — but it
fired because of the window, not because of the graph. The phase arm never ran.

For reference, the project's own standard is `SIGNAL_STEPS = 1500` in both
`e0_lif_controls.py` and `specificity_campaign.py` — **30 s**, sixty times what
run 1 used.

### Bug 2 — the controls were not activity-matched

Reported `signal` from run 1:

| arm | `frac_active` | `motor_hz` |
|---|---|---|
| real | 0.2404 | **6.29** |
| rewired | 0.3556 | **16.69** |
| random | 0.4105 | **16.45** |

The controls fire **2.6× harder** than the real graph. A control that active is
not a control, and every S2/S3/S4 comparison in run 1 is confounded by it. The
old guard checked `frac_active` at a 2× tolerance (0.41/0.24 = 1.71×, so it
passed) and **never checked `motor_hz` at all.** `amplitude_match` was named in
the instructions but evidently never constrained the readout the metrics were
computed from.

### Bug 3 — S4 compared real against itself

The old S4 required real's selectivity distance to exceed *real's own SD*. It
reported `S4: true` on real `0.09697 ± 0.01886` — while the **random** control
scored `0.20705`, more than double real. A criterion that a control can beat 2:1
while still reporting `true` is not a criterion.

### Bug 4 — the three artifacts were the same file

`labeled_line_selectivity_lif.json` and `labeled_line_phase_lif.json` are
**byte-identical** (`md5 213a88b561f719768c4f69ad3d6a1f9c`, both 15,741,101
bytes). The old §12 asked for three files without saying they must hold
different content, so one blob was written three times.

### What run 1 did get right — keep all of it

Preflight reproduced 14 types and the 20 testable cells exactly. Freeze audit
clean: `file_hashes_equal`, `real_tensor_hash_equal`, `zero_learning: true`,
`parameter_updates: 0`. Episode spec hash shared across arms. It refused to
claim a result it could not support and returned `OWNER_APPROVAL_REQUIRED`. The
per-iteration log is exemplary and is what made this diagnosis possible in ten
minutes. Keep that discipline exactly.

**Do not treat run 1's numbers as a baseline.** They are superseded. The only
thing to carry forward is the method and the freeze audit.

---

## 1. Hard constraints — violating any of these voids the result

1. **VNC core weights stay FROZEN.** The measured connectome is never trained,
   fine-tuned, or fitted. `core_parameters: 0`, `parameter_updates: 0`,
   `zero_learning: true`. Hash the weight files and the real weight tensor before
   and after every run and report both.
2. **No training anywhere in this loop.** No gradient descent, no evolutionary
   search, no reward. If you find yourself writing an optimiser, you have left
   the task.
3. **No constant may be selected from an outcome you are also reporting as a
   result.** Declare the grid, run it once, report all of it. If you pick a
   winner, its neighbours go in the artifact too so a knife-edge is visible.
4. **An honest null is a deliverable.** The previous loop's null is a good
   result. Do not tune your way out of one.
5. **No pod above `$0.20/hr`.** Hard requirement. Always pass `--cost 0.20`.
6. **No final claim without owner approval.** You produce artifacts and a log.
   You do not declare the project's headline finding.
7. **Never run experiment code on the laptop.** Edit locally, run on the pod.

---

## 2. What [SH] actually did — read this carefully, it is the whole point

Vendored at `fly_field_refs/shiu_brain_model/`. Files that matter:
`model.py` (the simulator), `utils.py` (the analysis), `example.ipynb` (the
protocol as they teach it). Read all three before writing code.

### 2.1 Their pipeline, step by step

**Step 1 — pick the input by biological identity, not by search.**
`example.ipynb` opens with a hard-coded list of 21 flywire IDs:

```python
neu_sugar = [720575940624963786, 720575940630233916, ...]
```

These are sugar-sensing gustatory receptor neurons. They were not found by
sweeping candidates for the best response. They were looked up by cell type. The
input set is a *biological commitment made before the simulation runs.*

**Step 2 — drive them as optogenetics, not as data.**
`model.poi()` attaches a `PoissonInput` to each named neuron:

```python
p = PoissonInput(target=neu[i], target_var='v', N=1,
                 rate=params['r_poi'],                       # 150 Hz default
                 weight=params['w_syn'] * params['f_poi'])   # .275 mV * 250
neu[i].rfc = 0 * ms   # Poisson targets lose their refractory period
```

The drive is unstructured — Poisson noise at a fixed rate. **It carries no
timing, no pattern, no information.** Every temporal structure downstream is
produced by the graph. This is the single most important property of their
design and it is what makes their claim a connectome claim.

**Step 3 — run many trials, because the model is stochastic.**
`default_params`: `t_run = 1000 ms`, `n_run = 30`. Thirty independent Poisson
realizations of the *same* graph, run in parallel via joblib. Our own
`lif_realization_spread` measured why this matters: tripod coherence has sd
`0.3428` across 30 realizations of one frozen graph. One rollout is not a
measurement.

**Step 4 — reduce spikes to per-neuron rate, keeping the spread.**
`utils.get_rate()` returns two frames — mean rate per neuron and **standard
deviation across trials**. The SD is reported alongside every rate. Any effect
smaller than it is not an effect.

**Step 5 — read out at a NAMED neuron with a known behavioural role.**
This is the move we have never made:

```python
id_mn9 = 720575940660219265   # id for MN9
x = df_rate.loc[id_mn9, "sugarR_100Hz"]
```

MN9 is the motor neuron that drives proboscis extension. Sugar in → MN9 fires →
the fly would extend its proboscis. **The behavioural claim is a labeled line
from a named input to a named output.** There is no decoder, no learned readout,
no aggregate score. The specificity lives in the *identity* of the endpoints.

**Step 6 — establish causality by silencing, one neuron at a time.**

```python
ids = df_rate.sort_values('sugarR_100Hz', ascending=False).index[:3]
for i in ids:
    run_exp(exp_name=f'sugarR-{i}', neu_exc=neu_sugar, neu_slnc=[i], params=params, **config)
```

`model.silence()` zeroes every synapse out of the named neuron. Re-run, and
measure how far MN9's rate falls. This ranks neurons by *necessity* for the
labeled line. It is an intervention inside one graph, not a comparison between
graphs.

**Step 7 — validate outside the model.** They compare against real optogenetic
activation in living flies. We cannot do this and must not pretend to.

### 2.2 The thing you must understand about their controls

**[SH] never compared against a rewired or random connectome.** Their
specificity comes from somewhere else entirely: from the claim that *this named
input drives that named output and not the others.* The contrast is **other
input sets** — sugar vs water vs bitter — and **silencing**, not other graphs.

Our specificity bar (real vs degree-preserving-rewired vs random-weight) is
**strictly harder than anything [SH] ran.** We should keep it — it is our moat —
but we have been applying it to a readout that cannot possibly pass it. A scalar
distance has no identity in it. Give the test something with identity to bite on.

### 2.3 What this loop is therefore about

> Does driving the frozen VNC connectome through **named** descending inputs
> produce **structured, identity-specific activation of named leg motor neuron
> pools** — antagonist pairs in anti-phase, left/right in anti-phase, distinct
> profiles for distinct descending populations — and is that structure destroyed
> by degree-preserving rewiring?

That is [SH]'s question, asked of the VNC, on our hexapod. It is answerable with
the assets we already have, and it is a question a random graph can actually fail.

---

## 3. What you already have — inventory before you write anything

### 3.1 The labeled lines are already in the index — verified contents

`eon-fly-brain/results/path2/all_legs_io_index.npz`. Load with
`np.load(path, allow_pickle=True)`.

| key | shape | what it is |
|---|---|---|
| `motor_out` | (260,) | node indices of the 260 leg motor neurons |
| `motor_labels` | (260,) `<U96` | **MANC type names** — these are our MN9s |
| `motor_leg` | (260,) | 0–5, index into `leg_names` = `T1L T1R T2L T2R T3L T3R` |
| `motor_joint` | (260,) | 0–4, index into `joint_names` = `coxa trochanter femur tibia tarsus` |
| `motor_sign` | (260,) float32 | **±1 — this IS the antagonist label** (see below) |
| `motor_pool_sizes` | (6,5) | neurons per leg × joint |
| `descending_in` | (1325,) | descending input node indices |
| `dng100_left` (1,) `dng100_right` (1,) | | named walk-command DN, one node each |
| `dnb08_left` (2,) `dnb08_right` (2,) | | named walk-command DN, two nodes each |
| `premotor` | (9292,) | |
| `published_coordination` | (212,) | named coordination interneurons |
| `published_coordination_types` | (212,) | eLife 96084 type names, 62 distinct present |
| `published_coordination_neuromeres` / `_sides` | (212,) | `T1/T2/T3`, `L/R` |
| `rhythm_core` | (18,) | nodes of `IN17A001`, `INXXX466`, `IN16B036` |
| `signed_edge_count` | () | `485074` |

**Label format.** `motor_labels` entries duplicate the type and suffix the side,
e.g. `'Ti flexor MN Ti flexor_L'`. Exact parse recipe used to produce the tables
below — use this one, do not invent another:

```python
mtype = ' '.join(str(label).split(' ')[:-1])   # -> 'Ti flexor MN Ti flexor'
side  = str(label).split('_')[-1]              # -> 'L' or 'R'
```

**The 14 types, with their joint and their sign — verified, not assumed:**

| joint | sign `+1` | sign `-1` |
|---|---|---|
| coxa | Tergopleural/Pleural promotor (8), Sternal anterior rotator (12) | Pleural remotor/abductor (14), Sternal posterior rotator (22), Sternal adductor (6) |
| trochanter | Tr flexor (35), Acc. tr flexor (22) | Tr extensor (11) |
| femur | *(none)* | Fe reductor (20) |
| tibia | Ti flexor (37), Acc. ti flexor (47) | Ti extensor (12) |
| tarsus | Ta levator (5) | Ta depressor (9) |

**`motor_sign` is the agonist/antagonist polarity.** `+1` = flexor / levator /
promotor / anterior-rotator. `-1` = extensor / depressor / remotor /
posterior-rotator / adductor. This is the labeled line you are testing.

**Which leg×joint pairs are actually testable** — both polarities present:

| joint | testable legs | note |
|---|---|---|
| coxa | **6 of 6** | all legs |
| trochanter | **6 of 6** | all legs |
| tibia | **6 of 6** | all legs |
| tarsus | **2 of 6** | T1L, T1R only — T2/T3 have no tarsus pool |
| femur | **0 of 6** | `Fe reductor` only, no antagonist — **not testable, do not try** |

So the antagonist test has **20 testable leg×joint cells**, not 30. Any success
criterion phrased over "6 legs" must be read per joint. Hard-code this table into
the preflight and assert it (§9.1).

**Warning — `leg_scores` destroys exactly the signal you need.**
`audit_vnc_legs.leg_scores` computes, per leg×joint:

```python
score += history[:, sign > 0].mean(axis=1)
score -= history[:, sign < 0].mean(axis=1)
```

That is the agonist pool **minus** the antagonist pool, collapsed to one number.
If flexor and extensor are perfectly anti-phase, this maximises the score — and
throws away the fact. **The antagonist readout must index `history[:, mask]`
directly and never route through `leg_scores`.** Keep `leg_scores` only for
backward comparison with the previous campaign.

### 3.2 Our code you will reuse, not rewrite

| file | what it gives you |
|---|---|
| `code/vnc_lif_core.py` | `ShiuLIFStepper` — [SH] dynamics, our graph. Has `.W`, `.reset()`, `.step()`, `.descending_current()`, `.core.generator.manual_seed()`, `.core.rate_ceiling_hz` |
| `code/audit_vnc_legs.py` | `leg_scores(history, io) -> (T, 6, 5)` — the existing aggregate. **You will need something finer; see §5.1** |
| `code/probe_vnc_phase.py` | `fft_bandpass`, `analytic_signal`, `wrap_cycles`, `band_power_share` — reuse these, do not reimplement Hilbert transforms. `phase_metrics` is fixed at `(T, 6)` legs and is **only** usable for §5.1(c); see §3.4 |
| `code/reservoir_3way.py` | `degree_preserving_rewire(W, seed)`, `randomize_weights(W, seed)` — the controls. Preserve row AND column degree plus weight multiset |
| `code/e0_lif_controls.py` | The pattern for a parallel multi-arm, multi-realization sweep. Copy its structure: one stepper per worker, weights swapped through `.W`, results keyed by `(kind, seed)` and sorted before writing |
| `code/audit_h5_lesion_localization.py` | Existing **lesion** machinery — this is our `model.silence()`. Read `LESION_CANDIDATES` |
| `code/audit_amplitude_control.py` | `amplitude_match`, `array_sha256` — activity matching so a control stays a control |
| `code/specificity_campaign.py` | The harness. `EPISODE_SEED`, `SIGNAL_SAMPLE_DT`, `ARM_SUFFIX`, `run_body_trace`, `episode_specs` |
| `code/nmf_cpg.py` | The **corrected** CPG from the last loop — `PreprogrammedHexapodSteps` with source-sized retargeting and fixed phase direction. Do not re-break it |
| `code/build_dn_command_map.py` | The fixed 4-channel descending expansion (`go`, `steer`, `family`, `walkdn`). Fixed, hashed, never trained |

### 3.3 Reference material, vendored

- `fly_field_refs/shiu_brain_model/` — [SH] code. `model.py`, `utils.py`, `example.ipynb`.
- `fly_field_refs/flygym/src/flygym_demo/complex_terrain/` — [NMF] CPG and
  `preprogrammed.py`.
- `fly_field_refs/nmf2-paper/` — [NMF] paper repo.

[SH] = Shiu et al., *Nature* 634:210–219 (2024).
[NMF] = Wang-Chen et al., *Nature Methods* 21:2353–2362 (2024).

### 3.4 Verified API — signatures checked against the code, use these exactly

```python
# --- core ---------------------------------------------------------------
from vnc_lif_core import ShiuLIFStepper
stepper = ShiuLIFStepper(RESULTS, seed=1, device="cuda",
                         weights_name="all_legs_weight_csr.pkl",
                         index_name="all_legs_io_index.npz")
stepper.io                       # the npz above
stepper.n                        # 11000
stepper.W                        # signed synapse COUNTS (the setter rescales to mV and calls reset())
stepper.W = counts_tensor        # swap a control graph in; CPU tensor is moved for you
stepper.reset()
stepper.core.generator.manual_seed(1000 + realization)   # the Poisson seed
stepper.core.rate_ceiling_hz     # divisor used to un-normalise motor output
stepper.descending_current(drive, turn=0.0, dnb08_gain=1.0, turn_scale=1.0)  # -> (11000,) float32
motor = stepper.step(current)    # -> torch (260,), rates ALREADY normalised by rate_ceiling_hz

# --- controls -----------------------------------------------------------
from reservoir_3way import degree_preserving_rewire, randomize_weights
W2 = degree_preserving_rewire(weight: torch.Tensor, seed: int)   # CPU, numpy-backed
W3 = randomize_weights(weight: torch.Tensor, seed: int)

# --- activity matching --------------------------------------------------
from audit_amplitude_control import amplitude_match, array_sha256
matched, report = amplitude_match(history, target_sd, allow_unmatchable=False)
#   rescales each channel's temporal SD to target_sd, preserving mean.
#   allow_unmatchable=True is REQUIRED for rewired arms: a rewire can leave a
#   channel with no surviving drive, and the default aborts the whole run.

# --- phase machinery ----------------------------------------------------
from probe_vnc_phase import fft_bandpass, analytic_signal, wrap_cycles, band_power_share
BANDPASS_HZ = (1.5, 4.0)          # module constant
PHASE_TOLERANCE_CYCLES = 0.15     # module constant
TRIPOD_A, TRIPOD_B = (0, 3, 4), (1, 2, 5)

# --- harness ------------------------------------------------------------
import specificity_campaign as SC
SC.EPISODE_SEED        # 20260829
SC.SIGNAL_SAMPLE_DT    # VNC_DT * VNC_SUBSTEPS -- the control-grid timestep
SC.ARM_SUFFIX          # "" for rate core, "_lif" for the published core
SC.base_vnc(SC.VNC_DATA)
SC.episode_specs(n, SC.EPISODE_SEED)
SC.run_body_trace(targets, io, spec, secs, perturbation=("leg_length", 1.0),
                  horizons=None, video=None, budget=None, model=None, body_record=None)
```

**`probe_vnc_phase.phase_metrics` is NOT reusable for the antagonist test.** Read
its body: it takes `scores` of shape `(T, 6)` and hard-codes the six legs and the
two tripod groups. It works for §5.1(c) inter-leg phase and nothing else. For
§5.1(b) antagonist phase you write a small new function using only
`fft_bandpass`, `analytic_signal` and `wrap_cycles`. Do not widen `phase_metrics`
— the existing campaign depends on its exact behaviour.


---

## 4. Pod operations

**§4 of `CODEX_CPG_TRANSLATION_LOOP.md` applies unchanged and in full. Read it.**
The essentials repeated here so this file is runnable on its own:

### 4.1 Cost ceiling — `$0.20/hr`, hard

| GPU | $/hr | RAM | vCPU | verdict |
|---|---|---|---|---|
| **RTX 4000 Ada** | **0.200** | 50 GB | 9 | first choice |
| RTX A5000 | 0.160 | 34 GB | 9 | fallback |
| RTX A4500 | 0.190 | 31 GB | 8 | works |
| RTX 4070 Ti | 0.190 | 24 GB | 12 | what the last loop used |
| RTX 3080 Ti | 0.180 | 30 GB | 12 | works |
| RTX 3070 | 0.130 | 17 GB | 8 | **banned — will OOM** |

Prices move. Check `runpodctl get cloud -c` before creating. Never raise
`--cost` above `0.20`. If nothing is available at the cap, wait or report blocked.

```bash
runpodctl create pod \
  --name flycraft-labeled-line \
  --gpuType 'NVIDIA RTX 4000 Ada Generation' --gpuCount 1 \
  --imageName 'runpod/pytorch:1.1.0-cu1281-torch271-ubuntu2204' \
  --containerDiskSize 30 --volumeSize 30 --mem 30 --vcpu 8 \
  --communityCloud --startSSH --ports '22/tcp' --cost 0.20
```

### 4.2 Connect

```bash
runpodctl pod list
runpodctl get pod <POD_ID> -o json     # ssh host/port live in the ports field

KEY=/Users/puneeth/.runpod/ssh/runpodctl-ssh-key
SSH="ssh -i $KEY -p <PORT> -o StrictHostKeyChecking=no -o BatchMode=yes -o ServerAliveInterval=15"
```

`ServerAliveInterval=15` on **every** long connection. Without it a watcher dies
with `client_loop: send disconnect: Broken pipe` and exit 255 — **the job keeps
running**. Reconnect and check state before concluding anything failed.

### 4.3 Deploy

```bash
rsync -av --exclude '__pycache__' --exclude '._*' -e "$SSH" \
  /Users/puneeth/Flycraft/eon-fly-brain/code/ root@<HOST>:/root/eon-fly-brain/code/
rsync -av -e "$SSH" \
  /Users/puneeth/Flycraft/eon-fly-brain/results/path2/all_legs_*.{pkl,npz} \
  root@<HOST>:/root/eon-fly-brain/results/path2/
rsync -av -e "$SSH" \
  /Users/puneeth/Flycraft/eon-fly-brain/data/manc/ root@<HOST>:/root/eon-fly-brain/data/manc/
$SSH root@<HOST> 'mkdir -p /root/eon-fly-brain/data/nmf'
rsync -av -e "$SSH" \
  /Users/puneeth/Flycraft/fly_field_refs/flygym/src/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl \
  root@<HOST>:/root/eon-fly-brain/data/nmf/
```

Exclude `._*` — macOS AppleDouble junk shipped to a pod once already.

### 4.4 Environment

```bash
pip install -q --only-binary=:all: mujoco imageio imageio-ffmpeg scipy pandas pyarrow matplotlib
apt-get update -qq && apt-get install -y -qq libegl1 libglvnd0 libgles2
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2     # MANDATORY — the LIF core OOM-killed a 62 GB pod
```

At most **3 experiment processes in parallel** on a 30 GB box. A job that vanishes
with no traceback is almost always the OOM killer — check `dmesg | tail` first.

### 4.5 Launch and monitor

Always detached:

```bash
$SSH root@<HOST> 'cd /root/eon-fly-brain/code && \
  setsid nohup python3 -u labeled_line_campaign.py --out ../results/path2/labeled_line.json \
  > ../results/path2/ll_logs/run.log 2>&1 < /dev/null & echo $!'
```

Poll in short-lived connections:

```bash
$SSH root@<HOST> 'pgrep -af labeled_line; tail -20 /root/eon-fly-brain/results/path2/ll_logs/run.log'
$SSH root@<HOST> 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
$SSH root@<HOST> 'free -g'
```

**0% GPU / 1 MiB means the job is on CPU.** That bug has cost this project hours
twice. Check `--device cuda` reached the core.

### 4.6 Pull, then terminate

```bash
rsync -av -e "$SSH" root@<HOST>:/root/eon-fly-brain/results/path2/ \
  /Users/puneeth/Flycraft/eon-fly-brain/results/path2/
```

Verify locally — open the JSON, confirm the row count you expected — **then**:

```bash
runpodctl remove pod <POD_ID>
runpodctl pod list          # must come back []
```

**Write every artifact to disk as it is produced.** A previous CPG frequency sweep
existed only in a pod's stdout and died with the pod; only the chosen value
survives, in a comment at `render_best_walk.py:84`. Do not repeat that.

Terminate after every iteration. Do not leave a pod idle overnight.

---

## 5. The actual work

### 5.1 Build the labeled-line readout — this replaces the scalar

**RULE 0 — window duration. This is not tunable and not a cost tradeoff.**

`SIGNAL_SECS = 30.0` at `SC.SIGNAL_SAMPLE_DT` (0.02 s) = **1500 steps**, matching
`e0_lif_controls.STEPS` and `specificity_campaign.SIGNAL_STEPS`. Burn-in discards
half, leaving 15 s analysed = **22 cycles** of the 1.5 Hz band edge and **1 / 15 s
= 0.067 Hz** resolution, i.e. ~37 usable bins across `BANDPASS_HZ`.

[SH]'s `t_run = 1000 ms` is right **for rates** and wrong for phase. Rates
converge in a second; a 1.5 Hz phase does not. Run 1 cut to 1 s on exactly this
reasoning and destroyed the measurement (§0b, Bug 1). If pod time is tight,
**reduce the rate grid or the number of control seeds — never the window.**

Compute and assert this before every launch:

```python
analysed_s   = SIGNAL_SECS * (1.0 - BURN_FRACTION)
cycles_low   = analysed_s * BANDPASS_HZ[0]
bins_in_band = analysed_s * (BANDPASS_HZ[1] - BANDPASS_HZ[0])
assert cycles_low   >= 10, f"only {cycles_low:.1f} cycles of the low band edge"
assert bins_in_band >= 8,  f"only {bins_in_band:.1f} FFT bins in the band"
```

Record `analysed_s`, `cycles_low` and `bins_in_band` in every artifact's
`protocol` block so any future reader can check this in one glance.


`stepper.step()` already returns exactly what you need: a `(260,)` vector of
per-motor-neuron normalised rates, one entry per named MANC motor neuron. Stack
it into `history` of shape `(T, 260)`. **That array is the labeled-line
substrate.** Everything below is computed from it directly — no `leg_scores`.

Write `code/labeled_line.py` producing four objects.

**(a) Per-type rate table — the direct [SH] analogue of `utils.get_rate`.**
For each of the 14 named types × 6 legs: mean rate over time, then mean and
**standard deviation across realizations**. Multiply by
`stepper.core.rate_ceiling_hz` to report in Hz so the numbers are comparable to
[SH]'s. Report rate and SD together, always, in the same row — that is the whole
point of `utils.get_rate` returning two frames.

**(b) Antagonist phase table — the core new measurement.**
For each of the **20 testable leg×joint cells** in §3.1 (coxa, trochanter, tibia
on all 6 legs; tarsus on T1L/T1R only; femur never):

```python
mask   = (motor_leg == L) & (motor_joint == J)
ago    = history[:, mask & (motor_sign > 0)].mean(axis=1)   # (T,)
anta   = history[:, mask & (motor_sign < 0)].mean(axis=1)   # (T,)
tail   = slice(len(ago) // 2, None)                          # burn-in, as phase_metrics does
fa, fb = fft_bandpass(np.stack([ago, anta], 1)[tail], SC.SIGNAL_SAMPLE_DT, *BANDPASS_HZ).T
z      = np.mean(np.exp(1j * (np.angle(analytic_signal(fa)) - np.angle(analytic_signal(fb)))))
phase_cycles  = np.angle(z) / (2 * np.pi)
concentration = abs(z)
```

Report `phase_cycles`, `concentration`, and the across-realization SD of
`phase_cycles` for every cell. **Prediction, written into the log before the
run: `|phase_cycles|` near `0.5` (anti-phase).** `concentration` near 0 means
there is no consistent phase relationship at all and the angle is meaningless —
treat a low-concentration cell as missing data, not as a value.

**(c) Inter-leg phase table.** Here `probe_vnc_phase.phase_metrics(scores,
SC.SIGNAL_SAMPLE_DT)` *is* the right tool, but it wants `(T, 6)`. Build that as
per-leg summed activity across all its motor neurons (**not** `leg_scores`,
which subtracts). It returns `tripod_coherence_scalar`, the within/cross pair
phases, and a verdict. Within-tripod ≈ 0, cross ≈ 0.5.

**(d) Selectivity matrix — the object the loop exists to produce.**
Rows = descending input sets (§5.2) × rate points. Columns = the 14 types.
Entry = mean rate. **Z-score within each row** so a graph cannot win by being
uniformly louder. Report the raw matrix too, so the z-scoring is auditable.

### 5.2 Define the named input sets — commit before running

**Do not touch the drive implementation. It is already [SH]-faithful.**
`vnc_lif_core.py` restored the Poisson source deliberately, and the reasoning is
in the comments at lines 191–220. One Bernoulli draw per substep at
`p = rate_hz * dt`, which is the Poisson process thinned to the substep grid
(exact to `O(p²)`), plus `refractory_per_neuron = 0` on drive nodes — both halves
of [SH] `model.py:85-92`, where `PoissonInput` has weight `w_syn * f_poi` =
`68.75 mV` and `neu[i].rfc = 0 * ms`. A previous version substituted the mean
current and silently saturated: requests of 50 / 100 / 150 Hz measured
341 / 489 / 498 Hz. **If you "fix" the drive you will reintroduce that bug.**
The units are already what [SH] hand `PoissonInput`: the value you pass is a
stimulation **rate in Hz**.

`descending_current(drive, turn, dnb08_gain, turn_scale)` writes only into
`dng100_left/right` and `dnb08_left/right` — **it does not drive all 1325
descending inputs.** So build the other input sets as raw `(11000,)` float32
vectors yourself. Declare all four in the log, with rationale, before any rollout:

| set | how to build | rationale |
|---|---|---|
| `dng100` | zeros; set `io["dng100_left"]` and `io["dng100_right"]` to `rate` | named walk-command DN, 2 nodes |
| `dnb08` | zeros; set `io["dnb08_left"]` and `io["dnb08_right"]` to `rate` | named walk-command DN, 4 nodes |
| `dng100_left_only` | `descending_current(rate, turn=+1.0)` — or zeros with only `dng100_left` set | unilateral; **should bias one side** |
| `all_descending` | zeros; set **all** of `io["descending_in"]` to `rate` | the undifferentiated drive used until now — the **negative control for identity** |

Rate grid, declared once and run in full: `(50, 100, 150, 200) Hz`. This spans
[SH]'s `r_poi` default of `150 Hz` and the `100 Hz` variant in their notebook.
Do not narrow it after seeing results.

**Note the asymmetry and control for it.** `dng100` is 2 nodes, `dnb08` is 4,
`all_descending` is 1325. Total injected drive therefore differs by ~600×. Report
`frac_active` and `motor_hz` for every set, and interpret the selectivity matrix
**within rows** (§5.1d), never across them, for exactly this reason.

**The core prediction:** `dng100` and `dnb08` produce **different** selectivity
rows. If all four sets give the same row, the graph routes nothing — that is a
publishable null and you should say so plainly rather than search for a set that
separates.

### 5.3 The graph controls — keep our bar, it is harder than theirs

**RULE 1 — activity matching is verified after it is applied, not merely called.**

Run 1 named `amplitude_match` and still shipped controls firing 2.6× harder than
real (§0b, Bug 2). Calling the function is not the check. Do this:

1. Compute `motor_hz` and `frac_active` for **every** arm, from the same
   `history` the metrics are computed from — not from a separate pass.
2. Match, then **re-measure and assert**:

```python
for arm in ("rewired", "random"):
    r = signal[arm]["motor_hz"] / signal["real"]["motor_hz"]
    assert 0.8 <= r <= 1.25, f"{arm} motor_hz ratio {r:.2f} — not a control"
    f = signal[arm]["frac_active"] / signal["real"]["frac_active"]
    assert 0.8 <= f <= 1.25, f"{arm} frac_active ratio {f:.2f} — not a control"
```

3. If an arm cannot be brought inside the band, **that is the result to report** —
   "the rewire changes activity so much it cannot be matched" is a real and
   interesting finding. Do not proceed to a phase comparison against an
   unmatched arm and do not widen the tolerance.

The old 2× `frac_active` gate was too loose to catch a 2.6× `motor_hz` gap.
Both metrics, both directions, 0.8–1.25. `allow_unmatchable=True` is still
required on rewired arms (a rewire can leave a channel with no drive), and the
count of unmatchable channels goes in the artifact.


Three arms, as in `e0_lif_controls.py`:

- `real` — the frozen measured graph, **30** Poisson realizations ([SH]'s `n_run`)
- `rewired` — `degree_preserving_rewire`, ≥12 seeds × 10 realizations
- `random` — `randomize_weights`, ≥12 seeds × 10 realizations

Amplitude-match with `audit_amplitude_control.amplitude_match` and report
`frac_active` and `motor_hz` for **every** arm. A control is only a control while
it is as active as the thing it controls. Every arm sees the identical set of
realization seeds so the arms differ by graph and by nothing else.

**Where the previous loop died:** its margin (`0.3845 m`) was smaller than the
real arm's own realization SD (`0.4083 m`). Compute the real arm's SD **first**,
and state it in the log **before** you look at any margin. Any effect under it is
noise. Say so.

### 5.4 The silencing ladder — [SH]'s causal test, ported

`model.silence()` zeroes all synapses out of a neuron. `audit_h5_lesion_localization.py`
already does the equivalent by source-edge lesion. Extend it to silence **by named
type**, not by segment:

- each of the 3 `rhythm_core_labels` (`IN17A001`, `INXXX466`, `IN16B036`)
- the top-15 `published_coordination_types` by out-degree onto `motor_out`

For each, re-run and report how far the **antagonist anti-phase (5.1b)** falls —
not how far the robot walked. Rank types by necessity, exactly as [SH] rank
neurons by MN9 rate drop.

Then the part [SH] could not do: **repeat the ranking under a rewired graph.** If
the real ranking is structured and the rewired ranking is flat or reshuffled, that
is a connectome-specific result with identity in it. This is the strongest test
available to us and it does not exist anywhere in the literature.

### 5.5 Keep the body, but demote it

Still run `specificity_campaign.run_body_trace` with the corrected `nmf_cpg`
prior so the readout stays embodied and videos still render. But forward distance
is now a **descriptive** number, not the decision variable. The decision variable
is §5.1's structure. Say this explicitly in the artifact so nobody later mistakes
a distance for the finding.

---

## 6. The loop

### 6.0 Iteration 1 is fixed: find where the power is. Do this first.

Do not run the full grid until this is answered. Run 1 assumed the rhythm lives
in `BANDPASS_HZ = (1.5, 4.0)` and measured a band share of `0.1489` — but in a
0.5 s window that number meant nothing (§0b). Nobody has yet established where
the real graph's motor-output power actually sits.

So: one short pod session, real graph only, `SIGNAL_SECS = 30.0`, primary drive
`dng100 @ 150 Hz`, 30 realizations. Dump the **full power spectrum** of the
per-leg and per-pool motor signals from 0 to 25 Hz. Write it to
`ll_logs/spectrum_iteration1.json` and plot it.

Then answer, in the log, before touching anything else:

- Where is the peak? Is there one?
- What fraction of total power is in `(1.5, 4.0)`, measured properly this time?
- Real mean `motor_hz` was `6.29` in run 1. Is there rhythmic power near there,
  well above the band the analysis uses?
- Does the spectrum differ between real and a single rewired seed?

**If the power is somewhere else, say so and stop.** Changing `BANDPASS_HZ` is
then a declared, logged, one-time change justified by the spectrum — with the
full spectrum published alongside — and never a search for a band that produces a
result. If there is no spectral peak at all, the honest finding is that the
frozen VNC under Poisson descending drive produces no rhythm to phase-lock, and
that is a real, reportable answer to the original question.

### 6.1 Then the loop proper


Repeat until a stop condition in §7 fires:

1. **State the hypothesis** in one sentence, in `ll_logs/ITERATION_LOG.md`, before
   any code.
2. **State the prediction** — what the selectivity matrix or phase table should
   look like if the hypothesis holds, and what it looks like if it does not. Write
   the numbers you expect where you can.
3. **Make exactly one change.** One. If a run needs two changes, that is two
   iterations.
4. **Verify locally** — `py_compile`, the self-checks in §9.1, a pure-math smoke
   test. No physics on the laptop.
5. **Create the pod** (§4.1), deploy, run detached, monitor.
6. **Pull artifacts, verify them locally, terminate the pod.** Every time.
7. **Run the full control set** (§5.3) — never report a real-arm number without
   its rewired and random counterparts and the real arm's realization SD.
8. **Run the self-interrogation** (§8.3) and paste the answers into the log.
9. **Write the result into the log** — including when it is a null, especially
   when it is a null.
10. **Decide**: continue, or stop per §7.

One iteration = one pod = one hypothesis. Do not batch hypotheses to save money;
a confounded run wastes more than a pod-hour.

---

## 7. Success and stop criteria — declared now, not later

### Success (all must hold)

Phrased over the **20 testable leg×joint cells** of §3.1, never over "6 legs".

- **S1.** For the real graph, `|phase_cycles|` sits in `0.5 ± 0.15`
  (`PHASE_TOLERANCE_CYCLES`) on **at least 12 of the 20** testable cells, counting
  only cells whose `concentration >= 0.3`. Cells below that threshold are missing
  data and must be reported as such, not scored.
- **S2.** The **rewired** arm's count of passing cells is lower than real's by
  more than the real arm's across-realization SD of that count.
- **S3.** The **random** arm likewise. Beating one control is not a result — the
  previous loop beat rewired and lost to random, and that is a null.
- **S4.** At least two named descending sets (§5.2) give selectivity rows whose
  separation (cosine distance on the z-scored rows) is greater for the real graph
  than for **both** the rewired and the random arms, by more than the real arm's
  across-realization SD of that distance.
  *Run 1's S4 compared real only against its own SD and reported `true` while
  random beat real 0.207 to 0.097 (§0b, Bug 3). A control must be in the
  comparison or the criterion is decoration.*
- **S6.** Every arm passes the activity-match assertion of §5.3 RULE 1
  (`motor_hz` and `frac_active` ratios both within 0.8–1.25 of real). An
  unmatched arm makes S2, S3 and S4 unreadable.
- **S7.** The window assertion of §5.1 RULE 0 passes: `cycles_low >= 10` and
  `bins_in_band >= 8`. Without it there is no phase measurement to interpret.
- **S5.** All of the above with `zero_learning: true`, `parameter_updates: 0`,
  `file_hashes_equal: true`, `real_tensor_hash_equal: true`.

### Stop and report a null

- Antagonist phase is unstructured for the real graph — fewer than 12 of 20 cells
  pass S1, or most cells fall below the concentration floor. There is no labeled
  line to test, and that is itself the finding.
- Real matches its controls within realization SD across the whole rate grid.
- All four descending sets produce the same selectivity row (no routing).
- **In-band power is too low to support a phase claim** — see §9.2b. If
  `band_power_share` is at the recorded `0.14–0.19`, the phase you are measuring
  may be the phase of noise. Resolve this before reporting either way.
- You have run **8 iterations** without S1 holding.

### Stop and report a bug, not a result

- Any arm's `frac_active` differs from real's by more than 2×.
- Any freeze hash mismatches.
- The real arm's realization SD exceeds the effect you are reporting.

**Write the null up as carefully as you would write up a win.** The previous
loop's null is one of the better artifacts in this project precisely because its
positive control worked.

---

## 8. Anti-cheat

### 8.1 The ablation ladder — mandatory on every reported arm

| # | arm | what it does | what it proves |
|---|---|---|---|
| A | `full` | Real connectome, named DN drive | the claim |
| B | `constant-connectome` | Replace the connectome's output with its own temporal mean | **if B ≈ A, the connectome's timing contributes nothing** |
| C | `time-shuffled` | Permute the history along the time axis | separates temporal structure from amplitude |
| D | `drive-only` | No connectome; Poisson straight to the readout | the floor |

**A-vs-B is the decisive test for the RATE and SELECTIVITY readouts** (§5.1a,
§5.1d). The previous loop's V2 lost to its own constant ablation by `0.0031 m`.
Report the A−B difference explicitly for every arm.

**But the ladder is degenerate for the PHASE readouts** (§5.1b, §5.1c), and you
must not report it as if it were not. Arm B replaces the signal with its temporal
mean — a constant has no phase, so B fails trivially and A "wins" by construction.
Arm D is white Poisson, which likewise has no phase structure. Neither tells you
anything about the connectome.

For the phase readouts the meaningful controls are therefore:

- **C, time-shuffled** — same amplitude distribution, destroyed timing. This is
  the real within-arm control for phase. Report it.
- **The graph controls of §5.3** — rewired and random. These carry the weight of
  the phase claim, because they preserve activity and destroy *wiring identity*,
  which is exactly the thing under test.

Report A, B, C and D for the rate/selectivity readouts. For phase, report A, C,
rewired and random, and state plainly in the artifact that B and D are degenerate
for that readout rather than quietly omitting them. Omitting a control without
saying why is how a ladder becomes decoration.

### 8.2 Known cheats — check each one before reporting

| cheat | how it sneaks in | the check |
|---|---|---|
| Structured drive | Poisson replaced by anything with timing | assert drive autocorrelation is flat (§9.1) |
| Learned interface | a per-DN gain, however innocent-looking | `build_dn_command_map` channels only; hash the map |
| Post-hoc rate pick | choosing the Hz that worked | report the full grid and neighbours |
| Louder-not-smarter | real arm simply more active | `amplitude_match`; report `frac_active` and `motor_hz` per arm |
| Dead controls | rewire silences the network | assert every arm's `frac_active` within 2× of real |
| Realization cherry-pick | one lucky Poisson seed | 30 realizations, report SD, same seeds every arm |
| Phase from the CPG | anti-phase inherited from the hand-written gait | **run the phase readout open-loop, with no body and no CPG at all** |
| Silent overwrite | LIF run clobbers a rate artifact | keep `ARM_SUFFIX` |

The **phase-from-the-CPG** cheat is the dangerous one here. If the CPG is running,
it imposes anti-phase on the joints regardless of what the connectome does.
Therefore §5.1(b) and (c) must be computed on the **core's own motor output with
the body and the CPG entirely absent.** The embodied run is separate and
descriptive. Do not mix them. Assert it (§9.1).

### 8.3 Self-interrogation — answer all of these in the artifact, every run

- `a` — A minus B, in the readout's own units. If ≈ 0, say so.
- `b` — parameter range fraction touched by each arm.
- `c` — was any constant selected from an outcome being reported? Name it.
- `d` — identical episode/realization specs across arms? Hash them.
- `e` — weight hashes match pre/post?
- `f` — realizations per arm, and the real arm's SD.
- `g` — the rewire and random replacement numbers, stated plainly.
- `h` — one sentence naming what the connectome is *not* doing in this result.

---

## 9. Debugging

### 9.1 Assertions — put these in a preflight that gates every launch

```
 1. io["motor_out"].shape == (260,) and motor_labels parse to exactly 14 types
 2. the testable-cell table of 3.1 reproduces EXACTLY:
      coxa 6 legs, trochanter 6, tibia 6, tarsus 2 (T1L,T1R), femur 0
      -> 20 testable leg x joint cells.  Any other count means the index
         changed and every number in this file is stale.
 3. femur is never entered into the antagonist test (it has no +1 pool)
 4. drive integrity: core.drive_spike_prob_per_hz == core.dt / 1000.0, and
    core.refractory_per_neuron[descending_in] == 0.  These are [SH]
    model.py:85-92.  If either fails, the drive was edited -- stop.
 5. drive carries no timing: autocorrelation of the injected drive at lags
    1..20 within +/- 3 sigma of zero
 6. phase readout is CPG-free: assert no CPG object exists in the code path
    computing 5.1(b) and 5.1(c).  This is the cheat in 8.2 that would
    manufacture a false positive.
 7. every arm's frac_active within 2x of the real arm's
 8. band_power_share computed and recorded for every arm (see 9.2b)
 8b. WINDOW (5.1 RULE 0): analysed_s = SIGNAL_SECS * (1 - BURN_FRACTION);
     cycles_low = analysed_s * BANDPASS_HZ[0] >= 10
     bins_in_band = analysed_s * (BANDPASS_HZ[1] - BANDPASS_HZ[0]) >= 8
     Run 1 had cycles_low = 0.75 and bins_in_band = 1.25.  Refuse to launch.
 8c. ACTIVITY (5.3 RULE 1): for every control arm, motor_hz ratio and
     frac_active ratio to real both inside [0.8, 1.25], measured AFTER
     amplitude_match, from the same history the metrics use.
     Run 1 shipped motor_hz ratios of 2.65 and 2.61.
 8d. the three output artifacts have three distinct md5 digests
 9. file hashes and real weight tensor hash equal pre/post; expected real
    tensor sha256 is
    128b5ceddec4d8e14f3a04fbb66a08a2e83564c674624741eb03fb2bf01517fc
10. episode_spec_hash identical across all arms in one comparison
11. torch.cuda.is_available() and the core's device is cuda when --device cuda
    was passed; log both
12. artifact path does not already exist unless --force-overwrite
    (this is [SH]'s own guard in run_exp)
13. ARM_SUFFIX is in every output filename
```

Fail loud. A preflight that warns and continues is not a preflight.

### 9.2 Symptoms and first causes

| symptom | look here first |
|---|---|
| job vanishes, no traceback | OOM killer — `dmesg \| tail`, then `OMP_NUM_THREADS` |
| 0% GPU, 1 MiB | `--device cuda` never reached the core |
| watcher exits 255 | ssh timeout, **not** job failure — reconnect, check `pgrep` |
| all phases ≈ 0 | you are reading the CPG, not the core (§8.2) |
| every arm identical | weights not actually swapped — check `.W` setter fired |
| rewired arm silent | rewire broke in-degree — check `degree_preserving_rewire` invariants |
| rates all zero | Poisson weight too low; [SH] use `w_syn * f_poi` = `.275 mV * 250` |
| SD larger than effect | not a bug, a null. Report it |
| phase looks great, band share ~0.15 | **you are measuring the phase of noise** — see below |
| `amplitude_match` aborts on a rewired arm | pass `allow_unmatchable=True`; a rewire can leave a channel with no drive |
| "found at least two devices, cpu and cuda:0" | control graph built on CPU; the `.W` setter moves it — you bypassed the setter |
| requests of 50/100/150 Hz measure 341/489/498 Hz | someone replaced the Poisson drive with a mean current (§5.2) |

### 9.2b The band-power landmine — read before trusting any phase number

`probe_vnc_phase` band-passes to `BANDPASS_HZ = (1.5, 4.0)` before extracting
phase. Coherence is well defined on a band-passed signal **even when that band
holds almost none of the signal's power** — and then it is the phase of noise.
The recorded `band_power_share` for both cores at four drives is **0.14–0.19**.
That is low.

So: call `band_power_share(scores, SC.SIGNAL_SAMPLE_DT)` on **every** arm and
report it beside every phase number. If the real arm's in-band share is not
meaningfully above its controls', a phase difference between them is not
evidence. This is the most likely way this loop produces a false positive, and
`MEASURED_PHASE_NOT_TRIPOD` firing on 30/30 realizations in the previous loop
(§11) is consistent with the band simply being empty.

If the band is empty, the honest next step is to ask where the power actually is
(dump the spectrum) rather than to widen the band until something appears. If you
do change the band, it becomes a declared, logged, one-time change with the full
spectrum reported — not a search.

### 9.3 Double-checking — do these before writing any number into the log

- Recompute the headline number by a second path (different script, or by hand
  from the raw arrays). Numbers that disagree get resolved, not averaged.
- Re-read the artifact **from disk**, not from memory, when writing the log.
- Confirm the row count matches the declared grid size. A short run is a bug.
- Confirm neighbours of any selected value are present in the artifact.
- Diff your interpretation against §7's declared criteria, literally, one by one.

---

## 10. Bug-hunt loop

When a result looks wrong — too good, too flat, or inconsistent — stop the
experiment loop and enter this one. Do not report anything from a run you have
not first tried to break.

1. **Name the suspicion** in one sentence in the log.
2. **Find the smallest reproduction** — one leg, one pair, one realization, no
   body, no video.
3. **Bisect the pipeline.** Feed the readout (a) a synthetic perfect anti-phase
   signal, (b) pure noise, (c) a constant. It must return `0.5`, unstructured,
   and undefined respectively. A readout that cannot pass this cannot be trusted
   on real data.
4. **Check the boring causes first** — shape mismatch, sign flip, off-by-one in
   the leg/joint index, wrong axis in a mean, stale artifact on disk.
5. **Fix one thing.** Re-run the reproduction. If it still fails, revert and try
   the next hypothesis; do not stack fixes.
6. **Write the diagnostic into the log as its own iteration**, with its numbers,
   even though it produced no result. The previous loop's Iteration 4 was exactly
   this and it is what unblocked the whole thing.
7. **Re-run everything downstream of the fix.** A corrected component invalidates
   every number measured with the broken one. Mark the old ones superseded.

The previous loop found two real bugs this way — full-range target saturation and
an inverted phase direction. Expect this one to have its own.

---

## 11. The numbers you are working against

From `eon-fly-brain/results/path2/cpg_specificity_variants.json`, the previous
loop's final state:

| arm | forward | path/forward | note |
|---|---|---|---|
| V3 phase, real | `0.8386 m` | `2.4624×` | walks |
| V3 phase, random control | `0.8429 m` | | **beat real** |
| V3 phase, rewired control | `0.4540 m` | | margin `0.3845 m` |
| real realization SD | `0.4083 m` | | **larger than the margin** |
| CPG-only, corrected | `0.7715 m` | `2.498×` | the prior alone |
| V2 frequency, real | `0.7647 m` | `2.5095×` | lost to its constant ablation |
| V1 amplitude, real | `0.0722 m` | `21.52×` | dead |

Diagnostic worth carrying forward: `MEASURED_PHASE_NOT_TRIPOD` fired on **30 of
30** realizations. The measured VNC phase, as currently extracted, is not a gait
signal. Either the extraction is wrong or the signal is not there — §10 step 3
distinguishes those, and doing so is a legitimate first iteration.

Freeze audit from that run, which you must match or better:

```
file_hashes_equal: true
real_tensor_sha256: 128b5ceddec4d8e14f3a04fbb66a08a2e83564c674624741eb03fb2bf01517fc
zero_learning: true, parameter_updates: 0
```

Earlier `probe_vnc_phase` E0 numbers (LIF coherence `0.980`, band `0.187`) predate
the core corrections and are **superseded**. Re-measure rather than cite them.

**Run 1 of this loop is also superseded (§0b).** Its numbers —
`4.33 ± 1.37` of 20 antagonist cells, selectivity distance real `0.09697` /
rewired `0.01620` / random `0.20705`, band share real `0.1489` — were produced in
a 0.5 s analysis window with unmatched controls. Do not cite them, do not treat
them as a baseline to beat, and do not report the loop as "already tried". The
artifacts stay on disk as a record of the failure mode:
`labeled_line_{selectivity,phase,silencing}_lif.json` and
`ll_logs/ITERATION_LOG.md`. Start the new log as `ll_logs/ITERATION_LOG_v2.md`
and write fresh artifacts under a `_v2` suffix so the two are never confused.

---

## 12. What to produce

**Artifacts** in `eon-fly-brain/results/path2/`:

- `labeled_line_selectivity.json` — **only** the §5.1(a) rate table and §5.1(d)
  selectivity matrix, all arms, all rate points
- `labeled_line_phase.json` — **only** the §5.1(b) antagonist table and §5.1(c)
  inter-leg table, **per cell**, each with concentration, across-realization SD,
  and that cell's in-band power share
- `labeled_line_silencing.json` — **only** the §5.4 rankings, real and rewired

**These three files must have different content.** Run 1 wrote one blob three
times — `labeled_line_selectivity_lif.json` and `labeled_line_phase_lif.json`
came back byte-identical at `md5 213a88b561f719768c4f69ad3d6a1f9c` (§0b, Bug 4).
Assert before finishing:

```python
digests = {f: hashlib.md5(f.read_bytes()).hexdigest() for f in outputs}
assert len(set(digests.values())) == len(outputs), digests
```

Report the §5.1(b) table **per cell**, never only as a pass count. `4.33 of 20`
tells nobody *which* four, or whether they are the same four across realizations —
which is the difference between a weak signal and no signal. Include leg, joint,
phase_cycles, concentration, SD, and in-band power for all 20 rows.
- `ll_logs/ITERATION_LOG.md` — one entry per iteration, in the previous loop's
  format: hypothesis / change / prediction / verification / status / result /
  freeze audit / interpretation
- `ll_logs/*.log` — raw stdout of every pod run, pulled before termination
- videos only where they show something a table cannot

**Every JSON carries:**

```
experiment, status, decision, protocol,
zero_learning, zero_learning_audit, parameter_updates,
integrity { file_hashes_pre/post, file_hashes_equal,
            real_tensor_sha256_pre/post, real_tensor_hash_equal },
episode_spec_hash,
ablation_ladder { full, constant_connectome, time_shuffled, drive_only },
self_interrogation { a..h },
signal { frac_active, motor_hz, rate_ceiling_hz } per arm,
realization_sd per arm,
reference_departures [ { reference, departure, reason } ],
rows
```

`reference_departures` is not optional. Every place you diverge from [SH]'s
method gets a file:line reference into `fly_field_refs/shiu_brain_model/`, what
you did instead, and why. The previous loop did this well — copy its format.

---

## 13. Final word

The previous loop proved the body works and the interface works. It also proved
that a distance in metres cannot carry a connectome claim, because a random graph
produces distances just as well.

[SH] never had that problem, because they never asked a graph for a number. They
asked it for an **address**: sugar neurons in, MN9 out. The address is the claim.

We have 260 named motor neurons across 14 identified types, arranged in
antagonist pairs across six legs, driven by named descending commands. That is an
address space. Nobody has asked the VNC connectome to route through it in a
non-fly body with frozen measured weights.

Ask it. If it routes, that is the result. If it does not, say so clearly and
stop — an honest null here is worth more than another metre of walking.
