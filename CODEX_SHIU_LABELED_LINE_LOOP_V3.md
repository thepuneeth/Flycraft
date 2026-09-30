# Codex loop v3 — port the [SH] labeled-line method into the VNC hexapod

**This is version 3. Two previous runs of this same experiment returned no
answer.** Run 1 was void (§0b — phase measured in a window shorter than one
cycle). Run 2 was blocked (§0c — a four-line matcher bug let four unmatchable
channels veto sixteen grid conditions). Both failures were instruction bugs, both
were fixable from the code alone, and both are diagnosed in full below with the
exact numbers.

Read §0b, §0c and §0d before anything else. Then §7.5, which sets out where you
have authority to decide and where you must stop — that section is new, and it
exists because the last two runs were damaged by errors a careful reader could
have caught but did not feel licensed to change.

**Supersedes `CODEX_SHIU_LABELED_LINE_LOOP.md`.** Where the two disagree, this
file wins. Where this file disagrees with the source code, **the source wins** —
implement what is correct and log the contradiction.

You are continuing Flycraft. The CPG loop
(`CODEX_CPG_TRANSLATION_LOOP.md`) is **finished**, and its verdict was
`CPG-INTERFACE-SPECIFICITY-NULL`. Read that file's iteration log before you start:
`eon-fly-brain/results/path2/cpg_logs/ITERATION_LOG.md`.

This is a different experiment. Do not redo the CPG one.

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

## 0c. RUN 2 (v2) DID NOT FAIL THE WAY RUN 1 DID — read this before §1

Run 2 fixed every bug in §0b. The window was correct, the assertions fired, the
freeze held, and the agent refused to report a comparison it could not make
fairly. **That refusal was correct behaviour and you should be prepared to do it
again.** But it stopped the run at the gate, so the experiment has no verdict.

This section tells you exactly why the gate fired, because the cause is a
four-line bug and you are going to fix it.

### What run 2 established (keep all of it)

Run 2's window arithmetic passed on the pod: 15 s analysed, 22.5 cycles at the
band's low edge, 37.5 bins in band. Freeze audit clean, real tensor hash equal to
`128b5cedde…517fc`, `parameter_updates: 0`. The §6.0 spectrum diagnostic ran on
an A4500 at `$0.190/hr` and produced the first real-versus-rewired separation
this project has ever measured:

| quantity | real | rewired |
|---|---|---|
| per-leg `(1.5,4.0) Hz` power share | `0.1217` | `0.0591` |
| per-pool power share | `0.0907` | `0.0528` |
| per-leg peak Hz | `1.33, 4.80, 5.00, 19.13, 23.33, 23.53` | `19.60, 20.33, 23.13, 23.73, 24.13, 24.33` |

Read the peak row carefully, because it is the most interesting number in the
project. The real graph's six legs peak at **six scattered frequencies spanning
1.33–23.53 Hz**. The rewired graph's six legs all collapse into a narrow
**19.60–24.33 Hz** band. Same degree sequence, same drive, same realization
seeds — and the frequency structure is qualitatively different.

This is a diagnostic, not the preregistered test, and you must not promote it to
a headline on its own. But it is a reportable artifact
(`ll_logs/spectrum_iteration1.json` and `.png`), it justifies keeping the
declared `(1.5, 4.0) Hz` band, and it says the phase test is worth running
properly rather than abandoning.

### Why the gate fired — the actual four-line bug

At rewired seed `20260830`, `dnb08 @ 100 Hz`:

```
post-match control motor_hz = 9.3718   (sd 5.5648)
real            motor_hz    = 6.9340   (sd 3.7511)
ratio                       = 1.352    > 1.25  ->  abort
unmatchable_channels_total  = 4        (max 2 in any one realization)
```

Reproduced identically on a second pod, so it is deterministic, not noise.

The cause is in `labeled_line_campaign.strict_activity_match`. That function
matches two things — active-channel count and total rate — and when it cannot
match the first, it abandons the second:

```python
if len(source_active) < target_active:
    reports.append({... "scale": None})
    out_row[:] = source_row      # <-- raw, unscaled, hot
    continue                     # <-- never reaches the rate scaling below
```

Every other realization gets `out_row *= scale` and lands at ratio 1.000. The
handful that hit this branch keep their raw rewired rate, and their heat drags
the batch mean to 1.352. **Four unmatchable channels vetoed the entire grid.**

The two metrics are independent. Failing to match channel *count* is not a reason
to leave total *rate* unmatched — you can always scale a rate. Fixing this is
§5.3 Strategy A and it is roughly a ten-line change.

### Two things run 2 surfaced that nobody has explained yet

**(1) `dnb08` is bimodal.** Look at the real arm's own spread:

```
dng100 @ 100 Hz   motor_hz 4.8552   sd 0.0525    <- tight
dnb08  @ 100 Hz   motor_hz 5.9827   sd 3.3477    <- sd is 56% of the mean
dnb08  @ 150 Hz   motor_hz 6.7420   sd 3.2056
dnb08  @ 200 Hz   motor_hz 5.0723   sd 0.0315    <- tight again
```

`dnb08` at 100 and 150 Hz has an across-realization SD two orders of magnitude
larger than its neighbours. That is not measurement noise; the same graph and the
same drive land in two different regimes depending on the Poisson seed. Matching
a control per-realization against a bimodal target is fragile by construction —
and `dnb08 @ 100 Hz` is exactly where the gate fired. **This is not a
coincidence and you must investigate it (§5.6) before you trust any `dnb08` row.**

**(2) `active_sample_fraction` differs 3× and is not in the gate.**

```
real     dng100 @ 150 Hz   active_sample_fraction 0.0985
rewired  dng100 @ 150 Hz   active_sample_fraction 0.2857
```

Same `frac_active` (0.327 vs 0.324) and, after matching, the same `motor_hz` —
but the rewired graph is emitting on **2.9× more sample-slots**. Real is sparse
and bursty; rewired is dense and even. The declared gate checks two metrics and
this third one walks straight through it.

You must decide what to do about this and defend the decision in the log. It is
a genuine open question, not a bug with an obvious fix — see §5.3 Strategy C and
§7.5, which give you the authority to settle it.

---

## 0d. What "100%" means in this run

You are being asked to take this experiment to completion. Completion is defined
by the ledger in §7.6, and it means **a decided experiment**, not a positive
result. A fully-run grid that ends in a clean, well-controlled null is 100%.
A half-run grid that ends in an exciting number is not.

Nobody can promise the connectome wins. What can be promised, and what is being
asked of you, is that every arm runs, every control is fair, every criterion is
evaluated, and the answer — whichever way it falls — is written down and
defensible. Run 1 was void. Run 2 was blocked. This one gets an answer.

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
**within rows** (§5.1(d)), never across them, for exactly this reason.

**The core prediction:** `dng100` and `dnb08` produce **different** selectivity
rows. If all four sets give the same row, the graph routes nothing — that is a
publishable null and you should say so plainly rather than search for a set that
separates.

### 5.3 The graph controls — the part that blocked run 2, and your first real decision

Three arms, unchanged from run 2, all seeing the identical realization seeds so
the arms differ by graph and by nothing else:

- `real` — the frozen measured graph, **30** Poisson realizations ([SH]'s `n_run`)
- `rewired` — `degree_preserving_rewire`, ≥12 seeds × 10 realizations
- `random` — `randomize_weights`, ≥12 seeds × 10 realizations

**RULE 1 stands: activity matching is verified after it is applied.** Both
`motor_hz` and `frac_active` ratios inside `0.8–1.25`, re-measured from the exact
history the readouts consume. Do not widen this tolerance. Run 2 honoured it and
was right to.

What changes in v3 is *how* you reach the band. Run 2 tried one method, it had a
bug, and the run died. You get three, ranked, and **you choose** — see §7.5 for
the scope of that authority.

#### Strategy A — fix the bail-out (mandatory, do this first)

The `continue` branch quoted in §0c abandons rate scaling whenever channel count
cannot be matched. Separate the two metrics:

```python
if len(source_active) < target_active:
    deficit = target_active - len(source_active)
    keep = source_active                      # keep every channel that exists
else:
    deficit = 0
    keep = source_active[np.argsort(source_mean[source_active])[::-1][:target_active]]

out_row[:, keep] = source_row[:, keep]

# Rate scaling now runs unconditionally. A channel deficit is a fact to
# report; it is not a reason to ship an unscaled control.
source_hz = float(out_row.mean() * ceiling)
target_hz = float(target_row.mean() * ceiling)
scale = (target_hz / source_hz) if source_hz > 0.0 else 1.0
out_row *= np.float32(scale)

reports.append({
    "source_active_channels": int(len(source_active)),
    "target_active_channels": target_active,
    "unmatchable_channels": int(deficit),
    "scale": float(scale),
    "rate_matched": True,
})
```

`unmatchable_channels` still lands in the artifact, still gets reported, and
still counts against S6b below. It just no longer vetoes the grid.

**Where Strategy A is legitimate and where it is not.** A post-hoc scalar
multiply is invisible to any scale-invariant readout, and both of the readouts
that carry the claim are scale-invariant:

| readout | scale-invariant? | Strategy A safe? |
|---|---|---|
| §5.1(b) antagonist phase | yes — phase of `x` = phase of `kx` | **yes** |
| §5.1(c) inter-leg phase | yes | **yes** |
| §5.1(d) cosine selectivity distance | yes — cosine ignores magnitude | **yes** |
| §5.1(a) absolute per-pool rate table | **no** — you are setting what you measure | **no, and say so** |

So: report §5.1(a) rates for the real arm and for the **unscaled** controls, and
mark clearly that cross-arm rate comparison is not meaningful under Strategy A.
Phase and selectivity are the claim; rates are context. If you catch yourself
about to write "real fires at a different rate than rewired" after rescaling
rewired to match real, stop — you wrote that number yourself.

#### Strategy B — calibrate at the drive, not the output (preferred if it fits budget)

The principled version. Instead of rescaling a control's output, give the control
its own **input** rate so it lands naturally where real does. This is the same
knob [SH] varied (their sugar-GRN rate sweep), it never touches the frozen
weights, and it leaves §5.1(a) rates meaningful across arms.

`descending_current(drive, turn=0.0, dnb08_gain=1.0, turn_scale=1.0)` takes drive
in Hz. Solve per (graph seed, input set, real rate) for the control drive whose
`motor_hz` matches real's:

```python
def calibrate_drive(stepper, input_set, target_motor_hz, ceiling, steps,
                    seeds, lo=10.0, hi=400.0, tol=0.02, max_iter=8):
    """Bisection on input drive Hz so the control's motor_hz matches real's.

    motor_hz is monotone in drive over this range -- verify that assumption on
    the first arm and record the check. Bisection on a non-monotone function
    silently returns nonsense.
    """
    history = []
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        current = make_drive(stepper, input_set, mid)
        hz = float(collect_history(stepper, current, seeds[:4], steps).mean() * ceiling)
        history.append({"drive_hz": mid, "motor_hz": hz})
        if abs(hz / target_motor_hz - 1.0) <= tol:
            return mid, {"converged": True, "iterations": history}
        lo, hi = (mid, hi) if hz < target_motor_hz else (lo, mid)
    return mid, {"converged": False, "iterations": history}
```

Four calibration realizations per solve is enough to locate the drive; run the
full 10 at the solved drive for the actual measurement. Budget it before you
commit: 8 bisection steps × 4 realizations × 30 s is roughly one full grid cell's
worth of compute per control condition, so Strategy B is meaningfully more
expensive than A. Cost it against §4.1 and say in the log what it came to.

**Check monotonicity first.** Sweep drive over `50, 100, 150, 200 Hz` on one
rewired seed and confirm `motor_hz` rises throughout. Real's own sweep is monotone
(`2.31 / 4.86 / 7.21 / 9.18` for `dng100`), so the assumption is reasonable — but
`dnb08` real goes `2.14 / 5.98 / 6.74 / 5.07`, which is **not** monotone. Do not
run bisection on `dnb08` until §5.6 explains that curve.

#### Strategy C — add the third metric to the gate

`active_sample_fraction` differs 3× between real and rewired (§0c) and is not
gated. You may add it as a **reported, non-gating** diagnostic on every arm, and
you may argue in the log that a 3× difference in emission sparsity means these
graphs are not activity-comparable no matter what is matched.

What you may **not** do is add it as a hard gate without saying so first. Adding a
new gate after seeing the data is how a null gets manufactured. Declare it in the
iteration log *before* the run that uses it, or leave it descriptive.

#### How to combine them

Do Strategy A always — it is a bug fix and it is not optional. Then:

- If §5.6 shows `dnb08`'s bimodality is a tractable, explainable thing and the
  budget allows, do **B on top of A** for the primary condition at minimum, and
  report both A-matched and B-matched results. Agreement between them is strong
  evidence the matching is not driving the answer. Disagreement is a finding.
- If budget is tight, A alone is sufficient to produce a valid, reportable
  result, provided you honour the scale-invariance table above.
- Do C as reporting in every case.

**Where the previous CPG loop died, still true:** its margin (`0.3845 m`) was
smaller than the real arm's own realization SD (`0.4083 m`). Compute the real
arm's SD **first**, state it in the log **before** you look at any margin, and
treat any effect under it as noise. With `dnb08`'s SD at 56% of its mean, this
trap is wide open in this run.

**If an arm still cannot be matched after Strategy A and, where used, B:** that
is a result to report, exactly as run 2 said. But it is now a much stronger claim
than run 2 could make, because you will have shown the failure survives a correct
matcher. Report *which* conditions fail, with their ratios, and run the full grid
on every condition that passes rather than aborting the whole campaign on one
cell. **A single unmatchable condition must not veto the other fifteen** — that
is the specific mistake that left run 2 at 0 rows.

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

### 5.6 The `dnb08` bimodality — mandatory investigation, and possibly the result

`dnb08` real at 100 and 150 Hz has an across-realization `motor_hz` SD of `3.35`
and `3.21` on means of `5.98` and `6.74`. Its neighbours sit at `sd 0.03–0.07`.
The same frozen graph under the same drive lands in two different regimes
depending only on the Poisson seed, and its rate-vs-drive curve is non-monotone
(`2.14 / 5.98 / 6.74 / 5.07` across `50/100/150/200 Hz`).

Run this before the full grid. It is cheap — no controls, no body, one input set.

1. Take the 30 real realizations at `dnb08 @ 100 Hz`. Plot or tabulate
   per-realization `motor_hz`. **Is it bimodal or is it heavy-tailed?** A
   histogram with two separated modes means something different from one long
   tail.
2. If bimodal, split the realizations at the gap and ask what differs. Which
   motor pools are live in the high mode and dead in the low one? Is one mode
   `dng100`-like? Does the switch correlate with early transient activity in the
   first 5 s?
3. Ask whether the high mode is runaway. `v_th - v_0 = 7 mV` and
   `w_syn = 0.275 mV`, so roughly 26 coincident inputs cross threshold. A
   recurrent graph near that boundary can latch. Check whether high-mode
   realizations show `motor_hz` still climbing at 30 s or already flat — a
   climbing trace is saturation, not signal, and would disqualify `dnb08` rather
   than characterise it.
4. Check the same question on `dng100` at every rate as a negative control. If
   `dng100` is unimodal everywhere and `dnb08` is not, the instability is
   specific to that descending channel.

**Then decide, and write the decision in the log before the grid runs:**

- **Bimodal and explained** (e.g. a named pool that latches) — keep `dnb08`, report
  both modes separately, and never average across the gap. An average of two modes
  is a number that describes neither.
- **Runaway saturation** — drop `dnb08` from the primary comparison, keep it as a
  reported limitation, and run the grid on the remaining three input sets. Say
  plainly that one of four declared sets was excluded and why.
- **Heavy-tailed, no structure** — keep it, raise its realization count for that
  condition only, and report the wider error bars honestly.

**This may be the most interesting thing in the run.** A specific named
descending channel driving the frozen VNC into a distinct second regime, when its
neighbour does not, is a structural fact about the measured graph. Check early
whether the rewired and random arms show the same bimodality at `dnb08`: if they
do not, that is identity-bearing evidence of exactly the kind this whole loop is
looking for — and it is a phenomenon, not a scalar, so a matched-activity control
cannot trivially reproduce it. Give it a real look before you file it as a
nuisance.

---

## 6. The loop

### 6.0 Iteration 1: the spectrum question is answered — confirm, then move

Run 1 assumed the rhythm lives in `BANDPASS_HZ = (1.5, 4.0)` and measured a band
share of `0.1489` in a 0.5 s window, where the number meant nothing (§0b). **Run 2
answered this properly** and the artifact is on disk:
`ll_logs/spectrum_iteration1.json` and `spectrum_iteration1.png`, from 30 real and
30 rewired realizations at `dng100 @ 150 Hz`, 30 s, window assertions passed.

The answer (§0c): real per-leg in-band share `0.1217`, rewired `0.0591`. Real
per-leg peaks scattered across `1.33–23.53 Hz`; rewired peaks all inside
`19.60–24.33 Hz`. The band has real power, real has roughly twice the rewired
share, and the two graphs have qualitatively different frequency structure.

**So you do not need to re-run the spectrum. Do this instead, in iteration 1:**

1. Read `spectrum_iteration1.json` and confirm the summary numbers above against
   its contents. If they do not reconcile, that is a bug and it outranks
   everything else in this file.
2. `BANDPASS_HZ` stays at `(1.5, 4.0)`. It is justified by run-2 evidence, it was
   declared before that evidence existed, and it is now locked (assertion 8g).
   Do not retune it.
3. Extend the diagnostic in the one direction run 2 did not: the spectrum was
   measured on **one** rewired seed. Run the same spectrum on a `random`
   (`randomize_weights`) graph and on two more rewired seeds, so the `0.0591`
   figure has a spread rather than being a single point. This is cheap — no body,
   no grid — and it turns a suggestive number into one you can report.
4. Then §5.6 (the `dnb08` bimodality), which is the other thing that must be
   settled before the grid runs.

If step 3 shows the real-versus-control power separation does not survive across
seeds, say so plainly in the log. That weakens the phase test's prospects, and
knowing it before you spend the grid budget is worth the hour.

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

## 7. Success, stop criteria, authority, and the completion ledger

### 7.1 Success (all must hold)

Phrased over the **20 testable leg×joint cells** of §3.1, never over "6 legs".

- **S1.** For the real graph, `|phase_cycles|` sits in `0.5 ± 0.15`
  (`PHASE_TOLERANCE_CYCLES`) on **at least 12 of the 20** testable cells, counting
  only cells whose `concentration >= 0.3`. Cells below that threshold are missing
  data and must be reported as such, not scored.
- **S2.** The **rewired** arm's count of passing cells is lower than real's by
  more than the real arm's across-realization SD of that count.
- **S3.** The **random** arm likewise. Beating one control is not a result — the
  CPG loop beat rewired and lost to random, and that is a null.
- **S4.** At least two named descending sets (§5.2) give selectivity rows whose
  separation (cosine distance on the z-scored rows) is greater for the real graph
  than for **both** the rewired and the random arms, by more than the real arm's
  across-realization SD of that distance.
  *Run 1's S4 compared real only against its own SD and reported `true` while
  random beat real 0.207 to 0.097 (§0b, Bug 3). A control must be in the
  comparison or the criterion is decoration.*
- **S5.** All of the above with `zero_learning: true`, `parameter_updates: 0`,
  `file_hashes_equal: true`, `real_tensor_hash_equal: true`.
- **S6.** Every **reported** arm passes §5.3 RULE 1 — `motor_hz` and `frac_active`
  ratios both inside `0.8–1.25` of real, re-measured after matching.
- **S6b.** *(new in v3)* For every reported arm, `unmatchable_channels_total` and
  `active_sample_fraction` are recorded per condition, and any condition with
  `unmatchable_channels_total > 0` is flagged in the results table. Flagged
  conditions still count; they are not silently dropped and not silently kept.
- **S7.** The window assertion of §5.1 RULE 0 passes: `cycles_low >= 10` and
  `bins_in_band >= 8`.
- **S8.** *(new in v3)* Every declared grid condition has one of exactly three
  terminal states in the artifact: `evaluated`, `excluded-with-reason`, or
  `unmatchable-after-strategy-A`. **No condition may be absent.** Run 2's artifacts
  had empty readout rows because one condition aborted the campaign; that must not
  recur.

### 7.2 Stop and report a null

- Antagonist phase is unstructured for the real graph — fewer than 12 of 20 cells
  pass S1, or most cells fall below the concentration floor. There is no labeled
  line to test, and that is itself the finding.
- Real matches its controls within realization SD across the whole rate grid.
- All four descending sets produce the same selectivity row (no routing).
- In-band power is too low to support a phase claim — see §9.2b. Note that §0c
  now gives you a real measurement here (`0.1217` real vs `0.0591` rewired), so
  "the band is empty" is no longer a live hypothesis; "the band has power but no
  coherent phase" still is.
- You have run **8 iterations** without S1 holding.

### 7.3 Stop and report a bug, not a result

- Any freeze hash mismatches.
- The real arm's realization SD exceeds the effect you are reporting.
- A control's post-match ratio is outside `0.8–1.25` **for every condition**, not
  just some.

**Write the null up as carefully as you would write up a win.** The CPG loop's
null is one of the better artifacts in this project precisely because its
positive control worked.

### 7.4 The distinction that matters most in this run

Three outcomes are all acceptable and all count as finishing:

1. **Positive** — S1–S8 hold. Real routes, controls do not.
2. **Null** — the grid ran, controls were fair, real did not beat them. Reported
   with the same care as a win.
3. **Structural** — the real graph and its degree-preserving rewire are not
   activity-comparable even under a correct matcher, demonstrated across
   conditions rather than inferred from one.

One outcome is **not** acceptable: a stop with no rows. That is what run 2
produced and it is the thing v3 exists to prevent.

### 7.5 Your authority — where you decide, and where you do not

You are running this at maximum reasoning with the whole codebase in front of
you. Prior runs failed partly because the instructions were wrong in places the
agent could see but did not feel licensed to change. That licence is now
explicit.

**You decide, without asking, and record the decision in the iteration log:**

- Which matching strategy from §5.3 (A alone, or A+B), and the budget reasoning.
- Whether `active_sample_fraction` is reported-only or gated — declared *before*
  the run that uses it.
- How §5.6 resolves and whether `dnb08` stays in the primary comparison.
- Realization counts above the declared minimums, grid ordering, batching,
  checkpointing, pod class and region within §4.1's ceiling.
- Any additional diagnostic, figure, or ablation you judge informative. More
  evidence is always welcome; adding a diagnostic is never a deviation.
- Refactoring, restructuring or rewriting any analysis code in
  `labeled_line_campaign.py`. It is not sacred. If a function is wrong, fix it and
  say what you fixed.
- Stopping a run early when you have found a bug. Finding a bug at hour two is a
  good outcome, not a failure.
- **Telling me an instruction in this file is wrong.** Every previous run was
  damaged by an error of mine that a careful reader could have caught. §3.4's API
  block, §3.1's cell tables, and §0c's bug diagnosis were all verified against the
  code, but verification is not proof. If something here contradicts what you read
  in the source, **the source wins** — implement what is correct, and write the
  contradiction into the log so the next version fixes it.

**You do not decide, and must stop and report instead:**

- Anything that unfreezes weights, trains, fits, or tunes a parameter against an
  outcome. `core_parameters: 0` is the moat.
- Widening the `0.8–1.25` activity tolerance, or loosening
  `PHASE_TOLERANCE_CYCLES`, `concentration >= 0.3`, or the S1 threshold of 12/20.
- Changing `BANDPASS_HZ` from `(1.5, 4.0)` after seeing results. §0c justifies the
  band on run-2 evidence; retuning it on run-3 outcomes is p-hacking.
- Reducing `SIGNAL_SECS` below `30.0`. This killed run 1.
- Dropping a declared input set, rate, or arm for any reason other than a written
  §5.6-style justification.
- Publishing a headline claim. `OWNER_APPROVAL_REQUIRED` on every final claim,
  positive or negative.

**When genuinely torn:** run both, report both, and say which you would trust.
Compute is cheaper than another blocked run.

### 7.6 Completion ledger — the definition of 100%

Tick every row or state explicitly why it cannot be ticked. This table goes in
the final artifact.

| # | item | done when |
|---|---|---|
| 1 | Strategy A bug fix applied | `strict_activity_match` rate-scales unconditionally; unit check that a deficit row still gets `scale != None` |
| 2 | §5.6 bimodality investigated | `dnb08` verdict written in the log **before** the grid run |
| 3 | Window assertions pass | `cycles_low >= 10`, `bins_in_band >= 8` recorded |
| 4 | Spectrum diagnostic carried forward | run-2 `spectrum_iteration1.json` cited or re-run |
| 5 | Full grid executed | every (input set × rate × arm × seed) has a terminal state per S8 |
| 6 | Activity match verified post-hoc | ratios recorded for every condition, not just the primary |
| 7 | §5.1(a) rate table | all arms, with the Strategy-A scale-invariance caveat attached |
| 8 | §5.1(b) antagonist phase | all 20 cells, each with phase, concentration, SD, in-band power |
| 9 | §5.1(c) inter-leg phase | reported |
| 10 | §5.1(d) selectivity matrix | all declared pairs, all arms |
| 11 | Silencing ladder (§5.4) | 3 rhythm types + top-15 coordination types, real arm |
| 12 | Silencing under rewire (§5.4) | the same ranking on a rewired graph — the test [SH] could not run |
| 13 | Ablation ladder (§8.1) | A/B/C/D for rate+selectivity; A/C/rewired/random for phase, with the degeneracy stated |
| 14 | S1–S8 evaluated | each with its number, `true`/`false`, and the margin against SD |
| 15 | Freeze audit | hashes equal pre/post, `parameter_updates: 0` |
| 16 | Body trace (§5.5) | rendered, labelled descriptive-only |
| 17 | Self-interrogation (§8.3) | every question answered |
| 18 | Three distinct artifacts | three different MD5s (§0b Bug 4) |
| 19 | Verdict written | one of §7.4's three outcomes, with `OWNER_APPROVAL_REQUIRED` |
| 20 | Pod terminated | `runpodctl pod list` empty, spend reported |

Rows 11–12 are the ones most likely to get squeezed by time. They are also the
strongest test in the design. If the budget forces a cut, cut realization counts
before you cut row 12.

---

## 8. Anti-cheat

### 8.1 The ablation ladder — mandatory on every reported arm

| # | arm | what it does | what it proves |
|---|---|---|---|
| A | `full` | Real connectome, named DN drive | the claim |
| B | `constant-connectome` | Replace the connectome's output with its own temporal mean | **if B ≈ A, the connectome's timing contributes nothing** |
| C | `time-shuffled` | Permute the history along the time axis | separates temporal structure from amplitude |
| D | `drive-only` | No connectome; Poisson straight to the readout | the floor |

**A-vs-B is the decisive test for the RATE and SELECTIVITY readouts** (§5.1(a),
§5.1(d)). The previous loop's V2 lost to its own constant ablation by `0.0031 m`.
Report the A−B difference explicitly for every arm.

**But the ladder is degenerate for the PHASE readouts** (§5.1(b), §5.1(c)), and you
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
 8e. MATCHER (v3, 0c): strict_activity_match returns scale != None on EVERY
     realization, including rows with unmatchable_channels > 0. Assert this
     on a synthetic case where the source has fewer active channels than the
     target. This is the exact branch that blocked run 2.
 8f. COVERAGE (S8): the set of (input_set, rate_hz, arm, graph_seed) keys in
     the artifact equals the declared grid exactly. Every key carries one of
     'evaluated' / 'excluded-with-reason' / 'unmatchable-after-strategy-A'.
     assert not any(state is None for state in grid_states.values())
     Run 2 wrote empty readout rows; this catches that at write time.
 8g. NO OUTCOME-TUNING: BANDPASS_HZ == (1.5, 4.0), SIGNAL_SECS == 30.0,
     PHASE_TOLERANCE_CYCLES == 0.15, concentration floor == 0.3, and the
     activity band == (0.8, 1.25). Assert the literal values at launch so a
     mid-run edit cannot pass silently.
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

**Run 1 of this loop is superseded (§0b).** Its numbers —
`4.33 ± 1.37` of 20 antagonist cells, selectivity distance real `0.09697` /
rewired `0.01620` / random `0.20705`, band share real `0.1489` — were produced in
a 0.5 s analysis window with unmatched controls. Do not cite them, do not treat
them as a baseline to beat, and do not report the loop as "already tried".

**Run 2 produced no comparison rows (§0c),** so it has no result to supersede or
cite. What it does leave you is genuine and reusable:

| quantity | value | status |
|---|---|---|
| real per-leg in-band power share | `0.1217` | **usable diagnostic** |
| rewired per-leg in-band power share | `0.0591` | **usable diagnostic** |
| real per-leg peak Hz | `1.33, 4.80, 5.00, 19.13, 23.33, 23.53` | usable |
| rewired per-leg peak Hz | `19.60–24.33` (all six) | usable |
| real `dng100` motor_hz by rate | `2.31 / 4.86 / 7.21 / 9.18` at `50/100/150/200 Hz` | monotone, usable |
| real `dnb08` motor_hz by rate | `2.14 / 5.98 / 6.74 / 5.07` | **non-monotone — see §5.6** |
| real `dnb08 @ 100 Hz` SD | `3.35` on mean `5.98` | **bimodal — see §5.6** |
| gate failure | `9.3718` vs `6.9340`, ratio `1.352` | **caused by the §0c bug** |
| freeze | hashes equal, `parameter_updates: 0` | clean |

Artifacts on disk as a record: `ll_logs/ITERATION_LOG.md` (run 1),
`ll_logs/ITERATION_LOG_v2.md` (run 2), `ll_logs/spectrum_iteration1.json` and
`.png` (run 2's diagnostic, **carry this forward**), and the `_lif_v2` JSONs
recording the stop.

Start the new log as `ll_logs/ITERATION_LOG_v3.md` and write fresh artifacts under
a `_v3` suffix. Three runs of artifacts now coexist; a wrong suffix will make the
result unciteable.

Earlier `probe_vnc_phase` E0 numbers (LIF coherence `0.980`, band `0.187`) predate
the core corrections and are **superseded**. Re-measure rather than cite them.

---

## 12. What to produce

**Artifacts** in `eon-fly-brain/results/path2/`:

- `labeled_line_selectivity_lif_v3.json` — **only** the §5.1(a) rate table and
  §5.1(d) selectivity matrix, all arms, all rate points
- `labeled_line_phase_lif_v3.json` — **only** the §5.1(b) antagonist table and
  §5.1(c) inter-leg table, **per cell**, each with concentration,
  across-realization SD, and that cell's in-band power share
- `labeled_line_silencing_lif_v3.json` — **only** the §5.4 rankings, real **and**
  rewired (ledger row 12 — do not drop the rewired ranking)
- `labeled_line_matching_lif_v3.json` — *(new in v3)* the matching audit: per
  condition, the strategy used, pre- and post-match `motor_hz` / `frac_active` /
  `active_sample_fraction`, `unmatchable_channels`, the applied scale, and (under
  Strategy B) the bisection trace. This is the file that would have made run 2's
  failure legible in one read.

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
- `ll_logs/ITERATION_LOG_v3.md` — one entry per iteration, in the previous loop's
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

Add three v3 blocks to every JSON:

```
completion_ledger { 1..20 : { done: bool, note: str } },   # section 7.6
grid_coverage     { key -> 'evaluated' | 'excluded-with-reason'
                         | 'unmatchable-after-strategy-A' },   # S8
decisions_taken   [ { decision, options_considered, chosen, why } ]  # section 7.5
```

`decisions_taken` is where your authority under §7.5 gets recorded. Every choice
you made without asking — matching strategy, the `dnb08` verdict, whether
`active_sample_fraction` was gated — goes in with its reasoning. A reader must be
able to reconstruct why the run looks the way it does without re-deriving it.

`reference_departures` is not optional. Every place you diverge from [SH]'s
method gets a file:line reference into `fly_field_refs/shiu_brain_model/`, what
you did instead, and why. The CPG loop did this well — copy its format.

**Also report, in plain prose at the top of the final log:** what you would do
next if you had another run, and what you now believe the single biggest threat to
this result is. Both previous runs died on something the instructions did not
anticipate. Your read on where the next one hides is worth more than another
table.

---

## 13. Final word

The CPG loop proved the body works and the interface works. It also proved that a
distance in metres cannot carry a connectome claim, because a random graph
produces distances just as well.

[SH] never had that problem, because they never asked a graph for a number. They
asked it for an **address**: sugar neurons in, MN9 out. The address is the claim.

We have 260 named motor neurons across 14 identified types, arranged in
antagonist pairs across six legs, driven by named descending commands. That is an
address space. Nobody has asked the VNC connectome to route through it in a
non-fly body with frozen measured weights.

Two runs have now tried and neither returned an answer. Run 1 measured phase in a
window shorter than one cycle. Run 2 fixed that, found something genuinely
interesting in the spectrum, and then let four unmatchable channels veto sixteen
grid conditions. Both failures were failures of instruction, not of the
connectome — and both were, in hindsight, visible in the code before the pod ever
started.

So the standing request for this run is narrower than "get a positive result",
because nobody can order that. It is: **come back with an answer.** Run every
condition. Match every control honestly, or show that it cannot be done and prove
that with a correct matcher rather than a broken one. Fill the ledger in §7.6.
Then say which of §7.4's three outcomes happened.

A clean null with all twenty ledger rows ticked is a complete result and a real
contribution. A stop at the gate is not. The difference is the whole point of
this version.

If you find an error in this file — and the base rate says there is one — fix the
work and write the error down. That is how the last two runs should have ended.
