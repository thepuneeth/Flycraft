# Codex task: get the hexapod walking through a connectome-driven CPG

You are working in `/Users/puneeth/Flycraft` (laptop) and `/root/eon-fly-brain` (GPU pod).
Read this whole file before running anything. It is a **loop**, not a checklist: you
iterate until a stopping condition fires, and the stopping conditions include failure.

---

## 0. The one-paragraph version

A frozen, untrained *Drosophila* VNC connectome (11,000 neurons, 485,074 signed
edges, measured synapse counts) currently drives an 18-DoF PhantomX-class hexapod
in MuJoCo by mapping motor-neuron firing rates straight onto joint angles. It does
not walk: best arm is 0.0672 m of forward travel in 8 s while covering 0.7645 m of
path — an 11.4:1 path-to-forward ratio. A hand-written CPG on the *same body, same
camera, same episode* gets 2.3944 m at a 1.1:1 ratio. The body is fine. The direct
mapping is what fails.

Your job is to replace the direct mapping with the locomotion interface from
Wang-Chen et al. (NeuroMechFly v2), translated as closely as the reference code
allows, so the connectome modulates a CPG instead of driving joints. **The
connectome stays frozen. Nothing trains. Ever.**

---

## 1. Hard constraints — violating any of these invalidates the entire result

These are not style preferences. They are the reason the project has a claim at all.

1. **VNC core weights are FROZEN.** They are measured synapse counts from the
   MANC/FlyWire connectome. No gradient ever touches them. Every artifact must
   carry `zero_learning: true`, `parameter_updates: 0`, `training_markers: []`,
   and pre/post SHA-256 hashes of the weight file that compare equal.
   The real weight tensor hash is `128b5ceddec4d8e14f3a04fbb66a08a2e83564c674624741eb03fb2bf01517fc`.
   The graph files hash to:
   - `all_legs_weight_csr.pkl` → `018a4890e01f3dd608693e8a040103829e8523d134290ae918e6e5dd4d815506`
   - `all_legs_io_index.npz` → `3f65f5b3ebed0d63161dca416bffa1a55afc0be4906113103426b16492d09449`
   If any of these change, **stop and report** — you have corrupted the substrate.

2. **No training, of anything, anywhere.** Not the core, not the interface, not a
   readout head, not a "small" calibration network. If you find yourself writing
   an optimiser, a loss, or a `.backward()`, you have left the task. Every constant
   in the new interface is either (a) taken from the reference implementation,
   (b) derived from anatomy, or (c) swept over a small declared grid and reported
   in full — never fitted.

3. **Every constant is declared and justified in a comment naming its source.**
   `[NMF]` for Wang-Chen, `[SH]` for Shiu. A number with no provenance is a bug.

4. **The CPG is a hand-written prior and must be labelled as one, always.**
   It supplies rhythm generation and inter-leg coordination — the two things that
   *are* locomotion. The honest claim ceiling for this whole task is
   **"the connectome modulates a biologically-derived gait"**, never "the
   connectome controls a hexapod". Write it that way in every artifact, every
   figure caption, every summary you produce. If you catch yourself writing the
   stronger claim, that is the failure mode this project exists to avoid.

5. **Do not commit to git.** Leave changes in the working tree. The owner commits.

5a. **No pod above $0.20/hr, ever.** Always pass `--cost 0.20`. See §4.1. If nothing
   is available at that price, wait or report blocked — do not spend more.

6. **An honest null is a deliverable.** If the specificity test in §7 comes back
   null — real connectome indistinguishable from rewired — that is a publishable
   result and you report it as such. Do not tune until it passes. Tuning until it
   passes is how you manufacture a false positive, and §7 is specifically designed
   to catch that.

---

## 2. Success criteria and kill criteria — read before you start, decide nothing later

You are done when **either** of these fires. Check them at the end of every loop.

### SUCCESS (report and stop)
- `forward_m >= 0.5` over 8 s, averaged over ≥ 20 episodes, **and**
- `path_per_forward <= 2.5` (motion is directed, not thrashing), **and**
- the real connectome beats **both** control arms (degree-preserving rewire,
  weight randomisation) on `forward_m` by more than the real arm's own
  realization SD, with n ≥ 12 control seeds, **and**
- **the full arm beats the constant-connectome ablation** (§8.1 row B) — without
  this, the connectome's timing contributes nothing and the result is a CPG demo,
  **and**
- all freeze checks in §1.1 pass.

Hitting the first two but not the third is **not** success. It is "the CPG walks",
which we already knew. Report it as such.

### KILL (report and stop)
- 6 full loop iterations with no improvement in `forward_m` over the best so far.
- Or: walking achieved but the specificity test is null across all three readout
  variants in §6, with n ≥ 12 seeds. → Report `CPG-INTERFACE-SPECIFICITY-NULL`.
  This is a legitimate scientific outcome. Say so plainly.
- Or: any freeze check fails and you cannot restore the substrate from git.
- Or: total pod spend exceeds **$15** (see §4.6). At the $0.20/hr cap that is
  75 GPU-hours, which is ample for this task.

### Do NOT
- Do not add sensory feedback / closed-loop terrain adaptation. Out of scope, and
  it is the *next* project. Open loop only.
- Do not tune `MAP_GAINS` or the existing direct map. That arm is recorded and finished.
- Do not touch anything under `results/path2/clobbered_unsuffixed/` — that is a
  quarantined record of a prior overwrite bug.

---

## 3. What exists already (read these files before writing any code)

### 3.1 Our code — `/Users/puneeth/Flycraft/eon-fly-brain/code/`

| file | what it is | your relationship to it |
|---|---|---|
| `vnc_lif_core.py` | Frozen [SH] LIF connectome core. Verified line-by-line against `philshiu/Drosophila_brain_model/model.py`. Has a `demo()` self-check that asserts the Poisson drive returns the rate it was asked for. | **Do not modify.** Read the module docstring — it explains three departures that were found and fixed, and why. |
| `nmf_cpg.py` | Our port of the [NMF] CPG. `NMFCPGNetwork` (oscillators) + `NMFCPGGait` (joint mapping). Reference-verified clean. | **Extend, do not rewrite.** This is your starting point. |
| `specificity_campaign.py` | The harness. `base_vnc()`, `target_history()`, `run_body_trace()`, `episode_specs()`, `EPISODE_SEED`, `SIGNAL_SAMPLE_DT`, `make_body_model()`. | Use it. Do not fork it. |
| `hexapod_gait.py` | Body constants: `resolve_stance()`, `LEFT = {0,2,4}`, joint layout. | Read `resolve_stance` — it defines the 6×3 stance pose you modulate around. |
| `audit_vnc_legs.py` | `leg_scores(history, io) -> (T, 6, 5)`. Groups motor neurons into 6 legs × 5 joints using `io["motor_leg"]`, `io["motor_joint"]`, `io["motor_sign"]`. | **This is your connectome→leg bridge.** It already exists. Use it. |
| `probe_vnc_phase.py` | `phase_metrics()` → `tripod_coherence_scalar`; `band_power_share()`. | Your phase diagnostics. |
| `reservoir_3way.py` | `degree_preserving_rewire(counts, seed)` (preserves row AND column degree plus weight multiset), `randomize_weights(counts, seed)`. | **Your control arms. Do not invent new ones.** |
| `build_dn_command_map.py` | Fixed, anatomy-derived 4→1325 descending command map. Hashed, never trained. Channels: `go`, `steer`, `family`, `walkdn`. | Read its docstring. It contains the argument for why the interface must not be learned. |
| `walk_gain_envelope.py` | Sweeps one scalar readout gain identically over both cores. | Your template for how to sweep honestly. |
| `render_best_walk.py` | Renders arms + the CPG ceiling on the same body/camera/episode. | Your video tool. |

### 3.2 Reference material — `/Users/puneeth/Flycraft/fly_field_refs/`

| path | what |
|---|---|
| `flygym/src/flygym_demo/complex_terrain/cpg_controller.py` | **The reference CPG.** 186 lines. `CPGNetwork`, `get_cpg_biases()` (tripod/tetrapod/wave), `make_tripod_cpg_network()`. |
| `flygym/src/flygym_demo/complex_terrain/preprogrammed.py` | **`PreprogrammedSteps` — the piece we never ported.** Real recorded fly leg kinematics, cubic-spline interpolated over phase. |
| `flygym/src/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl` | **The recorded step data itself. It is on disk. This is the most valuable unexploited asset in the repo.** |
| `flygym/tutorials/4a_cpg_controller.ipynb` | Tutorial walkthrough of the same controller. Good for intent. |
| `nmf2-paper/control_signal/cpg.ipynb` | Paper figure code for the CPG. |
| `nmf2-paper/controller_comparison/` | How the paper compares CPG vs rule-based vs hybrid. Read `README.md`. |
| `shiu_brain_model/` | The [SH] reference. Already fully exploited; consult only if a core question arises. |

---

## 4. Pod operations — read fully before creating anything

**Never run Flycraft experiment code on the laptop.** MuJoCo + the 485k-edge sparse
core will take hours on CPU and the numbers are not reproducible across machines
(a recorded example: identical config gave 0.2241 on the laptop and 0.125 on the pod).
Edit locally, run on the pod, always.

### 4.1 Which pod — HARD COST CEILING: $0.20/hr

**You must not create a pod costing more than $0.20/hr. This is a requirement, not
a preference.** Always pass `--cost 0.20`, which makes RunPod refuse rather than
silently give you a pricier box. Do not "temporarily" take a faster GPU.

Live community-cloud pricing at time of writing, options at or under the cap:

| GPU | $/hr | sys RAM | vCPU | verdict |
|---|---|---|---|---|
| **RTX 4000 Ada** | **0.200** | **50 GB** | 9 | **First choice.** Only option at the cap with real RAM headroom. |
| RTX A5000 | 0.160 | 34 GB | 9 | **Fallback.** Cheapest sane box. |
| RTX A4500 | 0.190 | 31 GB | 8 | Works. |
| RTX 4070 Ti | 0.190 | 24 GB | 12 | Tight on RAM. |
| RTX A4000 | 0.170 | 31 GB | 5 | Only 5 vCPU — cuts parallelism. |
| RTX 3080 Ti | 0.180 | 30 GB | 12 | Works. |
| RTX 3070 | 0.130 | 17 GB | 8 | **Do not use.** Will OOM. |

For reference and to keep expectations straight: the previously validated box was a
**4090 at $0.34/hr**, which is over the cap and is no longer available to you.

**The RAM tradeoff you must respect.** The LIF core once OOM-killed a **62 GB** pod
because per-thread buffers scaled with thread count. Every option under $0.20/hr has
*less* RAM than that box. So:

- `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` are **mandatory**, not tuning knobs.
- Run **at most 3 experiment processes in parallel**, not the 9 the old campaign
  script launched on the 4090. On a 30 GB box, 9 parallel jobs will OOM.
- If you see a job vanish with no traceback, check `dmesg | tail` for the OOM
  killer before assuming a code bug. A silent disappearance is almost always RAM.

**Expect roughly 1.5–2× the wall-clock of the 4090** on an A5000/4000 Ada for the
sparse core. At half the hourly rate that is still net cheaper per experiment, so
the cap costs you time, not money. Plan iterations accordingly — prefer fewer,
better-designed runs over many exploratory ones.

Do **not** take an A100/H100/L40. Aside from blowing the cap 10×, the workload is
latency-bound on small sparse matmuls, not throughput-bound; you would pay far more
for no speedup.

### 4.2 Create it

```bash
runpodctl create pod \
  --name flycraft-cpg \
  --gpuType 'NVIDIA RTX 4000 Ada Generation' \
  --gpuCount 1 \
  --imageName 'runpod/pytorch:1.1.0-cu1281-torch271-ubuntu2204' \
  --containerDiskSize 30 \
  --volumeSize 30 \
  --mem 30 \
  --vcpu 8 \
  --communityCloud \
  --startSSH \
  --ports '22/tcp' \
  --cost 0.20
```

If that GPU type is unavailable, fall back to `'NVIDIA RTX A5000'` ($0.160/hr).
Check live availability and prices first — they move:

```bash
runpodctl get cloud -c          # community cloud listings with $/hr
```

**Never raise `--cost` above 0.20.** If nothing is available at or under the cap,
wait and retry, or report that you are blocked on capacity. Do not solve a capacity
problem by spending more.

Then get connection details:

```bash
runpodctl pod list                    # confirm RUNNING
runpodctl get pod <POD_ID> -o json    # read the ssh host/port out of the ports field
```

SSH key is at `/Users/puneeth/.runpod/ssh/runpodctl-ssh-key`. Connection form:

```bash
KEY=/Users/puneeth/.runpod/ssh/runpodctl-ssh-key
ssh -i $KEY -p <PORT> -o StrictHostKeyChecking=no -o BatchMode=yes \
    -o ServerAliveInterval=15 root@<HOST>
```

Set `ServerAliveInterval` on every long connection. Without it, long-running
watchers die with `client_loop: send disconnect: Broken pipe` and exit 255 — the
**job keeps running**, only your watcher died. Reconnect and check state; do not
assume the job failed.

### 4.3 Deploy code and data up

```bash
KEY=/Users/puneeth/.runpod/ssh/runpodctl-ssh-key
SSH="ssh -i $KEY -p <PORT> -o StrictHostKeyChecking=no -o BatchMode=yes"

# code
rsync -av --exclude '__pycache__' --exclude '._*' -e "$SSH" \
  /Users/puneeth/Flycraft/eon-fly-brain/code/ root@<HOST>:/root/eon-fly-brain/code/

# graph + annotations (large; only needed once)
rsync -av -e "$SSH" \
  /Users/puneeth/Flycraft/eon-fly-brain/results/path2/all_legs_*.{pkl,npz} \
  root@<HOST>:/root/eon-fly-brain/results/path2/
rsync -av -e "$SSH" \
  /Users/puneeth/Flycraft/eon-fly-brain/data/manc/ \
  root@<HOST>:/root/eon-fly-brain/data/manc/

# the recorded fly steps — you WILL need this (§5.1)
$SSH root@<HOST> 'mkdir -p /root/eon-fly-brain/data/nmf'
rsync -av -e "$SSH" \
  /Users/puneeth/Flycraft/fly_field_refs/flygym/src/flygym_demo/complex_terrain/assets/single_steps_untethered.pkl \
  root@<HOST>:/root/eon-fly-brain/data/nmf/
```

macOS writes `._*` AppleDouble junk files. Always exclude them; a previous deploy
shipped a dozen of them to the pod.

### 4.4 Environment on the pod

```bash
pip install -q --only-binary=:all: mujoco imageio imageio-ffmpeg scipy pandas pyarrow matplotlib
apt-get update -qq && apt-get install -y -qq libegl1 libglvnd0 libgles2

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2     # MANDATORY, not a knob — see the OOM note in §4.1
export MKL_NUM_THREADS=2
```

`scipy` is required — `PreprogrammedSteps` uses `CubicSpline`.

### 4.5 Launch, monitor, survive disconnects

**Always launch detached.** Never leave a job attached to your ssh session.

```bash
$SSH root@<HOST> 'cd /root/eon-fly-brain/code && \
  setsid nohup python3 -u cpg_connectome_walk.py --out ../results/path2/cpg_walk.json \
  > ../results/path2/cpg_logs/run.log 2>&1 < /dev/null & echo $!'
```

Monitor by polling, in separate short-lived connections:

```bash
$SSH root@<HOST> 'pgrep -af cpg_connectome_walk; tail -20 /root/eon-fly-brain/results/path2/cpg_logs/run.log'
$SSH root@<HOST> 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

A healthy run shows **high GPU utilisation** (90%+ on a saturated box; somewhat
lower on the smaller cards is fine). If you see **0% / 1 MiB**, the job is on CPU —
check that `--device cuda` reached the core. That exact bug wasted hours previously.

Also watch RAM, which is the binding constraint on a sub-$0.20 box:

```bash
$SSH root@<HOST> 'free -g; dmesg | tail -5'
```

A job that vanishes with no traceback was almost certainly OOM-killed. Reduce
parallelism before suspecting your code.

**Killing jobs — read carefully.** `pkill -f "some_script.py"` will match *your own
ssh command line* and kill your session (observed: exit 255). Always kill by PID:

```bash
$SSH root@<HOST> 'pgrep -f cpg_connectome_walk'   # get PID
$SSH root@<HOST> 'kill <PID>'
```

Write a `DONE` sentinel file at the end of every batch script and poll for it.
Poll every ~5 minutes, not every 10 seconds.

### 4.6 Pull results, then terminate — in that order, verified

**Terminating a pod is irreversible and destroys everything on it.** The last pod
was deleted with a frequency-sweep result on it that had never been written to an
artifact; that data is gone permanently. Do not repeat this.

Before terminating, run all four checks:

```bash
# 1. Nothing still running
$SSH root@<HOST> 'pgrep -af "python3|ffmpeg" || echo NONE_RUNNING'

# 2. Pull every result
rsync -av -e "$SSH" root@<HOST>:/root/eon-fly-brain/results/path2/ \
  /Users/puneeth/Flycraft/eon-fly-brain/results/path2/

# 3. Confirm pod-side code has no edits you'd lose (checksum both sides, compare)
#    Do NOT blind-rsync pod code down — it can clobber local edits.

# 4. Confirm the artifacts you care about are actually local and non-empty
ls -la /Users/puneeth/Flycraft/eon-fly-brain/results/path2/cpg_*
```

Only then:

```bash
runpodctl pod delete <POD_ID>
runpodctl pod list        # must show [] or no longer list your pod
```

**Terminate as soon as a loop iteration's results are pulled.** Do not leave a pod
idle between iterations "in case" — at $0.20/hr an idle overnight is ~$2.40 of a $15
budget for nothing. Creating a fresh pod costs ~5 minutes.

Track cumulative spend and log it in `ITERATION_LOG.md` each iteration. At **$15
total**, stop and report regardless of progress.

---

## 5. The actual work — translation, ranked by expected value

The current `nmf_cpg.py` ports the oscillator network faithfully but substitutes a
**sine/lift heuristic** for the reference's joint mapping. That substitution is
where the fidelity was lost. Fix it in this order.

### 5.1 [HIGHEST VALUE] Port `PreprogrammedSteps` — use the real recorded fly steps

**This is the single biggest gap and you should do it first.**

Reference: `preprogrammed.py:40-120`. What it does:
- Loads `single_steps_untethered.pkl` — real *Drosophila* leg kinematics recorded
  from walking flies.
- Builds a `CubicSpline` per leg over a phase grid `np.linspace(0, 2π, length)`
  with `bc_type="periodic"`, mapping **phase → joint angles**.
- Right legs get roll/yaw sign-flipped (`preprogrammed.py:88-92`).
- Exposes `get_joint_angles(leg, phase, magnitude)` — offset from neutral, scaled
  by the oscillator's magnitude.
- Also has `get_adhesion_onoff()`. **Ignore adhesion** — the hexapod has no
  adhesion actuators. Note the omission explicitly in your docstring.

Ours (`nmf_cpg.py:126-135`, `NMFCPGGait.targets`) instead does:
```python
coxa  = stance[:,0] - side * swing_amp * gain * sin_phase
femur = stance[:,1] + side * lift_amp * gain * max(0, sin_phase)
tibia = stance[:,2]                      # tibia never moves at all
```
A sine and a half-wave rectifier. The real fly step is not a sine, and our tibia
is frozen.

**Your job:** load the real step data, spline it over phase exactly as the
reference does, and **retarget** it from fly leg geometry to the hexapod's 3-DoF
coxa/femur/tibia. Retargeting is the genuinely hard part and it must be
*declared*, not fitted:
- The fly has more DoFs per leg than the hexapod's 3. Pick the mapping by joint
  *role* (protraction/retraction → coxa; elevation/depression → femur;
  extension/flexion → tibia), document the choice, and **do not tune it against
  forward distance**.
- Normalise amplitude by the hexapod's joint range, taken from the MuJoCo model,
  not by hand-picked constants.
- Preserve the phase-to-swing relationship: `swing_period` in the reference tells
  you which phase interval is swing vs stance. Getting stance/swing backwards
  makes the robot walk backwards — and note that several of our recorded arms
  have *negative* forward distance, so this failure mode is live.

Write this as a new class in `nmf_cpg.py`, e.g. `PreprogrammedHexapodSteps`,
alongside the existing `NMFCPGGait`. Keep the old one — it is the recorded baseline.

### 5.2 Fix the frequency scaling and declare it

Reference default: `intrinsic_frequency = 12.0` Hz (`cpg_controller.py:170`).
Ours: `1.2` Hz, chosen as the best of a sweep whose artifact was lost with the
pod (only the chosen value survives, in a comment at `render_best_walk.py:84`).

12 Hz is a fly stepping at fly scale. The hexapod is orders of magnitude larger
and heavier; Froude scaling says its natural stride frequency is much lower, so
1.2 Hz is *physically reasonable* — but it is currently undocumented and
unreproducible.

**Do:** re-run the frequency sweep, write it to a real artifact this time
(`cpg_frequency_sweep.json`), and include the Froude-scaling justification in the
docstring. Sweep `(0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)` Hz, ≥ 8 episodes each.
Report the whole curve, plus the neighbours of the best point — a peak whose
neighbours are far worse is a knife edge, not an optimum, and must be reported
as such. `walk_gain_envelope.py` already does exactly this; copy its structure.

### 5.3 [THE CORE OF THE TASK] Connectome → CPG modulation

Now wire the frozen connectome into the CPG. Currently the CPG runs open-loop on
constants and the connectome is not involved at all.

The bridge already exists: `leg_scores(history, io)` returns `(T, 6, 5)` —
per-timestep, per-leg, per-joint signed activity from the frozen core. Six legs;
six oscillators. The correspondence is one-to-one and anatomical, not chosen.

Implement **three variants** and evaluate all three. Do not pick one early.

- **V1 — amplitude modulation.** Leg *i*'s motor activity sets oscillator *i*'s
  `intrinsic_amps[i]`. Connectome controls *how big* each leg's step is; the CPG
  keeps all timing. Weakest coupling, most likely to walk, least likely to be
  specific.
- **V2 — frequency modulation.** Leg *i*'s activity sets `intrinsic_freqs[i]`.
  Connectome controls *how fast* each leg steps; coupling still enforces the
  tripod. Stronger claim.
- **V3 — phase-bias modulation.** The connectome's measured inter-leg phase
  relationships set the `phase_biases` matrix, replacing the hand-written tripod
  from `get_cpg_biases("tripod")`. **This is the strongest possible claim** — the
  inter-leg coordination itself would come from the connectome rather than from
  us — and it is the variant most likely to fail outright. Do it anyway. If V3
  works, that is the result worth publishing.

For every variant, the mapping from `leg_scores` to CPG parameter must be an
**affine transform with declared constants**: normalise by the core's own rate
ceiling (`core.rate_ceiling_hz`, 250 Hz at dt=2 ms), then scale into the
parameter's physical range. Sweep the scale over a small declared grid. Never fit it.

Note the honest tension, and put it in the docstring: V1 is most likely to walk
and least likely to be specific; V3 is the reverse. Report all three regardless
of which "wins" — reporting only the flattering variant is the exact failure this
project's control design exists to prevent.

### 5.4 Keep the existing harness

Do not fork `specificity_campaign.py`. Your new controller produces an
`(T, 18)` array of joint targets, which is exactly what `run_body_trace(targets,
io, spec, secs, video=...)` already consumes. Same body, same cameras, same
`episode_specs(n, EPISODE_SEED)`. Comparability with the recorded arms is the
whole point — a number measured on a different body or different episodes cannot
be placed in the existing table.

---

## 6. The loop

Run this until §2 fires. One iteration ≈ one pod lifetime.

```
ITERATION N
├─ 1. PLAN (laptop, no pod)
│     Pick ONE change. State the hypothesis and the number you expect it to move,
│     in writing, BEFORE running. Append to results/path2/cpg_logs/ITERATION_LOG.md.
│     One change per iteration. Two changes and you cannot attribute the result.
│
├─ 2. IMPLEMENT (laptop)
│     Edit code/. Add or update a demo()/__main__ self-check that fails loudly if
│     the logic breaks (assert-based, no test framework). Reference constants get
│     a [NMF] comment naming file and line.
│
├─ 3. VERIFY LOCALLY (laptop, cheap only)
│     Run the self-check on the CPG/oscillator math only — pure numpy, no MuJoCo,
│     no connectome core. Anything touching the graph or the physics goes to a pod.
│
├─ 4. POD UP (§4.2), DEPLOY (§4.3), PREFLIGHT
│     Preflight gate, in this order, before any experiment:
│       python3 -u vnc_lif_core.py       # [SH] core self-check must pass
│       python3 -u nmf_cpg.py            # CPG must lock into a tripod
│       + verify the three SHA-256 hashes in §1.1
│     If preflight fails, nothing downstream is worth GPU time. Fix first.
│     Build the §9.1 assertions in before the first real run, not after the
│     first confusing result.
│
├─ 5. RUN (§4.5)
│     Launch detached. Run at most 3 processes in parallel (§4.1 RAM limit) —
│     the three variants of §5.3 fit exactly. Poll every ~5 min.
│
├─ 6. PULL AND TERMINATE (§4.6)
│     All four checks, then delete the pod. Do not skip this to "save time" —
│     an idle pod is the most common way this budget gets burned.
│
├─ 7. EVALUATE (laptop)
│     Run the §8.1 ablation ladder on any arm you intend to report — A vs B is
│     the row that decides whether the connectome did anything at all.
│     Compare against the recorded table in §11. Did the number move the way you
│     predicted in step 1? Write the answer down, including when it did not.
│     A prediction that failed is more informative than one that succeeded.
│     If anything looks wrong or surprising, go to the §10 bug loop — do not
│     start changing constants.
│
├─ 8. CHECK §2 CRITERIA
│     SUCCESS → §7 specificity, then report.
│     KILL    → report the null. Do not keep tuning.
│     Neither → ITERATION N+1.
```

**Suggested first three iterations** (deviate if you have a better reason, and say why):
1. §5.1 `PreprogrammedSteps` port with V1 amplitude modulation. Establishes whether
   real recorded kinematics alone close the gap to the 2.3944 m ceiling.
2. §5.2 frequency sweep on the winner of iteration 1, artifact written this time.
3. §5.3 all three variants at the iteration-2 best frequency.

---

## 7. Specificity — the experiment that decides whether any of this means anything

**Do not skip this and do not run it early.** It only makes sense once something walks.

Here is the trap, stated plainly. The CPG walks on its own — 2.3944 m with no
connectome at all. So a **rewired** connectome plugged into the same CPG will also
walk, because the CPG is carrying it. If you only measure "does it walk", you will
get a beautiful video and a meaningless result: a hand-written CPG with a
connectome-shaped decoration bolted on.

The question is not *does it walk*. It is **does it walk differently, and better,
when the connectome is real.**

Protocol:
- Arms: `real`, `degree_preserving_rewire` (n ≥ 12 seeds), `randomize_weights`
  (n ≥ 12 seeds). Use `reservoir_3way.py` — these are the project's established
  controls. Do not invent new ones.
- **Amplitude-match the controls.** A control is only a control while it is as
  active as the thing it controls. `audit_amplitude_control.py` has the machinery
  (`amplitude_match(history, target_sd, allow_unmatchable=False)`). A dead control
  arm proves nothing.
- **Average over Poisson realizations.** The [SH] core is stochastic: tripod
  coherence has SD **0.3428** across 30 realizations of the *same* frozen graph
  (measured — see `lif_realization_spread.py`). At n=1 per arm, the max over N
  controls is an order statistic of noise. This exact artifact previously produced
  a spurious "1 of 200 rewires beat real" that evaporated under proper trials.
  Use ≥ 10 realizations per control arm and 30 for the real arm ([SH]
  `default_params` uses `n_run = 30`).
- **Report the real arm's own realization SD beside every comparison.** A control
  "beating" real means nothing unless it beats it by more than real moves on its own.
- Metrics: `forward_m`, `path_per_forward`, `tripod_coherence_scalar`, `band_power_share`.
- Pre-register the null **in the artifact, before running**: *"CPG-interface forward
  drive is no greater with the real connectome than with degree-preserving rewires."*

Report `CPG-INTERFACE-SPECIFICITY-EARNED` or `...-NULL`. Both are results. The null
is genuinely publishable — "connectome structure does not survive translation
through a CPG interface" is a real finding about connectome-to-robot transfer, and
it is a finding the field does not currently have.

---

## 8. Anti-cheat — how this task gets faked, and how you prove it wasn't

Read this section twice. You are being asked to "get it walking", and there is a
hand-written CPG in the loop that already walks at 2.3944 m. That means **there is
a trivially available fake success**: wire the connectome in so loosely that the
CPG does everything, then report the CPG's number as the result. It will produce a
great video. It will pass every check except the ones below.

You are not being accused of bad faith. The point is that this failure happens by
*accident*, through ordinary small decisions each of which looked reasonable, and
the only defence is a mechanical test that does not care about intent.

### 8.1 The mandatory ablation ladder — run this on EVERY reported arm

No number gets reported without these four rows beside it. Same body, same camera,
same `episode_specs(n, EPISODE_SEED)`, same everything except the named change.

| # | arm | what it does | what it proves |
|---|---|---|---|
| A | **full** | Real connectome modulating the CPG. | The claim. |
| B | **constant-connectome** | Replace the connectome output with its own **temporal mean** — same magnitude, zero dynamics. | If `forward_m(B) ≈ forward_m(A)`, the connectome's *timing* contributes nothing. You have a CPG with a constant offset. |
| C | **time-shuffled connectome** | Randomly permute the connectome history along the time axis. Identical amplitude distribution per channel, destroyed temporal structure. | Isolates temporal structure from amplitude. |
| D | **CPG-only** | No connectome at all, reference constants. | The 2.3944 m ceiling. |

**The decisive comparison is A vs B, and it is the one that kills fake results.**
If A does not beat B, stop. You do not have a connectome result, you have a CPG
with decoration, and no amount of further tuning changes that. Report it.

Write these four into the artifact as `ablation_ladder`, always, even when the
answer is embarrassing. An artifact without this block is incomplete and must not
be presented as a result.

### 8.2 The specific cheats, and the check that catches each

| cheat | how it happens innocently | detection |
|---|---|---|
| **Connectome contributes nothing** | Modulation scale ends up near zero after a sweep "found" that small values walk best. | Ablation B (§8.1). Also print the actual variance the connectome injects into each CPG parameter — if it is <1% of the parameter's range, say so out loud. |
| **Hand-fitting = training** | You sweep a constant, look at `forward_m`, pick the winner, sweep again around it, repeat. That is gradient descent performed manually. | One declared grid, swept **once**, reported **in full**. If you find yourself re-sweeping a narrowed range around a winner, you are training. Log every grid you ever ran in `ITERATION_LOG.md`, including abandoned ones. |
| **Weak controls** | Rewired arm is less active, so it walks worse for reasons unrelated to structure. | Amplitude-match (§7). Report `frac_active` and `motor_hz` for **every** arm. A control quieter than real is not a control. |
| **Realization cherry-picking** | Real arm happens to draw a good Poisson realization. | ≥10 realizations per control, 30 for real. Report the real arm's SD (it is ~0.3428 for coherence — large). This exact artifact already produced a spurious result in this project once. |
| **Seed/episode mismatch** | Arms silently run on different episodes. | Assert identical `episode_specs` across arms. Hash the spec list and put the hash in the artifact. |
| **Variant cherry-picking** | Only V1 is reported because V2/V3 failed. | §5.3 requires all three, always, in the same artifact. |
| **Leaking the objective into a "declared" constant** | The fly→hexapod retargeting is chosen by trying options and keeping the one that walks. | Retargeting is fixed by **joint role**, decided and written down **before** the first rollout. Commit the mapping to `ITERATION_LOG.md` with a timestamp before you measure anything. |
| **Moving the goalposts** | Success threshold drifts down toward whatever was achieved. | §2 thresholds are frozen. If you believe one is wrong, say so explicitly in the report and give **both** the original and proposed verdict. Never silently substitute. |
| **Comparability drift** | Body, camera, episode length, or control rate changes, making new numbers incomparable to §11. | Use `run_body_trace` unmodified. Assert `secs`, `CTRL_HZ`, perturbation `("leg_length", 1.0)` match §11. |
| **Substrate corruption** | An in-place tensor op mutates the frozen weights. | Hash pre and post, every run (§1.1). |

### 8.3 Self-interrogation — answer these in writing before reporting success

Put the answers in the artifact. If you cannot answer one, you are not done.

1. What is `forward_m` for ablation B, and how much does A beat it by?
2. What fraction of each CPG parameter's range does the connectome actually move?
3. Did any constant get chosen by looking at `forward_m`? Which, and when?
4. Are all arms on identical episodes? What is the spec hash?
5. Do the weight hashes match §1.1 pre and post?
6. How many realizations per arm, and what is the real arm's SD?
7. If a reviewer replaced the real connectome with a rewire and reran, what number would they get?
8. Is there any sentence in the write-up that a reader could take as "the connectome generated the gait"?

---

## 9. Debugging playbook — symptoms you will actually hit

These are ranked by how likely they are. Diagnose in this order; do not guess.

| symptom | most likely cause | how to confirm | fix |
|---|---|---|---|
| **Robot walks backwards** (`forward_m` negative) | Swing/stance phase inverted, or coxa sign flipped. Several recorded arms are already negative, so this is live. | Plot one leg's coxa target vs phase. Foot should move *backward* relative to body during stance (foot planted), forward during swing. | Flip the swing interval, not the sign of the whole signal — flipping the sign also breaks left/right. |
| **Job vanishes, no traceback** | OOM killer. Binding constraint on a sub-$0.20 box. | `dmesg \| tail -5`, `free -g` | Reduce parallel processes (max 3), confirm `OMP_NUM_THREADS=2`. |
| **GPU at 0% / 1 MiB** | `--device cuda` never reached the core. | `nvidia-smi`; print `next(core.parameters()).device` or the buffer device at startup. | Thread the device through. This wasted hours previously. |
| **Robot vibrates / falls immediately** | Targets outside joint limits, or discontinuity at phase wrap. | Print `min`/`max` of targets per joint; compare to MuJoCo `jnt_range`. Check target continuity at phase 0/2π. | Clamp to `jnt_range`; ensure spline uses `bc_type="periodic"`. |
| **No movement at all** (`forward_m ≈ 0`, `path_m ≈ 0`) | Oscillator magnitudes never converge off zero, or targets equal stance every step. | Print `curr_magnitudes` over time — should converge to `intrinsic_amps`. Print `targets - stance` norm. | `convergence_coefs` too low, or `intrinsic_amps` driven to 0 by the connectome mapping. |
| **NaN in targets** | Oscillator blowup from too-large `intrinsic_freqs`, or Euler instability. | `np.isfinite(targets).all()` assert at every step. | Reduce timestep or frequency; the reference uses Euler and is only stable in a range. |
| **Tripod never forms** (coherence ≈ 0) | `phase_biases` wrong shape/values, or `coupling_weights` zero. | `nmf_cpg.demo()` — it already asserts tripod lock. Print the 6×6 bias matrix. | Compare against `get_cpg_biases("tripod")` element by element. |
| **Result changes run to run** | Poisson realization noise — expected, not a bug. Coherence SD is 0.3428. | Run same seed twice; if identical, it is realization noise, not nondeterminism. | Average over realizations (§7). Never compare single draws. |
| **Numbers differ from laptop** | Different torch build. Recorded example: 0.2241 laptop vs 0.125 pod, identical config. | — | Never mix machines in one comparison. All arms on one pod, one run. |
| **`'ShiuLIFStepper' object has no attribute 'W'`** or similar | Core API drift between rate and LIF paths. | — | `vnc_lif_core.py` has a `W` property (counts in, counts out). Read it before working around it. |
| **Watcher dies, exit 255** | ssh timeout. **Job is still running.** | Reconnect, `pgrep -af` | `ServerAliveInterval=15`. Never assume the job died. |
| **`pkill` killed your session** | Pattern matched your own ssh command line. | — | Kill by PID only (§4.5). |

### 9.1 Assertions to build in from the start

Cheaper than debugging. Put these in the code, not in your head:

```python
assert np.isfinite(targets).all(), "non-finite joint targets"
assert targets.shape == (steps, 18), f"bad target shape {targets.shape}"
assert (targets >= jnt_low).all() and (targets <= jnt_high).all(), "targets out of joint range"
assert abs(coherence_after_lock - 1.0) < 0.3, "CPG did not lock into tripod"
assert weight_hash_pre == weight_hash_post, "FROZEN WEIGHTS MUTATED"
```

The last one is not optional.

---

## 10. Bug-hunt loop — what to do when something is wrong

When a run gives a wrong or surprising number, **do not** start changing things.
Run this loop instead. It is short on purpose.

```
BUG LOOP
├─ 1. REPRODUCE
│     Same seed, same command, twice. Identical output?
│       identical  → deterministic bug, continue
│       different  → realization noise (SD 0.3428 is large), or true
│                    nondeterminism. Average before concluding anything.
│
├─ 2. LOCALISE — bisect the pipeline, do not read code hoping to spot it
│     The pipeline is:  connectome → leg_scores → CPG params → oscillator
│                       → joint targets → MuJoCo → forward_m
│     Check each boundary in order with a print or an assert:
│       a. Is the connectome history finite, non-constant, right shape?
│       b. Does leg_scores give 6 legs with real variance? (not all-NaN)
│       c. Are the CPG params in their physical range?
│       d. Do oscillator phases advance and magnitudes converge?
│       e. Are joint targets finite and inside jnt_range?
│       f. Does MuJoCo report the body actually moving?
│     The first boundary that looks wrong is where the bug is. Stop there.
│
├─ 3. MINIMISE
│     Shrink to the smallest failing case: 1 leg, 1 second, 1 episode, no video.
│     A bug that needs an 8-second 20-episode run to see is a bug you cannot
│     iterate on.
│
├─ 4. COMPARE TO REFERENCE
│     Open cpg_controller.py / preprogrammed.py at the corresponding line.
│     Ours vs theirs, side by side. A previous audit of this project found
│     THREE silent departures this way, and every number measured through them
│     was stale. This step finds real bugs at a high rate — do not skip it.
│
├─ 5. FIX, THEN RE-ASSERT
│     Add an assertion that would have caught it. If you cannot write that
│     assertion, you have not understood the bug yet — go back to step 2.
│
└─ 6. RECORD
      One line in ITERATION_LOG.md: symptom, real cause, the assertion added.
      A bug fixed without a recorded cause comes back.
```

**Time limit.** If a single bug eats more than one full pod iteration, stop, write
down exactly what you know and what you ruled out, and move to a different part of
the task. Report it as blocked rather than grinding. Grinding on one bug is the
most common way an agent burns a budget with nothing to show.

---

## 11. The recorded numbers you are trying to beat

All on the same body, same camera, 8 s, episode 0. Do not re-measure these; they
are recorded. Place your new numbers in this table.

| arm | forward (m) | path (m) | path/forward |
|---|---|---|---|
| **NMF CPG @ 1.2 Hz (hand-written ceiling, NOT a connectome arm)** | **2.3944** | 2.6918 | **1.1×** |
| rate core, direct map, gain 1 | 0.1554 | 1.0008 | 6.4× |
| LIF core, direct map, gain 5 | 0.0672 | 0.7645 | 11.4× |

Videos: `results/path2/best_walk_videos/`.

Context on how weak the direct-drive arms are — the structural audits found real
effects, but at millimetre scale:
- H4 dose response: real α=0 forward = **0.00208 m**; α=0.1 rewired mean −0.00164 m.
  Spearman −0.886 (monotone), but the whole effect is 3.7 mm.
- E3 worst case: real 0.00070 m vs random −0.0017 to −0.0098 m.
- H5 lesion: T2 lesion drives forward to −0.00912 m from a 2 mm baseline.
- E1 turning: `TURNING-NULL`.
- H1 amplitude-matched: `STRUCTURE-DRIVE-NULL` — amplitude-matched random ties real.

The owner reviewed the videos and judged the movement too limited to report as
walking. **That judgment stands.** Those numbers are the baseline you are trying to
escape, not results to defend.

---

## 12. What to produce

Every run writes a JSON artifact to `results/path2/` containing:
- `experiment`, `status`, `pre_registered_null`
- `protocol` — every constant, every sweep grid, every episode count
- `zero_learning: true`, `zero_learning_audit: {learning: false, parameter_updates: 0, training_markers: []}`
- `integrity` — pre/post file hashes, `file_hashes_equal`, real tensor SHA-256
- `rows` — per-episode records
- `ablation_ladder` — the four rows of §8.1 (full / constant / time-shuffled /
  CPG-only) for every reported arm. **An artifact without this block is
  incomplete and must not be presented as a result.**
- `self_interrogation` — the eight answers from §8.3
- `frac_active` and `motor_hz` per arm, so control activity is auditable
- `episode_spec_hash` — identical across arms, proving comparability
- `numeric_decision` and `decision`
- `reference_departures` — **every place you deviated from `cpg_controller.py` /
  `preprogrammed.py`, with file:line and the reason.** The adhesion omission and
  the fly→hexapod retargeting both go here. This section is not optional; a prior
  reference audit found three silent departures in a different port and every
  number measured through them was stale.

Name artifacts with a `cpg_` prefix and a variant suffix (`_v1_amplitude`,
`_v2_frequency`, `_v3_phase`) so nothing overwrites the recorded direct-drive
results. **This has bitten the project before**: four audits wrote unsuffixed
files and clobbered a prior core's results, which is why
`results/path2/clobbered_unsuffixed/` exists.

Render video for the best arm plus its matched control, on the same body and
camera, using `render_best_walk.py`. A path-to-forward ratio is not something a
table communicates.

Keep a running `results/path2/cpg_logs/ITERATION_LOG.md`: per iteration, the
hypothesis, the change, the predicted number, the measured number, and the verdict.
Include iterations that failed. That log is the actual deliverable if the task ends
in a null.

---

## 13. Final word

The gap between 0.0672 m and 2.3944 m is not a tuning problem. The direct map asks
fly motor neurons to be hexapod joint angles, which they are not and were never
going to be. The CPG interface is a real, published, defensible way to close that
gap — and it closes it by *handing the CPG the part that was failing*.

That trade is legitimate, and it is what NeuroMechFly itself does. But it is only
legitimate if you say so out loud, in the artifact, every time. The moment the
write-up implies the connectome is generating the gait rather than modulating a
prescribed one, the result becomes false — and it will be a *convincing* false
result, with a good video attached, which is the worst kind.

Report what the connectome actually contributes. If that turns out to be nothing,
say that. An honest null here is worth more than a walking robot with an
unsupported claim on top of it.
