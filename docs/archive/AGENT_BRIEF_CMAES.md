# Agent brief — Path 2 pivot: CMA-ES feedback/coupling tuning

A self-contained, multi-hour runbook. Read the whole thing, then `HANDOFF.md`
and `PATH2_METHODS.md`, before writing code. You are expected to work
autonomously for several hours: build the harness, run a long CMA-ES search,
run the 3-way control, produce all artifacts, and stop only before writing the
final research claim. Checkpoint often so a long run survives interruption.

---

## 0. Context — where the project is and why this experiment exists

Two clean negatives are committed:

- **No-learn (E0–E4):** the frozen fly VNC driving a PhantomX-class hexapod in
  MuJoCo produces leg **rhythm** but no alternating **tripod**, across four
  anatomical interventions (baseline, published-coordination completeness, real
  descending drive, anatomical proprioceptive feedback).
- **Reservoir readout:** the tripod gait is not linearly or small-nonlinearly
  decodable from the frozen VNC population (dynamic held-out R² < 0), and a
  connectome reservoir does no better than random/rewired. The reservoir was
  rich (~21 effective dims, 5,146/11,000 active neurons), so this is a genuine
  "gait not present in the frozen dynamics" result, not an under-excited
  artifact.

The literature says why, and what to do next. Every model that walks properly
supplies inter-leg coordination from **outside** the frozen connectome:
NeuroMechFly v2 uses a hand-designed CPG phase-bias matrix (its sensory rules
only add rough-terrain robustness); FlyGM uses the connectome as topology but
**learns** the weights (imitation + PPO); the 2025 connectome-CPG paper gets
**rhythm** from the real VNC but tripod only in embodied/coupled versions.

Nobody has produced walking from **frozen real connectome weights, open-loop,
zero learning** — which is exactly what we proved fails. This experiment is the
minimal honest version of NeuroMechFly's winning recipe: keep the real VNC as
the oscillator, and let CMA-ES set only the **coupling and feedback gains**
until a walking gait emerges. The connectome supplies the coupling *pattern*;
CMA-ES supplies the *strength*. We never hand-design a phase matrix.

---

## 1. The hard rules (the project's spine — violating any voids the result)

1. **VNC synaptic weight pattern is frozen.** CMA-ES never edits individual
   synapses, never rewires, never adds edges. It scales a small set of global
   gains that were previously fixed at arbitrary values.
2. **Frozen constants stay frozen** and are NOT in the search vector: `gW`,
   `tau`, `dt`, substeps, PD gains, MN firing caps, rest angles, threshold,
   activation gain. Use the exact frozen protocol (`tau=.02`, `dt=.005`,
   substeps 4, `gW=.03`, threshold 7.5, activation gain 1, cap 200,
   heterogeneity seed 1).
3. **Objective is locomotion only.** Fitness rewards forward distance and
   staying upright. It must contain NO prey-distance, catch, or tripod-coherence
   term. Grep your own fitness code to prove this before every commit.
4. **Tripod is measured, never rewarded.** We are testing whether a tripod
   *emerges* from tuning the real coupling. Rewarding it would design the
   answer. Measure it with the existing E0 metric and report it as an outcome.
5. **Steering stays brain-driven** at deploy, exactly as `run_vnc_chase.py`.
6. **Report everything honestly:** the tuned gene vector, the full fitness
   trajectory, emergent gait, and whether random/rewired match connectome. If it
   does not walk, say so. Do not widen the gene vector beyond section 3 to force
   a walk; that is a different, user-approved experiment.

When a decision is genuinely ambiguous, prefer the choice that keeps the
connectome load-bearing and the objective task-agnostic, log the decision, and
continue. Only hard-stop for the user at the final-claim gate (section 8).

---

## 2. Autonomy and time budget (work continuously, checkpoint always)

You run C0 → C1 → C2 without waiting for the user between them. Post a progress
report at each internal gate and **continue**. The only hard stop is before the
final claim.

Rough wall-clock (tune to the machine; parallelize rollouts across CPU workers):

| phase | work | rough time |
|---|---|---|
| C0 | harness + regression check | 30–60 min |
| C1 | CMA-ES on real connectome (the long run) | 2–4 h |
| C2 | same search on random + rewired | 2–4 h |
| wrap | audits, figures, report, integrity | 30–60 min |

**Checkpointing is mandatory.** CMA-ES exposes its full state; pickle the
optimizer every generation to
`results/path2/cmaes_checkpoint_<condition>.pkl`, and write the running best
gene + fitness to JSON every generation. On restart, resume from the checkpoint
rather than starting over. A multi-hour run must survive a killed process.

---

## 3. The gene vector (what CMA-ES may tune — ~18–30 scalars)

Structure is fixed by the connectome; only these global magnitudes are searched.
Keep an explicit ordered list mapping gene index → parameter → injection point,
and save it as `results/path2/cmaes_gene_spec.json`.

1. **Descending drive (4):** DNg100 level, DNb08 level, left/right balance,
   tonic offset. Previously flat 100 (E3). Bound each to a sane range
   (e.g. 0–300); log bounds.
2. **Intersegmental coupling gains (3–6):** one scalar multiplier per real
   T1↔T2↔T3 connectome pathway. Identify these edges in the built CSR by source
   and target neuromere (soma-neuromere annotations from `fetch_manc.py`); a
   pathway is the set of edges whose source is in neuromere A and target in
   neuromere B. Apply the gene as a scalar multiply on exactly those edge
   weights at load time. This scales existing edges only — it never creates a
   coupling that is not in the graph, and never sets a per-leg phase. The
   connectome supplies the pattern; the gene supplies the strength.
3. **Sensory feedback gains (6–12):** per proprioceptor-class gain and sign for
   the E4 afferent path — campaniform sensilla, hair plate, chordotonal organ,
   and leg-load — grouped by class (not per cell). Feedback encoding
   (contact/joint-angle → sensory-pool current) uses the fixed E4 encoding;
   only the per-class gain/sign is a gene. Reuse `run_sensory_feedback.py`.
4. **Optional single global feedback delay (1):** include only if C1 shows
   instability that a delay would fix; declare it if added.

No free phase-bias matrix. No per-leg designed oscillator. No new edges. No
change to `gW`/`tau`.

---

## 4. Fitness (declare once in code, never edit to chase a result)

```text
fitness(gene) = mean over a fixed seed set of:
    forward_distance_m(rollout)  -  fall_penalty(rollout)
fall_penalty = FALL_CONST if min body height < H_MIN at any step, else 0
```

- Rollout: reuse the `run_vnc_chase.py` MuJoCo setup (brain steering, VNC
  stepper, `mn_to_joint`, PD, body). Length 3–5 s, static prey, a fixed small
  seed set (e.g. 4 seeds) averaged to reduce noise. Declare the seeds.
- Optionally multiply by upright-fraction (fraction of steps above `H_MIN`);
  declare it up front if used. Nothing else enters fitness.
- `FALL_CONST`, `H_MIN`, seeds, and rollout length are fixed constants declared
  at the top of `cmaes_walk.py`, not tuned.
- Forward distance is body displacement along the initial heading, not toward
  prey (no prey term).

---

## 5. Algorithm and infrastructure

- Use `cma` (pycma). If not installed, `eon-fly-brain/.venv/bin/pip install cma`.
- Dimensions ~18–30. Population `popsize ≈ 4 + floor(3·ln(D))` (pycma default),
  bump to ~12–16. Budget a few hundred generations; set a generation cap and a
  wall-clock cap, whichever first.
- Fixed optimizer seed, fixed initial mean = the frozen-default gains (all drive
  at the E3/E4 values, coupling multipliers = 1, feedback gains = E4 values),
  fixed initial sigma (~0.3 of each bound range; use CMA bounds/scaling).
- Parallelize the `popsize` rollouts per generation across CPU workers
  (`multiprocessing` pool). Each rollout is independent and CPU-bound. No GPU.
- Log per generation: generation index, best fitness, mean fitness, best gene,
  wall-clock. Append to `results/path2/cmaes_fitness_<condition>.csv`.

---

## 6. Build steps

### C0 — harness  → `code/cmaes_walk.py`
1. Wrap the existing episode pipeline as `rollout(gene, seed) -> metrics`
   returning forward distance, min/final height, fell flag, and the measured E0
   tripod coherence on the joint targets.
2. Implement gene injection at the three points in section 3 (drive, coupling
   multipliers on identified intersegmental edges, per-class feedback gains).
3. **Regression check (must pass before C1):** with `gene = frozen defaults`
   (multipliers 1, drive/feedback at E3/E4 values), `rollout` must reproduce the
   current E4 open-loop/closed-loop behavior within numerical tolerance. If it
   does not, the injection is wrong — fix it before searching. Save the check to
   `results/path2/cmaes_regression_check.json`.

### C1 — CMA-ES on the real connectome  [internal gate, then continue]
1. Run the search with checkpointing (section 2). Let it run to the generation
   or wall-clock cap.
2. Take the best gene; run a clean evaluation rollout; render the multiview
   video (`cmaes_walk_multiview.mp4`); record best distance, upright, and
   measured tripod coherence.
3. **Post a progress report:** fitness curve, best distance, walks yes/no,
   emergent gait (tripod or other), honesty line. Then **continue to C2**
   (do not wait for the user).
4. If fitness is flat at/near the frozen-default value after a real budget
   (no improvement), that is the ceiling signal — still run C2 for completeness
   (random/rewired should then also fail to walk), and prepare the terminal-fork
   report. Allowed diagnostics if flat (no tuning, just inspection): confirm
   genes actually change rollouts (sensitivity check — vary each gene, confirm
   metrics move), confirm no rollout is silently erroring, confirm the parallel
   workers use distinct genes. Log findings.

### C2 — 3-way prior test  → `code/cmaes_3way.py`  [internal gate, then stop]
1. Repeat the **identical** CMA-ES procedure (same gene spec, bounds, popsize,
   generation/wall-clock cap, sigma, optimizer seed, fitness, evaluation seeds)
   on three graphs:
   - **connectome:** real MANC weights (reuse the C1 best if identical config),
   - **random:** random weights matched to size and degree distribution,
   - **rewired:** degree-preserving edge shuffle of the real graph.
   Use the existing shuffle machinery in `vnc_stepper.py` where possible;
   declare and seed the random/rewired generation.
2. Report per condition: best fitness, generations-to-first-walk, best distance,
   measured tripod coherence, and the fitness-vs-generation curve (sample
   efficiency). Emit `cmaes_3way.{json,csv,png}`.
3. Verdicts:
   - **PRIOR CONFIRMED:** connectome walks better / sooner / more efficiently
     than random and rewired — FlyGM's result transferred to a non-fly body,
     the moat.
   - **NULL:** random matches connectome — the wiring is not a special
     locomotor prior for this body; tuning alone explains the walk. Honest,
     publishable.
4. **Post the 3-way table, then STOP for user approval** before any final claim.

---

## 7. Terminal fork (state plainly; never hide a negative)

- **C1 walks:** movement achieved — the frozen real fly VNC plus tuned
  coupling/feedback walks a hexapod. Novel; moat intact (pattern frozen, only
  gains tuned). C2 then tests whether the connectome made it easier.
- **C1 will not walk after a real budget:** the honest ceiling — the fly VNC is
  not a sufficient locomotor controller for this body without retraining its
  actual weights. That definitively closes the no-retrain line. Also
  publishable. Do NOT then start editing synapses to force a walk; that is a
  separate experiment requiring explicit user approval.

---

## 8. Gates, deliverables, integrity

**Gate discipline:** C1 and C2 post a report and continue; the single hard stop
is before writing the final claim, after C2. At every post, include one line
confirming the honesty rules held: weights frozen, fitness = locomotion only,
constants untouched, tripod measured-not-rewarded.

**Deliverables:**
- `code/cmaes_walk.py`, `code/cmaes_3way.py`.
- `results/path2/cmaes_gene_spec.json`, `cmaes_regression_check.json`,
  `cmaes_fitness_<condition>.csv`, `cmaes_checkpoint_<condition>.pkl`,
  `cmaes_best_gene.json`, `cmaes_walk_metrics.json`,
  `cmaes_walk_multiview.mp4`, `cmaes_3way.{json,csv,png}`.
- A `PATH2_METHODS.md` section: gene-vector definition, fitness declaration,
  per-condition results, emergent-gait measurement, verdict. No final-claim
  wording until user approval.
- Large rebuildable `*weight_csr.pkl` remain gitignored; commit code, small
  artifacts, and report only.

**Integrity before any commit:**
- Python compile all `code/*.py`.
- Confirm the VNC weight *pattern* is unchanged: hash the base CSR before gene
  injection and after a rollout; gene injection must be applied to a copy, never
  persisted back to the canonical graph.
- Grep the fitness/rollout source for `prey`, `catch`, `distance_to`, `tripod`,
  `reward` used in scoring — the fitness must not reference prey or tripod.
- Confirm frozen constants (`gW`, `tau`, `dt`, substeps, PD) do not appear in
  the gene spec.
- `git diff --check`. Commit with a message stating the current verdict.

**Commit cadence:** commit after C0 (harness + regression), after C1 (with the
fitness curve and best-gene artifacts), and after C2 (with the 3-way table).
Each commit is a checkpoint of a multi-hour run; do not wait until the end.
