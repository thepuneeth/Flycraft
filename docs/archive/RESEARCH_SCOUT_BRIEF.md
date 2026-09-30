# Pre-Stage-2 research scouting brief

**Who this is for:** the research agent, before we spend GPU budget on Stage 2
of `EMBODY_3WAY_SPEC.md`.

**Why now:** Stage 1 is complete and clean (frozen core held, hashes matched,
connectome held-out loss `1.048560` vs random `2.666219` and rewired
`1.827670`). Stage 2 is the decisive and expensive part. Before committing, we
want to know what the field already knows — so we do not (a) rediscover a
published null, (b) miss a design choice that makes the experiment sound, or
(c) claim novelty that someone already took.

**This is a read-and-report job. Do not write training code. Do not run
training. Do not modify the repo's experiment files.**

---

## 0. Context you must load first

Read, in order:

1. `FIELD_CODE_REVIEW.md` — what flygm/flybody/flygym actually do, and the
   frozen-vs-learned seam.
2. `EMBODY_3WAY_SPEC.md` — the experiment we are about to run.
3. `eon-fly-brain/results/path2/RUN_EMBODY_3WAY.md` — Stage 1 status.
4. `NEXT_EXPERIMENTS.md` + `PATH2_METHODS.md` — the standing constraints and the
   four prior training nulls (Outcome C, R0–R3, Rung 1, Outcome D).

Our setup in one line: a **frozen, measured** MANC VNC connectome rate model is
the recurrent controller; only a **linear sensory encoder** and **linear motor
decoder** are trained; the body is a **non-fly 18-DoF hexapod**; every result is
gated against **amplitude-matched random** and **degree-preserving rewired**
cores.

---

## 1. The questions, in priority order

Answer each with evidence. "Not found" is a valid, valuable answer — say it
plainly rather than stretching a weak match.

### Q1 — Novelty: has anyone driven a NON-fly body with a frozen measured connectome? (highest priority)
Our claimed moat is: measured weights as the computer + non-fly morphology.
Find any work that does either half, and especially both.
- Connectome-derived controllers in robots, non-biological morphologies, or
  simulated bodies other than the source animal.
- *C. elegans* connectome robotics is the obvious precedent family — find what
  was actually frozen vs tuned in each, because most tune.
- If someone has already done this, we need to know **before** we spend the GPU,
  and the honest move is to reposition, not to proceed quietly.

### Q2 — Precedent for "frozen biological recurrent core + trained thin interface"
This is our exact architecture. Search both the neuro and ML sides:
- Reservoir computing with biologically-measured (not random) reservoirs.
- Frozen-RNN / random-feature literature: how much can a *linear* readout extract
  from a fixed recurrent system, and what is the known ceiling?
- Connectome-as-reservoir papers specifically.
- **Key question for us:** is there a published result showing that once you
  train an interface, measured connectivity stops mattering? That is our
  repeated finding across four campaigns. If it is already published, our null
  is confirmatory (still worth reporting, but not novel). If it is *not*
  published, our null becomes a contribution in its own right.

### Q3 — How wide should the descending encoder be? (this decides our design)
This is the live design fork. Our encoder currently maps observations onto the
descending input neurons; a wide per-neuron encoder can puppet the core and make
the connectome decorative. Find the biological ground truth:
- Drosophila **descending neuron** organization — how many DNs, what they encode,
  and the evidence on whether locomotor commands are low-dimensional (a few
  coarse signals) or high-dimensional (many independent channels).
- Papers on DN population coding for walking speed, turning, and stopping.
- Any quantitative estimate of the **dimensionality** of descending drive during
  walking.
- Deliverable: a defensible recommended encoder output width, with citation. If
  the literature says descending control is effectively 2–5 dimensional during
  walking, we bottleneck to that and can defend it.

### Q4 — What is the accepted control for connectome-model claims?
We use amplitude-matched random + degree-preserving rewired. Check this against
the field:
- What null models do connectome-modeling papers actually use?
- Is degree-preserving rewiring standard, or is there a stronger accepted control
  (e.g. weight-preserving, distance-preserving, motif-preserving nulls)?
- Are we missing a control a reviewer would demand? Name it.
- Also check: how do others handle **activity-amplitude confounds**? Our
  amplitude-matching excluded `208/260` motor channels — find whether that
  scale of exclusion is normal or a red flag.

### Q5 — Locomotion targets without animal motion capture
We have no fly-in-a-hexapod recording, so our teacher is a synthetic
constant-speed forward path (`Option B`). Find:
- How other embodiment papers construct imitation targets when no mocap of the
  target morphology exists.
- Whether a synthetic body-path target (versus joint-level cloning) is accepted
  practice, and what it costs in result strength.
- Any critique of synthetic-teacher imitation we should pre-empt.

### Q6 — Training practicalities we can steal (cheapest wins)
- Sample budgets that actually worked for hexapod / multi-legged locomotion via
  PPO in MuJoCo — is `5,000,000` steps per arm realistic for emergence, or an
  order of magnitude short?
- Known-good PPO hyperparameters for 18-DoF hexapods specifically.
- Whether anyone reports single-env throughput numbers we can use to sanity-check
  our wall-clock estimate.
- **Concrete blocker to solve:** our run has no CUDA, and MPS fails with
  `NotImplementedError: Could not run 'new_compressed_tensor' from the 'mps:0' device.`
  Find whether sparse CSR on MPS is supported in any current PyTorch release, or
  whether the practical fix is dense fallback / CPU core / CUDA-only. Report the
  cheapest correct fix.

### Q7 — Morphology transfer
- Literature on transferring a controller between morphologies (different limb
  counts, different DoF).
- What counts as evidence that a controller "generalizes across bodies" rather
  than being refit per body.
- Anything that would sharpen or threaten our non-fly-body claim.

---

## 2. Where to look

- The three local reference repos under `fly_field_refs/` — **re-read their
  papers and READMEs for citations we have not followed**, especially anything
  they cite for controls, null models, or descending drive.
- Standard venues: bioRxiv, Nature/Nature Methods/Neuron/eLife/Current Biology,
  NeurIPS/ICLR/ICML/CoRL/RSS for the robotics and RL side.
- FlyWire / MANC / Janelia and connectomics consortium publication lists.
- Follow citation graphs both directions (who cited flybody / flygm /
  NeuroMechFly, and what they cite).

Leads worth checking but **not verified by me — confirm before citing**:
connectome-constrained models of the fly visual system; MANC/VNC network models
from the connectome papers; leg motor-neuron connectome work; Drosophila
descending-neuron atlases; *C. elegans* connectome robotics. Treat every one of
these as an unconfirmed pointer.

---

## 3. Deliverable format

Write `RESEARCH_SCOUT_FINDINGS.md` at the repo root:

1. **Verdict per question (Q1–Q7)** — one short paragraph each, leading with the
   answer, then the evidence.
2. **Evidence table** — paper/repo, year, what they froze, what they trained,
   what body, what controls, and one line on relevance to us.
3. **Novelty assessment** — the honest state of our claim after this search.
   If someone scooped part of it, say exactly which part.
4. **Design changes recommended before Stage 2** — concrete, ranked, each with
   the citation that motivates it. Explicitly answer the encoder-width question
   with a number.
5. **Threats a reviewer will raise** — the controls, confounds, or comparisons
   we are missing.
6. **Open / unresolved** — what you could not find, and where you looked.

Rules:
- **Cite everything.** Title, authors, year, venue, and a link or DOI. No claim
  without a source.
- **Distinguish read from skimmed.** Mark which papers you actually read versus
  found by abstract.
- **Flag uncertainty explicitly.** Do not smooth over a weak match into a strong
  claim.
- **No new experiment code.** This is reconnaissance.
- Do not commit.

---

## 4. The decision this feeds

After your report we decide three things:
1. **Encoder width** — narrow bottleneck versus current width.
2. **Whether Stage 2 runs at all**, or whether the design changes first.
3. **How the result gets framed** — novel finding versus confirmatory null.

Write the report so those three decisions are answerable from it.
