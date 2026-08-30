# Path 2 — Walking experiment plan

Goal: reach ONE of two end states, honestly.

- **Outcome A (no-learn win):** the untrained FlyWire brain + real Male-CNS VNC
  drives a PhantomX-class hexapod into coherent alternating-tripod walking,
  with zero learning. Novel result. Lock it.
- **Outcome B (learning pivot):** we prove the no-learn chain cannot produce
  tripod, localize exactly why, and hand back a concrete plan for a
  connectome-as-prior training study (Path B) on the hexapod.

Either outcome is a publishable deliverable. A *tuned* catch is not. Same
discipline as before: freeze all constants (`gW`, `tau`, `dt`, substeps,
drive, gains, thresholds, rest angles). Every fix below is an **anatomical or
data-selection** change with a declared rule — never a search over
performance. No reward, target, distance-minimization, or catch-success
fitting anywhere. Stop and report at every GATE before proceeding.

The missing piece is **inter-leg tripod coordination**. The VNC oscillates but
the six legs do not phase-lock into two alternating tripods. Everything else
already works (rhythm, motor activity, stable physics, real ≫ shuffled). These
experiments find where the tripod signal is lost, or prove it never existed in
the extracted circuit.

---

## E0 — Phase probe at the VNC motor-output layer (diagnostic, run first)

This decides which branch we are on. No changes to anything.

Measure the phase of each of the six legs directly from the raw VNC
motor-neuron pool activity, **before** `mn_to_joint`, under the frozen constant
drive already used for the rhythm validation.

1. For each leg (L1,L2,L3,R1,R2,R3), take its coxa-promotor/remotor motor-pool
   population activity (the swing-drive channel). Bandpass at the dominant
   rhythm band (~2–3 Hz), extract instantaneous phase (Hilbert).
2. Build the 6×6 mean pairwise phase-difference matrix over the analysis window
   (half-run burn-in, same protocol as `analyze_vnc_rhythm.py`).
3. Tripod signature to test for: {L1,R2,L3} mutually in-phase, {R1,L2,R3}
   mutually in-phase, the two groups ~half-cycle apart (0.5 ± 0.15).

Report the full 6×6 matrix, a tripod-coherence scalar (mean within-group phase
concentration minus cross-group), and a clear verdict:

- **E0-PRESENT:** tripod phase structure exists at the VNC output → go to E1.
- **E0-ABSENT:** no tripod at the VNC output → go to E2.

GATE 0: post the matrix + verdict. Do not proceed past the branch it selects
without noting it.

---

## Branch if E0-PRESENT — the signal exists, the map is losing it

### E1 — Anatomical motor-map audit (`mn_to_joint`)

The reduction 5 fly leg joints → 3 hexapod joints (coxa/femur/tibia) may be
merging pools across legs or across left/right, destroying phase.

1. Print the current `mn_to_joint` mapping: which MN pools feed each of the 18
   hexapod joint targets. Look for cross-leg or L/R pooling.
2. Rebuild the map from **anatomical identity only**: each hexapod leg's three
   joints receive only that same leg's homologous fly motor pools
   (T1→front, T2→mid, T3→hind; promotor/remotor→coxa, levator/depressor→femur,
   flexor/extensor→tibia). Declared anatomical rule, no fitting.
3. Re-run E0's phase measure at the **joint-target** layer.

GATE 1: if joint-target tripod coherence now appears, run the full closed-loop
episode + control study. If a coherent alternating tripod + net forward
locomotion emerges → **Outcome A candidate**, go to VERIFY. If not, the map was
not the only loss; go to E3.

---

## Branch if E0-ABSENT — the coordination is not in the extracted circuit

### E2 — Intersegmental coordination-cell completeness

Tripod coordination in the fly is carried by intersegmental interneurons
linking the three leg neuromeres. Our subgraph may not include the path that
couples T1↔T2↔T3.

1. From the frozen graph, list which cells actually carry signal **between**
   neuromeres (T1↔T2↔T3 edges), by type. Compare against the published
   coordination-cell set and known ascending/descending intersegmental classes
   (e.g. the eLife 96084 coordination labels that span segments).
2. If anatomically-defined intersegmental coordination cells are missing under
   the current selection rule, widen the rule **by declared anatomy** (cell
   type / neuromere span), not by any performance criterion. Record exactly
   which cells and edges are added and why.
3. Rebuild, re-run E0.

GATE 2: if E0 now shows tripod → go to E1 (map), then VERIFY. If still absent
after including the anatomically-complete intersegmental set → the open-loop
CPG does not self-organize tripod; go to E3, then E4.

---

## E3 — Structured descending drive (still no learning)

We currently drive both sides flat at 100. Real walking uses structured
descending command, not a flat bilateral clamp.

1. Replace the flat drive with the **brain's actual computed descending output**
   (DNg100/DNb08 activity from the connectome forward pass), left/right as the
   brain produces it. This is computed, not fitted.
2. Re-run E0 at VNC output and joint layer.

GATE 3: report whether structured biological drive changes the phase matrix.
If tripod appears → VERIFY. If not → E4.

---

## E4 — Sensory feedback loop (the biologically-required piece)

Insect inter-leg coordination is not pure feedforward CPG; it depends on
load/proprioceptive feedback. Open-loop VNC may be physically unable to
self-organize tripod without body feedback. This is the strongest honest
hypothesis if E0–E3 fail.

1. Close the loop **through connectome sensory afferents only**: feed hexapod
   leg load + joint angle into the VNC's sensory neuron populations (campaniform
   / hair-plate / chordotonal homologs) using the connectome's own sensory→VNC
   wiring. The feedback path is anatomical, its gains frozen — no controller is
   fitted.
2. Run the closed-loop episode. Measure joint-layer tripod coherence and
   locomotion.

GATE 4: if biologically-wired sensory feedback produces coherent tripod +
forward walking → **Outcome A**, the headline becomes "connectome CPG walks a
hexapod only when closed through its own sensory loop." Go to VERIFY.
If it still fails → **Outcome B**: the no-learn chain is exhausted.

---

## VERIFY (only when a tripod + walking result appears)

Before any walking claim:

1. Confirm the change that produced it was anatomical/data-declared, not a
   performance search. State the rule in one sentence.
2. Zero-learning scan (optimizer/backprop/reward/loss/target-fit) — must pass.
3. Controls: real vs shuffled vs random-drive must still separate correctly;
   the walk must survive on the real wiring and break on shuffled.
4. Freeze constants unchanged from the rhythm-validation protocol.
5. Reproduce twice, same seed and fresh seed.
6. Render the multiview video; confirm visible alternating tripod, not sprawl.

Then STOP and post for approval before writing the headline.

---

## Outcome B — learning pivot plan (write this if E0–E4 exhausted)

If no anatomical/feedback fix yields tripod, the honest conclusion is: the
frozen fly VNC does not transfer to hexapod walking open-loop, and the deliver-
able becomes the **connectome-as-prior** study — novel because it is on a
non-fly body, which FlyGM never tested.

Deliver a plan, not a trained model, containing:

1. **Frozen substrate:** the real MANC VNC graph stays fixed (weights not
   trained). Only a minimal readout / sensory-feedback-gain layer is learned.
2. **The measured comparison (the actual science):** train the same minimal
   controller three ways —
   - connectome-initialized (real MANC),
   - random-initialized (matched size/degree),
   - rewired-initialized (degree-preserving shuffle).
   Measure walking performance and sample-efficiency for each. The claim is
   *connectome-init beats random/rewired on a hexapod* — FlyGM's result, but
   transferred to a body the fly never evolved for. That transfer is the moat.
3. **Cheapest saturating hardware:** single RTX 4090; cache the frozen
   connectome forward pass so the GPU trains only the small readout.
4. **Zero-learning control retained:** the open-loop no-learn result (this whole
   Path 2) becomes the honest baseline the learned version is measured against.

Report which experiment (E0–E4) forced the pivot and why anatomy alone could
not close it.

---

## Discipline (unchanged)

- No constant touched for performance. Every fix carries a one-line anatomical
  or data-selection justification.
- Stop at every GATE and post results before proceeding.
- A no-walk result with a clean localization is a real finding. A tuned walk is
  worthless. Do not manufacture tripod by fitting gains.
- Report negatives plainly. Do not soften E0–E4 failures.
