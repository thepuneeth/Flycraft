# Next experiments — menu for the research agent

> **Superseded for the active line (2026-09-05).** The embodiment work has moved
> to `EMBODY_V5_PLAN.md`. The menu below remains the reference for the Path-2
> zero-training and specificity experiments, which are unchanged and still valid;
> it is not the current queue.

**Where we are (verified, 2026-08-31; evidence assembly, not final claim).**
The real connectome's **wiring structure** produces forward drive through the
fixed anatomical map against amplitude-matched random structure, and the
effect survives the declared body perturbations. Direct bilateral descending
asymmetry also produces a turning advantage. H4 localizes loss under
fractional rewiring and H5 ranks source-lesion sensitivity. T1 earns a narrow
one-scalar target-speed error advantage; T2's identical memoryless forward
readouts do not preserve an advantage. H2 remains a candidate-only graph
screen. These are separate narrow results, not evidence that the connectome
walks or coordinates a tripod.

Caveats that ride with the forward result: (1) B1/B2/B3/B6 are one
descending-drive finding swept over horizon/drive/body, not four findings; (2)
it is **forward drive, not coordination** — canonical tripod, rhythm, and
visual-steering endpoints remain NULL; (3) it is **gain-regime-dependent** —
the declared operating region wins, while higher gains over-drive backward;
(4) H4's dose curve is strongly but not perfectly monotone point-by-point.

The canonical coordination endpoint remains NULL across six zero-learning
methods plus the trained forward readout; C waveform, D pose, visual steering,
E2 speed monotonicity, H2 mechanism, and the Rung 1 moat do not add a tripod.

**Current decision boundary:**
> Present narrow wiring-specific drive, turning, and robustness evidence while
> keeping canonical coordination negative; do not write a broader mechanism or
> walking claim without owner approval.

Every experiment below carries the standing constraints: zero-learning unless the
row says *training*; freeze constants; one shared engine; real/random(+rewired)
with **amplitude-matched** controls now mandatory; multi-seed (K=12); paired Holm
+ bootstrap CIs; watched real video; pre-registered null; hash-equal; **no final
claim without owner approval**; an honest null is a deliverable.

## Completion status (all planned H/E/T jobs done)

| job | decision | recorded result |
|---|---|---|
| H1 amplitude match | `STRUCTURE-DRIVE-EARNED` | forward `+.2358 m` vs matched random, 95% CI `[+.1868,+.2805]`, exact/Holm `p=.000244`, `0/12` |
| H2 mechanism | candidate-only | pooled graph-scalar screen unsupported; mechanism claim withheld |
| E1 turning | `TURNING-EARNED` | odd-yaw advantage `+6.916°`, CI `[+4.892,+8.899]`, exact/Holm `p=.000244`, `0/12` |
| E2 speed | `SPEED-MODULATION-NULL` | real `ρ=.400` vs random mean `.017`, `p=.0166`, CI `[+.117,+.650]`; `5/12` controls match/beat; gates fail |
| H3 gain envelope | reported | `16/25` cells pass; baseline CI `[+.1868,+.2804]`, Holm `p=.00610` |
| H4 dose response | `DOSE-RESPONSE-EARNED` | α=`0→1` means `.2241→-.0087 m`, Spearman `ρ=-.829`; α=1 CI `[+.2101,+.2538]`, Holm `p=.00122` |
| H5 lesion localization | `LESION-LOCALIZATION-EARNED` | T1 loss `+.2664 m` `[+.2555,+.2773]`; VNC-sensory `+.1949 m` `[+.1899,+.1999]`; T3 `+.1747 m` `[+.1622,+.1870]` |
| E3 body generalization | `BODY-GENERALIZATION-EARNED` | mean advantage `+.2346 m` `[+.1870,+.2765]`; worst-case `+.2416 m` `[+.1915,+.2862]`; all 9 perturbations significant |
| T2 forward readout | `T2-FORWARD-READOUT-NULL` | connectome−random `−.0094 m` `[-.0296,+.0098]`, `p=.814`; connectome−rewired `−.0653 m` `[-.1002,-.0327]`, `p≈.9996`; leak pass |
| T1 scalar drive | `T1-SCALAR-CONTROL-EARNED` | error `.0080 m/s` real vs `.0135` matched random and `.0123` rewired; Holm `p=.000976/.00269` |

Detailed methods, CIs, audit fields, and artifact paths are in
`PATH2_METHODS.md` and `HANDOFF.md`. All H1/E1/E2/H3/H4/H5/E3 jobs are
zero-learning with equal graph hashes; T1/T2 train only declared scalar/readout
objects and record zero VNC-weight updates. The two formerly omitted T1
`IN12B013` cells are included in the fixed 212/212 published-candidate census.

---

## Track H — Harden & explain the positive (zero-training; complete)

These completed runs bound, test, and localize the positive without promoting it
to a coordination or mechanism claim.

### H1 — Full confirmation with the amplitude control folded in (complete)
**Question:** does real beat random on forward drive across **all 12 seeds × 20
episodes** when random is **amplitude-matched** (the control the campaign skipped)?
**Method:** re-run B1's forward-distance test adding an `amplitude-matched-random`
condition (rescale each random motor history to real's per-channel SD, same map,
same body). Paired sign-flip, bootstrap CI, `n/12`.
**Pre-registered null:** amplitude-matched random ties real → the positive was
amplitude after all, `STRUCTURE-DRIVE-NULL`. The initial pilot had unmatchable
zero-source channels; the strict all-18-varying-channel rerun is the formal
result.
**Earns:** the positive as a defensible primary result. **Cost:** minutes.

### H2 — Mechanism: why does shuffling structure collapse drive? (complete; candidate-only)
**Question:** what property of the real recurrent weight matrix sustains
supra-threshold motor activity that the same-multiset shuffle loses?
**Method:** pure analysis, no body. Compare real vs 12 random/rewired on: spectral
radius / leading eigenvalues of W, self-sustaining-activity regime (does real sit
near criticality?), motor-pool in-degree concentration, presence of recurrent
excitatory loops feeding motor neurons. Correlate each structural scalar with the
per-seed forward drive.
**Pre-registered null:** no structural scalar predicts forward drive across seeds
→ effect is real but unexplained (still reportable, weaker).
**Earns:** the *mechanism* sentence of the paper. **Cost:** low, no rollouts.

### H3 — Gain-regime envelope (complete)
**Question:** over the 2-D gain space, where does real>random hold, and is the
declared operating point special or arbitrary?
**Method:** sweep map gains on a grid; chart the region where real beats the
random distribution on forward. Report the whole envelope honestly.
**Pre-registered null:** the win occupies a thin, arbitrary sliver → disclose as a
severe limitation. **Earns:** honest bounds on the claim. **Cost:** moderate.

### H4 — Dose–response (structure → drive association; complete)
**Question:** is forward drive **monotone** in how much real structure is
preserved? **Method:** interpolate real↔random by rewiring a fraction α of edges
(α = 0, .1, .25, .5, .75, 1.0), amplitude-matched throughout; measure forward vs α.
**Pre-registered null:** no monotone relation → structure is not the graded cause.
**Earns:** causal (not just contrastive) evidence structure drives locomotion.
**Cost:** moderate.

### H5 — Lesion localization (complete)
**Question:** which parts of the connectome carry the drive? **Method:** ablate by
segment / neuropil / pathway class, amplitude-matched, measure forward-drive loss;
rank contributors. **Pre-registered null:** drive is diffuse, no localizable
substrate → report as distributed. **Earns:** a circuit-level story. **Cost:**
moderate.

---

## Track E — Does the positive EXTEND beyond straight-line forward? (zero-training; complete)

The moat's real promise is generality. These test whether structure-specific drive
is a one-trick artifact or a usable controller.

### E1 — Turning from descending-drive asymmetry (complete)
**Question:** given a **left/right asymmetric** descending command, does real
structure convert it into turning better than random? (Distinct from VISION-EVAL,
which tested the full visual pathway and was null — this is direct drive asymmetry.)
**Method:** asymmetric drive input, measure yaw-rate vs random, amplitude-matched.
**Pre-registered null:** real turns no better than random → drive is
forward-only, not steerable. **Earns:** a second controllable axis. **Cost:** low.

### E2 — Speed modulation (re-mine B3 honestly; complete NULL)
**Question:** does forward speed scale **monotonically** with descending drive for
real where random stays flat? **Method:** re-analyze the existing B3 drive sweep
under amplitude-matching; test monotonic speed–drive coupling.
**Pre-registered null:** no real-specific speed control. **Earns:** "drive knob"
claim. **Cost:** reuses B3 data — cheap.

### E3 — Body-generalization done rigorously (the moat's core test; complete)
**Question:** does real's forward-drive advantage **generalize across
morphologies** where random fails — the actual non-fly-body moat promise?
**Method:** the B6 body sweep, but as its own study, amplitude-matched, many
bodies (leg/mass/friction grid), real vs random distribution, worst-case and
mean advantage. **Pre-registered null:** real's advantage is body-specific / no
more robust than random → the "works across bodies" claim dies. **Earns:** the
strongest, most novel honest claim available to this project. **Cost:** highest in
this track (overnight); run after H1–H2 confirm the base effect.

---

## Track T — Minimal-training frontier, scoped so training can't fake it (*training*; complete)

Rung 1 was null — but Rung 1 trained a readout to **clone the clean-clock tripod**,
an axis where the connectome is null. It never trained on the axis where structure
actually helps (forward drive). These retest training **on the right axis**, built
so a flexible readout can't manufacture the win.

### T1 — Train only a scalar drive controller (smallest possible training; complete)
**Question:** with the connectome **frozen**, train a tiny controller that sets the
**1-D descending drive level** to hit a target forward speed. Does the connectome
substrate reach target speed with **less control effort / fewer parameters / lower
sample cost** than amplitude-matched random?
**Metric:** control complexity & sample-efficiency, **not** raw walk distance —
so credit accrues to the substrate, not the controller.
**Pre-registered null:** connectome needs equal-or-more control effort than random
→ structure gives no control advantage. **Earns:** "connectome makes control
easier," a clean minimal-training claim. **Cost:** low training, CPU.

### T2 — The fair moat-training test (Rung 1 redone on the winning axis; complete NULL)
**Question:** train **identical** readouts on `connectome` vs
`amplitude-matched-random` vs `rewired`, target = **forward progress** (not tripod
cloning). Does connectome beat matched-random under identical budget?
**Guards:** memoryless readout, capacity-capped, no time/phase input, static-input
leak detector, connectome hash-equal, forbidden from loss:
path/coherence/feet/fall of the *evaluation* body (train on a held-out target
signal, evaluate on body) — reuse Rung 1's integrity gate.
**Pre-registered null:** connectome ties amplitude-matched random under training →
even on its best axis, training erases the structural edge; the moat is
zero-training-only. **Earns:** whether the positive **survives training** — the
question Rung 1 asked on the wrong axis. **Cost:** moderate, CPU.

### T3 — Closed-loop / curriculum (not run)
T1 earned, but T2 was `T2-FORWARD-READOUT-NULL`; the preregistered escalation
gate therefore remains closed. RL, recurrence, learned coupling, and larger
readouts remain out of scope.

---

## Housekeeping (closed except owner claim review)

- B1/B2/B3/B6 are documented as one descending-drive finding; H1 amplitude
  matching and H3 gain dependence are recorded in the working report.
- H1–H5/E1–E3/T1–T2 scripts and result artifacts are present in the working
  tree. No constant was changed after its clean rerun.
- The two `IN12B013` T1 cells are documented in the 212/212 census; their
  addition is hash-clean and orthogonal to the positive.
- Invalid H4 cache-collision and non-formal H1 pilot artifacts remain
  quarantined for provenance. Final claim wording still requires owner approval.

---

## Execution record

1. **H1 + H2:** complete. Amplitude-matched forward positive survives; graph
   mechanism screen remains candidate-only.
2. **E2 + E1:** complete. Speed modulation is NULL; direct turning earns.
3. **H3/H4/H5:** complete. Gain envelope, negative dose association, and lesion
   sensitivity are reported with their limits.
4. **E3:** complete. Forward advantage generalizes across the declared body
   perturbations.
5. **T2 then T1:** complete. Forward readout is NULL; scalar control earns a
   narrow held-out error advantage.
6. **Final claim:** paused for owner approval.

**Honest framing of the ceiling:** even the best-case outcome here is *"the frozen
connectome's structure supplies body-general forward locomotor drive"* — a real,
novel, narrow claim. It is **not** "the connectome walks." Coordination stays
null. Keep the scope where the evidence is.
