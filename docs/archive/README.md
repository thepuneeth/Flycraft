# Archived specs (superseded — historical only)

These were early build specs. Their content is absorbed into the live docs at
repo root. Kept for provenance; do not follow them for current work.

- `BUILD_PATH2.md` — original Path 2 build spec. Superseded by `HANDOFF.md` +
  `PATH2_METHODS.md`.
- `RUN_PATH2_WALKING.md` — the E0–E4 no-learn walking experiment plan. Executed
  and complete; results in `PATH2_METHODS.md` and commit `6481a2d`.
- `RUN_PATH2_RESERVOIR.md` — reservoir-readout pivot spec. Executed; verdict
  `READOUT TOO WEAK` in `PATH2_METHODS.md`.
- `AGENT_BRIEF_RESERVOIR.md` — self-contained reservoir agent brief. Executed.
- `AGENT_BRIEF_CMAES.md` — Outcome C CMA-ES gain-tuning brief. Executed; verdict
  `TUNED NEGATIVE; moat NULL`.
- `AGENT_BRIEF_OVERNIGHT.md` — Outcome C-close + Outcome D weight-learning brief.
  Executed; verdict `NO STABLE WALK; topology NULL` (commits `e4a1e00`–`093394d`).
- `RUN_PATH3_STEPDECODE.md` — zero-learn step-timing decode + fixed IK gait brief.
  Executed; superseded by the body-fix rerun.
- `RUN_BODYFIX_RERUN.md` — squat-stance body-fix + full ladder rerun on a standing
  body. Executed; results in `PATH2_METHODS.md` and commit `20c9fa5`. Body fix
  Gate A passed (walks 2.04 m path); prior body-level results marked
  confounded/superseded.
- `RUN_CONNECTOME_DRIVE.md` — Methods 1 (connectome-set coordination) + 2 (continuous
  phase drive). Executed; commit `b2deac4`. M1 NEGATIVE (real falls 16/20, worse than
  shuffled/random — VNC phase does not encode walkable coordination); M2 partial win
  (self-generated duty 0.495, 0/20 falls, positive coherence, beats random but not
  shuffled — rhythm earned). Connectome share ~25% → ~30–35%.

- `../../RUN_LAST_THREE_JOBS.md` — coordination (emergent coupling) / waveform
  (direct MN map) / pose (tonic stance) on the standing body. Executed; all three
  NULL. Coordination via dynamic inter-leg coupling falls 20/20.
- `../../RUN_UNTRIED_ZEROTRAIN.md` — the four genuinely-untried zero-training methods:
  COORD-212 (coupling from 212 identified coordination cells), COORD-INHIB (inhibitory
  edges only), WAVE-ANTAG (flexor/extensor antagonist map), VISION-EVAL (frozen
  retina→optic-lobe→DN steering). Executed; all four NULL.
- `../../RUN_DELAY_COUPLING.md` — delayed inter-leg oscillator coupling with a
  connectome-blind `(g, τ)` calibration and matched real/shuffled/random controls.
  Executed; DC-1/DC-2/DC-3 pass; registered verdict `DELAY-COUPLING-NULL`.
- `../../RUN_RUNG1_READOUT.md` — minimal-training memoryless readout moat test
  with connectome/random/rewired/static-input substrates. Executed; T-1/T-2/T-3
  pass; registered verdict `RUNG1-MOAT-NULL`.

**Established boundary (no active brief).** The frozen, untrained connectome
(FlyWire brain + MANC VNC) driving the non-fly hexapod has a real VNC-versus-
shuffle path positive and the registered seven-axis specificity audit earns
`SPECIFICITY-EARNED@{B1,B2,B3,B6}` on predeclared displacement/robustness
comparisons against 12 random and 12 rewired substrates. B5 rhythm and B7
steering are NULL; B4 fails its distribution/CI gate. These positives do not
establish canonical tripod coordination: coordination is NULL across six
independent zero-training methods (M1 static grouping, Phase B bulk coupling,
E4 proprioceptive feedback, COORD-212, COORD-INHIB, DC), while C direct MN
waveform remains `C1-ABSENT`. The only remaining zero-training walker is a
hand-designed CPG (WALKNET-class)—human-supplied coordination, outside the
frozen-connectome moat; it lives in the repo only as the clean-clock positive
control (Phase A gate T4), never credited to wiring. Full specificity evidence
is in `../../eon-fly-brain/results/path2/specificity_campaign.{json,csv}` and
`../../PATH2_METHODS.md`; final claim wording remains owner-gated.

Current source of truth: `../../HANDOFF.md`, `../../PATH2_METHODS.md`.

## Archived 2026-08-31 (executed briefs + superseded specs)

- `RUN_UNTRIED_ZEROTRAIN.md` — untried zero-training coordination/waveform/vision
  methods. Executed; coordination NULL.
- `RUN_LAST_THREE_JOBS.md` — coordination, waveform, pose. Executed; negative.
- `RUN_DELAY_COUPLING.md` — delayed inter-leg phase coupling, the last zero-training
  coordination lever. Executed.
- `RUN_RUNG1_READOUT.md` — Rung 1 minimal-training readout moat test. Executed;
  verdict `RUNG1-MOAT-NULL`.
- `RUN_SPECIFICITY_AUDIT.md` — wiring-specificity campaign. Executed and audited;
  registered result in its §11, methods in `PATH2_METHODS.md`.
- `FIELD_CODE_REVIEW.md` — review of flybody/flygym/flygm and the first embodiment
  build plan. Superseded by `RESEARCH_SCOUT_FINDINGS.md`.
- `RESEARCH_SCOUT_BRIEF.md` — the scouting task. Executed; answers in
  `RESEARCH_SCOUT_FINDINGS.md`.
- `EMBODY_3WAY_SPEC.md` — EMBODY-3WAY v1. Stage 1 executed and on record; Stage 2
  withdrawn unrun. Superseded by `EMBODY_3WAY_SPEC_V2.md` (wide 1,325-drive encoder
  replaced by the fixed 4-command bottleneck).

