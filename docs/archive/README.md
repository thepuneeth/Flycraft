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

**Established boundary (no active brief).** The frozen, untrained connectome
(FlyWire brain + MANC VNC) driving the non-fly hexapod earns **steering + step
timing + rhythm/duty (~35–40%)** and does **not** supply coordination, waveform,
or pose un-trained. Coordination is NULL across six independent zero-training
methods (M1 static grouping, Phase B bulk coupling, E4 proprioceptive feedback,
COORD-212, COORD-INHIB, DC). The only remaining zero-training walker is a hand-designed
CPG (WALKNET-class) — human-supplied coordination, outside the frozen-connectome
moat; it lives in the repo only as the clean-clock positive control (Phase A gate
T4), never credited to the wiring. This is the boundary result.

Current source of truth: `../../HANDOFF.md`, `../../PATH2_METHODS.md`.
