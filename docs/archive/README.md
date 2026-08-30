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

Current source of truth: `../../HANDOFF.md`, `../../PATH2_METHODS.md`.
No active next-experiment brief (paused; next steps = untried non-training
methods #1 connectome-set coordination / #2 continuous phase drive).
