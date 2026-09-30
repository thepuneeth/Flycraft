# Voice and visual audit, round 02

## Reference register

- Primary voice sample: the author's retinitis-pigmentosa GAN paper. It uses direct
  sentences, repeated ``This study'' purpose statements, explicit definitions, and
  cause-and-effect transitions. Its accessible cadence is useful; its unsupported
  superlatives and any project-specific errors are not copied.
- Format sample: the local BIBM UGHS decomposition. It uses a dense but readable
  two-column layout, one process/progression figure, compact metric comparisons,
  sample or state views, and captions that tell the reader what to inspect.
- Closest domain visual precedent: FlyGM's connectome-to-body pipeline and
  time-sequenced locomotion snapshots. These motivate a visible body rollout and a
  clear distinction between graph computation and body feedback.

## Current-draft audit

The draft preserves the measured results and has the right claim boundary, but its
abstract and introduction read more like compressed systems prose than the author's
original explanatory register. Repeated phrases such as ``registered transfer set,''
``wiring-specific,'' and ``robustness'' add abstraction or evaluation jargon where a
short concrete sentence would be clearer. Several negative-form sentences are factual
scope limits and must remain; rhetorical contrasts can be rewritten in positive form.
The future-work paragraph also needs to say plainly which observations enter the
translation layer and that only its encoder and decoder train.

## Evidence-preservation contract

The rewrite must preserve every reported number, endpoint, sample size, null-model
definition, citation key, and unrun-stage boundary. No result from the v4 follow-on
may enter the completed-study claims. The exact author and school affiliation remain
user-supplied fields.

## Visual changes for this round

Keep the existing two-figure plan. Use the data-derived MANC/VNC excerpt, a
brightness-adjusted crop of the actual MuJoCo rollout frame, and the existing compact
body/rewiring result plot. Label schematic elements as schematic, identify simulation
assets, and mark the future translation layer as unrun. Add ``trainable encoder'' to
the future strip so the trainable boundary is visible without adding a new figure.

## First re-draft targets

1. Rewrite abstract and introduction in the author's direct, explanatory cadence.
2. Replace generic evaluation adjectives with measured descriptions.
3. Make related-work positioning concrete and narrow.
4. State the future observation, four-latent bottleneck, fixed DN map, trainable
   encoder/decoder, and unrun PPO stage in plain language.
5. Run the mechanical gate, render all five pages, inspect figures, and send the
   revised draft to fresh independent reviewers.
