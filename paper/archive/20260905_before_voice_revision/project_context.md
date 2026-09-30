# Flycraft paper context

## Identity sentence

A frozen, measured Drosophila MANC/VNC can transfer wiring-specific motor drive
to a different hexapod through a fixed interface, but the completed
zero-learning transfer does not reconstruct canonical tripod coordination; a
later four-latent translation layer is designed to test whether embodiment-level
walking and balance can be learned without changing the core.

## Venue and audience

- BIBM 2026 Undergraduate and High School Student Research Forum (UGHS).
- High-school paper. Maximum five pages including figures, tables, and references.
- IEEE Computer Society conference format. Student is first author and must be
  identified as a high-school student in the affiliation.
- Audience: computational biology, connectomics, embodied AI, and robotics
  reviewers. Keep terms defined and claims bounded.

## Controlling claim for this draft

The completed paper reports a frozen, measured MANC/VNC rate core transferred
through a fixed motor map to one non-fly 18-DoF hexapod. Real wiring preserves
structured motor drive relative to declared nulls, but the direct zero-learning
interface does not produce the declared canonical alternating tripod endpoint.

The four-latent translation layer is a planned follow-on architecture, not a
result of this paper. Stage 2 PPO has not run. Do not report walking, balance,
target-reaching, or performance results for that stage.

## Term map

- MANC: male adult nerve cord connectome.
- VNC: ventral nerve cord; define at first use when separated from MANC.
- Frozen core: fixed signed MANC/VNC graph and rate-model parameters.
- Fixed motor map: antagonist-aware conversion from 260 motor outputs to 18
  hexapod joint targets.
- Translation layer: future trainable observation encoder and motor decoder
  surrounding the frozen core; not part of the completed zero-learning result.
- Canonical alternating tripod: within-tripod near-phase and cross-tripod
  approximately half-cycle phase relation.
- One configured body: current robustness result; never call it morphology
  generalization.

## Verified claims and results

- Complete published T1/T2/T3 coordination inventory: 212/212 candidates.
- All-leg graph: 11,000 nodes, 485,074 signed nonzero edges, 260 motor outputs.
- T1 graph: 4,418 nodes, 144,214 signed edges.
- T1 rhythm: dominant peak 2.467 Hz, autocorrelation frequency 2.500 Hz,
  autocorrelation peak 0.342, opposing-pool correlation -0.321, left/right
  correlation +0.031.
- Tripod phase: within-tripod correlations -0.057 and +0.307, cross-tripod
  correlation -0.185; analytic phase gate accepts 3/6 within and 3/9 cross.
- Strict amplitude-matched forward result: real 0.224070 m versus matched
  random -0.011718 m; difference +0.235788 m, across-seed CI
  [+0.186808,+0.280522], exact one-sided p=0.000244; 0/12 controls match/beat.
- Turning: real odd yaw 7.4408 degrees; difference +6.9155 degrees, CI
  [+4.8919,+8.8995], exact one-sided p=0.000244; 0/12 controls match/beat.
- Body sweep: nine declared perturbations, mean difference +0.23464 m, CI
  [+0.18702,+0.27648], worst-case +0.24159 m, CI [+0.19149,+0.28620]; all
  nine paired comparisons pass.
- Rewiring dose: forward means [0.2241, 0.0435, 0.0282, 0.0566, 0.0159,
  -0.0087] m at alpha [0, .1, .25, .5, .75, 1]; Spearman rho=-0.829;
  full-rewiring loss +0.2327 m, CI [+0.2101,+0.2538], Holm p=0.00122.
- Fixed-input feedback branches remain phase-negative; report only if space
  permits and label them fixed anatomical branches, not learned repairs.

## Excluded or future evidence

- v2/v3 translation-layer results are inconclusive and defective; omit.
- v4 Stage 1 teacher metrics are not walking performance; omit from main claim
  unless needed to explain the planned follow-on.
- v4 Stage 2 PPO has not run; no outcome exists.
- T1/T2 trained-readout and H5 lesion results are outside the compact main story
  unless a reviewer requests them.
- Do not claim first, novel, biological gait replication, causal circuit
  mechanism, morphology generalization, or successful prey pursuit.

## Figure plan

1. Overview pipeline: Drosophila/connectome evidence -> frozen MANC/VNC ->
   fixed map or future translation layer -> 18-DoF hexapod -> endpoints.
   Use a real rollout snapshot and an evidence-derived sparse connectome view.
   Solid path is completed; dashed path is planned.
2. Keep compact result figures for amplitude-matched body robustness and
   rewiring dose. Use captions to state what each graph does and does not show.
3. Preserve source result images and label MuJoCo output as simulation, not a
   physical robot photograph.

## Voice target

Use direct, explanatory sentences similar to the author's RP paper: define the
problem, state what was used, describe what the result shows, then state the
limitation. Keep technical nouns and measured values, but remove unsupported
superlatives, vague novelty language, and claims that exceed the endpoint.
