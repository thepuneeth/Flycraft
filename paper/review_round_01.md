# Paper review round 01

Date: 2026-09-03  
Scope: `flycraft_bibm_draft.tex`, `flycraft_bibm_draft.bib`, and generated figures  
Target: BIBM UGHS, five-page IEEE conference manuscript

## Panel dispositions

| Lens | Main finding | Disposition |
|---|---|---|
| Voice and humanization | Draft should stay direct, explanatory, and evidence-bound; avoid generic contribution language and claims stronger than controls support. | Applied. Preserved reported values and narrowed biological claims. |
| BIBM UGHS fit | High-school affiliation and five-page limit are required; figure and reference pages count. | Applied. Generic high-school affiliation is used pending school name; final PDF is five pages, letter size. |
| Connectome/domain | Treat coxa phase as a proxy; 212/212 means completeness under the fixed published rule only; nulls support a narrow graph-dependent effect, not a unique mechanism. | Applied throughout abstract, methods, results, and discussion. |
| Reproducibility/statistics | State endpoint hierarchy, seed counts, shared starts, matching procedure, and test families. | Applied in Controls and Endpoints; excluded unrun PPO results. |
| Adversarial estimand review | Separate candidate-cell count from graph size, seed-level null contrasts from episode-level paired tests, VNC phase from body phase, and fixed targets from learned targets. | Applied. Third 3-second comparison, endpoint definitions, E3/E4 run units, and control-seed intervals are now explicit. |
| Figures | Add an intuitive process figure, use actual connectome and simulated hexapod assets, avoid implying physical hardware or morphology generalization. | Applied. Solid completed path and dashed planned path are separated in Fig. 1; Fig. 2 replaces unordered category lines with grouped settings. |

## Figure 1 visual gate

The figure contract is: measured MANC/VNC connectivity excerpt -> frozen rate core -> fixed motor interface -> MuJoCo hexapod -> completed endpoints, with a separate dashed future translation-layer strip. The connectome excerpt is data-derived from the project CSR graph and the hexapod panel is a simulation frame; neither is presented as a physical robot or a spatial connectome reconstruction.

Checklist against the rendered `overview_pipeline.pdf`:

- Philosophy: pass. Reader sees completed solid path versus unrun dashed interface; measured graph and rollout assets prevent a generic box-only diagram.
- T1/T3/T4/T5: pass. Main path reads left-to-right, source/core/interface/body/endpoints have distinct roles, and the future strip is visually bounded.
- S1/S6/S8/S9: pass. Text is readable at the target full-width placement; boxes have consistent styling; geometry checker reports zero errors; image remains legible after raster preview.
- M1/M2/M3/M8: pass or not applicable. Box hierarchy, endpoint scorecard, and actual image anchors are clear; no repeated hero graphic is needed.
- E1/E2/E3/E9/E12: pass. Arrows terminate at the intended stages, solid/dashed semantics are explicit, and the remaining line crossings are intentional routing into the frozen core.
- A1/V1: pass. Color is paired with line style and labels; values and assets are visible without relying on color alone.

Automated geometry output: `error_count=0`; 21 warnings remain for tight text spacing or intentional line crossings. No warning identifies a text collision or clipped node. The linter's low element-count and two-line-type advisories are accepted because this is a medium-density pipeline with one completed path and one planned path.

## Mechanical checks

- Main PDF: 5 pages, US letter (`612 x 792` pt); all five rendered pages visually inspected, including complete conclusion and references.
- Main LaTeX build: succeeds with `latexmk -halt-on-error`.
- Main log: no undefined citations, fatal errors, missing characters, or overfull boxes; two non-fatal underfull boxes remain.
- Figure log: no overfull/underfull boxes, missing characters, or fatal errors.
- Source bundle: figure inputs used by the manuscript are inside `paper/figures/`; the main source no longer depends on the external results directory.
- Stage 2 PPO: not run and not reported as a result.

## Change log

- Reframed paper around wiring-specific motor drive versus canonical alternating-tripod coordination.
- Added actual data-derived MANC/VNC connectivity excerpt and MuJoCo hexapod rollout frame.
- Added closest Drosophila whole-brain locomotion precedent without broad novelty language.
- Added planned four-latent translation layer as a dashed, explicitly unrun branch.
- Added strict amplitude matching, null-model wording, fixed-input probe methods, endpoint hierarchy, and sample-size/statistical details.
- Replaced old robustness and rewiring panels with one compact two-panel figure.
- Removed redundant transfer-audit panel to hold the five-page limit after adding statistical clarifications.
- Kept `bodyfix_transfer_layers.png` in the self-contained figure directory as an optional audit asset.

## Remaining human edits

Replace `Puneeth` and `High School Student Researcher` with the exact author and school wording required for submission. Confirm final author order, contact details, and any BIBM-specific source-upload fields before submission.
