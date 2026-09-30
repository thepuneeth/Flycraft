# Voice and motor-map revision

## Before editing

Scope: main manuscript only; add the fixed motor-map figure (option 2). Keep the new brain/VNC control diagram separate for later use.

Voice sources: academic-humanizer, followed by `citation_and_writing_style_guide.md` in the UGHS project's `final paper version` folder, referenced by the local voice audits. Preserve paragraph coverage, numerical results, equations and citation keys. Use short, process-first sentences and claim → evidence → interpretation.

| Location | Pattern / evidence status | Planned action |
|---|---|---|
| Abstract, Introduction, Conclusion | Figurative “body that no fly has”; broad transfer claim | Name the simulated non-fly hexapod and restrict conclusions to tested interfaces. |
| Methods | Fixed conversion lacks reproducible explanation | Show the actual `mn_to_joint.translate` coefficients; explain signed pool means and fixed gains. |
| Controls | Temporal-variation matching described as eliminating output-scale effects | State what is matched; do not claim that all scale differences disappear. |
| Experiment A, Abstract | “Removing neurons” differs from stated edge-deletion method | Describe removal of outgoing connections from annotated neurons. |
| Experiment B, Abstract | “Monotonic” conflicts with Table II | Say overall decline, retaining every table value and rho. |
| Results D, captions | Dense clauses and unexplained row comparisons | Define the readout briefly and shorten sentence structure. |
| Experiment E, Discussion | Difference within SD treated as “within noise” | Report the same difference and SD without asserting statistical equivalence. |
| Conclusion | Localization and generalization stronger than lesion-size control permits | Preserve findings while retaining lesion-size and tested-body bounds. |

## Existing evidence issues, outside this writing edit

- `robustness_rewiring.pdf` contains older rate-model outcomes (its generator reads the unsuffixed E3 and H4 records), whereas the current main tables report LIF outcomes. The revised caption and text identify that distinction explicitly. Plotted values are unchanged. Regenerate from matching LIF artifacts before submission if this figure is intended to support the current tables.
- The technical notes describe an earlier rate model. Check their equations/settings against the current LIF implementation before calling them a complete specification of the current experiment.
- Existing numerical results and references are retained, not independently revalidated by this voice pass.

## New figure checks

The map is extracted from the actual fixed translation function by applying unit motor vectors with zero stance. Random-vector and nonzero-stance checks verify the resulting affine map. No dynamics or performance results are inferred from this visualization. Anatomical source pixels remain unchanged; brain rendering is context, not a claim that the whole brain is simulated.

## Completed changes and verification

- Applied academic-humanizer first (claims, clause length, paragraph coverage), then the saved author guide (active decisions, process-first explanations, concrete findings, narrow conclusions).
- Rewrote prose throughout the main paper, retaining its sections and paragraph coverage. Added one Methods paragraph explaining the actual motor map. Shortened experiment headings and figure captions.
- Preserved all original numerical tokens and all citation occurrences against `/tmp/flycraft_bibm_draft.before-map-voice.tex`. New numeric content is limited to the map description, leg labels, repeated map dimensions, and attribution. No original equations or table values changed.
- Added only the requested motor-map content to the manuscript, placed at the top of page 2 with its Methods explanation directly below. The separate brain/VNC control diagram remains outside the paper.
- Moved the map float before the bibliography. The bibliography now starts only after the final walking figure, so no figure is interleaved with reference entries.
- The map generator verifies 18 × 260 coefficients on 20 random activity vectors and confirms all cross-leg coefficients are zero. PDF and PNG use the same matrix; CSV records exact coefficients and channel identities.
- Main PDF remains five pages. All pages and both new figures inspected visually; no LaTeX overfull/underfull boxes or unresolved references in the checked build.
- This is a voice and visualization pass, not a new literature review, statistical reanalysis, or submission-compliance certification.
