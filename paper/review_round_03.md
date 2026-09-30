# Paper review round 06 — final redraft gate

Date: 2026-09-04  
Scope: `paper/flycraft_bibm_draft.tex`, bibliography, figures, canonical Path 2 result records, and compiled PDF.

## Verdict

No unresolved internal critical or major scientific, semantic, voice, or visual issue remains in the current five-page draft. The paper makes the narrow claim supported by the records: a frozen MaleCNS-v1.0-derived VNC graph produces graph-dependent motor drive through one fixed motor map and one configured hexapod under specified null models. The neural coxa-phase proxy fails, so the draft does not claim biological gait transfer. Stable artifact access and the remaining school/author-order metadata remain external submission requirements.

Submission metadata is partly complete. The author block now uses `Puneeth Pradeep` and the verified email from the prior manuscript; school or organization, exact affiliation wording, and author order remain human-supplied.

## Official UGHS compliance check

Checked 2026-09-04 against the [IEEE BIBM 2026 UGHS page](https://bibm2026-hs.github.io/). The page lists bioinformatics, biomedical data analysis, graph data, and biomedical imaging among its scope; this paper uses a measured neural graph and physics-based body simulation. It specifies a five-page maximum for high-school papers, including figures, tables, and references; requires the author affiliation to identify high-school or undergraduate status; and lists the review as single-blind. The current draft is five pages, includes `High School Student Researcher` in the affiliation, and shows authors. The page lists September 7, 2026 at 11:59 PM AoE as the paper deadline; its camera-ready date remains a placeholder, so recheck after acceptance. The page also prohibits dual submission and assigns authors responsibility for authorship, data integrity, and citations; those checks require author confirmation before upload.

## Independent review disposition

| Review lens | Critical | Major | Current disposition |
|---|---:|---:|---|
| BIBM UGHS format and scope | 0 | 0 | IEEEtran letter draft is five pages; school/author-order metadata remains open. |
| Figure geometry and print readability | 0 | 0 | Actual connectome excerpt and MuJoCo hexapod frame remain visible; both figures were regenerated or rebuilt and all pages were rendered. |
| Statistics and provenance | 0 | 0 | Canonical JSON values and endpoint definitions match the paper. H4 clustered falls are descriptive without an episode-level interval. |
| Semantic/domain review | 0 | 0 | Title names the tested proxy; v2/v3 49-D pilot status is separated from the untested 66-D layer; one-body scope is explicit. |
| Voice and mechanical review | 0 | 0 | Abstract and introduction use direct, claim-first prose; final gate has no banned construction except factual M2 negations. |

### Issues found and fixed in this loop

- The H4 `25/240` fall rate was clustered by rewired seed. The episode-level Wilson interval was removed, and the rate is labeled descriptive.
- The E3 mean margin was corrected from a rounded `0.234638` to `0.235` m.
- The E1 tail statement now identifies an exact one-sided conditional sign-flip test.
- The title now names the prespecified neural coxa-phase proxy instead of implying a universal neural null.
- The paper distinguishes 212 selected candidate cells from 260 motor-output channels.
- The planned translation paragraph now states the same 49-D joint-position/joint-velocity (qpos/qvel) interface used by the incomplete v3 attempt, expands Proximal Policy Optimization (PPO), and names the auxiliary value critic. It reports no 66-D walking, balance, or target-reaching result.
- The rate-model methods now identify NumPy seed 1, independent normal heterogeneity draws, and the implementation's lower clipping bounds; the primary graph-null endpoint and conditional inference wording are consistent.
- The H4 rewiring-dose wording now defines $\alpha$ as accepted swaps divided by half the edge count, matching the canonical construction code; the plot axis uses `edge-swap fraction`.
- Added a public-data statement covering the absence of new human or animal subjects, clarified that all 260 motor channels feed the fixed map while SD matching occurs in 18 joint-target dimensions, and identified the coxa phase trace as separate from the odd-yaw drive.
- Split the long methods blocks into protocol, inference, and provenance units; retained the full E3 uncertainty intervals while shortening repeated future-work wording.
- Rebuilt Figure 1 with a tighter, higher-contrast crop of the actual hexapod frame and a context crop of the actual MaleCNS/VNC graph that preserves T1--T3 and motor-output labels; the underlying simulation and graph image content is unchanged.
- Regenerated the standalone Figure 1 preview from the current compiled figure so the review bundle matches the embedded artwork.
- Figure 1 uses direct left-to-right rails, explicit completed/planned styling, an actual evidence-derived connectome context crop, an actual simulated hexapod frame, and readable forced line breaks.
- Figure 2 replaces broad `robustness` and `factor` labels with measured forward-displacement and parameter-value wording. Removing the forced bibliography break and shortening repeated caption/conclusion wording keeps all 11 references on page 5.
- Added nonbreaking spaces before all 15 LaTeX citation commands, split the remaining overlong results sentence, changed passive wording to active forms, and completed the final voice pass.

## Inspiration-figure audit

The local C. elegans connectome-robot paper separates neural dynamics, body, and behavior into interpretable stages. The local FlyGM paper uses an overview rail from observation to neural state, graph computation, and action, then follows it with compact behavior snapshots and representation plots. The original RP paper uses an early overview figure and dense but direct quantitative panels. Figure 1 applies this shared visual logic to this study: measured graph → frozen core → fixed map → hexapod → endpoints, with the proposed 66-dimensional translation layer isolated as a dashed future path. Figure 2 keeps completed measurements in two compact result plots. No unrun translation-layer outcome enters either figure.

## Claim → analysis → raw output

- Matched forward contrast: `eon-fly-brain/code/audit_amplitude_control.py` → `eon-fly-brain/results/path2/h1_amplitude_control_strict.json`.
- Odd-yaw contrast: `eon-fly-brain/code/audit_e1_turning.py` → `eon-fly-brain/results/path2/e1_turning.json`.
- Neural coxa-phase proxy: `eon-fly-brain/code/probe_vnc_phase.py` → `eon-fly-brain/results/path2/bodyfix_vnc_phase_probe_e0.json`; Pearson coxa correlations and body-map tracking → `eon-fly-brain/results/path2/all_legs_coordination.json`.
- Five-condition 3-s study: `eon-fly-brain/code/analyze_control_study.py` → `eon-fly-brain/results/path2/bodyfix_vnc_control_study_3s.csv` and `bodyfix_vnc_control_study_3s_stats.json`.
- One-body parameter sensitivity: `eon-fly-brain/code/audit_e3_body_generalization.py` → `eon-fly-brain/results/path2/e3_body_generalization_amplitude_matched.json`.
- Rewiring dose: `eon-fly-brain/code/audit_h4_dose_response.py` → `eon-fly-brain/results/path2/h4_dose_response.json`.
- Translation-layer status: `eon-fly-brain/results/path2/embody_3way_v2.json` and `v3_partial_eval.json`; these support an inconclusive 49-D pilot and incomplete amended attempt, not a walking or balance result.

## Numerical spot checks

- Graph inventory: 11,000 closure nodes, 485,074 signed edges, 212 selected cells across 62 of 66 labels, and 260 mapped motor-output channels.
- H1: 0.224070 m measured mean versus -0.011718 m null mean; difference +0.235788 m, paper value +0.236 m; seed interval [+0.186808, +0.280522] m; exact conditional sign-flip `p=0.000244`.
- E1: 7.440824 degrees measured odd yaw; difference +6.915546 degrees, paper value +6.916 degrees and +6.92 degrees in the abstract/figure; seed interval [+4.891895, +8.899470] degrees; exact conditional sign-flip `p=0.000244`.
- E3: mean margin +0.234638 m, paper value +0.235 m; worst-case margin +0.241590 m, paper value +0.242 m; all nine labeled runs are positive, while three nominal labels share one body setting.
- E3 intervals: equal-weight margin [0.187, 0.276] m and worst-case margin [0.191, 0.286] m, both conditional 95% null-seed intervals.
- Phase proxy: 3/6 within-tripod pairs accepted and 3/9 cross-tripod pairs accepted; correlations are -0.057, +0.307, and -0.185.
- H4: means at alpha `[0, .1, .25, .5, .75, 1]` are `[0.2241, .0435, .0282, .0566, .0159, -0.0087]` m; Spearman rho is -0.829; 25/240 falls at alpha 0.1 are descriptive because episodes share rewired seeds.

## Build and visual gate

- `latexmk -pdf -interaction=nonstopmode -halt-on-error flycraft_bibm_draft.tex` completed successfully.
- PDF: 5 pages, US Letter (612 x 792 pt), 2,580,427 bytes; PDF title and author metadata are populated.
- No fatal errors, undefined control sequences, missing characters, or overfull boxes. TeX reports one underfull abstract hbox; visual inspection found no clipping or malformed text.
- The audit-only reference verifier reports 11/11 `OK`, no duplicate findings, `submission_safe=true`, and `fully_verified=true`; it wrote `paper/qc/reference_audit.json` without modifying the bibliography.
- `pdffonts` reports embedded Type 1 fonts. `pdfimages -list` reports the Figure 1 connectome context crop at 2100 x 1000, tight hexapod frame at 400 x 300, and Figure 2 at 2250 x 840.
- All five PDF pages were rendered at 180 dpi in the final pass. Figure 1 and Figure 2 have no clipping, missing labels, or unreadable endpoint values; pages 4--5 were rechecked after the final caption and conclusion edits. The last-page columns remain materially close in length.
- Standalone Figure 1 compiles to one page and the TikZ validator reports 0 errors and 9 heuristic warnings. Warnings flag tight connector gaps and lower-box spacing; visual inspection shows intentional non-overlapping layout. The optional PDF overlap checker was not run because the available Python lacks `pdfplumber` and `PyMuPDF`.
- Figure 2 was regenerated from the canonical JSON records using a temporary `uv` runtime; no canonical result JSON changed.

## Mechanical gate

| Check | Hits | Disposition |
|---|---:|---|
| M1 em dash | 0 | Clean |
| M2 rhetorical negation pattern | 22 | All are factual scope, limitation, or comparison statements; no rhetorical decoration remains |
| M3/M4/M8/M9 editorializing, hype, grand setup, metaphor | 0 | Clean |
| M5/M16 banned or pompous words | 0 | Clean |
| M6 throat-clearing opener | 0 | Clean |
| M10 vague mechanism/futurist verb | 0 | Clean |
| M11 passive voice | 0 | Clean |
| M12 needless wording | 0 | Clean |
| M13 weak qualifier | 0 | Clean |
| M14 coined adverb/false ordinal | 0 | Clean |
| M15 exclamation/rhetorical question | 0 | Clean |
| M17 figurative verb | 0 | Clean |
| M18 content-free opener | 0 | Clean |
| Part B precision pairs | 0 | Clean |

Citation audit found 11 cited keys, 11 bibliography keys, no missing keys, and no uncited entries. Citation-spacing, duplicate-label, term-drift, and decomposition-cardinality checks returned no issues.

## Remaining human edits

Replace remaining school/organization and author-order metadata. Then perform the venue's final upload and anonymization check if BIBM requests one. The current paper intentionally does not claim a walking or balance result for the proposed 66-D layer, biological gait transfer, morphology generalization, or causal circuit mechanism.
