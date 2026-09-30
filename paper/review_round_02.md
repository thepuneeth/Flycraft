# Paper review round 02 — final closure audit

Date: 2026-09-04  
Scope: `paper/flycraft_bibm_draft.tex`, bibliography, figures, canonical Path 2 records, and compiled PDF.

## Closure verdict

Scientific closure: no unresolved critical or major issue identified after the independent BIBM, voice, domain, statistics, and figure reviews. One domain audit caught a paper/code mismatch in an earlier draft: the paper had declared a leg/world sign transform that the result runners did not use. The paper now documents the actual direct native-joint-order interface and default-state-plus-home-settling initialization; no body results were rerun because the result-generating code was already unchanged. The paper makes a narrow claim: a frozen MaleCNS-v1.0-derived VNC graph produced graph-dependent motor drive under the tested fixed map, body, and null models.

Submission metadata: pending human completion. The author block still contains `Puneeth` and `High School Student Researcher`; exact full name, school or organization, affiliation, author order, and optional email must be supplied before submission.

## Panel disposition

| Review lens | Critical | Major | Resolution |
|---|---:|---:|---|
| BIBM format/reproducibility | 0 | 0 scientific | Five-page IEEEtran draft; canonical `bodyfix_*` records and local claim map used. Human metadata remains open. |
| Voice/readability | 0 | 0 | VNC, coxa, neuromere, MaleCNS release, MJCF, and MuJoCo terms defined or expanded at first use; claims shortened and bounded. |
| Domain validity | 0 | 0 | Full-CNS release, VNC closure, premotor scope, node counts, six injected input recipients, native joint order, initialization, and fixed map are stated. |
| Statistics | 0 | 0 | E3 uses the stated minimum-over-labels comparison; H4 is descriptive; sign-flip assumptions and conditional units are stated. |
| Figures | 0 | 0 | Figure labels, source/output distinction, 300-dpi raster export, and preview were refreshed; no material overlap found. |

## Claim → analysis → raw output

- Matched forward contrast (+0.236 m): `eon-fly-brain/code/audit_amplitude_control.py` → `eon-fly-brain/results/path2/h1_amplitude_control_strict.json`.
- Odd-yaw contrast (+6.916 degrees): `eon-fly-brain/code/audit_e1_turning.py` → `eon-fly-brain/results/path2/e1_turning.json`.
- Neural coxa-phase proxy: `eon-fly-brain/code/probe_vnc_phase.py` → `eon-fly-brain/results/path2/bodyfix_vnc_phase_probe_e0.json`.
- Five-condition 3-s paired study: `eon-fly-brain/code/analyze_control_study.py` → `eon-fly-brain/results/path2/bodyfix_vnc_control_study_3s.csv` and `bodyfix_vnc_control_study_3s_stats.json`.
- One-body parameter sensitivity: `eon-fly-brain/code/audit_e3_body_generalization.py` → `eon-fly-brain/results/path2/e3_body_generalization_amplitude_matched.json`.
- Rewiring dose and descriptive fall rate: `eon-fly-brain/code/audit_h4_dose_response.py` → `eon-fly-brain/results/path2/h4_dose_response.json`.
- Translation-layer status: `eon-fly-brain/results/path2/embody_3way_v2.json` and `v3_partial_eval.json`; these support an inconclusive pilot and incomplete amended attempt, not a walking or balance result. Partial v3 trajectories are excluded.

## Build and visual gate

- `latexmk` completed without fatal errors; PDF is 5 pages on US Letter.
- No overfull boxes, undefined citations, missing glyphs, or unresolved references reported on the final pass; remaining underfull warnings are nonfatal.
- PDF fonts are embedded Type 1 fonts.
- Figure 1 overlap checker returned 0 errors; 72 heuristic warnings mark tight or intentional graph routes only.
- Citation audit: 11 cited entries, 11 bibliography entries, no missing or uncited entries.

## Remaining human edits

Replace placeholder author metadata. After that, set PDF metadata if desired and perform the venue's final upload check. No public repository URL or commit hash is claimed here; this audit maps the local project bundle used to produce the draft.
