# Unified LaTeX source

This source package contains only publication-facing material.

## Root files

- Manuscript: `manuscript/main_clean.tex`
- Response to the Editor and Reviewers: `response/response_to_editor_and_reviewers.tex`
- Cover letter: `cover/cover_letter.tex`
- Supplementary information: `supplementary/supplementary_information.tex`
- Main and supplementary figures: `figures/`

The clean manuscript is the authoritative LaTeX manuscript. The marked revision is supplied separately as a Word Track Changes document because whole-document color marking is not an appropriate substitute for tracked revisions.

Compile each root file from its own directory. The manuscript uses BibTeX through `natbib`; the remaining documents compile directly. All four root files have been compiled and verified in this package.
