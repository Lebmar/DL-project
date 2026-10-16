# Proposal draft

## Updated submission option, 16 October 2026

The instructor FAQ supplied by the student clarifies that one page is a minimum, not a maximum; meaningful extra content and appendices are permitted. Following the student's chosen structure, [the recommended submission version](../output/pdf/project_proposal_expanded.pdf) now has **one complete core proposal page, followed by one page containing Appendix A, Appendix B, and references**. [Editable source](project_proposal_expanded.tex) is retained. Page 1 covers the question, method, data, evaluation, contribution, feasibility, and plan independently. Appendix A gives the example and formal design; Appendix B gives preliminary feasibility and remaining analysis. Both pages were rendered and visually checked. It is a proposal, not a claim that all research goals or novelty have already been established.

The core-plus-appendix organization is the student's selected presentation, not a claim that the FAQ prohibits longer main text. The original one-page version below remains unchanged as an alternative. The 11 October compliance review and its manifest preserve the earlier review; their one-page-limit interpretation is superseded by the 16 October FAQ. The official preprint format remains in use. Neither version has been submitted to Canvas by the assistant.

Build the expanded version from this directory with two pdfLaTeX passes, using project_proposal_expanded.tex and the same output directory as below. Provenance and checks are in expanded_review_manifest.json.

The reviewed PDF is `../output/pdf/project_proposal.pdf`, one page including references. The student supplied the author name **Chengxuan YUAN** and student ID **50025581**, which now appear in the PDF. It has not been uploaded to a course submission system; compliance with the stated proposal requirements does not certify novelty or an A+ grade. The guideline does not explicitly require a full-name/ID field; these fields identify the student.

The original `neurips_2026.sty` is unmodified. It was downloaded from the [official NeurIPS 2026 formatting archive](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip), linked by the [official call for papers](https://neurips.cc/Conferences/2026/CallForPapers). We use the `preprint` option, numbered sections and standard margins/fonts. No layout squeezing is used.

Build with TeX Live:

```bash
cd proposal
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf project_proposal.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf project_proposal.tex
```

The two passes resolve citations. The PDF was rendered and visually checked on 2026-10-11; there are no undefined citations or overfull boxes in the final build. The course guideline requires a one-page PDF by October 16, 23:59. The student confirmed on 2026-10-11 that AI tools are permitted; the student remains responsible for understanding and revising the work.

The original guideline's pages 1–2 and 5 were rechecked on 2026-10-11. See [the compliance review](compliance_review.md) for mandatory requirements versus recommendations, and [the artifact manifest](review_manifest.json) for source/PDF/template hashes. The style matches a fresh download of the official formatting archive byte for byte. The bibliography uses the smaller reference font shown in the official example; main-text fonts and margins remain unchanged.
