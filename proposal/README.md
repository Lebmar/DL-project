# Proposal draft

The rendered draft is `../output/pdf/project_proposal.pdf`, one page including references. It is a planning draft for student review, not a submitted assessment or certified novel contribution. The author line uses only the known first name; replace it with the required official student name before submission.

The original `neurips_2026.sty` is unmodified. It was downloaded from the [official NeurIPS 2026 formatting archive](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip), linked by the [official call for papers](https://neurips.cc/Conferences/2026/CallForPapers). We use the `preprint` option, numbered sections and standard margins/fonts. No layout squeezing is used.

Build with TeX Live:

```bash
cd proposal
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf project_proposal.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf project_proposal.tex
```

The two passes resolve citations. The PDF was rendered and visually checked on 2026-10-11; there are no undefined citations or overfull boxes in the final build. The course guideline requires a one-page PDF by October 16, 23:59. The student confirmed on 2026-10-11 that AI tools are permitted; the student remains responsible for understanding and revising the work.
