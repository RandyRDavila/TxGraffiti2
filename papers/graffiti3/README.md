# Experiments and Paper Artifacts

This subdirectory houses **reproducible experiment notebooks** (used to generate results, figures, and tables for the TxGraffiti manuscript) and the **Calabi–Yau fully automated note (PDF)** referenced in the paper.

The intent is practical reproducibility: each notebook should run end-to-end and write the same kinds of artifacts used in the manuscript (run reports, tables, plots, and audit summaries).

---

## What’s in here

### 1) Notebooks (`*.ipynb`)
Jupyter notebooks that generate the experiments and paper-facing artifacts. Notebooks typically:
- load or build a dataset (often with caching)
- run a Graffiti-style conjecturing/auditing pipeline
- export artifacts (tables/figures/run reports) to disk

Notebooks are usually grouped by **domain/case study** (graphs, groups, integers, knots, Calabi–Yau, etc.).

### 2) Calabi–Yau note (PDF)
This directory also includes the Calabi–Yau “fully automated note” PDF referenced in the manuscript.

> If you rename or move the PDF, update any references in the paper and in this README.

---

## Quick start

From the repository root:

```bash
jupyter lab
```

Then open the notebooks in this directory.
