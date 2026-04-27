# experimental_grn

Benchmarks the effect of integrating **promoter motif** and **ATAC-seq chromatin accessibility** priors into GRN inference, across a C-axis of 7 filtering strategies (C0–C6) and 6 algorithms, evaluated on 10 datasets.

---

## C-axis (filtering strategies)

| Tag | Description |
|-----|-------------|
| C0 | No filter — raw top-50k edges |
| C1 | Motif skeleton ±1 kb of TSS |
| C2 | Motif skeleton ±100 kb of TSS |
| C3 | ATAC peaks ±1 kb of gene body *(ATAC datasets only)* |
| C4 | ATAC peaks ±100 kb of gene body *(ATAC datasets only)* |
| C5 | C4 + peak–gene correlation filter (r ≥ 0.05, p < 0.05) *(ATAC datasets only)* |
| C6 | C2 ∩ C4 — motif AND ATAC *(ATAC datasets only)* |

**Algorithms:** pearson, spearman, lasso, ridge, elasticnet, grnboost  
**Datasets:** op, ibd_uc, ibd_cd, 300BCG, nakatake, norman, replogle, parsebioscience, xaira_HEK293T, xaira_HCT116  
**ATAC datasets** (C3–C6 applicable): op, ibd_uc, ibd_cd

---

## Folder structure

```
grn_inference/      Stage 1 inference + C-tag prior integration
consensus/          Build consensus GRN (regulators + ws weights) per dataset
grn_evaluation/     Submit evaluation jobs → score files
results/            Analysis scripts and output plots
```

---

## Scripts

### `grn_inference/`

- **`run_jobs.py`** — submits sbatch jobs for Stage 1 inference (pearson/spearman/lasso/ridge/elasticnet). Outputs ~1M raw edges per algo×dataset to `predictions/`.
- **`submit_grnboost.py`** — same as above for grnboost (separate due to different compute requirements).
- **`stage1_infer.py`** — core inference: reads expression data, runs the chosen algorithm, writes raw ranked edges.
- **`stage1_grnboost.py`** — GRNBoost2 core logic called by `submit_grnboost.py`.
- **`run_apply_prior_jobs.py`** — submits one sbatch job per algo×dataset×c_tag. Covers C0–C2 for all datasets and C3–C6 for ATAC datasets.
- **`apply_prior.py`** — core prior integration: reads raw predictions, applies the requested C-tag filter (motif skeleton / ATAC accessibility / correlation), and writes the top-50k filtered edges to `predictions_final/`.

### `consensus/`

- **`submit_consensus.sh`** — submits one consensus job per dataset.
- **`build_consensus.py`** — builds the consensus GRN (regulators list + Wasserstein weights) needed by the evaluator.

### `grn_evaluation/`

- **`run_eval.py`** — submits one sbatch evaluation job per file in `predictions_final/`. Scores are written to `scores/` as `.score.h5ad` files.

### `results/`

- **`heatmap_analysis.py`** — loads all score files and produces three heatmaps: (1) algorithm × C-tag overall score, (2) algorithm × metric, (3) metric × C-tag. Normalization is global per (dataset, metric) across all algo+c_tag combinations.
- **`plot_algo_strategies.py`** — produces 6 per-algorithm bar plots (one per algo) showing raw scores per metric grouped by C-tag, with individual dataset dots.
