# stability_analysis

This folder contains experiments and post-processing scripts that probe the **robustness and reliability** of GRN inference metrics and methods. Each sub-folder is a self-contained experiment with its own runner script (`script.py` / `experiment_*.sh`) and a post-processing script (`post_*.py`) that produces figures saved under `results/figs/`.

## Sub-folders

| Folder | Purpose |
|--------|---------|
| `causal_directionality` | Tests whether evaluation metrics are sensitive to the direction of regulatory edges (TF → target vs target → TF). |
| `causality` | TF-masking experiment: evaluates whether metrics correctly reward GRNs that capture known causal TF–target relationships. |
| `causality_ws_distance` | Variant of the causality experiment using Wasserstein distance to quantify distributional shifts under TF masking. |
| `gb_vs_ridge` | Compares gradient-boosting (GRNBoost2) vs. ridge regression as GRN inference backbones on the same evaluation datasets. |
| `global_grns` | Evaluates context-specific vs. context-agnostic (global) GRN models across datasets. Plots sensitivity and ctx-vs-nonctx heatmaps. |
| `merged_plots` | Combined figures that overlay results from multiple experiments into a single panel for publication. Scripts: `data_aggregation_heatmaps.py` (Metacell + Imputation:KNN + Imputation:MAGIC heatmaps), `causal_directionality_heatmaps.py` (methods×datasets + methods×metrics heatmaps). |
| `metrics_applicibility` | Assesses which evaluation metrics are applicable per dataset (based on data availability and statistical criteria). |
| `metrics_stability` | Measures how stable metric scores are across datasets and GRN methods; produces summary statistics tables. |
| `normalization` | Investigates the effect of count normalisation strategies on GRN evaluation scores. |
| `op_network` | Analysis specific to the OPSCA (OpenProblems) network topology and method comparisons. |
| `permute_grn` | Permutation-based null model experiment: shuffles GRN edges and checks that metrics correctly penalise random networks. |
| `piecewise_analysis` | Donor-level cross-validation and piecewise stability analysis to assess how scores vary across biological replicates. |
| `repo` | Shared utilities and helper resources reused across multiple stability experiments. |
| `skeleton` | Template / scaffold experiment used as a starting point for new stability analyses. |

## Conventions

- **Runner scripts** (`script.py`, `experiment_*.sh`): submit or execute the actual benchmarking job.
- **Post-processing scripts** (`post_*.py`): load results CSVs and generate publication-ready figures.
- All output figures are written to `task_grn_inference/resources/results/figs/` (or sub-directories thereof).
- IBD datasets (`ibd_cd`, `ibd_uc`) are excluded from all plots; MSCIC is included where data is available.
