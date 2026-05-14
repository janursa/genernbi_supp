


# Supplementary Code for genernbi_supp

This repository provides supplementary code for the **genernbi_supp** manuscript.  
For more details on genernbi_supp, visit the main repository:  
[**github.com/openproblems-bio/task_grn_benchmark**](https://github.com/openproblems-bio/task_grn_benchmark).

## Setup Instructions

To run this repository, follow these steps:

1. **Download genernbi_supp** and place it alongside the main repository (`task_grn_inference`).
2. **Download the results** within `task_grn_inference` repo using the following command:

   ```bash
   aws s3 sync s3://openproblems-data/resources/grn/results/ resources/results --no-sign-request
   ```

3. **Configure paths** regenerate `env.yaml` (used by all Python scripts):

   ```bash
   bash generate_env_yaml.sh
   ```

## Repository Structure

### `src/stability_analysis/`
Experiments that probe the robustness and reliability of GRN inference metrics and methods. Each sub-folder is a self-contained experiment with a runner script and a post-processing script that produces publication figures.

| Sub-folder | Purpose |
|---|---|
| `causal_directionality` | Tests whether evaluation metrics are sensitive to the direction of regulatory edges (TF → target vs. target → TF). |
| `causality` | TF-masking experiment: checks whether regulatory relationships masked with known TFs outperform those without masking. |
| `gb_vs_ridge` | Compares gradient-boosting vs. ridge as core models used in the regression metrics. |
| `metrics_stability` | Checks whether the regression and Wasserstein-distance metrics reliably separate good GRN models from a poor ones by looking at it's stability across perturbations and donors as covariates.
| `normalization` | Investigates the effect of data normalisation strategies on GRN inference performance. |
| `permute_grn` | Permutation analysis of GRN models. |
| `piecewise_analysis` | Gene- and TF-level breakdown of metric scores: compared top performer to low peformers using gene and TF wise scors; identifies "easy" vs. "hard" genes and TFs and categorize them; and evaluates gene expression and variabilty as well as TF pertubration strength as factors of performance |
| `skeleton` | GRN inference performance analysis after filtering with the skeleton (a loose putative prior TF-gene connections built using TF motifs and chromatin accessibility). |

### `src/data_aggregation/`
Experiments evaluating how data aggregation affect GRN inference performance.

| Sub-folder | Purpose |
|---|---|
| `metacell` | Tests the effect of metacell aggregation granularity on GRN inference. |
| `bulk_vs_sc` | Compares GRN inference performance when using bulk RNA-seq vs. single-cell expression data. |

### `src/global_grns/`
Evaluates context-specific vs. context-agnostic (global) GRN models across datasets.

### `src/imputation/`
Tests how expression data imputation methods (KNN imputation, MAGIC) affect GRN inference and evaluation scores across datasets.

### `src/MMP/`
GRN analysis pipeline for the Multiple Myeloma Precursor (MMP) dataset. Covers preprocessing (peak merge, QC, CellTypist annotation, protocol split), GRN inference with five methods (SCENIC+, CellOracle, GRaNIE, GRNBoost2, PPCOR) across two protocols (10x Multiome and DOGMA-seq), and network topology analysis.

## Authors
Jalil Nourisa
Antoine Passemiers