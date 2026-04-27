# grn_inference

SLURM sbatch wrappers for GRN inference on MMP multiome and DOGMA-seq data.

## Methods

| Script prefix     | Method     | Notes                                      |
|-------------------|------------|--------------------------------------------|
| `run_scenicplus_` | SCENIC+    | Chromatin-aware; cistopic + GRN inference  |
| `run_celloracle_` | CellOracle | Motif-based; requires base GRN             |
| `run_granie_`     | GRaNIE     | Hi-C + motif + RNA; 20k subsample in R     |
| `run_grnboost_`   | GRNBoost2  | Gradient-boosted trees on RNA              |
| `run_ppcor_`      | PPCOR      | Partial correlation on RNA                 |

## Usage

```bash
sbatch grn_inference/run_scenicplus_multiome.sh
sbatch grn_inference/run_celloracle_dogma.sh
# etc.
```

All outputs → `task_grn_inference/resources/results/MMP/MMP_{dataset}_{method}_grn.h5ad`.
SCENIC+ jobs use job-specific temp dirs (`task_grn_inference/temp/scenicplus_{dataset}_{SLURM_JOB_ID}`)
to avoid stale cistopic state across re-runs.
