# MMP — src/MMP

Scripts for the Multiple Myeloma Precursor (MMP) GRN analysis pipeline.

## Structure

```
MMP/
├── preprocess/
│   ├── script.py          — Full preprocessing: bedtools peak merge + QC + CellTypist annotation
│   │                        + normalization + protocol split (multiome / dogma)
│   └── run_preprocess.sh  — SLURM sbatch wrapper (200 GB, 16 CPU)
│
├── grn_inference/
│   ├── run_scenicplus_{dogma,multiome}.sh   — SCENIC+ (chromatin-aware, cistopic)
│   ├── run_celloracle_{dogma,multiome}.sh   — CellOracle (motif-based)
│   ├── run_granie_{dogma,multiome}.sh       — GRaNIE (Hi-C + motif + RNA)
│   ├── run_grnboost_{dogma,multiome}.sh     — GRNBoost2 (gradient-boosted trees)
│   └── run_ppcor_{dogma,multiome}.sh        — PPCOR (partial correlation)
│
└── logs/                  — SLURM stdout/stderr logs
```

## Datasets

| Tag        | Protocol    | Batches            | Cells  | Donors          |
|------------|-------------|--------------------|--------|-----------------|
| multiome   | 10x Multiome | SD-2520           | ~12.8k | all donors      |
| dogma      | DOGMA-seq   | SD-2589, SD-2752   | 25,000 | top-3 by count  |

## Outputs

All GRN outputs → `task_grn_inference/resources/results/MMP/MMP_{dataset}_{method}_grn.h5ad`
Each file contains `uns['prediction']`: a DataFrame with columns `source`, `target`, `weight`.
