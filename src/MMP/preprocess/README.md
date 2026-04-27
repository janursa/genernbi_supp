# preprocess

Full MMP preprocessing pipeline in a single script.

## Steps (both run in sequence)

1. **Peak merge** (`merge_peaks`): bedtools consensus merge of raw ATAC outer-join peaks
   (~1.3M → ~271k peaks). Skipped automatically if `MMP_atac_merged.h5ad` already exists.
2. **Preprocessing**: QC, Ensembl→gene-symbol mapping, CellTypist annotation,
   log-normalization, barcode matching, protocol split (multiome / dogma, 25k subsample).

## Usage

```bash
sbatch preprocess/run_preprocess.sh
```

Requires `/vol/biotools/bin/bedtools` (bound into singularity via `--bind /vol/biotools`).
