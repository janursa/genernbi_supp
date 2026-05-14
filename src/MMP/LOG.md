# MMP Processing Log

## Dataset Overview

**GSE311602** — Multiple Myeloma Precursor (MMP) bone marrow multiome (10x Multiome + DOGMA-seq)

| Property | Value |
|----------|-------|
| Samples | 12 (3 batches: SD-2520, SD-2589, SD-2752) |
| Protocols | 10x Multiome (SD-2520) + DOGMA-seq (SD-2589, SD-2752) |
| Cells (raw) | 101,359 |
| RNA genes | 36,601 (Ensembl IDs) |
| Disease | Multiple myeloma precursor (MGUS/SMM/MM) |
| Timepoints | BL (Baseline) / ES (Early Stage) — not resolvable per-cell without HTO demux |
| Accession | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE311602 |

---

## Step-by-Step Work Log

### 1. Dataset Identification & Download
- Searched for publicly available multiome datasets relevant to bone marrow / immune disease
- Selected **GSE311602** (MMP disease) and **GSE194122** (healthy BMMC, NeurIPS 2021 / OpenProblems)
- Downloaded GSE311602_RAW.tar (836 MB) to `/vol/projects/CIIM/agentic_central/temp/multiome_datasets/`

### 2. Inspection of Raw TAR Contents
- TAR contains 24 files: 12 × `adataFinal_*.h5` (processed/annotated, per sample) + 12 × `filtered_feature_bc_matrix_*.h5` (raw 10x counts)
- Raw integer counts confirmed; 3,500–12,000 cells per sample
- Each raw H5 has GEX (36,601 genes, Ensembl IDs) + ATAC (varying per-sample peak set ~100k peaks each)
- **GSM accession quirk**: adataFinal and filtered_feature_bc_matrix files have *different* GSM numbers for the same sample — key parsing strips GSM prefix before matching

### 3. RNA + ATAC Merging (`merge_GSE311602.py`)
- Extracted 24 H5 files from TAR to `/tmp/GSE311602_extract/`
- Matched adataFinal (donor/sample metadata) to count matrix (raw counts) by sample name
- RNA: concatenated all 12 samples → `GSE311602_rna.h5ad` — **101,359 × 36,601**, raw integer counts
- ATAC outer-join: all per-sample peaks merged with `join='outer'` → `GSE311602_atac.h5ad` — **101,359 × 1,319,226** (one column per unique peak across all samples, zeros for missing)

### 4. ATAC Peak Merging Strategy
- Per-sample peak sets differ (each sample processed with own MACS2 call) → outer-join has 1.3M peaks
- Reference datasets use bedtools merge consensus:
  - `op_atac.h5ad`: 135,358 peaks, mean 839 bp (single pseudobulk MACS2)
  - `ibd_atac.h5ad`: 182,284 peaks, mean 947 bp (variable width bedtools merge)
- Decision: **bedtools merge** on union of all per-sample peaks → collapse overlapping intervals into consensus peaks

### 5. Bedtools Merge (`merge_atac_bedtools.py` → `merge_peaks.py`)
- Extracted all peak coordinates from all 12 samples
- Wrote to BED → `sort -k1,1 -k2,2n` → `bedtools merge` (`/vol/biotools/bin/bedtools`, host binary)
- Binary-search interval mapping: each of the 1.3M original peaks mapped to its consensus peak
- Aggregated counts via sparse matrix multiply: `X_merged = X_outer @ M.T` where M is (n_merged × n_orig)
- Result: `GSE311602_atac_merged.h5ad` — **101,359 × 271,667 peaks**, mean 943 bp, 100% peaks mapped

### 6. File Organization
- Renamed and copied raw files to `task_grn_inference/resources/datasets_raw/`:
  - `MMP_rna.h5ad` — RNA, 101,359 × 36,601, raw counts (input to script.py)
  - `MMP_atac.h5ad` — ATAC outer-join, 101,359 × 1,319,226 (input to merge_peaks.py)
- Derived files stored in `task_grn_inference/resources/results/MMP/`:
  - `MMP_atac_merged.h5ad` — bedtools consensus peaks
  - `MMP_rna_processed.h5ad` — QC'd, normalized, cell-typed
  - `MMP_atac_processed.h5ad` — QC'd, standard chromosomes only

### 7. Processing Pipeline (`script.py`)

#### 7a. obs Standardization (`format_obs`)
- `barcode`: raw 10x barcode from `barcode_orig`
- `donor_id`: extracted from original obs (e.g. '1003')
- `batch`: SD-2520 / SD-2589 / SD-2752
- `protocol`: multiome / DOGMA-seq
- `timepoint`: parsed from `sample_id`; stays 'unknown' when sample contains both BL+ES (HTO demux required)
- `disease`: standardized → 'mmp'
- `perturbation`: 'none' (no stimulation)
- `condition`: 'mmp'
- `is_control`: False

#### 7b. RNA QC (`qc_rna`)
- `sc.pp.filter_cells(min_genes=200)` — remove low-complexity cells
- `sc.pp.filter_genes(min_cells=10)` — remove genes detected in <10 cells
- Result: 101,359 → **93,899 cells**, 36,601 → **29,155 genes**

#### 7c. ATAC QC (`qc_atac`)
- `nCount_ATAC` (total fragments) filter: 500 – 100,000
- `nFeature_ATAC` (peaks detected) filter: 200 – 100,000
- No nucleosome_signal/TSS enrichment — not available in raw 10x H5 count matrices
- Result: 101,359 → **99,546 cells**

#### 7d. ATAC Peak Filtering (`format_atac_var`)
- Parsed `chr:start-end` index → `seqname`, `start`, `end`, `ranges`, `strand` columns
- Filtered to standard chromosomes (chr1-22, chrX, chrY, chrM)
- Result: 271,667 → **271,470 peaks**

#### 7e. Cell Matching
- Intersect RNA and ATAC by full obs_names (globally unique cell IDs = sample_id + barcode)
- **Important**: matching on raw barcode alone is wrong (barcodes repeat across samples)
- Common cells after QC: **92,117** #TODO: you should only keep the common cells

#### 7f. CellTypist Annotation (`annotate_celltype_celltypist`)
- Called on RNA adata after QC, before normalization
- `identify_counts_layer` detects raw counts in `.X`
- Internally normalizes (10k CPM + log1p) for CellTypist input
- Pre-clusters with Leiden (auto-res=20 for ~92k cells): HVG → z-score → PCA → kNN → Leiden
- Runs `Immune_All_Low.pkl` (98 fine types) with majority voting per cluster
- Adds to obs: `leiden`, `CT_Major`, `CT_Major_percell`, `CT_Minor`, `CT_Minor_percell`
- `cell_type` column set to `CT_Major`
- Cell type labels transferred to ATAC obs after cell matching

#### 7g. RNA Normalization + GTF Annotation
- `normalize_func`: stores raw counts in `layers['counts']`, log-normalizes (10k CPM + log1p) into `layers['lognorm']`, resets `.X = counts`
- GTF (Gencode v47): maps Ensembl IDs → `gene_name` and `interval` (chr:start-end of gene body)
  - 29,155/29,155 genes mapped to gene_name
  - 28,454/29,155 genes mapped to interval

#### 7h. Metadata (`uns`)
- `dataset_id`: 'MMP'
- `dataset_name`: 'Multiple Myeloma Precursor (MMP)'
- `dataset_organism`: 'human'
- `normalization_id`: 'lognorm'
- `data_reference`: 'GSE311602'

### 8. Final Output Shapes
| File | Shape | Notes |
|------|-------|-------|
| `MMP_rna_processed.h5ad` | 92,117 × 29,155 | `.X`=raw counts, `layers['lognorm']`=normalized |
| `MMP_atac_processed.h5ad` | 92,117 × 271,470 | `.X`=raw counts, standard chr only |

---

## Step 9. Regression Evaluation

### 9a. Consensus (`evaluation/compute_consensus.py`)
- For each dataset (multiome/dogma): reads the 5 GRN prediction h5ads
- Computes quantile-based consensus number of putative regulators per gene
- Output: `resources/results/MMP/evaluation/regulators_consensus_MMP_{dataset}.json`

### 9b. Regression (`evaluation/run_regression_job.sh` + `submit_regression.sh`)
- 10 SLURM jobs: 5 methods × 2 datasets (jobs 10259703–10259712)
- CV groups: `CT_Major` (8 cell types, min 200 cells each)
- Metrics: `r_precision`, `r_recall` (ridge regression, LOGO-CV)
- Output: `resources/results/MMP/evaluation/MMP_{dataset}_{method}_regression_score.h5ad`

**Note**: Added `MMP_multiome` and `MMP_dogma` to `DATASETS_METRICS` in `task_grn_inference/src/utils/config.py` with `['regression']`.

---

## Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `merge_peaks.py` | `genernbi_supp/src/MMP/` | Bedtools merge on outer-join ATAC → consensus peaks |
| `script.py` | `genernbi_supp/src/MMP/preprocess/` | Full processing: QC, CellTypist, normalize, metadata |
| `compute_consensus.py` | `genernbi_supp/src/MMP/evaluation/` | Compute regulators_consensus JSON for multiome + dogma |
| `run_regression_job.sh` | `genernbi_supp/src/MMP/evaluation/` | SLURM job: regression metric for one dataset × method |
| `submit_regression.sh` | `genernbi_supp/src/MMP/evaluation/` | Launcher: submits all 10 regression jobs |

**Singularity image**: `ciim.sif` (has celltypist, scanpy, anndata, scipy)

**Inputs**: `task_grn_inference/resources/datasets_raw/MMP_rna.h5ad` + `MMP_atac_merged.h5ad`

**Outputs**: `task_grn_inference/resources/results/MMP/`
