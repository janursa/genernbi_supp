# Experimental GRN Inference — LOG

## Objective
Design and benchmark new GRN inference methods to beat GRNBoost2 (score=0.9515, rank #1).

## Experimental Design (3 Factors)

| Factor | Levels |
|--------|--------|
| S1 — Algorithm | pearson, spearman, ridge, lasso, elasticnet |
| S2 — Prior     | P0 (no skeleton), P1 (skeleton.csv filter) |
| S3 — ATAC      | C0 (none), C1 (accessibility filter), C3 (peak-gene corr), C5 (joint embedding) |

Naming: `{dataset}.S1-{algo}.S2-{P0|P1}.S3-{C0|C1|C3|C5}.h5ad`

---

## Stage 1 — Core Algorithm Inference

### Implementation (`grn_inference/stage1_infer.py`)
- All 5 algorithms in one script selected via `--algorithm`
- TF filtering: `tf_all.csv` loaded with `np.loadtxt`, intersected with dataset genes
- Output: **1,000,000 edges** (not 50k) to allow downstream pruning
- Regression methods: subsample to 5000 cells; auto-alpha = `clip(0.001 * mean_gene_std, 1e-4, 0.1)`
- Spearman = rank-transform X, then Pearson on ranks (avoids scipy overhead)

### Submission (`grn_inference/run_jobs.py`)
- 50 sbatch jobs submitted: job IDs 10210919–10210968
- Large PBMC datasets (op, 300BCG, parsebioscience, ibd_uc, ibd_cd): 6-8h / 120GB
- Small datasets: 3h / 60GB

### Status (2026-04-14)
- Jobs completed: 42/50 as of ~11:30 UTC
- Remaining: lasso × {op, ibd_uc, ibd_cd, parsebioscience}, elasticnet × same 4 datasets
- **Sanity check PASSED**: NonTF_src = 0 for ALL files; 1M edges except norman (<1M, expected)

### Warnings (non-fatal)
- LASSO/ElasticNet: ConvergenceWarning on large PBMC datasets (fine)
- Ridge: ill-conditioned matrix on nakatake/norman (fine)

---

## Stage 2 — Skeleton Prior Filter

### Implementation (`grn_inference/stage2_skeleton.py`)
- P0: top 50k edges by |weight| (no skeleton filter)
- P1: intersect with skeleton.csv (13.4M edges), then top 50k
- Input: `predictions/` (1M edges), Output: `predictions_final/` (50k edges)
- Tested on norman/pearson: ✅ 50k P0, 50k P1 (243k skeleton hits)

### Submission
- sbatch job 10211110 submitted — processing all 42 completed S1 files
- Will re-run after remaining 8 S1 jobs complete to pick up the rest

---

## Stage 3 — ATAC Integration (Planned)

Applies only to datasets with paired ATAC: op, ibd_uc, ibd_cd

- **C1**: filter edges where target gene has no ATAC peak within ±500kb of TSS
- **C3**: require peak-gene Pearson r > 0.1 for ≥1 nearby peak
- **C5**: SCGLUE joint embedding (GRN pruning via attention/dot product)

Status: Not yet implemented — waiting for S1 completion

---

## Evaluation

### Scripts
- `grn_evaluation/run_eval.py` — submit sbatch eval jobs for all `predictions_final/*.h5ad`
- `grn_evaluation/compare_scores.py` — aggregate scores, fixed-pool normalization, leaderboard

### Invocation pattern (from `src/metrics/all_metrics/run_local.sh`)
```bash
python src/metrics/all_metrics/script.py \
  --prediction ... --evaluation_data ..._rna_all.h5ad \
  --evaluation_data_de ..._de.h5ad (if exists) \
  --regulators_consensus ...json \
  --ground_truth_unibind/chipatlas/remap (if cell type != '') \
  --ws_consensus / --ws_distance_background (if exist) \
  --layer lognorm --reg_type ridge --num_workers 20 \
  --tf_all ... --score ...
```

### Cell type mapping
| Dataset | Cell Type |
|---------|-----------|
| replogle, norman | K562 |
| xaira_HEK293T | HEK293T |
| xaira_HCT116 | HCT116 |
| op, parsebioscience, 300BCG, ibd_uc, ibd_cd | PBMC |
| nakatake | (empty) |

---

## Stage 3 — ATAC Integration

### Implementation (`grn_inference/stage3_atac.py`)
- Reads paired ATAC+RNA, builds chromosomal peak index (binary search for overlaps)
- Gene windows derived from RNA var `interval` column (no GTF needed)
- **C1**: mark gene accessible if ≥1 peak overlaps [gene_start-500kb, gene_end+500kb]; drop unaccessible
- **C3**: Pearson(peak_accessibility, gene_expression) per nearby peak; keep if max|r| ≥ 0.1
- Unannotated genes (no `interval`) conservatively kept in both modes

### C1 test result (op/pearson/P0)
- 13,577/13,582 genes accessible; 50,000 → 49,953 edges (47 removed)
- op has very dense peaks (135k peaks), nearly all genes accessible

### Submitted jobs (`grn_inference/run_stage3_jobs.py`)
| Dataset | C1 Job | C3 Job | Status |
|---------|--------|--------|--------|
| op      | 10211269 (running) | 10211270 (running) | Partial |
| ibd_uc  | 10211271 | 10211272 | ✅ Done |
| ibd_cd  | 10211273 | 10211274 | ✅ Done |

### Files in predictions_final/ (2026-04-14 ~12:20)
- C0: 84 files (42 S1 × P0+P1)
- C1: 26 files (ibd_uc×10, ibd_cd×10, op×6)
- C3: 20 files (ibd_uc×10, ibd_cd×10)

---

## Evaluation

### Jobs submitted
- `run_eval.py` submitted **105 eval jobs** (IDs 10211275–10211379)
- Covers all existing predictions_final/*.h5ad files
- Will re-run after op C1/C3 + xaira_HEK293T elasticnet complete

---

## Next Steps
1. ⏳ Wait for op stage3 jobs (10211269, 10211270) → re-run run_eval.py
2. ⏳ Wait for xaira_HEK293T elasticnet S1 (10210965) → stage2 → eval
3. ✅ Run `grn_evaluation/compare_scores.py` once eval scores arrive
