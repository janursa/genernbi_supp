"""
apply_prior.py — Unified prior-integration script (C0–C6).

Reads one S1 raw prediction (1M edges) from predictions/ and writes filtered
predictions to predictions_final/.

C-axis:
  C0  top-50k by |weight|                                   (all datasets)
  C1  intersect motif skeleton ±1 kb TSS, top-50k           (all datasets)
  C2  intersect motif skeleton ±100 kb TSS, top-50k         (all datasets)
  C3  ATAC peak ±1 kb of gene body, top-50k                 (ATAC datasets only)
  C4  ATAC peak ±100 kb of gene body, top-50k               (ATAC datasets only)
  C5  ATAC peak ±100 kb + peak-gene corr ≥ threshold, top-50k (ATAC datasets only)
  C6  C2 ∩ C4  (motif 100kb AND ATAC 100kb), top-50k        (ATAC datasets only)

Input:  predictions/{ds}.S1-{algo}.h5ad
Output: predictions_final/{ds}.S1-{algo}.{c_tag}.h5ad

Usage:
  python apply_prior.py --dataset 300BCG --algorithm pearson --c_tag C0
  python apply_prior.py --dataset op     --algorithm lasso   --c_tag C5
  python apply_prior.py --dataset op     --algorithm lasso   --c_tag all_atac
  python apply_prior.py --dataset 300BCG --algorithm all     --c_tag all_nonatac
"""

import os, sys, bisect, argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse

# ── Paths ────────────────────────────────────────────────────────────────────

REPO      = '/home/jnourisa/projs/ongoing/task_grn_inference'
DATA_DIR  = f'{REPO}/resources/grn_benchmark/inference_data'
PRED_IN   = f'{REPO}/temp/experimental_grn/predictions'
PRED_OUT  = f'{REPO}/temp/experimental_grn/predictions_final'
PRIOR_DIR = f'{REPO}/resources/grn_benchmark/prior'

ATAC_DATASETS = {'op', 'ibd_uc', 'ibd_cd'}
ALGOS = ['pearson', 'lasso', 'ridge', 'elasticnet', 'spearman', 'grnboost']

TOP_K    = 50_000
MIN_CORR = 0.05     # C5 peak-gene correlation threshold
MIN_CORR_PVAL = 0.05  # C5 significance threshold for peak-gene correlation

# C-tags applicable per dataset type
C_ALL      = ['C0', 'C1', 'C2']
C_ATAC     = ['C3', 'C4', 'C5', 'C6']
C_ALL_ATAC = C_ALL + C_ATAC


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',   required=True,
                   help='Dataset name, e.g. 300BCG, op. Use "all" for everything.')
    p.add_argument('--algorithm', required=True,
                   help='Algorithm name, e.g. pearson. Use "all" for all.')
    p.add_argument('--c_tag',     required=True,
                   help='C0-C6, "all_nonatac" (C0-C2), "all_atac" (C0-C6), "all"')
    p.add_argument('--top_k',     type=int, default=TOP_K)
    p.add_argument('--min_corr',  type=float, default=MIN_CORR)
    p.add_argument('--layer',     default='lognorm',
                   help='RNA layer for expression (used in C5)')
    return p.parse_args()


# ── Top-K helper ─────────────────────────────────────────────────────────────

def top_k(df, k):
    """Return top-k edges by |weight|."""
    df = df.copy()
    df['abs_w'] = df['weight'].astype(float).abs()
    out = df.nlargest(k, 'abs_w').drop(columns='abs_w')
    return out.reset_index(drop=True)


# ── Save prediction ───────────────────────────────────────────────────────────

def save_pred(df, src_adata, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = df.copy()
    df['weight'] = df['weight'].astype(str)
    out = ad.AnnData(
        X=None,
        uns={
            'method_id':  src_adata.uns.get('method_id', ''),
            'dataset_id': src_adata.uns.get('dataset_id', ''),
            'prediction': df[['source', 'target', 'weight']],
        }
    )
    out.write_h5ad(out_path)


# ── Skeleton (motif-based) filters ───────────────────────────────────────────

_skel_cache = {}

def load_skeleton_set(name):
    """Load skeleton CSV and return a set of 'source__target' strings."""
    if name in _skel_cache:
        return _skel_cache[name]
    path = os.path.join(PRIOR_DIR, f'{name}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Skeleton not found: {path}')
    skel = pd.read_csv(path, usecols=['source', 'target'])
    skel_set = set(skel['source'] + '__' + skel['target'])
    print(f"  Loaded {name}: {len(skel_set):,} edges", flush=True)
    _skel_cache[name] = skel_set
    return skel_set


def apply_skeleton_filter(df, skeleton_name):
    """Intersect df edges with skeleton, return filtered df."""
    skel_set = load_skeleton_set(skeleton_name)
    key = df['source'] + '__' + df['target']
    df_out = df[key.isin(skel_set)].copy()
    print(f"  {skeleton_name}: {len(df):,} → {len(df_out):,} edges", flush=True)
    return df_out


# ── ATAC / genomic helpers ───────────────────────────────────────────────────

def parse_interval(interval_str):
    """'chr1:149706-173862' → ('chr1', 149706, 173862)"""
    try:
        chrom, coords = interval_str.rsplit(':', 1)
        s, e = coords.split('-')
        return chrom, int(s), int(e)
    except Exception:
        return None, None, None


def build_peak_index(atac_var):
    """Build chr → (sorted_starts, ends, peak_indices) for fast overlap query."""
    idx = {}
    for i, (_, row) in enumerate(atac_var.iterrows()):
        chrom = row['seqname']
        rng   = row['ranges']
        try:
            s, e = map(int, rng.split('-'))
        except Exception:
            continue
        idx.setdefault(chrom, []).append((s, e, i))

    for chrom in idx:
        idx[chrom].sort(key=lambda x: x[0])
        arr = idx[chrom]
        idx[chrom] = (
            np.array([x[0] for x in arr]),
            np.array([x[1] for x in arr]),
            np.array([x[2] for x in arr]),
        )
    n_peaks = sum(len(v[0]) for v in idx.values())
    print(f"  Peak index: {len(idx)} chromosomes, {n_peaks:,} peaks", flush=True)
    return idx


def overlapping_peaks(peak_index, chrom, win_start, win_end):
    """Return array of peak indices overlapping [win_start, win_end]."""
    if chrom not in peak_index:
        return np.array([], dtype=int)
    starts, ends, orig = peak_index[chrom]
    lo = bisect.bisect_right(starts, win_end)
    candidates = np.where(ends[:lo] >= win_start)[0]
    return orig[candidates]


def build_gene_windows(rna_var, window):
    """Return dict: gene → (chrom, win_start, win_end) from RNA var 'interval' col."""
    windows = {}
    for gene, row in rna_var.iterrows():
        iv = row.get('interval', '')
        if not iv or iv == 'NA':
            continue
        chrom, s, e = parse_interval(iv)
        if chrom is None:
            continue
        windows[gene] = (chrom, max(0, s - window), e + window)
    print(f"  Gene windows ({window:,} bp): {len(windows):,}/{len(rna_var):,} annotated",
          flush=True)
    return windows


def accessible_genes(gene_windows, peak_index):
    """Return set of genes with ≥1 ATAC peak in their window."""
    acc = set()
    for gene, (chrom, ws, we) in gene_windows.items():
        if len(overlapping_peaks(peak_index, chrom, ws, we)) > 0:
            acc.add(gene)
    print(f"  Accessible genes: {len(acc):,}/{len(gene_windows):,}", flush=True)
    return acc


def apply_accessibility_filter(df, accessible):
    """Keep edges whose target gene is accessible."""
    before = len(df)
    df_out = df[df['target'].isin(accessible)].copy()
    print(f"  Accessibility filter: {before:,} → {len(df_out):,}", flush=True)
    return df_out


# ── C5: peak-gene Pearson correlation ────────────────────────────────────────

def pearson_col(X, y):
    """Pearson correlation between each column of X and vector y."""
    y = y - y.mean()
    ys = y.std()
    if ys < 1e-9:
        return np.zeros(X.shape[1])
    Xc = X - X.mean(axis=0)
    Xs = Xc.std(axis=0)
    Xs[Xs < 1e-9] = np.inf
    return (Xc.T @ y) / (Xs * ys * len(y))


def corr_pvalue(r, n):
    """Two-sided p-value for Pearson r with n observations."""
    from scipy import stats
    if n <= 2 or abs(r) >= 1.0:
        return 0.0
    t = r * np.sqrt(n - 2) / np.sqrt(max(1 - r**2, 1e-15))
    return float(2 * stats.t.sf(abs(t), df=n - 2))


def apply_correlation_filter(df, rna_X, rna_genes, atac_X, gene_windows,
                              peak_index, min_corr, pval_thresh=MIN_CORR_PVAL):
    """
    Keep edges where target gene has at least one peak with:
      |correlation| >= min_corr  AND  p-value < pval_thresh.
    Genes with no window or not in RNA are kept conservatively.
    """
    n_cells    = rna_X.shape[0]
    gene_idx   = {g: i for i, g in enumerate(rna_genes)}
    target_genes = df['target'].unique()
    print(f"  C5: correlation filter for {len(target_genes):,} target genes "
          f"(r≥{min_corr}, p<{pval_thresh}, n={n_cells:,})...", flush=True)

    gene_pass = {}
    for k, gene in enumerate(target_genes):
        if gene not in gene_windows:
            gene_pass[gene] = True   # unannotated → keep
            continue
        if gene not in gene_idx:
            gene_pass[gene] = True   # not in RNA → keep
            continue

        chrom, ws, we = gene_windows[gene]
        peak_cols = overlapping_peaks(peak_index, chrom, ws, we)

        if len(peak_cols) == 0:
            gene_pass[gene] = False  # no nearby peak → remove
            continue

        gi = gene_idx[gene]
        y = rna_X[:, gi].astype(np.float32)
        A = atac_X[:, peak_cols].astype(np.float32)
        if issparse(A):
            A = A.toarray()

        corrs = pearson_col(A, y)
        # Gene passes if ANY peak has |r| >= min_corr AND p < pval_thresh
        passes = False
        for r in corrs:
            r = float(r)
            if np.isnan(r):
                continue
            if abs(r) >= min_corr and corr_pvalue(abs(r), n_cells) < pval_thresh:
                passes = True
                break
        gene_pass[gene] = passes

        if (k + 1) % 2000 == 0:
            n_pass = sum(gene_pass.values())
            print(f"    {k+1:,}/{len(target_genes):,} genes done, "
                  f"{n_pass:,} passing...", flush=True)

    before = len(df)
    df_out = df[df['target'].map(gene_pass)].copy()
    n_pass = sum(gene_pass.values())
    print(f"  C5: {n_pass:,}/{len(target_genes):,} genes pass r≥{min_corr} & p<{pval_thresh}",
          flush=True)
    print(f"  C5 filter: {before:,} → {len(df_out):,} edges", flush=True)
    return df_out


# ── Load paired RNA+ATAC ─────────────────────────────────────────────────────

def load_rna_atac(ds, layer='lognorm'):
    print(f"  Loading RNA ({ds})...", flush=True)
    rna = ad.read_h5ad(f'{DATA_DIR}/{ds}_rna.h5ad')
    print(f"  Loading ATAC ({ds})...", flush=True)
    atac = ad.read_h5ad(f'{DATA_DIR}/{ds}_atac.h5ad')

    shared = sorted(set(rna.obs_names) & set(atac.obs_names))
    print(f"  Shared barcodes: {len(shared):,}", flush=True)
    rna  = rna[shared]
    atac = atac[shared]

    rna_X = rna.layers[layer] if layer in rna.layers else rna.X
    if issparse(rna_X):
        rna_X = rna_X.toarray()
    rna_X = np.array(rna_X, dtype=np.float32)

    return rna_X, np.array(rna.var_names), rna.var, atac.X, atac.var


# ── Per-dataset per-algorithm per-C pipeline ─────────────────────────────────

def apply_c_tag(ds, algo, c_tag, top_k_n, min_corr, layer):
    """Apply one C-tag filter and write output."""
    in_path  = os.path.join(PRED_IN,  f'{ds}.S1-{algo}.h5ad')
    out_path = os.path.join(PRED_OUT, f'{ds}.S1-{algo}.{c_tag}.h5ad')

    if not os.path.exists(in_path):
        print(f"  SKIP (S1 not found): {in_path}", flush=True)
        return

    if os.path.exists(out_path):
        print(f"  SKIP (exists): {out_path}", flush=True)
        return

    print(f"\n{'─'*60}", flush=True)
    print(f"  {ds}.S1-{algo}  →  {c_tag}", flush=True)

    a  = ad.read_h5ad(in_path)
    df = a.uns['prediction'].copy()
    df['weight'] = df['weight'].astype(float)

    if c_tag == 'C0':
        df_out = top_k(df, top_k_n)

    elif c_tag == 'C1':
        df_f   = apply_skeleton_filter(df, 'skeleton_motif')
        df_out = top_k(df_f, top_k_n)

    elif c_tag == 'C2':
        df_f   = apply_skeleton_filter(df, 'skeleton_motif_100kb')
        df_out = top_k(df_f, top_k_n)

    elif c_tag in ('C3', 'C4', 'C5', 'C6'):
        if ds not in ATAC_DATASETS:
            print(f"  SKIP: {c_tag} not applicable to non-ATAC dataset {ds}", flush=True)
            return

        rna_X, rna_genes, rna_var, atac_X, atac_var = load_rna_atac(ds, layer)
        peak_index = build_peak_index(atac_var)

        if c_tag == 'C3':
            gene_windows = build_gene_windows(rna_var, 1_000)
            acc = accessible_genes(gene_windows, peak_index)
            df_f   = apply_accessibility_filter(df, acc)
            df_out = top_k(df_f, top_k_n)

        elif c_tag == 'C4':
            gene_windows = build_gene_windows(rna_var, 100_000)
            acc = accessible_genes(gene_windows, peak_index)
            df_f   = apply_accessibility_filter(df, acc)
            df_out = top_k(df_f, top_k_n)

        elif c_tag == 'C5':
            # ATAC 100 kb + peak-gene correlation (r >= min_corr AND p < pval_thresh)
            gene_windows = build_gene_windows(rna_var, 100_000)
            df_f   = apply_correlation_filter(df, rna_X, rna_genes, atac_X,
                                              gene_windows, peak_index, min_corr,
                                              pval_thresh=MIN_CORR_PVAL)
            df_out = top_k(df_f, top_k_n)

        elif c_tag == 'C6':
            # C2 (motif 100kb) ∩ C4 (ATAC 100kb)
            df_c2  = apply_skeleton_filter(df, 'skeleton_motif_100kb')
            gene_windows = build_gene_windows(rna_var, 100_000)
            acc = accessible_genes(gene_windows, peak_index)
            df_f   = apply_accessibility_filter(df_c2, acc)
            df_out = top_k(df_f, top_k_n)

    else:
        raise ValueError(f'Unknown c_tag: {c_tag}')

    print(f"  → {len(df_out):,} edges → {out_path}", flush=True)
    save_pred(df_out, a, out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve dataset list
    if args.dataset == 'all':
        in_files = sorted(os.listdir(PRED_IN))
        datasets = sorted({f.split('.S1-')[0] for f in in_files if '.S1-' in f})
    else:
        datasets = [args.dataset]

    # Resolve algorithm list
    if args.algorithm == 'all':
        algorithms = ALGOS
    else:
        algorithms = [args.algorithm]

    # Resolve C-tag list
    tag = args.c_tag
    if tag == 'all_nonatac':
        c_tags = C_ALL
    elif tag == 'all_atac':
        c_tags = C_ALL_ATAC
    elif tag == 'all':
        c_tags = C_ALL_ATAC
    else:
        c_tags = [tag]

    os.makedirs(PRED_OUT, exist_ok=True)

    print(f"Datasets  : {datasets}", flush=True)
    print(f"Algorithms: {algorithms}", flush=True)
    print(f"C-tags    : {c_tags}", flush=True)

    for ds in datasets:
        for algo in algorithms:
            for c_tag in c_tags:
                # Skip ATAC-only tags for non-ATAC datasets
                if c_tag in C_ATAC and ds not in ATAC_DATASETS:
                    continue
                apply_c_tag(ds, algo, c_tag, args.top_k, args.min_corr, args.layer)

    print(f"\nDone. Output in {PRED_OUT}", flush=True)


if __name__ == '__main__':
    main()
