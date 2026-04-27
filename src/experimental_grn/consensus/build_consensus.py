"""
Build experiment-specific consensus files from our experimental predictions.

Two consensus types are generated:
  1. regulators_consensus_{dataset}.json  — for regression metric (all datasets)
  2. ws_consensus_{dataset}.csv           — for WS-distance metric (perturbation datasets only)

The consensus is built from ALL C0 predictions for the dataset
(S2-P0 and S2-P1, all algorithms). C1/C2 are ATAC-filtered subsets and
should not contribute to the consensus baseline.

Outputs go to temp/experimental_grn/prior/ (isolated from resources/grn_benchmark/prior/).

Usage:
    python build_consensus.py --dataset norman
    python build_consensus.py --dataset op        # regression only (no WS)
"""

import argparse
import glob
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

REPO        = '/home/jnourisa/projs/ongoing/task_grn_inference'
PRED_DIR    = f'{REPO}/temp/experimental_grn/predictions_final'
OUT_DIR     = f'{REPO}/temp/experimental_grn/prior'
EVAL_DIR    = f'{REPO}/resources/grn_benchmark/evaluation_data'
PRIOR_DIR   = f'{REPO}/resources/grn_benchmark/prior'

# Only these datasets support WS-distance metric (perturbation-based)
WS_DATASETS = {'norman', 'replogle', 'xaira_HEK293T', 'xaira_HCT116'}

os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, f'{REPO}/src/utils')
from util import process_links


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_predictions_c0(dataset):
    """Return list of all C0 prediction .h5ad paths for this dataset."""
    pattern = os.path.join(PRED_DIR, f'{dataset}.S1-*.C0.h5ad')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No C0 predictions found for {dataset} in {PRED_DIR}')
    print(f'  Found {len(files)} C0 predictions for {dataset}:', flush=True)
    for f in files:
        print(f'    {os.path.basename(f)}', flush=True)
    return files


def load_grn(fpath, par=None):
    """Load prediction df from h5ad, apply process_links if available."""
    a = ad.read_h5ad(fpath)
    df = a.uns['prediction'].copy()
    df['weight'] = df['weight'].astype(float)
    if par is not None:
        df = process_links(df, par)
    return df


# ─── regression consensus ─────────────────────────────────────────────────────

def build_regression_consensus(dataset, pred_files):
    """
    For each gene: compute quantile of 'number of regulators' across models
    at thetas [0, 0.25, 0.5, 0.75, 1].
    Output: regulators_consensus_{dataset}.json
    """
    out_path = os.path.join(OUT_DIR, f'regulators_consensus_{dataset}.json')
    if os.path.exists(out_path):
        print(f'  Regression consensus already exists: {out_path}', flush=True)
        return

    bulk = ad.read_h5ad(f'{EVAL_DIR}/{dataset}_rna_all.h5ad', backed='r')
    gene_names = list(bulk.var_names)
    gene_idx   = {g: i for i, g in enumerate(gene_names)}
    n_genes    = len(gene_names)

    print(f'  Building regression consensus: {n_genes} genes, {len(pred_files)} models', flush=True)

    grns = []
    par  = {'max_n_links': 50_000}
    for fpath in pred_files:
        df = load_grn(fpath, par)
        A  = np.zeros((n_genes, n_genes), dtype=np.float32)
        for src, tgt, w in zip(df['source'], df['target'], df['weight']):
            if src in gene_idx and tgt in gene_idx:
                A[gene_idx[src], gene_idx[tgt]] = float(w)
        grns.append(A)
        print(f'    loaded {os.path.basename(fpath)}, sparsity={np.mean(A==0):.4f}', flush=True)

    grns   = np.asarray(grns)          # (n_models, n_genes, n_genes)
    thetas = [0, 0.25, 0.5, 0.75, 1]
    n_tfs  = {
        theta: np.round(
            np.quantile(np.sum(grns != 0, axis=1), theta, axis=0)
        ).astype(int)
        for theta in thetas
    }

    results = {
        gene: {theta: int(n_tfs[theta][i]) for theta in thetas}
        for i, gene in enumerate(gene_names)
    }
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f'  Saved regression consensus → {out_path}', flush=True)


# ─── WS-distance consensus ────────────────────────────────────────────────────

def build_ws_consensus(dataset, pred_files):
    """
    For each TF that was perturbed: compute 0.25 and 0.75 quantile of
    'number of target edges per model', across all models.
    Output: ws_consensus_{dataset}.csv

    Available TFs are derived from the DE file obs_names (perturbed genes).
    """
    out_path = os.path.join(OUT_DIR, f'ws_consensus_{dataset}.csv')
    if os.path.exists(out_path):
        print(f'  WS consensus already exists: {out_path}', flush=True)
        return

    de_path = f'{EVAL_DIR}/{dataset}_de.h5ad'
    if not os.path.exists(de_path):
        print(f'  WS consensus: no DE file found ({de_path}), skipping', flush=True)
        return

    tf_all       = np.loadtxt(f'{PRIOR_DIR}/tf_all.csv', dtype=str)
    de           = ad.read_h5ad(de_path, backed='r')
    available_tfs = np.intersect1d(de.obs_names, tf_all)
    print(f'  WS consensus: {len(available_tfs)} perturbed TFs', flush=True)

    par = {'max_n_links': 50_000}
    grn_store = []
    for fpath in pred_files:
        df = load_grn(fpath, par)
        df['model'] = os.path.basename(fpath)
        grn_store.append(df)

    grn_all = pd.concat(grn_store, ignore_index=True)
    grn_all = grn_all[grn_all['source'].isin(available_tfs)]

    edges_count = (
        grn_all.groupby(['source', 'model'])
        .size()
        .reset_index(name='n_edges')
        .pivot(index='source', columns='model')
        .fillna(0)
    )

    consensus = []
    for tf, row in edges_count.iterrows():
        row_nz = row[row != 0]
        if len(row_nz) == 0:
            continue
        consensus.append({'source': tf, 'theta': 0.25, 'value': int(np.quantile(row_nz, 0.25))})
        consensus.append({'source': tf, 'theta': 0.75, 'value': int(np.quantile(row_nz, 0.75))})

    consensus_df = pd.DataFrame(consensus)
    consensus_df.to_csv(out_path)
    print(f'  Saved WS consensus ({len(consensus_df)} rows) → {out_path}', flush=True)


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Dataset name (e.g. norman, op, ibd_uc)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ds   = args.dataset

    print(f'\n=== Consensus for {ds} ===', flush=True)
    pred_files = load_predictions_c0(ds)

    print(f'\n[1] Regression consensus', flush=True)
    build_regression_consensus(ds, pred_files)

    if ds in WS_DATASETS:
        print(f'\n[2] WS-distance consensus', flush=True)
        build_ws_consensus(ds, pred_files)
    else:
        print(f'\n[2] WS-distance consensus: skipped (not a perturbation dataset)', flush=True)

    print(f'\nDone: {ds}', flush=True)
