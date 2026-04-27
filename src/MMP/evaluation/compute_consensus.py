#!/usr/bin/env python3
"""
Compute regulators_consensus JSON for MMP multiome and dogma datasets.

Run from task_grn_inference repo root:
    python3 /home/jnourisa/projs/ongoing/geneRNBI/src/MMP/evaluation/compute_consensus.py

Output: resources/results/MMP/evaluation/regulators_consensus_MMP_{dataset}.json
"""

import sys
sys.path.insert(0, 'src/metrics/regression/consensus')
sys.path.insert(0, 'src/utils')

import os
import json
import numpy as np
import anndata as ad

from util import process_links

RESULTS_MMP  = 'resources/results/MMP'
EVAL_OUT_DIR = 'resources/results/MMP/evaluation'
METHODS      = ['grnboost', 'scenicplus', 'celloracle', 'ppcor', 'granie']
DATASETS     = ['multiome', 'dogma']

os.makedirs(EVAL_OUT_DIR, exist_ok=True)


def compute_consensus(dataset: str):
    rna_path = os.path.join(RESULTS_MMP, f'MMP_{dataset}_rna_processed.h5ad')
    out_json  = os.path.join(EVAL_OUT_DIR, f'regulators_consensus_MMP_{dataset}.json')

    print(f'\n=== {dataset} ===')
    adata_rna = ad.read_h5ad(rna_path)
    gene_names = adata_rna.var_names.tolist()
    gene_dict  = {g: i for i, g in enumerate(gene_names)}
    print(f'  Genes: {len(gene_names)}')

    grns = []
    for method in METHODS:
        grn_path = os.path.join(RESULTS_MMP, f'MMP_{dataset}_{method}_grn.h5ad')
        if not os.path.exists(grn_path):
            print(f'  [MISSING] {grn_path}')
            continue
        net = ad.read_h5ad(grn_path).uns['prediction']
        par_dummy = {'max_n_links': 50000}
        net = process_links(net, par_dummy)

        A = np.zeros((len(gene_names), len(gene_names)), dtype=float)
        for src, tgt, w in zip(net['source'], net['target'], net['weight']):
            if src in gene_dict and tgt in gene_dict:
                A[gene_dict[src], gene_dict[tgt]] = float(w)
        sparsity = np.mean(A == 0)
        print(f'  {method}: sparsity={sparsity:.4f}')
        grns.append(A)

    assert len(grns) > 0, 'No GRN files found!'
    grns = np.asarray(grns)  # (n_methods, n_genes, n_genes)

    thetas = [0, 0.25, 0.5, 0.75, 1]
    n_tfs  = {}
    for theta in thetas:
        n_tfs[theta] = np.round(
            np.quantile(np.sum(grns != 0, axis=1), theta, axis=0)
        ).astype(int)

    results = {}
    for i, gene in enumerate(gene_names):
        results[gene] = {theta: int(n_tfs[theta][i]) for theta in thetas}

    with open(out_json, 'w') as f:
        json.dump(results, f)
    print(f'  Saved: {out_json}')


if __name__ == '__main__':
    for ds in DATASETS:
        compute_consensus(ds)
    print('\nDone.')
