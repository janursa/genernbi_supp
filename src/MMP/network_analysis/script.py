#!/usr/bin/env python3
"""
MMP GRN Network Biology Analysis
Adapted from src/stability_analysis/ibd_biology/script.py

Disease cell type: Plasma cells (Plasma_B in CT_Minor) are the disease-defining cells
in Multiple Myeloma and its precursors (MGUS/SMM). The KNOWN_TFS list reflects
plasma cell differentiation and MMP-relevant TF biology.

Figures:
  1. Top 20 TFs by out-degree centrality per model — one figure per dataset
  2. Top 20 MMP GWAS genes by in-degree centrality per model — one figure per dataset
  3. MMP GWAS TF summary — normalized out-degree of GWAS-associated TFs per model

GWAS: 255 MMP-associated genes from GWAS Catalog (228 rows; traits: multiple myeloma,
      myeloma, mgus, smoldering, plasma cell).

Run from genernbi_supp repo root:
    python3 src/MMP/network_analysis/script.py

Methods: GRNBoost2, SCENIC+, CellOracle, PPCOR, GRaNIE (5 methods)
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────
TASK_GRN_DIR  = '/home/jnourisa/projs/ongoing/task_grn_inference'
GENERNBI_DIR  = '/home/jnourisa/projs/ongoing/genernbi_supp'
RESULTS_MMP   = os.path.join(TASK_GRN_DIR, 'resources/results/MMP')
OUT_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, GENERNBI_DIR)
from src.helper import surrogate_names, palette_methods, colors_blind
from src.MMP.constants import MMP_KNOWN_TFS, get_mmp_gwas_tfs

# ── methods & datasets ────────────────────────────────────────────────────────
METHODS = ['grnboost', 'scenicplus', 'celloracle', 'ppcor', 'granie']
METHOD_COLORS = {m: palette_methods.get(surrogate_names.get(m, m), 'grey') for m in METHODS}
METHOD_COLORS['celloracle'] = '#4CAF50'

DATASETS = ['multiome', 'dogma']

KNOWN_TFS = MMP_KNOWN_TFS


# ── helpers ───────────────────────────────────────────────────────────────────
def load_grn(dataset_tag, method):
    """Load GRN prediction DataFrame for a given dataset and method. Returns None if missing."""
    path = os.path.join(RESULTS_MMP, f'MMP_{dataset_tag}_{method}_grn.h5ad')
    if not os.path.exists(path):
        print(f'  [MISSING] {path}', flush=True)
        return None
    return ad.read_h5ad(path).uns['prediction']


get_mmp_gwas_genes = get_mmp_gwas_tfs  # alias for backward compat


def _hbar(ax, labels, values, highlight_set):
    bar_colors = [colors_blind[1] if lbl in highlight_set else colors_blind[0]
                  for lbl in labels]
    y = range(len(labels))
    ax.barh(list(y), values, color=bar_colors, edgecolor='white', linewidth=0.4, alpha=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)


# ── Figure 1: Top 20 TFs by out-degree ───────────────────────────────────────
def fig_top_tfs(dataset):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(f'Top 20 TFs by Out-degree · MMP:{dataset}',
                 fontsize=13, fontweight='bold')
    for i, (ax, method) in enumerate(zip(axes, METHODS)):
        grn = load_grn(dataset, method)
        if grn is None:
            ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='grey')
            ax.set_title(surrogate_names.get(method, method), fontsize=10)
            continue
        od = grn.groupby('source')['target'].count().sort_values(ascending=False).head(20)
        _hbar(ax, od.index.tolist(), od.values.tolist(), KNOWN_TFS)
        ax.set_xlabel('Out-degree (# targets)', fontsize=8)
        ax.set_title(surrogate_names.get(method, method),
                     fontweight='bold', color=METHOD_COLORS[method], fontsize=10)

    legend_handles = [mpatches.Patch(color=colors_blind[1], label='Known MMP/plasma-cell TF')]
    fig.legend(handles=legend_handles, ncol=1, fontsize=9, frameon=False,
               bbox_to_anchor=(.85, .08))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'fig1_top20_tfs_{dataset}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Figure 2: Top 20 MMP GWAS genes by in-degree ────────────────────────────
def fig_gwas_indegree(dataset, gwas_genes):
    fig, axes = plt.subplots(1, 5, figsize=(27, 7))
    fig.suptitle(
        f'Top 20 MMP GWAS Genes by In-degree Centrality · MMP:{dataset}',
        fontsize=13, fontweight='bold'
    )
    for ax, method in zip(axes, METHODS):
        grn = load_grn(dataset, method)
        if grn is None:
            ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='grey')
            ax.set_title(surrogate_names.get(method, method), fontsize=10)
            continue
        indeg = grn.groupby('target')['source'].count()
        gwas_indeg = indeg[indeg.index.isin(gwas_genes)].sort_values(ascending=False).head(20)
        if gwas_indeg.empty:
            ax.text(0.5, 0.5, 'No GWAS genes\nin network', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            _hbar(ax, gwas_indeg.index.tolist(), gwas_indeg.values.tolist(), set())
        ax.set_xlabel('In-degree (# regulating TFs)', fontsize=9)
        ax.set_title(surrogate_names.get(method, method), fontweight='bold',
                     color=METHOD_COLORS[method], fontsize=11)
        ax.annotate(f'n={len(indeg[indeg.index.isin(gwas_genes)])} GWAS genes\nin network',
                    xy=(0.97, 0.03), xycoords='axes fraction', ha='right',
                    fontsize=7.5, color='grey')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'fig2_gwas_indegree_{dataset}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Figure 3: MMP GWAS TF out-degree summary — all models + datasets ─────────
def fig_gwas_tf_summary():
    """For each GWAS-associated TF: show out-degree per model per dataset as dots."""
    gwas = pickle.load(open(GWAS_PATH, 'rb'))
    tf_all = set(pd.read_csv(TF_ALL_PATH).iloc[:, 0].astype(str).str.strip())

    def parse_genes(s):
        if pd.isna(s):
            return []
        return [g.strip() for g in s.replace(' - ', ', ').split(',')
                if g.strip() and g.strip() != 'NR']

    mask = gwas['DISEASE/TRAIT'].str.lower().str.contains(
        'multiple myeloma|plasma cell|myeloma|mgus|smoldering', na=False
    )
    gwas_genes_all = set()
    for col in ['REPORTED GENE(S)', 'MAPPED_GENE']:
        if col not in gwas.columns:
            continue
        gwas[mask][col].dropna().apply(parse_genes).apply(gwas_genes_all.update)
    gwas_tfs = sorted(gwas_genes_all & tf_all)
    print(f'  GWAS TFs (overlap with tf_all): {len(gwas_tfs)}', flush=True)

    # Collect out-degree for each GWAS TF, method, dataset
    records = []
    for dataset in DATASETS:
        for method in METHODS:
            grn = load_grn(dataset, method)
            if grn is None:
                continue
            od = grn.groupby('source')['target'].count()
            for tf in gwas_tfs:
                if tf in od.index:
                    records.append({'tf': tf, 'method': method,
                                    'dataset': dataset, 'outdegree': od[tf]})

    if not records:
        print('  fig3 skipped — no GRN outputs available yet', flush=True)
        return

    df = pd.DataFrame(records)
    # rank TFs by mean out-degree across all models/datasets
    tf_order = (df.groupby('tf')['outdegree'].mean()
                  .sort_values(ascending=False).index.tolist())
    tf_order = [tf for tf in tf_order if tf in set(df['tf'])]

    fig, ax = plt.subplots(figsize=(4, max(5, len(tf_order) * 0.28)))
    y_pos = {tf: i for i, tf in enumerate(tf_order)}

    for method in METHODS:
        for dataset in DATASETS:
            sub = df[(df['method'] == method) & (df['dataset'] == dataset)]
            if sub.empty:
                continue
            ys = [y_pos[r['tf']] for _, r in sub.iterrows()]
            xs = sub['outdegree'].tolist()
            marker = 'o' if dataset == 'multiome' else 's'
            ax.scatter(xs, ys, color=METHOD_COLORS[method], s=20, zorder=3,
                       marker=marker, alpha=0.8, edgecolors='white', linewidths=0.2)

    ax.set_yticks(list(range(len(tf_order))))
    ax.set_yticklabels(
        [f'*{tf}' if tf in KNOWN_TFS else tf for tf in tf_order],
        fontsize=7
    )
    ax.invert_yaxis()
    ax.set_xlabel('Out-degree (# targets)', fontsize=9)
    ax.set_title('MMP GWAS TFs — Out-degree\nacross models & datasets', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    # legend: methods (color) + datasets (marker)
    method_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=METHOD_COLORS[m], markersize=6,
                   label=surrogate_names.get(m, m))
        for m in METHODS
    ]
    dataset_handles = [
        plt.Line2D([0], [0], marker='o', color='grey', markersize=6,
                   linestyle='None', label='Multiome'),
        plt.Line2D([0], [0], marker='s', color='grey', markersize=6,
                   linestyle='None', label='DOGMA'),
    ]
    leg1 = ax.legend(handles=method_handles, title='Model', loc='lower right',
                     fontsize=7, title_fontsize=8, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=dataset_handles, title='Dataset', loc='upper right',
              fontsize=7, title_fontsize=8, frameon=True)

    ax.annotate('* known MMP/plasma-cell TF', xy=(0, -0.06), xycoords='axes fraction',
                fontsize=7, color='grey')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig3_gwas_tf_outdegree.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading MMP GWAS genes...')
    gwas_genes = get_mmp_gwas_genes()
    print(f'  {len(gwas_genes)} MMP-associated genes from GWAS Catalog')

    for dataset in DATASETS:
        print(f'\n── Dataset: MMP:{dataset} ──')
        fig_top_tfs(dataset)
        fig_gwas_indegree(dataset, gwas_genes)

    print('\n── GWAS TF out-degree summary (all models) ──')
    fig_gwas_tf_summary()

    print('\nDone.')
