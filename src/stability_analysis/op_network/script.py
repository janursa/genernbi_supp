#!/usr/bin/env python3
"""
OP Dataset GRN Network Analysis
Adapted from src/stability_analysis/ibd_biology/script.py

Dataset: op (OpenProblems PBMC multiome)
  - 25,551 cells x 13,595 genes
  - Cell types: T cells, Myeloid cells, NK cells, B cells
  - 3 donors

Analysis: Top TFs by out-degree centrality per model, with known PBMC/immune
          TF markers highlighted. No GWAS analysis.

Methods: GRNBoost2, SCENIC+, CellOracle, PPCOR, GRaNIE (5 methods)

Run from geneRNBI repo root:
    python3 src/stability_analysis/op_network/script.py
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────
TASK_GRN_DIR = '/home/jnourisa/projs/ongoing/task_grn_inference'
GENERNBI_DIR = '/home/jnourisa/projs/ongoing/geneRNBI'
RESULTS_OP   = os.path.join(TASK_GRN_DIR, 'resources/results/op')
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, GENERNBI_DIR)
sys.path.insert(0, '/home/jnourisa/projs/ongoing')
from src.helper import surrogate_names, palette_methods, colors_blind

# ── methods ───────────────────────────────────────────────────────────────────
METHODS = ['grnboost', 'scenicplus', 'celloracle', 'ppcor', 'granie']
METHOD_COLORS = {m: palette_methods.get(surrogate_names.get(m, m), 'grey') for m in METHODS}
METHOD_COLORS['celloracle'] = '#4CAF50'

# ── known PBMC / immune TF markers — highlighted in plots ────────────────────
# Same cell types as IBD (T, Myeloid, NK, B), so we use canonical PBMC TFs
KNOWN_TFS = {
    # T cell lineage
    'TBX21', 'GATA3', 'RORC', 'FOXP3', 'TCF7', 'LEF1', 'EOMES',
    'BCL11B', 'RUNX3', 'TOX', 'NR4A1', 'IKZF2',
    # B cell lineage
    'PAX5', 'EBF1', 'IRF4', 'BACH2', 'PRDM1', 'XBP1',
    # Myeloid / NK
    'SPI1', 'CEBPA', 'CEBPB', 'IRF8', 'MAFB', 'KLF4', 'ZEB2',
    'NFKB1', 'NFKB2', 'REL', 'RELB',
    # Pan-immune
    'STAT1', 'STAT3', 'RUNX1', 'IKZF1', 'IRF1', 'IRF2',
    'EGR1', 'SP1', 'MYC', 'FLI1', 'ETS1', 'AHR',
    'BATF', 'JUN', 'FOSB', 'NFE2L2', 'HIF1A',
}


# ── helpers ───────────────────────────────────────────────────────────────────
def load_grn(method):
    path = os.path.join(RESULTS_OP, f'op.{method}.{method}.prediction.h5ad')
    if not os.path.exists(path):
        print(f'  [MISSING] {path}', flush=True)
        return None
    return ad.read_h5ad(path).uns['prediction']


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
def fig_top_tfs():
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Top 20 TFs by Out-degree · OP (PBMC multiome)',
                 fontsize=13, fontweight='bold')

    for ax, method in zip(axes, METHODS):
        grn = load_grn(method)
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

        n_known = sum(1 for tf in od.index if tf in KNOWN_TFS)
        ax.annotate(f'{n_known}/20 known\nimmune TFs',
                    xy=(0.97, 0.03), xycoords='axes fraction',
                    ha='right', fontsize=7.5, color='grey')

    legend_handles = [mpatches.Patch(color=colors_blind[1], label='Known PBMC/immune TF')]
    fig.legend(handles=legend_handles, ncol=1, fontsize=9, frameon=False,
               bbox_to_anchor=(.88, .08))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig1_top20_tfs_op.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Figure 2: Out-degree distribution — all TFs across methods ───────────────
def fig_outdegree_summary():
    """Dot plot: each TF appearing in top 20 of ≥2 models, with out-degree per model."""
    from collections import Counter

    # Collect top-20 TFs per method and their out-degrees
    all_records = []
    top20_counts = Counter()
    for method in METHODS:
        grn = load_grn(method)
        if grn is None:
            continue
        od = grn.groupby('source')['target'].count().sort_values(ascending=False).head(20)
        top20_counts.update(od.index.tolist())
        for tf, deg in od.items():
            all_records.append({'tf': tf, 'method': method, 'outdegree': deg})

    # Keep TFs that appear in top-20 of ≥2 methods
    recurrent_tfs = {tf for tf, cnt in top20_counts.items() if cnt >= 2}
    df = pd.DataFrame(all_records)
    df = df[df['tf'].isin(recurrent_tfs)]

    if df.empty:
        print('  fig2 skipped — no recurrent TFs found', flush=True)
        return

    tf_order = (df.groupby('tf')['outdegree'].mean()
                  .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(4, max(4, len(tf_order) * 0.3)))
    y_pos = {tf: i for i, tf in enumerate(tf_order)}

    for method in METHODS:
        sub = df[df['method'] == method]
        if sub.empty:
            continue
        ys = [y_pos[r['tf']] for _, r in sub.iterrows()]
        xs = sub['outdegree'].tolist()
        ax.scatter(xs, ys, color=METHOD_COLORS[method], s=25, zorder=3,
                   alpha=0.85, edgecolors='white', linewidths=0.3)

    ax.set_yticks(list(range(len(tf_order))))
    ax.set_yticklabels(
        [f'*{tf}' if tf in KNOWN_TFS else tf for tf in tf_order],
        fontsize=8
    )
    ax.invert_yaxis()
    ax.set_xlabel('Out-degree (# targets)', fontsize=9)
    ax.set_title('Recurrent hub TFs (top-20 in ≥2 models)\nOP PBMC multiome', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    method_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=METHOD_COLORS[m], markersize=7,
                   label=surrogate_names.get(m, m))
        for m in METHODS
    ]
    ax.legend(handles=method_handles, title='Model', fontsize=7,
              title_fontsize=8, frameon=True, loc='lower right')
    ax.annotate('* known PBMC/immune TF', xy=(0, -0.06), xycoords='axes fraction',
                fontsize=7, color='grey')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig2_recurrent_hub_tfs_op.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('── OP dataset: Top TFs by out-degree ──')
    fig_top_tfs()

    print('── OP dataset: Recurrent hub TFs ──')
    fig_outdegree_summary()

    print('\nDone.')
