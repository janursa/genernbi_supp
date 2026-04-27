#!/usr/bin/env python3
# IBD GRN Biology Case Study
# Figures:
#   1. Top 20 TFs by out-degree centrality per model — one figure per dataset
#   2. Top 20 IBD GWAS genes by in-degree centrality per model — one figure per dataset
#   3. TF centrality shift UC->CD per model — one combined figure (4 panels)
#
# Run from project root after: source env.sh
# python3 src/stability_analysis/ibd_biology/script.py

import os, sys, pickle
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.environ['RESULTS_DIR']
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

from src.helper import surrogate_names, palette_methods, colors_blind

GWAS_PATH = '/vol/projects/BIIM/agentic_central/agentic/datalake/general/biomni/gwas_catalog.pkl'

METHODS = ['grnboost', 'scenicplus', 'celloracle', 'ppcor']
METHOD_COLORS = {m: palette_methods.get(surrogate_names[m], 'red') for m in METHODS}
METHOD_COLORS['celloracle'] = '#4CAF50'  # not in project palette

DATASETS = {'UC': 'ibd_uc', 'CD': 'ibd_cd'}


# Known PBMC / IBD TFs — highlighted in red in centrality plots
KNOWN_TFS = {
    'SPI1', 'IRF8', 'IRF4', 'BATF', 'TBX21', 'GATA3', 'RORC', 'FOXP3',
    'PAX5', 'EBF1', 'TCF7', 'LEF1', 'EOMES', 'IKZF1', 'NFKB1', 'STAT3',
    'STAT1', 'CEBPA', 'CEBPB', 'RUNX1', 'RUNX3', 'BACH2', 'ZEB2', 'AHR',
    'MAFB', 'NFE2L2', 'IRF1', 'HIF1A', 'EGR1', 'NFKB2', 'SP1', 'RORA',
    'SMAD3', 'FOXO3', 'BCL11B', 'TCF4', 'REL', 'JUN', 'FOSB', 'MYC',
    'IKZF1', 'FLI1', 'KLF2', 'ETS1',
}

# ── helpers ──────────────────────────────────────────────────────────────────
def load_grn(dataset, method):
    path = f"{RESULTS_DIR}/{dataset}/{dataset}.{method}.{method}.prediction.h5ad"
    return ad.read_h5ad(path).uns['prediction']


def get_ibd_gwas_genes():
    """Return set of IBD-associated gene symbols from GWAS Catalog."""
    gwas = pickle.load(open(GWAS_PATH, 'rb'))
    mask = gwas['DISEASE/TRAIT'].str.lower().str.contains(
        'inflammatory bowel|ulcerative colitis|crohn', na=False
    )
    ibd = gwas[mask]
    genes = set()
    for col in ['REPORTED GENE(S)', 'MAPPED_GENE']:
        ibd[col].dropna().apply(
            lambda s: genes.update(
                g.strip() for g in s.replace(' - ', ', ').split(',')
                if g.strip() and g.strip() != 'NR'
            )
        )
    return genes


def _hbar(ax, labels, values, color, highlight_set, highlight_color='#D32F2F'):
    """Horizontal bar chart: rank 1 at top, known TFs highlighted."""
    bar_colors = [colors_blind[1] if lbl in highlight_set else colors_blind[0]
                  for lbl in labels]
    y = range(len(labels))
    ax.barh(list(y), values, color=bar_colors, edgecolor='white', linewidth=0.4, alpha=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()   # rank 1 at top
    ax.spines[['top', 'right']].set_visible(False)


# ── Figure 1: Top 20 TFs by out-degree ───────────────────────────────────────
def fig_top_tfs(dataset_label, dataset_key):
    fig, axes = plt.subplots(1, 4, figsize=(6, 3.7))
    fig.suptitle(
        f'IBD:{dataset_label}',
        fontsize=13, fontweight='bold'
    )
    for i, (ax, method) in enumerate(zip(axes, METHODS)):
        grn = load_grn(dataset_key, method)
        od = grn.groupby('source')['target'].count().sort_values(ascending=False).head(20)
        _hbar(ax, od.index.tolist(), od.values.tolist(), METHOD_COLORS[method], KNOWN_TFS)
        if i == 0:
            ax.set_xlabel('Out-degree (# of targets)', fontsize=9)
        else:
            ax.set_xlabel('')
        ax.set_title(surrogate_names[method],  fontsize=10)

    legend_handles = [
        mpatches.Patch(color=colors_blind[1], label='PBMC/IBD related'),
    ]
    fig.legend(handles=legend_handles, ncol=1,
               fontsize=9, frameon=False, bbox_to_anchor=(.8, .1))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'fig1_top20_tfs_{dataset_label.lower()}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Figure 2: Top 20 IBD GWAS genes by in-degree ─────────────────────────────
def fig_gwas_indegree(dataset_label, dataset_key, gwas_genes):
    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    fig.suptitle(
        f'Top 20 IBD GWAS Genes by In-degree Centrality  ·  IBD:{dataset_label}',
        fontsize=13, fontweight='bold'
    )
    for ax, method in zip(axes, METHODS):
        grn = load_grn(dataset_key, method)
        indeg = grn.groupby('target')['source'].count()
        gwas_indeg = indeg[indeg.index.isin(gwas_genes)].sort_values(ascending=False).head(20)
        if gwas_indeg.empty:
            ax.text(0.5, 0.5, 'No GWAS genes\nin network',
                    ha='center', va='center', transform=ax.transAxes)
        else:
            _hbar(ax, gwas_indeg.index.tolist(), gwas_indeg.values.tolist(),
                  METHOD_COLORS[method], set())
        ax.set_xlabel('In-degree (# regulating TFs)', fontsize=9)
        ax.set_title(surrogate_names[method], fontweight='bold',
                     color=METHOD_COLORS[method], fontsize=11)
        ax.annotate(f'n={len(indeg[indeg.index.isin(gwas_genes)])} GWAS genes\nin network',
                    xy=(0.97, 0.03), xycoords='axes fraction', ha='right',
                    fontsize=7.5, color='grey')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'fig2_gwas_indegree_{dataset_label.lower()}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def fig_centrality_shift():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, method in zip(axes, METHODS):
        grns = {cond: load_grn(ds, method) for cond, ds in DATASETS.items()}
        ods = {cond: grn.groupby('source')['target'].count()
               for cond, grn in grns.items()}

        common = set(ods['UC'].index) & set(ods['CD'].index)
        delta = pd.Series(
            {tf: ods['CD'].get(tf, 0) - ods['UC'].get(tf, 0) for tf in common}
        ).sort_values()

        top_uc = delta.head(10)
        top_cd = delta.tail(10)
        plot_data = pd.concat([top_uc, top_cd.iloc[::-1]])

        colors = ['#2e7d32' if v < 0 else '#c62828' for v in plot_data.values]
        y = range(len(plot_data))
        ax.barh(list(y), plot_data.values, color=colors,
                edgecolor='white', linewidth=0.4)
        ax.set_yticks(list(y))
        labels = [
            f'*{tf}' if tf in KNOWN_TFS else tf
            for tf in plot_data.index
        ]
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.invert_yaxis()
        ax.set_xlabel('IBD: UC vs CD centrality shift', fontsize=9)
        ax.set_title(surrogate_names[method], fontweight='bold', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)

    legend_handles = [
        mpatches.Patch(color='#2e7d32', label='High in UC'),
        mpatches.Patch(color='#c62828', label='High in CD'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    ax.annotate('* known PBMC/IBD TF', xy=(1.01, 0), xycoords='axes fraction',
                fontsize=7, color='grey')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig3_centrality_shift_uc_cd.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── main (placeholder — real main is at bottom) ──────────────────────────────


# ── Figure 4: GWAS TF summary — all models ────────────────────────────────────
def fig_gwas_tf_summary():

    # Method key → display name + color (using project palette)
    METHOD_MAP = {m: (surrogate_names[m], METHOD_COLORS[m]) for m in METHODS}

    # GWAS TFs
    gwas = pickle.load(open(GWAS_PATH, 'rb'))
    def parse_genes(s):
        if pd.isna(s): return []
        return [g.strip() for g in s.replace(' - ', ', ').split(',')
                if g.strip() and g.strip() != 'NR']

    tf_all = set(pd.read_csv('/vol/projects/jnourisa/genernbi/resources/grn_benchmark/prior/tf_all.csv')
                 .iloc[:, 0].astype(str).str.strip())

    def get_gwas_tfs(pattern):
        mask = gwas['DISEASE/TRAIT'].str.lower().str.contains(pattern, na=False)
        genes = set()
        for col in ['REPORTED GENE(S)', 'MAPPED_GENE']:
            gwas[mask][col].dropna().apply(parse_genes).apply(genes.update)
        return genes & tf_all

    uc_tfs = get_gwas_tfs('ulcerative colitis')
    cd_tfs = get_gwas_tfs('crohn')
    uc_only = sorted(uc_tfs - cd_tfs)   # 14
    cd_only = sorted(cd_tfs - uc_tfs)   # 29

    # Compute normalized delta per method: divide by max out-degree in the network
    # so shifts are expressed as a fraction of the network's largest hub
    method_deltas = {}
    for mkey in METHOD_MAP:
        grns = {cond: load_grn(ds, mkey) for cond, ds in DATASETS.items()}
        ods  = {cond: grn.groupby('source')['target'].count() for cond, grn in grns.items()}
        common = set(ods['UC'].index) & set(ods['CD'].index)
        delta = pd.Series({tf: ods['CD'].get(tf, 0) - ods['UC'].get(tf, 0) for tf in common})
        max_deg = max(ods['UC'].max(), ods['CD'].max())
        method_deltas[mkey] = delta / max_deg if max_deg > 0 else delta

    # Keep only TFs present in at least one model's network
    all_present = set().union(*[set(d.index) for d in method_deltas.values()])
    uc_only = [tf for tf in uc_only if tf in all_present]
    cd_only = [tf for tf in cd_only if tf in all_present]

    all_tfs   = uc_only + cd_only
    group_col = ['#2e7d32'] * len(uc_only) + ['#c62828'] * len(cd_only)

    n = len(all_tfs)
    fig, ax = plt.subplots(figsize=(3, 5))
    y_pos = np.arange(len(all_tfs))

    # average shift across models (for bar length)
    avg_shifts = [np.nanmean([method_deltas[m].get(tf, np.nan) for m in METHOD_MAP])
                  for tf in all_tfs]

    # horizontal bars colored by avg shift direction
    bar_colors = ['#2e7d32' if v < 0 else '#c62828' for v in avg_shifts]
    ax.barh(y_pos, avg_shifts, color=bar_colors, alpha=0.30, height=0.6,
            zorder=1, label='_nolegend_')

    # one dot per model per TF (if TF is in that model's network)
    dot_handles = []
    for mkey, (mlabel, mcolor) in METHOD_MAP.items():
        delta_ser = method_deltas[mkey]
        xs = [delta_ser.get(tf, np.nan) for tf in all_tfs]
        ys = list(y_pos)
        valid = [(x, y) for x, y in zip(xs, ys) if not np.isnan(x)]
        if valid:
            vx, vy = zip(*valid)
            sc = ax.scatter(vx, vy, color=mcolor, s=30, zorder=3,
                            edgecolors='white', linewidths=0.3, label=mlabel)
            dot_handles.append(sc)

    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_tfs, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel('UC vs CD centrality shift\n(normalized by max out-degree)', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    # separator line between UC-only and CD-only
    sep = len(uc_only) - 0.5
    ax.axhline(sep, color='grey', linewidth=0.8, linestyle=':')
    ax.text(ax.get_xlim()[0], sep - 0.3, 'UC TFs', fontsize=7,
            color='#2e7d32', ha='left', va='bottom', style='italic')
    ax.text(ax.get_xlim()[0], sep + 0.3, 'CD TFs', fontsize=7,
            color='#c62828', ha='left', va='top', style='italic')

    # legends
    legend1 = ax.legend(handles=dot_handles, title='Model', loc='lower right',
                        fontsize=7, title_fontsize=8, frameon=True)
    ax.add_artist(legend1)
    bar_handles = [
        mpatches.Patch(color='#2e7d32', alpha=0.5, label='High in UC (avg)'),
        mpatches.Patch(color='#c62828', alpha=0.5, label='High in CD (avg)'),
    ]
    ax.legend(handles=bar_handles, loc='upper right', fontsize=7, frameon=True)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig4_gwas_tf_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading IBD GWAS genes...')
    gwas_genes = get_ibd_gwas_genes()
    print(f'  {len(gwas_genes)} IBD-associated genes from GWAS Catalog')

    for label, key in DATASETS.items():
        print(f'\n── Dataset: IBD:{label} ──')
        fig_top_tfs(label, key)
        fig_gwas_indegree(label, key, gwas_genes)

    print('\n── Centrality shift UC→CD ──')
    fig_centrality_shift()

    print('\n── GWAS TF summary (all models) ──')
    fig_gwas_tf_summary()

    print('\nDone.')
