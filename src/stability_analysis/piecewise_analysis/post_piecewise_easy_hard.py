"""
Easy/Hard-to-score gene and TF analysis for piecewise GRN evaluation.

Produces five figures:
  1. Gene expression vs gene-wise R² — one axis per model
  2. Gene std (expression variability) vs gene-wise R² — one axis per model
  3. Gene expression per PBMC cell type vs R² — one axis per cell type
  4. Enrichment analysis for easy and hard genes — combined gene sets, one plot each
  5. TF global KD fold-change vs WS score — easy/hard coloured
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as ss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scanpy as sc
import gseapy as gp
warnings.filterwarnings('ignore')

from genernbi_supp.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
FIGS_DIR    = f"{RESULTS_DIR}/figs"
os.makedirs(FIGS_DIR, exist_ok=True)

sys.path.append(env['genernbi_supp_DIR'])
from src.helper import surrogate_names, palette_methods

# ── Constants ─────────────────────────────────────────────────────────────────
EASY_COLOR = '#2ca02c'
MID_COLOR  = '#d9d9d9'
HARD_COLOR = '#d62728'
PALETTE    = {'easy': EASY_COLOR, 'mid': MID_COLOR, 'hard': HARD_COLOR}

MODELS_4 = ['Pearson Corr.', 'Scenic+', 'GRNBoost2', 'PPCOR']

MODEL_NAMES = {
    'scenicplus':   'Scenic+',
    'pearson_corr': 'Pearson Corr.',
    'grnboost':     'GRNBoost2',
    'ppcor':        'PPCOR',
    'scenic':       'Scenic',
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def classify_groups(series, easy_q=0.75, hard_q=0.25):
    hi = series.quantile(easy_q)
    lo = series.quantile(hard_q)
    return pd.cut(series,
                  bins=[-np.inf, lo, hi, np.inf],
                  labels=['hard', 'mid', 'easy'])


def _spearman_label(x, y, show_p=False):
    mask = ~(np.isnan(x) | np.isnan(y))
    r, p = ss.spearmanr(x[mask], y[mask])
    if show_p:
        p_str = f'{p:.1e}' if p < 0.001 else f'{p:.3f}'
        return f'r={r:.2f}, p={p_str}'
    return f'r={r:.2f}'


def _scatter_group(ax, df, x_col, y_col, alpha=0.35, s=6):
    """Scatter plot coloured by easy/mid/hard group."""
    for grp in ['mid', 'hard', 'easy']:   # draw mid first so easy/hard on top
        sub = df[df['group'] == grp]
        ax.scatter(sub[x_col], sub[y_col],
                   c=PALETTE[grp], s=s, alpha=alpha,
                   linewidths=0, rasterized=True)
    ax.spines[['right', 'top']].set_visible(False)


def _legend_patches(ax, loc='best'):
    handles = [mpatches.Patch(color=PALETTE[g], label=g.capitalize())
               for g in ['easy', 'hard']]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc=loc,
              markerscale=1.5)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _load_gene_data():
    """Return per-gene table with R² per model, expression, std, cell-type means, group."""
    reg = pd.read_csv(
        f'{RESULTS_DIR}/experiment/metrics_stability/op_regression.csv',
        index_col=0)
    reg['model'] = reg['model'].map(MODEL_NAMES)
    reg_raw = reg[reg['theta'] == 'r2_raw'].copy()

    # Mean R² per gene (for classification)
    gene_mean_r2 = (reg_raw.groupby('gene')['r2']
                    .mean().rename('mean_r2').reset_index())
    gene_mean_r2['group'] = classify_groups(gene_mean_r2['mean_r2'])

    # Per-model R² pivot
    r2_pivot = reg_raw.pivot_table(
        index='gene', columns='model', values='r2', aggfunc='mean')

    # Expression data from op_bulk
    adata = sc.read(
        f'{RESULTS_DIR}/../grn_benchmark/evaluation_data/op_bulk.h5ad')
    X = adata.layers['lognorm']
    if sp.issparse(X): X = X.toarray()

    gene_mean_expr = pd.Series(X.mean(axis=0),  index=adata.var_names, name='mean_expr')
    gene_std_expr  = pd.Series(X.std(axis=0),   index=adata.var_names, name='std_expr')

    # Mean expression per cell type
    cell_types = adata.obs['cell_type'].unique()
    ct_means = {}
    for ct in cell_types:
        mask = adata.obs['cell_type'] == ct
        ct_means[ct] = X[mask, :].mean(axis=0)
    ct_df = pd.DataFrame(ct_means, index=adata.var_names)   # genes × cell_types

    # Assemble
    df = (gene_mean_r2
          .set_index('gene')
          .join(r2_pivot)
          .join(gene_mean_expr)
          .join(gene_std_expr)
          .join(ct_df)
          .reset_index())
    return df, cell_types


def _load_tf_data():
    """Return per-TF table with mean WS, group, KD global log2FC."""
    ws = pd.read_csv(
        f'{RESULTS_DIR}/experiment/metrics_stability/replogle_ws.csv',
        index_col=0)
    ws['model'] = ws['model'].map(MODEL_NAMES)
    ws_theta = ws[ws['theta'] != 'ws_raw'].copy()

    tf_mean = (ws_theta.groupby('tf')['ws_distance']
               .mean().rename('mean_ws').reset_index())
    tf_mean['group'] = classify_groups(tf_mean['mean_ws'])

    perturb = pd.read_csv(f'{RESULTS_DIR}/exp_analysis/perturb_effect_all.csv')
    perturb_rep = (perturb[perturb['Dataset'] == 'replogle']
                   [['perturbation', 'Expression fold change']]
                   .drop_duplicates()
                   .rename(columns={'Expression fold change': 'global_log2fc_kd'}))
    tf_mean = tf_mean.merge(perturb_rep, left_on='tf', right_on='perturbation', how='inner')
    return tf_mean


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Gene expression vs R² per model
# ══════════════════════════════════════════════════════════════════════════════
def plot_expression_vs_r2(df, out_path):
    models = [m for m in MODELS_4 if m in df.columns]
    fig, axes = plt.subplots(1, len(models),
                             figsize=(3.2 * len(models), 3.2), sharey=False)
    for ax, model in zip(axes, models):
        sub = df[['gene', 'group', 'mean_expr', model]].dropna()
        _scatter_group(ax, sub.rename(columns={model: 'r2'}),
                       x_col='mean_expr', y_col='r2')
        ax.set_xlabel('Mean expression\n(log-norm)', fontsize=9)
        ax.set_ylabel('R²' if ax == axes[0] else '', fontsize=9)
        ax.set_title(model, fontsize=9)
        lbl = _spearman_label(sub['mean_expr'].values, sub[model].values)
        # ax.text removed
    _legend_patches(axes[-1], loc='lower right')
    plt.suptitle('Gene expression vs regression R² per model', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → {os.path.abspath(out_path)}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Gene expression std vs R² per model
# ══════════════════════════════════════════════════════════════════════════════
def plot_std_vs_r2(df, out_path):
    models = [m for m in MODELS_4 if m in df.columns]
    fig, axes = plt.subplots(1, len(models),
                             figsize=(3.2 * len(models), 3.2), sharey=False)
    for ax, model in zip(axes, models):
        sub = df[['gene', 'group', 'std_expr', model]].dropna()
        _scatter_group(ax, sub.rename(columns={model: 'r2'}),
                       x_col='std_expr', y_col='r2')
        ax.set_xlabel('Std. of expression\n(log-norm)', fontsize=9)
        ax.set_ylabel('R²' if ax == axes[0] else '', fontsize=9)
        ax.set_title(model, fontsize=9)
        lbl = _spearman_label(sub['std_expr'].values, sub[model].values)
        # ax.text removed
    _legend_patches(axes[-1], loc='lower right')
    plt.suptitle('Gene expression variability vs regression R² per model', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → {os.path.abspath(out_path)}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Cell-type expression vs R² (one ax per PBMC subtype)
# ══════════════════════════════════════════════════════════════════════════════
def plot_celltype_expr_vs_r2(df, cell_types, out_path):
    ct_list = [ct for ct in cell_types if ct in df.columns]
    fig, axes = plt.subplots(1, len(ct_list),
                             figsize=(3.2 * len(ct_list), 3.2), sharey=False)
    if len(ct_list) == 1:
        axes = [axes]
    for ax, ct in zip(axes, ct_list):
        sub = df[['gene', 'group', 'mean_r2', ct]].dropna()
        _scatter_group(ax, sub.rename(columns={ct: 'ct_expr'}),
                       x_col='ct_expr', y_col='mean_r2')
        ax.set_xlabel(f'Mean expression\n({ct})', fontsize=9)
        ax.set_ylabel('Mean R² (all models)' if ax == axes[0] else '', fontsize=9)
        ax.set_title(ct, fontsize=9)
        lbl = _spearman_label(sub[ct].values, sub['mean_r2'].values)
        # ax.text removed
    _legend_patches(axes[-1], loc='lower right')
    plt.suptitle('Cell-type expression vs mean R² across models', fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → {os.path.abspath(out_path)}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Combined enrichment (easy / hard), all gene-set sources merged
# ══════════════════════════════════════════════════════════════════════════════
def _run_combined_enrichr(gene_list, n_top=10):
    """Return combined top-N terms from GO-BP, KEGG, Reactome."""
    dbs = ['GO_Biological_Process_2023', 'KEGG_2021_Human', 'Reactome_Pathways_2024']
    frames = []
    for db in dbs:
        try:
            enr = gp.enrichr(gene_list=gene_list, gene_sets=db,
                              organism='human', outdir=None, verbose=False)
            df = enr.results[enr.results['Adjusted P-value'] < 0.05].copy()
            df['db'] = db
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values('Adjusted P-value').head(n_top)
    combined['-log10(padj)'] = -np.log10(combined['Adjusted P-value'] + 1e-300)
    combined['count'] = combined['Overlap'].str.split('/').str[0].astype(int)
    # Shorten GO term names (strip GO ID)
    combined['Term_short'] = (combined['Term']
                              .str.replace(r'\s*\(GO:\d+\)', '', regex=True)
                              .str[:30])
    return combined


def plot_enrichment_combined(easy_genes, hard_genes, out_dir):
    for gene_list, label, fname in [
        (easy_genes, 'Easy genes', 'enrichment_easy_genes.png'),
        (hard_genes, 'Hard genes', 'enrichment_hard_genes.png'),
    ]:
        df = _run_combined_enrichr(gene_list, n_top=15)
        if df.empty:
            print(f"  [{label}] No significant enrichment terms.")
            continue
        df = df.sort_values('-log10(padj)').reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(4, 3))

        sig = df['-log10(padj)'].values
        counts = df['count'].values
        norm = plt.Normalize(sig.min(), sig.max())
        cmap = plt.cm.YlOrRd
        colors = cmap(norm(sig))

        count_min, count_max = counts.min(), counts.max()
        denom = max(count_max - count_min, 1)
        sizes = 50 + 200 * (counts - count_min) / denom

        for i, x in enumerate(sig):
            ax.plot([0, x], [i, i], color='#cccccc', lw=1, zorder=1)

        sc = ax.scatter(sig, range(len(df)), 
                        c='green', 
                        alpha=0.7,
                        # cmap='YlOrRd',
                        norm=norm, s=sizes, zorder=2,
                        edgecolors='white', linewidth=0.5)

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Term_short'], fontsize=8)
        ax.set_xlabel('−log₁₀(adjusted p-value)', fontsize=9)
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
        ax.margins(x=0.1, y=0.1)

        # Size legend for gene count — spaced out vertically
        legend_counts = sorted({count_min, int(np.median(counts)), count_max})
        handles = [
            plt.scatter([], [], s=50 + 200 * (cnt - count_min) / denom,
                        c='#aaaaaa', label=f'{cnt}')
            for cnt in legend_counts
        ]
        ax.legend(handles=handles, title='Gene count', frameon=False,
                  fontsize=7, title_fontsize=8, loc='lower right',
                  labelspacing=1.2, bbox_to_anchor=(1.0, 0.15))

        plt.tight_layout()
        out = os.path.join(out_dir, fname)
        plt.savefig(out, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"  → {os.path.abspath(out)}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — TF KD global fold-change vs WS score
# ══════════════════════════════════════════════════════════════════════════════
def plot_tf_kd_vs_ws(tf_df, out_path):
    tf_df = tf_df.copy()
    tf_df['abs_log2fc'] = tf_df['global_log2fc_kd'].abs()

    # Print top-5 easy (lowest WS) and top-5 hard (highest WS)
    top5_easy = tf_df[tf_df['group'] == 'easy'].nlargest(5, 'mean_ws')['tf'].tolist()
    top5_hard = tf_df[tf_df['group'] == 'hard'].nsmallest(5, 'mean_ws')['tf'].tolist()
    print(f"  Top-5 easy TFs (lowest WS):  {top5_easy}")
    print(f"  Top-5 hard TFs (highest WS): {top5_hard}")

    fig, ax = plt.subplots(figsize=(4, 2.5))
    for grp in ['mid', 'hard', 'easy']:
        sub = tf_df[tf_df['group'] == grp]
        label = {'easy': 'Easy-to-score', 'hard': 'Hard-to-score', 'mid': None}[grp]
        ax.scatter(sub['abs_log2fc'], sub['mean_ws'],
                   c=PALETTE[grp], s=35, alpha=0.85,
                   edgecolors='white', linewidth=0.4,
                   label=label,
                   zorder={'mid': 1, 'hard': 2, 'easy': 3}[grp])

    ax.set_xlabel('log2FC', fontsize=9)
    ax.set_ylabel('WS distance', fontsize=9)
    ax.margins(x=0.1, y=0.1)
    ax.legend(frameon=False, fontsize=8, loc=[1.02, 0.5])
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → {os.path.abspath(out_path)}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
def run_easy_hard_analysis():
    out_dir = f'{RESULTS_DIR}/figs/easy_hard'
    os.makedirs(out_dir, exist_ok=True)

    print("Loading gene data…")
    gene_df, cell_types = _load_gene_data()

    easy_genes = gene_df[gene_df['group'] == 'easy']['gene'].tolist()
    hard_genes  = gene_df[gene_df['group'] == 'hard']['gene'].tolist()
    print(f"  Easy genes: {len(easy_genes)}, Hard genes: {len(hard_genes)}")

    print("Loading TF data…")
    tf_df = _load_tf_data()
    print(f"  TFs with KD data: {len(tf_df)}")

    print("Fig 4 — combined enrichment (easy + hard)")
    plot_enrichment_combined(easy_genes, hard_genes, out_dir)

    

    print("Fig 5 — TF KD effect vs WS score")
    plot_tf_kd_vs_ws(tf_df, f'{out_dir}/tf_kd_vs_ws_score.png')

    print("\nFig 1 — expression vs R² per model")
    plot_expression_vs_r2(gene_df, f'{out_dir}/gene_expression_vs_r2_per_model.png')

    print("Fig 2 — std vs R² per model")
    plot_std_vs_r2(gene_df, f'{out_dir}/gene_std_vs_r2_per_model.png')

    print("Fig 3 — cell-type expression vs R²")
    plot_celltype_expr_vs_r2(gene_df, cell_types, f'{out_dir}/gene_celltype_expr_vs_r2.png')

    

    print(f"\n✓ All figures saved to {os.path.abspath(out_dir)}")
if __name__ == '__main__':
    run_easy_hard_analysis()