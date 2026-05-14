"""
Post-analysis for causal directionality experiment.
Loads reversed GRN scores (100% edge flip) vs original GRN scores across all datasets
and produces two heatmap summary tables:
  1. methods × datasets  (mean ratio across metrics)
  2. methods × metrics   (mean ratio across datasets)

Usage:
    python src/exp_analysis/post_causal_directionality.py
"""
import os
import sys
import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")

from genernbi_supp.src.helper import load_env
env = load_env()

sys.path.insert(0, env['genernbi_supp_DIR'])
from src.helper import surrogate_names, METHODS, palette_metrics, palette_methods
from task_grn_inference.src.utils.config import DATASETS

RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{RESULTS_DIR}/figs"
os.makedirs(figs_dir, exist_ok=True)

from task_grn_inference.src.utils.config import METRICS as ALL_METRICS

# tfb excluded: direction-sensitive by design (reversed GRN = gene→TF, metric expects TF→gene)
# replicate_consistency excluded: measures stability across replicates, not directionality
SKIP_METRICS = {'tfb_f1', 'gs_f1'} #'replicate_consistency', 'tfb_f1', 'tfb_precision', 'tfb_recall'
METRICS_COLS = [m for m in ALL_METRICS if m not in SKIP_METRICS]
SKIP_METHODS = {'spearman_corr'}

# ── load and compute ratios across all datasets ───────────────────────────────
exp_dir = f"{RESULTS_DIR}/experiment/causal_directionality"
all_scores = pd.read_csv(f"{RESULTS_DIR}/all_scores.csv")

SKIP_DATASETS = {'ibd_cd', 'ibd_uc', '300BCG', 'nakatake', 'soundlife',  'soundlife_vaccine', 'norman'}

records = []
for f in sorted(os.listdir(exp_dir)):
    if not f.endswith('-scores.csv'):
        continue
    dataset = f.split('-direction-')[0]
    if dataset.lower() in SKIP_DATASETS:
        continue
    scores_rev = pd.read_csv(os.path.join(exp_dir, f), index_col=0)
    scores_orig = (all_scores[all_scores['dataset'].str.lower() == dataset.lower()]
                   .drop(columns='dataset').set_index('method'))
    cols = [c for c in METRICS_COLS if c in scores_rev.columns and c in scores_orig.columns]
    common_methods = scores_rev.index.intersection(scores_orig.index).difference(SKIP_METHODS)
    if common_methods.empty or not cols:
        continue
    rev  = scores_rev.loc[common_methods, cols]
    orig = scores_orig.loc[common_methods, cols]
    ratio = (rev / orig.replace(0, np.nan)).abs()
    ratio = ratio.reset_index().melt(id_vars='index', var_name='metric', value_name='ratio')
    ratio.columns = ['method', 'metric', 'ratio']
    ratio['dataset'] = dataset
    records.append(ratio)

df = pd.concat(records, ignore_index=True)
df['method_name']  = df['method'].map(lambda x: surrogate_names.get(x, x))
df['metric_name']  = df['metric'].map(lambda x: surrogate_names.get(x, x))
df['dataset_name'] = df['dataset'].map(lambda x: surrogate_names.get(x, x))

# method order from global METHODS list
ordered_methods = list(dict.fromkeys(surrogate_names.get(m, m) for m in METHODS))

def _sort_methods(pivot):
    idx = [m for m in ordered_methods if m in pivot.index]
    idx += [m for m in pivot.index if m not in idx]
    return pivot.reindex(idx)

# cap value for colour scale (outliers don't dominate)
CAP = 2.0

def _heatmap(pivot, title, out_path, xlabel):
    nan_mask = pivot.isna()
    pivot_display = pivot.clip(upper=CAP).fillna(0)
    annot_data = pivot.clip(upper=CAP).applymap(lambda v: '' if pd.isna(v) else f'{v:.2f}')

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'rg', ['#d73027', '#ffffbf', '#1a9850'])   # red (0) → yellow (1) → green (2)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(pivot_display, ax=ax, cmap=cmap, vmin=0, vmax=CAP,
                mask=nan_mask,
                annot=annot_data, fmt='', linewidths=0.4, linecolor='white',
                annot_kws={'size': 7},
                cbar_kws={'label': 'Sensitivity', 'shrink': 0.6, 'aspect': 20,
                          'ticks': [0, 1, 2]})

    # mark cells that were capped
    for (i, j), val in np.ndenumerate(pivot.values):
        if val > CAP:
            pass  # capped cells shown via colour scale alone

    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    plt.xticks(rotation=45, ha='right')
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)

# ordered column lists (surrogate names)
ordered_datasets = [surrogate_names.get(d, d) for d in DATASETS
                    if d.lower() not in SKIP_DATASETS]
ordered_metrics  = [surrogate_names.get(m, m) for m in ALL_METRICS if m not in SKIP_METRICS]


def get_causal_directionality_pivots():
    """Return (pivot_datasets, pivot_metrics, CAP) for the two causal directionality heatmaps."""
    pivot_ds = (df.groupby(['method_name', 'dataset_name'])['ratio']
                  .mean().unstack('dataset_name'))
    pivot_ds = _sort_methods(pivot_ds)
    pivot_ds = pivot_ds.reindex(columns=[d for d in ordered_datasets if d in pivot_ds.columns])

    pivot_met = (df.groupby(['method_name', 'metric_name'])['ratio']
                   .mean().unstack('metric_name'))
    pivot_met = _sort_methods(pivot_met)
    pivot_met = pivot_met.reindex(columns=[m for m in ordered_metrics if m in pivot_met.columns])

    return pivot_ds, pivot_met, CAP


if __name__ == '__main__':
    pivot_ds, pivot_met, _ = get_causal_directionality_pivots()

    # ── Table 1: methods × datasets ───────────────────────────────────────────────
    _heatmap(pivot_ds,
             title='Causal directionality — methods × datasets\n(ratio reversed / original, mean over metrics)',
             out_path=f"{figs_dir}/causal_directionality_methods_datasets.png",
             xlabel='Dataset')

    # ── Table 2: methods × metrics ────────────────────────────────────────────────
    _heatmap(pivot_met,
             title='Causal directionality — methods × metrics\n(ratio reversed / original, mean over datasets)',
             out_path=f"{figs_dir}/causal_directionality_methods_metrics.png",
             xlabel='Metric')

    # ── Fraction degraded (ratio < 1) plots ──────────────────────────────────────
    def _fraction_bar(frac_series, n_series, out_path, entity_label, palette=None):
        """Horizontal bar chart: fraction of degraded combinations per entity."""
        frac = frac_series.sort_values(ascending=False)
        n    = n_series.reindex(frac.index)

        _FALLBACK = '#aab4be'
        colors = [palette.get(name, _FALLBACK) if palette else _FALLBACK for name in frac.index]

        n_items = len(frac)
        fig, ax = plt.subplots(figsize=(3.5, 0.22 * n_items + 0.8))
        bars = ax.barh(frac.index, frac.values, color=colors, edgecolor='white', height=0.65)

        # annotate with count
        for bar, (name, f), nt in zip(bars, frac.items(), n.values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{int(round(f * nt))}/{int(nt)}',
                    va='center', ha='left', fontsize=9)

        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlim(0, 1.22)
        ax.set_xlabel('Sensitivity')
        ax.set_ylabel(entity_label)
        ax.invert_yaxis()
        for side in ['right', 'top']:
            ax.spines[side].set_visible(False)
        plt.tight_layout()
        fig.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}", flush=True)

    df['degraded'] = df['ratio'] < 1.0

    # per method: fraction of (dataset × metric) pairs that degrade
    method_stats = df.groupby('method_name')['degraded'].agg(frac='mean', n='count')
    method_stats = method_stats.reindex([m for m in ordered_methods if m in method_stats.index]
                                        + [m for m in method_stats.index if m not in ordered_methods])
    _fraction_bar(method_stats['frac'], method_stats['n'],
                  out_path=f"{figs_dir}/causal_directionality_frac_methods.png",
                  entity_label='Method', palette=palette_methods)

    # per metric: fraction of (dataset × method) pairs that degrade
    metric_stats = df.groupby('metric_name')['degraded'].agg(frac='mean', n='count')
    _fraction_bar(metric_stats['frac'], metric_stats['n'],
                  out_path=f"{figs_dir}/causal_directionality_frac_metrics.png",
                  entity_label='Metric', palette=palette_metrics)

    # ── Subset barplot helper ─────────────────────────────────────────────────────
    def _subset_barplot(dataset, methods, out_path):
        _rev_file = [f for f in os.listdir(exp_dir)
                     if f.lower() == f"{dataset.lower()}-direction-100-scores.csv"][0]
        _scores_rev  = pd.read_csv(os.path.join(exp_dir, _rev_file), index_col=0)
        _scores_orig = (all_scores[all_scores['dataset'].str.lower() == dataset.lower()]
                        .drop(columns='dataset').set_index('method'))

        _cols   = [c for c in METRICS_COLS if c in _scores_rev.columns and c in _scores_orig.columns]
        _subset = [m for m in methods if m in _scores_rev.index and m in _scores_orig.index]

        _rev  = _scores_rev.loc[_subset, _cols]
        _orig = _scores_orig.loc[_subset, _cols]
        _ratio = (_rev / _orig.replace(0, np.nan)).abs()

        _ratio.index   = _ratio.index.map(lambda x: surrogate_names.get(x, x))
        _ratio.columns = _ratio.columns.map(lambda x: surrogate_names.get(x, x))

        CAP_local = float(np.nanmax(_ratio.values[np.isfinite(_ratio.values)]))
        _ratio_capped = _ratio.clip(upper=CAP_local)
        _capped_mask  = _ratio > CAP_local

        _long = (_ratio_capped.reset_index()
                 .melt(id_vars='index', var_name='Metric', value_name='Relative score')
                 .rename(columns={'index': 'Method'}))

        n_methods = len(_subset)
        n_metrics = len(_cols)
        fig, ax = plt.subplots(figsize=(.7 * n_methods * n_metrics * 0.22 + 1.5, 2.2))

        _pal = {m: palette_metrics[m] for m in _long['Metric'].unique() if m in palette_metrics}
        sns.barplot(_long, x='Method', y='Relative score', hue='Metric', ax=ax, palette=_pal)

        for patch, (_, row) in zip(ax.patches, _long.iterrows()):
            if row['Method'] in _capped_mask.index and row['Metric'] in _capped_mask.columns:
                if _capped_mask.loc[row['Method'], row['Metric']]:
                    ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.03,
                            '▲', ha='center', va='bottom', fontsize=7, color='black')

        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_xlabel('')
        ax.set_ylabel('Relative performance \n (reversed / original)', labelpad=10)
        ax.set_ylim(0, _ratio_capped.values[np.isfinite(_ratio_capped.values)].max() * 1.15)
        ax.margins(x=0.1)
        for side in ['right', 'top']:
            ax.spines[side].set_visible(False)
        ax.legend(title='Metric', loc=(1.02, 0), frameon=False, fontsize=9)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}", flush=True)

    _subset_barplot(
        dataset='op',
        methods=['scenicplus', 'celloracle', 'grnboost', 'pearson_corr', 'ppcor'],
        out_path=f"{figs_dir}/causal_directionality_subset_op.png")

    _subset_barplot(
        dataset='replogle',
        methods=['scenic', 'grnboost', 'pearson_corr', 'ppcor', 'scprint'],
        out_path=f"{figs_dir}/causal_directionality_subset_replogle.png")

    _subset_barplot(
        dataset='300BCG',
        methods=['scenic', 'grnboost', 'pearson_corr', 'ppcor', 'scprint'],
        out_path=f"{figs_dir}/causal_directionality_subset_bcg.png")
