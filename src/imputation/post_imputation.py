"""
Post-analysis for imputation experiment.
Plots relative performance of imputed vs. single-cell GRNs across datasets.

Outputs (all in figs/imputation/):
  - imputation_{dataset}.png       : per-dataset barplot (KNN vs Magic, metrics as hue)
  - imputation_heatmap_{method}.png: heatmap of relative perf (metrics x datasets) per imputation method

Also outputs (in figs/):
  - imputation_{inference_method}.png: avg barplot across datasets

Usage:
    python src/imputation/post_imputation.py
"""
import os
import sys
import matplotlib.colors as mcolors
import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from genernbi_supp.src.helper import load_env
env = load_env()

sys.path.insert(0, env['TASK_GRN_INFERENCE_DIR'])
from src.utils.config import METRICS as FINAL_METRICS
from genernbi_supp.src.helper import surrogate_names, colors_blind

RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{RESULTS_DIR}/figs"
figs_imputation_dir = f"{figs_dir}/imputation"
os.makedirs(figs_dir, exist_ok=True)
os.makedirs(figs_imputation_dir, exist_ok=True)

DATASETS = ["op", "300BCG", "norman", "soundlife_vaccine"]  # ibd datasets excluded
exp_dir = f"{RESULTS_DIR}/experiment/imputation"

frames = []
for dataset in DATASETS:
    csv_path = f"{exp_dir}/metrics_{dataset}.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping {dataset}: {csv_path} not found")
        continue
    df = pd.read_csv(csv_path)
    df['dataset'] = dataset
    frames.append(df)

rr_raw = pd.concat(frames, ignore_index=True)
metric_raw_cols = [m for m in FINAL_METRICS if m in rr_raw.columns]
rr_raw = rr_raw[['dataset', 'imputation_method', 'inference_method'] + metric_raw_cols]

imputation_mapping = {'original': 'Single-cell', 'magic': 'Magic imput.', 'knn': 'KNN imput.'}
rr_raw['imputation_method'] = rr_raw['imputation_method'].map(imputation_mapping)

renamed_metrics = [surrogate_names.get(m, m) for m in metric_raw_cols]
rr_raw.columns = ['dataset', 'imputation_method', 'inference_method'] + renamed_metrics
metric_cols = renamed_metrics

def compute_relative(df_method):
    """Average relative performance (imputed / single-cell) across datasets."""
    rows = []
    for ds, grp in df_method.groupby('dataset'):
        baseline_rows = grp[grp['imputation_method'] == 'Single-cell']
        if baseline_rows.empty:
            continue
        baseline = baseline_rows[metric_cols].iloc[0]
        imput = grp[grp['imputation_method'] != 'Single-cell'].copy()
        norm = imput[metric_cols].div(baseline).clip(upper=2.0)
        norm['imputation_method'] = imput['imputation_method'].values
        rows.append(norm)
    if not rows:
        return None
    combined = pd.concat(rows, ignore_index=True)
    return combined.groupby('imputation_method')[metric_cols].mean().reset_index()

def plot_imput_vs_singlecell(ax, df_avg):
    long = df_avg.melt(id_vars='imputation_method', var_name='Metric', value_name='value')
    sns.barplot(long, x='imputation_method', y='value', hue='Metric', ax=ax, palette=colors_blind)
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('')
    ax.set_ylabel('Relative performance\n(imputed / single-cell)')
    ax.margins(x=0.1, y=0.15)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc=(1.02, 0.1), title='Metric', frameon=False, fontsize=6, title_fontsize=7)

KEEP_METHODS = ['pearson_corr']


def get_imputation_heatmap_pivot(imput_key, cap=2.0):
    """Return (pivot_clipped, nan_mask) for a given imputation method ('knn' or 'magic')."""
    imput_label_map = {'knn': 'KNN imput.', 'magic': 'Magic imput.'}
    imput_label = imput_label_map[imput_key]
    heatmap_data = {}
    for dataset in DATASETS:
        df_ds = rr_raw[
            (rr_raw['dataset'] == dataset) &
            (rr_raw['inference_method'].isin(KEEP_METHODS))
        ]
        baseline_rows = df_ds[df_ds['imputation_method'] == 'Single-cell']
        imput_rows = df_ds[df_ds['imputation_method'] == imput_label]
        if baseline_rows.empty or imput_rows.empty:
            continue
        baseline = baseline_rows[metric_cols].iloc[0]
        ratio = (imput_rows[metric_cols].iloc[0] / baseline).clip(upper=cap)
        heatmap_data[surrogate_names.get(dataset, dataset)] = ratio
    pivot = pd.DataFrame(heatmap_data)
    return pivot.clip(upper=cap), pivot.isna()


if __name__ == '__main__':
    inference_methods = [m for m in rr_raw['inference_method'].unique() if m in KEEP_METHODS]

    for method in inference_methods:
        df_method = rr_raw[rr_raw['inference_method'] == method].copy()
        df_avg = compute_relative(df_method)
        if df_avg is None:
            print(f"Skipping {method}: no 'original' baseline found in any dataset")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        plot_imput_vs_singlecell(ax, df_avg)
        plt.tight_layout()

        out = f"{figs_dir}/imputation_{method}.png"
        fig.savefig(out, dpi=300, transparent=True, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")

    # --- Per-dataset barplots (KNN and Magic side by side, metrics as hue) ---
    imput_display = {'KNN imput.': 'KNN', 'Magic imput.': 'Magic'}

    for dataset in DATASETS:
        df_ds = rr_raw[(rr_raw['dataset'] == dataset) & (rr_raw['inference_method'].isin(KEEP_METHODS))].copy()
        baseline_rows = df_ds[df_ds['imputation_method'] == 'Single-cell']
        if baseline_rows.empty:
            continue
        baseline = baseline_rows[metric_cols].iloc[0]
        df_imput = df_ds[df_ds['imputation_method'] != 'Single-cell'].copy()
        norm = df_imput[metric_cols].div(baseline).clip(upper=2.0)
        norm['imputation_method'] = df_imput['imputation_method'].map(imput_display).values
        long = norm.melt(id_vars='imputation_method', var_name='Metric', value_name='value')

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        sns.barplot(long, x='imputation_method', y='value', hue='Metric', ax=ax, palette=colors_blind)
        ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_title(surrogate_names.get(dataset, dataset), fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Relative performance\n(imputed / single-cell)')
        ax.margins(x=0.1, y=0.15)
        ax.spines[['right', 'top']].set_visible(False)
        ax.legend(loc=(1.02, 0.1), title='Metric', frameon=False, fontsize=6, title_fontsize=7)
        plt.tight_layout()
        out = f"{figs_imputation_dir}/imputation_{dataset}.png"
        fig.savefig(out, dpi=300, transparent=True, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")

    # --- Heatmaps: one per imputation method (metrics x datasets) ---
    CAP = 2.0
    cmap = mcolors.LinearSegmentedColormap.from_list('rg', ['#d73027', '#ffffbf', '#1a9850'])
    imputation_methods_plot = [('knn', 'knn'), ('magic', 'magic')]

    for imput_label, imput_key in imputation_methods_plot:
        pivot, nan_mask = get_imputation_heatmap_pivot(imput_key, cap=CAP)

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(3, len(pivot.columns) * 0.9 + 1.5), 3.5))
        sns.heatmap(
            pivot, ax=ax, cmap=cmap, vmin=0, vmax=CAP,
            mask=nan_mask,
            annot=pivot.applymap(lambda v: '' if pd.isna(v) else f'{v:.2f}'),
            fmt='', linewidths=0.4, linecolor='white',
            annot_kws={'size': 7},
            cbar_kws={
                'label': 'Relative performance\n(imputed / single-cell)',
                'shrink': 0.6, 'aspect': 20, 'ticks': [0, 1, 2]
            }
        )
        ax.set_title(f'Imputation: {imput_key.upper()}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('')
        plt.xticks(rotation=35, ha='right', fontsize=8)
        ax.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        out = f"{figs_imputation_dir}/imputation_heatmap_{imput_key}.png"
        fig.savefig(out, dpi=300, transparent=True, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")
