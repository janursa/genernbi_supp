import pandas as pd
import numpy as np
import anndata as ad
import tqdm
import json
import warnings
import matplotlib
import sys
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scanpy as sc 
import itertools
import warnings
import os
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from geneRNBI.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = F"{env['RESULTS_DIR']}/figs"

from geneRNBI.src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores, palette_metrics
from task_grn_inference.src.utils.config import METRICS

DATASETS = ['op', '300BCG', 'norman']  # ibd datasets excluded
dataset_labels = {'op': 'OPSCA', '300BCG': '300BCG', 'norman': 'Norman'}

def load_scores(dataset, method=None):
    df = pd.read_csv(f'{RESULTS_DIR}/experiment/granular_pseudobulk/metrics_{dataset}.csv')
    if 'dataset' in df.columns:
        df = df.drop(columns=['dataset'])
    if method is not None and 'method' in df.columns:
        df = df[df['method'] == method]
    df = df[[c for c in METRICS if c in df.columns] + ['granularity']]
    df.columns = df.columns.map(lambda name: surrogate_names.get(name, name))
    mask = df['granularity'] == -1
    df.loc[mask, 'granularity'] = np.inf
    df = df.sort_values(by='granularity', ascending=False)
    df.index = df['granularity'].map(lambda name: 'Original' if name == np.inf else name)
    df = df.drop(columns=['granularity'])
    return df

def get_methods(dataset):
    df = pd.read_csv(f'{RESULTS_DIR}/experiment/granular_pseudobulk/metrics_{dataset}.csv')
    if 'method' not in df.columns:
        return [None]
    methods = [m for m in df['method'].unique() if m != 'portia']
    ordered = [m for m in ['pearson_corr', 'grnboost'] if m in methods]
    return ordered if ordered else methods

def plot_line_pseudobulking_effect(scores_mat, ax, show_ylabel=True, show_xlabel=False):
    scores_mat = scores_mat.reset_index()
    df_melted = scores_mat.melt(
        id_vars='granularity', var_name='Metric', value_name='Score'
    )

    def normalize_by_original(group):
        original_value = group[group['granularity'] == 'Original']['Score'].values
        if len(original_value) > 0 and original_value[0] != 0:
            group['Normalized Score'] = group['Score'] / original_value[0]
        else:
            group['Normalized Score'] = group['Score']
        return group

    df_melted = df_melted.groupby('Metric', group_keys=False).apply(normalize_by_original)
    granularities = sorted([x for x in df_melted['granularity'].unique() if x != 'Original'], reverse=True)
    position_map = {'Original': 0}
    for i, gran in enumerate(granularities, start=1):
        position_map[gran] = i
    df_melted['x_position'] = df_melted['granularity'].map(position_map)

    df_melted['Normalized Score'] = df_melted['Normalized Score'].clip(upper=2)

    sns.lineplot(
        data=df_melted,
        x='x_position',
        y='Normalized Score',
        hue='Metric',
        marker='o',
        markersize=3,
        ax=ax,
        palette=palette_metrics,
        legend=False,
    )
    ax.set_ylabel('Relative performance' if show_ylabel else '', fontsize=8)
    ax.set_xlabel('Clustering granularity (larger = finer)' if show_xlabel else '', fontsize=8)
    ax.margins(y=0.1)
    xticks = list(range(len(position_map)))
    xtick_labels = ['SC'] + [str(int(np.round(float(g)))) for g in granularities]
    max_shown = 5
    step = max(1, len(xtick_labels) // max_shown)
    sparse_labels = [label if i % step == 0 else '' for i, label in enumerate(xtick_labels)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(sparse_labels, rotation=45, ha='right', fontsize=7)
    ax.spines[['right', 'top']].set_visible(False)
    return df_melted  # return for legend extraction

def get_granularity_heatmap_pivot(cap=2.0):
    """Return (pivot_clipped, nan_mask) for the granularity heatmap (datasets panel)."""
    heatmap_data = {}
    for dataset in DATASETS:
        df = pd.read_csv(f'{RESULTS_DIR}/experiment/granular_pseudobulk/metrics_{dataset}.csv')
        if 'dataset' in df.columns:
            df = df.drop(columns=['dataset'])
        metric_cols_loc = [c for c in METRICS if c in df.columns]
        if 'method' in df.columns:
            df_avg = df.groupby('granularity')[metric_cols_loc].mean().reset_index()
        else:
            df_avg = df[metric_cols_loc + ['granularity']]
        df_avg.columns = df_avg.columns.map(lambda name: surrogate_names.get(name, name))
        sc_row = df_avg[df_avg['granularity'] == -1].drop(columns='granularity').iloc[0]
        pseudobulk_rows = df_avg[df_avg['granularity'] != -1].drop(columns='granularity')
        ratio = pseudobulk_rows.mean() / sc_row
        heatmap_data[dataset_labels[dataset]] = ratio
    pivot = pd.DataFrame(heatmap_data).T.T
    return pivot.clip(upper=cap), pivot.isna()


if __name__ == '__main__':
    # Collect which datasets have each method
    all_methods = ['pearson_corr', 'grnboost']
    method_datasets = {}
    for method in all_methods:
        ds_list = [d for d in DATASETS if method in (get_methods(d) or [None])]
        if ds_list:
            method_datasets[method] = ds_list

    for method, ds_list in method_datasets.items():
        n = len(ds_list)
        fig, axes = plt.subplots(1, n, figsize=(8, 2.0), sharey=True)
        if n == 1:
            axes = [axes]
        for i, (dataset, ax) in enumerate(zip(ds_list, axes)):
            scores_mat = load_scores(dataset, method)
            plot_line_pseudobulking_effect(scores_mat, ax,
                                           show_ylabel=(i == 0),
                                           show_xlabel=(i == n // 2))
            ax.set_title(dataset_labels[dataset], fontsize=9)

        # Build a single figure-level legend from palette_metrics
        metric_display = [surrogate_names.get(m, m) for m in METRICS if surrogate_names.get(m, m) in palette_metrics]
        legend_handles = [
            Line2D([0], [0], color=palette_metrics[m], marker='o', markersize=4, linewidth=1.5, label=m)
            for m in metric_display if m in palette_metrics
        ]
        fig.legend(handles=legend_handles, title='Metric', loc='center left',
                   bbox_to_anchor=(1.02, 0.5), bbox_transform=axes[-1].transAxes,
                   frameon=False, fontsize=7, title_fontsize=8)

        method_label = surrogate_names.get(method, method)
        fig.suptitle(method_label, fontsize=10, y=1.08)
        fig.subplots_adjust(wspace=0.15)
        file_name = f"{figs_dir}/granularity_lineplot_{method}.png"
        fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
        print(f"Saved: {file_name}")
        plt.close(fig)

    # --- Heatmap 1: mean pseudobulked / original, metrics x datasets ---
    CAP = 2.0
    pivot, pivot_nan_mask = get_granularity_heatmap_pivot(cap=CAP)
    pivot_display = pivot  # already clipped inside function
    cmap = mcolors.LinearSegmentedColormap.from_list('rg', ['#d73027', '#ffffbf', '#1a9850'])
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(pivot_display, ax=ax, cmap=cmap, vmin=0, vmax=CAP,
                mask=pivot_nan_mask,
                annot=pivot_display.applymap(lambda v: '' if pd.isna(v) else f'{v:.2f}'),
                fmt='', linewidths=0.4, linecolor='white',
                annot_kws={'size': 7},
                cbar_kws={'label': 'Mean relative performance\n(pseudobulk / single-cell)', 'shrink': 0.6, 'aspect': 20,
                          'ticks': [0, 1, 2]})
    ax.set_xlabel('Dataset')
    ax.set_ylabel('')
    plt.xticks(rotation=35, ha='right', fontsize=8)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    heatmap_file = f"{figs_dir}/granularity_heatmap.png"
    fig.savefig(heatmap_file, dpi=300, transparent=True, bbox_inches='tight')
    print(f"Saved: {heatmap_file}")

    # --- Heatmap 2: mean pseudobulked / original, metrics x methods ---
    method_data = {}
    for dataset in DATASETS:
        df = pd.read_csv(f'{RESULTS_DIR}/experiment/granular_pseudobulk/metrics_{dataset}.csv')
        if 'dataset' in df.columns:
            df = df.drop(columns=['dataset'])
        if 'method' not in df.columns:
            continue
        metric_cols = [c for c in METRICS if c in df.columns]
        for method, grp in df.groupby('method'):
            sc_row = grp[grp['granularity'] == -1][metric_cols].iloc[0]
            pb_rows = grp[grp['granularity'] != -1][metric_cols]
            ratio = pb_rows.mean() / sc_row
            if method not in method_data:
                method_data[method] = []
            method_data[method].append(ratio)

    method_pivot = {}
    for method, ratios in method_data.items():
        avg = pd.concat(ratios, axis=1).mean(axis=1)
        avg.index = [surrogate_names.get(c, c) for c in avg.index]
        method_pivot[method] = avg

    method_pivot_df = pd.DataFrame(method_pivot)  # metrics x methods
    method_pivot_df = method_pivot_df.reindex(
        columns=[m for m in ['pearson_corr', 'grnboost'] if m in method_pivot_df.columns])
    method_pivot_df.columns = [surrogate_names.get(c, c) for c in method_pivot_df.columns]

    fig, ax = plt.subplots(figsize=(4, 4))
    method_nan_mask = method_pivot_df.isna()
    method_display = method_pivot_df.clip(upper=CAP)
    sns.heatmap(method_display, ax=ax, cmap=cmap, vmin=0, vmax=CAP,
                mask=method_nan_mask,
                annot=method_pivot_df.clip(upper=CAP).applymap(lambda v: '' if pd.isna(v) else f'{v:.2f}'),
                fmt='', linewidths=0.4, linecolor='white',
                annot_kws={'size': 7},
                cbar_kws={'label': 'Mean relative performance\n(pseudobulk / single-cell)', 'shrink': 0.6, 'aspect': 20,
                          'ticks': [0, 1, 2]})
    ax.set_xlabel('GRN method')
    ax.set_ylabel('')
    plt.xticks(rotation=35, ha='right', fontsize=8)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    method_heatmap_file = f"{figs_dir}/granularity_heatmap_methods.png"
    fig.savefig(method_heatmap_file, dpi=300, transparent=True, bbox_inches='tight')
    print(f"Saved: {method_heatmap_file}")
