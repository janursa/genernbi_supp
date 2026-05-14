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
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scanpy as sc 
import itertools
import warnings
import os
import warnings
from scipy import stats
from genernbi_supp.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = F"{env['RESULTS_DIR']}/figs"

sys.path.append(env['genernbi_supp_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, palette_metrics, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores

def format_permute_grn_results(noise_type='net', metrics=None, dataset=None):
    if metrics is None:
        raise ValueError("You must provide a list of metrics.")
    degrees = [0, 10, 20, 50, 100]
    results = {}
    for degree in degrees:
        rr_file = f"{RESULTS_DIR}/experiment/permute_grn/{dataset}-{noise_type}-{degree}-scores.csv"
        if not os.path.exists(rr_file):
            # print(f"File not found: {rr_file}")
            continue
        # Load data
        df = pd.read_csv(
            rr_file,
            index_col=0
        )

        for metric in metrics:
            if metric not in df.columns:
                # print(f"Metric '{metric}' not found in {dataset}-{noise_type}-{degree}-scores.csv")
                continue
            df_metric = df.loc[:, [metric]].copy()
            df_metric['degree'] = degree
            if metric not in results:
                results[metric] = []
            results[metric].append(df_metric)
    metric_matrices = {}
    for metric in results.keys():
        df_metric_all = pd.concat(results[metric], axis=0)
        df_metric_all.index.name = 'model'
        df_metric_all = df_metric_all.reset_index().pivot(
            index="degree", columns="model", values=metric
        )
        metric_matrices[metric] = df_metric_all
    # print(metric_matrices)
    return metric_matrices
    

def merge_df(reg_mat_net=None, reg_mat_sign=None, reg_mat_weight=None, reg_mat_direction=None):
    dfs = []
    type_map = {
        'TF-gene link': reg_mat_net,
        'TF-gene sign': reg_mat_sign,
        'TF-gene weight': reg_mat_weight,
        'TF-gene direction': reg_mat_direction
    }

    # Collect available DataFrames with proper labels
    for label, df in type_map.items():
        if df is not None:
            df = df.copy()
            df['Permute type'] = label
            dfs.append(df)

    if not dfs:
        return None  # no data available

    df_concat = pd.concat(dfs)

    # Rename columns with surrogate names
    df_concat.columns = df_concat.columns.map(lambda name: surrogate_names.get(name, name))

    # Reshape
    df_concat.reset_index(inplace=True)
    df_concat.rename(columns={'degree': 'Degree'}, inplace=True)
    df_concat = df_concat.melt(
        id_vars=['Permute type', 'Degree'],
        var_name='model',
        value_name='r2score'
    )

    # Filter unwanted rows
    df_concat = df_concat[~((df_concat['Permute type'] == 'TF-gene sign') & (df_concat['Degree'] == 100))]

    return df_concat

def plot_line(df_all, ax, col='TF-gene connections', ylabel="R² score decline (%)"):
    def normalize(df):
        df = df.set_index('Degree')
        df['share'] = 0
        baseline = df.loc[0, 'r2score']
        for i, degree in enumerate(df.index):
            if degree == 0:
                share = 0
            else:
                share = 100*np.abs(df.loc[degree, 'r2score'] - baseline)/baseline
                df.loc[degree, 'share'] = share - previous_degree
                
            previous_degree = share
        return df
    df = df_all[df_all['Permute type'] == col].drop(columns='Permute type')
    if False:
        df = df.groupby(['Permute type', 'model']).apply(normalize).reset_index(level='Degree').drop(columns='r2score')
        df['Degree'] = df['Degree'].astype('category')
        df_pivot = df.pivot(index='model', columns='Degree', values='share')
    else:
        df_pivot = df.pivot(index='model', columns='Degree', values='r2score')
    for model in df_pivot.index:
        ax.plot(df_pivot.columns, df_pivot.loc[model], label=model, marker='o', color=palette_methods[model])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_title(col, pad=12)
    ax.margins(x=.1)
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
def main(metrics, dataset):
    noise_type = 'net'
    matrices_net = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'sign'
    matrices_sign = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'weight'
    matrices_weight = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'direction'
    matrices_direction = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    merged_dfs = {}
    for metric in metrics:
        reg_net = matrices_net.get(metric)
        reg_sign = matrices_sign.get(metric)
        reg_weight = matrices_weight.get(metric)
        reg_dir = matrices_direction.get(metric)

        # Check if all are None or empty
        if not any([
            df is not None and not df.empty
            for df in [reg_net, reg_sign, reg_weight, reg_dir]
        ]):
            # print(f"Skipping metric '{metric}' due to missing all data variants.")
            continue

        merged_df = merge_df(reg_net, reg_sign, reg_weight, reg_dir)
        if merged_df is not None:
            merged_dfs[metric] = merged_df
    for metric in metrics:
        if metric not in merged_dfs:
            # print(f"Skipping metric '{metric}' due to missing data.")
            continue
        # print(metric)
        ylabel = "Performance"
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.5), sharey=False)
        df_metric = merged_dfs[metric]
        # df_metric_short = df_metric[df_metric['model'].isin(['GRNBoost2', 'Scenic+', 'Pearson Corr.', 'Portia', 'PPCOR'])]
        df_metric_short = df_metric.copy()
        
        ax = axes[0]
        plot_line(df_metric_short, ax, col='TF-gene link', ylabel=ylabel)
        ax.set_xlabel('Permutation intensity (%)')
        ax.margins(y=.2)
        ax.set_xticks([0, 20, 50, 100])
        legend = ax.legend(loc=(1.1, .2), frameon=False)
        ax = axes[1]
        plot_line(df_metric_short, ax, col='TF-gene sign', ylabel=ylabel)
        ax.set_ylabel('')
        ax.margins(y=.2)
        ax.set_xticks([0, 20, 50])
        ax = axes[2]
        plot_line(df_metric_short, ax, col='TF-gene direction', ylabel=ylabel)
        ax.set_ylabel('')
        ax.margins(y=.2)
        ax.set_xticks([0, 20, 50, 100])
        ax = axes[3] 
        plot_line(df_metric_short, ax, col='TF-gene weight', ylabel=ylabel)
        ax.set_ylabel('')
        ax.margins(y=.2)
        ax.set_xticks([0, 20, 50, 100])
        # legend = ax.legend(loc=(1.1, .2), frameon=False)
        plt.suptitle(f"Metric: {surrogate_names.get(metric, metric)}", y=1.05, weight='bold')
        plt.tight_layout()
    # fig.savefig(f"{results_folder}/figs/robustnes_analysis.png", dpi=300, transparent=True, bbox_inches='tight')


def plot_metrics_as_axes(metrics, dataset, save_tag='_all', use_raw_scores=False):
    """
    Plot permutation analysis with one figure per permutation type, and one axis per metric.
    Instead of the main function which creates one figure per metric with 4 axes (one per permutation type),
    this creates 4 figures (one per permutation type) with multiple axes (one per metric).
    
    Args:
        metrics: list of metrics to plot
        dataset: dataset name
        save_tag: tag to append to saved figure names
        use_raw_scores: if True, plot raw scores instead of normalized 0-100 scale
    """
    # Load all permutation results
    noise_type = 'net'
    matrices_net = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'sign'
    matrices_sign = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'weight'
    matrices_weight = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    noise_type = 'direction'
    matrices_direction = format_permute_grn_results(noise_type=noise_type, metrics=metrics, dataset=dataset)
    
    # Merge data for all metrics
    merged_dfs = {}
    for metric in metrics:
        reg_net = matrices_net.get(metric)
        reg_sign = matrices_sign.get(metric)
        reg_weight = matrices_weight.get(metric)
        reg_dir = matrices_direction.get(metric)

        # Check if all are None or empty
        if not any([
            df is not None and not df.empty
            for df in [reg_net, reg_sign, reg_weight, reg_dir]
        ]):
            continue

        merged_df = merge_df(reg_net, reg_sign, reg_weight, reg_dir)
        if merged_df is not None:
            merged_dfs[metric] = merged_df
    
    # Count available metrics
    available_metrics = list(merged_dfs.keys())
    n_metrics = len(available_metrics)
    
    if n_metrics == 0:
        print("No metrics available to plot")
        return
    
    if use_raw_scores:
        ylabel = "Raw score"
    else:
        ylabel = "Performance (%)"
    # permute_types = ['TF-gene link', 'TF-gene sign', 'TF-gene direction', 'TF-gene weight']
    permute_types = ['TF-gene link', 'TF-gene sign', 'TF-gene weight', 'TF-gene direction']
    
    # Helper function to normalize data with min-max scaling to 0-100
    def normalize_minmax(df_all, col):
        """Apply min-max normalization to scale values to 0-100"""
        df = df_all[df_all['Permute type'] == col].drop(columns='Permute type').copy()
        if df.empty:
            return df
        
        df_pivot = df.pivot(index='model', columns='Degree', values='r2score')
        
        # Find global min and max across all models and degrees
        global_min = df_pivot.min().min()
        global_max = df_pivot.max().max()
        
        # Avoid division by zero
        if global_max == global_min:
            df_pivot_norm = df_pivot * 0 + 50  # Set to middle if all values are the same
        else:
            # Min-max normalization: (x - min) / (max - min) * 100
            df_pivot_norm = ((df_pivot - global_min) / (global_max - global_min)) * 100
        
        return df_pivot_norm
    
    # Metrics that encode sign information
    SIGN_METRICS = {'vc', 'sem'}

    # Create one figure per permutation type
    for permute_type in permute_types:
        # For sign permutation, only show metrics that encode sign
        if permute_type == 'TF-gene sign':
            plot_metrics = [m for m in available_metrics if m in SIGN_METRICS]
        else:
            plot_metrics = available_metrics

        if not plot_metrics:
            continue

        n_cols = len(plot_metrics)
        n_rows = 1

        # Make figure bigger for raw scores
        if use_raw_scores:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.5*n_rows), sharey=False)
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.7*n_cols, 2.4*n_rows), sharey=False)
        if n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for idx, metric in enumerate(plot_metrics):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            df_metric = merged_dfs[metric]
            
            
            
            # Normalize the data for this permutation type (or use raw scores)
            _SELECTED_METHODS = ['PPCOR', 'GRNBoost2', 'Pearson Corr.', 'Portia', 'Scenic+', 'scPRINT', 'Scenic']
            if use_raw_scores:
                # Use raw scores instead of normalization, but same method subset
                df = df_metric[df_metric['Permute type'] == permute_type].drop(columns='Permute type').copy()
                if df.empty:
                    continue
                df = df[df['model'].isin(_SELECTED_METHODS)]
                df_pivot = df.pivot(index='model', columns='Degree', values='r2score')
            else:
                df_metric_short = df_metric[df_metric['model'].isin(_SELECTED_METHODS)]
                # df_metric_short = df_metric.copy()
                df_pivot = normalize_minmax(df_metric_short, permute_type)

            
            if df_pivot.empty:
                continue
            
            # Plot data (normalized or raw)
            for model in df_pivot.index:
                ax.plot(df_pivot.columns, df_pivot.loc[model], 
                       label=model, marker='o', color=palette_methods.get(model, 'gray'))
            
            ax.set_ylabel(ylabel if idx == 0 else '')
            ax.margins(x=.1)
            for side in ['right', 'top']:
                ax.spines[side].set_visible(False)
            
            # Only show x-axis label on first axis
            if idx == 0:
                ax.set_xlabel('Permutation intensity (%)')
            else:
                ax.set_xlabel('')
                if not use_raw_scores:
                    ax.set_yticks([])
            
            # Wrap long titles: if > 10 characters, replace " (" with " \n("
            title = surrogate_names.get(metric, metric)
            if len(title) > 10:
                title = title.replace(' (', ' \n(')
            ax.set_title(title, pad=12)
            ax.margins(x=.1, y=.2)
            
            # Add legend to the last axis only
            if idx == len(plot_metrics) - 1:
                ax.legend(loc=[1.07, .2], frameon=False, fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(plot_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        # plt.suptitle(f"{permute_type}", y=1.02, weight='bold', fontsize=14)
        plt.tight_layout()
        
        # Add _raw tag to filename if using raw scores
        if use_raw_scores:
            fig_name = f"{figs_dir}/permutation_{permute_type.replace(' ', '_').replace('-', '_')}_{save_tag}_{dataset}_raw.png"
        else:
            fig_name = f"{figs_dir}/permutation_{permute_type.replace(' ', '_').replace('-', '_')}_{save_tag}_{dataset}.png"
        print(f"Saving figure to: {fig_name}")
        fig.savefig(fig_name, 
                   dpi=200, transparent=True, bbox_inches='tight')


# ── Run for completed datasets ────────────────────────────────────────────────
from task_grn_inference.src.utils.config import DATASETS_METRICS, METRICS as ALL_METRIC_COLS

# Map module-level metric names → actual CSV column names
MODULE_TO_COLS = {
    'regression':            ['r_precision', 'r_recall'],
    'ws_distance':           ['ws_precision', 'ws_recall'],
    'tf_recovery':           ['t_rec_precision', 't_rec_recall'],
    'tf_binding':            ['tfb_f1'],
    'sem':                   ['sem'],
    'vc':                    ['vc'],
    'gs_recovery':           ['gs_f1'],
    'replicate_consistency': ['replicate_consistency'],
}

def dataset_metric_cols(dataset):
    """Return CSV column names relevant for this dataset."""
    cols = []
    for m in DATASETS_METRICS.get(dataset, []):
        cols.extend(MODULE_TO_COLS.get(m, []))
    return [c for c in cols if c in ALL_METRIC_COLS]

# datasets with at least net + sign + weight complete
DATASETS_TO_RUN = ['op', 'replogle', 'parsebioscience', 'norman']

for ds in DATASETS_TO_RUN:
    metrics = dataset_metric_cols(ds)
    print(f"\n=== {ds}: {metrics} ===")
    plot_metrics_as_axes(metrics=metrics, dataset=ds, save_tag='all', use_raw_scores=True)


# ── Which metric modules are relevant per permutation type ───────────────────
PERMUTATION_METRIC_MAP = {
    'net':    ['regression', 'ws_distance', 'sem', 'tf_recovery', 'tf_binding', 'vc', 'gs_recovery'],
    'weight': ['regression', 'ws_distance', 'tf_recovery', 'tf_binding', 'gs_recovery'],
}

ALL_DATASETS_PERM = [
    'op', 'replogle', 'parsebioscience', 'norman',
    'nakatake', 'xaira_HCT116', 'xaira_HEK293T',
]

PERM_DIR = f"{RESULTS_DIR}/experiment/permute_grn"


def _load_ratio_df(ptype):
    """For each (dataset, method, metric), return whether score@100 < score@0 (degraded=True)."""
    rel_cols = []
    for m in PERMUTATION_METRIC_MAP.get(ptype, []):
        rel_cols.extend(MODULE_TO_COLS.get(m, []))

    rows = []
    for ds in ALL_DATASETS_PERM:
        f0   = f"{PERM_DIR}/{ds}-{ptype}-0-scores.csv"
        f100 = f"{PERM_DIR}/{ds}-{ptype}-100-scores.csv"
        if not os.path.exists(f0) or not os.path.exists(f100):
            continue
        df0   = pd.read_csv(f0,   index_col=0)
        df100 = pd.read_csv(f100, index_col=0)
        ds_applicable = dataset_metric_cols(ds)
        avail = [c for c in rel_cols if c in df0.columns and c in df100.columns and c in ds_applicable]
        for method in df0.index.intersection(df100.index):
            for col in avail:
                v0, v100 = df0.loc[method, col], df100.loc[method, col]
                if pd.notna(v0) and pd.notna(v100):
                    rows.append({'dataset': ds, 'method': method, 'metric': col,
                                 'degraded': float(v100 < v0)})
    return pd.DataFrame(rows)


def _sensitivity_bar(ptype, out_path):
    """Horizontal bar: fraction of (dataset x method) pairs where 100% permutation
    degrades the metric, shown per metric. Annotated with n_degraded/n_total.
    """
    df = _load_ratio_df(ptype)
    if df.empty:
        print(f"No data for ptype={ptype}")
        return

    df['metric_name'] = df['metric'].map(lambda x: surrogate_names.get(x, x))
    grp   = df.groupby('metric_name')['degraded']
    stats = grp.agg(frac='mean', n='count')

    # order by sensitivity descending (most degraded first)
    stats = stats.sort_values('frac', ascending=False)

    colors = [palette_metrics.get(m, '#aab4be') for m in stats.index]

    n_items = len(stats)
    fig, ax = plt.subplots(figsize=(3.5, 0.22 * n_items + 0.5))
    bars = ax.barh(stats.index, stats['frac'].values, color=colors,
                   edgecolor='white', height=0.65)

    # annotate with n_degraded / n_total
    for bar, (name, row) in zip(bars, stats.iterrows()):
        n_deg = int(round(row['frac'] * row['n']))
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{n_deg}/{int(row['n'])}", va='center', ha='left', fontsize=8)

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlim(0, 1.22)
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('Metric')
    ax.invert_yaxis()
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def _subset_degree_plot(dataset, methods_list, ptype, out_path):
    """Line plot: raw scores vs permutation degree for a subset of methods.

    One subplot per relevant metric. Used for OP and Replogle detailed views.
    """
    rel_cols = []
    for m in PERMUTATION_METRIC_MAP.get(ptype, []):
        rel_cols.extend(MODULE_TO_COLS.get(m, []))
    ds_cols  = dataset_metric_cols(dataset)
    cols     = [c for c in rel_cols if c in ds_cols]
    if not cols:
        print(f"No relevant metrics for {dataset}/{ptype}")
        return

    degrees = [0, 10, 20, 50, 100]
    # data[col][method][degree] = value
    data = {col: {m: {} for m in methods_list} for col in cols}
    for deg in degrees:
        fpath = f"{PERM_DIR}/{dataset}-{ptype}-{deg}-scores.csv"
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, index_col=0)
        for col in cols:
            if col not in df.columns:
                continue
            for m in methods_list:
                if m in df.index:
                    data[col][m][deg] = df.loc[m, col]

    # convert to DataFrames, drop empty cols
    col_dfs = {}
    for col in cols:
        method_data = {m: data[col][m] for m in methods_list if data[col][m]}
        if method_data:
            col_dfs[col] = pd.DataFrame(method_data, index=degrees)

    if not col_dfs:
        print(f"No data for {dataset}/{ptype}")
        return

    n = len(col_dfs)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n + 0.5, 2), sharey=False)
    if n == 1:
        axes = [axes]

    for idx, (col, df_col) in enumerate(col_dfs.items()):
        ax = axes[idx]
        for method in df_col.columns:
            m_name = surrogate_names.get(method, method)
            color  = palette_methods.get(m_name, '#aab4be')
            ax.plot(df_col.index, df_col[method].values,
                    marker='o', label=m_name, color=color, linewidth=1.5)

        ax.margins(x=0.1, y=0.15)
        ax.set_title(surrogate_names.get(col, col), fontsize=9)
        ax.set_xlabel('Permutation (%)')
        ax.set_ylabel('Score' if idx == 0 else '')
        ax.set_xticks([0, 20, 50, 100])
        for side in ['right', 'top']:
            ax.spines[side].set_visible(False)
        if idx == n - 1:
            ax.legend(loc=(1.05, 0.1), frameon=False, fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


# ── Sensitivity summary bars (all datasets × methods aggregated) ─────────────
_sensitivity_bar('net',    f"{figs_dir}/permutation_sensitivity_net.png")
_sensitivity_bar('weight', f"{figs_dir}/permutation_sensitivity_weight.png")

# ── Subset degree plots for OP and Replogle ───────────────────────────────────
_OP_METHODS       = ['scenicplus', 'celloracle', 'grnboost', 'pearson_corr', 'ppcor']
_REPLOGLE_METHODS = ['scenic', 'grnboost', 'pearson_corr', 'ppcor', 'scprint']

for _ptype in ['net', 'weight']:
    _subset_degree_plot('op', _OP_METHODS, _ptype,
                        f"{figs_dir}/permutation_subset_op_{_ptype}.png")
    _subset_degree_plot('replogle', _REPLOGLE_METHODS, _ptype,
                        f"{figs_dir}/permutation_subset_replogle_{_ptype}.png")