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
warnings.filterwarnings("ignore")

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{env['RESULTS_DIR']}/figs/consensus_regulators"
os.makedirs(figs_dir, exist_ok=True)


sys.path.insert(0, env['genernbi_supp_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores

TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
from task_grn_inference.src.utils.config import DATASETS_METRICS, DATASETS_CELLTYPES, DATASETS, METRICS, METHODS

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='op', help='Dataset to analyze')
args = parser.parse_args()
dataset = args.dataset
output_dir = f"{RESULTS_DIR}/experiment/normalization"
output_file = f"{output_dir}/metrics_{dataset}.csv"

output = pd.read_csv(output_file)

def map_method(pred_str):
    for m in METHODS:
        if m.lower() in str(pred_str).lower():
            return m
    return "unknown"

def map_normalizaton_method(pred_str):
    if 'defualt' in str(pred_str).lower():
        return 'SLA'
    else:
        return 'PR'

# Apply mapping
output["method"] = output["prediction"].apply(map_method)
output["normalization"] = output["prediction"].apply(map_normalizaton_method)
output.drop('prediction', axis=1, inplace=True)

output = output[output["method"] != "portia"]

cols = ['method', 'normalization'] + METRICS
output = output[[c for c in cols if c in output.columns]]

# Option 1: Side-by-side ranking plot (Slope/Bump chart)
def plot_ranking_comparison(output_df, metrics=None, norm_methods=['SLA', 'PR']):
    """
    Show how method rankings compare across normalization methods.
    Stable rankings indicate stable comparative performance.
    """
    metrics = [c for c in metrics if c in output_df.columns]
    n_metrics = len(metrics)
    n_norms = len(norm_methods)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(1.4*n_metrics, 2), sharey=True)
    if n_metrics == 1:
        axes = [axes]
    for idx, metric in enumerate(metrics):
        if metric not in output_df.columns:
            continue
        ax = axes[idx]
        ranking_data = []
        for norm in norm_methods:
            subset = output_df[output_df['normalization'] == norm][['method', metric]].copy()
            subset = subset.sort_values(metric, ascending=False)
            subset['rank'] = range(1, len(subset) + 1)
            subset['normalization'] = norm
            ranking_data.append(subset[['method', 'rank', 'normalization']])
        ranking_df = pd.concat(ranking_data)
        ranking_df['method'] = ranking_df['method'].map(lambda name: surrogate_names.get(name, name))
        ranking_pivot = ranking_df.pivot(index='method', columns='normalization', values='rank')
        # ranking_pivot = ranking_pivot[['SLA', 'PR']]  # Ensure consistent order
        for method in ranking_pivot.index:
            if all(pd.notna(ranking_pivot.loc[method])):
                ax.plot(
                    range(n_norms), 
                    ranking_pivot.loc[method].values,
                    marker='o', 
                    label=method,
                    color=palette_methods.get(method, 'gray'),
                    alpha=0.7,
                    linewidth=2,
                    markersize=8
                )
        ax.set_xticks(range(n_norms))
        ax.set_xticklabels([surrogate_names.get(n, n) for n in norm_methods], rotation=0)
        ax.set_ylabel('Rank (1=best)' if idx == 0 else '')
        ax.set_xlabel('')
        ax.set_title(surrogate_names.get(metric, metric).replace(' (', ' \n(').replace('Replicate consistency', 'Replica \nconsistency').replace('Gene sets recovery', 'Gene sets \nrecovery'), pad=10)
        ax.invert_yaxis()  # Lower rank (1) at top
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        valid_ranks = sorted(ranking_df['rank'].unique())
        ax.set_yticks(valid_ranks)
        ax.margins(x=0.25, y=0.2)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        if idx == n_metrics - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=9)
        if idx != 0:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)

    plt.tight_layout()
    return fig

fig1 = plot_ranking_comparison(output, metrics=METRICS)
file_name = f'{figs_dir}/normalization_ranking_comparison.png'
print(f"Saving figure to {file_name}")
plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')


fig1 = plot_ranking_comparison(output, metrics=['r_precision', 'r_recall', 'vc', 'sem'])
file_name = f'{figs_dir}/normalization_ranking_comparison_selected.png'
print(f"Saving figure to {file_name}")
plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
