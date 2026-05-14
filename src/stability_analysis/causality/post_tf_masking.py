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
figs_dir = f"{env['RESULTS_DIR']}/figs/consensus_regulators"
os.makedirs(figs_dir, exist_ok=True)


sys.path.append(env['genernbi_supp_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores

TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
from src.utils.config import DATASETS_METRICS, DATASETS_CELLTYPES, DATASETS, METRICS
df = pd.read_csv(f"{env['RESULTS_DIR']}/experiment/causality/scores.csv", index_col=0)
df = df.drop('r2_raw', axis=1)
df

# Filter for dataset_mask == 'pert'
df_pert = df[df['dataset_mask'] == 'pert'].copy()

# Prepare data for plotting
metrics_cols = ['r_precision', 'r_recall']
df_plot = df_pert[['dataset', 'method'] + metrics_cols].copy()

# Calculate relative performance (TF masked / TF not masked)
datasets = df_plot['dataset'].unique()
relative_data = []

for dataset in datasets:
    dataset_data = df_plot[df_plot['dataset'] == dataset]
    
    tf_masked = dataset_data[dataset_data['method'] == 'pearson_tf_applied']
    not_masked = dataset_data[dataset_data['method'] == 'pearson']
    
    if len(tf_masked) > 0 and len(not_masked) > 0:
        for metric in metrics_cols:
            tf_val = tf_masked[metric].values[0]
            not_tf_val = not_masked[metric].values[0]
            
            # Calculate ratio (handle division by zero)
            if not_tf_val != 0 and not np.isnan(tf_val) and not np.isnan(not_tf_val):
                ratio = tf_val / not_tf_val
                relative_data.append({
                    'dataset': dataset,
                    'metric': metric,
                    'relative_performance': ratio
                })

# Create DataFrame
df_relative = pd.DataFrame(relative_data)

# Apply surrogate names
df_relative['dataset'] = df_relative['dataset'].map(lambda x: surrogate_names.get(x, x))
df_relative['metric'] = df_relative['metric'].map(lambda x: surrogate_names.get(x, x))

# Order metrics according to METRICS
ordered_metric_names = [surrogate_names.get(m, m) for m in METRICS if surrogate_names.get(m, m) in df_relative['metric'].unique()]
df_relative['metric'] = pd.Categorical(df_relative['metric'], categories=ordered_metric_names, ordered=True)
df_relative = df_relative.sort_values('metric')

datasets_plot = df_relative['dataset'].unique()
metrics_plot = ordered_metric_names

fig, ax = plt.subplots(1, 1, figsize=(2, 2.2))
x = np.arange(len(metrics_plot))  # Metric positions
width = 0.15  # Width of each bar
dataset_colors = [palette_datasets.get(ds, colors_blind[i % len(colors_blind)]) 
                  for i, ds in enumerate(datasets)]

for i, dataset in enumerate(datasets_plot):
    dataset_data = df_relative[df_relative['dataset'] == dataset]
    ratios = []
    for metric in metrics_plot:
        metric_dataset = dataset_data[dataset_data['metric'] == metric]
        if len(metric_dataset) > 0:
            ratios.append(metric_dataset['relative_performance'].values[0])
        else:
            ratios.append(np.nan)
    offset = width * i
    original_dataset = [d for d in datasets if surrogate_names.get(d, d) == dataset][0]
    color = palette_datasets.get(original_dataset, colors_blind[i % len(colors_blind)])
    ax.bar(x + offset, ratios, width, label=dataset, 
           color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No difference')
ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Relative Performance\n(TF-masked / not masked)', fontsize=12)
ax.set_xticks(x + width * (len(datasets_plot) - 1) / 2)
ax.set_xticklabels(metrics_plot, rotation=45, ha='right')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=10, title='Dataset')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.margins(x=0.15, y=0.15)
file_name = f"{figs_dir}/causality_relative_performance.png"
print(f"Saving figure to: {file_name}")
plt.savefig(file_name, dpi=300, bbox_inches='tight')