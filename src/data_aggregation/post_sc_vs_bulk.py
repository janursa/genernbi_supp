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
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from geneRNBI.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = F"{env['RESULTS_DIR']}/figs"

sys.path.append(env['geneRNBI_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
from src.utils.config import METRICS

df_all = []
for dataset in ['replogle', 'xaira_HCT116', 'xaira_HEK293T']:
    df_d = pd.read_csv(f"{RESULTS_DIR}/experiment/bulk_vs_sc/metrics_{dataset}.csv")
    df_all.append(df_d)
df_all = pd.concat(df_all, ignore_index=True)
metric_cols = [m for m in METRICS if m in df_all.columns]
group_cols = ['dataset', 'data_type']
# average across methods if method column present
df_all = df_all[metric_cols + group_cols + (['method'] if 'method' in df_all.columns else [])]
if 'method' in df_all.columns:
    df_all = df_all.groupby(group_cols)[metric_cols].mean().reset_index()

df = df_all.copy()

sc = df[df['data_type'] == 'sc'].set_index('dataset')
bulk = df[df['data_type'] == 'bulk'].set_index('dataset')
diff = sc.drop('data_type', axis=1) / \
       bulk.drop('data_type', axis=1)
diff = diff.reset_index().melt(id_vars='dataset', var_name='metric', value_name='sc_minus_bulk')
diff['dataset'] = diff['dataset'].map(lambda name: surrogate_names.get(name, name))
diff['metric'] = diff['metric'].map(lambda name: surrogate_names.get(name, name))

fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
palette_datasets = {surrogate_names.get(ds, ds): color for ds, color in palette_datasets.items()}
sns.barplot(data=diff, x='metric', y='sc_minus_bulk', hue='dataset', ci=None, ax=ax, palette=palette_datasets)
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(1, color='red', linestyle='--')
plt.ylabel('Single-cell / Pseudobulk')
plt.xlabel('Metric')
plt.margins(y=0.1, x=0.05)
# plt.title('Difference between Single-cell (SC) and Bulk per Metric')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Dataset', loc=[1.05, .3], frameon=False)
ax.spines[['top', 'right']].set_visible(False)
file_name = f"{figs_dir}/evaluation_scores_sc_vs_bulk_barplot.png"
print(f"Saving figure to {file_name}")
plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')