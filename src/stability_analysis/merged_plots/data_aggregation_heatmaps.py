"""
Merged heatmap figure: Metacell | Imputation: KNN | Imputation: MAGIC

Imports data extraction functions from:
  - geneRNBI.src.data_aggregation.post_granular
  - geneRNBI.src.imputation.post_imputation

Output:
  results/figs/merged_metacell_imputation_heatmap.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

from geneRNBI.src.helper import load_env
from geneRNBI.src.data_aggregation.post_granular import get_granularity_heatmap_pivot
from geneRNBI.src.imputation.post_imputation import get_imputation_heatmap_pivot

env = load_env()
figs_dir = f"{env['RESULTS_DIR']}/figs"
os.makedirs(figs_dir, exist_ok=True)

CAP = 2.0
cmap = mcolors.LinearSegmentedColormap.from_list('rg', ['#d73027', '#ffffbf', '#1a9850'])

# --- Retrieve pivot DataFrames from each source ---
pivot_meta, nan_meta = get_granularity_heatmap_pivot(cap=CAP)
pivot_knn,  nan_knn  = get_imputation_heatmap_pivot('knn',   cap=CAP)
pivot_magic, nan_magic = get_imputation_heatmap_pivot('magic', cap=CAP)

# Align all three panels on the same row order (metrics)
all_metrics = pivot_meta.index.union(pivot_knn.index).union(pivot_magic.index)
pivots    = [pivot_meta.reindex(all_metrics),
             pivot_knn.reindex(all_metrics),
             pivot_magic.reindex(all_metrics)]
nan_masks = [p.isna() for p in pivots]
titles    = ['Metacell', 'Imputation: KNN', 'Imputation: MAGIC']

n_rows = len(all_metrics)
panel_widths = [max(1, len(p.columns)) for p in pivots]
total_w = 4
fig_h   = 2.5

fig, axes = plt.subplots(1, 3, figsize=(total_w, fig_h),
                         gridspec_kw={'width_ratios': panel_widths})

for i, (ax, pivot, nan_mask, title) in enumerate(zip(axes, pivots, nan_masks, titles)):
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=CAP,
        mask=nan_mask,
        annot=pivot.applymap(lambda v: '' if pd.isna(v) else f'{v:.2f}'),
        fmt='',
        linewidths=0.4,
        linecolor='white',
        annot_kws={'size': 7},
        cbar=False,
        yticklabels=(i == 0),
    )
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('Dataset' if i == 0 else '', fontsize=9)
    ax.set_ylabel('Metric' if i == 0 else '', fontsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)

plt.subplots_adjust(wspace=0.08, right=0.88)

# Shared colorbar in a dedicated axes to the right of all panels
cbar_ax = fig.add_axes([0.90, 0.20, 0.025, 0.55])
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=CAP))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0, 1, 2])
cbar.set_label('Relative performance\n(experiment / single-cell)', fontsize=7)
cbar.ax.tick_params(labelsize=7)

out = f"{figs_dir}/merged_metacell_imputation_heatmap.png"
fig.savefig(out, dpi=300, transparent=True, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
