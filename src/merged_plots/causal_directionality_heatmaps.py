"""
Merged heatmap figure for causal directionality experiment:
  Panel 1: methods × datasets
  Panel 2: methods × metrics

Both share the y-axis (methods) and a single colorbar.

Output:
  results/figs/merged_causal_directionality_heatmap.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

from genernbi_supp.src.helper import load_env
from genernbi_supp.src.stability_analysis.causal_directionality.post_causal_directionality import (
    get_causal_directionality_pivots,
)

env = load_env()
figs_dir = f"{env['RESULTS_DIR']}/figs"
os.makedirs(figs_dir, exist_ok=True)

pivot_ds, pivot_met, CAP = get_causal_directionality_pivots()
cmap = mcolors.LinearSegmentedColormap.from_list('rg', ['#d73027', '#ffffbf', '#1a9850'])

pivots    = [pivot_ds,  pivot_met]
titles    = ['Datasets', 'Metrics']
xlabels   = ['Dataset', 'Metric']
nan_masks = [p.isna() for p in pivots]

panel_widths = [max(1, len(p.columns)) for p in pivots]
fig, axes = plt.subplots(1, 2, figsize=(6, 3.5),
                         gridspec_kw={'width_ratios': panel_widths})

for i, (ax, pivot, nan_mask, title, xlabel) in enumerate(
        zip(axes, pivots, nan_masks, titles, xlabels)):
    sns.heatmap(
        pivot.clip(upper=CAP),
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=CAP,
        mask=nan_mask,
        annot=pivot.clip(upper=CAP).applymap(lambda v: '' if pd.isna(v) else f'{v:.2f}'),
        fmt='',
        linewidths=0.4,
        linecolor='white',
        annot_kws={'size': 6},
        cbar=False,
        yticklabels=(i == 0),
    )
    ax.set_title('', fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel('', fontsize=8)
    ax.set_ylabel('Method' if i == 0 else '', fontsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=7)

plt.subplots_adjust(wspace=0.08, right=0.88)

cbar_ax = fig.add_axes([0.90, 0.20, 0.025, 0.55])
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=CAP))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0, 1, 2])
cbar.set_label('Sensitivity\n(reversed / original)', fontsize=7)
cbar.ax.tick_params(labelsize=7)

out = f"{figs_dir}/merged_causal_directionality_heatmap.png"
fig.savefig(out, dpi=300, transparent=True, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
