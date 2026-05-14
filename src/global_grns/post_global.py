import pandas as pd
import numpy as np
import warnings
import os
import seaborn as sns
import matplotlib.pyplot as plt
from genernbi_supp.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{env['RESULTS_DIR']}/figs"

from genernbi_supp.src.helper import plot_heatmap, surrogate_names, palette_metrics, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, CONTROLS3, retrieve_grn_path, \
                        plot_raw_scores

from task_grn_inference import METRICS

INDIVIDUAL_DATASETS = ['op', 'parsebioscience', '300BCG', 'soundlife', 'soundlife_vaccine']
EXCLUDE_DATASETS = ['ibd_cd', 'ibd_uc']


def extract_tissue_name(x):
    if ':' not in x:
        return x
    method_part = x.split(':')[0]
    tissue_part = x.split(':')[1].strip()
    if tissue_part.lower().startswith('whole blood'):
        tissue_name = 'Blood'
    elif tissue_part.lower().startswith('bone'):
        tissue_name = 'Bone marrow'
    else:
        tissue_name = tissue_part.split()[0]
    return f"{method_part}: {tissue_name}"


def load_and_process_metrics(dataset):
    path = f'{RESULTS_DIR}/experiment/global_grns/metrics_{dataset}.csv'
    if not os.path.exists(path):
        return None
    metrics = pd.read_csv(path).fillna(0)
    metrics['model'] = metrics['model'].apply(lambda x: x.replace('.csv', ''))
    metrics['model'] = metrics['model'].apply(lambda x: ':'.join(x.split(':')[:2]) if ':' in x else x)
    metrics['model'] = metrics['model'].apply(extract_tissue_name)
    metrics = metrics[[c for c in METRICS if c in metrics.columns] + ['model']]
    metrics = metrics[~metrics['model'].isin(['scenic', 'Scenic'])]
    metrics.set_index('model', inplace=True)
    return metrics  # original (processed) model names, no surrogate mapping applied yet


# ── Load all available datasets ───────────────────────────────────────────────
global_grns_dir = f'{RESULTS_DIR}/experiment/global_grns'
available_datasets = sorted([
    f.replace('metrics_', '').replace('.csv', '')
    for f in os.listdir(global_grns_dir)
    if f.startswith('metrics_') and f.endswith('.csv') and f != 'metrics_all.csv'
    and f.replace('metrics_', '').replace('.csv', '') not in EXCLUDE_DATASETS
])

all_metrics = {}
for ds in available_datasets:
    m = load_and_process_metrics(ds)
    if m is not None:
        all_metrics[ds] = m

# ── Compute sensitivity across all datasets ───────────────────────────────────
# Context-specific = inference methods (no ":" in model name, inferred from the dataset)
# Non-context-specific = global pre-built GRNs (Gtex, Ananse, Cellnet — ":" in model name)
# Sensitivity ratio per (dataset, metric) = mean(context-specific) / mean(non-context-specific)
records = []
for ds_name, metrics_df in all_metrics.items():
    is_global = metrics_df.index.map(lambda x: ':' in x)
    ctx     = metrics_df[~is_global]   # inference methods (context-specific)
    non_ctx = metrics_df[is_global]    # global GRNs (non-context-specific)

    if ctx.empty or non_ctx.empty:
        continue

    for metric in metrics_df.columns:
        ctx_mean     = ctx[metric].mean()
        non_ctx_mean = non_ctx[metric].mean()
        ratio = ctx_mean / non_ctx_mean if non_ctx_mean != 0 else np.nan
        records.append({'dataset': ds_name, 'metric': metric, 'ratio': ratio})

sens_df = pd.DataFrame(records)
summary = (
    sens_df.groupby('metric')
    .agg(
        mean_ratio=('ratio', 'mean'),
        count_above_1=('ratio', lambda x: (x > 1).sum()),
        n_datasets=('ratio', 'count'),
    )
    .reset_index()
)
# Map metric names to display names and sort by mean_ratio
summary['metric_name'] = summary['metric'].map(lambda x: surrogate_names.get(x, x))
summary = summary.sort_values('mean_ratio', ascending=False)

# ── Sensitivity summary bar plot ──────────────────────────────────────────────
_pal = palette_metrics if palette_metrics else {}
bar_colors = [_pal.get(m, '#aab4be') for m in summary['metric_name']]

n_metrics = len(summary)
fig, ax = plt.subplots(figsize=(3.5, 0.2 * n_metrics + 1.2))
bars = ax.barh(summary['metric_name'], summary['mean_ratio'],
               color=bar_colors, edgecolor='white', height=0.65)

for bar, (_, row) in zip(bars, summary.iterrows()):
    ax.text(bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['count_above_1'])}/{int(row['n_datasets'])}",
            va='center', ha='left', fontsize=9)

ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlim(0, summary['mean_ratio'].max() * 1.3)
ax.set_xlabel('Sensitivity', labelpad=10)
ax.set_ylabel('Metric')
ax.invert_yaxis()
for side in ['right', 'top']:
    ax.spines[side].set_visible(False)
plt.tight_layout()

file_name = f"{figs_dir}/global_models_sensitivity.png"
print(file_name)
fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
plt.close(fig)

# ── Heatmap: metrics × datasets, values = mean(ctx) / mean(non-ctx) ──────────
heatmap_df = sens_df.pivot(index='metric', columns='dataset', values='ratio')
heatmap_df.index   = heatmap_df.index.map(lambda x: surrogate_names.get(x, x))
heatmap_df.columns = heatmap_df.columns.map(lambda x: surrogate_names.get(x, x))
heatmap_df = heatmap_df.sort_index()

# Order datasets alphabetically
heatmap_df = heatmap_df[sorted(heatmap_df.columns)]

n_metrics  = len(heatmap_df)
n_datasets = len(heatmap_df.columns)
from matplotlib.colors import LinearSegmentedColormap
grim_cmap = LinearSegmentedColormap.from_list('ctx_specificity', ['#8b0000', '#d9d9d9', '#2d7d2d'], N=256)
fig, ax = plt.subplots(figsize=(0.6 * n_datasets + 2.2, 0.45 * n_metrics + 0.7))
sns.heatmap(
    heatmap_df,
    ax=ax,
    cmap=grim_cmap,
    vmin=0,
    vmax=2,
    annot=True,
    fmt='.2f',
    linewidths=0.4,
    linecolor='white',
    cbar_kws={'label': 'Context specificity', 'shrink': 0.9, 'ticks': [0, 1, 2]},
)
ax.set_xlabel('Dataset')
ax.set_ylabel('Metric')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()

file_name = f"{figs_dir}/global_models_ctx_vs_nonctx_heatmap.png"
print(file_name)
fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
plt.close(fig)

# ── Individual dataset heatmaps (op and replogle only) ────────────────────────
for dataset in INDIVIDUAL_DATASETS:
    metrics_df = all_metrics.get(dataset)
    if metrics_df is None:
        print(f"Dataset '{dataset}' not found, skipping.")
        continue

    metrics_display = metrics_df.copy()
    metrics_display.index   = metrics_display.index.map(lambda x: surrogate_names.get(x, x))
    metrics_display.columns = metrics_display.columns.map(lambda x: surrogate_names.get(x, x))

    fig, ax = plt.subplots(1, 1, figsize=(.5 * len(metrics_display.columns), 6), sharey=False)
    plot_heatmap(metrics_display, name='', ax=ax, cmap="viridis")
    ax.set_ylabel('')
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')
    file_name = f"{figs_dir}/global_models_{dataset}.png"
    print(file_name)
    fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)