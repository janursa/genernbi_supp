"""
Post-analysis for skeleton-filtering experiment.
Compares skeleton-filtered GRN scores vs original GRN scores.
Produces a bar plot: x=methods, bars=metrics, y=relative score (skeleton/original).
Also produces two summary heatmaps across all datasets:
  1. methods × metrics  (mean ratio across datasets)
  2. metrics × datasets (mean ratio across methods)

Usage:
    python src/exp_analysis/post_skeleton.py --dataset op
    python src/exp_analysis/post_skeleton.py --dataset op --all_datasets
"""
import argparse
import os
import sys
import warnings
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")

from geneRNBI.src.helper import load_env
env = load_env()

sys.path.insert(0, env['geneRNBI_DIR'])
from src.helper import surrogate_names, METHODS, palette_metrics, palette_methods

RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{RESULTS_DIR}/figs"
os.makedirs(figs_dir, exist_ok=True)

from task_grn_inference.src.utils.config import METRICS as ALL_METRICS

# replicate_consistency excluded: measures stability across replicates, not structure
SKIP_METRICS = {'replicate_consistency'}
METRICS_COLS  = [m for m in ALL_METRICS if m not in SKIP_METRICS]

GS_MIN_THRESHOLD = 0.05  # mask gs_f1 cells where original score < threshold


def plot_dataset(dataset: str, heatmap_records: list = None):
    import glob as _glob
    all_scores_path = f"{RESULTS_DIR}/all_scores.csv"

    # collect per-method score files (one per sbatch job); fall back to legacy single file
    per_method_files = _glob.glob(
        f"{RESULTS_DIR}/experiment/skeleton/{dataset}/{dataset}-*-skeleton-scores.csv"
    )
    if not per_method_files:
        legacy = f"{RESULTS_DIR}/experiment/skeleton/{dataset}/{dataset}-skeleton-scores.csv"
        if os.path.exists(legacy):
            per_method_files = [legacy]
        else:
            print(f"No skeleton scores found for {dataset} — skipping")
            return

    scores_skel = pd.concat(
        [pd.read_csv(f, index_col=0) for f in per_method_files]
    )

    all_scores = pd.read_csv(all_scores_path)
    scores_orig = (
        all_scores[all_scores["dataset"] == dataset]
        .drop(columns="dataset")
        .set_index("method")
    )

    cols = [c for c in METRICS_COLS if c in scores_skel.columns and c in scores_orig.columns]
    common_methods = scores_skel.index.intersection(scores_orig.index)
    if common_methods.empty:
        print(f"No common methods for {dataset} — skipping")
        return

    scores_skel = scores_skel.loc[common_methods, cols]
    scores_orig = scores_orig.loc[common_methods, cols]

    # ── compute ratio skeleton / original ──────────────────────────────────────
    ratio_raw = (scores_skel / scores_orig.replace(0, np.nan)).abs()

    # mask near-zero baseline for gs_f1
    if "gs_f1" in ratio_raw.columns and "gs_f1" in scores_orig.columns:
        ratio_raw.loc[scores_orig["gs_f1"] < GS_MIN_THRESHOLD, "gs_f1"] = float("nan")

    # drop unreliable metrics (median baseline too low)
    reliable = scores_orig.median() > 0.01
    cols = [c for c in cols if reliable.get(c, False)]
    ratio_raw = ratio_raw[cols]

    # accumulate long-form records for heatmaps (before capping)
    if heatmap_records is not None:
        long = (ratio_raw.reset_index()
                .melt(id_vars="index", var_name="metric", value_name="ratio")
                .rename(columns={"index": "method"}))
        long["dataset"] = dataset
        heatmap_records.append(long)

    ratio = ratio_raw.clip(lower=0, upper=2)

    # surrogate names
    ratio.index   = ratio.index.map(lambda x: surrogate_names.get(x, x))
    ratio.columns = ratio.columns.map(lambda x: surrogate_names.get(x, x))

    # order methods by METHODS list
    ordered = list(dict.fromkeys(surrogate_names.get(m, m) for m in METHODS))
    ratio = ratio.reindex([m for m in ordered if m in ratio.index])

    scores_long = (
        ratio.reset_index()
        .melt(id_vars="index", var_name="Metric", value_name="Relative score")
        .rename(columns={"index": "Method"})
    )

    # ── plot ───────────────────────────────────────────────────────────────────
    n_metrics = len(cols)
    palette = {m: palette_metrics[m] for m in scores_long['Metric'].unique() if m in palette_metrics}

    fig, ax = plt.subplots(1, 1, figsize=(max(7, len(ratio) * 0.6), 3.2))
    sns.barplot(scores_long, x="Method", y="Relative score", hue="Metric",
                ax=ax, palette=palette)

    ax.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.6, label="No change")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Relative score\n(skeleton-filtered / original GRN)")
    ax.set_title(surrogate_names.get(dataset, dataset), weight="bold")
    ax.margins(x=0.05, y=0.15)
    for side in ["right", "top"]:
        ax.spines[side].set_visible(False)
    ax.legend(title="Metric", loc=(1.05, 0.1), frameon=False)

    plt.tight_layout()
    out_path = f"{figs_dir}/skeleton_{dataset}.png"
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


# ── heatmap helper (same style as causal_directionality) ──────────────────────

CAP = 2.0

def _heatmap(pivot, out_path, xlabel):
    nan_mask = pivot.isna()
    pivot_display = pivot.clip(upper=CAP).fillna(0)
    annot_data = pivot.clip(upper=CAP).applymap(lambda v: '' if pd.isna(v) else f'{v:.2f}')

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'rg', ['#d73027', '#ffffbf', '#1a9850'])  # red (0) → yellow (1) → green (2)

    fig, ax = plt.subplots(figsize=(max(4, pivot.shape[1] * 0.7 + 1),
                                    max(3, pivot.shape[0] * 0.4 + 1)))
    sns.heatmap(pivot_display, ax=ax, cmap=cmap, vmin=0, vmax=CAP,
                mask=nan_mask,
                annot=annot_data, fmt='', linewidths=0.4, linecolor='white',
                annot_kws={'size': 7},
                cbar_kws={'label': 'Relative score', 'shrink': 0.6, 'aspect': 20,
                          'ticks': [0, 1, 2]})
    ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    plt.xticks(rotation=45, ha='right')
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


def plot_heatmaps(records):
    """Build two summary heatmaps from a list of per-dataset ratio records."""
    if not records:
        print("No records for heatmaps — skipping", flush=True)
        return

    df = pd.concat(records, ignore_index=True)
    df['method_name']  = df['method'].map(lambda x: surrogate_names.get(x, x))
    df['metric_name']  = df['metric'].map(lambda x: surrogate_names.get(x, x))
    df['dataset_name'] = df['dataset'].map(lambda x: surrogate_names.get(x, x))

    ordered_methods = list(dict.fromkeys(surrogate_names.get(m, m) for m in METHODS))
    ordered_metrics = [surrogate_names.get(m, m) for m in ALL_METRICS if m not in SKIP_METRICS]

    def _sort_methods(pivot):
        idx = [m for m in ordered_methods if m in pivot.index]
        idx += [m for m in pivot.index if m not in idx]
        return pivot.reindex(idx)

    # ── Plot 1: methods (y) × metrics (x), mean across datasets ──────────────
    pivot_met = (df.groupby(['method_name', 'metric_name'])['ratio']
                   .mean().unstack('metric_name'))
    pivot_met = _sort_methods(pivot_met)
    pivot_met = pivot_met.reindex(columns=[m for m in ordered_metrics if m in pivot_met.columns])
    _heatmap(pivot_met,
             out_path=f"{figs_dir}/skeleton_heatmap_methods_metrics.png",
             xlabel='Metric')

    # ── Plot 2: metrics (y) × datasets (x), mean across methods ──────────────
    pivot_ds = (df.groupby(['metric_name', 'dataset_name'])['ratio']
                  .mean().unstack('dataset_name'))
    pivot_ds = pivot_ds.reindex(index=[m for m in ordered_metrics if m in pivot_ds.index])
    _heatmap(pivot_ds,
             out_path=f"{figs_dir}/skeleton_heatmap_metrics_datasets.png",
             xlabel='Dataset')


# ── entry point ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="op")
parser.add_argument("--all_datasets", action="store_true",
                    help="Plot all datasets that have skeleton scores available")
args = parser.parse_args()

if args.all_datasets:
    skeleton_root = f"{RESULTS_DIR}/experiment/skeleton"
    # subdirectory-based datasets (new style)
    datasets = [d for d in os.listdir(skeleton_root)
                if os.path.isdir(os.path.join(skeleton_root, d))]
    # legacy single-file datasets (e.g. op-skeleton-scores.csv at root)
    import glob as _glob
    for f in _glob.glob(f"{skeleton_root}/*-skeleton-scores.csv"):
        ds = os.path.basename(f).replace("-skeleton-scores.csv", "")
        has_skeleton = os.path.exists(f"{skeleton_root}/{ds}/skeleton.csv")
        if ds not in datasets and has_skeleton:
            datasets.append(ds)
    print(f"Found datasets: {datasets}")
    heatmap_records = []
    for ds in datasets:
        plot_dataset(ds, heatmap_records=heatmap_records)
    plot_heatmaps(heatmap_records)
else:
    plot_dataset(args.dataset)
