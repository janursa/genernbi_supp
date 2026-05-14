"""
heatmap_analysis.py — Modular heatmap analysis for GRN experimental results.

Callable functions:
  load_scores(score_dir)            → raw long-format DataFrame with parsed factors
  normalize_scores(df, metrics_yaml)→ double-normalized DataFrame (same logic as compute_overall_scores)
  plot_algo_vs_variation(df_norm, metrics_yaml, out_path)  → heatmap 1
  plot_algo_vs_metric(df_norm, metrics_yaml, out_path)     → heatmap 2
  plot_metric_vs_variation(df_norm, metrics_yaml, out_path)→ heatmap 3
  run_all(score_dir, out_dir, metrics_yaml_path)           → generate all 3 heatmaps

Usage:
  python heatmap_analysis.py
  or: from heatmap_analysis import run_all; run_all(...)
"""

import os, sys, glob
import numpy as np
import pandas as pd
import anndata as ad
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

REPO         = '/home/jnourisa/projs/ongoing/task_grn_inference'
SCORE_DIR    = f'{REPO}/temp/experimental_grn/scores'
OUT_DIR      = f'{REPO}/temp/experimental_grn/results'
METRICS_YAML = f'{REPO}/resources/results/exp_analysis/metrics_kept_per_dataset.yaml'

# Human-readable labels for C-axis variations
VARIATION_LABELS = {
    'C0': 'C0\nBaseline',
    'C1': 'C1\nMotif ±1kb',
    'C2': 'C2\nMotif ±100kb',
    'C3': 'C3\nATAC ±1kb',
    'C4': 'C4\nATAC ±100kb',
    'C5': 'C5\nATAC ±100kb\n+corr',
    'C6': 'C6\nC2∩C4',
}
VARIATION_ORDER = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

ALGO_ORDER = ['pearson', 'spearman', 'lasso', 'ridge', 'elasticnet', 'grnboost', 'negative_control']


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_scores(score_dir=SCORE_DIR):
    """Load all score files → long-format DataFrame with columns:
    dataset, algo, c_tag, variation, <metric cols...>

    New naming: {ds}.S1-{algo}.{c_tag}.score.h5ad
    """
    records = []
    for fpath in sorted(glob.glob(os.path.join(score_dir, '*.score.h5ad'))):
        a = ad.read_h5ad(fpath)
        ds = str(a.uns.get('dataset_id', ''))
        fname = os.path.basename(fpath).replace('.score.h5ad', '')

        # Parse: {ds}.S1-{algo}.{c_tag}
        try:
            after_ds = fname[len(ds)+1:]          # e.g. "S1-pearson.C0"
            s1_part, c_tag = after_ds.split('.', 1)
            algo = s1_part.replace('S1-', '')
        except Exception:
            print(f"  [WARN] Cannot parse: {fname}")
            continue

        row = dict(dataset=ds, algo=algo, c_tag=c_tag, variation=c_tag)
        for mid, mval in zip(a.uns.get('metric_ids', []), a.uns.get('metric_values', [])):
            row[mid] = pd.to_numeric(mval, errors='coerce')
        records.append(row)

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. NORMALIZATION  (mirrors compute_overall_scores logic)
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(series, clip_negative=True):
    """Min-max normalize a series to [0, 1]; clip negatives to 0 first."""
    s = series.copy().astype(float)
    if clip_negative:
        s[s < 0] = 0
    mn, mx = s.min(), s.max()
    if mx > mn:
        return (s - mn) / (mx - mn)
    return pd.Series(0.0, index=s.index)


def normalize_scores(df, metrics_yaml_path=METRICS_YAML):
    """
    Double-normalize raw scores:
      Step 1 — per dataset × metric: min-max across all models (negatives → 0)
      Step 2 — per metric:           min-max across all models of their dataset-mean
      Step 3 — per dataset:          min-max across all models of their metric-mean

    Returns df_norm: same index structure as df, metric columns replaced with
    normalized values in [0,1].  Also returns `final_metrics` (list) and
    `metrics_per_dataset` (dict).
    """
    with open(metrics_yaml_path) as f:
        metrics_per_ds = yaml.safe_load(f)
    final_metrics = sorted(set(m for ms in metrics_per_ds.values() for m in ms))

    meta_cols = ['dataset', 'algo', 'c_tag', 'variation']
    raw_metrics = [c for c in df.columns if c not in meta_cols]

    # Step 1: per-dataset min-max
    def _norm_dataset_group(grp):
        grp = grp.copy()
        for m in raw_metrics:
            if m in grp.columns:
                grp[m] = _minmax(grp[m])
        return grp

    df_n = df.groupby('dataset', group_keys=False).apply(_norm_dataset_group)

    # Step 2: per-metric re-normalize (mean across datasets first)
    metric_means = df_n.groupby(['algo', 'c_tag', 'variation'])[raw_metrics].mean()
    for m in raw_metrics:
        if m in metric_means.columns:
            metric_means[m] = _minmax(metric_means[m], clip_negative=False)

    # Step 3: per-dataset re-normalize (mean across metrics first)
    dataset_means = df_n.groupby(['dataset', 'algo', 'c_tag', 'variation'])[raw_metrics].mean()
    dataset_means['_ds_mean'] = dataset_means[raw_metrics].mean(axis=1, skipna=True)
    ds_pivot = dataset_means['_ds_mean'].unstack('dataset')
    for ds in ds_pivot.columns:
        ds_pivot[ds] = _minmax(ds_pivot[ds], clip_negative=False)

    return df_n, metric_means, ds_pivot, final_metrics, metrics_per_ds


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_final_metric_cols(metric_means, final_metrics):
    return [m for m in final_metrics if m in metric_means.columns]


def compute_variation_algo_matrix(df_raw, metrics_per_ds, final_metrics):
    """
    Heatmap 1: rows=algo, cols=c_tag → normalized score.

    Normalization: once per (dataset, metric) across ALL (algo, c_tag) rows
    using the global min/max in that cell.  No re-scaling afterward — so if
    grnboost is at C0 peak but drops at C3, that drop is visible.
    """
    meta_cols = ['dataset', 'algo', 'c_tag', 'variation']

    # build long-form (dataset, algo, c_tag, metric, value)
    raw_metrics = [c for c in df_raw.columns if c not in meta_cols and c in
                   set(m for ms in metrics_per_ds.values() for m in ms)]
    long = df_raw[meta_cols + raw_metrics].melt(
        id_vars=meta_cols, var_name='metric', value_name='value')
    long = long.dropna(subset=['value'])

    # keep only applicable (dataset, metric) pairs
    long = long[long.apply(
        lambda r: r['metric'] in metrics_per_ds.get(r['dataset'], []), axis=1)]

    # global min/max per (dataset, metric) across ALL algo+c_tag
    stats = long.groupby(['dataset', 'metric'])['value'].agg(['min', 'max'])
    long = long.join(stats, on=['dataset', 'metric'])
    span = long['max'] - long['min']
    long['norm'] = np.where(span > 1e-9,
                            (long['value'] - long['min']) / span, 0.0)

    # exclude negative_control from the display (it anchors min anyway)
    long = long[long.algo != 'negative_control']

    # aggregate: median over (dataset, metric) per (algo, c_tag)
    matrix = long.groupby(['algo', 'c_tag'])['norm'].median().unstack('c_tag')
    return matrix


def compute_algo_metric_matrix(df_n, metrics_per_ds):
    """Heatmap 2: rows=algo, cols=final_metric → median normalized score."""
    meta_cols = ['dataset', 'algo', 'c_tag', 'variation']
    raw_metrics = [c for c in df_n.columns if c not in meta_cols]

    records = []
    for (algo,), grp in df_n.groupby(['algo']):
        for m in raw_metrics:
            # only include (dataset, metric) pairs where metric is applicable
            applicable = [ds for ds, ms in metrics_per_ds.items() if m in ms]
            vals = grp[grp['dataset'].isin(applicable)][m].dropna()
            if len(vals) > 0:
                records.append({'algo': algo, 'metric': m, 'score': vals.median()})

    mat_df = pd.DataFrame(records)
    if mat_df.empty:
        return pd.DataFrame()
    # re-normalize across algos per metric
    for m, grp in mat_df.groupby('metric'):
        mat_df.loc[mat_df['metric']==m, 'score'] = _minmax(grp['score'], clip_negative=False).values

    matrix = mat_df.pivot_table(index='algo', columns='metric', values='score', aggfunc='median')
    return matrix


def compute_metric_variation_matrix(df_n, metrics_per_ds):
    """Heatmap 3: rows=metric, cols=variation → median normalized score."""
    meta_cols = ['dataset', 'algo', 'c_tag', 'variation']
    raw_metrics = [c for c in df_n.columns if c not in meta_cols]

    records = []
    for (var,), grp in df_n.groupby(['variation']):
        for m in raw_metrics:
            applicable = [ds for ds, ms in metrics_per_ds.items() if m in ms]
            vals = grp[grp['dataset'].isin(applicable)][m].dropna()
            if len(vals) > 0:
                records.append({'variation': var, 'metric': m, 'score': vals.median()})

    mat_df = pd.DataFrame(records)
    if mat_df.empty:
        return pd.DataFrame()
    for m, grp in mat_df.groupby('metric'):
        mat_df.loc[mat_df['metric']==m, 'score'] = _minmax(grp['score'], clip_negative=False).values

    matrix = mat_df.pivot_table(index='metric', columns='variation', values='score', aggfunc='median')
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# 4. HEATMAP PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _draw_heatmap(matrix, title, out_path, figsize=None, fmt='.2f',
                  row_order=None, col_order=None, cmap='RdYlGn',
                  col_labels=None, annot=True):
    """Core heatmap renderer."""
    # reorder rows/cols if requested
    if row_order is not None:
        row_order = [r for r in row_order if r in matrix.index]
        remaining = [r for r in matrix.index if r not in row_order]
        matrix = matrix.loc[row_order + remaining]
    if col_order is not None:
        col_order = [c for c in col_order if c in matrix.columns]
        remaining = [c for c in matrix.columns if c not in col_order]
        matrix = matrix[col_order + remaining]

    if col_labels is not None:
        matrix.columns = [col_labels.get(c, c) for c in matrix.columns]

    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (max(8, n_cols * 1.4), max(4, n_rows * 0.7))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix, ax=ax,
        cmap=cmap, vmin=0, vmax=1,
        annot=annot, fmt=fmt,
        linewidths=0.5, linecolor='#dddddd',
        cbar_kws={'label': 'Normalized score', 'shrink': 0.7},
        annot_kws={'size': 9}
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=9, rotation=0)
    ax.tick_params(axis='y', labelsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PUBLIC PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_algo_vs_variation(df_raw, metrics_per_ds, final_metrics, out_path):
    """Heatmap 1: algo (rows) × variation (cols), globally normalized score."""
    matrix = compute_variation_algo_matrix(df_raw, metrics_per_ds, final_metrics)
    _draw_heatmap(
        matrix,
        title='Algorithm × Variation — Normalized Score\n(global min/max per dataset-metric; median across applicable dataset-metric pairs)',
        out_path=out_path,
        row_order=ALGO_ORDER,
        col_order=VARIATION_ORDER,
        col_labels=VARIATION_LABELS,
        figsize=(12, 5),
    )
    return matrix


def plot_algo_vs_metric(df_n, metrics_per_ds, out_path):
    """Heatmap 2: algo (rows) × metric (cols), normalized score."""
    matrix = compute_algo_metric_matrix(df_n, metrics_per_ds)
    # group metrics by prefix for ordering
    metric_order = sorted(matrix.columns, key=lambda m: (m.split('_')[0], m))
    _draw_heatmap(
        matrix,
        title='Algorithm × Metric — Normalized Score\n(median across datasets and variations)',
        out_path=out_path,
        row_order=ALGO_ORDER,
        col_order=metric_order,
        figsize=(max(18, len(matrix.columns)*0.9), 5),
        fmt='.2f',
    )
    return matrix


def plot_metric_vs_variation(df_n, metrics_per_ds, out_path):
    """Heatmap 3: metric (rows) × variation (cols), normalized score."""
    matrix = compute_metric_variation_matrix(df_n, metrics_per_ds)
    metric_order = sorted(matrix.index, key=lambda m: (m.split('_')[0], m))
    _draw_heatmap(
        matrix,
        title='Metric × Variation — Normalized Score\n(median across algorithms and applicable datasets)',
        out_path=out_path,
        row_order=metric_order,
        col_order=VARIATION_ORDER,
        col_labels=VARIATION_LABELS,
        figsize=(10, max(10, len(matrix.index)*0.45)),
        fmt='.2f',
    )
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_all(score_dir=SCORE_DIR, out_dir=OUT_DIR, metrics_yaml_path=METRICS_YAML):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading scores...")
    df = load_scores(score_dir)
    print(f"  {len(df)} rows | algos: {sorted(df.algo.unique())} | variations: {sorted(df.variation.unique())}")

    # ── Coverage check
    print("\n── Coverage (rows per c_tag):")
    print(df.groupby(['c_tag', 'algo']).size().unstack(fill_value=0).to_string())

    print("\nNormalizing scores...")
    df_n, metric_means, ds_pivot, final_metrics, metrics_per_ds = normalize_scores(df, metrics_yaml_path)

    with open(metrics_yaml_path) as _f:
        import yaml as _yaml
        _metrics_per_ds = _yaml.safe_load(_f)

    print("\nPlotting heatmap 1: algo × variation...")
    m1 = plot_algo_vs_variation(df, _metrics_per_ds, final_metrics,
                                 out_path=f'{out_dir}/heatmap1_algo_variation.png')

    print("\nPlotting heatmap 2: algo × metric...")
    m2 = plot_algo_vs_metric(df_n, metrics_per_ds,
                              out_path=f'{out_dir}/heatmap2_algo_metric.png')

    print("\nPlotting heatmap 3: metric × variation...")
    m3 = plot_metric_vs_variation(df_n, metrics_per_ds,
                                   out_path=f'{out_dir}/heatmap3_metric_variation.png')

    # ── Diagnostic analysis
    print("\n" + "="*60)
    print("DIAGNOSTIC ANALYSIS")
    print("="*60)

    meta_cols = ['dataset', 'algo', 'c_tag', 'variation']
    raw_metrics = [c for c in df.columns if c not in meta_cols]

    # 1. Negative control raw scores vs best algo per dataset
    print("\n[1] Negative control raw scores vs algo median (per metric):")
    nc = df[df.algo == 'negative_control']
    al = df[df.algo != 'negative_control']
    for m in raw_metrics:
        nc_med = nc[m].dropna().median()
        al_med = al[m].dropna().median()
        flag = " *** SUSPICIOUS: neg_ctrl >= algo" if (not np.isnan(nc_med) and not np.isnan(al_med) and nc_med >= al_med) else ""
        print(f"  {m:40s}  neg_ctrl={nc_med:.4f}  algos={al_med:.4f}{flag}")

    # 2. C5 gene coverage check - look at raw score counts
    print("\n[2] Score count per (dataset × algo × c_tag) — suspicious zeros or NaNs:")
    for m in raw_metrics:
        zero_rows = df[(df[m].isna()) | (df[m] == 0)][['dataset','algo','c_tag',m]]
        if len(zero_rows) > 0:
            print(f"  {m}: {len(zero_rows)} zero/NaN rows")
            print(zero_rows[zero_rows.algo != 'negative_control'].to_string(index=False))

    # 3. C-axis trend: does score increase from C0 → C1 → C2? C3 → C4 → C5?
    print("\n[3] C-axis trend (median normalized score across algos, excl. negative_control):")
    df_n_algos = df_n[df_n.algo != 'negative_control']
    for m in raw_metrics:
        if m not in df_n_algos.columns:
            continue
        trend = df_n_algos.groupby('c_tag')[m].median().reindex(VARIATION_ORDER)
        if trend.notna().sum() < 2:
            continue
        print(f"  {m}:")
        print("    " + "  ".join(f"{c}={v:.3f}" for c,v in trend.items() if not np.isnan(v)))

    # 4. Best algo overall
    print("\n[4] Best algorithm overall (normalized score, excl. negative_control):")
    fm = [m for m in final_metrics if m in df_n_algos.columns]
    algo_scores = df_n_algos.groupby('algo')[fm].median().median(axis=1).sort_values(ascending=False)
    print(algo_scores.to_string())

    # 5. C5 edge count sanity
    print("\n[5] C5 prediction file edge counts (sample):")
    import anndata as ad2
    for f in sorted(glob.glob(f'{REPO}/temp/experimental_grn/predictions_final/*.C5.h5ad'))[:6]:
        a2 = ad2.read_h5ad(f)
        pred = a2.uns.get('prediction', pd.DataFrame())
        print(f"  {os.path.basename(f)}: {len(pred)} edges")

    return m1, m2, m3


if __name__ == '__main__':
    run_all()
