"""
6 plots (one per algorithm): raw scores, x=metrics, grouped bars=C-tags (strategies),
dots=individual datasets, bar height=median across applicable datasets.
"""
import pandas as pd, numpy as np, anndata as ad, glob, os, yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCORE_DIR  = '/home/jnourisa/projs/ongoing/task_grn_inference/temp/experimental_grn/scores'
METRICS_YAML = '/home/jnourisa/projs/ongoing/task_grn_inference/resources/results/metrics_kept_per_dataset.yaml'
OUT_DIR    = '/home/jnourisa/projs/ongoing/task_grn_inference/temp/experimental_grn/results'

ALGOS   = ['pearson','spearman','lasso','ridge','elasticnet','grnboost']
C_TAGS  = ['C0','C1','C2','C3','C4','C5','C6']

C_COLORS = {
    'C0': '#888888',
    'C1': '#4e9af1', 'C2': '#1a5fb4',
    'C3': '#f5a623', 'C4': '#e67e00',
    'C5': '#e74c3c',
    'C6': '#27ae60',
}

with open(METRICS_YAML) as f:
    metrics_per_ds = yaml.safe_load(f)
final_metrics = sorted(set(m for ms in metrics_per_ds.values() for m in ms))

# ── load scores ──────────────────────────────────────────────────────────────
records = []
for fpath in sorted(glob.glob(os.path.join(SCORE_DIR, '*.score.h5ad'))):
    a   = ad.read_h5ad(fpath)
    ds  = str(a.uns.get('dataset_id', ''))
    fname = os.path.basename(fpath).replace('.score.h5ad', '')
    try:
        after = fname[len(ds)+1:]
        s1, c_tag = after.split('.', 1)
        algo = s1.replace('S1-', '')
    except:
        continue
    row = dict(dataset=ds, algo=algo, c_tag=c_tag)
    for mid, mval in zip(a.uns.get('metric_ids', []), a.uns.get('metric_values', [])):
        row[mid] = pd.to_numeric(mval, errors='coerce')
    records.append(row)
df = pd.DataFrame(records)

# ── one plot per algorithm ────────────────────────────────────────────────────
for algo in ALGOS:
    sub = df[df.algo == algo]

    n_metrics = len(final_metrics)
    n_tags    = len(C_TAGS)
    bar_w     = 0.11
    group_gap = 0.15
    group_w   = n_tags * bar_w + group_gap

    fig, ax = plt.subplots(figsize=(max(18, n_metrics * 2.2), 6))

    xtick_pos, xtick_lab = [], []

    for mi, metric in enumerate(final_metrics):
        applicable_ds = [ds for ds, ms in metrics_per_ds.items() if metric in ms]
        group_center  = mi * group_w
        tag_positions = [group_center + (ti - n_tags/2 + 0.5) * bar_w for ti in range(n_tags)]

        for ti, (c_tag, xpos) in enumerate(zip(C_TAGS, tag_positions)):
            color = C_COLORS[c_tag]
            vals  = sub[(sub.c_tag == c_tag) & (sub.dataset.isin(applicable_ds))][metric].dropna()
            if vals.empty:
                continue
            med = vals.median()
            ax.bar(xpos, med, width=bar_w * 0.85, color=color, alpha=0.85, zorder=2)
            # individual dataset dots
            jitter = np.random.default_rng(ti + mi * 100).uniform(-bar_w*0.25, bar_w*0.25, size=len(vals))
            ax.scatter(xpos + jitter, vals.values, color='black', s=14, alpha=0.6, zorder=3, linewidths=0)

        xtick_pos.append(group_center)
        xtick_lab.append(metric.replace('_', '\n'))

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lab, fontsize=9)
    ax.set_ylabel('Raw score', fontsize=11)
    ax.set_title(f'{algo} — raw scores by metric and C-tag strategy\n(bars = median across applicable datasets, dots = individual datasets)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_xlim(-group_w * 0.6, (n_metrics - 1) * group_w + group_w * 0.6)

    legend_patches = [mpatches.Patch(color=C_COLORS[t], label=t) for t in C_TAGS]
    ax.legend(handles=legend_patches, title='Strategy', fontsize=9, title_fontsize=9,
              loc='upper right', ncol=4, framealpha=0.9)

    out = f'{OUT_DIR}/strategy_raw_{algo}.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

print('Done.')
