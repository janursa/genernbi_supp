"""
Post-analysis for GBM vs Ridge experiment.
Loads saved regression scores and produces comparison plots.

Usage:
    python src/exp_analysis/post_gb_vs_ridge.py --dataset op
"""
import argparse
import os
import sys
import warnings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from genernbi_supp.src.helper import load_env
env = load_env()

sys.path.insert(0, env['genernbi_supp_DIR'])
from src.helper import plot_raw_scores, surrogate_names, colors_blind

RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{RESULTS_DIR}/figs"
os.makedirs(figs_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='op')
args = parser.parse_args()
dataset = args.dataset

rr_dir = f"{RESULTS_DIR}/experiment/gb_vs_ridge/{dataset}"


METRICS_COLS = ['r_precision', 'r_recall']


def load_scores(reg_type):
    path = f"{rr_dir}/scores_{reg_type}.csv"
    df = pd.read_csv(path, index_col=0)
    df = df.set_index('model').drop(columns=['dataset'], errors='ignore')
    # keep only the two regression sub-metrics
    cols = [c for c in METRICS_COLS if c in df.columns]
    return df[cols]


# Load available score files
ridge_path = f"{rr_dir}/scores_ridge.csv"
gb_path = f"{rr_dir}/scores_GB.csv"

has_ridge = os.path.exists(ridge_path)
has_gb = os.path.exists(gb_path)

if not has_ridge and not has_gb:
    raise FileNotFoundError(f"No score files found in {rr_dir}. Run script.py first.")

scores_ridge = load_scores('ridge') if has_ridge else None
scores_gb = load_scores('GB') if has_gb else None

# --- Side-by-side raw scores plot ---
left_scores = scores_ridge if has_ridge else scores_gb
right_scores = scores_gb if has_gb else scores_ridge
left_label = 'Ridge' if has_ridge else 'GBM'
right_label = 'GBM' if has_gb else 'Ridge'

n_models = max(len(left_scores), len(right_scores))
fig, axes = plt.subplots(1, 2, figsize=(7, max(3, n_models * 0.65)), sharey=False)

ax = axes[0]
plot_raw_scores(left_scores, ax)
ax.set_xlabel('')
ax.set_title(left_label, pad=15, weight='bold')

ax = axes[1]
plot_raw_scores(right_scores, ax)
ax.set_title(right_label, pad=15, weight='bold')
ax.set_xlabel('')
ax.set_yticklabels([])

plt.tight_layout()
out_path = f"{figs_dir}/gb_vs_ridge_raw_{dataset}.png"
fig.savefig(out_path, dpi=300, transparent=True, bbox_inches='tight')
print(f"Saved: {out_path}", flush=True)


# --- Relative performance plot (GBM / Ridge ratio) ---
# When only ridge is available (test mode), use ridge/ridge so layout can be verified.
numerator = scores_gb if has_gb else scores_ridge
denominator = scores_ridge if has_ridge else scores_gb
ylabel = 'Relative performance\n(GBM / Ridge)' if (has_gb and has_ridge) else 'Relative performance\n(Ridge / Ridge — test mode)'

common_models = numerator.index.intersection(denominator.index)
common_cols = numerator.columns.intersection(denominator.columns)

scores_mat_n = numerator.loc[common_models, common_cols] / denominator.loc[common_models, common_cols]

scores_long = scores_mat_n.reset_index().melt(id_vars='model')
scores_long['model'] = scores_long['model'].map(lambda x: surrogate_names.get(x, x))
scores_long['variable'] = scores_long['variable'].map(lambda x: surrogate_names.get(x, x))

n_methods = scores_long['model'].nunique()
fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.8))
sns.barplot(scores_long, x='model', y='value', hue='variable', ax=ax, palette=colors_blind)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel('')
ax.set_ylabel(ylabel)
ax.set_title(surrogate_names.get(dataset, dataset), weight='bold')
ax.margins(x=0.05, y=0.15)
for side in ['right', 'top']:
    ax.spines[side].set_visible(False)
ax.legend(title='Metric', loc=(1.05, 0.1), frameon=False)
plt.tight_layout()

out_path2 = f"{figs_dir}/gb_vs_ridge_{dataset}.png"
fig.savefig(out_path2, dpi=300, transparent=True, bbox_inches='tight')
print(f"Saved: {out_path2}", flush=True)
