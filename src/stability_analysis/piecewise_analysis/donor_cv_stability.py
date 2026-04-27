"""
Donor CV Stability Analysis
-----------------------------
Mirrors the old metrics_stability framework but uses donor leave-one-out CV
(train on 2 donors, hold out 1 — 3 folds total) instead of random 5-fold
within each donor.

For each gene × GRN model:
  - Select top-n TFs from the GRN (same as old: fill_zeros_in_grn + regulators_consensus)
  - Fit Ridge on 2 donors, hold out 1 — for all 3 donor combinations
  - stability = mean( |mean(coef)| / (std(coef) + eps) ) across the 3 coef vectors
  - present = gene in model's GRN target set

Plots (per theta):
  Left  — per-model violin (present genes), star on least-stable model
  Right — GRN-derived (present) vs Random (not present), star on less-stable group
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, LabelEncoder
from scipy.stats import mannwhitneyu
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from geneRNBI.src.helper import load_env
env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
sys.path.append(env['geneRNBI_DIR'])
sys.path.append(env['UTILS_DIR'])
sys.path.append(env['METRICS_DIR'])

from util import naming_convention, process_links
from regression.helper import net_to_matrix, fill_zeros_in_grn
from src.helper import palette_methods, colors_blind
from src.params import get_par

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET    = 'op'
GRN_MODELS = ['pearson_corr', 'scenicplus', 'grnboost', 'ppcor']
THETAS     = [0.25, 0.75]
EPS        = 1e-6
N_JOBS     = 10
OUT_DIR    = f'{RESULTS_DIR}/temp/grn_weight_dynamics'
os.makedirs(OUT_DIR, exist_ok=True)

SURROGATE = {
    'pearson_corr': 'Pearson Corr.',
    'scenicplus':   'Scenic+',
    'grnboost':     'GRNBoost2',
    'ppcor':        'PPCOR',
}
ORDER = ['Pearson Corr.', 'Scenic+', 'GRNBoost2', 'PPCOR']


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data(par):
    adata = ad.read_h5ad(par['evaluation_data'])
    X = adata.layers['lognorm']
    if hasattr(X, 'toarray'): X = X.toarray()
    X = RobustScaler().fit_transform(X.astype(float))
    groups = LabelEncoder().fit_transform(adata.obs['donor_id'].values)
    gene_names = np.asarray(adata.var_names)
    return X, gene_names, groups

def load_grn(model, par, gene_names):
    path = f"{par['grn_models_dir']}/{naming_convention(DATASET, model)}"
    net  = ad.read_h5ad(path).uns['prediction']
    net  = process_links(net, par)
    mat  = net_to_matrix(net, gene_names)
    return fill_zeros_in_grn(mat), set(net['target'].unique())

def load_n_features(par, gene_names, theta):
    with open(par['regulators_consensus']) as f:
        data = json.load(f)
    key = str(theta)
    return np.array([int(data.get(g, {}).get(key, 0)) for g in gene_names])

def get_tf_mask(par, gene_names):
    tf_names = np.loadtxt(par['tf_all'], dtype=str)
    return np.isin(gene_names, tf_names)


# ── Core: donor-LOO stability per gene ────────────────────────────────────────
def _process_gene(j, grn, X, groups, n_features, present):
    if n_features == 0:
        return None

    importance = np.abs(grn[:, j]).copy()
    importance[j] = -1
    feat_idx = np.argsort(importance)[-n_features:]
    feat_idx = feat_idx[importance[feat_idx] > 0]
    if len(feat_idx) == 0:
        return None

    X_feat = X[:, feat_idx]
    y      = X[:, j]

    # 3-fold donor LOO: fold k → held-out donor group k
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import r2_score as r2_fn
    coefs, r2s = [], []
    for tr, te in LeaveOneGroupOut().split(X_feat, y, groups):
        reg = Ridge(alpha=1.0, random_state=0)
        reg.fit(X_feat[tr], y[tr])
        coefs.append(reg.coef_)
        r2s.append(float(np.clip(r2_fn(y[te], reg.predict(X_feat[te])), 0, 1)))

    coefs = np.array(coefs)   # (3, n_features)
    stability = float(np.mean(
        np.abs(np.mean(coefs, axis=0) + EPS) / (np.std(coefs, axis=0) + EPS)
    ))
    return {'j': j, 'present': present, 'stability': stability,
            'n_regulator': len(feat_idx),
            'r2_d0': r2s[0], 'r2_d1': r2s[1], 'r2_d2': r2s[2]}


def compute_donor_cv(X, grn, gene_names, groups, n_features_arr, grn_targets):
    # mask out non-TF rows
    results = Parallel(n_jobs=N_JOBS, prefer='threads')(
        delayed(_process_gene)(
            j, grn, X, groups, int(n_features_arr[j]),
            gene_names[j] in grn_targets
        )
        for j in range(len(gene_names))
    )
    rows = []
    for j, r in enumerate(results):
        if r is not None:
            r['gene'] = gene_names[j]
            rows.append(r)
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_donor_cv(df, theta, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(2.7, 2),
                             width_ratios=[1.5, 1], sharey=True)

    # Left: per-model, present genes
    ax = axes[0]
    pres = df[df['present']].copy()
    cap = pres['stability'].quantile(0.99)
    pres = pres[pres['stability'] <= cap]
    sns.violinplot(data=pres, x='model', y='stability', order=ORDER,
                   palette=palette_methods, inner=None, linewidth=0.5, cut=0, ax=ax)
    ax.set_ylabel('Stability', fontsize=7)
    ax.set_xlabel('')
    ax.set_yticks([])
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    for lbl in ax.get_xticklabels(): lbl.set_ha('right')

    medians = pres.groupby('model')['stability'].median()
    least = ORDER.index(medians[ORDER].idxmin())
    sigs = ['']*4; sigs[least] = '*'
    for i, s in enumerate(sigs):
        ax.text(i, cap * 0.95, s, ha='center', va='bottom', fontsize=14, color='red')

    # Right: GRN-derived (present) vs random (absent)
    ax = axes[1]
    cap2 = df['stability'].quantile(0.98)
    sub2 = df[df['stability'] <= cap2].copy()
    sns.violinplot(data=sub2, x='present', y='stability', order=[True, False],
                   palette=colors_blind, inner=None, linewidth=0.5, cut=0, ax=ax)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.set_xticklabels(['GRN derived', 'Random'])
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    ax.set_yticks([])
    for lbl in ax.get_xticklabels(): lbl.set_ha('right')

    g = sub2[sub2['present']==True]['stability'].dropna()
    r = sub2[sub2['present']==False]['stability'].dropna()
    _, p = mannwhitneyu(g, r, alternative='two-sided')
    lower_x = 1 if r.median() <= g.median() else 0
    sigs2 = ['', '']
    if p < 0.05: sigs2[lower_x] = '*'
    for i, s in enumerate(sigs2):
        ax.text(i, cap2 * 0.95, s, ha='center', va='bottom', fontsize=14, color='red')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  → {os.path.abspath(out_path)}")
    print(f"    MW p={p:.2e} | GRN median={g.median():.3f} | Random median={r.median():.3f}")
    print(f"    Least-stable model: {medians[ORDER].idxmin()} (median={medians[ORDER].min():.3f})")


# ── Donor heatmap ─────────────────────────────────────────────────────────────
def plot_donor_heatmap(df, theta, out_path):
    """3×3 Spearman correlation of per-gene R² across held-out donors.
    Left panel = GRN-derived genes; Right panel = randomly-assigned (absent) genes.
    Mirrors old plot_regression_perfromance_similarity_donors."""

    donor_labels = ['Donor 1', 'Donor 2', 'Donor 3']

    def _corr_matrix(sub):
        # pivot: rows = held-out donor, cols = (model, gene)
        long = sub[['gene', 'model', 'r2_d0', 'r2_d1', 'r2_d2']].melt(
            id_vars=['gene', 'model'], var_name='held_out', value_name='r2'
        )
        long['held_out'] = long['held_out'].map(
            {'r2_d0': donor_labels[0], 'r2_d1': donor_labels[1], 'r2_d2': donor_labels[2]}
        )
        tbl = long.pivot_table(index='held_out', columns=['model', 'gene'],
                               values='r2', aggfunc='first')
        tbl = tbl.dropna(axis=1)
        corr = tbl.T.corr(method='spearman')
        return corr.reindex(index=donor_labels, columns=donor_labels)

    corr_grn    = _corr_matrix(df[df['present']])
    corr_random = _corr_matrix(df[~df['present']])

    vmin = min(corr_grn.min().min(), corr_random.min().min())
    vmax = max(corr_grn.max().max(), corr_random.max().max())

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5), sharey=True)

    for ax, corr, title in zip(axes,
                               [corr_grn, corr_random],
                               ['GRN-derived', 'Randomly-assigned']):
        m = corr.copy()
        np.fill_diagonal(m.values, np.nan)
        sns.heatmap(m, annot=True, fmt='.2f', cmap='viridis',
                    cbar=False, ax=ax, vmin=vmin, vmax=vmax,
                    annot_kws={'size': 6}, linewidths=0.3)
        ax.set_title(title, fontsize=7, pad=6)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=6)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params('y', rotation=0)
        for lbl in ax.get_xticklabels(): lbl.set_ha('right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  → {os.path.abspath(out_path)}")
    print(f"    GRN corr (off-diag mean):    {corr_grn.values[~np.eye(3,dtype=bool)].mean():.3f}")
    print(f"    Random corr (off-diag mean): {corr_random.values[~np.eye(3,dtype=bool)].mean():.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    par = get_par(DATASET)
    par['grn_models_dir'] = f'{RESULTS_DIR}/{DATASET}'

    print("Loading expression data…")
    X, gene_names, groups = load_data(par)
    tf_mask = get_tf_mask(par, gene_names)
    print(f"  {X.shape[0]} cells × {X.shape[1]} genes | groups: {np.unique(groups, return_counts=True)}")

    for theta in THETAS:
        csv_path = f'{OUT_DIR}/donor_cv_stability_theta{theta}.csv'
        print(f"\n{'='*55}\nTheta = {theta}")

        n_features_arr = load_n_features(par, gene_names, theta)

        if os.path.exists(csv_path):
            print(f"  Loading cached {csv_path}")
            all_df = pd.read_csv(csv_path)
        else:
            all_results = []
            for model in GRN_MODELS:
                print(f"  Processing {SURROGATE[model]}…")
                grn, grn_targets = load_grn(model, par, gene_names)
                # zero out non-TF rows (same as old analysis)
                grn[~tf_mask, :] = 0

                df = compute_donor_cv(X, grn, gene_names, groups,
                                      n_features_arr, grn_targets)
                df['model'] = SURROGATE[model]
                all_results.append(df)

                pres = df['present']
                print(f"    genes={len(df)} | present={pres.sum()} | "
                      f"GRN stab={df.loc[pres,'stability'].median():.3f} | "
                      f"absent stab={df.loc[~pres,'stability'].median():.3f}")

            all_df = pd.concat(all_results, ignore_index=True)
            all_df.to_csv(csv_path, index=False)
            print(f"  CSV → {os.path.abspath(csv_path)}")

        plot_donor_cv(all_df, theta,
                      f'{OUT_DIR}/donor_cv_stability_theta{theta}.png')
        plot_donor_heatmap(all_df, theta,
                           f'{OUT_DIR}/donor_cv_heatmap_theta{theta}.png')

    print(f"\n✓ Done. Outputs in {OUT_DIR}")


if __name__ == '__main__':
    main()
