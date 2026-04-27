"""
GRN Weight Dynamics Experiment
--------------------------------
Two complementary analyses of TF weight stability across folds:

  1. Per-donor 5-fold CV  — matches old metrics_stability/script.py methodology.
     For each donor, run 5-fold CV; stability = mean(|coef_mean|/(coef_std+eps));
     aggregate (mean) across donors per gene.

  2. Donor leave-one-out (3-fold)  — each of the 3 folds trains on cells from 2
     donors and holds out the 3rd. Stability = consistency of Ridge coefficients
     across the 3 donor combinations.

Random baseline: same n TFs per gene drawn uniformly from the TF pool.
Cached to CSV; re-plotting skips recomputation.
Gene loop parallelised with joblib.
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
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
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
N_FOLDS    = 5       # for per-donor analysis
MIN_TFS    = 3
EPS        = 1e-6
RNG_SEED   = 42
N_JOBS     = 10
OUT_DIR    = f'{RESULTS_DIR}/temp/grn_weight_dynamics'
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_DISPLAY = {
    'pearson_corr': 'Pearson Corr.',
    'scenicplus':   'Scenic+',
    'grnboost':     'GRNBoost2',
    'ppcor':        'PPCOR',
}


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data(par):
    """Returns scaled X, gene_names, and donor_indices dict."""
    adata = ad.read_h5ad(par['evaluation_data'])
    X = adata.layers['lognorm']
    if hasattr(X, 'toarray'): X = X.toarray()
    X = RobustScaler().fit_transform(X.astype(float))
    gene_names = np.asarray(adata.var_names)
    donor_indices = {
        d: np.where(adata.obs['donor_id'] == d)[0]
        for d in adata.obs['donor_id'].unique()
    }
    return X, gene_names, donor_indices

def load_grn(model, par, gene_names):
    path = f"{par['grn_models_dir']}/{naming_convention(DATASET, model)}"
    net  = ad.read_h5ad(path).uns['prediction']
    net  = process_links(net, par)
    return net_to_matrix(net, gene_names), set(net['target'].unique())

def load_n_features(par, gene_names, theta):
    with open(par['regulators_consensus']) as f:
        data = json.load(f)
    key = str(theta)
    return {g: int(data[g][key]) for g in gene_names if g in data and key in data[g]}


# ── Core metric ────────────────────────────────────────────────────────────────
def _stability(coef_matrix):
    """mean( |mean_coef| / (std_coef + eps) ) — higher = more stable."""
    mean = np.abs(coef_matrix.mean(axis=0))
    std  = coef_matrix.std(axis=0)
    return float((mean / (std + EPS)).mean())

def _fit_coef(X_sub, y_sub, feat_idx):
    reg = Ridge(alpha=1.0, random_state=0)
    reg.fit(X_sub[:, feat_idx], y_sub)
    return reg.coef_


# ── Analysis 1: per-donor 5-fold CV ───────────────────────────────────────────
def _process_gene_per_donor(j, gene, X, grn_filled, tf_indices,
                             n_features_dict, grn_targets, donor_indices, rng_seed):
    n_features = n_features_dict.get(gene, 0)
    if n_features < MIN_TFS:
        return None

    importance = np.abs(grn_filled[:, j]).copy()
    importance[j] = -1
    grn_idx = np.argsort(importance)[::-1][:n_features]
    grn_idx = grn_idx[importance[grn_idx] > 0]
    if len(grn_idx) < MIN_TFS:
        return None

    rng = np.random.default_rng(rng_seed + j)
    pool = [i for i in tf_indices if i != j]
    rand_idx = rng.choice(pool, size=len(grn_idx), replace=False)

    grn_stabs, rand_stabs = [], []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rng_seed)

    for d_idx in donor_indices.values():
        # per-donor 5-fold
        grn_coefs, rand_coefs = [], []
        for tr, _ in kf.split(d_idx):
            tr_cells = d_idx[tr]
            y_tr = X[tr_cells, j]
            grn_coefs.append(_fit_coef(X[tr_cells], y_tr, grn_idx))
            rand_coefs.append(_fit_coef(X[tr_cells], y_tr, rand_idx))
        grn_stabs.append(_stability(np.array(grn_coefs)))
        rand_stabs.append(_stability(np.array(rand_coefs)))

    return {
        'gene':          gene,
        'present':       gene in grn_targets,
        'grn_stability': float(np.mean(grn_stabs)),
        'rand_stability': float(np.mean(rand_stabs)),
        'n_tfs':         len(grn_idx),
    }

def compute_per_donor(X, grn_matrix, gene_names, tf_indices,
                      n_features_dict, grn_targets, donor_indices):
    grn_filled = fill_zeros_in_grn(grn_matrix)
    non_tf = np.ones(len(gene_names), bool); non_tf[tf_indices] = False
    grn_filled[non_tf, :] = 0

    results = Parallel(n_jobs=N_JOBS, prefer='threads')(
        delayed(_process_gene_per_donor)(
            j, gene, X, grn_filled, tf_indices,
            n_features_dict, grn_targets, donor_indices, RNG_SEED
        )
        for j, gene in enumerate(gene_names)
    )
    return pd.DataFrame([r for r in results if r is not None])


# ── Analysis 2: 3-fold donor leave-one-out ────────────────────────────────────
def _process_gene_donor_loo(j, gene, X, grn_filled, tf_indices,
                             n_features_dict, grn_targets, donor_indices, rng_seed):
    n_features = n_features_dict.get(gene, 0)
    if n_features < MIN_TFS:
        return None

    importance = np.abs(grn_filled[:, j]).copy()
    importance[j] = -1
    grn_idx = np.argsort(importance)[::-1][:n_features]
    grn_idx = grn_idx[importance[grn_idx] > 0]
    if len(grn_idx) < MIN_TFS:
        return None

    rng = np.random.default_rng(rng_seed + j)
    pool = [i for i in tf_indices if i != j]
    rand_idx = rng.choice(pool, size=len(grn_idx), replace=False)

    donors = list(donor_indices.keys())  # [donor_0, donor_1, donor_2]
    grn_coefs, rand_coefs = [], []

    for held_out in donors:
        train_cells = np.concatenate([donor_indices[d] for d in donors if d != held_out])
        y_train = X[train_cells, j]
        grn_coefs.append(_fit_coef(X[train_cells], y_train, grn_idx))
        rand_coefs.append(_fit_coef(X[train_cells], y_train, rand_idx))

    return {
        'gene':           gene,
        'present':        gene in grn_targets,
        'grn_stability':  _stability(np.array(grn_coefs)),
        'rand_stability': _stability(np.array(rand_coefs)),
        'n_tfs':          len(grn_idx),
    }

def compute_donor_loo(X, grn_matrix, gene_names, tf_indices,
                      n_features_dict, grn_targets, donor_indices):
    grn_filled = fill_zeros_in_grn(grn_matrix)
    non_tf = np.ones(len(gene_names), bool); non_tf[tf_indices] = False
    grn_filled[non_tf, :] = 0

    results = Parallel(n_jobs=N_JOBS, prefer='threads')(
        delayed(_process_gene_donor_loo)(
            j, gene, X, grn_filled, tf_indices,
            n_features_dict, grn_targets, donor_indices, RNG_SEED
        )
        for j, gene in enumerate(gene_names)
    )
    return pd.DataFrame([r for r in results if r is not None])


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_combined(df, out_path):
    model_order = [MODEL_DISPLAY[m] for m in GRN_MODELS
                   if MODEL_DISPLAY[m] in df['model'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 4),
                             width_ratios=[1.5, 1], sharey=True)

    # Left: per-model (GRN-present only, clipped 99th)
    ax = axes[0]
    sub = df[df['present']].copy()
    cap = sub['grn_stability'].quantile(0.99)
    sub = sub[sub['grn_stability'] <= cap]
    sns.violinplot(data=sub, x='model', y='grn_stability', order=model_order,
                   palette=palette_methods, inner=None, linewidth=0.5, cut=0, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Stability', fontsize=7)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')

    # red star on least-stable model (lowest median)
    medians = sub.groupby('model')['grn_stability'].median()
    least   = model_order.index(medians[model_order].idxmin())
    sigs    = [''] * len(model_order)
    sigs[least] = '*'
    ypos = cap * 0.95
    for i, s in enumerate(sigs):
        ax.text(i, ypos, s, ha='center', va='bottom', fontsize=20, color='red')

    # Right: Random vs GRN-derived
    ax = axes[1]
    long = pd.concat([
        df[['grn_stability']].rename(columns={'grn_stability': 'stability'}).assign(source=False),
        df[['rand_stability']].rename(columns={'rand_stability': 'stability'}).assign(source=True),
    ])
    cap2 = long['stability'].quantile(0.98)
    long = long[long['stability'] <= cap2]
    sns.violinplot(data=long, x='source', y='stability', order=[True, False],
                   palette=colors_blind, inner=None, linewidth=0.5, cut=0, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(['Random', 'GRN derived'])
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')

    g = df.loc[df['grn_stability']  <= cap2, 'grn_stability'].dropna()
    r = df.loc[df['rand_stability'] <= cap2, 'rand_stability'].dropna()
    _, p = mannwhitneyu(g, r, alternative='two-sided')
    lower_x = 0 if r.median() <= g.median() else 1
    sigs2 = ['', '']
    if p < 0.05:
        sigs2[lower_x] = '*'
    ypos2 = cap2 * 0.95
    for i, s in enumerate(sigs2):
        ax.text(i, ypos2, s, ha='center', va='bottom', fontsize=20, color='red')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"  → {os.path.abspath(out_path)}")


# ── Main ───────────────────────────────────────────────────────────────────────
def run_analysis(tag, compute_fn, X, gene_names, tf_indices, donor_indices,
                 par, theta, n_features_dict):
    """Load from cache or compute, then plot."""
    csv_path = f'{OUT_DIR}/{tag}_theta{theta}.csv'

    if os.path.exists(csv_path):
        print(f"  [{tag}] Loading cached {csv_path}")
        return pd.read_csv(csv_path)

    all_results = []
    for model in GRN_MODELS:
        print(f"  [{tag}] Processing {MODEL_DISPLAY[model]}…")
        grn_matrix, grn_targets = load_grn(model, par, gene_names)
        df = compute_fn(X, grn_matrix, gene_names, tf_indices,
                        n_features_dict, grn_targets, donor_indices)
        df['model'] = MODEL_DISPLAY[model]
        all_results.append(df)
        pres = df['present']
        print(f"    genes={len(df)} | present={pres.sum()} | "
              f"GRN stab={df.loc[pres,'grn_stability'].median():.3f} "
              f"rand stab={df['rand_stability'].median():.3f}")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"  CSV → {os.path.abspath(csv_path)}")
    return combined


def main():
    par = get_par(DATASET)
    par['grn_models_dir'] = f'{RESULTS_DIR}/{DATASET}'

    print("Loading expression data…")
    X, gene_names, donor_indices = load_data(par)
    tf_indices = np.where(np.isin(gene_names, np.loadtxt(par['tf_all'], dtype=str)))[0]
    print(f"  {X.shape[0]} cells × {X.shape[1]} genes | {len(tf_indices)} TFs")
    print(f"  Donors: { {d: len(idx) for d, idx in donor_indices.items()} }")

    for theta in THETAS:
        print(f"\n{'='*50}\nTheta = {theta}")
        n_features_dict = load_n_features(par, gene_names, theta)

        # Analysis 1: per-donor 5-fold CV
        df_pd = run_analysis('per_donor', compute_per_donor,
                             X, gene_names, tf_indices, donor_indices,
                             par, theta, n_features_dict)
        plot_combined(df_pd, f'{OUT_DIR}/stability_per_donor_theta{theta}.png')

        # Analysis 2: donor leave-one-out (3-fold)
        df_loo = run_analysis('donor_loo', compute_donor_loo,
                              X, gene_names, tf_indices, donor_indices,
                              par, theta, n_features_dict)
        plot_combined(df_loo, f'{OUT_DIR}/stability_donor_loo_theta{theta}.png')

    print(f"\n✓ Done. Outputs in {OUT_DIR}")


if __name__ == '__main__':
    main()
