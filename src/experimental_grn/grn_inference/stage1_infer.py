"""
Stage 1 GRN Inference — all algorithms in one script.

Algorithms (selected via --algorithm):
  pearson     Pearson TF-gene correlation
  spearman    Spearman TF-gene correlation
  ridge       Ridge regression per gene (L2)
  lasso       LASSO regression per gene (L1)
  elasticnet  Elastic Net regression per gene (L1+L2)

Output: top --max_n_links edges (default 1,000,000) in h5ad with uns['prediction'].
Naming: {dataset}.S1-{algo}.S2-P0.S3-C0.h5ad
"""
import sys, os, argparse
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from scipy.stats import spearmanr

REPO = '/home/jnourisa/projs/ongoing/task_grn_inference'
sys.path.insert(0, f'{REPO}/src/utils')
from util import manage_layer

# ── Argument parsing ────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--rna',        required=True)
    p.add_argument('--tf_all',     default=f'{REPO}/resources/grn_benchmark/prior/tf_all.csv')
    p.add_argument('--algorithm',  required=True,
                   choices=['pearson','spearman','ridge','lasso','elasticnet'])
    p.add_argument('--prediction', required=True)
    p.add_argument('--max_n_links', type=int, default=1_000_000)
    p.add_argument('--max_cells',   type=int, default=5000,
                   help='Max cells for regression methods (subsampling)')
    p.add_argument('--n_jobs',      type=int, default=20)
    p.add_argument('--layer',       default='lognorm')
    return p.parse_args()

# ── Shared helpers ───────────────────────────────────────────────────

def load_data(par):
    adata = ad.read_h5ad(par.rna)
    layer_key = manage_layer(adata, {'layer': par.layer})
    X = adata.layers[layer_key]
    X = X.toarray() if hasattr(X, 'toarray') else np.array(X)
    X = np.array(X, dtype=np.float32)

    genes = np.array(adata.var_names)
    tf_all = np.loadtxt(par.tf_all, dtype=str)
    tfs = np.array([t for t in tf_all if t in set(genes)])

    # Remove zero-variance genes
    std = X.std(axis=0)
    keep = std > 0
    X = X[:, keep]
    genes = genes[keep]
    tfs = np.array([t for t in tfs if t in set(genes)])

    gene_idx = {g: i for i, g in enumerate(genes)}
    tf_idx = np.array([gene_idx[t] for t in tfs])

    print(f"  {X.shape[0]} cells x {len(genes)} genes, {len(tfs)} TFs")
    return adata, X, genes, tfs, tf_idx


def to_dataframe(sources, targets, weights, max_n_links):
    """Build edge DataFrame, drop self-loops, keep top max_n_links by |weight|."""
    df = pd.DataFrame({'source': sources, 'target': targets, 'weight': weights})
    df = df[df['source'] != df['target']]
    df['abs_weight'] = df['weight'].abs()
    df = df.nlargest(max_n_links, 'abs_weight').drop(columns='abs_weight')
    return df.reset_index(drop=True)


def save_output(df, adata, algorithm, par):
    os.makedirs(os.path.dirname(par.prediction), exist_ok=True)
    df['weight'] = df['weight'].astype(str)
    output = ad.AnnData(
        X=None,
        uns={
            'method_id': f'S1-{algorithm}',
            'dataset_id': adata.uns.get('dataset_id', ''),
            'prediction': df[['source', 'target', 'weight']],
        }
    )
    output.write_h5ad(par.prediction)
    print(f"Saved {len(df):,} edges → {par.prediction}")

# ── Algorithm implementations ────────────────────────────────────────

def run_pearson(X, genes, tfs, tf_idx, max_n_links):
    print("  Computing Pearson correlation matrix...")
    # Mean-centre
    Xc = X - X.mean(axis=0)
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    Xn = Xc / std  # (cells, genes)

    X_tf = Xn[:, tf_idx]   # (cells, n_tfs)
    # dot product normalised by n_cells gives correlation
    corr = (X_tf.T @ Xn) / X.shape[0]  # (n_tfs, n_genes)

    print(f"  Building edge list from {len(tfs)}×{len(genes)} matrix...")
    rows, cols = np.meshgrid(np.arange(len(tfs)), np.arange(len(genes)), indexing='ij')
    sources = tfs[rows.ravel()]
    targets = genes[cols.ravel()]
    weights = corr.ravel().astype(np.float32)
    return to_dataframe(sources, targets, weights, max_n_links)


def run_spearman(X, genes, tfs, tf_idx, max_n_links, max_cells):
    """Spearman via rank-transforming X first, then Pearson on ranks."""
    print("  Rank-transforming expression for Spearman...")
    from scipy.stats import rankdata
    # Subsample cells for memory on large datasets
    n_cells = X.shape[0]
    if n_cells > max_cells:
        print(f"  Subsampling {n_cells} → {max_cells} cells")
        rng = np.random.default_rng(42)
        sel = rng.choice(n_cells, max_cells, replace=False)
        X = X[sel, :]

    # Rank each gene across cells (converts to Spearman)
    X_ranked = np.apply_along_axis(rankdata, 0, X).astype(np.float32)
    # Pearson on ranks = Spearman
    Xc = X_ranked - X_ranked.mean(axis=0)
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    Xn = Xc / std

    X_tf = Xn[:, tf_idx]
    corr = (X_tf.T @ Xn) / Xn.shape[0]

    rows, cols = np.meshgrid(np.arange(len(tfs)), np.arange(len(genes)), indexing='ij')
    sources = tfs[rows.ravel()]
    targets = genes[cols.ravel()]
    weights = corr.ravel().astype(np.float32)
    return to_dataframe(sources, targets, weights, max_n_links)


def _fit_regression_gene(y, X_tf, model_cls, model_kwargs):
    """Fit one regression model; returns coefficient vector (n_tfs,)."""
    if y.std() < 1e-6:
        return np.zeros(X_tf.shape[1], dtype=np.float32)
    m = model_cls(**model_kwargs)
    m.fit(X_tf, y)
    return m.coef_.astype(np.float32)


def run_regression(X, genes, tfs, tf_idx, max_n_links, max_cells, n_jobs,
                   model_cls, model_kwargs, label):
    print(f"  Running {label} regression for {len(genes)} genes (n_jobs={n_jobs})...")
    n_cells = X.shape[0]
    if n_cells > max_cells:
        print(f"  Subsampling {n_cells} → {max_cells} cells")
        rng = np.random.default_rng(42)
        sel = rng.choice(n_cells, max_cells, replace=False)
        X = X[sel, :]

    X_tf = X[:, tf_idx]
    scaler = StandardScaler()
    X_tf_s = scaler.fit_transform(X_tf)

    # Auto alpha: scale with expression variance
    gene_stds = X.std(axis=0)
    alpha = float(np.clip(0.001 * gene_stds.mean(), 1e-4, 0.1))
    print(f"  Alpha = {alpha:.5f}")
    model_kwargs['alpha'] = alpha

    coefs = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(_fit_regression_gene)(X[:, j], X_tf_s, model_cls, dict(model_kwargs))
        for j in range(len(genes))
    )
    B = np.stack(coefs, axis=0)  # (n_genes, n_tfs)

    n_nonzero = np.count_nonzero(B)
    print(f"  Non-zero coefficients: {n_nonzero:,} / {B.size:,} ({100*n_nonzero/B.size:.1f}%)")

    # Build edge list: B[gene_idx, tf_idx] = weight of tf → gene
    gi, ti = np.nonzero(B)
    sources = tfs[ti]
    targets = genes[gi]
    weights = B[gi, ti]
    return to_dataframe(sources, targets, weights, max_n_links)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    par = parse_args()
    print(f"\n=== Stage 1 inference: {par.algorithm.upper()} ===")
    print(f"  Input:  {par.rna}")
    print(f"  Output: {par.prediction}")

    adata, X, genes, tfs, tf_idx = load_data(par)

    algo = par.algorithm
    if algo == 'pearson':
        df = run_pearson(X, genes, tfs, tf_idx, par.max_n_links)

    elif algo == 'spearman':
        df = run_spearman(X, genes, tfs, tf_idx, par.max_n_links, par.max_cells)

    elif algo == 'ridge':
        df = run_regression(X, genes, tfs, tf_idx, par.max_n_links, par.max_cells,
                            par.n_jobs, Ridge,
                            {'max_iter': 1000, 'fit_intercept': True},
                            'Ridge')

    elif algo == 'lasso':
        df = run_regression(X, genes, tfs, tf_idx, par.max_n_links, par.max_cells,
                            par.n_jobs, Lasso,
                            {'max_iter': 2000, 'fit_intercept': True},
                            'LASSO')

    elif algo == 'elasticnet':
        df = run_regression(X, genes, tfs, tf_idx, par.max_n_links, par.max_cells,
                            par.n_jobs, ElasticNet,
                            {'l1_ratio': 0.5, 'max_iter': 2000, 'fit_intercept': True},
                            'ElasticNet')

    print(f"  Final edges: {len(df):,}  |  TFs: {df['source'].nunique()}")
    save_output(df, adata, algo, par)


if __name__ == '__main__':
    main()
