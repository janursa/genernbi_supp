import os
import pandas as pd
import numpy as np
import sys
import anndata as ad
import scanpy as sc
from sklearn.impute import KNNImputer
import magic

import os
env = os.environ


def get_par(dataset):
    par = {
        "rna": f"{env['TASK_GRN_INFERENCE_DIR']}/resources/grn_benchmark/inference_data/{dataset}_rna.h5ad",
        'layer': 'lognorm',
        
    }
    return par
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='op', help='Dataset name')
parser.add_argument('--imputation_methods', type=str, nargs='+', help='Imputation methods to apply')
parser.add_argument('--output_dir', type=str, help='Output directory for imputed data')

args = parser.parse_args()
dataset = args.dataset
imputation_methods = args.imputation_methods
output_dir = args.output_dir
print("Imputation methods:", imputation_methods, flush=True)

def naming_convention(dataset, imputation):
    return f"{output_dir}/{dataset}_{imputation}_rna.h5ad"


def impute(par):
    print('Processing dataset with parameters:', par, flush=True)
    print('Loading data...', flush=True)
    rna = ad.read_h5ad(par['rna'])

    print('Data loaded. Performing PCA...', flush=True)
    sc.pp.pca(rna, layer=par['layer'])
    sc.pp.neighbors(rna)

    # Keep X_original in sparse format if possible
    X_original = rna.layers[par['layer']]

    for imputation in imputation_methods:
        print(f"Running imputation: {imputation}", flush=True)

        if imputation == 'original':
            X = X_original

        elif imputation == 'knn':
            # KNN requires dense
            if not isinstance(X_original, np.ndarray):
                X_temp = X_original.toarray() if hasattr(X_original, "toarray") else np.array(X_original)
            else:
                X_temp = X_original.copy()
            X_temp[X_temp == 0] = np.nan
            knn_imputer = KNNImputer(n_neighbors=10, keep_empty_features=True)
            X = knn_imputer.fit_transform(X_temp)
            del X_temp

        elif imputation == 'magic':
            # MAGIC works fine with dense, so convert once here
            if not isinstance(X_original, np.ndarray):
                X_temp = X_original.toarray() if hasattr(X_original, "toarray") else np.array(X_original)
            else:
                X_temp = X_original
            magic_operator = magic.MAGIC()
            X = magic_operator.fit_transform(X_temp)
            del X_temp

        else:
            raise ValueError('Unknown imputation method')

        # Create a lightweight AnnData with just imputed layer
        rna_out = rna.copy()
        rna_out.layers.clear()
        rna_out.layers['X_norm'] = X

        out_path = naming_convention(dataset, imputation)
        rna_out.write(out_path)
        print(f"Saved: {out_path}", flush=True)

if __name__ == '__main__':
    par = get_par(dataset)
    impute(par)