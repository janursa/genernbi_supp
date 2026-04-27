import os
import pandas as pd
import numpy as np
import anndata as ad
import sys

env = os.environ
assert 'TASK_GRN_INFERENCE_DIR' in env, "Please set the TASK_GRN_INFERENCE_DIR environment variable."
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
RESULT_DIR = env['RESULTS_DIR']

GRN_BENCHMARK_DIR = env.get('GRN_BENCHMARK_DIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(GRN_BENCHMARK_DIR)  # contains params.py

sys.path.append(f'{TASK_GRN_INFERENCE_DIR}/src')
sys.path.append(f'{TASK_GRN_INFERENCE_DIR}/src/metrics')
sys.path.append(f'{TASK_GRN_INFERENCE_DIR}/src/utils')
sys.path.append(f'{TASK_GRN_INFERENCE_DIR}/src/methods/portia')  # pre-inject portia deps
from all_metrics.helper import main as main_metrics
from params import get_par
from methods.portia.script import main as main_portia


def prediction_file_name(dataset, method, data_type):
    return f'{results_dir}/{dataset}.{method}.prediction_{data_type}.h5ad'


def infer_pearson_corr(par, dataset):
    from util import corr_net
    adata = ad.read_h5ad(par["rna"], backed='r')
    tf_all = np.loadtxt(par["tf_all"], dtype=str)
    perturbs = sorted(adata.obs['perturbation'].unique())[:10] if dataset == 'parsebioscience' else tf_all
    adata = adata[adata.obs['perturbation'].isin(perturbs)].to_memory()
    print(adata.shape, flush=True)
    net = corr_net(adata, tf_all, par).astype(str)
    return ad.AnnData(X=None, uns={
        "method_id": 'pearson_corr',
        "dataset_id": adata.uns['dataset_id'],
        "prediction": net[["source", "target", "weight"]]
    })


def infer_portia(par, dataset):
    result = main_portia(par)
    if result is not None and hasattr(result, 'to_csv'):
        dataset_id = ad.read_h5ad(par['rna'], backed='r').uns['dataset_id']
        return ad.AnnData(X=None, uns={
            "method_id": 'portia',
            "dataset_id": dataset_id,
            "prediction": result[["source", "target", "weight"]]
        })


METHODS = {
    'pearson_corr': infer_pearson_corr,
    'portia': infer_portia,
}

if __name__ == '__main__':
    dataset = env.get('DATASET', 'xaira_HEK293T')
    INFER_GRN = env.get('INFER_GRN', 'false').lower() == 'true'
    RUN_METRICS = env.get('METRICS', 'true').lower() == 'true'

    results_dir = f'{RESULT_DIR}/experiment/bulk_vs_sc/'
    os.makedirs(results_dir, exist_ok=True)

    par = get_par(dataset)
    par.setdefault('num_workers', 20)
    print(f'Processing dataset: {dataset}  INFER_GRN={INFER_GRN}  METRICS={RUN_METRICS}', flush=True)

    if INFER_GRN:
        for method_name, method_fn in METHODS.items():
            for data_type in ['sc', 'bulk']:
                print(f"  Inferring GRN [{method_name}] on {data_type} data...", flush=True)
                par['rna'] = (
                    f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/inference_data/{dataset}_rna.h5ad'
                    if data_type == 'bulk'
                    else f'{TASK_GRN_INFERENCE_DIR}/resources/extended_data/{dataset}_train_sc.h5ad'
                )
                par['prediction'] = prediction_file_name(dataset, method_name, data_type)
                net = method_fn(par, dataset)
                if net is not None:
                    net.write_h5ad(par['prediction'])

    if RUN_METRICS:
        rr_all_store = []
        for method_name in METHODS:
            for data_type in ['sc', 'bulk']:
                print(f"  Calculating metrics [{method_name}] on {data_type} data...", flush=True)
                par['prediction'] = prediction_file_name(dataset, method_name, data_type)
                rr = main_metrics(par)
                rr['data_type'] = data_type
                rr['method'] = method_name
                rr_all_store.append(rr)
        rr_all = pd.concat(rr_all_store, axis=0)
        rr_all['dataset'] = dataset
        rr_all.to_csv(f'{results_dir}/metrics_{dataset}.csv', index=False)
        print(f"Saved metrics: {results_dir}/metrics_{dataset}.csv", flush=True)
        