import os
import pandas as pd
import numpy as np
import sys
import anndata as ad
import scanpy as sc
from tqdm import tqdm

import os
env = os.environ

assert 'TASK_GRN_INFERENCE_DIR' in env, "Please set the TASK_GRN_INFERENCE_DIR environment variable."
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']


meta = {
    "main_dir": f'{TASK_GRN_INFERENCE_DIR}/src/',
}
sys.path.append(meta["main_dir"])

meta = {
    "resources_dir": './',
    'util_dir': f'{TASK_GRN_INFERENCE_DIR}/src/utils/',
    'metrics_dir': f'{TASK_GRN_INFERENCE_DIR}/src/metrics/'
}
sys.path.append(meta["resources_dir"])
sys.path.append(meta["util_dir"])
sys.path.append(meta["metrics_dir"])
# Pre-inject method helper dirs so their try/except sys.path appends find the right modules
sys.path.append(f'{TASK_GRN_INFERENCE_DIR}/src/methods/portia')

from methods.pearson_corr.script import main as main_pearson_corr
from methods.portia.script import main as main_portia
from all_metrics.helper import main as main_metrics
# from metrics.ws_distance.consensus.helper import main as main_consensus_ws_distance
from process_data.helper_data import sum_by
from src.params import get_par

SCENIC_IMAGE = '/home/jnourisa/projs/images/scenic'

def main_grnboost(par):
    """Run grnboost via singularity (pyscenic grn), matching the standard pipeline."""
    import subprocess
    cmd = [
        'singularity', 'exec',
        '--bind', '/home,/vol',
        '--pwd', f'{TASK_GRN_INFERENCE_DIR}',
        SCENIC_IMAGE,
        'python', 'src/methods/grnboost/script.py',
        '--rna', par['rna'],
        '--prediction', par['prediction'],
        '--tf_all', par['tf_all'],
        '--temp_dir', par['temp_dir'],
        '--num_workers', str(par.get('num_workers', 20)),
    ]
    print(f'  Running grnboost via singularity...', flush=True)
    result = subprocess.run(cmd, check=True, text=True)
    print(result.stdout or '', flush=True)

METHODS = {
    'pearson_corr': main_pearson_corr,
    'portia': main_portia,
    'grnboost': main_grnboost,
}
GRNBOOST_DEGREES = [-1.0, 9.0, 19.0]  # single cell, middle, max granularity

# def def_par(dataset):
#     par = {
#         'evaluation_data': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/evaluation_data/{dataset}_bulk.h5ad',
#         'tf_all': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/tf_all.csv',
#         'apply_skeleton': False,
#         'apply_tf': True,
#         'max_n_links': 50000,
#         'layer': 'lognorm',
#         'apply_tf_methods': True,
#         'reg_type': 'ridge',
#         'num_workers': 10,
#         'ws_consensus': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/ws_consensus_{dataset}.csv',
#         'ws_distance_background': f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/ws_distance_background_{dataset}.csv',
#         'evaluation_data_sc': f'{TASK_GRN_INFERENCE_DIR}/resources/processed_data/{dataset}_sc.h5ad'

#     }
#     return par
def prediction_file_name(dataset, granularity, method='pearson_corr'):
    return f'{results_dir}/{dataset}.{method}.prediction_{granularity}.h5ad'

def main_pseudobulk(par):

    # - read inputs and cluster with differen resolutions
    rna = ad.read_h5ad(par['rna'])
    granularity = par['granularity']

    if granularity == -1:
        pass
    else:
        sc.pp.pca(rna, layer=par['layer'])
        sc.pp.neighbors(rna)
        sc.tl.leiden(rna, resolution=granularity, key_added=f'leiden_{granularity}')
        rna_bulk = sum_by(rna, f'leiden_{granularity}', unique_mapping=True)
        rna_bulk.layers[par['layer']] = rna_bulk.X
        for key in rna.uns.keys():
            rna_bulk.uns[key] = rna.uns[key]
        rna = rna_bulk

    rna.write(par['rna_pseudobulked'])


if __name__ == '__main__':
    import sys
    dataset = os.environ.get('DATASET', 'op')

    results_dir = os.path.join(env['RESULTS_DIR'], 'experiment/granular_pseudobulk')
    os.makedirs(results_dir, exist_ok=True)

    INFER_GRN = os.environ.get('INFER_GRN', 'false').lower() == 'true'
    PSEUDOBULK = os.environ.get('PSEUDOBULK', 'false').lower() == 'true'
    METRICS = os.environ.get('METRICS', 'true').lower() == 'true'
    SKIP_EXISTING = os.environ.get('SKIP_EXISTING', 'true').lower() == 'true'

    par = get_par(dataset)
    par['temp_dir'] = f'{results_dir}/temp_grnboost'
    par.setdefault('num_workers', 20)
    par.setdefault('seed', 32)
    degrees = [-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
    # - pseudobulk
    if PSEUDOBULK:
        print('Pseudobulking data...', flush=True)
        par['rna'] = f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/inference_data/{dataset}_rna.h5ad'
        print(ad.read_h5ad(par['rna']))
        
        for granuality in tqdm(degrees, desc='Pseudobulking'):
            par['granularity'] = granuality
            par['rna_pseudobulked'] = f'{results_dir}/{dataset}_granularity_{granuality}.h5ad'
            main_pseudobulk(par)
    # - infer grns
    if INFER_GRN:
        print('Inferring GRNs for pseudobulked data...', flush=True)
        for method_name, method_fn in METHODS.items():
            method_degrees = GRNBOOST_DEGREES if method_name == 'grnboost' else degrees
            print(f'  Method: {method_name}, granularities: {method_degrees}', flush=True)
            for granuality in tqdm(method_degrees, desc=f'Inferring GRNs ({method_name})'):
                par['rna'] = f'{results_dir}/{dataset}_granularity_{granuality}.h5ad'
                par['prediction'] = prediction_file_name(dataset, granuality, method_name)
                if SKIP_EXISTING and os.path.exists(par['prediction']):
                    print(f'    Skipping {method_name} @ {granuality} (already exists)', flush=True)
                    continue
                result = method_fn(par)
                # portia returns a DataFrame without writing to disk; write it here
                if result is not None and hasattr(result, 'to_csv'):
                    dataset_id = ad.read_h5ad(par['rna'], backed='r').uns['dataset_id']
                    output = ad.AnnData(X=None, uns={
                        "method_id": method_name,
                        "dataset_id": dataset_id,
                        "prediction": result[["source", "target", "weight"]]
                    })
                    output.write(par['prediction'])
    
    # - consensus 
    
    if False:
        par['regulators_consensus'] = f'{results_dir}/regulators_consensus_{dataset}.json'
        print('Calculating consensus for pseudobulked data...', flush=True)
        def naming_convention(dataset, model):
            return f'{dataset}.{model}.h5ad'
        from src.metrics.regression.consensus.helper import main as main_consensus_regression
        par['naming_convention'] = naming_convention
        par['dataset'] = dataset
        par['models_dir'] = results_dir
        par['models'] = [f'prediction_{i}' for i in degrees]
        
        _ = main_consensus_regression(par)
    
        # par['ws_consensus'] = f'{results_dir}/ws_consensus_{dataset}.json'
        # main_consensus_ws_distance(par)
    else:
        par['regulators_consensus'] =  f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/regulators_consensus_{dataset}.json'
    # - grn evaluation
    if METRICS:
        print('Calculating metrics for pseudobulked data...', flush=True)
        rr_all_store = []
        for method_name in METHODS:
            method_degrees = GRNBOOST_DEGREES if method_name == 'grnboost' else degrees
            for granuality in tqdm(method_degrees, desc=f'Calculating metrics ({method_name})'):
                print(f"Calculating metrics for {method_name} @ granularity {granuality} ...")
                par['prediction'] = prediction_file_name(dataset, granuality, method_name)
                metric_rr = main_metrics(par)
                metric_rr['granularity'] = granuality
                metric_rr['method'] = method_name
                rr_all_store.append(metric_rr)
        rr_all = pd.concat(rr_all_store, axis=0)
        rr_all['dataset'] = dataset
        rr_all.to_csv(f'{results_dir}/metrics_{dataset}.csv', index=False)
