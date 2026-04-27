import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import sys
import os
env = os.environ
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
RESULTS_DIR = env['RESULTS_DIR']

sys.path.append(env["METRICS_DIR"])
sys.path.append(env["UTILS_DIR"])


from util import naming_convention as naming_convention_main
from ws_distance.consensus.helper import main as main_consensus_ws_distance
from regression.consensus.helper import main as main_consensus_regression

sys.path.append(env['TASK_GRN_INFERENCE_DIR'])
from src.metrics.all_metrics.helper import main as main_metrics

from src.params import get_par

def naming_convention(dataset, model):
    return f'{dataset}.{model}.h5ad'
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate global GRNs')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., op, 300BCG)')
    args = parser.parse_args()
    
    dataset = args.dataset
    results_dir = f'{RESULTS_DIR}/experiment/global_grns/'
    os.makedirs(results_dir, exist_ok=True)
    global_grn_dir = f'{TASK_GRN_INFERENCE_DIR}/resources/grn_models/global/'
    
    global_grns = os.listdir(global_grn_dir) #all the files in the global_grn_dir
    inferred_methods = ['negative_control', 'pearson_corr', 'grnboost', 'scenicplus', 'scenic']
    

    for name in global_grns:
        file_path = f'{global_grn_dir}/{name}'
        net = pd.read_csv(file_path, index_col=0)
        
        net = ad.AnnData(X=None, uns={"method_id": name, "dataset_id": dataset, "prediction": net[["source", "target", "weight"]]})
        net.write(f'{results_dir}/{naming_convention(dataset, name)}')
    available_methods = []
    for name in inferred_methods:
        net_file = f"{TASK_GRN_INFERENCE_DIR}/resources/results/{dataset}/{naming_convention_main(dataset, name)}"
        if not os.path.exists(net_file):
            continue
        net = ad.read_h5ad(net_file)
        net.write(f'{results_dir}/{naming_convention(dataset, name)}')
        available_methods.append(name)
    all_models = available_methods + global_grns
    # - consensus 
    par = get_par(dataset)
    par['dataset'] = dataset
    
    metrics_all = []
    for model in all_models:
        # - grn evaluation
        print(f"Calculating metrics for {model}...", flush=True)
        par['prediction'] = f'{results_dir}/{naming_convention(dataset, model)}'
        metric_rr = main_metrics(par)
        metric_rr['model'] = model
        metrics_all.append(metric_rr)
    metrics_all = pd.concat(metrics_all, axis=0)
    metrics_all.to_csv(f'{results_dir}/metrics_{dataset}.csv', index=False)