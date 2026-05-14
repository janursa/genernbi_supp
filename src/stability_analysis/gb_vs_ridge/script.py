"""
GBM vs Ridge experiment: runs regression metric with a configurable reg_type
for all available GRN model prediction files in a dataset.

Usage:
    python src/stability_analysis/gb_vs_ridge/script.py --dataset op --reg_type ridge
"""
import argparse
import os
import sys
import pandas as pd

env = os.environ

sys.path.insert(0, env['UTILS_DIR'])
sys.path.insert(0, env['METRICS_DIR'])

from util import naming_convention
from regression.helper import main as main_reg

from src.params import get_par as get_base_par

RESULTS_DIR = env['RESULTS_DIR']


def get_par(dataset, reg_type):
    par = get_base_par(dataset)
    par['reg_type'] = reg_type
    return par


def main_metrics(par):
    _, output = main_reg(par)
    return output


SELECTED_METHODS = ['pearson_corr', 'grnboost', 'ppcor', 'portia', 'scenic']
MULTIOMICS_EXTRA_METHODS = ['scenicplus', 'celloracle']
MULTIOMICS_DATASETS = ['op', 'ibd_uc', 'ibd_cd', 'MSCIC']


def run(dataset, reg_type):
    rr_dir = f"{RESULTS_DIR}/experiment/gb_vs_ridge/{dataset}"
    os.makedirs(rr_dir, exist_ok=True)

    par = get_par(dataset, reg_type)

    grn_models_dir = f"{RESULTS_DIR}/{dataset}/"

    methods = SELECTED_METHODS + MULTIOMICS_EXTRA_METHODS if dataset in MULTIOMICS_DATASETS else SELECTED_METHODS

    scores_store = []
    for model in methods:
        prediction_file = f"{grn_models_dir}/{naming_convention(dataset, model)}"
        if not os.path.exists(prediction_file):
            continue
        print(f"[{reg_type}] Evaluating model: {model} ...", flush=True)
        par['prediction'] = prediction_file
        try:
            score = main_metrics(par)
            score['model'] = model
            score['dataset'] = dataset
            scores_store.append(score)
        except Exception as e:
            print(f"  Warning: failed for {model}: {e}", flush=True)

    if scores_store:
        scores_df = pd.concat(scores_store)
        out_path = f"{rr_dir}/scores_{reg_type}.csv"
        scores_df.to_csv(out_path)
        print(f"Saved results to {out_path}", flush=True)
    else:
        print("No results to save.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='op', help='Dataset name')
    parser.add_argument('--reg_type', type=str, default='GB', choices=['ridge', 'GB'],
                        help='Regression type: ridge (fast, for testing) or GB (GBM)')
    args = parser.parse_args()
    run(args.dataset, args.reg_type)
