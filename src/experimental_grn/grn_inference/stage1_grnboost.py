"""
Stage 1 GRNBoost2 inference — runs pyscenic grn via the grnboost singularity image.

Keeps up to 1,000,000 edges (no 50k default subsetting).
Output naming mirrors other Stage 1 predictions:
  {dataset}.S1-grnboost.S2-P0.S3-C0.h5ad  →  temp/experimental_grn/predictions/

Usage (called from submit_grnboost.py via sbatch singularity exec):
    python stage1_grnboost.py --dataset op
"""

import argparse
import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO    = '/home/jnourisa/projs/ongoing/task_grn_inference'
PRED_DIR = f'{REPO}/temp/experimental_grn/predictions'
TF_ALL   = f'{REPO}/resources/grn_benchmark/prior/tf_all.csv'
INFER    = f'{REPO}/resources/grn_benchmark/inference_data'

sys.path.insert(0, f'{REPO}/src/utils')
sys.path.insert(0, f'{REPO}/src/methods/grnboost')
from util import process_links
from helper import format_data, run_grn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      required=True)
parser.add_argument('--num_workers',  type=int, default=20)
parser.add_argument('--seed',         type=str, default='32')
parser.add_argument('--max_n_links',  type=int, default=1_000_000)
args = parser.parse_args()

ds   = args.dataset
pred = f'{PRED_DIR}/{ds}.S1-grnboost.S2-P0.S3-C0.h5ad'

if os.path.exists(pred):
    print(f'Already exists: {pred}  — skipping', flush=True)
    sys.exit(0)

os.makedirs(PRED_DIR, exist_ok=True)
temp_dir = f'{REPO}/temp/experimental_grn/grnboost_tmp/{ds}'
os.makedirs(temp_dir, exist_ok=True)

par = {
    'rna':                  f'{INFER}/{ds}_rna.h5ad',
    'tf_all':               TF_ALL,
    'prediction':           pred,
    'temp_dir':             temp_dir,
    'expr_mat_adjacencies': f'{temp_dir}/expr_mat_adjacencies.tsv',
    'expression_data':      f'{temp_dir}/expression_data.tsv',
    'num_workers':          args.num_workers,
    'seed':                 args.seed,
    'max_n_links':          args.max_n_links,
}

print(f'Running GRNBoost2 for {ds}', flush=True)
print(f'  RNA: {par["rna"]}', flush=True)
print(f'  max_n_links: {par["max_n_links"]}', flush=True)

dataset_id = ad.read_h5ad(par['rna'], backed='r').uns.get('dataset_id', ds)

format_data(par)
run_grn(par)

network = pd.read_csv(par['expr_mat_adjacencies'], sep='\t')
network.rename(columns={'TF': 'source', 'importance': 'weight'}, inplace=True)
network.reset_index(drop=True, inplace=True)
network = process_links(network, par)

print(f'  Edges after process_links: {len(network)}', flush=True)

network['weight'] = network['weight'].astype(str)
output = ad.AnnData(
    X=None,
    uns={
        'method_id':  'grnboost',
        'dataset_id': dataset_id,
        'prediction': network[['source', 'target', 'weight']],
    }
)
output.write(pred)
print(f'Saved → {pred}', flush=True)
