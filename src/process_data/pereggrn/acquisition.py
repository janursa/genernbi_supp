"""
Get the raw datasets and networks from the pereggrn package.

https://github.com/ekernf01/pereggrn/blob/main/docs/tutorial.md
"""

import pereggrn_perturbations
import pereggrn_networks

import os 
import anndata as ad 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import scanpy as sc

pereggrn_data_dir = '/home/jnourisa/projs/external/pereggrn'


meta = {
    "local_dir": './',
}
sys.path.append(meta["local_dir"])
from src.helper import TASK_GRN_INFERENCE_DIR, RESULT_DIR

sys.path.append(f'{TASK_GRN_INFERENCE_DIR}/src/utils/')
from util import process_links

def get_dataset(par):
    pereggrn_perturbations.set_data_path(f"{pereggrn_data_dir}/perturbation_data/perturbations")
    pereggrn_perturbations.load_perturbation_metadata()
    print('Load datasets ...')
    for file_name in par['datasets']:

        adata = pereggrn_perturbations.load_perturbation(file_name) 
        # pereggrn_perturbations.check_perturbation_dataset(ad = adata)
        if file_name == 'replogle2':
            file_name = 'replogle'
        adata.write(f"{par['raw_datasets_dir']}/{file_name}.h5ad")

def get_networks(par):
    print('Load networks ...')
    os.makedirs(par['grns_dir'], exist_ok=True)

    names = []
    for model in par['nets']:
        net = pd.read_parquet(f"{pereggrn_data_dir}/network_collection/networks/{model}")
        net.columns = ['source','target','weight']
        method = model.split('/')[0].split('_')[0].capitalize()
        tissue = model.split('/')[-1].split('.')[0].replace('_', ' ').capitalize()
        name = method+':'+tissue

        net = process_links(net, par)

        net.to_csv(f"{par['grns_dir']}/{name}.csv")
    
        
if __name__ == '__main__':
    par = {
        'datasets': ['norman', 'adamson', 'nakatake'],
        'nets': [
                            'ANANSE_tissue/networks/lung.parquet',
                            'ANANSE_tissue/networks/stomach.parquet', 
                            'ANANSE_tissue/networks/heart.parquet',
                            'ANANSE_tissue/networks/bone_marrow.parquet',
                            
                            'gtex_rna/networks/Whole_Blood.parquet',
                            'gtex_rna/networks/Brain_Amygdala.parquet', 
                            'gtex_rna/networks/Breast_Mammary_Tissue.parquet', 
                            'gtex_rna/networks/Lung.parquet',
                            'gtex_rna/networks/Stomach.parquet',

                            'cellnet_human_Hg1332/networks/bcell.parquet',
                            'cellnet_human_Hg1332/networks/tcell.parquet',
                            'cellnet_human_Hg1332/networks/skin.parquet',
                            'cellnet_human_Hg1332/networks/neuron.parquet',
                            'cellnet_human_Hg1332/networks/heart.parquet',
                            ],
        'raw_datasets_dir': f'{RESULT_DIR}/resources/datasets_raw/',
        'grns_dir': f'{RESULT_DIR}/resources/output/global_models/',
        'max_n_links': 50_000,
    }
    os.makedirs(f'{RESULT_DIR}/resources/output/global_models/', exist_ok=True)

    if False:
        print('Getting data ...')
        get_dataset(par)
    if True:
        print('Getting networks ...')
        get_networks(par)