import pandas as pd
import numpy as np
import anndata as ad
import tqdm
import json
import warnings
import matplotlib
import sys
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scanpy as sc 
import itertools
import warnings
import os
import warnings
from scipy import stats
from genernbi_supp.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{env['RESULTS_DIR']}/figs/consensus_regulators"
os.makedirs(figs_dir, exist_ok=True)


sys.path.append(env['genernbi_supp_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores

TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
from src.utils.config import DATASETS_METRICS, DATASETS_CELLTYPES, DATASETS

thetas=['0.25', '0.75']

def load_data(dataset, input_dir):
    """
    Load the JSON file for a given dataset and extract the data.
    """
    file_path = os.path.join(input_dir, 'resources/grn_benchmark/prior', f'regulators_consensus_{dataset}.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    gene_names = np.asarray(list(data.keys()), dtype=object)
    return data, gene_names

def process_features(data, gene_names, thetas):
    """
    Extract the number of features (regulators) for each theta value.
    """
    n_features = {
        theta: np.asarray([data[gene_name][theta] for gene_name in gene_names], dtype=int)
        for theta in thetas
    }
    return n_features

def plot_consensus_number_of_regulators(dataset, axes,thetas=['0', '0.5', '1'], color='#56B4E9'):
    """
    Create a plot for the consensus number of regulators for a dataset.
    """
    # Load and process data
    data, gene_names = load_data(dataset, TASK_GRN_INFERENCE_DIR)
    n_features = process_features(data, gene_names, thetas)
    
    
    for i, theta in enumerate(thetas):
        ax = axes[i]
        sns.histplot(
            data=n_features[theta], 
            ax=ax, 
            discrete=True, 
            color=color, 
            linewidth=.5, 
            edgecolor=None
        )
        ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='blue')
        for side in ['right', 'top']:
            ax.spines[side].set_visible(False)
        ax.set_yscale('log')
        ax.set_ylabel('Number of target genes')
        # ax.set_title(fr'$\theta$ = {theta}')
        if theta=='0.25':
            metric = "Regression (precision)"
        if theta=='0.75':
            metric = "Regression (recall)"
        ax.set_title(metric, pad=15, )
        ax.set_xlabel(r'Number of regulators')

    
    # output_path = os.path.join(output_dir, f"consensus_{dataset}.png")
    # fig.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight')

def extract_nregulators_func(datasets, TASK_GRN_INFERENCE_DIR):
    """
    Analyze and plot the consensus number of regulators for a list of datasets.
    """
    n_genes_with_regulators_dict = {}

    for dataset in datasets:
        n_genes_with_regulators_dict[dataset] = []
        # Load and process data
        data, gene_names = load_data(dataset, TASK_GRN_INFERENCE_DIR)
        n_features = process_features(data, gene_names, thetas)

        # Calculate number of genes with at least one regulator for each theta
        for theta in thetas:
            n_genes_with_regulators_dict[dataset].append((n_features[theta] != 0).sum())
    return n_genes_with_regulators_dict

def plot_n_genes_with_regulators(n_genes_with_regulators_dict, ax):
    df = pd.DataFrame(n_genes_with_regulators_dict)
    df.index = ['r_precision', 'r_recall']
    df.index = df.index.map(surrogate_names)
    df.columns = df.columns.map(surrogate_names)

    df = df.reset_index().melt(id_vars = 'index', var_name='dataset')
    df.index.name = 'Metric'
    sns.barplot(df, x='dataset', y='value', hue='index', palette=colors_blind[2:])
    # ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='blue')
    # for side in ['right', 'top']:
    #     ax.spines[side].set_visible(False)
    ax.margins(x=.1)
    ax.margins(y=.1)
    ax.set_yscale('log')
    # ax.set_xlabel(r'Number of selected regulators')
    ax.set_ylabel('Number of genes (log)')
    ax.set_xlabel(r'')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.title('Consensus genes with regulators', fontsize=12, fontweight='bold', pad=15)
    plt.legend(loc=(1.05, .5),frameon=False, title='Metric')
    plt.xticks(rotation=45, ha='right')

def main_consensus_regression_repo():
    # - format it and store in one df
    df_store = []
    for dataset in DATASETS:
        data, gene_names = load_data(dataset, TASK_GRN_INFERENCE_DIR)
        n_features = process_features(data, gene_names, thetas)
        df = pd.DataFrame(n_features).melt(var_name='theta')
        df['dataset'] = dataset
        df_store.append(df)
    regulatorys_consensus = pd.concat(df_store)
    regulatorys_consensus['dataset'] = regulatorys_consensus['dataset'].map(surrogate_names)
    regulatorys_consensus['theta'] = regulatorys_consensus['theta'].map({'0.25':"Regression (precision)", '0.75':"Regression (recall)"})
    regulatorys_consensus['theta'].unique()

    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 2, figsize=(5, 2), sharey=True)
        color=palette_datasets[dataset]
        data, gene_names = load_data(dataset, TASK_GRN_INFERENCE_DIR)
        n_features = process_features(data, gene_names, thetas)    
        for i, theta in enumerate(thetas):
            ax = axes[i]
            sns.histplot(
                alpha=.7,
                data=n_features[theta], 
                ax=ax, 
                discrete=True, 
                color=color, 
                linewidth=.01, 
                edgecolor='white',
                label=surrogate_names.get(dataset, dataset)
            )
            for side in ['right', 'top']:
                ax.spines[side].set_visible(False)
            ax.set_yscale('log')
            ax.set_ylabel('Number of target genes')
            if theta=='0.25':
                metric = "R2 (precision)"
            if theta=='0.75':
                metric = "R2 (recall)"
            ax.set_title(metric, pad=15, )
            ax.set_xlabel(r'Number of regulators')
        plt.legend(loc=(1,.2), frameon=False, title='Dataset')
        file_name = f"{figs_dir}/{dataset}.png"
        print(f"Saving figure to {file_name}")
        fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
        plt.close()
    
    # Number of genes with actual regulators 
    n_genes_with_regulators_dict = extract_nregulators_func(
        datasets=DATASETS,
        TASK_GRN_INFERENCE_DIR=TASK_GRN_INFERENCE_DIR
    )
    fig, ax = plt.subplots(1,1, figsize=(4, 2), sharey=True)
    plot_n_genes_with_regulators(n_genes_with_regulators_dict, ax)
    file_name = f"{figs_dir}/consensus_all.png"
    print(f"consensus_all: {file_name}")
    fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
def main_consensus_regression():
    # - format it and store in one df
    df_store = []
    for dataset in DATASETS:
        data, gene_names = load_data(dataset, TASK_GRN_INFERENCE_DIR)
        n_features = process_features(data, gene_names, thetas)
        df = pd.DataFrame(n_features).melt(var_name='theta')
        df['dataset'] = dataset
        df_store.append(df)
    regulatorys_consensus = pd.concat(df_store)
    regulatorys_consensus['dataset'] = regulatorys_consensus['dataset'].map(surrogate_names)
    regulatorys_consensus['theta'] = regulatorys_consensus['theta'].map({'0.25':"R2 (precision)", '0.75':"R2 (recall)"})
    
    # Plot all datasets in one figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    sns.stripplot(
        data=regulatorys_consensus, 
        x='theta', 
        y='value', 
        hue='dataset', 
        ax=ax, 
        alpha=.5, 
        edgecolor='black', 
        dodge=True, 
        jitter=0.1,  
        palette={surrogate_names[name]: color for name, color in palette_datasets.items()}
    )
    plt.yscale('log')  
    plt.xlabel('Metric')  
    plt.ylabel('Number of regulators') 
    plt.margins(y=.15) 
    ax.legend(loc=(1.1, -.1), frameon=False, title='Dataset')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    file_name = f"{figs_dir}/regression_consensus_all.png"
    print(f"regression_consensus_all: {file_name}")
    fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    
    # Number of genes with actual regulators 
    n_genes_with_regulators_dict = extract_nregulators_func(
        datasets=DATASETS,
        TASK_GRN_INFERENCE_DIR=TASK_GRN_INFERENCE_DIR
    )
    fig, ax = plt.subplots(1,1, figsize=(4, 2), sharey=True)
    plot_n_genes_with_regulators(n_genes_with_regulators_dict, ax)
    file_name = f"{figs_dir}/regression_genes_with_regulators.png"
    print(f"regression_genes_with_regulators: {file_name}")
    fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()

def main_ws():
    consensus_store = []
    for dataset in ['norman', 'replogle', 'xaira_HCT116', 'xaira_HEK293T']:
        consensus = pd.read_csv(f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/prior/ws_consensus_{dataset}.csv', index_col=0)
        consensus['dataset'] = dataset
        consensus_store.append(consensus)
    consensus = pd.concat(consensus_store)
    consensus['theta'] = consensus['theta'].map({0.25: 'WS (precision)', 0.75: 'WS (recall)'})
    consensus.groupby(['dataset'])['source'].nunique()

    consensus['dataset'] = consensus['dataset'].map(surrogate_names)
    dataset_counts = consensus.groupby(['dataset'])['source'].nunique()
    legend_labels = {name: f"{name} (TFs={count})" for name, count in dataset_counts.items()}
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    sns.stripplot(
        data=consensus, x='theta', y='value', hue='dataset', ax=ax, alpha=.7, 
        edgecolor='black', dodge=True, jitter=0.25,  
        palette={surrogate_names[name]: color for name, color in palette_datasets.items()}
    )
    plt.yscale('log')  
    plt.xlabel('Metric')  
    plt.ylabel('Number of edges') 
    plt.margins(y=.15) 
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [legend_labels[label] for label in labels]
    ax.legend(handles, new_labels, loc=(1.1, .2), frameon=False, title='Dataset')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    file_name = f"{figs_dir}/ws_consensus_all.png"
    print(f"ws_consensus_all: {file_name}")
    fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    main_consensus_regression()
    main_ws()

