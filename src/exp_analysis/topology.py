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
warnings.filterwarnings("ignore")

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{env['RESULTS_DIR']}/figs/consensus_regulators"
os.makedirs(figs_dir, exist_ok=True)


sys.path.append(env['genernbi_supp_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores, METHODS

TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
from src.utils.config import DATASETS_METRICS, DATASETS_CELLTYPES, DATASETS


from src.exp_analysis.helper import Exp_analysis, create_interaction_info, jaccard_similarity_net

from src.utils.util import read_prediction

exp_objs_dict_dict = {}
for dataset in DATASETS:
    print(dataset, flush=True)
# dataset = 'op' #'op', nakatake, adamson
    par_top_analysis = {
            'grn_models': METHODS,
            'shortlist': ['pearson_corr', 'ppcor', 'portia', 'grnboost'],
            'peak_gene_models': [], #['celloracle', 'scenicplus', 'figr', 'granie'],
            'peak_gene_dir': f'{TASK_GRN_INFERENCE_DIR}/resources/results/{dataset}/peak_gene/',
    }
    exp_objs_dict = {}
    nets_dict = {}
    for model in par_top_analysis['grn_models']:
        grn_file_name = retrieve_grn_path(dataset, model)
        if not os.path.exists(grn_file_name):
            # print(dataset, model, ' is skipped')
            continue
        net = read_prediction(par={'prediction': grn_file_name, 'max_links': 50_000, 'verbose':0})
        nets_dict[model] = net
        if model in par_top_analysis['peak_gene_models']:
            peak_gene_net = pd.read_csv(f"{par_top_analysis['peak_gene_dir']}/{model}.csv")
        else:
            peak_gene_net = None
        # print(model, len(net))
        obj = Exp_analysis(net, peak_gene_net)
        obj.calculate_basic_stats()
        obj.calculate_centrality()

        exp_objs_dict[model] = obj
    exp_objs_dict = {surrogate_names[key]:value for key,value in exp_objs_dict.items()}
    exp_objs_dict_dict[dataset] = exp_objs_dict


for i, dataset in enumerate(DATASETS):
    print(dataset, flush=True)
    exp_objs_dict = exp_objs_dict_dict[dataset]
    links_n = {}
    source_n = {}
    target_n = {}
    nets = {}
    for name, obj in exp_objs_dict.items():
        net = obj.net
        if 'cell_type' in net.columns: # for cell specific grn models, take the mean
            n_grn = net.groupby('cell_type').size().mean()
        else:
            n_grn = len(net)

        links_n[name] = n_grn
        source_n[name] = obj.stats['n_source']
        target_n[name] = obj.stats['n_target']
    data = {
        'Model': [],
        'Count': [],
        'Type': []
    }
    for model in links_n.keys():
        data['Model'].append(model)
        data['Count'].append(links_n[model])
        data['Type'].append('Putative links')

    for model in source_n.keys():
        data['Model'].append(model)
        data['Count'].append(source_n[model])
        data['Type'].append('Putative TFs')

    for model in target_n.keys():
        data['Model'].append(model)
        data['Count'].append(target_n[model])
        data['Type'].append('Putative target genes')
    df = pd.DataFrame(data)
    df['Dataset'] = dataset
    if i == 0:
        topology_stats =df
    else:
        topology_stats = pd.concat([topology_stats, df]).reset_index(drop=True)
topology_stats['Dataset'] = topology_stats['Dataset'].map(surrogate_names)

order_names =[surrogate_names[name] for name in ['pearson_corr', 'grnboost', 'portia', 'ppcor']]

def plot_topology_short(axes):
    topology_stats_short = topology_stats[topology_stats['Model'].isin(order_names)]
    topology_stats_short = topology_stats_short[topology_stats_short['Type'].isin(['Putative TFs', 'Putative target genes'])]
    for i, type in enumerate(topology_stats_short['Type'].unique()):
        ax = axes[i]
        topology_stats_sub = topology_stats_short[topology_stats_short['Type']==type]
        sns.barplot(
            ax=ax,
            data=topology_stats_sub,
            hue='Model',
            hue_order=order_names,
            x='Dataset',
            y='Count',
            alpha=1,
            palette=palette_methods
        )
        ax.get_legend().remove()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.margins(x=.05, y=.1)
        # ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(type, pad=15)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i == 1:
            ax.set_xlabel('')
            ax.set_ylabel('')
        else:
            ax.set_xlabel('Dataset')
def plot_indegree_centrality(exp_objs_dict_dict, axes):
    print(exp_objs_dict_dict.keys())
    for i, dataset in enumerate(exp_objs_dict_dict.keys()):
        exp_objs_dict = exp_objs_dict_dict[dataset]
        if len(exp_objs_dict)==0:
            print(f"Skipping {dataset} as no data is available.")
            continue
        ax = axes[i]
        for name in ['pearson_corr', 'grnboost', 'portia', 'ppcor']:
            name = surrogate_names.get(name,name)
            if name not in exp_objs_dict:
                print(f"Skipping {name} as no data is available for {dataset}.")
                continue
            obj = exp_objs_dict[name]
            obj.calculate_centrality()
            values = obj.tf_gene.in_deg.degree.values

            obj.plot_cumulative_density(values, title='', x_label='Number of regulators', ax=ax, alpha=.8, label=name, c=palette_methods[name], linestyle=linestyle_methods[name], linewidth=2)
        if i != 0:
            ax.set_ylabel('')
            ax.set_xlabel('')
        ax.set_title(surrogate_names[dataset], pad=15)
        ax.grid(False)

fig = plt.figure(figsize=(11, 2))
gs = fig.add_gridspec(1, 5)
ax1 = fig.add_subplot(gs[0, 0]) 
ax2 = fig.add_subplot(gs[0, 1]) 
ax3 = fig.add_subplot(gs[0, 2]) 
ax4 = fig.add_subplot(gs[0, 3]) 
plot_topology_short([ax1, ax2])
plot_indegree_centrality({key:value for key, value in exp_objs_dict_dict.items() if key in ['op', 'nakatake']}, [ax3, ax4])
ax4.set_yticklabels([])
ax4.legend(frameon=False, loc=(1.1, .2), title='GRN model')

ax1.set_position([0.01, 0.1, 0.20, 0.75])  
ax2.set_position([0.27, 0.1, 0.20, 0.75])  
ax3.set_position([0.55, 0.20, 0.11, 0.55])
ax4.set_position([0.69, 0.20, 0.11, 0.55]) 
file_name = f"{RESULTS_DIR}/figs/topology_stats.png"
print(f"Saving figure to {file_name}")
plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')

additional_models = [model for model in topology_stats['Model'].unique() if model not in palette_methods.keys()]
palette_methods_all = {**{name:color for name, color in zip(additional_models, palette_celltype)}, **palette_methods}


single_modality = ['PPCOR', 'Positive Ctrl', 'Pearson Corr.', 'Portia', 'GRNBoost2', 'Scenic', 'scPRINT', 'scGPT', 'Geneformer']

g = sns.catplot(
    data=topology_stats[topology_stats['Model'].isin(single_modality)],
    kind='bar',
    hue='Model',
    # hue_order=order_names,  # Specify the desired order of hue categories
    x='Dataset',
    y='Count',
    col='Type',
    # alpha=.5,
    palette=palette_methods_all,
    sharey=False,
    height=2.5,  # Adjust plot size (smaller)
    aspect=1.5    # Adjust aspect ratio
)
g.set_axis_labels("")
for ax, col_name in zip(g.axes.flat, topology_stats['Type'].unique()):
    ax.set_title(col_name, fontsize=12, pad=20)
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
for ax in g.axes.flat:
    ax.margins(x=.05, y=.1)
g._legend.set_title("Model")
g._legend.set_bbox_to_anchor((1, 0.7))  
file_name = f"{RESULTS_DIR}/figs/topology_stats_1.png"
print(f"Saving figure to {file_name}")
plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')

g = sns.catplot(
    data=topology_stats[(~topology_stats['Model'].isin(single_modality))&(topology_stats['Dataset'].isin(['OPSCA', 'MSCIC']))],
    kind='bar',
    hue='Model',
    # hue_order=order_names,  # Specify the desired order of hue categories
    x='Dataset',
    y='Count',
    col='Type',
    # alpha=.5,
    palette=palette_methods_all,
    sharey=False,
    height=2.5,  # Adjust plot size (smaller)
    aspect=1    # Adjust aspect ratio
)
g.set_axis_labels("")
for ax, col_name in zip(g.axes.flat, topology_stats['Type'].unique()):
    ax.set_title(col_name, fontsize=12, pad=20) 
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
for ax in g.axes.flat:
    ax.margins(x=.2, y=.1)
g._legend.set_title("Model")
g._legend.set_bbox_to_anchor((.98, 0.6))  # Adjust legend position
file_name = f"{RESULTS_DIR}/figs/topology_stats_2.png"
print(f"Saving figure to {file_name}")
plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')



def plot_jaccard_similarity(dataset, ax):
    exp_objs_dict = exp_objs_dict_dict[dataset]
    exp_objs_dict = {key:value for key, value in exp_objs_dict.items() if key in ['Positive Ctrl', 'Pearson Corr.', 'PPCOR', 'Portia', 'GRNBoost2', 'Scenic+']}
    nets = {}
    for name, obj in exp_objs_dict.items():
        nets[name] = obj.net
    _ = jaccard_similarity_net(nets, ax=ax)
    ax.set_title(f'', pad=20, fontsize=12, fontweight='bold')
for i, dataset in enumerate(['op']):
    if dataset == 'op':
        figsize=(3, 3)
    else:
        figsize=(5, 4)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_jaccard_similarity(dataset, ax)
    file_name = f"{RESULTS_DIR}/figs/jaccard_similarity_{dataset}.png"
    print(f"Saving figure to {file_name}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')