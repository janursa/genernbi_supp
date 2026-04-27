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
from geneRNBI.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = f"{env['RESULTS_DIR']}/figs/datasets_stats"
os.makedirs(figs_dir, exist_ok=True)


sys.path.append(env['geneRNBI_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores

TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
from src.utils.config import DATASETS_METRICS, DATASETS_CELLTYPES, DATASETS, DATASET_INFO
DOCS_IMAGES_DIR = env['DOCS_IMAGES_DIR']

def plot_table(summary, figsize=(6,6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)
    tbl.auto_set_column_width(col=list(range(summary.shape[1])))
    n_rows = summary.shape[0] + 1  # +1 for header
    n_cols = summary.shape[1]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.4)    # thin inner lines by default
        cell.get_text().set_color("black")
        cell.get_text().set_fontsize(10)
    for c in range(n_cols):
        cell = tbl[(0, c)]
        cell.get_text().set_fontweight("bold")
        cell.set_facecolor("#f2f2f2")
        cell.set_linewidth(1.2)
    for r in range(n_rows):
        tbl[(r, 0)].set_linewidth(1.2)
        tbl[(r, n_cols - 1)].set_linewidth(1.2)
    for c in range(n_cols):
        tbl[(n_rows - 1, c)].set_linewidth(1.2)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
  
def main_datasets_stats(force=False):
    stats_cache_file = f'{RESULTS_DIR}/datasets_stats_cache.csv'
    
    # Check if cached stats exist and force is not set
    if not force and os.path.exists(stats_cache_file):
        print(f"Loading cached stats from {stats_cache_file}")
        stats_df = pd.read_csv(stats_cache_file)
    else:
        print("Calculating dataset stats...")
        stats_store = []
        cell_counts = []
        for dataset in DATASETS: 
            print(dataset)
            adata = ad.read_h5ad(f"{TASK_GRN_INFERENCE_DIR}/resources/extended_data/{dataset}_rna_all.h5ad", backed="r")
            try:
                cell_count = adata.obs['cell_count'].sum()
                
            except:
                cell_count = len(adata)
            if dataset in ['nakatake']:
                cell_count = '-'
            total_perturbations = adata.obs["perturbation"].nunique()
            if dataset not in DATASET_INFO:
                raise ValueError(f"Unknown dataset: {dataset}")
            info = DATASET_INFO[dataset]
            n_genes = adata.shape[1]
            adata = ad.read_h5ad(f"{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/inference_data/{dataset}_rna.h5ad", backed="r")
            num_samples_inference = adata.n_obs
            if "perturbation" not in adata.obs:
                adata.obs["perturbation"] = "control"
            num_unique_perturbations_inference = adata.obs["perturbation"].nunique()

            eval_suffix = "_sc" if dataset == "MSCIC" else "_bulk"
            adata = ad.read_h5ad(f"{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/evaluation_data/{dataset}{eval_suffix}.h5ad", backed="r")
            num_samples_eval = adata.n_obs
            if "perturbation" not in adata.obs:
                adata.obs["perturbation"] = "control"
            num_unique_perturbations_eval = adata.obs["perturbation"].nunique()
            stats_store.append({
            "Dataset": surrogate_names.get(dataset, dataset),
            "Unique perturbs": total_perturbations,
            "Cell type": info["cell_type"],
            "Perturb type": info["perturbation_type"],
            "Genes": n_genes,
            "Inference (samples)": num_samples_inference,
            "Inference (perturbs)": num_unique_perturbations_inference,
            "Eval. (samples)": num_samples_eval,
            "Eval. (perturbs)": num_unique_perturbations_eval,
            "Inference type": info["Inference data"],
            "Measurement time": info["Measurement time"],
            "Cell count": cell_count,
            "Modality": info["Modality"],
            "Raw/Norm.": ("✓" if info.get("has_raw_counts") else "-") + "/✓",
                # 'Condition': "Crohn's disease" if dataset=='ibd_cd' else  ('Ulcerative colitis' if dataset=='ibd_uc' else 'Healthy')
            })
        stats_df = pd.DataFrame(stats_store)
        
        # Save to cache
        stats_df.to_csv(stats_cache_file, index=False)
        print(f"Saved stats cache to {stats_cache_file}")
    
    print('Unique perturbs: ', stats_df['Unique perturbs'].sum())
    cell_count = stats_df[stats_df['Cell count'] != '-']['Cell count'].astype(int)
    print('Cell count: ', cell_count.sum())
    print('Perturb type: ', stats_df['Perturb type'].nunique())
    print('Cell type: ', stats_df['Cell type'].nunique())

      
    plot_table(stats_df, figsize=(1.3*stats_df.shape[1], .3*stats_df.shape[0]))
    file_name = f'{figs_dir}/table_datasets_summary.png'
    print('dataset summary table: ', file_name)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.01, dpi=300)

    file_name = f'{DOCS_IMAGES_DIR}/table_datasets_summary.png'
    print('dataset summary table: ', file_name)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.close()

    df_sub = stats_df[['Dataset', 'Cell type', 'Perturb type', 'Unique perturbs', 'Modality']]
    plot_table(df_sub, figsize=(1.3*df_sub.shape[1], .3*df_sub.shape[0]))
    file_name = f'{figs_dir}/table_datasets_summary_short.pdf'
    print('dataset summary table short: ', file_name)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.close()

def wrapper_plot_fc(perturb_effect_df, title=None):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.5))
    sns.stripplot(
        data=perturb_effect_df,
        x="Expression fold change", 
        y="Dataset",
        hue="Perturbation type",
        dodge=False,
        jitter=True,
        alpha=0.6,
        size=5,  # increase point size
        ax=ax,
        palette="tab10"
    )
    ax.set_ylabel("Dataset")
    ax.set_xlabel("Expression log2fc")
    if title:
        ax.set_title(title, fontsize=10)
    perturb_types = perturb_effect_df['Perturbation type'].unique()
    palette = sns.color_palette("tab10", n_colors=len(perturb_types))
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=palette[i], markersize=10,
                                label=perturb_types[i]) for i in range(len(perturb_types))]
    ax.margins(y=0.1)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1, 1), frameon=False, title="Perturbation type")
    for side in ["right", "top"]:
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    return fig, ax
def main_perturbation_effects():
    assert os.path.exists(f'{RESULTS_DIR}/perturb_effect_all.csv'), "File not found"
    perturb_effect_all = pd.read_csv(f'{RESULTS_DIR}/perturb_effect_all.csv')
    perturb_effect_all = perturb_effect_all[perturb_effect_all['Dataset'].isin(DATASETS)]
    perturb_effect_all['Dataset'] = perturb_effect_all['Dataset'].map(lambda name: surrogate_names[name])
    perturb_effect_all['Perturbation type'] = perturb_effect_all['Dataset'].map({
        'OPSCA': 'Chemical', 
        'Nakatake': 'Overexpression', 
        'Norman': 'Activation', 
        'Adamson': 'KD', 
        'Replogle': 'KD', 
        'Xaira:HCT116': 'KD', 
        'Xaira:HEK293T': 'KD', 
        'ParseBioscience': 'Cytokine',
        'IBD_UC': "Chemical and bacterial",
        'IBD_CD': "Chemical and bacterial",
        '300BCG': "Chemical"  
    })
    
    wrapper_plot_fc(perturb_effect_all)
    file_name = f'{figs_dir}/perturbation_effects_all_datasets.png'
    print('perturbation_effects_all_datasets ', file_name)
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')


    # - perturbation effect: std vs mean fc
    def wrapper_plot(perturb_effect_df):
        fig, ax = plt.subplots(1, 1, figsize=(2.5 , 2.5))
        def plot_perturbation_strength_datasets(perturb_effect_df, ax):
            sns.scatterplot(perturb_effect_df, x='STD fold change', y='Expression fold change', hue='Dataset', s=10,
                    alpha=.7, ax=ax, linewidth=.1, edgecolor='white', palette={surrogate_names[name]: color for name, color in palette_datasets.items()})
            legend = ax.legend(bbox_to_anchor=(1, 1), frameon=False)
            legend.set_title("Dataset") 
            ax.set_yscale('log')
            ax.set_xscale('log')
            for side in ['right', 'top']:
                ax.spines[side].set_visible(False)
        plot_perturbation_strength_datasets(perturb_effect_df, ax)
        # fig.savefig(f"{RESULTS_DIR}/figs/perturbation_strength_datasets.png", dpi=300, transparent=True, bbox_inches='tight')
    if False:
        wrapper_plot(perturb_effect_all[perturb_effect_all['Dataset'].isin(['OPSCA', 'ParseBioscience'])])
    # plt.close()
def main_gene_wise():
    for i, dataset in enumerate(DATASETS):   
        adata = ad.read_h5ad(f'{TASK_GRN_INFERENCE_DIR}/resources/extended_data/{dataset}_rna_all.h5ad')
        control_mask = adata.obs['is_control']
        non_control_mask = ~adata.obs['is_control']
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        mean_control = X[control_mask, :].mean(axis=0)
        mean_non_control = X[non_control_mask, :].mean(axis=0)
        plt.figure(figsize=(2.5, 2.5))
        plt.scatter(mean_control, mean_non_control, alpha=0.3, s=5)
        plt.plot([0, max(mean_control.max(), mean_non_control.max())],
                [0, max(mean_control.max(), mean_non_control.max())],
                color='red', linestyle='--', lw=1)
        plt.xlabel('Mean expression \n (control)')
        plt.ylabel('Mean expression \n (perturb)')
        plt.title(f'{surrogate_names[dataset]}')
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main_datasets_stats()
    main_perturbation_effects()
