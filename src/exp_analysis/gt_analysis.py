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
from pathlib import Path
from genernbi_supp.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = F"{env['RESULTS_DIR']}/figs/gt_analysis"
os.makedirs(figs_dir, exist_ok=True)

sys.path.append(env['genernbi_supp_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
# Define cell types and their corresponding ground truth files
cell_types_info = {
    'PBMC': {'gt_files': ['PBMC_remap.csv', 'PBMC_chipatlas.csv', 'PBMC_unibind.csv']},
    'K562': {'gt_files': ['K562_remap.csv', 'K562_chipatlas.csv', 'K562_unibind.csv']},
    'HEK293': {'gt_files': ['HEK293T_remap.csv', 'HEK293T_chipatlas.csv', 'HEK293T_unibind.csv']},
    'HCT116': {'gt_files': ['HCT116_remap.csv', 'HCT116_chipatlas.csv', 'HCT116_unibind.csv']}
}

ground_truth_types = ['remap2022', 'chipatlas', 'unibind']
gt_base_path = Path(f'{TASK_GRN_INFERENCE_DIR}/resources/grn_benchmark/ground_truth/')

# Collect statistics for each cell type and ground truth type
stats_data = []

for cell_type, info in cell_types_info.items():
    print(f"Processing cell type: {cell_type}")
    
    for i, gt_type in enumerate(ground_truth_types):
        gt_file = info['gt_files'][i]
        gt_path = f"{gt_base_path}/{gt_file}"
        
        if os.path.exists(gt_path):
            try:
                # Read ground truth file
                df = pd.read_csv(gt_path)
                
                # Count unique TFs
                n_tfs = df['source'].nunique()
                n_edges = len(df)
                n_targets = df['target'].nunique()
                
                stats_data.append({
                    'cell_type': cell_type,
                    'ground_truth': gt_type,
                    'n_tfs': n_tfs,
                    'n_edges': n_edges,
                    'n_targets': n_targets,
                    'avg_targets_per_tf': n_edges / n_tfs if n_tfs > 0 else 0
                })
                
                print(f"  {gt_type}: {n_tfs} TFs, {n_edges} edges, {n_targets} targets")
                
            except Exception as e:
                print(f"  Error reading {gt_file}: {e}")
                stats_data.append({
                    'cell_type': cell_type,
                    'ground_truth': gt_type,
                    'n_tfs': 0,
                    'n_edges': 0,
                    'n_targets': 0,
                    'avg_targets_per_tf': 0
                })
        else:
            print(f"  File not found: {gt_file}")
            stats_data.append({
                'cell_type': cell_type,
                'ground_truth': gt_type,
                'n_tfs': 0,
                'n_edges': 0,
                'n_targets': 0,
                'avg_targets_per_tf': 0
            })

# Convert to DataFrame
stats_df = pd.DataFrame(stats_data)
print(f"\nCollected stats for {len(stats_df)} cell type-GT combinations")
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(stats_df.pivot_table(index='cell_type', columns='ground_truth', values='n_tfs', fill_value=0))
stats_df

gt_pretty_names = {
    'remap2022': 'ReMap',
    'chipatlas': 'ChIP-Atlas', 
    'unibind': 'UniBind'
}

# Colors for ground truth types
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
gt_colors = dict(zip(['remap2022', 'chipatlas', 'unibind'], colors))

# Get cell types in order
cell_types = list(cell_types_info.keys())
x = np.arange(len(cell_types))
width = 0.25

# Create TF counts plot
fig, ax = plt.subplots(1, 1, figsize=(5, 3))

for i, gt_type in enumerate(['remap2022', 'chipatlas', 'unibind']):
    gt_data = stats_df[stats_df['ground_truth'] == gt_type]
    values = []
    for cell_type in cell_types:
        cell_data = gt_data[gt_data['cell_type'] == cell_type]
        if len(cell_data) > 0:
            values.append(cell_data['n_tfs'].iloc[0])
        else:
            values.append(0)
    
    bars = ax.bar(x + i * width, values, width, 
                  label=gt_pretty_names[gt_type], 
                  color=gt_colors[gt_type], alpha=0.8)
    

ax.set_xlabel('Cell Types')
ax.set_ylabel('Number of Transcription Factors')
ax.set_xticks(x + width)
ax.set_xticklabels(cell_types)
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
plt.tight_layout()

fig_name = f"{figs_dir}/tf_counts_by_celltype.png"
print(f"Saving figure to: {fig_name}")
plt.savefig(fig_name, dpi=200, transparent=True, bbox_inches='tight')
plt.close()

# Create Edge counts plot
fig, ax = plt.subplots(1, 1, figsize=(5, 3))

for i, gt_type in enumerate(['remap2022', 'chipatlas', 'unibind']):
    gt_data = stats_df[stats_df['ground_truth'] == gt_type]
    values = []
    for cell_type in cell_types:
        cell_data = gt_data[gt_data['cell_type'] == cell_type]
        if len(cell_data) > 0:
            values.append(cell_data['n_edges'].iloc[0])
        else:
            values.append(0)
    
    bars = ax.bar(x + i * width, values, width, 
                  label=gt_pretty_names[gt_type], 
                  color=gt_colors[gt_type], alpha=0.8)

ax.set_xlabel('Cell Types')
ax.set_ylabel('Number of Regulatory Edges')
ax.set_xticks(x + width)
ax.set_xticklabels(cell_types_info)
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

plt.tight_layout()

fig_name = f"{figs_dir}/edge_counts_by_celltype.png"
print(f"Saving figure to: {fig_name}")
plt.savefig(fig_name, dpi=200, transparent=True, bbox_inches='tight')
plt.close()


# 2. Calculate and plot Jaccard similarity between ground truth datasets
def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Calculate overlaps for each cell type
for cell_type in cell_types:
    print(f"\n{'='*60}")
    print(f"OVERLAP ANALYSIS FOR {cell_type}")
    print(f"{'='*60}")
    
    # Load the ground truth data for this cell type
    gt_data = {}
    gt_tfs = {}
    gt_edges = {}
    
    cell_info = cell_types_info[cell_type]
    
    for i, gt_type in enumerate(['remap2022', 'chipatlas', 'unibind']):
        gt_file = cell_info['gt_files'][i]
        gt_path = gt_base_path / gt_file
        
        if gt_path.exists():
            df = pd.read_csv(gt_path)
            gt_data[gt_type] = df
            gt_tfs[gt_type] = set(df['source'].unique())
            gt_edges[gt_type] = set(df.apply(lambda row: f"{row['source']}_{row['target']}", axis=1))
        else:
            gt_data[gt_type] = pd.DataFrame()
            gt_tfs[gt_type] = set()
            gt_edges[gt_type] = set()
    
    # Calculate Jaccard similarities for TFs and Edges
    gt_names = ['remap2022', 'chipatlas', 'unibind']
    pretty_names = [gt_pretty_names[gt] for gt in gt_names]
    
    # TF overlap matrix
    tf_jaccard = np.zeros((3, 3))
    for i, gt1 in enumerate(gt_names):
        for j, gt2 in enumerate(gt_names):
            tf_jaccard[i, j] = jaccard_similarity(gt_tfs[gt1], gt_tfs[gt2])
    
    # Edge overlap matrix  
    edge_jaccard = np.zeros((3, 3))
    for i, gt1 in enumerate(gt_names):
        for j, gt2 in enumerate(gt_names):
            edge_jaccard[i, j] = jaccard_similarity(gt_edges[gt1], gt_edges[gt2])
    
    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # TF overlap heatmap
    sns.heatmap(tf_jaccard, annot=True, fmt='.3f', 
                xticklabels=pretty_names, yticklabels=pretty_names,
                cmap='Blues', vmin=0, vmax=1, ax=ax1,
                cbar_kws={'label': 'Jaccard Similarity'})
    ax1.set_title(f'{cell_type}: TF Overlap', fontsize=12, fontweight='bold')
    # ax1.set_xlabel('Ground Truth Database', fontweight='bold')
    # ax1.set_ylabel('Ground Truth Database', fontweight='bold')
    
    # Edge overlap heatmap
    sns.heatmap(edge_jaccard, annot=True, fmt='.3f',
                xticklabels=pretty_names, yticklabels=pretty_names, 
                cmap='Reds', vmin=0, vmax=1, ax=ax2,
                cbar_kws={'label': 'Jaccard Similarity'})
    ax2.set_title(f'{cell_type}: Edge Overlap', fontsize=12, fontweight='bold')
    # ax2.set_xlabel('Ground Truth Database', fontweight='bold')
    ax2.set_ylabel('', fontweight='bold')
    
    plt.tight_layout()
    
    fig_name = f"{figs_dir}/overlap_{cell_type}.png"
    print(f"Saving figure to: {fig_name}")
    plt.savefig(fig_name, dpi=200, transparent=True, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nTF Overlap Summary:")
    for i, gt1 in enumerate(gt_names):
        for j, gt2 in enumerate(gt_names):
            if i < j:  # Only print upper triangle (avoid duplicates)
                print(f"  {gt_pretty_names[gt1]} ∩ {gt_pretty_names[gt2]}: {tf_jaccard[i,j]:.3f}")
    
    print(f"\nEdge Overlap Summary:")
    for i, gt1 in enumerate(gt_names):
        for j, gt2 in enumerate(gt_names):
            if i < j:  # Only print upper triangle (avoid duplicates)
                print(f"  {gt_pretty_names[gt1]} ∩ {gt_pretty_names[gt2]}: {edge_jaccard[i,j]:.3f}")
    
    # Print unique counts
    print(f"\nUnique TF counts:")
    for gt in gt_names:
        print(f"  {gt_pretty_names[gt]}: {len(gt_tfs[gt])} TFs")
    
    print(f"\nUnique edge counts:")
    for gt in gt_names:
        print(f"  {gt_pretty_names[gt]}: {len(gt_edges[gt])} edges")
