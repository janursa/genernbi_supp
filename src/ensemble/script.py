"""
Gene-wise Ensemble Algorithm

This algorithm creates an ensemble GRN by:
1. Running gene-wise regression for each model at a specified quantile (theta)
2. Identifying the best performing model for each gene based on R2 scores
3. Selecting regulators from the best model for each gene
4. Assembling all selected edges into a final ensemble network

Usage:
    python script_genewise_ensemble.py --rr_folder <output_folder> --dataset <dataset_name> --theta <quantile>
    
Example:
    python script_genewise_ensemble.py --rr_folder results/ensemble --dataset op --theta 0.25
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import anndata as ad
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, RobustScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from geneRNBI.src.helper import load_env

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run gene-wise ensemble GRN analysis')
parser.add_argument('--rr_folder', type=str, required=True, help='Results folder path')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
parser.add_argument('--theta', type=float, default=0.25, help='Quantile for feature selection (default: 0.25)')
parser.add_argument('--skip_metrics', action='store_true', help='Skip metrics evaluation')
args = parser.parse_args()

rr_folder = args.rr_folder
dataset = args.dataset
theta = args.theta
skip_metrics = args.skip_metrics

# Create output directory
os.makedirs(rr_folder, exist_ok=True)

env = load_env()
sys.path.append(env['geneRNBI_DIR'])

# Define models to ensemble
grns = ['scenicplus', 'grnboost', 'pearson_corr', 'ppcor'] if dataset == 'op' else ['scenic', 'grnboost', 'pearson_corr', 'ppcor']
# grns = ['scenicplus']

TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
sys.path.append(TASK_GRN_INFERENCE_DIR)
sys.path.append(env['UTILS_DIR'])
sys.path.append(env['METRICS_DIR'])

from src.utils.util import read_prediction, naming_convention, process_links
from regression.helper import cross_validate_gene, net_to_matrix, fill_zeros_in_grn
from src.params import get_par

# Get parameters
par = get_par(dataset)
par['grn_models_dir'] = f'{env["RESULTS_DIR"]}/{dataset}/'


print(f"Running gene-wise ensemble for dataset: {dataset}")
print(f"Using theta (quantile): {theta}")
print(f"Models to ensemble: {grns}")

# Load evaluation data
perturb_data = ad.read_h5ad(par['evaluation_data'])
gene_names = perturb_data.var_names.to_numpy()
n_genes = len(gene_names)

# Load consensus regulators
with open(par['regulators_consensus'], 'r') as f:
    consensus_data = json.load(f)
n_features_per_gene = np.asarray([consensus_data[gene][str(theta)] for gene in gene_names], dtype=int)

# Load TF list
tf_names = np.loadtxt(par['tf_all'], dtype=str)
if par['apply_tf'] == False:
    tf_names = gene_names

# Prepare expression data
layer = par['layer']
X = perturb_data.layers[layer]
try:
    X = X.todense().A
except:
    pass
X = RobustScaler().fit_transform(X)

# Create random groups for cross-validation
n_cells = perturb_data.shape[0]
random_groups = np.random.choice(range(1, 6), size=n_cells, replace=True)
groups = LabelEncoder().fit_transform(random_groups)
reg_type = par['reg_type']

from joblib import Parallel, delayed

# Reusable function to evaluate genes
def evaluate_genes_parallel(grn, model_name="model"):
    """Evaluate all genes in parallel for a given GRN matrix"""
    def evaluate_gene(j):
        gene = gene_names[j]
        n_features = int(n_features_per_gene[j])
        
        if n_features == 0:
            return gene, -np.inf, 'skipped'
        try:
            result = cross_validate_gene(reg_type, X, groups, grn, j, n_features, n_jobs=1)
            r2score = result['avg-r2']
            return gene, r2score, 'evaluated'
        except Exception as e:
            return gene, -np.inf, f'failed: {str(e)}'
    
    results = Parallel(n_jobs=20, backend='loky')(
        delayed(evaluate_gene)(j) for j in range(n_genes)
    )
    
    gene_scores = {}
    genes_evaluated = 0
    genes_skipped = 0
    
    for gene, score, status in results:
        gene_scores[gene] = score
        if status == 'evaluated':
            genes_evaluated += 1
        elif status == 'skipped':
            genes_skipped += 1
    
    avg_r2 = np.mean([s for s in gene_scores.values() if s > -np.inf]) if genes_evaluated > 0 else 0.0
    print(f"  {model_name}: Evaluated {genes_evaluated} genes, skipped {genes_skipped}, avg R2 = {avg_r2:.4f}")
    
    return gene_scores, genes_evaluated, genes_skipped

if skip_metrics==False:
    if 'donor_id' not in perturb_data.obs:
        perturb_data.obs['donor_id'] = 'donor_0'
    if 'cell_type' not in perturb_data.obs:
        perturb_data.obs['cell_type'] = 'cell_type'

    print("\n" + "="*80)
    print("STEP 1: Evaluating gene-wise performance for each model")
    print("="*80)

    # Store gene-wise performance for each model
    gene_performance = {}  # {model: {gene: r2_score}}
    model_networks = {}    # {model: network_dataframe}
    
    def evaluate_model(model):
        print(f"\nEvaluating model: {model}")
        # Load network
        net = ad.read_h5ad(f"{par['grn_models_dir']}/{naming_convention(dataset, model)}").uns['prediction']
        net = process_links(net, par)
        
        # Convert to matrix
        net_matrix = net_to_matrix(net, gene_names)
        grn = fill_zeros_in_grn(net_matrix)
        
        # Remove interactions when first gene in pair is not in TF list
        mask = np.isin(gene_names, list(tf_names))
        grn[~mask, :] = 0
        
        # Evaluate genes in parallel using shared function
        gene_scores, genes_evaluated, genes_skipped = evaluate_genes_parallel(grn, model)
        
        return model, gene_scores, net
    
    # Evaluate all models
    results = [evaluate_model(model) for model in grns]
    
    for model, gene_scores, net in results:
        gene_performance[model] = gene_scores
        model_networks[model] = net

    print("\n" + "="*80)
    print("STEP 2: Selecting best model for each gene")
    print("="*80)

    # For each gene, find the model with the best R2 score
    best_model_per_gene = {}
    gene_performance_summary = []

    for gene in gene_names:
        # Get R2 scores from all models for this gene
        scores = {model: gene_performance[model].get(gene, -np.inf) for model in grns}
        
        # Find best model
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]
        
        best_model_per_gene[gene] = {
            'model': best_model,
            'r2_score': best_score,
            'all_scores': scores
        }
        
        gene_performance_summary.append({
            'gene': gene,
            'best_model': best_model,
            'best_r2': best_score,
            **{f'{model}_r2': scores[model] for model in grns}
        })

    # Save gene-wise performance summary
    gene_performance_df = pd.DataFrame(gene_performance_summary)
    gene_performance_df.to_csv(f'{rr_folder}/gene_wise_performance_theta{theta}.csv', index=False)

if True:
    # Load gene-wise performance
    gene_performance_df = pd.read_csv(f'{rr_folder}/gene_wise_performance_theta{theta}.csv')
    
    # Load consensus regulators
    with open(par['regulators_consensus'], 'r') as f:
        consensus_data = json.load(f)
    n_features_per_gene = np.asarray([consensus_data[gene][str(theta)] for gene in gene_names], dtype=int)
    
    model_counts = gene_performance_df['best_model'].value_counts()
    # print("\nGenes assigned to each model:")
    # for model, count in model_counts.items():
    #     print(f"  {model}: {count} genes ({count/n_genes*100:.1f}%)")

    print("\n" + "="*80)
    print("STEP 3: Assembling ensemble network")
    print("="*80)

    # Load all networks once
    print("Loading networks...")
    model_networks = {}
    for model in gene_performance_df['best_model'].unique():
        net = ad.read_h5ad(f"{par['grn_models_dir']}/{naming_convention(dataset, model)}").uns['prediction']
        net = process_links(net, par)
        model_networks[model] = net
    
    # Collect edges for the ensemble network
    ensemble_edges = []

    for j, gene in enumerate(gene_names):
        row = gene_performance_df[gene_performance_df['gene'] == gene].iloc[0]
        best_model = row['best_model']
        best_r2 = row['best_r2']
        n_features = int(n_features_per_gene[j])
        
        # Skip genes with no regulators
        if n_features == 0:
            continue
        
        # Get network for best model
        net = model_networks[best_model]
        
        # Get edges from the best model for this target gene
        gene_edges = net[net['target'] == gene].copy()
        
        if len(gene_edges) > 0:
            # Take only top N regulators based on consensus
            gene_edges = gene_edges.nlargest(min(n_features, len(gene_edges)), 'weight')
            
            # Add source model information
            gene_edges['source_model'] = best_model
            gene_edges['target_r2'] = best_r2
            ensemble_edges.append(gene_edges)

    # Combine all edges
    ensemble_net = pd.concat(ensemble_edges, ignore_index=True)
    print(f"\nEnsemble network statistics:")
    print(f"  Total edges: {len(ensemble_net)}")
    print(f"  Unique target genes: {ensemble_net['target'].nunique()}")
    print(f"  Unique source genes (TFs): {ensemble_net['source'].nunique()}")
    print(f"  Average regulators per gene: {len(ensemble_net) / ensemble_net['target'].nunique():.1f}")

    # Distribution of edges by source model
    print("\nEdge distribution by source model:")
    edge_counts = ensemble_net['source_model'].value_counts()
    for model, count in edge_counts.items():
        print(f"  {model}: {count} edges ({count/len(ensemble_net)*100:.1f}%)")

    # Save ensemble network with metadata
    ensemble_net.to_csv(f'{rr_folder}/ensemble_network_theta{theta}_with_metadata.csv', index=False)
    
    # Create final network in standard format
    final_network = ensemble_net[['source', 'target', 'weight']].reset_index(drop=True)
    
    # Save as h5ad file
    prediction_template = ad.read_h5ad(f"{par['grn_models_dir']}/{naming_convention(dataset, 'grnboost')}")
    prediction_template.uns['prediction'] = final_network
    prediction_template.uns['method_id'] = f'ensemble'
    output_file = f"{rr_folder}/{naming_convention(dataset, f'ensemble')}"
    prediction_template.write_h5ad(output_file)
    
    print(f"\nEnsemble network saved to: {output_file}")
    
if not skip_metrics:
    # Run metrics evaluation
    print("\n" + "="*80)
    print("STEP 5: Evaluating ensemble network (gene-wise)")
    print("="*80)

    # Load the ensemble network
    ensemble_net_data = ad.read_h5ad(output_file)
    net = ensemble_net_data.uns['prediction']
    net = process_links(net, par)
    
    # Convert to matrix
    net_matrix = net_to_matrix(net, gene_names)
    grn = fill_zeros_in_grn(net_matrix)
    
    # Remove interactions when first gene in pair is not in TF list
    tf_names = np.loadtxt(par['tf_all'], dtype=str)
    mask = np.isin(gene_names, list(tf_names))
    grn[~mask, :] = 0
    
    # Evaluate genes in parallel using shared function
    gene_scores_dict, genes_evaluated, genes_skipped = evaluate_genes_parallel(grn, "ensemble")
    
    # Convert to list format for saving
    ensemble_gene_scores = []
    for gene in gene_names:
        ensemble_gene_scores.append({
            'gene': gene,
            'r2': gene_scores_dict[gene],
            'n_features': int(n_features_per_gene[np.where(gene_names == gene)[0][0]])
        })
    
    # Save ensemble gene-wise scores
    ensemble_scores_df = pd.DataFrame(ensemble_gene_scores)
    ensemble_scores_df.to_csv(f'{rr_folder}/ensemble_gene_scores_theta{theta}.csv', index=False)
    
    avg_r2 = np.mean([s for s in ensemble_scores_df['r2'].values if s > -np.inf]) if genes_evaluated > 0 else 0.0
    print(f"  Ensemble: Evaluated {genes_evaluated} genes, skipped {genes_skipped}, avg R2 = {avg_r2:.4f}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Theta (quantile): {theta}")
    print(f"Models ensembled: {', '.join(grns)}")
    print(f"\nGene-wise model selection:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} genes ({count/n_genes*100:.1f}%)")
    print(f"\nFinal network:")
    print(f"  Total edges: {len(final_network)}")
    print(f"  Target genes: {final_network['target'].nunique()}")
    print(f"  Source TFs: {final_network['source'].nunique()}")
    print(f"  Avg regulators/gene: {len(final_network) / final_network['target'].nunique():.1f}")
    print(f"\nEnsemble performance:")
    print(f"  Average R2: {avg_r2:.4f}")
    print(f"\nOutput files:")
    print(f"  Network (h5ad): {output_file}")
    print(f"  Gene performance: {rr_folder}/gene_wise_performance_theta{theta}.csv")
    print(f"  Ensemble scores: {rr_folder}/ensemble_gene_scores_theta{theta}.csv")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

if True:
    # Visualization - INDEPENDENT SECTION
    print("\n" + "="*80)
    print("STEP 6: Generating visualizations")
    print("="*80)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    
    sys.path.append(env['geneRNBI_DIR'])
    from src.helper import palette_methods, surrogate_names

    
    # Add ensemble to palette if not present
    if 'ensemble' not in palette_methods:
        palette_methods['ensemble'] = '#e74c3c'  # Red color for ensemble
    
    # Map model names
    model_name_map = {**surrogate_names, 'ensemble': 'Ensemble'}
    
    # Load data from files
    gene_perf_df = pd.read_csv(f'{rr_folder}/gene_wise_performance_theta{theta}.csv')
    ensemble_perf_df = pd.read_csv(f'{rr_folder}/ensemble_gene_scores_theta{theta}.csv')
    
    # Get model counts from gene_perf_df - only genes with valid (non-inf) predictions
    # Filter genes where best_r2 is not -inf
    valid_genes_df = gene_perf_df[~np.isinf(gene_perf_df['best_r2'])]
    model_counts = valid_genes_df['best_model'].value_counts()
    
    # PLOT 1: Number of genes selected per method (vertical bar plot)
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    
    model_counts_sorted = model_counts.sort_values(ascending=True)  # Ascending for horizontal bars

    
    labels = [model_name_map.get(model, model) for model in model_counts_sorted.index]
    colors = [palette_methods[model] for model in labels]
    
    # Horizontal bar plot
    ax.barh(range(len(model_counts_sorted)), model_counts_sorted.values, color=colors)
    ax.set_yticks(range(len(model_counts_sorted)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Genes')
    # ax.set_title(f'Gene selection per model (theta={theta})')
    ax.spines[['right', 'top']].set_visible(False)
    ax.margins(y=0.1, x=0.1)
    
    plt.tight_layout()
    plot1_file = f'{rr_folder}/genes_per_model_theta{theta}.png'
    plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot1_file}")
    
    
    # PLOT 2: Comparative performance (strip plot with stats)
    
    # Prepare data for plotting - only genes with n_features > 0
    plot_data = []
    genes_with_regulators = ensemble_perf_df[ensemble_perf_df['n_features'] > 0]['gene'].values
    
    # Add ensemble scores
    for _, row in ensemble_perf_df[ensemble_perf_df['n_features'] > 0].iterrows():
        if not np.isinf(row['r2']):
            plot_data.append({
                'gene': row['gene'],
                'model': 'ensemble',
                'r2': row['r2']
            })
    
    # Add individual model scores for the same genes
    for model in grns:
        r2_col = f'{model}_r2'
        for gene in genes_with_regulators:
            gene_row = gene_perf_df[gene_perf_df['gene'] == gene]
            if len(gene_row) > 0:
                r2_val = gene_row[r2_col].values[0]
                if not np.isinf(r2_val):
                    plot_data.append({
                        'gene': gene,
                        'model': model,
                        'r2': r2_val
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Statistical tests: Compare ensemble vs each other model
    print("\nStatistical tests (Mann-Whitney U):")
    ensemble_scores = plot_df[plot_df['model'] == 'ensemble']['r2'].values
    
    pvals = []
    test_models = []
    for model in grns:
        model_scores = plot_df[plot_df['model'] == model]['r2'].values
        if len(ensemble_scores) > 0 and len(model_scores) > 0:
            stat, pval = mannwhitneyu(ensemble_scores, model_scores, alternative='greater')
            pvals.append(pval)
            test_models.append(model)
            print(f"  Ensemble vs {model_name_map.get(model, model)}: p = {pval:.4e}")
    
    # Adjust for multiple testing
    if len(pvals) > 0:
        _, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')
        print("\nAdjusted p-values (FDR):")
        for model, pval_adj in zip(test_models, pvals_adj):
            stars = '***' if pval_adj < 0.001 else '**' if pval_adj < 0.01 else '*' if pval_adj < 0.05 else 'ns'
            print(f"  Ensemble vs {model_name_map.get(model, model)}: p_adj = {pval_adj:.4e} ({stars})")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    
    # Map model names for plotting
    plot_df['model_display'] = plot_df['model'].map(model_name_map)
    
    # Order: ensemble first, then others by mean performance
    model_order = ['Ensemble'] + [model_name_map.get(m, m) for m in grns]
    colors_ordered = [palette_methods.get('ensemble', 'gray')] + [palette_methods.get(m, 'gray') for m in grns]
    
    # Box plot with strip plot overlay
    sns.boxplot(data=plot_df, x='model_display', y='r2', order=model_order,
                fliersize=0, ax=ax, boxprops=dict(facecolor='none', edgecolor='black'))
    
    sns.stripplot(data=plot_df, x='model_display', y='r2', order=model_order,
                  palette=dict(zip(model_order, colors_ordered)),
                  size=2, alpha=0.2, ax=ax, jitter=0.2)
    
    # Add significance stars
    if len(pvals_adj) > 0:
        ymax = plot_df['r2'].max()
        ymin = plot_df['r2'].min()
        offset = (ymax - ymin) * 0.05
        
        for i, (model, pval_adj) in enumerate(zip(test_models, pvals_adj), start=1):
            stars = '***' if pval_adj < 0.001 else '**' if pval_adj < 0.01 else '*' if pval_adj < 0.05 else ''
            if stars:
                model_display = model_name_map.get(model, model)
                x_pos = model_order.index(model_display)
                y_pos = plot_df[plot_df['model'] == model]['r2'].max() + offset
                ax.text(x_pos, y_pos, stars, ha='center', va='bottom', 
                       fontsize=12, color='red', weight='bold')
    
    ax.set_xlabel('')
    ax.margins(y=0.2, x=0.1)
    ax.set_ylabel('R² score')
    # ax.set_title(f'Gene-wise performance comparison (theta={theta})')
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.spines[['right', 'top']].set_visible(False)
    
    plt.tight_layout()
    plot2_file = f'{rr_folder}/performance_comparison_theta{theta}.png'
    plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {plot2_file}")
    plt.close()
    
    print("\nVisualization complete!")
    print("="*80)
