import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import matplotlib
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import r2_score, make_scorer, accuracy_score
import os
import sys
import anndata as ad
import scanpy as sc
import subprocess
import seaborn as sns
import io
import itertools
import os

from task_grn_inference.src.utils.config import DATASETS, METHODS, METRICS, surrogate_names



colors_blind = [
          '#E69F00',  # Orange
          '#56B4E9',  # Sky Blue
          '#009E73',  # Bluish Green
          '#F0E442',  # Yellow
          '#0072B2',  # Blue
          '#D55E00',  # Vermillion
          '#CC79A7']  # Reddish Purple

LINESTYLES = {
    'Positive Control': '-',
    'Negative Control': '-',
    'CollectRI': '--',
    'FigR': '-.',
    'CellOracle': ':',
    'GRaNIE': ':',
    'ANANSE': '--',
    'scGLUE': '-.',
    'Scenic+': '-',
    'GRNBoost2': '--',
    'GENIE3': '-.',
    'Pearson corr.': '-' 
}

MARKERS = {
    'Random': '.',
    'CollectRI': '.',
    'FigR': 'o',
    'CellOracle': 'd',
    'GRaNIE': 'v',
    'ANANSE': 's',
    'scGLUE': 'x',
    'Scenic+': '*',
    'Positive Control': '.',
    'Negative Control': '.',
    'GRNBoost2': 'x',
    'GENIE3': 'v',
    'Pearson corr.': '-' 
}

surrogate_names = {**surrogate_names, **{
    'bulk': 'Bulk',
    'sc': 'Single cell',
    'de': 'Differential expression',

    'prediction': 'Inferred GRN',
    'evaluation_data': 'Evaluation data',
    'tf_all': 'Known TFs list',
    'regulators_consensus': 'Consensus regulators (Regression)',
    'ws_consensus': 'Consensus edges (WS distance)',
    'ws_distance_background': 'WS distance background scores',
    'evaluation_data_de': 'Differential expression data',
    'ground_truth': 'Ground truth',
    'Gtex:Whole blood': 'Gtex:Blood',
    'Gtex:Brain amygdala': 'Gtex:Brain',
    'Gtex:Breast mammary tissue': 'Gtex:Breast',
    
    }}

NEGATIVE_CONTROL = 'Dimethyl Sulfoxide'
CONTROLS3 = ['Dabrafenib', 'Belinostat', 'Dimethyl Sulfoxide']
SELECTED_MODELS = [surrogate_names[name] for name in ['ppcor', 'pearson_corr',  'portia', 'grnboost',  'granie', 'scenicplus', 'scenic', 'negative_control', 'positive_control', 'celloracle']]


if False:
    collectRI = pd.read_csv("https://github.com/pablormier/omnipath-static/raw/main/op/collectri-26.09.2023.zip")
    collectRI.to_csv(f'{work_dir}/collectri.csv')


# - color palettes 

palette_datasets = {key: color for key, color in zip(DATASETS, sns.color_palette("deep", len(DATASETS)))}
palette_methods = {key:color for key, color in zip(SELECTED_MODELS, sns.color_palette('Set2', len(SELECTED_MODELS)))}
# Stable metric display-name → color dict (colors_blind for ≤7; tab20 pairs beyond that)
_extra_metric_colors = sns.color_palette('tab20', 20)[14:]  # distinct colors not in colors_blind
_metric_color_list = colors_blind + list(_extra_metric_colors)
palette_metrics = {surrogate_names.get(m, m): _metric_color_list[i] for i, m in enumerate(METRICS)}
# Override TF binding and Gene sets recovery with more distinct colors (avoid confusion with WS greens/yellows)
palette_metrics['TF binding'] = '#882255'          # dark magenta
palette_metrics['Gene sets recovery'] = '#44AA99'  # teal

palette_celltype = ['#c4d9b3', '#c5bc8e', '#c49e81', '#c17d88', 'gray', 'lightsteelblue']

# - line styles
linestyle_methods = {key: linestyle for key, linestyle in zip(SELECTED_MODELS, itertools.cycle(['-', '--', '-.', ':']))}

def retrieve_grn_path(dataset, model):
    env = load_env()
    TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
    return f'{TASK_GRN_INFERENCE_DIR}/resources/results/{dataset}/{dataset}.{model}.{model}.prediction.h5ad'

def determine_fold_change_effect(adata, pseudocount=1e-6):
    """
    Compute fold change and standardized effect of perturbations per cell type.

    Parameters
    ----------
    adata : AnnData
        Single-cell expression data
    pseudocount : float
        Small value to avoid division by zero

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: perturbation, Expression fold change, STD fold change, cell_type
    """
    results = []

    obs_all = adata.obs.reset_index(drop=True)
    
    # Use normalized layer if available
    if 'X_norm' in adata.layers:
        X_norm = adata.layers['X_norm']
    else:
        X_norm = adata.X
    X_norm = X_norm.toarray() if hasattr(X_norm, 'toarray') else X_norm
    # Check for negative values or NaNs
    if np.any(X_norm < 0):
        raise ValueError("X_norm contains negative values")
    if np.any(np.isnan(X_norm)):
        raise ValueError("X_norm contains NaN values")
    if 'cell_type' not in obs_all:
        obs_all['cell_type'] = 'celltype'

    if 'cell_type' not in obs_all:
        obs_all['cell_type'] = 'celltype'
        
    for cell_type in obs_all.cell_type.unique():
        mask_celltype = obs_all.cell_type == cell_type
        obs = obs_all.loc[mask_celltype, :]
        X = X_norm[mask_celltype, :]

        # Identify control cells
        control_mask = obs['is_control']
        control_matrix = X[control_mask, :]

        # Mean and std per gene
        control_mean_expression = np.mean(control_matrix, axis=0)
        control_std_expression = np.std(control_matrix, axis=1)

        mask = control_mean_expression != 0

        # Skip if control is empty
        if control_matrix.shape[0] == 0:
            print(f'No control cells for cell type {cell_type}, skipping.')
            continue
        
        for perturbation in obs['perturbation'].unique():
            sample_mask = obs['perturbation'] == perturbation
            sample_matrix = X[sample_mask, :]
            mean_expression = np.mean(sample_matrix, axis=0)
            std_expression = np.std(sample_matrix, axis=1)
            fold_change = (mean_expression[mask] + pseudocount) / (control_mean_expression[mask] + pseudocount)
            fold_change_log2 = np.log2(fold_change)
            std_change = np.log2(np.mean(std_expression)+pseudocount) - np.log2(np.mean(control_std_expression)+pseudocount)
            results.append({
                'perturbation': perturbation,
                'Expression fold change': np.median(fold_change_log2),
                'STD fold change': std_change,
                'cell_type': cell_type
            })

    results = pd.DataFrame(results)
    return results

def plot_heatmap(scores, ax=None, name='', fmt='0.02f', cmap="viridis"):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharey=True)

    # Ensure numeric values first
    scores = scores.apply(pd.to_numeric, errors='coerce')
    
    # Normalize column-wise with guards for edge cases
    def safe_normalize(x):
        x_min = np.nanmin(x)
        x_max = np.nanmax(x)
        # If all values are the same or all NaN, return 0.5 (middle of colormap)
        if np.isnan(x_min) or np.isnan(x_max) or x_max == x_min:
            return pd.Series([0.5] * len(x), index=x.index)
        # Otherwise normalize
        return (x - x_min) / (x_max - x_min)
    
    scores_normalized = scores.apply(safe_normalize, axis=0)
    scores_normalized = scores_normalized.round(2)
    
    vmin = 0
    vmax = 1
    seaborn.heatmap(scores_normalized, ax=ax, square=False, cbar=False, annot=True, fmt=fmt, vmin=vmin, vmax=vmax, cmap=cmap)
    
    # Replace annotations with original values
    for text, (i, j) in zip(ax.texts, np.ndindex(scores.shape)):
        value = scores.iloc[i, j]
        if pd.isna(value):
            text.set_text('NaN')
        elif isinstance(value, (np.int64, int)):  # Check if the value is an integer for 'Rank'
            text.set_text(f'{value:d}')
        else:
            text.set_text(f'{value:.2f}')
    
    ax.tick_params(left=False, bottom=False)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=0)
    ax.set_title(name, pad=10)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='left')
def custom_jointplot(data, x, y, hue, ax, scatter_kws=None, 
    kde_kws={"fill": True, "common_norm": False, "alpha": 0.4}, alpha=0.5, top_plot=True):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    scatter_kws = scatter_kws or {}
    kde_kws = kde_kws or {"fill": False, "common_norm": False}
    if 'palette' in scatter_kws:
        kde_kws['palette'] = scatter_kws['palette']

    # Create axes for marginal plots using Divider
    divider = make_axes_locatable(ax)
    top_ax = divider.append_axes("top", size="20%", pad=0.1)
    side_ax = divider.append_axes("right", size="20%", pad=0.1)

    # Scatter plot on the central axis
    sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, alpha=alpha, **scatter_kws)

    # KDE marginal distributions
    if top_plot:
        sns.kdeplot(data=data, x=x, hue=hue, ax=top_ax, **kde_kws)
    sns.kdeplot(data=data, y=y, hue=hue, ax=side_ax, **kde_kws)

    # Styling for marginal axes (top_ax)
    top_ax.get_yaxis().set_visible(False)
    top_ax.set_xticks([])
    top_ax.set_yticks([])
    top_ax.set_ylabel('')
    top_ax.set_xlabel('')
    for spine in top_ax.spines.values():
        spine.set_visible(False)
    if top_ax.get_legend() is not None:
        top_ax.get_legend().remove()

    # Styling for marginal axes (side_ax)
    side_ax.get_yaxis().set_visible(False)
    side_ax.set_xticks([])
    side_ax.set_yticks([])
    side_ax.set_ylabel('')
    side_ax.set_xlabel('')
    for spine in side_ax.spines.values():
        spine.set_visible(False)
    if side_ax.get_legend() is not None:
        side_ax.get_legend().remove()
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)

    # Ensure that ax is not modified (e.g., keeping its ticks and labels)
    ax.set_aspect("auto")

def plot_cumulative_density(data, title='', ax=None, s=1, **kwdgs):
    # Step 1: Sort the data
    sorted_data = np.sort(data)
    
    # Step 2: Compute the cumulative density values
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Step 3: Plot the data
    if ax is None:
    	fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
    	fig = None
    ax.step(sorted_data, cdf, where='post', label=title, **kwdgs)
    ax.set_xlabel('Data')
    ax.set_ylabel('Cumulative Density')
    ax.set_title(title)
    # ax.grid(True)
    return fig, ax
def plot_bar(data_dict: dict[str, np.array], title: str=''):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(data_dict.keys(), data_dict.values())

    ax.set_xlabel('Methods')
    ax.set_ylabel('Values')
    ax.grid(True, axis='y')  # Add grid only on the y-axis for better readability
    ax.set_title(title)
    aa = plt.xticks(rotation=45)


def plot_umap(adata, color='', palette=None, ax=None, X_label='X_umap', on_data=False,
              bbox_to_anchor=None, legend=True, legend_title='', **kwrds):
    latent = adata.obsm[X_label]
    var_unique_sorted = sorted(adata.obs[color].unique())
    legend_handles = []
    
    for i_group, group in enumerate(var_unique_sorted):
        mask = adata.obs[color] == group
        sub_data = latent[mask]
        
        if palette is None:
            c = None 
        else:
            c = palette[i_group]
        
        scatter = ax.scatter(sub_data[:, 0], sub_data[:, 1], label=group, **kwrds, c=c)
        legend_handles.append(plt.Line2D([0], [0], linestyle='none', marker='o', markersize=8, color=scatter.get_facecolor()[0]))
        
        if on_data:
            mean_x = np.mean(sub_data[:, 0])
            mean_y = np.mean(sub_data[:, 1])
            ax.text(mean_x, mean_y, group, fontsize=9, ha='center', va='top', color='black', weight='bold')

    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.margins(0.4)
    if legend and not on_data:
        legend  =ax.legend(handles=legend_handles, labels=var_unique_sorted, loc='upper left', 
                  bbox_to_anchor=bbox_to_anchor, frameon=False, title=legend_title, title_fontproperties={'weight': 'bold',  'size':9})
        legend.get_title().set_ha('left')

        # Increase the distance between the title and legend entries
        legend._legend_box.align = "left"  # Align the entries to the left
        # legend._legend_box.set_spacing(2)  # Increase the spacing between the title and entries


def plot_scatter(obs, obs_index, xs, ys, x_label='', y_label='', log=True, log_y=False, figsize=(5, 7)):
    """
        Scatter plot to showcase the distribution of given variables across different groups. 
    """
    n_axes = len(obs_index)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    alpha = .6
    size = 4
    for i_index, index in enumerate(obs_index):
        # i = i_index // n_axes
        j = i_index % n_axes
        ax = axes[j]

        index_vars = obs[index]
        
        if (index=='sm_name'):
            # included_vars = train_sm_names
            # included_vars = index_vars.unique()
            mask = (index_vars.isin(controls2))
            ax.scatter(xs[~mask], ys[~mask], label='Rest', alpha=alpha, color='blue', s=size-1)
            mask = (index_vars.isin(controls2))
            ax.scatter(xs[mask], ys[mask], label='Positive control', alpha=alpha, color='cyan', s=size)
            
        else:
            included_vars = index_vars.unique()
            for i, var in enumerate(included_vars):
                label = var
                mask = (index_vars == var)
                ax.scatter(xs[mask], ys[mask], label=var, alpha=alpha, color=palette_celltype[i], s=size)

        ax.grid(alpha=0.4, linewidth=1, color='grey', linestyle='--')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if log:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.margins(0.05)
        ax.spines[['right', 'top']].set_visible(False)
        # ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
        prop = {'size': 9}
        
        handles = []
        for kk, label in enumerate(included_vars if (index != 'sm_name') else ['Rest', 'Positive control']):
            if index == 'sm_name':
                handles.append(Patch(facecolor=colors_positive_controls[kk], label=label))
            else:
                handles.append(Patch(facecolor=palette_celltype[kk], label=label))
        
        ax.legend(handles=handles, prop=prop, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, frameon=False)
    plt.tight_layout()
    return fig, axes
    
def plot_stratified_scatter(obs, ax, xs, ys, palette, size=4,  x_label='', y_label='', log_x=False, log_y=False, extra_labels=None, bbox_to_anchor=(1,1)):
    """
        Scatter plot to showcase the distribution of a given variable across different groups. 
    """
    alpha = .6
    included_vars = np.unique(obs)
    # included_vars = obs
    for i, sub_var in enumerate(included_vars):        
        mask = (obs == sub_var)
        ax.scatter(xs[mask], ys[mask], label=sub_var, alpha=alpha, color=palette[i], s=size)

    ax.grid(alpha=0.4, linewidth=.5, color='grey', linestyle='--')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.margins(0.05)
    ax.spines[['right', 'top']].set_visible(False)
    # ax.grid(alpha=0.4, linestyle='--', linewidth=0.5, color='grey')
    # prop = {'size': 9}
    prop = {}
    
    handles = []
    for kk, label in enumerate(included_vars):
        if extra_labels is not None:
            extra = extra_labels[label]
            label = f'{label} ({extra})'
        else:
            label = label 
        handles.append(Patch(facecolor=palette[kk], label=label))
    ax.legend(handles=handles, prop=prop, bbox_to_anchor=bbox_to_anchor, loc='upper left', borderaxespad=0, frameon=False)


    for spine in ax.spines.values():
        spine.set_linewidth(0.5)  # Adjust the border thickness here

    # Customize grid lines
    ax.tick_params(axis='both', which='both', width=0.5, length=2)  # Adjust the tick thickness here


def plot_stacked_bar_chart(cell_types_in_drops, title='', xticks=None, 
                           xticklabels=None, colors=None, figsize=(25, 4), 
                           ax=None, legend=False, color_map=None):
    """
        Stacked bar plot to showcase the compound based distribution of cell counts. Adopted from AmbrosM. 
    """
    # Add a column of zeros to the left and compute the cumulative sums
    cc = np.hstack([np.zeros((len(cell_types_in_drops), 1)), cell_types_in_drops])
    cc_cs = cc.cumsum(axis=1)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    cell_types = cell_types_in_drops.columns
    for i, cell_type in enumerate(cell_types):
        if color_map is None:
            color=palette_celltype[i]
        else:
            color=color_map[cell_type]
        ax.bar(np.arange(len(cc_cs)),
               cc_cs[:,i+1] - cc_cs[:,i],
               bottom=cc_cs[:,i],
               label=cell_types[i], color=color)
         
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(np.arange(len(cc_cs)))
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, rotation=90)
    if colors is not None:
        for ticklabel, color in zip(ax.get_xticklabels(), colors):
            ticklabel.set_color(color)
    if legend: 
        ax.legend()
    color_legend_handles = [
        matplotlib.patches.Patch(facecolor='red', label='-'),
        matplotlib.patches.Patch(facecolor='blue', label='-'),
        matplotlib.patches.Patch(facecolor='green', label='-'),
    ]
    return fig, ax



def run_scib(bulk_adata, layer='lognorm', layer_baseline='n_counts', batch_key='plate_name', label_key='cell_type'):
    import scib
    bulk_adata.X = bulk_adata.layers[layer_baseline].copy()

    bulk_adata_c = bulk_adata.copy()
    bulk_adata_c.X = bulk_adata_c.layers[layer].copy()

    scib.pp.reduce_data(
        bulk_adata_c, n_top_genes=None, batch_key=batch_key, pca=True, neighbors=True
    )
    rr = scib.metrics.metrics(bulk_adata, bulk_adata_c, batch_key, label_key, organism='human', 
                            # biological conservation (label)
                            nmi_=True, 
                            ari_=False,
                            silhouette_=True,
                            isolated_labels_f1_=False, # there is no isolated cell type
                            isolated_labels_asw_=False, # there is no isolated cell type
                            # biological conservation (label free)
                            cell_cycle_=True,
                            hvg_score_=False,
                            trajectory_=False,
                            # batch correction
                            pcr_=False, 
                            graph_conn_=False,
                            kBET_=True,
                            ilisi_=False,
                            clisi_=False,
                            # Not sure what they are
                            isolated_labels_=False,  # backwards compatibility
                            n_isolated=None,
                            lisi_graph_=False,
                            )
    rr = rr.dropna().T
    return rr 
def run_classifier(adata, layer, batch_key):
    import lightgbm as lgb

    print('GB classifier')
    model = lgb.LGBMClassifier(silent=True, verbose=-1)
    # model = RidgeClassifier()
    X = adata.layers[layer].copy()
    y = adata.obs[batch_key]
    scoring = {
        'accuracy_score': make_scorer(accuracy_score)
    }
    score = 1 - cross_validate(model, X, y, cv=5, scoring=scoring, return_train_score=False)['test_accuracy_score'].mean()
    
    return pd.DataFrame({'Batch classifier':[score]})


def isolation_forest(df_subset, group=['index'], cell_type_col='cell_type', values_col='cell_count'):
    """
        Identifies outlier compounds based on ratio of cell type in pseudobulked samples. 
    """
    from sklearn.ensemble import IsolationForest
    cell_count_m = df_subset.pivot(index=group, columns=cell_type_col, values=values_col)
    cell_count_ratio = cell_count_m.div(cell_count_m.sum(axis=1), axis=0)
    cell_count_ratio = cell_count_ratio.fillna(0)
    clf = IsolationForest(max_samples=100, random_state=0)
    clf.fit(cell_count_ratio.values)
    outlier_compounds = cell_count_ratio.index[clf.predict(cell_count_ratio.values)==-1]
    return outlier_compounds

def process_trace_local(job_ids_dict):
    def get_sacct_data(job_id):
        command = f'sacct -j {job_id} --format=JobID,JobName,AllocCPUS,Elapsed,State,MaxRSS,MaxVMSize'
        output = subprocess.check_output(command, shell=True).decode()
        
        # Load the output into a DataFrame
        df = pd.read_csv(io.StringIO(output), delim_whitespace=True)
        df = df.iloc[[2]]
        return df
    def elapsed_to_hours(elapsed_str):
        time = elapsed_str.split('-')
        if len(time) > 1:
            day = int(time[0])
            time = time[1]
        else:
            day = 0
            time = time[0]
        h, m, s = map(int, time.split(':'))
        hours = day*24 + h + m / 60 + s / 3600
        hours_total = 20*hours # because we used 20 cps
        return hours_total
    def reformat_data(df_local):
        # Remove 'K' and convert to integers
        df_local['MaxRSS'] = df_local['MaxRSS'].str.replace('K', '').astype(int)
        df_local['MaxVMSize'] = df_local['MaxVMSize'].str.replace('K', '').astype(int)
        df_local['Elapsed'] = df_local['Elapsed'].apply(lambda x: (elapsed_to_hours(x)))

        # Convert MaxRSS and MaxVMSize from KB to GB
        df_local['MaxRSS'] = df_local['MaxRSS'] / (1024 ** 2)  # Convert KB to GB
        df_local['MaxVMSize'] = df_local['MaxVMSize'] / (1024 ** 2)  # Convert KB to GB
        return df_local
    for i, (name, job_id) in enumerate(job_ids_dict.items()):
        if type(job_id)==list:
            
            for i_sub, job_id_ in enumerate(job_id):
                df_ = get_sacct_data(job_id_)
                df_ = reformat_data(df_)
                if i_sub == 0:
                    df = df_
                else:
                    concat_df = pd.concat([df, df_], axis=0)
                    df['MaxVMSize'] = concat_df['MaxVMSize'].max()
                    df['MaxRSS'] = concat_df['MaxRSS'].max()
                    df['Elapsed'] = concat_df['Elapsed'].sum()
        else: 
            df = get_sacct_data(job_id)
            df = reformat_data(df)
        df.index = [name]
        if i==0:
            df_local = df
        else:
            df_local = pd.concat([df_local, df], axis=0)
        
    
    return df_local

def load_env(env_file="env.yaml"):
    import yaml
    import os
    
    # Get the grn_benchmark root directory (parent of src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    grn_benchmark_root = os.path.dirname(current_dir)
    env_path = os.path.join(grn_benchmark_root, env_file)
    
    def load_config(config_path=env_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    env = load_config()
    return env

def read_yaml_raw(file_path):
    import yaml
    with open(file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    record_store = []
    for entry in yaml_content:
        dataset_id = entry['dataset_id']
        if dataset_id == 'None':
            continue
        method_id = entry['method_id']
        try:
            metric_ids = entry['metric_ids']
        except:
            continue
        metric_values = entry['metric_values']

        for metric_id, metric_value in zip(metric_ids, metric_values):
            record_store.append({
                'dataset_id': dataset_id,
                'method_id': method_id,
                'metric_id': metric_id,
                'metric_value': float(metric_value)
            })
    df = pd.DataFrame(record_store)
    return df
def pivot_table(df):
    # print(df.groupby(['dataset_id', 'method_id']).size().sort_values(ascending=False))
    df = df.pivot(index=['dataset_id', 'method_id'], columns='metric_id', values='metric_value').reset_index()
    df.rename(columns={'dataset_id': 'dataset', 'method_id': 'model'}, inplace=True)
    return df
def read_yaml(file_path):
    df = read_yaml_raw(file_path).reset_index()
    df = df[(df['dataset_id']!= 'missing') ]
    df = df[df['metric_value'] != "None"]
    df = pivot_table(df)
    return df


def plot_raw_scores(scores_mat, ax):
    scores_mat = scores_mat.dropna(how='all', axis=1)
    
    available_methods = list(dict.fromkeys(method for method in METHODS if method in scores_mat.index))
    scores_mat = scores_mat.loc[available_methods]
    scores_mat.index = scores_mat.index.map(lambda name: surrogate_names.get(name, name))
    scores_mat.columns = scores_mat.columns.map(lambda name: surrogate_names.get(name, name))
    
    
    plot_heatmap(scores_mat.fillna(0), name='', ax=ax, cmap="viridis")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    ax.set_ylabel('')


