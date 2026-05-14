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
from statsmodels.stats.multitest import multipletests
from genernbi_supp.src.helper import load_env

env = load_env()
RESULTS_DIR = env['RESULTS_DIR']
figs_dir = F"{env['RESULTS_DIR']}/figs"

sys.path.append(env['genernbi_supp_DIR'])
from src.helper import plot_heatmap, surrogate_names, custom_jointplot, palette_celltype, \
                       palette_methods, \
                       palette_datasets, colors_blind, linestyle_methods, palette_datasets, CONTROLS3, linestyle_methods, retrieve_grn_path, \
                        plot_raw_scores

os.makedirs(f'{RESULTS_DIR}/experiment/metrics_stability', exist_ok=True)

def config_regression():
    dataset = 'op' 
    gene_wise_output = f'{RESULTS_DIR}/experiment/metrics_stability/{dataset}_regression.csv'
    gene_wise_feature_importance = f'{RESULTS_DIR}/experiment/metrics_stability/{dataset}_regression_fi.csv'
    return dataset, gene_wise_output, gene_wise_feature_importance

def config_ws():
    dataset = 'replogle'
    ws_output = f'{RESULTS_DIR}/experiment/metrics_stability/{dataset}_ws.csv'
    return dataset, ws_output

def bh_adjust(pvals):
    # multipletests returns adjusted pvals in index 1 when method='fdr_bh'
    _, adj, _, _ = multipletests(pvals, method='fdr_bh')
    return adj

def pval_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''

def plot_model_comparison_with_significance( df,
    control_model="pearson_corr",
    y_label='WS distance',
    model_order=None,
    per_theta=False,
    q_low=0.01,
    q_high=0.99,
    col_wrap=4,
    s=5,
    figsize=(12, 6),
    cmap='viridis',
    jitter_strength=0.12,
    random_state=0,
    test_type='ttest',
    value_col='ws_distance_pc', ylim=None):
    import math
    import numpy as np

    # 1) compute median per (model, theta, source) if not already aggregated
    if not set(['model', 'theta', 'source']).issubset(df.columns):
        raise ValueError("df must contain 'model','theta','source' columns")
    if control_model not in df['model'].unique():
        print(f"Available models: {df['model'].unique()}")
        raise ValueError(f"control_model '{control_model}' not found in df['model']")
    df_med = (
        df
        .groupby(['model', 'theta', 'source'], as_index=False)
        .agg({value_col: 'median', 'present_edges_n': 'median'})
    )
    df_med['present_edges_q'] = df_med['present_edges_n'].rank(method='average', pct=True)
    df_med['present_edges_q_clipped'] = np.clip((df_med['present_edges_q'] - q_low) / (q_high - q_low), 0.0, 1.0)
    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=0, vmax=1)
     
    np.random.seed(random_state)
    def compute_pvals(sub_df):
        models = [m for m in sub_df['model'].unique() if m != control_model]
        pvals = []
        tested_models = []
        ctrl_vals = sub_df.loc[sub_df['model'] == control_model, value_col].dropna().values
        for m in models:
            vals = sub_df.loc[sub_df['model'] == m, value_col].dropna().values
            
            if len(ctrl_vals) < 2 or len(vals) < 2:
                raise ValueError(f"Not enough data to perform statistical test between control model '{control_model}' and model '{m}'")
            else:
                if test_type=='ttest':
                    from scipy.stats import ttest_ind
                    _, p = ttest_ind(ctrl_vals, vals, equal_var=False)
                else:
                    from scipy.stats import mannwhitneyu
                    _, p = mannwhitneyu(ctrl_vals, vals, alternative='greater')
                
            pvals.append(p)
            tested_models.append(m)
        if len(pvals) == 0:
            return pd.DataFrame(columns=['model','pval','pval_adj','stars'])
        pvals = np.array(pvals, dtype=float)
        adj = bh_adjust(pvals)
        out = pd.DataFrame({'model': tested_models, 'pval': pvals, 'pval_adj': adj})
        out['stars'] = out['pval_adj'].apply(pval_to_stars)
        return out
    # --------- PER-THETA plots -------------
    thetas = sorted(df_med['theta'].unique())
    n = len(thetas)
    ncols = min(col_wrap, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=False)
    # ensure axes is always a 1D iterable
    if nrows * ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = np.ravel(axes)
    all_sig_dfs = []
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    i_plot = 0
    for ax, theta in zip(axes.flatten(), thetas):
        sub = df_med[df_med['theta'] == theta]
        if sub.empty:
            ax.axis('off')
            continue
        sig_df_theta = compute_pvals(sub)
        # print(sig_df_theta)
        all_sig_dfs.append((theta, sig_df_theta))

        sns.boxplot(data=sub, x='model', y=value_col, order=model_order,
                    fliersize=0, ax=ax,  boxprops=dict(facecolor='none', edgecolor='black'))
        
        x = sub['model'].map({m:i for i,m in enumerate(model_order)}).values
        x_jitter = x + np.random.normal(loc=0.0, scale=jitter_strength, size=len(x))
        sc = ax.scatter(x_jitter, sub[value_col].values,
                        # c=sub['present_edges_q_clipped'].values, 
                        c='black',
                        cmap=cmap_obj, norm=norm,
                        s=s, alpha=0.9, edgecolors='none')

        # annotate stars for this theta
        sig_map_theta = {row['model']: row['stars'] for _, row in sig_df_theta.iterrows()}
        ymax_all = sub[value_col].max()
        ymin_all = sub[value_col].min()
        offset = (ymax_all - ymin_all) * 0.1 if (ymax_all > ymin_all) else 0.01
        for i, model in enumerate(model_order):
            if model == control_model:
                continue
            star = sig_map_theta.get(model, '')
            subset_ymax = sub.loc[sub['model'] == model, value_col].max()
            if np.isnan(subset_ymax):
                subset_ymax = ymax_all
            y = subset_ymax + offset
            ax.text(i, y, star, ha='center', va='bottom', fontsize=14, color='red', weight='bold')
        ax.set_title(f"{surrogate_names.get(theta, theta)}", pad=15)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('')
        if i_plot!=0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel(y_label)
        ax.spines[['right', 'top']].set_visible(False)    
        if ylim is not None:
            ax.set_ylim(*ylim)
        i_plot+=1
    for ax in axes.flatten()[len(thetas):]:
        ax.axis('off')
    if False:
        fig.subplots_adjust(right=0.85)
        cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(mappable, cax=cax, label='Number of edges')
        pct_ticks = np.linspace(0.0, 1.0, 5)
        actual_vals = np.quantile(df_med['present_edges_n'].values, pct_ticks)
        cax.set_yticks(pct_ticks)
        cax.set_yticklabels([str(int(v)) for v in actual_vals])
    plt.tight_layout()
    return fig, all_sig_dfs
def plot_lowness_reg(df, x_col='WS distance (raw)', y_col='Expression fold change (abs)'):
    from scipy.stats import spearmanr, pearsonr
    spearman_corr, spearman_p = spearmanr(df[x_col], df[y_col])
    pearson_corr, pearson_p = pearsonr(df[x_col], df[y_col])

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue='model',
        alpha=1,
        s=20,
        palette=palette_methods,
        ax=ax
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.spines[['right', 'top']].set_visible(False)

    # Model legend
    handles1, labels1 = ax.get_legend_handles_labels()
    model_legend = ax.legend(handles1, labels1, loc=(1.02, 0.5), frameon=False, title='Model')

    # Rectangle handle for correlation info
    rect_patch = mpatches.Patch(facecolor='black', edgecolor='black', linewidth=.2, 
                               )
    fit_label = f'Spearman = {spearman_corr:.2f}\n(p = {spearman_p:.2e})'
    fit_legend = ax.legend(
        [rect_patch], [fit_label],
        loc=(1.02, 0.1),
        frameon=False,
    )
    ax.add_artist(model_legend)

    sns.regplot(data=df, x=x_col, y=y_col, scatter=False, lowess=True, color='black', line_kws={'linewidth': 1})

def wrapper_plot_regression(gene_wise_output):
    regression_rr = pd.read_csv(gene_wise_output, index_col=0)
    regression_rr['theta'] = regression_rr['theta'].map(lambda x: surrogate_names.get(x, x))
    regression_rr['model'] = regression_rr['model'].map(lambda x: surrogate_names.get(x, x))
    df_thetas = regression_rr[regression_rr['theta']=='Regression (recall)']
    df_raw = regression_rr[regression_rr['theta']=='r2_raw']
    # print(df_raw['model'].unique())
    df_raw.head()
    # print(df_thetas.head())

    df = df_raw.rename({'r2': 'R2 score', 'n_regulators': 'Number of regulators'}, axis=1)
    plot_lowness_reg(df[df['Number of regulators']<df['Number of regulators'].quantile(0.99)], x_col='Number of regulators', y_col='R2 score')
    file_name = f"{RESULTS_DIR}/figs/lowess_nreg_vs_r2.png"
    print(f"Saved: {os.path.abspath(file_name)}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')

    _ = plot_model_comparison_with_significance(df_thetas.rename({'gene':'source', 'n_regulators':'present_edges_n'}, axis=1), 
                                            model_order=['Pearson Corr.', 'Scenic+', 'GRNBoost2', 'PPCOR'],
                                            per_theta=True, 
                                            control_model='Pearson Corr.', 
                                            y_label='R2 score',
                                            value_col='r2', 
                                            figsize=(2.7, 3),
                                            s=1,
                                            test_type='mannwhitneyu',
                                            ylim=None)
    plt.title('')
    file_name = f"{RESULTS_DIR}/figs/model_comparison_regression_thetas.png"
    print(f"Saved: {os.path.abspath(file_name)}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    _ = plot_model_comparison_with_significance(df_raw.rename({'gene':'source', 'n_regulators':'present_edges_n'}, axis=1), 
                                                model_order=['Pearson Corr.', 'Scenic+', 'GRNBoost2', 'PPCOR'],
                                                per_theta=True, 
                                                control_model='Pearson Corr.', 
                                                y_label='R2 score',
                                                value_col='r2', 
                                                figsize=(2.7, 3),
                                                s=1,
                                                test_type='mannwhitneyu',
                                                # ylim=(-.1, .1)
                                                )
    plt.title('')
    file_name = f"{RESULTS_DIR}/figs/model_comparison_regression_raw.png"
    print(f"Saved: {os.path.abspath(file_name)}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')



def plot_regression_feature_stability_scores(scores_store_all_regression, ax):
    from scipy.stats import mannwhitneyu
    scores_store_present = scores_store_all_regression[scores_store_all_regression['present']]
    df_common_pivot = scores_store_present.pivot(index=['model'], columns=['donor_id','gene'], values='feature_importance_mean2std').dropna(axis=1)
    common_genes = df_common_pivot.columns.get_level_values('gene').unique()
    df = df_common_pivot
    top_95_quantile = scores_store_present['contextual_tf_activity'].quantile(0.99)
    filtered_data = scores_store_present[scores_store_present['contextual_tf_activity'] <= top_95_quantile]
    sns.violinplot(filtered_data, x='model', y='contextual_tf_activity', linewidth=0.5,  cut=-1, inner=None, ax=ax, palette=palette_methods)
    ax.set_xlabel("")
    ax.set_ylabel("Contextual TF\nactivity score")
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')  # Set horizontal alignment to 'left'
    sigs = ['','','','*','']
    positions = [.4]*4
    i = 0
    for ii, row_name in enumerate(range(scores_store_present['model'].nunique())):
        ax.text(
            i,  # x-coordinate
            positions[ii],  # Position above the highest point in the box
            sigs[ii],  # Format p-value in scientific notation
            ha='center',
            va='bottom',
            fontsize=20,
            color='red'
        )
        i+=1
def plot_tf_activity_grn_derived(scores_store_all_regression, ax):
    top_95_quantile = scores_store_all_regression['contextual_tf_activity'].quantile(.98)
    filtered_data = scores_store_all_regression[scores_store_all_regression['contextual_tf_activity'] <= top_95_quantile]
    sns.violinplot(filtered_data, x='present', y='contextual_tf_activity', linewidth=0.5,  cut=0, inner=None, ax=ax, palette=colors_blind)

    ax.set_ylabel('Contextual TF\nactivation score')

    ax.set_xlabel("")
    ax.set_xticklabels(['Random','GRN derived'])

    # plt.title('TF activation stability across perturbation (OPSCA)', fontsize=12, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=0)

    sigs = ['*','','','','']
    positions = [.3]*5
    i = 0
    for ii, row_name in enumerate(range(scores_store_all_regression['model'].nunique())):
        ax.text(
            i,  # x-coordinate
            positions[ii],  # Position above the highest point in the box
            sigs[ii],  # Format p-value in scientific notation
            ha='center',
            va='bottom',
            fontsize=20,
            color='red'
        )
        i+=1
def plot_joint_regression_tf_activity_vs_nregulators(scores_store_all_regression, ax, top_plot=True):
    custom_jointplot(scores_store_all_regression,x = 'n_regulator', 
             y = 'contextual_tf_activity', 
             hue= 'present', ax=ax, scatter_kws={'s':20})
    ax.legend(title="GRN-derived", frameon=False)
    ax.set_xlabel("Number of regulators")
    ax.set_ylabel("contextual_tf_activity")
def plot_joint_regression_r2scores_vs_nregulators(scores_store_all_regression, ax, palette_present, top_plot=True):
    custom_jointplot(
        scores_store_all_regression,
        x='n_regulator',
        y='r2score',
        hue='present',
        ax=ax,
        scatter_kws=dict(s=20, palette=palette_present),
        top_plot=top_plot
    )
    ax.set_xlabel("Number of regulators")
    ax.set_ylabel("Performance")
def plot_regression_perfromance_similarity_donors(scores_store_all_regression, axes):
    def plot_heatmap_local(corr_matrix, ax, vmin, vmax):
        np.fill_diagonal(corr_matrix.values, np.nan)  # Optional: mask diagonal
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap="viridis", 
            cbar=False, 
            ax=ax, 
            vmin=vmin, 
            vmax=vmax
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params('y', rotation=0)
    scores_store_table = scores_store_all_regression[scores_store_all_regression['present']][['donor_id', 'r2score', 'model', 'gene']].pivot(
        index='donor_id', 
        values='r2score', 
        columns=['model', 'gene']
    )
    corr_matrix_grn = scores_store_table.T.corr(method='spearman')
    scores_store_table = scores_store_all_regression[~scores_store_all_regression['present']][['donor_id', 'r2score', 'model', 'gene']].pivot(
        index='donor_id', 
        values='r2score', 
        columns=['model', 'gene']
    )
    corr_matrix_random = scores_store_table.T.corr(method='spearman')
    vmin = min(corr_matrix_grn.min().min(), corr_matrix_random.min().min())
    vmax = max(corr_matrix_grn.max().max(), corr_matrix_random.max().max())
    plot_heatmap_local(corr_matrix_grn, axes[0], vmin, vmax)
    axes[0].set_title('GRN-derived', pad=15)
    axes[0].tick_params(axis='x', rotation=45)
    for label in axes[0].get_xticklabels():
        label.set_ha('right')  # Set horizontal alignment to 'left'
    plot_heatmap_local(corr_matrix_random, axes[1], vmin, vmax)
    axes[1].set_title('Randomly-assigned', pad=15)
    axes[1].tick_params(axis='x', rotation=45)
    for label in axes[1].get_xticklabels():
        label.set_ha('right')  # Set horizontal alignment to 'left'

def plot_performance_similarity_models_reg(df, ax, score_col='r2score'):
    scores_store_table = df[['donor_id', score_col, 'model', 'gene']].pivot(
                index='model', 
                values=score_col, 
                columns=['donor_id', 'gene']
            )

    spearman_corr = scores_store_table.T.corr(method='spearman')
    np.fill_diagonal(spearman_corr.values, np.nan)
    sns.heatmap(spearman_corr, annot=True, cmap="viridis", cbar=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params('y', rotation=0)
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')  

def plot_joint_scores(df, method1, method2, ax, color_map, value_col='r2score', col='gene'):
    sys.path.append('../')
    from src.helper import custom_jointplot
    scores_store_table = df.pivot_table(index=col, columns='model', values=value_col, aggfunc='mean')
    present_table = df.pivot_table(index=col, columns='model', values='present', aggfunc='mean')
    present_table = present_table.astype(bool)
    scores_store_table['present'] = 'Neither models'
    scores_store_table.loc[present_table[method1], 'present'] = method1
    scores_store_table.loc[present_table[method2], 'present'] = method2
    scores_store_table.loc[present_table[method1]&present_table[method2], 'present'] = 'Both models'
    custom_jointplot(scores_store_table,x = method1, 
             y = method2, 
             hue= 'present', ax=ax,
             scatter_kws={'s':15, 
             'palette':[color_map[name] for name in scores_store_table['present'].unique()]},
            #  'palette':[color_map[name] for name in ['Positive Ctrl', 'Pearson Corr.', 'GRNBoost2', 'PPCOR', 'None', 'Both']]},
              alpha=.5)
    return scores_store_table

def wrapper_regression_feature_analysis(dataset, gene_wise_feature_importance):
    # Load the full data once
    scores_reg_all = pd.read_csv(gene_wise_feature_importance, index_col=0)
    scores_reg_all.model = scores_reg_all.model.map(surrogate_names)
    if 'donor_id' not in scores_reg_all.columns:
        scores_reg_all['donor_id'] = 'donor_0'
    scores_reg_all['donor_id'] = scores_reg_all['donor_id'].map({'donor_0':'Donor 1', 'donor_1':'Donor 2', 'donor_2':'Donor 3'})
    for theta in scores_reg_all['theta'].unique():
        scores_reg = scores_reg_all[scores_reg_all['theta']==theta].copy()
        if True:
            scores_reg['feature_importance_mean2std_log'] = np.log1p(scores_reg['feature_importance_mean2std'])
            scores_reg['contextual_tf_activity'] = 1/scores_reg['feature_importance_mean2std']
            scores_store_present = scores_reg[scores_reg['present']]

            fig, axes = plt.subplots(1, 2, figsize=(2.7, 2), width_ratios=[1.5, 1], sharey=True)
            ax = axes[0]
            plot_regression_feature_stability_scores(scores_reg, ax)
            ax.set_ylabel('')
            ax = axes[1]
            plot_tf_activity_grn_derived(scores_reg, ax)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylabel('')
            file_name = f"{RESULTS_DIR}/figs/reg_feature_importance_stability_{dataset}_{theta}.png"
            print(f"Saved: {os.path.abspath(file_name)}")
            plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')

        if True:
            ## Number of regulators vs r2 scores and tf activation stability
            fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), sharex=True)
            palette_present = {True: colors_blind[0], False: colors_blind[1]}
            print(palette_present)
            plot_joint_regression_r2scores_vs_nregulators(scores_reg, ax, palette_present, top_plot=True)
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=palette_present[val],
                    markersize=10, label=str(val))
                for val in [True, False]
            ]
            ax.legend(handles=legend_elements, loc=(1.22, 0.3),
                    frameon=False, title="Gene has regulator")
            file_name = f"{RESULTS_DIR}/figs/regression_nregulators_vs_r2scoes_{dataset}_{theta}.png"
            print(f"Saved: {os.path.abspath(file_name)}")
            plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
        if True:
            ### Similary of scores across donors
            fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.5), sharey=True)

            plot_regression_perfromance_similarity_donors(scores_reg, axes)

            file_name = f"{RESULTS_DIR}/figs/regression_scores_similarity_donors_{dataset}_{theta}.png"
            print(f"Saved: {os.path.abspath(file_name)}")
            fig.savefig(
                file_name, 
                dpi=300, 
                transparent=True, 
                bbox_inches='tight'
            )
        if True:
            ### Similariy of scores across models
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)
            plot_performance_similarity_models_reg(scores_reg, ax)
            file_name = f"{RESULTS_DIR}/figs/regression_models_corr_{dataset}_{theta}.png"
            print(f"Saved: {os.path.abspath(file_name)}")
            fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')


        ### Joint distribution of gene wise scores
        color_map = {'Neither models':'green', 'Both models':'grey', **palette_methods}
        figsize = (7, 2.5)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax = axes[0]
        aa = plot_joint_scores(scores_reg, 'Pearson Corr.', 'Scenic+', ax=ax, color_map=color_map)
        ax.get_legend().remove()
        ax = axes[1]
        aa = plot_joint_scores(scores_reg, 'PPCOR', 'GRNBoost2', ax=ax, color_map=color_map)
        ax.get_legend().remove()
        legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[name], markersize=10, label=name)
                for name in ['Both models', 'Neither models', 'PPCOR', 'GRNBoost2', 'Pearson Corr.', 'Scenic+' ]
            ]
        ax.legend(handles=legend_elements, loc=(1.3, 0), frameon=False, title='Gene has regulator in:')
        plt.tight_layout()
        file_name = f"{RESULTS_DIR}/figs/regression_joint_tfactivity_vs_r2scores_{dataset}_{theta}.png"
        print(f"Saved: {os.path.abspath(file_name)}")
        plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')

def plot_performance_similarity_models_ws(df, ax, score_col='r2score'):
    scores_store_table = df[[score_col, 'model', 'source']].pivot(
                index='model', 
                values=score_col, 
                columns=['source']
            )

    spearman_corr = scores_store_table.T.corr(method='spearman')
    spearman_corr = spearman_corr.round(2)
    np.fill_diagonal(spearman_corr.values, np.nan)
    sns.heatmap(spearman_corr, annot=True, cmap="viridis", cbar=False, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params('y', rotation=0)
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')  

def wrapper_ws_analysis(dataset, ws_output):
    # - correlations of scores
    assert dataset in ['replogle'], f"{dataset} not supported in WS analysis"
    ws_rr = pd.read_csv(ws_output, index_col=0)
    ws_rr['theta'] = ws_rr['theta'].map(lambda x: surrogate_names.get(x, x))
    ws_rr['model'] = ws_rr['model'].map(lambda x: surrogate_names.get(x, x))
    ws_rr['WS distance (standardized)'] = ws_rr['ws_distance_pc']
    ws_rr['WS distance (raw)'] = ws_rr['ws_distance']
    
    # Add 'present' column if not exists - indicates if TF has edges in the network
    if 'present' not in ws_rr.columns:
        if 'present_edges_n' in ws_rr.columns:
            ws_rr['present'] = ws_rr['present_edges_n'] > 0
        else:
            # Assume all TFs are present if we don't have this info
            ws_rr['present'] = True
    
    ws_thetas = ws_rr[ws_rr['theta']!='ws_raw']
    ws_raw = ws_rr[ws_rr['theta']=='ws_raw']

    for theta in ws_thetas['theta'].unique():
        df = ws_thetas[ws_thetas['theta']==theta]
        df_mean = df.groupby(['model', 'source'], as_index=False).agg({'ws_distance_pc':'mean'})
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)
        
        plot_performance_similarity_models_ws(df_mean, ax, score_col='ws_distance_pc')
        plt.title(theta)
        file_name = f"{RESULTS_DIR}/figs/ws_models_corr_{theta}.png".replace(' ', '_').replace('(', '').replace(')', '')
        print(f"Saved: {os.path.abspath(file_name)}")
        fig.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    # ----- WS vs perturbation effect
    perturb_effect_all = pd.read_csv(f'{RESULTS_DIR}/exp_analysis/perturb_effect_all.csv')
    perturb_effect_t = perturb_effect_all[perturb_effect_all['Dataset']==dataset]
    df_raw_mean = ws_thetas.groupby(['model', 'tf'])['WS distance (raw)'].mean().reset_index()
    df_raw_mean = df_raw_mean.merge(perturb_effect_t, left_on=['tf'], right_on=['perturbation'], how='left')
    df_raw_mean = df_raw_mean[~df_raw_mean['perturbation'].isna()]
    df_raw_mean['Expression fold change (abs)'] = df_raw_mean['Expression fold change'].abs()
    subset = df_raw_mean[(df_raw_mean['WS distance (raw)'] > 0) & (df_raw_mean['Expression fold change (abs)'] > 0)]

    # Remove top 5% outliers (keep bottom 95%)
    percentile_95 = subset['WS distance (raw)'].quantile(0.95)
    n_before = len(subset)
    subset = subset[subset['WS distance (raw)'] <= percentile_95]
    n_after = len(subset)
    # print(f"Removed top 5% outliers: {n_before - n_after} cases ({(n_before - n_after)/n_before*100:.2f}%)")
    # print(f"WS distance threshold at 95th percentile: {percentile_95:.4f}")

    plot_lowness_reg(subset, x_col='WS distance (raw)', y_col='Expression fold change (abs)')
    file_name = f"{RESULTS_DIR}/figs/ws_vs_perturbation_effect_{dataset}.png"
    print(f"Saved: {os.path.abspath(file_name)}")
    plt.savefig(file_name,
                dpi=300, transparent=True, bbox_inches='tight')

    # ----- Model comparison plots
    _ = plot_model_comparison_with_significance(ws_thetas, per_theta=True, 
                                                control_model='Pearson Corr.', 
                                                value_col='WS distance (raw)', 
                                                figsize=(7, 3),
                                                model_order=['Pearson Corr.', 'Scenic', 'GRNBoost2', 'PPCOR'],
                                                ylim=(-.01, .2))
    file_name = f"{RESULTS_DIR}/figs/ws_distance_comparision_{dataset}.png"
    print(f"Saved: {os.path.abspath(file_name)}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')
    
    _ = plot_model_comparison_with_significance(ws_thetas, per_theta=True, 
                                                control_model='Pearson Corr.', 
                                                value_col='WS distance (standardized)', 
                                                figsize=(7, 3),
                                                model_order=['Pearson Corr.', 'Scenic', 'GRNBoost2', 'PPCOR'],
                                                ylim=(-.1, 1.3)) 
    file_name = f"{RESULTS_DIR}/figs/ws_distance_normalized_comparision_{dataset}.png"
    print(f"Saved: {os.path.abspath(file_name)}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')

    # ----- Joint distribution of TF-wise scores (similar to regression analysis)
    color_map = {'Neither models':'green', 'Both models':'grey', **palette_methods}
    figsize = (7, 2.5)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[0]
    aa = plot_joint_scores(ws_thetas, 'Pearson Corr.', 'Scenic', ax=ax, color_map=color_map, 
                          value_col='WS distance (raw)', col='tf')
    ax.get_legend().remove()
    ax = axes[1]
    aa = plot_joint_scores(ws_thetas, 'PPCOR', 'GRNBoost2', ax=ax, color_map=color_map,
                          value_col='WS distance (raw)', col='tf')
    ax.get_legend().remove()
    legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[name], markersize=10, label=name)
            for name in ['Both models', 'Neither models', 'PPCOR', 'GRNBoost2', 'Pearson Corr.', 'Scenic']
        ]
    ax.legend(handles=legend_elements, loc=(1.3, 0), frameon=False, title='TF present in:')
    plt.tight_layout()
    file_name = f"{RESULTS_DIR}/figs/ws_joint_tfwise_scores_{dataset}.png"
    print(f"Saved: {os.path.abspath(file_name)}")
    plt.savefig(file_name, dpi=300, transparent=True, bbox_inches='tight')