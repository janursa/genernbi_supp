"""
Gene-wise Ensemble (Custom) – Inference-Blind Evaluation

Unlike the standard ensemble (src/ensemble/script.py) which uses evaluation data
for model selection (data leakage), this version performs all model selection
exclusively on inference data, keeping evaluation data as a truly blind test.

Algorithm
---------
1. Build consensus regulators from the specified GRN methods (for dataset/theta)
2. Load inference data; assign CV groups:
     - op / MSCIC  →  leave-donor-out  (donor_id column)
     - replogle     →  leave-perturbation-out  (perturbation column)
3. For each model, evaluate gene-wise R² using CV on inference data
4. For each gene, select the best model → take its top-θ regulators
5. Evaluate ensemble GRN vs individual GRNs on evaluation data (blind)

Usage
-----
    python script.py --rr_folder <out_dir> --dataset op --theta 0.25

Example (from geneRNBI root after sourcing env.sh):
    python src/ensemble_custom/script.py \\
        --rr_folder $RESULTS_DIR/experiment/ensemble_custom \\
        --dataset op --theta 0.25
"""

import os
import sys
import argparse
import json
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Gene-wise ensemble using inference-data CV")
parser.add_argument("--rr_folder", type=str, required=True, help="Root results folder")
parser.add_argument("--dataset",   type=str, required=True, help="Dataset name (e.g. op, MSCIC, replogle)")
parser.add_argument("--theta",     type=float, default=0.25, help="Quantile for number of regulators (default: 0.25)")
parser.add_argument("--grns",      type=str,   nargs="+",
                    default=None,
                    help="GRN methods to ensemble. Defaults: op/MSCIC → scenicplus scenic grnboost pearson_corr ppcor; replogle → scenic grnboost pearson_corr ppcor granie")
parser.add_argument("--skip_metrics", action="store_true", help="Skip per-model CV (reuse existing gene_wise_performance CSV)")
args = parser.parse_args()

dataset     = args.dataset
theta       = args.theta
rr_folder   = os.path.join(args.rr_folder, dataset)
skip_metrics = args.skip_metrics

os.makedirs(rr_folder, exist_ok=True)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
from geneRNBI.src.helper import load_env
env = load_env()

TASK_GRN_DIR  = env["TASK_GRN_INFERENCE_DIR"]
INFERENCE_DIR = env["INFERENCE_DIR"]
EVALUATION_DIR = os.path.join(env["RESOURCES_DIR"], "grn_benchmark", "evaluation_data")
PRIOR_DIR     = env["PRIOR_DIR"]

for p in [TASK_GRN_DIR, env["UTILS_DIR"], env["METRICS_DIR"],
          os.path.join(TASK_GRN_DIR, "src", "metrics", "regression"),
          os.path.join(TASK_GRN_DIR, "src", "metrics", "regression", "consensus")]:
    sys.path.insert(0, p)

from util import naming_convention, process_links
from helper import cross_validate_gene, net_to_matrix, fill_zeros_in_grn

# Consensus helper (imported from consensus sub-package)
import importlib.util as _ilu
_cspec = _ilu.spec_from_file_location(
    "consensus_helper",
    os.path.join(TASK_GRN_DIR, "src", "metrics", "regression", "consensus", "helper.py")
)
_cmod = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
build_consensus = _cmod.main  # signature: main(par) → results dict

# ---------------------------------------------------------------------------
# Dataset-specific configuration
# ---------------------------------------------------------------------------
MULTIOMICS_DATASETS = {"op", "MSCIC"}

DEFAULT_GRNS = {
    "op":       ["scenicplus", "scenic", "grnboost", "pearson_corr", "ppcor"],
    "MSCIC":    ["scenicplus", "scenic", "grnboost", "pearson_corr", "ppcor"],
    "replogle": ["scenic",     "grnboost", "pearson_corr", "ppcor", "granie"],
}

CV_GROUP_COL = {
    "op":       "donor_id",
    "MSCIC":    "donor_id",
    "replogle": "perturbation",   # leave-perturbation-out
}

grns = args.grns if args.grns else DEFAULT_GRNS.get(dataset, DEFAULT_GRNS["replogle"])

print(f"\n{'='*70}")
print(f"ensemble_custom  |  dataset={dataset}  |  theta={theta}")
print(f"Methods: {grns}")
print(f"Results → {rr_folder}")
print("="*70)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
grn_models_dir   = os.path.join(env["RESULTS_DIR"], dataset)
inference_rna    = os.path.join(INFERENCE_DIR, f"{dataset}_rna.h5ad")
evaluation_data  = os.path.join(EVALUATION_DIR, f"{dataset}_bulk.h5ad")
tf_all_path      = os.path.join(PRIOR_DIR, "tf_all.csv")
consensus_path   = os.path.join(rr_folder, f"consensus_custom_theta{theta}.json")

reg_type = "ridge"
layer    = "lognorm"

# ---------------------------------------------------------------------------
# STEP 1 — Build consensus from the 5 ensemble methods
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("STEP 1: Building consensus from ensemble methods")
print("="*70)

prediction_paths = [
    os.path.join(grn_models_dir, naming_convention(dataset, m))
    for m in grns
]
missing = [p for p in prediction_paths if not os.path.exists(p)]
if missing:
    print("WARNING: missing prediction files:")
    for p in missing:
        print(f"  {p}")
    prediction_paths = [p for p in prediction_paths if os.path.exists(p)]
    grns = [m for m, p in zip(grns, [os.path.join(grn_models_dir, naming_convention(dataset, m)) for m in grns])
            if os.path.exists(p)]
    print(f"Continuing with {len(grns)} available methods: {grns}")

consensus_par = {
    "evaluation_data": evaluation_data,
    "regulators_consensus": consensus_path,
    "predictions": prediction_paths,
    "max_n_links": 50000,
}
consensus_data = build_consensus(consensus_par)
print(f"Consensus built for {len(consensus_data)} genes → {consensus_path}")

# ---------------------------------------------------------------------------
# STEP 2 — Load inference data + set CV groups
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("STEP 2: Loading inference data and setting CV groups")
print("="*70)

inference_adata = ad.read_h5ad(inference_rna)
print(f"Inference data: {inference_adata.shape[0]} cells × {inference_adata.shape[1]} genes")

cv_col = CV_GROUP_COL.get(dataset)
if cv_col and cv_col in inference_adata.obs.columns:
    group_labels = inference_adata.obs[cv_col].values
    unique_groups = np.unique(group_labels)
    print(f"CV strategy: leave-{cv_col}-out  ({len(unique_groups)} groups: {list(unique_groups)[:8]})")
    groups = LabelEncoder().fit_transform(group_labels)
else:
    print("CV strategy: random 5-fold (no donor/perturbation column found)")
    n_cells = inference_adata.shape[0]
    groups = LabelEncoder().fit_transform(np.random.choice(range(1, 6), size=n_cells, replace=True))

# Expression matrix
X_inf = inference_adata.layers[layer]
try:
    X_inf = X_inf.todense().A
except AttributeError:
    pass
X_inf = RobustScaler(with_centering=False).fit_transform(X_inf)

gene_names = inference_adata.var_names.to_numpy()
n_genes    = len(gene_names)

with open(consensus_path, "r") as f:
    consensus_loaded = json.load(f)

# Only evaluate genes present in both inference data and consensus
gene_names = np.array([g for g in gene_names if g in consensus_loaded])
n_genes    = len(gene_names)
gene_idx   = {g: i for i, g in enumerate(inference_adata.var_names)}
X_inf      = X_inf[:, [gene_idx[g] for g in gene_names]]

n_features_per_gene = np.asarray([consensus_loaded[g][str(theta)] for g in gene_names], dtype=int)

tf_names = np.loadtxt(tf_all_path, dtype=str)

print(f"Evaluating {n_genes} genes  |  {(n_features_per_gene > 0).sum()} with ≥1 regulator at θ={theta}")

# ---------------------------------------------------------------------------
# STEP 3 — Per-gene, per-model CV on inference data
# ---------------------------------------------------------------------------
def evaluate_model_on_inference(model):
    """Load model GRN, evaluate all genes via CV on inference data."""
    net_path = os.path.join(grn_models_dir, naming_convention(dataset, model))
    net = ad.read_h5ad(net_path).uns["prediction"]
    net = process_links(net, {"max_n_links": 50000, "apply_tf": True})

    net_matrix = net_to_matrix(net, gene_names)
    grn = fill_zeros_in_grn(net_matrix)
    mask = np.isin(gene_names, list(tf_names))
    grn[~mask, :] = 0

    def _eval_gene(j):
        n_feat = int(n_features_per_gene[j])
        if n_feat == 0:
            return gene_names[j], -np.inf, "skipped"
        try:
            res = cross_validate_gene(reg_type, X_inf, groups, grn, j, n_feat, n_jobs=1)
            return gene_names[j], res["avg-r2"], "evaluated"
        except Exception as e:
            return gene_names[j], -np.inf, f"failed: {e}"

    results = Parallel(n_jobs=20, backend="loky")(delayed(_eval_gene)(j) for j in range(n_genes))

    gene_scores = {gene: score for gene, score, _ in results}
    n_eval    = sum(1 for _, _, s in results if s == "evaluated")
    n_skip    = sum(1 for _, _, s in results if s == "skipped")
    avg_r2    = np.mean([s for s in gene_scores.values() if s > -np.inf]) if n_eval > 0 else 0.0
    print(f"  {model:20s}: evaluated={n_eval}, skipped={n_skip}, avg_R²={avg_r2:.4f}")
    return model, gene_scores


gene_perf_csv = os.path.join(rr_folder, f"gene_wise_performance_theta{theta}.csv")

if not skip_metrics:
    print(f"\n{'='*70}")
    print("STEP 3: Per-model CV on inference data")
    print("="*70)

    model_gene_scores = {}
    for model in grns:
        m, scores = evaluate_model_on_inference(model)
        model_gene_scores[m] = scores

    # Save per-gene per-model performance
    rows = []
    for gene in gene_names:
        row = {"gene": gene}
        for m in grns:
            row[f"{m}_r2"] = model_gene_scores[m].get(gene, -np.inf)
        scores_valid = {m: row[f"{m}_r2"] for m in grns if row[f"{m}_r2"] > -np.inf}
        if scores_valid:
            best = max(scores_valid, key=scores_valid.get)
            row["best_model"] = best
            row["best_r2"]    = scores_valid[best]
        else:
            row["best_model"] = grns[0]
            row["best_r2"]    = -np.inf
        rows.append(row)

    gene_perf_df = pd.DataFrame(rows)
    gene_perf_df.to_csv(gene_perf_csv, index=False)
    print(f"\nGene-wise performance saved → {gene_perf_csv}")
else:
    print(f"\nSkipping CV (--skip_metrics). Loading from {gene_perf_csv}")

# ---------------------------------------------------------------------------
# STEP 4 — Gene-wise model selection + assemble ensemble GRN
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("STEP 4: Assembling ensemble GRN")
print("="*70)

gene_perf_df = pd.read_csv(gene_perf_csv)

# Log model selection distribution
model_counts = gene_perf_df[~np.isinf(gene_perf_df["best_r2"])]["best_model"].value_counts()
print("Genes assigned per model:")
for m, cnt in model_counts.items():
    print(f"  {m:20s}: {cnt:5d}  ({cnt/n_genes*100:.1f}%)")

# Load networks for assigned models only
used_models = gene_perf_df["best_model"].unique()
model_networks = {}
for model in used_models:
    net = ad.read_h5ad(os.path.join(grn_models_dir, naming_convention(dataset, model))).uns["prediction"]
    model_networks[model] = process_links(net, {"max_n_links": 50000, "apply_tf": True})

ensemble_edges = []
for j, gene in enumerate(gene_names):
    row = gene_perf_df[gene_perf_df["gene"] == gene]
    if row.empty:
        continue
    row = row.iloc[0]
    best_model = row["best_model"]
    n_feat     = int(n_features_per_gene[j])
    if n_feat == 0:
        continue

    gene_edges = model_networks[best_model][model_networks[best_model]["target"] == gene].copy()
    if len(gene_edges) == 0:
        continue

    gene_edges = gene_edges.nlargest(min(n_feat, len(gene_edges)), "weight")
    gene_edges["source_model"] = best_model
    gene_edges["target_r2"]    = row["best_r2"]
    ensemble_edges.append(gene_edges)

ensemble_net = pd.concat(ensemble_edges, ignore_index=True)
print(f"\nEnsemble network: {len(ensemble_net)} edges | "
      f"{ensemble_net['target'].nunique()} targets | "
      f"{ensemble_net['source'].nunique()} TFs")

ensemble_net.to_csv(
    os.path.join(rr_folder, f"ensemble_network_theta{theta}_with_metadata.csv"), index=False
)

# Save as h5ad (same format as individual methods)
final_net = ensemble_net[["source", "target", "weight"]].reset_index(drop=True)
template  = ad.read_h5ad(os.path.join(grn_models_dir, naming_convention(dataset, "grnboost")))
template.uns["prediction"] = final_net
template.uns["method_id"]  = "ensemble_custom"
out_h5ad = os.path.join(rr_folder, naming_convention(dataset, "ensemble_custom"))
template.write_h5ad(out_h5ad)
print(f"Ensemble GRN saved → {out_h5ad}")

# ---------------------------------------------------------------------------
# STEP 5 — Evaluate ensemble on evaluation data (blind)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("STEP 5: Evaluating ensemble on evaluation data (blind)")
print("="*70)

eval_adata = ad.read_h5ad(evaluation_data)
X_eval     = eval_adata.layers[layer]
try:
    X_eval = X_eval.todense().A
except AttributeError:
    pass
X_eval = RobustScaler(with_centering=False).fit_transform(X_eval)

eval_genes = eval_adata.var_names.to_numpy()
eval_genes_in_consensus = np.array([g for g in eval_genes if g in consensus_loaded])

n_feat_eval = np.asarray([consensus_loaded[g][str(theta)] for g in eval_genes_in_consensus], dtype=int)
eval_gene_idx = {g: i for i, g in enumerate(eval_adata.var_names)}
X_eval_sub = X_eval[:, [eval_gene_idx[g] for g in eval_genes_in_consensus]]

# Random CV groups for evaluation data (standard approach)
n_eval_cells = eval_adata.shape[0]
eval_groups  = LabelEncoder().fit_transform(
    np.random.choice(range(1, 6), size=n_eval_cells, replace=True)
)


def evaluate_grn_on_eval(net_df, model_label):
    """Evaluate a GRN (as a DataFrame) on evaluation data via CV regression."""
    net_matrix = net_to_matrix(net_df, eval_genes_in_consensus)
    grn = fill_zeros_in_grn(net_matrix)
    mask = np.isin(eval_genes_in_consensus, list(tf_names))
    grn[~mask, :] = 0

    def _eval_gene(j):
        n_feat = int(n_feat_eval[j])
        if n_feat == 0:
            return eval_genes_in_consensus[j], -np.inf, "skipped"
        try:
            res = cross_validate_gene(reg_type, X_eval_sub, eval_groups, grn, j, n_feat, n_jobs=1)
            return eval_genes_in_consensus[j], res["avg-r2"], "evaluated"
        except Exception as e:
            return eval_genes_in_consensus[j], -np.inf, f"failed: {e}"

    results  = Parallel(n_jobs=20, backend="loky")(
        delayed(_eval_gene)(j) for j in range(len(eval_genes_in_consensus))
    )
    gene_r2  = {gene: score for gene, score, _ in results}
    n_ev     = sum(1 for _, _, s in results if s == "evaluated")
    avg_r2   = np.mean([s for s in gene_r2.values() if s > -np.inf]) if n_ev > 0 else 0.0
    print(f"  {model_label:22s}: avg_R²={avg_r2:.4f}  (n_genes={n_ev})")
    return gene_r2


eval_scores = {}

# Ensemble
net_ens = process_links(final_net, {"max_n_links": 50000, "apply_tf": True})
eval_scores["ensemble_custom"] = evaluate_grn_on_eval(net_ens, "ensemble_custom")

# Individual models
for model in grns:
    net = ad.read_h5ad(os.path.join(grn_models_dir, naming_convention(dataset, model))).uns["prediction"]
    net = process_links(net, {"max_n_links": 50000, "apply_tf": True})
    eval_scores[model] = evaluate_grn_on_eval(net, model)

# Save evaluation scores
eval_rows = []
for method, gene_r2 in eval_scores.items():
    for gene, r2 in gene_r2.items():
        eval_rows.append({"method": method, "gene": gene, "r2_eval": r2})
eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(os.path.join(rr_folder, f"eval_gene_scores_theta{theta}.csv"), index=False)

# ---------------------------------------------------------------------------
# STEP 6 — Visualizations
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("STEP 6: Generating visualizations")
print("="*70)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, env["geneRNBI_DIR"])
from src.helper import palette_methods, surrogate_names

palette_methods.setdefault("ensemble_custom", "#e74c3c")
model_name_map = {**surrogate_names, "ensemble_custom": "Ensemble\n(custom)"}

# --- Plot 1: genes per model (bar) ---
valid_df      = gene_perf_df[~np.isinf(gene_perf_df["best_r2"])]
model_counts2 = valid_df["best_model"].value_counts().sort_values()

fig, ax = plt.subplots(figsize=(2.5, 2))
labels  = [model_name_map.get(m, m) for m in model_counts2.index]
colors  = [palette_methods.get(m, "gray") for m in model_counts2.index]
ax.barh(range(len(model_counts2)), model_counts2.values, color=colors)
ax.set_yticks(range(len(model_counts2)))
ax.set_yticklabels(labels)
ax.set_xlabel("Genes selected")
ax.spines[["right", "top"]].set_visible(False)
ax.margins(y=0.1, x=0.1)
plt.tight_layout()
p1 = os.path.join(rr_folder, f"genes_per_model_theta{theta}.png")
plt.savefig(p1, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {p1}")

# --- Plot 2: R² on evaluation data comparison ---
plot_data = []
all_methods = ["ensemble_custom"] + grns
for method in all_methods:
    for gene, r2 in eval_scores[method].items():
        if not np.isinf(r2):
            plot_data.append({"model": method, "model_display": model_name_map.get(method, method), "r2": r2})
plot_df = pd.DataFrame(plot_data)

ens_r2    = plot_df[plot_df["model"] == "ensemble_custom"]["r2"].values
pvals, test_models = [], []
for m in grns:
    m_r2 = plot_df[plot_df["model"] == m]["r2"].values
    if len(ens_r2) > 0 and len(m_r2) > 0:
        _, pval = mannwhitneyu(ens_r2, m_r2, alternative="greater")
        pvals.append(pval)
        test_models.append(m)

_, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh") if pvals else (None, [], None, None)

model_order  = [model_name_map.get("ensemble_custom", "ensemble_custom")] + \
               [model_name_map.get(m, m) for m in grns]
colors_order = [palette_methods.get("ensemble_custom", "gray")] + \
               [palette_methods.get(m, "gray") for m in grns]

fig, ax = plt.subplots(figsize=(3.5, 2.8))
sns.boxplot(data=plot_df, x="model_display", y="r2", order=model_order,
            fliersize=0, ax=ax, boxprops=dict(facecolor="none", edgecolor="black"))
sns.stripplot(data=plot_df, x="model_display", y="r2", order=model_order,
              palette=dict(zip(model_order, colors_order)),
              size=2, alpha=0.2, ax=ax, jitter=0.2)

if len(pvals_adj):
    ymax = plot_df["r2"].max()
    ymin = plot_df["r2"].min()
    offset = (ymax - ymin) * 0.05
    for m, pval_adj in zip(test_models, pvals_adj):
        stars = "***" if pval_adj < 0.001 else "**" if pval_adj < 0.01 else "*" if pval_adj < 0.05 else ""
        if stars:
            xpos = model_order.index(model_name_map.get(m, m))
            ypos = plot_df[plot_df["model"] == m]["r2"].max() + offset
            ax.text(xpos, ypos, stars, ha="center", va="bottom", fontsize=12, color="red", weight="bold")

ax.set_xlabel("")
ax.set_ylabel("R² (evaluation data)")
ax.tick_params(axis="x", rotation=45)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
ax.spines[["right", "top"]].set_visible(False)
ax.margins(y=0.2, x=0.1)
plt.tight_layout()
p2 = os.path.join(rr_folder, f"performance_comparison_theta{theta}.png")
plt.savefig(p2, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {p2}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("DONE")
print("="*70)
print(f"Dataset : {dataset}")
print(f"Theta   : {theta}")
print(f"Methods : {grns}")
print(f"\nGene assignment (inference-side CV):")
for m, cnt in model_counts.items():
    print(f"  {m:20s}: {cnt}")
print(f"\nMean R² on evaluation data:")
for m in ["ensemble_custom"] + grns:
    vals = [r for r in eval_scores[m].values() if r > -np.inf]
    print(f"  {m:22s}: {np.mean(vals):.4f}" if vals else f"  {m:22s}: N/A")
print(f"\nOutputs in: {rr_folder}")
