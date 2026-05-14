"""
Skeleton-filtering experiment.

For each method × dataset:
  1. Load the original GRN prediction (.h5ad)
  2. Filter edges to only those present in the skeleton (TF→gene pairs)
  3. Save filtered GRN to tmp/
  4. Run all_metrics on the filtered GRN
  5. Collect scores alongside the baseline (from all_scores.csv)

Results are written to $RESULTS_DIR/experiment/skeleton/{dataset}/{dataset}-scores.csv
"""

import os
import sys
import glob
import argparse
import pandas as pd
import anndata as ad
from concurrent.futures import ProcessPoolExecutor, as_completed

env = os.environ

sys.path.insert(0, env['UTILS_DIR'])
sys.path.insert(0, env['METRICS_DIR'])

from util import naming_convention, process_links
from src.params import get_par

sys.path.insert(0, env['genernbi_supp_DIR'])

# ------------------------------------------------------------------ helpers --

def load_skeleton(skeleton_path: str) -> pd.DataFrame:
    """Return skeleton as a DataFrame with (source, target) columns."""
    return pd.read_csv(skeleton_path, usecols=["source", "target"])


def filter_grn(prediction: pd.DataFrame, skeleton: pd.DataFrame) -> pd.DataFrame:
    """Keep only edges whose (source, target) pair exists in the skeleton."""
    skeleton_marked = skeleton.copy()
    skeleton_marked["_keep"] = True
    merged = prediction.merge(skeleton_marked, on=["source", "target"], how="left")
    return prediction[merged["_keep"].fillna(False).values].reset_index(drop=True)


def main_metrics(par):
    from all_metrics.helper import main as _main_metrics
    return _main_metrics(par)


# --------------------------------------------------------------- per-method worker --

def process_method(method, par, skeleton):
    pred_path = f"{par['grns_dir']}/{naming_convention(par['dataset'], method)}"
    if not os.path.exists(pred_path):
        print(f"Skipping {pred_path} (not found)", flush=True)
        return method, None

    net = ad.read_h5ad(pred_path)
    prediction = pd.DataFrame(net.uns["prediction"])
    prediction = process_links(prediction, par={"max_n_links": 50_000})

    n_before = len(prediction)
    prediction_filtered = filter_grn(prediction, skeleton)
    n_after = len(prediction_filtered)
    print(f"{method}: {n_before} → {n_after} edges after skeleton filtering", flush=True)

    if len(prediction_filtered) == 0:
        print(f"  Skipping {method}: no edges remain after filtering", flush=True)
        return method, None

    tmp_path = f"{par['write_dir']}/tmp/{par['dataset']}_{method}_skeleton.h5ad"
    net_filtered = ad.AnnData(
        X=None,
        uns={
            "method_id":  net.uns.get("method_id", method),
            "dataset_id": net.uns.get("dataset_id", par["dataset"]),
            "prediction": prediction_filtered[["source", "target", "weight"]].astype(str),
        },
    )
    net_filtered.write(tmp_path)

    par_eval = {**par, "prediction": tmp_path}
    score = main_metrics(par_eval)
    score.index = [method]
    print(f"  Scores computed for {method}", flush=True)
    return method, score


# --------------------------------------------------------------- main logic --

def main(par):
    os.makedirs(par["write_dir"], exist_ok=True)
    os.makedirs(f"{par['write_dir']}/tmp/", exist_ok=True)

    skeleton = load_skeleton(par["skeleton"])
    print(f"Skeleton loaded: {len(skeleton):,} edges", flush=True)

    num_workers = min(len(par["methods"]), int(env.get("SLURM_CPUS_PER_TASK", 8)))
    print(f"Running {len(par['methods'])} methods with {num_workers} parallel workers", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_method, method, par, skeleton): method
            for method in par["methods"]
        }
        for future in as_completed(futures):
            method, score = future.result()
            if score is not None:
                results[method] = score

    if results:
        df_all = pd.concat(results.values())
        method_name = par["methods"][0] if len(par["methods"]) == 1 else "all"
        out_csv = f"{par['write_dir']}/{par['dataset']}-{method_name}-skeleton-scores.csv"
        df_all.to_csv(out_csv)
        print(f"Saved: {out_csv}", flush=True)
    else:
        print("No scores to save.", flush=True)


# --------------------------------------------------------------- entry point --

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    par = get_par(args.dataset)

    grns_dir = f"{env['RESULTS_DIR']}/{args.dataset}/"
    skeleton_path = f"{env['RESULTS_DIR']}/experiment/skeleton/{args.dataset}/skeleton.csv"
    print(f"Using skeleton: {skeleton_path}", flush=True)

    par = {
        **par,
        "grns_dir":  grns_dir,
        "write_dir": f"{env['RESULTS_DIR']}/experiment/skeleton/{args.dataset}/",
        "methods":   [args.method],
        "dataset":   args.dataset,
        "skeleton":  skeleton_path,
    }

    main(par)
