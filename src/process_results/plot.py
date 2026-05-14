"""
Plotting functions for GRN benchmark results.
"""

import sys
import os
import argparse
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Arial"


def plot_raw_scores(
    scores_all: pd.DataFrame,
    datasets: list[str],
    output_dirs: list[str],
    surrogate_names: dict,
    metrics: list[str],
):
    """
    Produce one heatmap per dataset and save to every directory in output_dirs.

    Parameters
    ----------
    scores_all   : DataFrame with columns 'model', 'dataset', + metric columns
    datasets     : ordered list of dataset ids to plot
    output_dirs  : list of directory paths to save figures into
    surrogate_names : display-name mapping for metrics and methods
    metrics      : ordered list of metric column names to include
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.helper import plot_heatmap

    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)

    for dataset in datasets:
        scores = scores_all[scores_all["dataset"] == dataset].copy()
        scores = scores.loc[:, ~scores.isna().all()]
        if scores.empty:
            print(f"  No data for dataset {dataset} — skipping")
            continue

        scores = scores.set_index("model").drop(columns="dataset", errors="ignore")
        scores = scores[[c for c in metrics if c in scores.columns]]
        if scores.empty:
            print(f"  No metric columns for dataset {dataset} — skipping")
            continue

        scores.columns = scores.columns.map(lambda n: surrogate_names.get(n, n))
        scores.index = scores.index.map(lambda n: surrogate_names.get(n, n))
        scores = scores.astype(float)

        ranks = scores.rank(axis=0, ascending=False, method="min")
        scores["_rank"] = ranks.mean(axis=1, skipna=True)
        scores = scores.sort_values("_rank").drop(columns=["_rank"])

        n_rows, n_cols = scores.shape
        fig, ax = plt.subplots(1, 1, figsize=(max(n_cols * 0.6, 4), max(n_rows * 0.4, 3)))
        plot_heatmap(scores, ax=ax, cmap="viridis")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
        ax.set_ylabel("")
        plt.suptitle(surrogate_names.get(dataset, dataset), y=1.01, fontsize=14, weight="bold")
        plt.tight_layout()

        for out_dir in output_dirs:
            fpath = os.path.join(out_dir, f"raw_scores_{dataset}.png")
            fig.savefig(fpath, dpi=150, transparent=True, bbox_inches="tight")
            print(f"  Saved: {fpath}")

        plt.close(fig)


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.helper import load_env, surrogate_names
    env = load_env()

    task_dir = env["TASK_GRN_INFERENCE_DIR"]
    grn_dir = env["genernbi_supp_DIR"]
    sys.path.insert(0, task_dir)
    from task_grn_inference.src.utils.config import DATASETS_METRICS, METHODS, METRICS

    parser = argparse.ArgumentParser(description="Plot GRN benchmark results")
    parser.add_argument("--scores_file",
                        default=f"{env['RESULTS_DIR']}/all_scores.csv",
                        help="Path to all_scores.csv")
    parser.add_argument("--output_dir_local",
                        default=f"{grn_dir}/output/figs/raw_scores",
                        help="Local output directory for figures")
    parser.add_argument("--output_dir_docs",
                        default=f"{task_dir}/docs/source/images",
                        help="Docs output directory for figures")
    args = parser.parse_args()

    from src.process_results.helper import load_scores
    scores_all = load_scores(args.scores_file, METRICS, METHODS)

    datasets = list(DATASETS_METRICS.keys())
    output_dirs = [args.output_dir_local, args.output_dir_docs]
    plot_raw_scores(scores_all, datasets, output_dirs, surrogate_names, METRICS)
    print("Done.")


if __name__ == "__main__":
    main()
