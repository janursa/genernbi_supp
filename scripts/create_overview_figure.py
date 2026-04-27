"""
Create overview figure from combined results.
This script processes the combined trace.csv and score_uns.yaml files,
then calls the R script to generate the summary figure.

This follows the same logic as process_results.ipynb "Overview of performance" section.
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import argparse

# Add grn_benchmark to path and load environment
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.helper import load_env, surrogate_names, read_yaml

env = load_env()
TASK_GRN_INFERENCE_DIR = env['TASK_GRN_INFERENCE_DIR']
GRN_BENCHMARK_DIR = env['geneRNBI_DIR']

# Add task_grn_inference to path; clear cached src so it re-resolves to task_grn_inference's src
sys.path.insert(0, TASK_GRN_INFERENCE_DIR)
sys.modules.pop('src', None)
from src.utils.config import DATASETS, METHODS, METRICS, DATASET_INFO
from src.utils.util import compute_overall_scores


def process_scores_from_yaml(score_file):
    """Process score_uns.yaml the same way as read_yaml does."""
    with open(score_file, 'r') as f:
        scores_data = yaml.safe_load(f)
    
    # Convert to rows
    rows = []
    for entry in scores_data:
        if entry is None or 'missing' in str(entry):
            continue
        
        dataset_id = entry.get('dataset_id', '')
        method_id = entry.get('method_id', '')
        metric_ids = entry.get('metric_ids', [])
        metric_values = entry.get('metric_values', [])
        
        for metric, value in zip(metric_ids, metric_values):
            if value != "None":
                try:
                    rows.append({
                        'dataset': dataset_id,
                        'model': method_id,
                        'metric': metric,
                        'value': float(value)
                    })
                except (ValueError, TypeError):
                    pass
    
    df = pd.DataFrame(rows)
    # Pivot to get the same format as read_yaml
    scores_all = df.pivot_table(index=['dataset', 'model'], columns='metric', values='value').reset_index()
    
    return scores_all


def process_scores_from_csv(score_file):
    """Process all_scores.csv for local runs."""
    scores_all = pd.read_csv(score_file)
    # Rename 'method' to 'model' to match expected format
    if 'method' in scores_all.columns:
        scores_all = scores_all.rename(columns={'method': 'model'})
    return scores_all


def parse_duration(duration_str):
    """Parse duration string like '10m 20s' or '1h 30m' to hours."""
    if pd.isna(duration_str):
        return 0
    
    duration_str = str(duration_str).strip()
    total_seconds = 0
    
    parts = duration_str.split()
    for part in parts:
        if 'h' in part:
            total_seconds += float(part.replace('h', '')) * 3600
        elif 'm' in part:
            total_seconds += float(part.replace('m', '')) * 60
        elif 's' in part:
            total_seconds += float(part.replace('s', ''))
    
    return total_seconds / 3600


def convert_to_gb(value):
    """Convert memory values to GB."""
    if pd.isna(value):
        return 0
    
    value = str(value).strip()
    unit_to_bytes = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "B": 1
    }
    
    try:
        parts = value.split()
        if len(parts) == 2:
            num, unit = parts
            num = float(num)
            unit = unit.upper()
            if unit in unit_to_bytes:
                return num * unit_to_bytes[unit] / (1024**3)
        return float(value)
    except Exception:
        return 0


def process_trace_to_csv(trace_file):
    """
    Process trace file to create a CSV with method as index and resource columns.
    Only uses 'op' dataset traces as requested.
    Specifically captures the actual GRN inference process, not metric evaluations.
    """
    trace_df = pd.read_csv(trace_file, sep='\t')
    
    # The actual GRN inference for op dataset has pattern: (op.METHOD)
    # The metric evaluation has pattern: (METHOD_op.METRIC)
    # We want the inference, not the evaluation!
    
    print(f"   Total trace entries: {len(trace_df)}")
    
    resource_data = []
    
    for method in METHODS:
        # Look for the actual GRN inference process
        # Pattern: run_grn_inference:run_wf:bench:runEachWf:METHOD:processWf:METHOD_process (op.METHOD)
        inference_pattern = f'run_grn_inference.*:{method}:processWf:{method}_process \\(op\\.{method}\\)'
        method_inference = trace_df[trace_df['name'].str.contains(inference_pattern, regex=True, na=False)]
        
        if len(method_inference) == 0:
            # Try alternative patterns
            # Sometimes it might be (op.method) without the full path
            alt_pattern = f'{method}_process \\(op\\.{method}\\)$'
            method_inference = trace_df[trace_df['name'].str.contains(alt_pattern, regex=True, na=False)]
        
        if len(method_inference) == 0:
            print(f"   Warning: No inference trace found for {method}")
            continue
        
        if len(method_inference) > 1:
            print(f"   Warning: Multiple inference traces found for {method}, using first one")
        
        # Get the values from the inference process
        row = method_inference.iloc[0]
        
        duration = parse_duration(row['duration'])
        peak_rss = convert_to_gb(row['peak_rss'])
        
        # Handle CPU usage
        cpu_val = row['%cpu']
        if pd.notna(cpu_val):
            if isinstance(cpu_val, str):
                cpu_usage = float(cpu_val.replace('%', ''))
            else:
                cpu_usage = float(cpu_val)
        else:
            cpu_usage = 0
        
        resource_data.append({
            'method': method,
            'Duration (hour)': duration,
            'Peak memory (GB)': peak_rss,
            'CPU Usage (%)': cpu_usage
        })
        
        print(f"   {method}: duration={duration:.2f}h, memory={peak_rss:.1f}GB, cpu={cpu_usage:.1f}%")
    
    df_resources = pd.DataFrame(resource_data)
    df_resources = df_resources.set_index('method')
    
    # Handle duplicates if any
    if df_resources.index.duplicated().any():
        print(f"   Warning: Found duplicate methods in resources, taking max")
        df_resources = df_resources.groupby(df_resources.index).max()
    
    print(f"\n   Processed resources for {len(df_resources)} methods")
    
    return df_resources


def main(local_run=False, methods=None, datasets=None):
    
    # Get paths from environment
    results_folder = env['RESULTS_DIR']
    
    combined_dir = Path(results_folder) / 'all_new'
    trace_file = combined_dir / 'trace.csv'
    
    # Choose score file based on mode
    if local_run:
        score_file = Path(results_folder) / 'all_scores.csv'
    else:
        score_file = combined_dir / 'score_uns.yaml'
    
    print("=" * 80)
    print("Creating Overview Figure")
    print("=" * 80)
    print(f"\nUsing base directory: {TASK_GRN_INFERENCE_DIR}")
    print(f"Results folder: {results_folder}")
    print(f"Combined directory: {combined_dir}")
    print(f"Mode: {'LOCAL RUN' if local_run else 'AWS'}")
    print(f"Score file: {score_file}")
    
    # Step 1: Process trace file (only op dataset)
    print("\n1. Processing trace data (op dataset only)...")
    df_res = process_trace_to_csv(trace_file)
    
    # Step 2: Process scores (all datasets)
    print("\n2. Processing scores (all datasets)...")
    if local_run:
        print("   Using local mode: reading from all_scores.csv...")
        scores_all = process_scores_from_csv(score_file)
    else:
        print("   Using AWS mode: reading from score_uns.yaml...")
        scores_all = process_scores_from_yaml(score_file)
    
    # Filter by methods if specified
    if methods is not None:
        scores_all = scores_all[scores_all['model'].isin(methods)]
    
    # Filter by datasets if specified
    if datasets is not None:
        scores_all = scores_all[scores_all['dataset'].isin(datasets)]
    
    print(f"   Loaded scores for {len(scores_all)} method-dataset combinations")
    print(f"   Datasets: {sorted(scores_all['dataset'].unique().tolist())}")
    print(f"   Methods: {sorted(scores_all['model'].unique().tolist())}")
    
    # Step 2.5: Load metrics that passed criteria from metrics_applicibility
    print("\n2.5. Loading metrics that passed applicability criteria...")
    metrics_kept_file = Path(results_folder) / 'metrics_kept_per_dataset.yaml'
    
    if not metrics_kept_file.exists():
        raise FileNotFoundError(f"Metrics kept file not found: {metrics_kept_file}")
    else:
        with open(metrics_kept_file, 'r') as f:
            metrics_kept_per_dataset = yaml.safe_load(f)
        
        # Get metrics that passed for at least one dataset
        all_kept_metrics = set()
        for dataset, metric_list in metrics_kept_per_dataset.items():
            all_kept_metrics.update(metric_list)
        
        print(f"   Loaded metrics that passed applicability criteria: {len(all_kept_metrics)} metrics")
        print(f"   Metrics: {sorted(all_kept_metrics)}")
    
    
    # Step 3: Process scores - EXACT LOGIC FROM NOTEBOOK
    print("\n3. Creating summary dataframe...")
    
    # Get all metrics from METRICS
    metrics = [m for m in scores_all.columns.tolist() if m in METRICS]
    print(f"   Using all METRICS for display: {len(metrics)} metrics")
    
    # Get all METRICS that passed applicability for ranking
    final_metrics = [m for m in METRICS if m in all_kept_metrics]
    print(f"   Using applicability-filtered METRICS for ranking: {len(final_metrics)} metrics")
    print(f"   Ranking metrics: {final_metrics}")
    
    # Build modality groups for balanced ranking
    modality_groups = {}
    for dataset in DATASETS:
        if dataset not in scores_all['dataset'].unique():
            continue
        modality = DATASET_INFO.get(dataset, {}).get('Modality', 'Transcriptomics')
        modality_groups.setdefault(modality, []).append(dataset)

    df_scores = compute_overall_scores(scores_all, final_metrics, DATASETS, modality_groups=modality_groups)

    final_metrics_cols = [m for m in final_metrics if m in df_scores.columns]
    dataset_cols = [c for c in df_scores.columns if c in DATASETS]  # display only non-IBD datasets
    print(f"   Overall scores calculated using applicability-filtered metrics ({len(final_metrics_cols)} metrics) + datasets ({len(dataset_cols)} datasets)")
    
    # Keep track of which methods are in the filtered scores
    methods_in_scores = df_scores.index.tolist()
    
    # Merge scores with resources
    df_summary = pd.concat([df_scores, df_res], axis=1)
    
    # Filter to only include methods that were in the filtered scores
    df_summary = df_summary[df_summary.index.isin(methods_in_scores)]
    
    df_summary = df_summary.fillna(0)
    df_summary.index.name = 'method_name'
    df_summary = df_summary.reset_index()
    
    # Sort by overall score
    df_summary = df_summary.sort_values(by='overall_score', ascending=False)
    
    # Map method names
    df_summary['method_name'] = df_summary['method_name'].map(lambda x: surrogate_names.get(x, x))
    
    # Add user complexity
    df_summary['User-friendly'] = df_summary['method_name'].map({
        'Scenic+': 1,
        'GRNBoost2': 8,
        'Positive Ctrl': 10,
        'Pearson Corr.': 10,
        'Spearman Corr.': 10,
        'CellOracle': 6,
        'Portia': 9,
        'scGLUE': 6,
        'Scenic': 7,
        'FigR': 6,
        'PPCOR': 7,
        'Negative Ctrl': 10,
        'GRaNIE': 6,
        'scPRINT': 5,
        'Geneformer': 5,
        'scGPT': 3,
    })
    
    df_summary['Complexity'] = df_summary['User-friendly'].max() - df_summary['User-friendly']
    
    # Map column names to surrogate names
    df_summary.columns = [surrogate_names.get(col, col) for col in df_summary.columns]
    
    df_summary = df_summary.fillna(0)
    
    # Add log-transformed resource metrics
    df_summary['memory_log'] = np.log(df_summary['Peak memory (GB)'] + 1)
    df_summary['memory_log'] = np.max(df_summary['memory_log']) - df_summary['memory_log']
    
    df_summary['complexity_log'] = np.log(df_summary['Complexity'] + 1)
    df_summary['complexity_log'] = np.max(df_summary['complexity_log']) - df_summary['complexity_log']
    
    df_summary['duration_log'] = np.log(df_summary['Duration (hour)'] + 1)
    df_summary['duration_log'] = np.max(df_summary['duration_log']) - df_summary['duration_log']
    
    df_summary['duration_str'] = df_summary['Duration (hour)'].round(1).astype(str)
    df_summary['memory_str'] = df_summary['Peak memory (GB)'].round(1).astype(str)
    
    # Step 4: Save summary
    summary_file = f"{results_folder}/summary.tsv"
    df_summary.to_csv(summary_file, sep='\t', index=False)
    print(f"\n4. Saved summary to: {summary_file}")
    print(f"   Summary has {len(df_summary)} methods and {len(df_summary.columns)} columns")
    
    # Step 5: Call R script
    print("\n5. Calling R script to create figure...")
    r_script = f'{GRN_BENCHMARK_DIR}/src/summary_figure.R'
    summary_figure = f"{results_folder}/summary_figure"
    
    import subprocess
    cmd = f"Rscript {r_script} {summary_file} {summary_figure}"
    print(f"   Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"\n✓ Success! Figure saved to:")
        print(f"   - {summary_figure}.pdf")
        print(f"   - {summary_figure}.png")
    else:
        print(f"\n✗ Error running R script:")
        print(result.stderr)
        print("\nStdout:")
        print(result.stdout)
        return 1
    # also save it to docs folder
    DOCS_IMAGES_DIR = Path(env['DOCS_IMAGES_DIR'])
    doc_png = DOCS_IMAGES_DIR / 'summary_figure.png'
    cmd = f"cp {summary_figure}.png {doc_png}"
    subprocess.run(cmd, shell=True)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create overview figure from combined results'
    )
    parser.add_argument(
        '--aws_run',
        dest='local_run',
        action='store_false',
        help='Use AWS run mode - read scores from score_uns.yaml instead of all_scores.csv'
    )
    parser.set_defaults(local_run=True)
    parser.add_argument(
        '--methods',
        nargs='+',
        default=None,
        help='List of methods to include (space-separated). If not provided, uses METHODS from config.'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='List of datasets to include (space-separated). If not provided, uses DATASETS from config.'
    )
    
    args = parser.parse_args()
    
    # If methods not provided, use METHODS from config
    methods_to_use = args.methods if args.methods else METHODS
    
    # If datasets not provided, use DATASETS from config
    datasets_to_use = args.datasets if args.datasets else DATASETS
    
    exit(main(local_run=args.local_run, methods=methods_to_use, datasets=datasets_to_use))

