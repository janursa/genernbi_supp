#!/bin/bash
#SBATCH --job-name=imputation
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -euo pipefail

dataset=$1
run_imputation=${2:-false}
run_grn_inference=${3:-false}
run_metrics=${4:-false}
imputation_methods=("original" "knn" "magic")
inference_methods=('pearson_corr')

source env.sh
output_dir="${RESULTS_DIR}/experiment/imputation"
output_file="${output_dir}/metrics_${dataset}.csv"



if [ "$run_imputation" = true ]; then
    echo "Running imputations..."
    python src/stability_analysis/imputation/script.py \
    --dataset "$dataset" \
    --imputation_methods "${imputation_methods[@]}" \
    --output_dir "$output_dir"

fi

if [ "$run_grn_inference" = true ]; then
    for inference_method in "${inference_methods[@]}"; do
        echo "Running GRN inference with $inference_method ..."
        for imputation in "${imputation_methods[@]}"; do
            rna_file="${output_dir}/${dataset}_${imputation}_rna.h5ad"
            prediction_file="${output_dir}/${dataset}_${imputation}_${inference_method}_prediction.h5ad"

            if [ ! -f "$rna_file" ]; then
                echo "RNA file not found: $rna_file"
                exit 1
            fi

            script_file="src/methods/${inference_method}/run_local.sh"
        
            cd $TASK_GRN_INFERENCE_DIR && bash "$script_file" \
                --rna "$rna_file" \
                --prediction "$prediction_file" 

            
        done
    done
fi

if [ "$run_metrics" = true ]; then
    predictions=""
    for inference_method in "${inference_methods[@]}"; do
        for imputation in "${imputation_methods[@]}"; do
            predictions="${predictions} ${output_dir}/${dataset}_${imputation}_${inference_method}_prediction.h5ad"
        done
    done

    echo "Running metrics..."
    score_files=()

    for prediction in $predictions; do
        score_file="${output_dir}/$(basename "${prediction}" .h5ad)_score.h5ad"
        cd "$TASK_GRN_INFERENCE_DIR" && bash src/metrics/all_metrics/run_local.sh \
            --dataset "$dataset" \
            --prediction "$prediction" \
            --score "$score_file"
        score_files+=("$score_file")
    done

    echo "Aggregating results..."
    python - "$output_file" "${score_files[@]}" <<'EOF'
import anndata as ad
import pandas as pd
import sys
import re
import os

output_file = sys.argv[1]
score_files = sys.argv[2:]  # all remaining args are the score files

results_all = []
for f in score_files:
    adata = ad.read_h5ad(f)
    print(f"Processing: {f}")
    print(adata.uns)
    metrics_keys = adata.uns['metric_ids']
    metrics_values = adata.uns['metric_values']
    df = {k: [v] for k, v in zip(metrics_keys, metrics_values)}
    df = pd.DataFrame(df)

    # Extract imputation method and inference method from filename
    # Expected format: {dataset}_{imputation}_{inference_method}_prediction_score.h5ad
    basename = os.path.basename(f)
    # Remove _score.h5ad suffix
    basename = basename.replace('_score.h5ad', '')
    # Remove _prediction suffix
    basename = basename.replace('_prediction', '')
    
    # Split by underscore and extract components
    parts = basename.split('_')
    
    # The format should be: dataset_imputation_inference_method
    # e.g., op_original_grnboost or op_knn_pearson_corr
    if len(parts) >= 3:
        dataset = parts[0]
        imputation_method = parts[1]
        # inference method might have underscores (e.g., pearson_corr)
        inference_method = '_'.join(parts[2:])
    else:
        # Fallback
        dataset = parts[0] if len(parts) > 0 else "unknown"
        imputation_method = parts[1] if len(parts) > 1 else "unknown"
        inference_method = parts[2] if len(parts) > 2 else "unknown"
    
    df["imputation_method"] = imputation_method
    df["inference_method"] = inference_method
    df["prediction"] = f
    
    results_all.append(df)

results = pd.concat(results_all, ignore_index=True)
results.to_csv(output_file, index=False)
print(f"\nSaved results to {output_file}")
print(results.head())
EOF
fi

echo "Done!"