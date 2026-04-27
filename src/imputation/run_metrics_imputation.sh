#!/bin/bash
# Runs metrics on all imputation predictions and regenerates metrics_op.csv.
# Called by wrapper_imputation.sh after inference jobs complete.
#
# Usage: bash src/stability_analysis/imputation/run_metrics_imputation.sh <dataset>

set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source env.sh

DATASET="${1:-op}"
OUTPUT_DIR="${RESULTS_DIR}/experiment/imputation"
OUTPUT_FILE="${OUTPUT_DIR}/metrics_${DATASET}.csv"
ALL_METHODS=("pearson_corr" "grnboost" "ppcor" "portia" "scenic")
IMPUTATION_METHODS=("original" "knn" "magic")

score_files=()

for method in "${ALL_METHODS[@]}"; do
    for imputation in "${IMPUTATION_METHODS[@]}"; do
        pred_file="${OUTPUT_DIR}/${DATASET}_${imputation}_${method}_prediction.h5ad"
        score_file="${OUTPUT_DIR}/${DATASET}_${imputation}_${method}_prediction_score.h5ad"

        if [ ! -f "$pred_file" ]; then
            echo "WARNING: prediction not found, skipping: $pred_file"
            continue
        fi

        if [ -f "$score_file" ]; then
            echo "Score already exists, reusing: $score_file"
        else
            echo "Computing metrics for $DATASET / $imputation / $method ..."
            cd "$TASK_GRN_INFERENCE_DIR" && bash src/metrics/all_metrics/run_local.sh \
                --dataset "$DATASET" \
                --prediction "$pred_file" \
                --score "$score_file"
            cd - > /dev/null
        fi

        score_files+=("$score_file")
    done
done

echo "Aggregating ${#score_files[@]} score files into $OUTPUT_FILE ..."
python - "$OUTPUT_FILE" "${score_files[@]}" <<'EOF'
import anndata as ad
import pandas as pd
import sys
import os

output_file = sys.argv[1]
score_files = sys.argv[2:]

results_all = []
for f in score_files:
    adata = ad.read_h5ad(f)
    metrics_keys = adata.uns['metric_ids']
    metrics_values = adata.uns['metric_values']
    df = pd.DataFrame({k: [v] for k, v in zip(metrics_keys, metrics_values)})

    # Filename format: {dataset}_{imputation}_{method}_prediction_score.h5ad
    basename = os.path.basename(f).replace('_score.h5ad', '').replace('_prediction', '')
    parts = basename.split('_')
    imputation_method = parts[1]
    inference_method  = '_'.join(parts[2:])

    df['imputation_method'] = imputation_method
    df['inference_method']  = inference_method
    df['prediction']        = f
    results_all.append(df)

results = pd.concat(results_all, ignore_index=True)
results.to_csv(output_file, index=False)
print(f"Saved {len(results)} rows to {output_file}")
print(results[['imputation_method', 'inference_method']].to_string())
EOF

echo "Done!"
