#!/bin/bash
# Submits GRN inference jobs for new methods on imputed data,
# then submits a dependent metrics+aggregation job.
#
# Usage: bash src/stability_analysis/imputation/wrapper_imputation.sh

set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source env.sh

DATASET="op"
OUTPUT_DIR="${RESULTS_DIR}/experiment/imputation"
IMPUTATION_METHODS=("original" "knn" "magic")
NEW_METHODS=("grnboost" "ppcor" "portia" "scenic")

# grnboost already has original+knn — only submit magic
SKIP=("op_original_grnboost" "op_knn_grnboost")

job_ids=()

for method in "${NEW_METHODS[@]}"; do
    for imputation in "${IMPUTATION_METHODS[@]}"; do
        key="${DATASET}_${imputation}_${method}"
        pred_file="${OUTPUT_DIR}/${key}_prediction.h5ad"

        # Skip already-completed predictions
        if [[ " ${SKIP[*]} " =~ " ${DATASET}_${imputation}_${method} " ]]; then
            echo "Skipping $key (already exists)"
            continue
        fi
        if [ -f "$pred_file" ]; then
            echo "Skipping $key (prediction file already exists)"
            continue
        fi

        rna_file="${OUTPUT_DIR}/${DATASET}_${imputation}_rna.h5ad"
        if [ ! -f "$rna_file" ]; then
            echo "ERROR: RNA file not found: $rna_file"
            exit 1
        fi

        script_file="src/methods/${method}/run_local.sh"
        extra_args=""
        if [[ "$method" == "scenic" ]]; then
            extra_args="--temp_dir /tmp/scenic_${DATASET}_${imputation}"
        fi
        jid=$(cd "$TASK_GRN_INFERENCE_DIR" && sbatch "$script_file" \
                --rna "$rna_file" \
                --prediction "$pred_file" \
                $extra_args | awk '{print $4}')
        echo "Submitted $key → job $jid"
        job_ids+=("$jid")
    done
done

if [ ${#job_ids[@]} -eq 0 ]; then
    echo "No new inference jobs submitted — all predictions already exist."
    echo "Running metrics directly..."
    bash src/stability_analysis/imputation/run_metrics_imputation.sh "$DATASET"
    exit 0
fi

# Join job IDs for dependency
dep=$(IFS=:; echo "${job_ids[*]}")
echo "Submitting metrics job with dependency on: $dep"

jid_metrics=$(sbatch --dependency=afterok:$dep \
    --job-name=imputation_metrics \
    --output=logs/%j.out \
    --error=logs/%j.err \
    --ntasks=1 --cpus-per-task=10 --time=4:00:00 --mem=100GB --partition=cpu \
    --wrap="cd $(pwd) && bash src/stability_analysis/imputation/run_metrics_imputation.sh $DATASET" \
    | awk '{print $4}')

echo "Submitted metrics+aggregation job → $jid_metrics (runs after all inference jobs complete)"
