#!/bin/bash
# Submit regression jobs for MMP multiome + dogma × 5 methods.
# Run from anywhere; cd to task_grn_inference is handled per job.
#
# Usage: bash submit_regression.sh

set -euo pipefail

METHODS=(grnboost scenicplus celloracle ppcor granie)
DATASETS=(multiome dogma)
JOB_SCRIPT=/home/jnourisa/projs/ongoing/geneRNBI/src/MMP/evaluation/run_regression_job.sh

echo "Submitting MMP regression jobs..."
for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        score=/home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/evaluation/MMP_${dataset}_${method}_regression_score.h5ad
        if [[ -f "$score" ]]; then
            echo "  [SKIP] ${dataset}/${method} — score already exists"
            continue
        fi
        job_id=$(sbatch --parsable "$JOB_SCRIPT" "$dataset" "$method")
        echo "  Submitted ${dataset}/${method} → job $job_id"
    done
done
echo "Done submitting."
