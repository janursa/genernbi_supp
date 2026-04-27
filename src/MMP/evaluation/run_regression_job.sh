#!/bin/bash
#SBATCH --job-name=mmp_regression
#SBATCH --output=/home/jnourisa/projs/ongoing/task_grn_inference/logs/mmp_regression_%j.out
#SBATCH --error=/home/jnourisa/projs/ongoing/task_grn_inference/logs/mmp_regression_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=3:00:00
#SBATCH --mem=80GB
#SBATCH --partition=cpu

set -euo pipefail

DATASET="$1"   # multiome or dogma
METHOD="$2"    # grnboost | scenicplus | celloracle | ppcor | granie

REPO=/home/jnourisa/projs/ongoing/task_grn_inference
cd "$REPO"

RESULTS_MMP="${REPO}/resources/results/MMP"
EVAL_DIR="${RESULTS_MMP}/evaluation"

prediction="${RESULTS_MMP}/MMP_${DATASET}_${METHOD}_grn.h5ad"
evaluation_data="${RESULTS_MMP}/MMP_${DATASET}_rna_processed.h5ad"
regulators_consensus="${EVAL_DIR}/regulators_consensus_MMP_${DATASET}.json"
score="${EVAL_DIR}/MMP_${DATASET}_${METHOD}_regression_score.h5ad"
tf_all="${REPO}/resources/grn_benchmark/prior/tf_all.csv"

echo "Dataset:  $DATASET"
echo "Method:   $METHOD"
echo "Prediction: $prediction"
echo "Score out:  $score"

python src/metrics/all_metrics/script.py \
    --prediction           "$prediction" \
    --evaluation_data      "$evaluation_data" \
    --regulators_consensus "$regulators_consensus" \
    --tf_all               "$tf_all" \
    --layer                lognorm \
    --reg_type             ridge \
    --num_workers          20 \
    --apply_tf \
    --cv_groups            CT_Major \
    --score                "$score"

echo "Done: $score"
