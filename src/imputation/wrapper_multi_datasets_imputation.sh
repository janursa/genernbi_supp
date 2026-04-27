#!/bin/bash
# Runs the full imputation experiment (imputation → GRN inference → metrics)
# for multiple datasets, chaining phases with SLURM dependencies.
#
# Usage: bash src/stability_analysis/imputation/wrapper_multi_datasets_imputation.sh

set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source env.sh

DATASETS=("norman" "300BCG" "ibd_cd" "ibd_uc")
SCRIPT="src/stability_analysis/imputation/experiment_imputation.sh"
WORKDIR=$(pwd)

for dataset in "${DATASETS[@]}"; do
    echo "=== Submitting pipeline for: $dataset ==="

    # Phase 1: imputation (memory/time intensive for large datasets)
    jid_imp=$(sbatch \
        --job-name=imp_${dataset} \
        --output=logs/%j.out \
        --error=logs/%j.err \
        --ntasks=1 --cpus-per-task=20 \
        --time=10:00:00 --mem=250GB \
        --partition=cpu \
        --wrap="cd ${WORKDIR} && bash ${SCRIPT} ${dataset} true false false" \
        | awk '{print $4}')
    echo "  Imputation     → job $jid_imp"

    # Phase 2: GRN inference (runs pearson_corr inline, after imputation)
    jid_grn=$(sbatch \
        --dependency=afterok:${jid_imp} \
        --job-name=grn_${dataset} \
        --output=logs/%j.out \
        --error=logs/%j.err \
        --ntasks=1 --cpus-per-task=20 \
        --time=20:00:00 --mem=250GB \
        --partition=cpu \
        --wrap="cd ${WORKDIR} && bash ${SCRIPT} ${dataset} false true false" \
        | awk '{print $4}')
    echo "  GRN inference  → job $jid_grn (after $jid_imp)"

    # Phase 3: metrics + aggregation (after GRN inference)
    jid_met=$(sbatch \
        --dependency=afterok:${jid_grn} \
        --job-name=metrics_${dataset} \
        --output=logs/%j.out \
        --error=logs/%j.err \
        --ntasks=1 --cpus-per-task=10 \
        --time=4:00:00 --mem=100GB \
        --partition=cpu \
        --wrap="cd ${WORKDIR} && bash ${SCRIPT} ${dataset} false false true" \
        | awk '{print $4}')
    echo "  Metrics        → job $jid_met (after $jid_grn)"
    echo ""
done
