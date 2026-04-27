#!/bin/bash
# Submit bulk_vs_sc experiment jobs for one or more datasets.
# Usage:
#   ./wrapper_submit.sh                          # default: xaira_HEK293T xaira_HCT116 replogle
#   ./wrapper_submit.sh replogle                 # specific dataset
#   INFER_GRN=true ./wrapper_submit.sh replogle  # run GRN inference + metrics
#
# Env vars (defaults: INFER_GRN=false, METRICS=true):
#   INFER_GRN=true/false    METRICS=true/false

DATASETS="${@:-xaira_HEK293T xaira_HCT116 replogle}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRNBI_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

INFER_GRN="${INFER_GRN:-false}"
METRICS="${METRICS:-true}"

for ds in $DATASETS; do
    sbatch \
        --job-name=bulk_vs_sc_${ds} \
        --output="${GRNBI_DIR}/logs/bulk_vs_sc_${ds}_%j.out" \
        --error="${GRNBI_DIR}/logs/bulk_vs_sc_${ds}_%j.err" \
        --ntasks=1 \
        --cpus-per-task=20 \
        --time=20:00:00 \
        --mem=250GB \
        --partition=cpu \
        --mail-type=END,FAIL \
        --mail-user=jalil.nourisa@gmail.com \
        --export=ALL,DATASET=${ds},INFER_GRN=${INFER_GRN},METRICS=${METRICS} \
        "${SCRIPT_DIR}/run_bulk_vs_sc.sh"
    echo "Submitted: $ds (INFER_GRN=${INFER_GRN}, METRICS=${METRICS})"
done
