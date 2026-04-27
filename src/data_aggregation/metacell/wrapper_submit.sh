#!/bin/bash
# Submit granularity experiment jobs for one or more datasets.
# Usage:
#   ./wrapper_submit.sh                          # default: op 300BCG ibd_cd ibd_uc norman
#   ./wrapper_submit.sh 300BCG norman            # specific datasets
#   INFER_GRN=true ./wrapper_submit.sh op        # run GRN inference + metrics for op
#
# Env vars passed to script.py (all default false except METRICS=true):
#   PSEUDOBULK=true/false   INFER_GRN=true/false   METRICS=true/false

DATASETS="${@:-op 300BCG ibd_cd ibd_uc norman}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRNBI_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

INFER_GRN="${INFER_GRN:-false}"
PSEUDOBULK="${PSEUDOBULK:-false}"
METRICS="${METRICS:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

for ds in $DATASETS; do
    sbatch \
        --job-name=granularity_${ds} \
        --output="${GRNBI_DIR}/logs/granularity_${ds}_%j.out" \
        --error="${GRNBI_DIR}/logs/granularity_${ds}_%j.err" \
        --ntasks=1 \
        --cpus-per-task=20 \
        --time=20:00:00 \
        --mem=250GB \
        --partition=cpu \
        --mail-type=END,FAIL \
        --mail-user=jalil.nourisa@gmail.com \
        --export=ALL,DATASET=${ds},INFER_GRN=${INFER_GRN},PSEUDOBULK=${PSEUDOBULK},METRICS=${METRICS},SKIP_EXISTING=${SKIP_EXISTING} \
        "${SCRIPT_DIR}/run_granularity.sh"
    echo "Submitted: $ds (INFER_GRN=${INFER_GRN}, METRICS=${METRICS}, SKIP_EXISTING=${SKIP_EXISTING})"
done
