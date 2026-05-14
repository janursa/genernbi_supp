#!/bin/bash

set -e

DATASETS=('soundlife' 'norman' 'MSCIC')
REG_TYPES=('ridge' 'GB')

mkdir -p logs

for dataset in "${DATASETS[@]}"; do
    for reg_type in "${REG_TYPES[@]}"; do
        echo "Submitting: dataset=$dataset reg_type=$reg_type"
        sbatch --job-name="gbr_${dataset}_${reg_type}" run.sh "$dataset" "$reg_type"
    done
done
