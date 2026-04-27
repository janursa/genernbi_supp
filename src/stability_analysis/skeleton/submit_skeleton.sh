#!/bin/bash
# Submit one sbatch job per dataset × method for the skeleton-filtering experiment.
# Usage: bash src/stability_analysis/skeleton/submit_skeleton.sh

DATASETS=(ibd_cd ibd_uc)

for dataset in "${DATASETS[@]}"; do
    source env.sh 2>/dev/null
    methods=($(ls $RESULTS_DIR/${dataset}/*.prediction.h5ad | xargs -n1 basename | cut -d. -f2 | sort -u))
    for method in "${methods[@]}"; do
        echo "Submitting: dataset=$dataset  method=$method"
        sbatch --job-name=skel_${dataset}_${method} \
               src/stability_analysis/skeleton/experiment_skeleton.sh \
               $dataset $method
    done
done
