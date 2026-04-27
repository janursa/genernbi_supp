#!/bin/bash
# Submit consensus building jobs for all 10 experimental datasets.
# Outputs → temp/experimental_grn/prior/
#
# Usage: bash temp/experimental_grn/consensus/submit_consensus.sh

REPO=/home/jnourisa/projs/ongoing/task_grn_inference
SCRIPT=$REPO/temp/experimental_grn/consensus/build_consensus.py
LOGS=$REPO/logs

mkdir -p $LOGS

DATASETS=(op ibd_uc ibd_cd 300BCG nakatake norman replogle parsebioscience xaira_HEK293T xaira_HCT116)

for ds in "${DATASETS[@]}"; do
    echo "Submitting consensus for $ds"
    sbatch \
        --job-name=cons_${ds} \
        --output=$LOGS/cons_${ds}_%j.out \
        --error=$LOGS/cons_${ds}_%j.err \
        --time=4:00:00 \
        --mem=120G \
        --cpus-per-task=4 \
        --partition=cpu \
        --wrap="cd $REPO && \
            source /home/jnourisa/miniconda3/etc/profile.d/conda.sh && \
            conda activate py10 && \
            python $SCRIPT --dataset $ds"
done
