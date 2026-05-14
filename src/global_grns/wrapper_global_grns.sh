#!/bin/bash
# Submit global_grns evaluation jobs for one or more datasets.
# Usage (run from project root):
#   bash src/global_grns/wrapper_global_grns.sh                 # default datasets
#   bash src/global_grns/wrapper_global_grns.sh soundlife MSCIC # specific datasets

DATASETS="${@:-soundlife soundlife_vaccine MSCIC}"

for ds in $DATASETS; do
    job_id=$(sbatch \
        --job-name=global_grns_${ds} \
        --output="logs/global_grns_${ds}_%j.out" \
        --error="logs/global_grns_${ds}_%j.err" \
        --ntasks=1 \
        --cpus-per-task=20 \
        --time=2-00:00:00 \
        --mem=250GB \
        --partition=cpu \
        --mail-type=END,FAIL \
        --mail-user=jalil.nourisa@gmail.com \
        src/global_grns/experiment_global_grns.sh "$ds" \
        | awk '{print $NF}')
    echo "Submitted $ds: job $job_id"
done
