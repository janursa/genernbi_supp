#!/bin/bash
#SBATCH --job-name=ensemble_custom
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

source env.sh

# Usage:
#   sbatch experiment_ensemble_custom.sh --dataset op --theta 0.25
#   sbatch experiment_ensemble_custom.sh --dataset MSCIC --theta 0.25
#   sbatch experiment_ensemble_custom.sh --dataset replogle --theta 0.25

python src/ensemble_custom/script.py \
    --rr_folder "$RESULTS_DIR/experiment/ensemble_custom" \
    "$@"
