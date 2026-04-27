#!/bin/bash
#SBATCH --job-name=global_grns
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

set -e

# Default dataset if not provided
DATASET=${1:-op}

source env.sh
python src/stability_analysis/global_grns/script.py --dataset $DATASET
