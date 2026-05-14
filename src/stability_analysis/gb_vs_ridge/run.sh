#!/bin/bash
#SBATCH --job-name=gb_vs_ridge
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

dataset=${1:-op}
reg_type=${2:-GB}

source env.sh
python src/stability_analysis/gb_vs_ridge/script.py --dataset "$dataset" --reg_type "$reg_type"
