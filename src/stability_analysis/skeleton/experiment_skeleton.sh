#!/bin/bash
#SBATCH --job-name=skeleton_exp
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=jalil.nourisa@gmail.com   

set -e

dataset=$1
method=$2
if [ -z "$dataset" ] || [ -z "$method" ]; then
    echo "Usage: $0 <dataset> <method>"
    exit 1
fi
source env.sh
python src/stability_analysis/skeleton/script.py --dataset $dataset --method $method
