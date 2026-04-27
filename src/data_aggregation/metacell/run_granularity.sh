#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --mem=250GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

cd /home/jnourisa/projs/ongoing/geneRNBI
source env.sh
/home/jnourisa/miniconda3/envs/py10/bin/python src/stability_analysis/pseudobulk/granularity/script.py
