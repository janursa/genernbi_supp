#!/bin/bash
#SBATCH --job-name=granie_dogma
#SBATCH --output=/home/jnourisa/projs/ongoing/genernbi_supp/src/MMP/logs/granie_dogma_%j.out
#SBATCH --error=/home/jnourisa/projs/ongoing/genernbi_supp/src/MMP/logs/granie_dogma_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -e
mkdir -p /home/jnourisa/projs/ongoing/genernbi_supp/src/MMP/logs
echo "=== GRaNIE MMP dogma ==="
echo "Start: $(date)"
cd /home/jnourisa/projs/ongoing/task_grn_inference

singularity exec \
  --bind /vol/projects:/vol/projects \
  --bind /home/jnourisa:/home/jnourisa \
  --bind /tmp:/tmp \
  --env RETICULATE_PYTHON=/usr/bin/python3 \
  --env R_FUTURE_GLOBALS_MAXSIZE=8589934592 \
  /home/jnourisa/projs/images/granie.sif \
  Rscript src/methods/granie/script.R \
    --rna  /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_dogma_rna_processed.h5ad \
    --atac /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_dogma_atac_processed.h5ad \
    --prediction /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_dogma_granie_grn.h5ad \
    --forceRerun TRUE

echo "End: $(date)"
