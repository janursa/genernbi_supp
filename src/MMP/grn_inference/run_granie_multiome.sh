#!/bin/bash
#SBATCH --job-name=granie_multiome
#SBATCH --output=/home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs/granie_multiome_%j.out
#SBATCH --error=/home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs/granie_multiome_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=300GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -e
mkdir -p /home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs
echo "=== GRaNIE MMP multiome ==="
echo "Start: $(date)"
cd /home/jnourisa/projs/ongoing/task_grn_inference

singularity exec \
  --bind /vol/projects:/vol/projects \
  --bind /home/jnourisa:/home/jnourisa \
  --bind /tmp:/tmp \
  --env RETICULATE_PYTHON=/usr/bin/python3 \
  /home/jnourisa/projs/images/granie.sif \
  Rscript src/methods/granie/script.R \
    --rna  /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_multiome_rna_processed.h5ad \
    --atac /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_multiome_atac_processed.h5ad \
    --prediction /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_multiome_granie_grn.h5ad \
    --forceRerun TRUE

echo "End: $(date)"
