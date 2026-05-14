#!/bin/bash
#SBATCH --job-name=ppcor_multiome
#SBATCH --output=/home/jnourisa/projs/ongoing/genernbi_supp/src/MMP/logs/ppcor_multiome_%j.out
#SBATCH --error=/home/jnourisa/projs/ongoing/genernbi_supp/src/MMP/logs/ppcor_multiome_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=300GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -e
mkdir -p /home/jnourisa/projs/ongoing/genernbi_supp/src/MMP/logs
echo "=== ppcor MMP multiome ==="
echo "Start: $(date)"
cd /home/jnourisa/projs/ongoing/task_grn_inference

singularity exec \
  --bind /vol/projects:/vol/projects \
  --bind /home/jnourisa:/home/jnourisa \
  /home/jnourisa/projs/images/ppcor \
  Rscript src/methods/ppcor/script.R \
    --rna  /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_multiome_rna_processed.h5ad \
    --prediction /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_multiome_ppcor_grn.h5ad

echo "End: $(date)"
