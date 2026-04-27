#!/bin/bash
#SBATCH --job-name=mmp_preprocess
#SBATCH --output=/home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs/preprocess_%j.out
#SBATCH --error=/home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs/preprocess_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=200GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -e
mkdir -p /home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs
echo "=== MMP Preprocessing ==="
echo "Start: $(date)"

singularity exec \
  --bind /vol/projects:/vol/projects \
  --bind /home/jnourisa:/home/jnourisa \
  --bind /vol/biotools:/vol/biotools \
  /vol/projects/CIIM/agentic_central/singularity/ciim.sif \
  python3 /home/jnourisa/projs/ongoing/geneRNBI/src/MMP/preprocess/script.py

echo "End: $(date)"
