#!/bin/bash
#SBATCH --job-name=scenicplus_dogma
#SBATCH --output=/home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs/scenicplus_dogma_%j.out
#SBATCH --error=/home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs/scenicplus_dogma_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=500GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -e
mkdir -p /home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs
echo "=== SCENIC+ MMP dogma ==="
echo "Start: $(date)"
cd /home/jnourisa/projs/ongoing/task_grn_inference
TEMP_DIR=/home/jnourisa/projs/ongoing/task_grn_inference/temp/scenicplus_dogma_${SLURM_JOB_ID}
mkdir -p ${TEMP_DIR}

singularity exec \
  --bind /vol/projects:/vol/projects \
  --bind /home/jnourisa:/home/jnourisa \
  --bind /tmp:/tmp \
  /home/jnourisa/projs/images/scenicplus \
  python3 src/methods/scenicplus/script.py \
    --rna  /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_dogma_rna_processed.h5ad \
    --atac /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_dogma_atac_processed.h5ad \
    --prediction /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_dogma_scenicplus_grn.h5ad \
    --temp_dir ${TEMP_DIR}

echo "End: $(date)"
