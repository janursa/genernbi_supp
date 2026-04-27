#!/bin/bash
#SBATCH --job-name=celloracle_multiome
#SBATCH --output=/home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs/celloracle_multiome_%j.out
#SBATCH --error=/home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs/celloracle_multiome_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=300GB
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jalil.nourisa@gmail.com

set -e
mkdir -p /home/jnourisa/projs/ongoing/geneRNBI/src/MMP/logs
echo "=== CellOracle MMP multiome ==="
echo "Start: $(date)"
cd /home/jnourisa/projs/ongoing/task_grn_inference

NEW_CACHE=$TMPDIR/cache
mkdir -p $NEW_CACHE
if [ -z "$XDG_CACHE_HOME" ]; then XDG_CACHE_HOME=$HOME/.cache; fi
if [ -d "$XDG_CACHE_HOME/gimmemotifs" ]; then cp -r $XDG_CACHE_HOME/gimmemotifs $NEW_CACHE/; fi
export XDG_CACHE_HOME=$NEW_CACHE

singularity exec \
  --bind /vol/projects:/vol/projects \
  --bind /home/jnourisa:/home/jnourisa \
  --bind /tmp:/tmp \
  /home/jnourisa/projs/images/celloracle \
  python3 src/methods/celloracle/script.py \
    --rna  /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_multiome_rna_processed.h5ad \
    --atac /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_multiome_atac_processed.h5ad \
    --temp_dir /home/jnourisa/projs/ongoing/task_grn_inference/output/celloracle_multiome \
    --prediction /home/jnourisa/projs/ongoing/task_grn_inference/resources/results/MMP/MMP_multiome_celloracle_grn.h5ad

echo "End: $(date)"
