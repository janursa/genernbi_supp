#!/bin/bash

# datasets=(300BCG ibd_cd ibd_uc nakatake norman op parsebioscience replogle xaira_HCT116 xaira_HEK293T)
datasets=(MSCIC soundlife soundlife_vaccine)

for dataset in "${datasets[@]}"; do
    sbatch "$(dirname "$0")/experiment_causal_directionality.sh" $dataset
done
