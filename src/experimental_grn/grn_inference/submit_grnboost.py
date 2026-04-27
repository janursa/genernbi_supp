"""
Submit GRNBoost2 Stage 1 inference jobs for all 10 datasets.

Uses the grnboost singularity image (aertslab/pyscenic:0.12.1).
Keeps 1M edges (no 50k default subsetting).
"""
import os
import subprocess

REPO     = '/home/jnourisa/projs/ongoing/task_grn_inference'
SCRIPT   = f'{REPO}/temp/experimental_grn/grn_inference/stage1_grnboost.py'
IMAGE    = '/home/jnourisa/projs/images/grnboost'
PRED_DIR = f'{REPO}/temp/experimental_grn/predictions'

DATASETS = [
    'op', 'parsebioscience', '300BCG', 'ibd_uc', 'ibd_cd',
    'replogle', 'xaira_HEK293T', 'xaira_HCT116', 'nakatake', 'norman'
]

# Large PBMC datasets need more resources
LARGE = {'op', 'parsebioscience', '300BCG', 'ibd_uc', 'ibd_cd'}

os.makedirs(f'{REPO}/logs', exist_ok=True)

for ds in DATASETS:
    pred = f'{PRED_DIR}/{ds}.S1-grnboost.S2-P0.S3-C0.h5ad'
    if os.path.exists(pred):
        print(f'  [{ds}] already exists, skipping')
        continue

    rna = f'{REPO}/resources/grn_benchmark/inference_data/{ds}_rna.h5ad'
    if not os.path.exists(rna):
        print(f'  [{ds}] RNA not found, skipping')
        continue

    time    = '24:00:00' if ds in LARGE else '12:00:00'
    mem     = '120GB'    if ds in LARGE else '80GB'
    workers = 20

    wrap = (
        f'singularity exec '
        f'--bind /home/jnourisa/projs:/home/jnourisa/projs '
        f'{IMAGE} '
        f'python3 {SCRIPT} '
        f'--dataset {ds} '
        f'--num_workers {workers} '
        f'--max_n_links 1000000'
    )

    cmd = [
        'sbatch',
        '--job-name',      f's1_grn_{ds}',
        '--output',        f'{REPO}/logs/s1_grnboost_{ds}_%j.out',
        '--error',         f'{REPO}/logs/s1_grnboost_{ds}_%j.err',
        '--ntasks',        '1',
        '--cpus-per-task', str(workers),
        '--time',          time,
        '--mem',           mem,
        '--partition',     'cpu',
        '--wrap',          wrap,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        jid = result.stdout.strip().split()[-1]
        print(f'  [{ds}] submitted → job {jid}  (time={time}, mem={mem})')
    else:
        print(f'  [{ds}] FAILED: {result.stderr.strip()}')
