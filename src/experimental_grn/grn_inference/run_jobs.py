"""
Submit Stage 1 inference sbatch jobs for all algorithm × dataset combinations.
Outputs go to temp/experimental_grn/predictions/
Naming: {dataset}.S1-{algo}.S2-P0.S3-C0.h5ad
"""
import os, subprocess

REPO = '/home/jnourisa/projs/ongoing/task_grn_inference'
PRED_DIR = f'{REPO}/temp/experimental_grn/predictions'
SCRIPT = f'{REPO}/temp/experimental_grn/grn_inference/stage1_infer.py'
TF_ALL = f'{REPO}/resources/grn_benchmark/prior/tf_all.csv'
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(f'{REPO}/logs', exist_ok=True)

ALGORITHMS = ['pearson', 'spearman', 'ridge', 'lasso', 'elasticnet']

DATASETS = [
    'op', 'parsebioscience', '300BCG', 'ibd_uc', 'ibd_cd',
    'replogle', 'xaira_HEK293T', 'xaira_HCT116', 'nakatake', 'norman'
]

# Large PBMC datasets need more time and memory
LARGE = {'op', 'parsebioscience', '300BCG', 'ibd_uc', 'ibd_cd'}

def submit(ds, algo):
    pred = f'{PRED_DIR}/{ds}.S1-{algo}.S2-P0.S3-C0.h5ad'
    if os.path.exists(pred):
        print(f"  [{ds} / {algo}] already exists, skipping")
        return

    rna = f'{REPO}/resources/grn_benchmark/inference_data/{ds}_rna.h5ad'
    if not os.path.exists(rna):
        print(f"  [{ds} / {algo}] RNA file not found, skipping")
        return

    time   = '6:00:00' if ds in LARGE else '3:00:00'
    mem    = '120GB'   if ds in LARGE else '60GB'
    # Spearman is slower than Pearson due to ranking
    if algo == 'spearman' and ds in LARGE:
        time = '8:00:00'

    wrap = (
        f'cd {REPO} && python {SCRIPT} '
        f'--rna {rna} '
        f'--tf_all {TF_ALL} '
        f'--algorithm {algo} '
        f'--max_n_links 1000000 '
        f'--max_cells 5000 '
        f'--n_jobs 20 '
        f'--prediction {pred}'
    )
    cmd = [
        'sbatch',
        '--job-name',    f's1_{algo[:4]}_{ds}',
        '--output',      f'{REPO}/logs/s1_{algo}_{ds}_%j.out',
        '--error',       f'{REPO}/logs/s1_{algo}_{ds}_%j.err',
        '--ntasks',      '1',
        '--cpus-per-task', '20',
        '--time',        time,
        '--mem',         mem,
        '--partition',   'cpu',
        '--wrap',        wrap,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [{ds} / {algo}] ERROR: {r.stderr.strip()}")
    else:
        jid = r.stdout.strip().split()[-1]
        print(f"  [{ds} / {algo}] submitted job {jid}")


if __name__ == '__main__':
    print(f"Submitting Stage 1 jobs: {len(ALGORITHMS)} algorithms × {len(DATASETS)} datasets\n")
    for algo in ALGORITHMS:
        print(f"--- {algo.upper()} ---")
        for ds in DATASETS:
            submit(ds, algo)
        print()
