"""
Submit evaluation sbatch jobs for all predictions in predictions_final/.

One job per prediction file. Reads cell_type from config.env.
Outputs scores to temp/experimental_grn/scores/
"""
import os, glob, subprocess

REPO      = '/home/jnourisa/projs/ongoing/task_grn_inference'
PRED_DIR  = f'{REPO}/temp/experimental_grn/predictions_final'
SCORE_DIR = f'{REPO}/temp/experimental_grn/scores'
EVAL_DIR  = f'{REPO}/resources/grn_benchmark/evaluation_data'
PRIOR     = f'{REPO}/resources/grn_benchmark/prior'
# Experiment-specific consensus (regulators_consensus + ws_consensus)
EXP_PRIOR = f'{REPO}/temp/experimental_grn/prior'
GT_DIR    = f'{REPO}/resources/grn_benchmark/ground_truth'
SCRIPT    = f'{REPO}/src/metrics/all_metrics/script.py'

os.makedirs(SCORE_DIR, exist_ok=True)
os.makedirs(f'{REPO}/logs', exist_ok=True)

CELL_TYPE = {
    'replogle':     'K562',
    'norman':       'K562',
    'xaira_HEK293T':'HEK293T',
    'xaira_HCT116': 'HCT116',
    'op':           'PBMC',
    'parsebioscience':'PBMC',
    '300BCG':       'PBMC',
    'ibd_uc':       'PBMC',
    'ibd_cd':       'PBMC',
    'nakatake':     '',
}

LARGE = {'op', 'parsebioscience', '300BCG', 'ibd_uc', 'ibd_cd', 'xaira_HEK293T', 'xaira_HCT116'}


def optional_flag(flag, path):
    return f'--{flag} "{path}"' if os.path.exists(path) else ''


def build_eval_cmd(ds, pred_path, score_path):
    ct = CELL_TYPE[ds]
    gt_prefix = f'{GT_DIR}/{ct}' if ct else None

    # Use experiment-specific consensus if available, otherwise fall back to original
    reg_consensus = (f'{EXP_PRIOR}/regulators_consensus_{ds}.json'
                     if os.path.exists(f'{EXP_PRIOR}/regulators_consensus_{ds}.json')
                     else f'{PRIOR}/regulators_consensus_{ds}.json')

    parts = [
        f'python {SCRIPT}',
        f'  --prediction "{pred_path}"',
        f'  --evaluation_data "{EVAL_DIR}/{ds}_rna_all.h5ad"',
        f'  --regulators_consensus "{reg_consensus}"',
        f'  --tf_all "{PRIOR}/tf_all.csv"',
        f'  --layer lognorm',
        f'  --reg_type ridge',
        f'  --num_workers 20',
        f'  --score "{score_path}"',
    ]

    if gt_prefix:
        for suffix in ['unibind', 'chipatlas', 'remap']:
            p = f'{gt_prefix}_{suffix}.csv'
            if os.path.exists(p):
                parts.append(f'  --ground_truth_{suffix} "{p}"')

    for opt, exp_fname, orig_fname in [
        ('evaluation_data_de',     f'{EVAL_DIR}/{ds}_de.h5ad',                  None),
        ('ws_consensus',           f'{EXP_PRIOR}/ws_consensus_{ds}.csv',         f'{PRIOR}/ws_consensus_{ds}.csv'),
        ('ws_distance_background', None,                                          f'{PRIOR}/ws_distance_background_{ds}.csv'),
    ]:
        chosen = None
        if exp_fname and os.path.exists(exp_fname):
            chosen = exp_fname
        elif orig_fname and os.path.exists(orig_fname):
            chosen = orig_fname
        if chosen:
            parts.append(f'  --{opt} "{chosen}"')

    return ' \\\n'.join(parts)


def submit(pred_path):
    fname = os.path.basename(pred_path)
    # extract dataset from fname: {ds}.S1-{algo}.{c_tag}.h5ad
    ds = fname.split('.S1-')[0]
    if ds not in CELL_TYPE:
        print(f"  [{fname}] unknown dataset '{ds}', skipping")
        return

    score_path = os.path.join(SCORE_DIR, fname.replace('.h5ad', '.score.h5ad'))
    if os.path.exists(score_path):
        print(f"  [{fname}] score already exists, skipping")
        return

    time = '12:00:00' if ds in LARGE else '3:00:00'
    mem  = '120GB'   if ds in LARGE else '80GB'

    wrap = f'cd {REPO} && ' + build_eval_cmd(ds, pred_path, score_path)

    jname = fname.replace('.h5ad', '')[:40]
    cmd = [
        'sbatch',
        '--job-name',      f'eval_{jname}',
        '--output',        f'{REPO}/logs/eval_{fname}_%j.out',
        '--error',         f'{REPO}/logs/eval_{fname}_%j.err',
        '--ntasks',        '1',
        '--cpus-per-task', '20',
        '--time',          time,
        '--mem',           mem,
        '--partition',     'cpu',
        '--wrap',          wrap,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [{fname}] ERROR: {r.stderr.strip()}")
    else:
        jid = r.stdout.strip().split()[-1]
        print(f"  [{fname}] submitted job {jid}")


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(PRED_DIR, '*.h5ad')))
    print(f"Found {len(files)} prediction files in {PRED_DIR}")
    for f in files:
        submit(f)
    print("\nDone submitting eval jobs.")
