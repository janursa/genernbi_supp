"""
run_apply_prior_jobs.py — Submit sbatch jobs for all prior integration predictions.

Generates:
  - C0/C1/C2 for ALL 10 datasets × 6 algos  = 180 predictions
  - C3/C4/C5/C6 for ATAC datasets (op, ibd_uc, ibd_cd) × 6 algos  = 72 predictions
  Total: 252 predictions

Each job runs apply_prior.py for a single dataset×algo×c_tag combination.

C1/C2 jobs are gated: they wait for skeleton_motif.csv / skeleton_motif_100kb.csv.
C3–C6 jobs require no extra skeleton files (use ATAC data directly).

Usage:
  python run_apply_prior_jobs.py                  # submit all
  python run_apply_prior_jobs.py --dataset op     # only op
  python run_apply_prior_jobs.py --c_tags C0 C1   # only these C-tags
  python run_apply_prior_jobs.py --dry_run        # print commands, don't submit
"""

import os, sys, argparse, subprocess, glob

REPO      = '/home/jnourisa/projs/ongoing/task_grn_inference'
IMAGE     = '/home/jnourisa/projs/images/scglue'
LOGS      = f'{REPO}/logs/apply_prior'
SCRIPT    = f'{REPO}/temp/experimental_grn/grn_inference/apply_prior.py'
PRED_OUT  = f'{REPO}/temp/experimental_grn/predictions_final'

ATAC_DATASETS    = ['op', 'ibd_uc', 'ibd_cd']
NONATAC_DATASETS = ['300BCG', 'nakatake', 'norman', 'parsebioscience',
                    'replogle', 'xaira_HEK293T', 'xaira_HCT116']
ALL_DATASETS     = NONATAC_DATASETS + ATAC_DATASETS
ALGOS            = ['pearson', 'lasso', 'ridge', 'elasticnet', 'spearman', 'grnboost']

# C3-C6 are ATAC-only
ATAC_C_TAGS      = ['C3', 'C4', 'C5', 'C6']
NONATAC_C_TAGS   = ['C0', 'C1', 'C2']
ALL_C_TAGS       = NONATAC_C_TAGS + ATAC_C_TAGS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',  nargs='+', default=None,
                   help='Restrict to these datasets')
    p.add_argument('--c_tags',   nargs='+', default=None,
                   help='Restrict to these C-tags, e.g. C0 C1 C3')
    p.add_argument('--dry_run',  action='store_true')
    p.add_argument('--overwrite', action='store_true',
                   help='Resubmit even if output file already exists')
    return p.parse_args()


def already_done(ds, algo, c_tag):
    out_path = os.path.join(PRED_OUT, f'{ds}.S1-{algo}.{c_tag}.h5ad')
    return os.path.exists(out_path)


def mem_for_c_tag(c_tag):
    # C5 uses RNA+ATAC correlation → more RAM
    return '48G' if c_tag == 'C5' else '16G'


def time_for_c_tag(c_tag):
    return '2:00:00' if c_tag == 'C5' else '0:30:00'


def submit_job(ds, algo, c_tag, dry_run):
    jname = f'ap_{ds[:6]}_{algo[:4]}_{c_tag}'
    log   = os.path.join(LOGS, f'{ds}.{algo}.{c_tag}.%j.out')
    err   = os.path.join(LOGS, f'{ds}.{algo}.{c_tag}.%j.err')

    cmd = (
        f'singularity exec '
        f'--bind /home/jnourisa/projs:/home/jnourisa/projs '
        f'{IMAGE} '
        f'python3 {SCRIPT} '
        f'--dataset {ds} '
        f'--algorithm {algo} '
        f'--c_tag {c_tag}'
    )
    sbatch = (
        f'sbatch '
        f'--job-name={jname} '
        f'--output={log} '
        f'--error={err} '
        f'--time={time_for_c_tag(c_tag)} '
        f'--mem={mem_for_c_tag(c_tag)} '
        f'--cpus-per-task=4 '
        f'--partition=cpu '
        f'--wrap="{cmd}"'
    )

    if dry_run:
        print(f'[DRY RUN] {jname}')
        return

    result = subprocess.run(sbatch, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        jid = result.stdout.strip().split()[-1]
        print(f'  Submitted {jname} → job {jid}')
    else:
        print(f'  FAILED {jname}: {result.stderr.strip()}')


def main():
    args = parse_args()

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(PRED_OUT, exist_ok=True)

    datasets = args.dataset or ALL_DATASETS
    c_tags   = args.c_tags   or ALL_C_TAGS

    submitted = skipped_done = skipped_na = 0

    for ds in datasets:
        is_atac = ds in ATAC_DATASETS
        for algo in ALGOS:
            for c_tag in c_tags:
                # C3-C6 only for ATAC datasets
                if c_tag in ATAC_C_TAGS and not is_atac:
                    skipped_na += 1
                    continue

                if not args.overwrite and already_done(ds, algo, c_tag):
                    print(f'  SKIP (exists): {ds}.S1-{algo}.{c_tag}.h5ad')
                    skipped_done += 1
                    continue

                submit_job(ds, algo, c_tag, args.dry_run)
                submitted += 1

    print(f'\nSummary: {submitted} submitted, '
          f'{skipped_done} skipped (done), '
          f'{skipped_na} skipped (N/A dataset type)')


if __name__ == '__main__':
    main()
