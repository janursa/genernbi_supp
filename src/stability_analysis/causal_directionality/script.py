import os
import sys
import glob
import argparse

env = os.environ

sys.path.insert(0, env['UTILS_DIR'])
sys.path.insert(0, env['METRICS_DIR'])
from src.params import get_par

sys.path.insert(0, env['genernbi_supp_DIR'])
from src.stability_analysis.causal_directionality.helper import main as main_causal

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

par = get_par(args.dataset)

grns_dir = f"{env['RESULTS_DIR']}/{args.dataset}/"
# auto-detect all methods with prediction files present for this dataset
available_files = glob.glob(f"{grns_dir}/*.prediction.h5ad")
methods = sorted(set(os.path.basename(f).split('.')[1] for f in available_files))
print(f"Detected methods for {args.dataset}: {methods}")

par = {
  **par,
  **{
  'grns_dir': grns_dir,
  'write_dir': f"{env['RESULTS_DIR']}/experiment/causal_directionality/",
  'methods': methods,
  'dataset': args.dataset
}
}

if __name__ == "__main__":
  main_causal(par)
