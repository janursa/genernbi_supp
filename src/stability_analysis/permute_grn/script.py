import pandas as pd
import anndata as ad
import sys
import numpy as np
import os 
import random
import os
import argparse
env = os.environ

sys.path.insert(0, env['UTILS_DIR'])
sys.path.insert(0, env['METRICS_DIR'])
# from all_metrics.helper import main as main_metrics
from src.params import get_par

sys.path.insert(0, env['genernbi_supp_DIR'])
from src.stability_analysis.permute_grn.helper import main as main_permute
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--methods', nargs='+', default=None,
                    help='Subset of methods to run (default: all)')
parser.add_argument('--analysis_types', nargs='+', default=None,
                    help='Analysis types to run: net, sign, weight (default: all)')

args = parser.parse_args()

par = get_par(args.dataset)

par = {
  **par, 
  **{
  'grns_dir': f"{env['RESULTS_DIR']}/{args.dataset}/",
  'write_dir': f"{env['RESULTS_DIR']}/experiment/permute_grn/",
  'degrees': [0, 10, 20, 50, 100],
  'analysis_types': args.analysis_types if args.analysis_types else ["net", "sign", "weight"],
  'methods': args.methods if args.methods else [
      'ppcor', 'grnboost', 'pearson_corr', 'portia', 'scenicplus', 'scprint', 'scenic'],
  'dataset': args.dataset
}
}


if __name__ == "__main__":
  main_permute(par)