"""
Piece-wise analysis for GRN metrics (regression and WS distance).
Migrated from notebooks/supp_analysis.ipynb (cells 79-88).
"""
from src.stability_analysis.piecewise_analysis.post_piecewise import (
    config_regression,
    wrapper_regression_feature_analysis,
    wrapper_plot_regression,
    config_ws,
    wrapper_ws_analysis,
)
from src.stability_analysis.piecewise_analysis.post_piecewise_easy_hard import (
    run_easy_hard_analysis,
)

# --- Regression ---
dataset, gene_wise_output, gene_wise_feature_importance = config_regression()
wrapper_plot_regression(gene_wise_output)
wrapper_regression_feature_analysis(dataset, gene_wise_feature_importance)

# --- WS distance ---
dataset, ws_output = config_ws()
wrapper_ws_analysis(dataset, ws_output)

# --- Easy / Hard gene and TF analysis ---
run_easy_hard_analysis()
