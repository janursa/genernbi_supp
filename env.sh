# env.sh
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    base_dir="/home/jnourisa/projs/ongoing"
    resources_dir="${base_dir}/task_grn_inference/resources"
else
    base_dir="/Users/jno24/Documents/projs/ongoing"
    resources_dir="${base_dir}/task_grn_inference/resources"

fi

echo "Base directory set to: $base_dir"
export geneRNBI_DIR="${base_dir}/geneRNBI" #->file directiory
export TASK_GRN_INFERENCE_DIR="${base_dir}/task_grn_inference" #-> it's ../../task_grn_inference
export DOCS_SOURCE_DIR="${TASK_GRN_INFERENCE_DIR}/docs/source"
export DOCS_IMAGES_DIR="${DOCS_SOURCE_DIR}/images"

export PYTHONPATH="$geneRNBI_DIR:${PYTHONPATH:-}"

export RESOURCES_DIR="${resources_dir}"
export RESULTS_DIR="${RESOURCES_DIR}/results"
export IMAGES_DIR="/home/jnourisa/projs/images"
export INFERENCE_DIR="${RESOURCES_DIR}/grn_benchmark/inference_data"
export EVALUATION_DIR="${RESOURCES_DIR}/grn_benchmark/evaluation_data"
export PRIOR_DIR="${RESOURCES_DIR}/grn_benchmark/prior"
export EXTENDED_DIR="${RESOURCES_DIR}/extended_data"
export METHODS_DIR="${TASK_GRN_INFERENCE_DIR}/src/methods"
export METRICS_DIR="${TASK_GRN_INFERENCE_DIR}/src/metrics"
export UTILS_DIR="${TASK_GRN_INFERENCE_DIR}/src/utils"

# echo "Environment variables set:"
# echo "TASK_GRN_INFERENCE_DIR=$TASK_GRN_INFERENCE_DIR"
# echo "geneRNBI_DIR=$geneRNBI_DIR"
# echo "RESULTS_DIR=$RESULTS_DIR"
# echo "IMAGES_DIR=$IMAGES_DIR"
# echo "RESOURCES_DIR=$RESOURCES_DIR"
# echo "INFERENCE_DIR=$INFERENCE_DIR"
# echo "EVALUATION_DIR=$EVALUATION_DIR"
# echo "PRIOR_DIR=$PRIOR_DIR"
# echo "EXTENDED_DIR=$EXTENDED_DIR"
# echo "METHODS_DIR=$METHODS_DIR"
# echo "METRICS_DIR=$METRICS_DIR"
# echo "UTILS_DIR=$UTILS_DIR"

