#!/bin/bash
# Run this script manually to regenerate env.yaml from env.sh.
# Do NOT source this from SLURM jobs — concurrent writes will corrupt env.yaml.

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source env.sh

output="$(pwd)/env.yaml"
echo "# Auto-generated from env.sh" > "$output"

for var in TASK_GRN_INFERENCE_DIR genernbi_supp_DIR PYTHONPATH RESULTS_DIR IMAGES_DIR RESOURCES_DIR INFERENCE_DIR EVALUATION_DIR PRIOR_DIR EXTENDED_DIR METHODS_DIR METRICS_DIR UTILS_DIR DOCS_IMAGES_DIR; do
    value="$(eval echo "\"\$$var\"")"
    echo "$var: \"$value\"" >> "$output"
done

echo "Wrote $output"
