#!/bin/bash
#SBATCH --job-name=lm-svd-exp2-inference
#SBATCH --partition=h100
#SBATCH --qos=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=bluehive_run/logs/slurm-%x-%j.out
#SBATCH --error=bluehive_run/logs/slurm-%x-%j.err

set -euo pipefail

# Submit with sbatch from the repository root, not from bluehive_run/.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"

log_dir="bluehive_run/logs/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$log_dir"

printf 'Job ID: %s\nNode: %s\nWorking directory: %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$PWD"

nvidia-smi
source .venv/bin/activate

# Override RUN_NAME for a separate direct-WikiText run.
run_name="${RUN_NAME:-wikitext-direct}"
run_args=(--run-name "$run_name" --model-path models/gpt2-small --device cuda)

# Inference cannot resume; --overwrite recomputes and replaces any earlier matrix.
printf 'Running exp2 inference on %s; log: %s/exp2-inference.out\n' "$run_name" "$log_dir"
exp2 inference "${run_args[@]}" --overwrite > "$log_dir/exp2-inference.out" 2>&1
