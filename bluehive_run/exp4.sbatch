#!/bin/bash
#SBATCH --job-name=lm-svd-exp4
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=bluehive_run/logs/slurm-exp4-%x-%j.out
#SBATCH --error=bluehive_run/logs/slurm-exp4-%x-%j.err

set -euo pipefail

# Submit with sbatch from the repository root, not from bluehive_run/.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"

log_dir="bluehive_run/logs/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$log_dir"

printf 'Job ID: %s\nNode: %s\nWorking directory: %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$PWD"

nvidia-smi
source .venv/bin/activate

run_args=(--dataset wikitext-large --model-path models/gpt2-small --device cuda)

# No --overwrite: resume the latest checkpoint instead of discarding training.
printf 'Running exp4 train; log: %s/exp4-train.out\n' "$log_dir"
exp4 train "${run_args[@]}" > "$log_dir/exp4-train.out" 2>&1

# Training failure or a Slurm timeout skips analysis; saved snapshots remain usable.
printf 'Running exp4 analysis; log: %s/exp4-analysis.out\n' "$log_dir"
exp4 analysis "${run_args[@]}" --overwrite > "$log_dir/exp4-analysis.out" 2>&1
