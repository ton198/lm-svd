#!/bin/bash
#SBATCH --job-name=lm-svd-exp4-train
#SBATCH --partition=preempt
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
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

# Override at submit time, e.g. DATASET=task-single-text sbatch bluehive_run/exp4-train.sbatch
dataset="${DATASET:-wikitext-large}"
run_args=(--dataset "$dataset" --model-path models/gpt2-small --device cuda)

# No --overwrite: resume the latest checkpoint instead of discarding training.
printf 'Running exp4 train on %s; log: %s/exp4-train.out\n' "$dataset" "$log_dir"
exp4 train "${run_args[@]}" > "$log_dir/exp4-train.out" 2>&1
