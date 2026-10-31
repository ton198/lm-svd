#!/bin/bash
#SBATCH --job-name=lm-svd-exp15-exp14
#SBATCH --partition=h100
#SBATCH --qos=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --no-requeue
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

pair_batch="${PAIR_BATCH:-4096}"
anchor_futures="${ANCHOR_FUTURES:-64}"
printf 'Running exp15 exp14; log: %s/exp15-exp14.out\n' "$log_dir"
exp15 exp14 --device cuda --pair-batch "$pair_batch" --anchor-futures "$anchor_futures" --overwrite > "$log_dir/exp15-exp14.out" 2>&1
