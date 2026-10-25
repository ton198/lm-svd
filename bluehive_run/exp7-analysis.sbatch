#!/bin/bash
#SBATCH --job-name=lm-svd-exp7-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
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

# Must match the exp7 train arguments (including STREAM) so analysis reads the same run directory.
stream="${STREAM:-wikitext-103-train}"
run_args=(--stream "$stream" --model-path models/gpt2-small --device cuda)

# --overwrite makes repeated runs replace earlier analysis outputs (writes are atomic).
printf 'Running exp7 analysis on %s; log: %s/exp7-analysis.out\n' "$stream" "$log_dir"
exp7 analysis "${run_args[@]}" --overwrite > "$log_dir/exp7-analysis.out" 2>&1
