#!/bin/bash
#SBATCH --job-name=lm-svd-exp
#SBATCH --partition=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --array=1-3
#SBATCH --output=bluehive_run/logs/slurm-%x%a-%A.out
#SBATCH --error=bluehive_run/logs/slurm-%x%a-%A.err

set -euo pipefail

# Submit with sbatch from the repository root, not from bluehive_run/.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"

log_dir="bluehive_run/logs/slurm-${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$log_dir"

printf 'Job ID: %s\nNode: %s\nWorking directory: %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$PWD"

nvidia-smi
source .venv/bin/activate

# Each array task runs only its assigned experiment; inference failure skips analysis.
experiment="exp${SLURM_ARRAY_TASK_ID}"
printf 'Running %s inference; log: %s/%s.out\n' "$experiment" "$log_dir" "$experiment"
"$experiment" inference --dataset task-single-text > "$log_dir/$experiment.out" 2>&1

printf 'Running %s analysis; log: %s/%s-analysis.out\n' "$experiment" "$log_dir" "$experiment"
"$experiment" analysis --dataset task-single-text > "$log_dir/$experiment-analysis.out" 2>&1
