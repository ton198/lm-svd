#!/bin/bash
#SBATCH --job-name=lm-svd-exp5-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
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

# Requires exp1 inference and exp1 analysis (same DATASET and centering) under data/experiment_results.
dataset="${DATASET:-wikitext}"
run_args=(--dataset "$dataset" --model-path models/gpt2-small --device cuda)

# Run all centering modes; --overwrite replaces only that mode's analysis outputs.
for centering in none global row deflate1; do
    printf 'Running exp5 analysis on %s with centering %s; log: %s/exp5-analysis-%s.out\n' "$dataset" "$centering" "$log_dir" "$centering"
    exp5 analysis "${run_args[@]}" --centering "$centering" --overwrite > "$log_dir/exp5-analysis-${centering}.out" 2>&1
done
