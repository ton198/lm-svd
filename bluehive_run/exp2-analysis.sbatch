#!/bin/bash
#SBATCH --job-name=lm-svd-exp2-analysis
#SBATCH --partition=h100
#SBATCH --qos=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
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

# Must match the exp2 inference arguments (including DATASET) so analysis reads the same run directory.
dataset="${DATASET:-wikitext}"
run_args=(--dataset "$dataset" --model-path models/gpt2-small --device cuda)

# Run all centering modes; --overwrite replaces only that mode's analysis outputs.
for centering in none global row deflate1; do
    printf 'Running exp2 analysis on %s with centering %s; log: %s/exp2-analysis-%s.out\n' "$dataset" "$centering" "$log_dir" "$centering"
    exp2 analysis "${run_args[@]}" --centering "$centering" --overwrite > "$log_dir/exp2-analysis-${centering}.out" 2>&1
done
