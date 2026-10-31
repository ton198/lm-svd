#!/bin/bash
#SBATCH --job-name=lm-svd-exp14-generate
#SBATCH --partition=h100
#SBATCH --qos=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

eval_model="${EVAL_MODEL:-models/gpt2-small}"
basis_chunk="${BASIS_CHUNK:-2}"
log_file="$log_dir/exp14-generate-$(basename "$eval_model").out"
printf 'Running exp14 generate for %s; log: %s\n' "$eval_model" "$log_file"
exp14 generate --dataset wikitext-large --model-path models/gpt2-small \
    --eval-model-path "$eval_model" --device cuda --basis-chunk "$basis_chunk" \
    > "$log_file" 2>&1
