#!/bin/bash
#SBATCH --job-name=lm-svd-sample-store
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
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

# Override GPU with sbatch --gres=gpu:H100:1 when needed.
stream="${STREAM:-wikitext-103-train}"
max_gib="${MAX_GIB:-64}"
extra_args=()
if [[ -n "${MAX_BLOCKS:-}" ]]; then
    extra_args+=(--max-blocks "$MAX_BLOCKS")
fi

# Requeue resumes production without overwriting the store.
requeue() {
    printf 'Time limit near; requeueing job %s\n' "$SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue USR1
data-pipline sample-store produce --stream "$stream" --device cuda --worker-id "$SLURM_JOB_ID" --max-gib "$max_gib" "${extra_args[@]}" >> "$log_dir/sample-store-${stream}-worker${SLURM_JOB_ID}.out" 2>&1 &
wait $!
