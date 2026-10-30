#!/bin/bash
#SBATCH --job-name=lm-svd-exp11-distill
#SBATCH --partition=preempt
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

stream="${STREAM:-wikitext-103-train}"

# No --overwrite: resume the latest checkpoint instead of discarding training.
printf 'Running exp11 distill on %s; log (appended): %s/exp11-distill.out\n' "$stream" "$log_dir"
# Near the time limit, requeue this job; the next run resumes from distill/checkpoint.pt.
requeue() {
    printf 'Time limit near; requeueing job %s\n' "$SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue USR1
# Append: requeues reuse the job ID and log directory.
exp11 distill --stream "$stream" --model-path models/gpt2-small --device cuda >> "$log_dir/exp11-distill.out" 2>&1 &
wait $!
