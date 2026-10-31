#!/bin/bash
#SBATCH --job-name=lm-svd-exp16-train
#SBATCH --partition=preempt
#SBATCH --gres=gpu:H100:1
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

source="${SOURCE:?set SOURCE}"
variant="${VARIANT:?set VARIANT}"
max_steps="${MAX_STEPS:-30000}"

# No --overwrite: resume the latest checkpoint instead of discarding training.
printf 'Running exp16 train for %s %s; log (appended): %s/exp16-%s-%s.out\n' "$source" "$variant" "$log_dir" "$source" "$variant"
# Near the time limit, requeue this job; the next run resumes from train/checkpoint.pt.
requeue() {
    printf 'Time limit near; requeueing job %s\n' "$SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap requeue USR1
# Append: requeues reuse the job ID and log directory.
exp16 train --source "$source" --variant "$variant" --device cuda --max-steps "$max_steps" >> "$log_dir/exp16-${source}-${variant}.out" 2>&1 &
wait $!
