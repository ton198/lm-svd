#!/bin/bash
#SBATCH --job-name=exp7-train
#SBATCH --partition=h100
#SBATCH --qos=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=bluehive_run/logs/slurm-%x-%j.out
#SBATCH --error=bluehive_run/logs/slurm-%x-%j.err

set -euo pipefail

# Submit from the repository root; create bluehive_run/logs before sbatch.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Resubmitting continues the current stage from its last evaluation. Extra arguments
# pass through. Train the first mode first, then add modes in the same directory:
#   sbatch bluehive_run/exp7-train.sbatch --subnet gpt2 --modes 1
#   sbatch bluehive_run/exp7-train.sbatch --subnet gpt2 --modes 8
# Needs the exp4 tables (exp4-inference.sbatch) and a sample store with published blocks,
# or a producer running at the same time (sample-store-produce.sbatch).
exec .venv/bin/python -u -m experiments.exp7_independent_subnets train "$@"
