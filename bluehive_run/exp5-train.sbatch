#!/bin/bash
#SBATCH --job-name=exp5-train
#SBATCH --partition=h100
#SBATCH --qos=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=bluehive_run/logs/slurm-%x-%j.out
#SBATCH --error=bluehive_run/logs/slurm-%x-%j.err

set -euo pipefail

# Submit from the repository root; create bluehive_run/logs before sbatch.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Resubmitting continues from the last checkpoint. Extra arguments pass through,
# e.g. sbatch exp5-train.sbatch --mlp-width 8192
# Needs the exp4 tables (exp4-inference.sbatch) and a sample store with published blocks,
# or a producer running at the same time (sample-store-produce.sbatch).
exec .venv/bin/python -u -m experiments.exp5_gpt2_mlp_table train "$@"
