#!/bin/bash
#SBATCH --job-name=svd-inference-analysis
#SBATCH --partition=h100
#SBATCH --qos=h100
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=bluehive_run/logs/slurm-%x-%j.out
#SBATCH --error=bluehive_run/logs/slurm-%x-%j.err

set -euo pipefail

# Submit from the repository root; create bluehive_run/logs before sbatch.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg

exec .venv/bin/python -u -m experiments.svd_inference.lingen
