#!/bin/bash
#SBATCH --job-name=exp8
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --exclude=bhg00[46-48]
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=bluehive_run/logs/slurm-%x-%j.out
#SBATCH --error=bluehive_run/logs/slurm-%x-%j.err

set -euo pipefail

# Submit from the repository root; create bluehive_run/logs before sbatch. The
# arguments are an exp8 command, for example
#   sbatch -J exp8-spectrum bluehive_run/exp8.sbatch spectrum
#   sbatch -J exp8-center   bluehive_run/exp8.sbatch center --encoder gpt2 --modes 32
#   sbatch -J exp8-mean     bluehive_run/exp8.sbatch mean --prefix-tokens 64
# A requeued job repeats its command and continues from the run's checkpoint.
# Needs the exp4 tables (exp4-inference.sbatch) and the token stream; no sample store.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# The checkout the job was submitted from, not the installed one.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -u -m experiments.exp8_hypotheses "$@"
