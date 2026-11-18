#!/bin/bash
#SBATCH --job-name=exp7-center
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --gres=gpu:1
#SBATCH --exclude=bhg00[45-48]
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=bluehive_run/logs/slurm-%x-%j.out
#SBATCH --error=bluehive_run/logs/slurm-%x-%j.err

set -euo pipefail

# Submit from the repository root; create bluehive_run/logs before sbatch. The
# arguments are an exp7 command. Build the center pool once, then train on it:
#   sbatch -J exp7-pool bluehive_run/exp7-center.sbatch center-pool
#   sbatch -J exp7-c64-scratch bluehive_run/exp7-center.sbatch train --subnet scratch \
#     --center-pool data/center_pools/wikitext-103-train__gpt2-small__h64-f32__m64
# Add --joint --steps N to train all modes at once instead of one stage per mode.
# A requeued job repeats its command: an unfinished pool and an unfinished stage
# both continue. Needs the exp4 tables (exp4-inference.sbatch) and the token
# stream; a centered run reads no sample store.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# The checkout the job was submitted from, not the installed one.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -u -m experiments.exp7_independent_subnets "$@"
