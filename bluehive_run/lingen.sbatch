#!/bin/bash
#SBATCH --job-name=lm-svd-lingen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
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

train_stream="${STREAM:-wikitext-103-train}"
experiments="${EXPERIMENTS:-exp6 exp7 exp8 exp9 exp10}"
eval_streams="${EVAL_STREAMS:-wikitext-103-train wikitext-103-validation}"
if [[ ! -f data/token_streams/wikitext-103-validation.npy ]]; then
    data-pipline build-token-stream --split validation \
        --output data/token_streams/wikitext-103-validation.npy > "$log_dir/build-validation-stream.out" 2>&1
fi
for e in $experiments; do
    for s in $eval_streams; do
        for mode in onestep autoregressive; do
            printf 'Running lingen %s %s on %s\n' "$mode" "$e" "$s"
            lingen "$mode" --experiment "$e" --stream "$train_stream" --eval-stream "$s" \
                --model-path models/gpt2-small --device cuda --overwrite \
                > "$log_dir/lingen-$e-$s-$mode.out" 2>&1
        done
    done
done
