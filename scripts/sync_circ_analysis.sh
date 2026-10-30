#!/bin/bash
# Pull only <exp>/<run>/{analysis,lingen}/ trees of data/experiment_results from circ.
# Extra args go to rsync, e.g. -n (dry run).
set -euo pipefail

REMOTE="${REMOTE:-circ}"
REMOTE_ROOT="${REMOTE_ROOT:-/scratch/lli81/projects/ur_research/lm-svd}"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p data/experiment_results

rsync -avzhP --delete --prune-empty-dirs \
    --include='/*/' \
    --include='/*/*/' \
    --include='/*/*/analysis/***' \
    --include='/*/*/lingen/***' \
    --exclude='*' \
    "$@" \
    "${REMOTE}:${REMOTE_ROOT}/data/experiment_results/" \
    data/experiment_results/
