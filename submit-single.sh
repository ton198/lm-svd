#!/bin/bash
set -euo pipefail

experiment="${1-}"
if [[ $# -ne 1 || ! "$experiment" =~ ^exp[1-3]$ ]]; then
    printf 'Usage: bash %s exp1|exp2|exp3\n' "$0" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p bluehive_run/logs
# Override the batch script's array range with exactly one experiment.
exec sbatch --array="${experiment#exp}" --time=6:00:00 bluehive_run/task-single-text.sbatch
