---
name: circ
description: Operate the lm-svd checkout on the CIRC BlueHive cluster over SSH — sync code with git, submit and monitor Slurm jobs, and pull results back.
---

# CIRC (BlueHive) operations

## Connection
- SSH alias `circ` (bluehive3.circ.rochester.edu, user `lli81`, ControlMaster). Always run `ssh -o BatchMode=yes circ '<commands>'`, never interactive.
- Every remote command prints harmless module-init noise (`_module_raw: not a function`, `Loading slurm/...`, `Loading requirement: circ`). Ignore it. Slurm commands are already on PATH.
- `git` needs `module load git/2.50.1` in the same remote command. `uv` is `~/.local/bin/uv` (not on PATH).
- The login node is only for git, submission and inspection. Never run GPU or long CPU work there. Tiny CPU unit tests are fine.

## Checkout
- Repo root: `/home/lli81/scratch/projects/ur_research/lm-svd` (same as `/scratch/lli81/projects/ur_research/lm-svd`, physical `/gpfs/fs2/scratch/...`). Remote `origin` = `git@github.com:ton198/lm-svd.git`.
- The checkout may have uncommitted user edits (e.g. `submit.sh`, `submit-single.sh`). Never `reset`, `checkout --`, `stash`, `clean` or commit them. Update only with `git pull --ff-only`. If that refuses, stop and report `git status` instead of forcing.
- `.venv` holds an editable install, so pulled source changes are live. Run `~/.local/bin/uv sync --frozen` only when `uv.lock` or `pyproject.toml` `[project.scripts]` changed in the pulled range.

## Deploy flow
1. Locally: run the relevant tests, commit, `git push origin main`.
2. Remote: `module load git/2.50.1 && cd <repo root> && git pull --ff-only && git log --oneline -1`. The printed hash must equal local `git rev-parse --short HEAD`.

## Submitting jobs
- Submit from the repo root. The sbatch scripts in `bluehive_run/` `cd "$SLURM_SUBMIT_DIR"`. Run `mkdir -p bluehive_run/logs` first.
- Dataset override: `DATASET=<name> sbatch ...` (default `wikitext`). sbatch exports the submit environment.
- Chain stages with `--parsable` and `--dependency=afterok:<id>`. `--parsable` may print `id;cluster`, so strip with `${id%%;*}`.
- Example: `inf=$(DATASET=wikitext sbatch --parsable bluehive_run/exp2-inference.sbatch); DATASET=wikitext sbatch --parsable --dependency=afterok:${inf%%;*} bluehive_run/exp2-analysis.sbatch`.
- Partition `gpu`: A100 and V100 nodes.

## Monitoring
- Queue: `squeue -u lli81 -o "%i %j %T %M %l %R"`.
- History: `sacct -u lli81 -S <YYYY-MM-DD> -X -o JobID,JobName%35,State,Elapsed,ExitCode`.
- Logs: `bluehive_run/logs/slurm-<job-name>-<id>.{out,err}` plus the per-job directory `bluehive_run/logs/slurm-<job-name>-<id>/` holding the per-command output files.
- A dependent job whose parent fails stays pending with reason `DependencyNeverSatisfied`. Cancel it with `scancel <id>` after reporting.

## Pulling results
- Analysis outputs: run `scripts/sync_circ_analysis.sh` locally (extra args go to rsync, e.g. `-n`). Single files: `rsync -avP circ:<repo root>/<path> <local path>`.
