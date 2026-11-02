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
- exp2 inference and analysis default to partition `h100`, QoS `h100`, and `gpu:H100:1` (5 nodes, 4 H100 GPUs per node). Keep both stages on this partition; the account `xxu67_lab` has `h100` QoS access.
- When moving an existing exp2 run between partitions, cancel its old inference and dependent analysis jobs before resubmitting. Both stages overwrite the same dataset output directories; rebuild each `afterok` dependency with the new inference job ID.

## Monitoring
- Queue: `squeue -u lli81 -o "%i %j %T %M %l %R"`.
- History: `sacct -u lli81 -S <YYYY-MM-DD> -X -o JobID,JobName%35,State,Elapsed,ExitCode`.
- Logs: `bluehive_run/logs/slurm-<job-name>-<id>.{out,err}` plus the per-job directory `bluehive_run/logs/slurm-<job-name>-<id>/` holding the per-command output files.
- A dependent job whose parent fails stays pending with reason `DependencyNeverSatisfied`. Cancel it with `scancel <id>` after reporting.

## Pulling results
Use the existing `scripts/sync_circ_analysis.sh` locally after the relevant jobs report `COMPLETED` with exit code `0:0`. Do not reimplement its rsync filters.

### Scope and safety
- The script pulls **all experiments'** `<experiment>/<run>/analysis/` trees from CIRC into local `data/experiment_results/`, preserving their directory layout. It is not restricted to the most recent jobs or exp2.
- It downloads analysis metadata, SVD factors and plots, not `inference/` matrices, training checkpoints or Slurm logs. Request those separately only when needed.
- Defaults: `REMOTE=circ`, `REMOTE_ROOT=/scratch/lli81/projects/ur_research/lm-svd`. Override these environment variables when targeting another checkout. The script resolves the local repo root itself.
- Extra arguments go to rsync. The script uses `--delete`, so preview changes first. Use `--max-delete=0` when downloading to preserve local-only results; do not use `--delete-excluded` or delete unrelated local files to silence a warning.
- `--max-delete=0` forbids deletions; rsync can exit with code 25 if deletions were skipped. Inspect the reported paths before proceeding, rather than rerunning without the guard.

### Download
```bash
# Preview additions, updates and potential deletions.
RSYNC_RSH='ssh -o BatchMode=yes' bash scripts/sync_circ_analysis.sh -n --itemize-changes

# Download without deleting local-only files.
RSYNC_RSH='ssh -o BatchMode=yes' bash scripts/sync_circ_analysis.sh --max-delete=0 --itemize-changes
```

Module-init noise is harmless. A `cannot delete non-empty directory` warning can arise for a local-only run whose contents are excluded from synchronization; leave that run intact and check rsync's exit status and the requested outputs.

### Verify and report
- For each requested run, check every expected centering under local `data/experiment_results/<experiment>/<run>/analysis/`. exp2 expects `none`, `global`, `row`, and `deflate1`, each containing `metadata.json`, `svd_U.npy`, `svd_S.npy`, `svd_Vt.npy`, and `svd_spectrum.png`.
- Check `metadata.json`: `centering.strategy` matches the directory; `input.matrix_shape` matches the run. For current exp2, verify `input.inference_metadata.value` describes no-BOS PMI and `input.inference_metadata.source.kind` identifies the expected `parquet` or `token_stream` source; check its paths and sample IDs or sampled starts and settings against the requested run. The first token is unscored, and the history log probability cancels. Full wikitext and task-single-text Parquet matrices are `[512, 512]` and `[1309, 1309]`, respectively.
- Load SVD factors with NumPy; check compatible dimensions and finite values, descending nonnegative singular values, and a readable spectrum plot. For byte-for-byte transfer verification, compare local and remote SHA-256 hashes of the downloaded files.
- Report the local paths, datasets/centerings downloaded, verification and anything intentionally excluded. Do not claim inference matrices were downloaded by this script.
- For a separately requested file, use `rsync -avP -e 'ssh -o BatchMode=yes' circ:<repo-root>/<path> <local-path>`.
