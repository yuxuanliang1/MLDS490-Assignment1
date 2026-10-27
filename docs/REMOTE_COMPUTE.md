# Deepdish: local source, remote compute, separate evaluation

**Both Part 1 and Part 2 training/evaluation on Deepdish5 GPU 1 are complete.
The dated discovery and launch notes below are historical reproduction records.** See [Part 2 status](PART2_STATUS.md) and
[the completed Part 1 run record](PONG_ACTIVE_RUN.md) for the
benchmark, detached process, configuration, logs and checkpoints. Deepdish3
discovery/pilot notes below are historical fallback and reproduction information.

## Observed and executed on 2026-10-26

**Earlier selected host, now fallback: `deepdish3`, physical GPU 4 (RTX 2080 Ti, CC 7.5).** Direct
single-GPU use was explicitly confirmed by the user. See the
[five-host comparison](DEEPDISH_COMPARISON.md). The isolated environment is
`/nfs/home/fqu0882/.venvs/mlds490-pg-deepdish3`, with Python 3.10.12,
PyTorch 2.8.0+cu128, Gymnasium 1.4.0 and ALE 0.10.2. The deployed source is
`/nfs/home/fqu0882/mlds490/releases/part1-20261026-run1`; its 43 source hashes
were verified before execution.

CPU numerical/startup validation and the GPU smoke passed in `tmux`: real
forward/backward kernels, two time-limited diagnostic rollouts, two complete
training games, checkpoints, and two separately loaded diagnostic evaluation
games. Complete training games took 7.428 and 4.106 seconds; peak allocated
tensor memory was 34,109,440 bytes. The two-game evaluation scored -21 with
zero sample SD; **this is not a learned final result**. Logs/checkpoints/rewards
and `remote/run1/gpu_smoke_passed.json` have been retrieved locally.

Bounded GPU pilots completed 30 games each at 0.01/0.003/0.001, sequentially on
that same GPU in `tmux`. Their means were -20.433/-20.467/-20.333; the predeclared
last-20 training-reward rule selected 0.001. All remained stochastic; this weak
separation is not evidence of LR superiority or meaningful learning. Measured
runtime averaged 4.049 seconds/game. Complete artifacts were retrieved locally.
Formal training and the final 500-game evaluation have not started; final
experiment configuration is pending. The pilot session exits normally when done.
For a running future stage, attach to its matching tmux session; completed pilot
logs remain available:

```powershell
ssh -t deepdish3 'tmux attach -t mlds490-pong-pilots'
ssh deepdish3 'tail -n 20 ~/mlds490/releases/part1-20261026-run1/results/policy_gradient_part1/remote/gpu_pipeline_pilots.log'
```

Inspect `pong/gpu_pilot_selection.json` in that release or the local results.
Selection uses training rewards and collapse diagnostics, never final evaluation.
The separate formal driver stage (`--stage formal --formal-episodes <BUDGET>`)
launches the 500-game evaluator in a fresh process only after training completes.
`launch_remote_stage.py --stage formal --episodes <BUDGET>` starts it in tmux;
see `--help` for required root, GPU UUID, and Python arguments.

### Initial Deepdish1 inspection

The authorized SSH inspection succeeded as `fqu0882` on
`mlds-deepdish1.ads.northwestern.edu`. See
`results/policy_gradient_part1/remote/inspection.txt` for actual command output.

- Ubuntu 22.04.5; `/usr/bin/python3` is 3.10.12.
- Four NVIDIA GeForce GTX 1080 cards, each 8192 MiB. GPU 0 was using 8071 MiB
  and 100% utilization at inspection. Availability is transient.
- Driver 580.178.04 advertises CUDA 13.0 compatibility; `nvcc` toolkit is 11.5.
  These are different from the runtime bundled in a PyTorch wheel.
- `/opt/anaconda/bin/python` is 3.9.12 with shared PyTorch 2.4.1+cu124,
  compiled for `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`.
  Gymnasium/ALE were absent. The shared environment was not changed.
- `sbatch`, `srun`, `sinfo`, `squeue`, and `qsub` were not on PATH.
  `tmux` and `screen` exist. Absence from PATH does not establish site policy.
- This initial inspection allocated no GPU. Subsequent authorized setup and
  GPU smoke ran only on the Deepdish3 device above. Other hosts were inspected
  without GPU kernels.

The project requires Python >=3.10; do not use the shared Python 3.9 directly.
Use an isolated home-directory venv. A known older PyTorch CUDA build is proposed
for these Pascal GPUs; the wrapper tests a real forward/backward kernel **after**
the allocation is permitted. See [official PyTorch wheel commands](https://pytorch.org/get-started/previous-versions/).
No CUDA compiler is needed to use that prebuilt wheel.

## Inspect again without running compute

From PowerShell (the local workspace):

```powershell
scp scripts/inspect_deepdish.sh fqu0882@mlds-deepdish1.ads.northwestern.edu:~/inspect_mlds490.sh
ssh fqu0882@mlds-deepdish1.ads.northwestern.edu 'bash ~/inspect_mlds490.sh'
```

Ask the course/admin which procedure allocates a GPU. Do not infer permission
from an idle card. If a scheduler is required, obtain its partition/account/time
and GPU request syntax, allocate exactly one GPU, and preserve the allocation's
`CUDA_VISIBLE_DEVICES`. No speculative Slurm submission file is supplied because
Slurm was not discovered here.

## Deploy authoritative local source

This workspace contained no `.git` metadata at inspection. No repository was
initialized and no commit/push/pull was attempted. If the real Git checkout is
available, inspect `git status` in both checkouts, commit/push reviewed local
source, then clone or fast-forward a clean remote checkout using its actual URL.
Do not overwrite remote edits or run destructive Git commands.

The supported fallback creates an allowlisted source archive with per-file hashes,
excluding credentials, venvs, checkpoints, and experimental results:

```powershell
.venv\Scripts\python.exe scripts/package_pg_source.py
scp tmp/pg_source.tar.gz fqu0882@mlds-deepdish1.ads.northwestern.edu:~/pg_source.tar.gz
ssh fqu0882@mlds-deepdish1.ads.northwestern.edu
```

In the remote shell, choose a **fresh** release directory for each deployment:

```bash
release="$HOME/mlds490/releases/part1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$release"
tar -xzf "$HOME/pg_source.tar.gz" -C "$release"
cd "$release"
python3 - <<'PY'
import hashlib,json,pathlib
for name,expected in json.loads(pathlib.Path('source_manifest.json').read_text()).items():
    assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() == expected, name
print('All deployed source hashes match')
PY
bash scripts/setup_pg_remote.sh
```

Setup uses only `~/.venvs/mlds490-pg`; it refuses an existing path. It performs
CPU validation but does not train or allocate a GPU. Review installation output
and keep `environment.txt`. Modern ALE 0.10.2 includes the Pong ROM used locally;
verify actual creation with `run_pg_validation.py --pong`. Do not substitute v4
if registration or ROM loading fails. See [ALE installation](https://ale.farama.org/main/getting-started/)
and [Pong specification](https://ale.farama.org/environments/pong/).

## Stages and persistence after policy confirmation

1. CPU smoke test, distinct from complete games or evidence of learning:

```bash
~/.venvs/mlds490-pg/bin/python scripts/train_pong.py --episodes 2 --batch-episodes 1 \
  --max-episode-steps 256 --output results/policy_gradient_part1/pong/cpu_smoke
```

2. If **direct single-GPU use in tmux is explicitly permitted**, start a named
   session and select a currently available, permitted physical card:

```bash
tmux new-session -s mlds490-pong
# Inside tmux, return to the deployed release directory.
nvidia-smi
export DEEPDISH_GPU_METHOD=direct-permitted
export CUDA_VISIBLE_DEVICES=1  # EXAMPLE ONLY: use the card assigned/permitted now
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
bash scripts/run_pong_remote.sh pilot results/policy_gradient_part1/pong/pilot_lr001 \
  --learning-rate 0.01 --seed 101
```

Run LR 0.003 and 0.001 pilots **sequentially**, in fresh output directories,
with equal episode budgets, seeds, and preprocessing. Optionally repeat with
`--raw-frames`. Compare reward, entropy, action frequencies, gradient norms,
episode duration, and peak GPU allocation. Determine an affordable formal
episode budget from measured runtime and training progression. Record the
decision before formal training; do not tune on final evaluation.

Detach with Ctrl-B D; reconnect with `tmux attach -t mlds490-pong`.
Logs go to each run's `process.log`. A normal disconnected SSH shell has no
persistence guarantee. If scheduler allocation is required, use its supported
persistent batch mechanism instead; tmux does not replace allocation.

3. Formal training (replace both placeholders with pilot-supported values):

```bash
bash scripts/run_pong_remote.sh train results/policy_gradient_part1/pong/formal \
  --episodes <BUDGET> --learning-rate <SELECTED_LR> --seed 11 \
  --checkpoint-interval 100 --log-interval 10 \
  --figure figures/policy_gradient_part1/pong/training_rewards.png
```

Resume in the same directory with the same algorithm/environment configuration
and a larger total target if needed, adding
`--resume results/policy_gradient_part1/pong/formal/latest.pt`. Checkpoints include
weights, Adam state, RNG state, complete committed history, and config. They are
saved atomically at update boundaries. SIGINT/SIGTERM requests finish the current
update and save; a hard kill can lose work since the last checkpoint. Resume
recovers CSV history from that checkpoint. Episode reset seeds are explicit;
mid-episode emulator state and cross-device bitwise identity are not promised.
Only load trusted `.pt` files (Python/NumPy state uses pickle).

4. Final evaluation in a **separate invocation**, once the checkpoint is selected:

```bash
bash scripts/run_pong_remote.sh evaluate results/policy_gradient_part1/pong/evaluation \
  --checkpoint results/policy_gradient_part1/pong/formal/latest.pt --seed 1000000 \
  --figure figures/policy_gradient_part1/pong/evaluation_histogram.png
```

This requires exactly 500 games, stochastic sampling, matching preprocessing,
and no truncated Pong games. It writes every raw reward, sample SD (`ddof=1`),
seed, checkpoint hash, and settings. A partial/failed evaluation has no completed
summary; use a fresh directory to restart. Evaluate a random/fixed reference
separately with `evaluate_pg.py --role diagnostic --reference random` or `fixed`.
This does not update the learned policy or subtract a training baseline.

The wrapper enforces one visible GPU, tests CUDA, logs process/GPU information,
and holds a per-user lock to prevent overlapping Pong GPU tasks. It does not
reserve resources against other users. An OOM raises an error; reduce the
gradient chunk size in a new documented configuration, not the number of GPUs.

## Retrieve artifacts without overwriting local source

After completion, in the remote release directory:

```bash
tar -czf "$HOME/pg_artifacts.tar.gz" source_manifest.json \
  results/policy_gradient_part1/pong figures/policy_gradient_part1/pong
```

From local PowerShell:

```powershell
New-Item -ItemType Directory -Force remote_artifacts | Out-Null
scp fqu0882@mlds-deepdish1.ads.northwestern.edu:~/pg_artifacts.tar.gz remote_artifacts/
tar -xzf remote_artifacts/pg_artifacts.tar.gz -C remote_artifacts
```

Inspect the downloaded source manifest, configs, checkpoint hashes, process logs,
episode counts, reward CSVs, and numerical summaries. Run
`python scripts/audit_pg_artifacts.py` after explicitly copying reviewed Pong
results/figures into their matching local component directories. Retain the
remote checkpoints and a local backup; `.pt` files are ignored by Git. Source
changes always originate locally and deploy as a new snapshot/commit. Build the
unified report locally after importing verified results. No automatic artifact
import or remote job submission is hidden in these scripts.
