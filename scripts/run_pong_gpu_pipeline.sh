#!/usr/bin/env bash
# Explicitly invoked only after resource permission and the GPU smoke pass.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${1:?Selected GPU UUID required}"
PG_PYTHON="${2:?Isolated Python executable required}"
PG_FORMAL_EPISODES="${3:?Explicit pilot-supported total episode budget required}"
PG_STAGE="${4:-pilots}"
PG_WEIGHTING="${5:-relative}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$HOME/.cache/mlds490-pg" results/policy_gradient_part1/remote
exec 9>"$HOME/.cache/mlds490-pg/pong-gpu.lock"
flock -n 9 || { echo 'Another Pong GPU task holds the account lock.' >&2; exit 1; }
exec > >(tee "results/policy_gradient_part1/remote/gpu_pipeline_${PG_WEIGHTING}_${PG_STAGE}.log") 2>&1
date -u
"$PG_PYTHON" scripts/verify_source_snapshot.py
nvidia-smi
"$PG_PYTHON" scripts/run_pong_gpu_experiments.py --formal-episodes "$PG_FORMAL_EPISODES" --stage "$PG_STAGE" --discount-weighting "$PG_WEIGHTING"
date -u
