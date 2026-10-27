#!/usr/bin/env bash
# Launched under nohup/setsid or tmux; no dependency on the Codex connection.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${1:?Exactly one selected GPU UUID required}"
PG_PYTHON="${2:?Python executable required}"
PG_STAGE="${3:?benchmark or formal}"
mkdir -p "$HOME/.cache/mlds490-pg" results/policy_gradient_part1/remote
exec 9>"$HOME/.cache/mlds490-pg/pong-gpu.lock"
flock -n 9 || { echo 'Another Pong GPU task holds the account lock.' >&2; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PG_PYTHON" scripts/verify_source_snapshot.py
date -u
nvidia-smi
if [ "$PG_STAGE" = benchmark ]; then
    "$PG_PYTHON" -u scripts/benchmark_pong.py --episodes 10 --output results/policy_gradient_part1/benchmarks/selected_gpu_10ep
elif [ "$PG_STAGE" = formal ]; then
    "$PG_PYTHON" -u scripts/train_pong.py --config configs/pong_part1_absolute.json \
      --discount-weighting absolute --device cuda:0 --episodes 1000 --learning-rate 0.01 \
      --checkpoint-interval 50 --log-interval 10 --output results/policy_gradient_part1/pong/formal \
      --figure figures/policy_gradient_part1/pong/training_rewards.png
    "$PG_PYTHON" -u scripts/evaluate_pg.py --device cuda:0 --episodes 500 --seed 1000000 \
      --checkpoint results/policy_gradient_part1/pong/formal/latest.pt \
      --output results/policy_gradient_part1/pong/evaluation \
      --figure figures/policy_gradient_part1/pong/evaluation_histogram.png
    tar -czf "$HOME/pg_absolute_completed_artifacts.tar.gz" source_manifest.json \
      results/policy_gradient_part1/pong/formal results/policy_gradient_part1/pong/evaluation \
      figures/policy_gradient_part1/pong
else
    echo 'Unknown stage' >&2; exit 1
fi
date -u
