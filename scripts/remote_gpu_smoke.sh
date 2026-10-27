#!/usr/bin/env bash
# Run via tmux after inspecting and selecting exactly one idle GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${1:?Selected GPU UUID required}"
export PG_PYTHON="${2:?Isolated Python executable required}"
export DEEPDISH_GPU_METHOD=direct-permitted
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p results/policy_gradient_part1/remote
exec > >(tee results/policy_gradient_part1/remote/gpu_smoke.log) 2>&1
date -u
bash scripts/run_pong_remote.sh pilot results/policy_gradient_part1/pong/gpu_smoke \
  --episodes 2 --batch-episodes 1 --max-episode-steps 256 --checkpoint-interval 1 --log-interval 1
# Verify complete uncapped gameplay, not just a forced truncation smoke.
bash scripts/run_pong_remote.sh pilot results/policy_gradient_part1/pong/gpu_full_game_smoke \
  --episodes 2 --batch-episodes 1 --checkpoint-interval 1 --log-interval 1
"$PG_PYTHON" scripts/evaluate_pg.py \
  --checkpoint results/policy_gradient_part1/pong/gpu_full_game_smoke/latest.pt \
  --output results/policy_gradient_part1/pong/gpu_smoke_evaluation \
  --episodes 2 --role diagnostic --device cuda:0 --seed 7000000
"$PG_PYTHON" - <<'PY'
import json,pathlib,torch
root=pathlib.Path('results/policy_gradient_part1')
checkpoint=torch.load(root/'pong/gpu_full_game_smoke/latest.pt',map_location='cpu',weights_only=False)
assert checkpoint['completed_episodes']==2
assert all(row['terminated'] and not row['truncated'] for row in checkpoint['history'])
assert all(row['gpu_peak_memory_bytes']>0 for row in checkpoint['history'])
summary=json.loads((root/'pong/gpu_smoke_evaluation/summary.json').read_text())
assert summary['episodes']==2 and summary['truncated_count']==0
(root/'remote/gpu_smoke_passed.json').write_text(json.dumps(dict(
    passed=True, gpu=checkpoint['metadata']['gpu'], history=checkpoint['history'],
    evaluation=summary),indent=2)+'\n')
print('GPU SMOKE PASSED: real forward/backward, checkpoints, complete games, separate evaluation')
PY
date -u
