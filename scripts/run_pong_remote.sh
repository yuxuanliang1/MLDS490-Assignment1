#!/usr/bin/env bash
# Start inside an approved persistent session/allocation. Never launches itself.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${DEEPDISH_GPU_METHOD:-}" != direct-permitted ] && [ "${DEEPDISH_GPU_METHOD:-}" != scheduler-allocated ]; then
    printf '%s\n' 'Confirm site policy; set DEEPDISH_GPU_METHOD to direct-permitted or scheduler-allocated.' >&2
    exit 1
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    printf '%s\n' 'Expose exactly one assigned GPU with CUDA_VISIBLE_DEVICES.' >&2
    exit 1
fi
PG_PYTHON="${PG_PYTHON:-$HOME/.venvs/mlds490-pg/bin/python}"
PG_STAGE="${1:?Usage: bash scripts/run_pong_remote.sh pilot|train|evaluate RUN_DIRECTORY [arguments]}"
PG_RUN="${2:?Supply a new run directory}"
shift 2
mkdir -p "$HOME/.cache/mlds490-pg" "$PG_RUN"
# One simultaneous Pong GPU process per account, including evaluation.
exec 9>"$HOME/.cache/mlds490-pg/pong-gpu.lock"
flock -n 9 || { printf '%s\n' 'Another Pong GPU task holds this account lock.' >&2; exit 1; }
exec > >(tee -a "$PG_RUN/process.log") 2>&1
date -u
printf 'host=%s pid=%s visible_device=%s method=%s\n' "$(hostname)" "$$" "$CUDA_VISIBLE_DEVICES" "$DEEPDISH_GPU_METHOD"
nvidia-smi
"$PG_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('compiled architectures:', torch.cuda.get_arch_list())
x = torch.randn(16, 16, device='cuda:0', requires_grad=True)
x.square().sum().backward()
torch.cuda.synchronize()
print('Allocated-device forward/backward kernel passed')
PY
case "$PG_STAGE" in
    pilot)
        "$PG_PYTHON" scripts/train_pong.py --device cuda:0 --episodes 10 --output "$PG_RUN" "$@"
        ;;
    train)
        # Require an explicit budget justified by pilot runtime and learning diagnostics.
        if [[ " $* " != *' --episodes '* ]]; then
            printf '%s\n' 'Formal training requires explicit --episodes and a documented pilot-based budget.' >&2
            exit 1
        fi
        "$PG_PYTHON" scripts/train_pong.py --device cuda:0 --output "$PG_RUN" "$@"
        ;;
    evaluate)
        "$PG_PYTHON" scripts/evaluate_pg.py --device cuda:0 --episodes 500 --output "$PG_RUN" "$@"
        ;;
    *) printf '%s\n' 'Unknown stage' >&2; exit 1 ;;
esac
date -u
