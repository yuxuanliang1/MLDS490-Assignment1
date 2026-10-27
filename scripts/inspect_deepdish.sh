#!/usr/bin/env bash
# Read-only inspection; no GPU kernels, installs, scheduler submission or environment changes.
set -u
date -u
hostname
id -un
cat /etc/motd 2>/dev/null || true
for tool in python python3 conda sbatch srun squeue sinfo qsub tmux screen nvidia-smi nvcc; do
    command -v "$tool" || true
done
python3 --version
nvidia-smi
nvcc --version 2>/dev/null || true
if [ -x /opt/anaconda/bin/python ]; then
    /opt/anaconda/bin/python --version
    /opt/anaconda/bin/conda env list
    /opt/anaconda/bin/python -m pip show torch gymnasium ale-py 2>/dev/null || true
    /opt/anaconda/bin/python - <<'PY'
import torch
print('torch:', torch.__version__, 'wheel CUDA:', torch.version.cuda)
print('compiled architectures (build flag):', torch._C._cuda_getArchFlags())
PY
fi
free -h
df -h ~
printf '%s\n' 'Inspection does not establish permission for direct GPU use. Confirm the site allocation policy.'
