"""Provenance, one-device enforcement, RNG snapshots and atomic checkpoints."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure_runtime(config):
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda:0")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        if torch.cuda.device_count() != 1 or device.index not in (None, 0):
            raise RuntimeError("Expose exactly ONE allocated GPU via CUDA_VISIBLE_DEVICES; use cuda:0")
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return device


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(0) if device.type == "cuda" else None)


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), 0)


def save_checkpoint(path, model, optimizer, config, completed, history, device, metadata, baseline=None):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    payload = dict(format_version=1, kind="resumable_episode_boundary", model=model.state_dict(),
                    optimizer=optimizer.state_dict(), config=config.to_dict(), completed_episodes=completed,
                    history=history, rng=rng_state(device), metadata=metadata,
                    continuation="Per-episode reset seed = training seed + zero-based episode. "
                    "No mid-episode emulator state; no cross-device bitwise guarantee.")
    if baseline is not None:
        payload["baseline"] = baseline.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Contains Python/NumPy RNG state: only load checkpoints from trusted runs.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def provenance(config, model):
    packages = {}
    for name in ("numpy", "torch", "gymnasium", "ale-py", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ("policy_gradient_part1", "scripts", "configs")
                     for p in sorted((ROOT / folder).glob("*")) if p.suffix in (".py", ".json")}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    result = dict(python=sys.version, platform=platform.platform(), packages=packages,
                  git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
                  git_status=status.stdout if status.returncode == 0 else "No Git metadata",
                  source_sha256=source_hashes, model=str(model), parameters=sum(p.numel() for p in model.parameters()),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), pid=os.getpid(),
                  objective="negative time-sum of sampled log probabilities times "
                  + ("gamma^t G_t (strict initial-state discounted objective)" if config.discount_weighting == "absolute"
                     else "G_t (relative-discount reference, no outer gamma^t)")
                  + "; mean over episodes; no baseline, entropy bonus or return scaling")
    if config.device.startswith("cuda"):
        result["gpu"] = dict(name=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                             torch_cuda=torch.version.cuda, compiled_arches=torch.cuda.get_arch_list())
    return result
