"""Launch the explicit selected-GPU smoke in tmux, from a deployed snapshot."""
import argparse
from pathlib import Path
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--session", default="mlds490-pong-smoke")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Arguments are joined with shell quoting inside tmux's single shell-command field.
    import shlex
    command = shlex.join(["bash", "scripts/remote_gpu_smoke.sh", args.gpu, args.python])
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session, "-c", str(root), command], check=True)
    print(f"Started tmux session {args.session}; log: {root}/results/policy_gradient_part1/remote/gpu_smoke.log")
