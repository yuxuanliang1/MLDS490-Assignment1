"""Launch a selected remote stage in tmux; usable through `ssh python3 - ...`."""
import argparse
from pathlib import Path
import shlex
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--stage", choices=["pilots", "formal"], required=True)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--discount-weighting", choices=["relative", "absolute"], default="relative")
    args = parser.parse_args()
    command = shlex.join(["bash", "scripts/run_pong_gpu_pipeline.sh", args.gpu, args.python,
                          str(args.episodes), args.stage, args.discount_weighting])
    session = "mlds490-pong-"+args.discount_weighting+"-"+args.stage
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(args.root), command], check=True)
    print(f"Started {session} in {args.root}")
