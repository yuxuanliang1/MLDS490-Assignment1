"""Detach a bounded benchmark or formal/evaluation pipeline with durable logs."""
import argparse
import json
from pathlib import Path
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--stage", choices=["benchmark", "formal"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "results/policy_gradient_part1/remote"
    output.mkdir(parents=True, exist_ok=True)
    log = output / f"{args.stage}_independent.log"
    with log.open('xb') as handle:
        process = subprocess.Popen(['nohup', 'bash', 'scripts/run_pong_independent.sh', args.gpu, args.python, args.stage],
                                   cwd=root, stdin=subprocess.DEVNULL, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(pid=process.pid, stage=args.stage, log=str(log), root=str(root),
                  gpu_uuid=args.gpu, python=args.python, persistence='nohup + new POSIX session; stdin closed; durable stdout/stderr')
    (output / f"{args.stage}_process.json").write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record), flush=True)
