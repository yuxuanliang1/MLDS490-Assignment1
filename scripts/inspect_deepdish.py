"""Read-only host inventory. Pipe to `ssh deepdishN python3 -`; no GPU kernels."""
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def command(args, timeout=30):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return dict(exit_code=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except Exception as exc:
        return dict(error=str(exc))


def main():
    report = dict(hostname=platform.node(), inspected_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                  cpu_count=os.cpu_count(), cpu_affinity_count=len(os.sched_getaffinity(0)),
                  load_average=os.getloadavg(),
                  tools={name: shutil.which(name) for name in
                         ("python", "python3", "conda", "tmux", "screen", "sbatch", "srun", "qsub", "nvcc")})
    report["cpu_model"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                               if line.startswith("model name")), "unknown")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            memory[key+"_KiB"] = int(value.strip().split()[0])
    report["memory"] = memory
    report["nvidia_smi"] = command(["nvidia-smi"])
    match = re.search(r"CUDA(?: UMD)? Version:\s*([\d.]+)", report["nvidia_smi"].get("stdout", ""))
    report["driver_cuda_compatibility"] = match.group(1) if match else None
    query = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version,compute_cap",
                     "--format=csv,noheader,nounits"])
    report["gpu_query"] = query
    if query.get("exit_code") == 0:
        names = ("index", "uuid", "name", "total_MiB", "used_MiB", "free_MiB", "utilization_percent", "driver", "compute_capability")
        report["gpus"] = [dict(zip(names, [value.strip() for value in row]))
                          for row in csv.reader(io.StringIO(query["stdout"]))]
    report["gpu_processes"] = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                        "--format=csv,noheader,nounits"])
    report["nvcc"] = command(["nvcc", "--version"])
    candidates = [sys.executable, "/opt/anaconda/bin/python", "/opt/conda/bin/python",
                  "/opt/miniconda3/bin/python", str(Path.home()/"miniconda3/bin/python")]
    report["python_environments"] = {}
    probe = """import importlib.util,json,sys
r={'version':sys.version,'executable':sys.executable,'packages':{n:bool(importlib.util.find_spec(n)) for n in ['torch','gymnasium','ale_py']}}
if r['packages']['torch']:
 import torch
 r.update(torch_version=torch.__version__,wheel_cuda=torch.version.cuda,compiled_arches=torch._C._cuda_getArchFlags())
print(json.dumps(r))
"""
    for candidate in dict.fromkeys(candidates):
        if Path(candidate).is_file():
            result = command([candidate, "-c", probe])
            try:
                report["python_environments"][candidate] = json.loads(result["stdout"])
            except Exception:
                report["python_environments"][candidate] = result
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
