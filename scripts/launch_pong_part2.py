"""Detach the Part 2 pipeline; save PID, session, command, and durable log path."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    out=root/'results/policy_gradient_part2/remote';out.mkdir(parents=True,exist_ok=True)
    log=out/('formal_resume.log' if a.resume else 'formal.log')
    command=['nohup',sys.executable,'-u','scripts/pong_part2_pipeline.py','--gpu',a.gpu,'--stage','formal']
    if a.resume: command+=['--resume']
    with log.open('xb') as stream:
        process=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=process.pid,log=str(log),root=str(root),command=command,gpu_uuid=a.gpu,
                persistence='nohup, new POSIX session, closed stdin, file stdout/stderr')
    (out/'process.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
