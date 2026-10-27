"""Locked sequential GPU smoke or formal training -> evaluation -> analysis."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
PART1=Path('/nfs/home/fqu0882/mlds490/releases/part1-20261026-blackwell')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True)
    p.add_argument('--stage',choices=['smoke','formal'],required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    lock=Path.home()/'.cache/mlds490-pg/pong-gpu.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        final=json.loads((PART1/'results/policy_gradient_part1/pong/evaluation/summary.json').read_text())
        if final['episodes']!=500 or final['truncated_count'] or final['terminated_count']!=500:
            raise RuntimeError('Part 1 final evaluation is not complete')
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                               '--format=csv,noheader'],text=True)
        if any(line.split(',')[0].strip()==a.gpu for line in gpu_processes.splitlines()):
            raise RuntimeError('Selected GPU is occupied; no kernels launched')
        def run(*args):
            subprocess.run([sys.executable,'-u',*map(str,args)],check=True,cwd=ROOT)
        run('scripts/verify_source_snapshot.py')
        if a.stage=='smoke':
            run('scripts/validate_pg2.py','--device','cuda:0','--no-integration',
                '--output','results/policy_gradient_part2/gpu_validation')
            for episodes in (2,4):
                extra=['--resume','results/policy_gradient_part2/pong/gpu_smoke/latest.pt'] if episodes==4 else []
                run('scripts/train_pg2.py','--config','configs/pong_part2.json','--device','cuda:0',
                    '--episodes',episodes,'--output','results/policy_gradient_part2/pong/gpu_smoke',*extra)
            run('scripts/evaluate_pg.py','--device','cuda:0','--episodes',2,'--role','diagnostic',
                '--checkpoint','results/policy_gradient_part2/pong/gpu_smoke/latest.pt',
                '--output','results/policy_gradient_part2/pong/gpu_smoke_evaluation')
            sys.path.insert(0,str(ROOT))
            from policy_gradient_part1.runtime import load_checkpoint,write_json
            ck=load_checkpoint('results/policy_gradient_part2/pong/gpu_smoke/latest.pt')
            assert ck['baseline']['count']==4 and ck['completed_episodes']==4 and ck['optimizer']['state']
            assert ck['history'][2]['baseline_used']==ck['history'][1]['baseline_after_episode']
            assert ck['metadata']['gpu']['capability']==(12,0) or ck['metadata']['gpu']['capability']==[12,0]
            write_json('results/policy_gradient_part2/remote/gpu_smoke_passed.json',dict(
                passed=True,complete_training_games=4,separate_evaluation_games=2,
                baseline_state=ck['baseline'],baseline_used_after_resume=ck['history'][2]['baseline_used'],
                gpu=ck['metadata']['gpu']))
            print('GPU smoke passed, including nonzero baseline after resume.',flush=True)
            return
        passed=ROOT/'results/policy_gradient_part2/remote/gpu_smoke_passed.json'
        if not passed.exists() or not json.loads(passed.read_text())['passed']:
            raise RuntimeError('Successful smoke required before formal launch')
        training=ROOT/'results/policy_gradient_part2/pong/formal'
        extra=['--resume',training/'latest.pt'] if a.resume else []
        run('scripts/train_pg2.py','--config','configs/pong_part2.json','--device','cuda:0',
            '--output',training,'--figure','figures/policy_gradient_part2/pong/training_rewards.png',*extra)
        summary=json.loads((training/'summary.json').read_text())
        if not summary['complete'] or summary['completed_episodes']!=1000:
            raise RuntimeError('Training stopped early; final evaluation not started')
        run('scripts/evaluate_pg.py','--device','cuda:0','--episodes',500,'--seed',1000000,
            '--checkpoint',training/'latest.pt','--output','results/policy_gradient_part2/pong/evaluation',
            '--figure','figures/policy_gradient_part2/pong/evaluation_histogram.png')
        run('scripts/diagnose_pg2.py','--device','cuda:0','--episodes',64,
            '--checkpoint',training/'latest.pt','--output','results/policy_gradient_part2/pong/gradient_diagnostics',
            '--figure','figures/policy_gradient_part2/pong/gradient_variance.png')
        sys.path.insert(0,str(ROOT))
        from policy_gradient_part2.analysis import compare
        from policy_gradient_part1.runtime import write_json
        compare(PART1/'results/policy_gradient_part1/pong',ROOT/'results/policy_gradient_part2/pong',
                ROOT/'figures/policy_gradient_part2/pong/part1_vs_part2.png',
                'Pong: gamma^t weighting; only training baseline differs')
        write_json('results/policy_gradient_part2/remote/completed.json',dict(
            complete=True,training_episodes=1000,final_evaluation_episodes=500,
            gpu_uuid=a.gpu,baseline='EMA previous discounted G0, alpha .05',
            checkpoint_sha256=hashlib.sha256((training/'latest.pt').read_bytes()).hexdigest()))
        with tarfile.open(Path.home()/'pg_part2_completed_artifacts.tar.gz','w:gz') as archive:
            for name in ['source_manifest.json','results/policy_gradient_part2','figures/policy_gradient_part2']:
                archive.add(ROOT/name,arcname=name)
        print('PART2 PIPELINE COMPLETE',flush=True)


if __name__=='__main__':
    main()
