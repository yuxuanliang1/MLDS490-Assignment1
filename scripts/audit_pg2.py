"""Audit persisted numbers, baseline timing, checkpoints, and preserved prior artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from policy_gradient_part1.config import Config
from policy_gradient_part1.runtime import load_checkpoint,write_json
from policy_gradient_part2.baseline import RunningConstantBaseline


def audit():
    checks={};pending=[]
    def check(name,condition):
        checks[name]=bool(condition)
        if not condition: raise AssertionError(name)
    manifest=json.loads((ROOT/'results/policy_gradient_part2/preservation_manifest.json').read_text())
    check('all_360_prior_files_preserved',all((ROOT/n).exists() and hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
                                             for n,h in manifest.items()))
    v=json.loads((ROOT/'results/policy_gradient_part2/validation/validation.json').read_text())
    check('mathematical_and_resume_checks',v['all_passed'])
    for name,primary in [('cartpole','seed_11'),('pong','formal')]:
        root=ROOT/f'results/policy_gradient_part2/{name}';run=root/primary
        reference=ROOT/f'results/policy_gradient_part1/{name}/{primary}'
        planned=Config.load(ROOT/f'configs/{name}_part2.json')
        original=Config.load(reference/'config.json')
        check(name+'_planned_settings_match_actual_part1',
              dict(planned.to_dict(),device=original.device)==original.to_dict())
        if not (run/'summary.json').exists():pending.append(name+' training');continue
        cfg=Config.load(run/'config.json');ref=Config.load(ROOT/f'configs/{name}_part2.json')
        check(name+'_matched_config',cfg.to_dict()==dict(ref.to_dict(),device=cfg.device))
        ck=load_checkpoint(run/'latest.pt');history=ck['history'];batch=cfg.batch_episodes
        with (reference/'history.csv').open(encoding='utf-8') as f:
            prior=list(csv.DictReader(f))
        columns=['reward','discounted_return','steps','loss_contribution','entropy','probability_action0','action0_fraction']
        check(name+'_first_zero_baseline_batch_matches_part1',all(
            all(np.isclose(float(a[k]),float(b[k]),rtol=1e-6,atol=1e-6) for k in columns)
            for a,b in zip(history[:batch],prior[:batch])))
        b=RunningConstantBaseline()
        for start in range(0,len(history),batch):
            before=b.estimate()
            for row in history[start:start+batch]:
                check(f'{name}_episode_{row["episode"]}_baseline_is_past_only',row['baseline_used']==before)
                b.update(row['discounted_return'])
                if row['baseline_after_episode']!=b.estimate():raise AssertionError('EMA recurrence')
        check(name+'_baseline_checkpoint_count',b.state_dict()==ck['baseline'] and b.count==ck['completed_episodes']==len(history))
        check(name+'_optimizer_saved',bool(ck['optimizer']['state']))
        check(name+'_on_policy_recomputed_logprobs',max(r['logprob_recompute_max_error'] for r in history)<1e-4)
        ma=np.convolve([r['reward'] for r in history],np.ones(100)/100,'valid')
        check(name+'_saved_MA100',np.allclose(ma,[r['moving_average_100'] for r in history[99:]]))
        for seed in (11,22,33) if name=='cartpole' else ():
            d=root/f'seed_{seed}'
            if not (d/'summary.json').exists():pending.append(f'cartpole seed {seed}');continue
            sc=load_checkpoint(d/'latest.pt')
            check(f'cartpole_seed{seed}_baseline_persisted',sc['baseline']['count']==sc['completed_episodes']==1500)
        evaluation=root/'evaluation'
        if not (evaluation/'summary.json').exists():pending.append(name+' 500-episode evaluation');continue
        summary=json.loads((evaluation/'summary.json').read_text())
        rows=list(csv.DictReader((evaluation/'rewards.csv').open(encoding='utf-8')))
        rewards=np.array([float(r['reward']) for r in rows])
        check(name+'_exact_500',len(rewards)==summary['episodes']==500)
        check(name+'_evaluation_statistics',abs(rewards.mean()-summary['mean'])<1e-10 and
              abs(rewards.std(ddof=1)-summary['sample_std'])<1e-10)
        check(name+'_evaluation_checkpoint_hash',hashlib.sha256((run/'latest.pt').read_bytes()).hexdigest()==summary['checkpoint_sha256'])
        if name=='pong':check('pong_all_complete',all(r['terminated']=='True' and r['truncated']=='False' for r in rows))
        for figure in ('training_rewards','evaluation_histogram','part1_vs_part2','gradient_variance'):
            path=ROOT/f'figures/policy_gradient_part2/{name}/{figure}.png'
            if not path.exists():pending.append(name+' '+figure);continue
            with Image.open(path) as im: check(name+'_'+figure+'_image',min(im.size)>=1000 and im.info.get('dpi',(0,0))[0]>290)
        diag=root/'gradient_diagnostics'
        if (diag/'summary.json').exists():
            ds=json.loads((diag/'summary.json').read_text());data=np.load(diag/'gradient_statistics.npz')
            check(name+'_gradient_statistics_finite',all(np.isfinite(data[k]).all() for k in data.files))
            check(name+'_gradient_variance_matches',np.isclose(data['variance_without'].sum(),ds['trace_covariance_without']) and
                  np.isclose(data['variance_with'].sum(),ds['trace_covariance_with']))
    result=dict(all_available_checks_passed=all(checks.values()),checks_count=len(checks),
                preserved_files=len(manifest),pending=pending,
                scope='Pending remote artifacts are not treated as completed results')
    write_json(ROOT/'results/policy_gradient_part2/audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':audit()
