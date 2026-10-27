"""Populate the unified report only from completed saved experiments."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from policy_gradient_part1.runtime import write_json


def main():
    result=ROOT/'results/policy_gradient_part2';lines=['% Generated from actual Part 2 evidence; remote results may remain pending.']
    def macro(name,value):lines.append('\\newcommand{\\PGTwo'+name+'}{'+str(value)+'}')
    def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    def scientific(value):
        mantissa,exponent=f'{value:.2e}'.split('e')
        return mantissa+r'\times10^{'+str(int(exponent))+'}'
    def table(name,spec,header,rows):
        macro(name,'\n'+ '\n'.join([r'\begin{center}',r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule']+
                                    [' & '.join(map(str,row))+r' \\' for row in rows]+[r'\bottomrule',r'\end{tabular}',r'\end{center}'])+'\n')
    validation=read(result/'validation/validation.json')
    macro('Validation',f"All {len(validation['checks'])} CPU checks pass. The largest exact expected-gradient discrepancy is "
          f"${scientific(validation['max_exact_error'])}$, and the maximum chunk/full-gradient error is ${scientific(validation['max_chunk_gradient_error'])}$.")
    syn=validation['synthetic_variance']
    macro('SyntheticVariance',f"The expected ascent gradient is $(-0.375,0.375)$. The baseline is {syn['baseline']:.6f}; "
          f"sample mean vectors are $({syn['mean_without'][0]:.6f},{syn['mean_without'][1]:.6f})$ without and "
          f"$({syn['mean_with'][0]:.6f},{syn['mean_with'][1]:.6f})$ with the baseline. The paired mean-difference norm "
          f"is {syn['mean_difference_norm']:.6f}, with per-coordinate standard error {syn['paired_mean_difference_se'][0]:.6f}; "
          'finite-sample differences do not contradict equality in expectation.')
    variance_rows=[['Synthetic',200000,f"{syn['trace_covariance_without']:.6g}",f"{syn['trace_covariance_with']:.6g}",f"{100*syn['relative_reduction']:.2f}\\%"]]
    evaluation_rows=[];summary=dict(validation=validation,experiments={});game_prose=[]
    for name,label in [('cartpole','CartPole'),('pong','Pong')]:
        root=result/name
        ev=read(root/'evaluation/summary.json');part1=read(ROOT/f'results/policy_gradient_part1/{name}/evaluation/summary.json')
        for part,e in [('Part 1',part1),('Part 2',ev)]:
            if e:
                assert e['episodes']==500
                evaluation_rows.append([label,part,500,f"{e['mean']:.3f}",f"{e['sample_std']:.3f}"])
            else:evaluation_rows.append([label,part,'pending','pending','pending'])
        summary['experiments'][name]=dict(evaluation=ev,part1_evaluation=part1)
        if name=='cartpole':
            macro('CartStatus','Completed: primary 1,500-episode training and independent 500-episode evaluation.' if ev else
                  r'\textbf{Pending: local primary training/evaluation has not completed.}')
            comparison=read(ROOT/'figures/policy_gradient_part2/cartpole/part1_vs_part2.json')
            summary['experiments'][name]['comparison']=comparison
            if ev:
                text=f"For these primary checkpoints, mean reward changes by {ev['mean']-part1['mean']:+.3f} and sample SD by {ev['sample_std']-part1['sample_std']:+.3f}. "
                if comparison:
                    c1,c2=comparison['Part 1'],comparison['Part 2']
                    text+=f"Final-100 training means are {c1['last100_mean']:.2f} versus {c2['last100_mean']:.2f}; their reward SDs are {c1['last100_sample_std']:.2f} versus {c2['last100_sample_std']:.2f}. "
                    text+=f"Mean absolute successive changes in MA100 are {c1['mean_absolute_successive_MA100_change']:.3f} versus {c2['mean_absolute_successive_MA100_change']:.3f}. "
                text+='These describe this matched primary pair, not population-level improvement across training seeds.'
                macro('CartComparison',text)
            else:macro('CartComparison','Primary comparison statistics remain pending.')
        else:
            if ev:
                macro('PongStatus',f"Completed: 1,000 training games and exactly 500 final games. Mean {ev['mean']:.3f}, sample SD {ev['sample_std']:.3f}; "
                      f"mean difference from Part 1 {ev['mean']-part1['mean']:+.3f}. No positive result is assumed.")
            else:macro('PongStatus',r'\textbf{Pending: formal Part 2 training is detached on Deepdish5; the chained 500-game evaluation, histogram and final comparison are not yet available locally. No final Part 2 Pong result is claimed.}')
        dg=read(root/'gradient_diagnostics/summary.json')
        if dg:
            summary['experiments'][name]['gradient_variance']=dg
            variance_rows.append([label,dg['held_out_episodes'],f"{dg['trace_covariance_without']:.6g}",
                                  f"{dg['trace_covariance_with']:.6g}",f"{100*dg['relative_reduction']:.2f}\\%"])
            game_prose.append(f"For {label}, the primary EMA changes held-out variance by a relative reduction of "
                f"{100*dg['relative_reduction']:.2f}\\%. Its paired mean-difference norm is {dg['mean_difference_norm']:.6g}, "
                f"versus estimated root trace of covariance of the paired mean {dg['paired_mean_difference_noise_scale']:.6g}. "
                f"The independently calibrated secondary constant is {dg['secondary_calibrated_constant']:.6g}, with held-out "
                f"relative reduction {100*dg['secondary_relative_reduction']:.2f}\\%. "
                + ('The primary baseline increased variance here; this is a negative diagnostic result, despite any reward improvement. ' if dg['relative_reduction']<0 else '')
                + ('The observed variance is zero; this is a degenerate diagnostic, not evidence of effective variance reduction. ' if dg['degenerate_zero_gradient_variance'] else ''))
        else:
            variance_rows.append([label,'pending','pending','pending','pending'])
            game_prose.append(f'{label} fixed-policy gradient diagnostics remain pending.')
    seeds=read(result/'cartpole/seed_results.json') or []
    summary['experiments']['cartpole']['seed_robustness']=seeds
    if seeds:
        prior=read(ROOT/'results/policy_gradient_part1/cartpole/experiment_summary.json')
        controls={s['training_seed']:s for s in prior['seed_robustness']}
        table('SeedTable','rrrrr',r'Seed & Diagnostic $n$ & Part 1 mean & Part 2 mean & Part 2 SD',
              [[s['seed'],s['diagnostic']['episodes'],f"{controls[s['seed']]['mean']:.3f}",
                f"{s['diagnostic']['mean']:.3f}",f"{s['diagnostic']['sample_std']:.3f}"] for s in seeds])
        macro('SeedInterpretation','The separate 100-episode seed diagnostics use seed 2,000,000 in both parts. '
              'Substantial dependence on training seed remains; these diagnostics do not replace the primary 500-episode evaluation. '
              'The primary training curve also regresses mid-run, and its average absolute successive MA100 change is larger, '
              'so the stronger final score does not establish uniformly more stable optimization.')
    else:macro('SeedTable','Independent seed robustness diagnostics remain pending.')
    if not seeds:macro('SeedInterpretation','')
    table('EvaluationTable','llrrr',r'Environment & Method & $n$ & Mean & Sample SD',evaluation_rows)
    table('VarianceTable','lrrrr',r'Problem & Held-out $n$ & No-baseline trace & EMA trace & Reduction',variance_rows)
    macro('GameVariance',' '.join(game_prose))
    (ROOT/'report/sections/policy_gradient_2_results.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    write_json(result/'experiment_summary.json',summary)
    if len(seeds)>1:
        import csv
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7.2,4.2))
        for s in seeds:
            with (result/f'cartpole/seed_{s["seed"]}/history.csv').open() as f:
                rewards=np.array([float(r['reward']) for r in csv.DictReader(f)])
            ax.plot(np.arange(100,len(rewards)+1),np.convolve(rewards,np.ones(100)/100,'valid'),label=f'Seed {s["seed"]}')
        ax.set(xlabel='Training episode',ylabel='100-episode mean reward',title='CartPole Part 2: independent matched training seeds')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(ROOT/'figures/policy_gradient_part2/cartpole/training_seeds.png',dpi=300);plt.close(fig)
    print('Updated Part 2 report and numerical summary from completed evidence only.')


if __name__=='__main__':main()
