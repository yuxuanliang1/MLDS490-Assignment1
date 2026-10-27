"""Three predeclared matched seeds; primary final evaluation is seed 11 only."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from policy_gradient_part1.config import Config
from policy_gradient_part1.evaluator import evaluate
from policy_gradient_part1.runtime import write_json
from policy_gradient_part1.trainer import train
from policy_gradient_part2.baseline import RunningConstantBaseline
from policy_gradient_part2.analysis import compare, diagnose

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"results/policy_gradient_part2/cartpole")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,22,33])
    a=p.parse_args();base=Config.load(ROOT/"configs/cartpole_part2.json")
    figures=ROOT/"figures/policy_gradient_part2/cartpole"
    write_json(a.output/"protocol.json",dict(primary_seed=11,seeds=a.seeds,alpha=.05,episodes=1500,learning_rate=.003,
        discount_weighting="relative",selection="Predeclared seed 11, final budget checkpoint; no retuning",
        final_evaluation_seed=1000000,final_evaluation_episodes=500,
        robustness_episodes=100,robustness_seed=2000000))
    results=[]
    for seed in a.seeds:
        directory=a.output/f"seed_{seed}"
        summary=train(replace(base,seed=seed),directory,baseline=RunningConstantBaseline(),
                      figure=figures/"training_rewards.png" if seed==11 else None)
        if seed==11:
            evaluation=evaluate(directory/"latest.pt",a.output/"evaluation",figure=figures/"evaluation_histogram.png")
            compare(ROOT/"results/policy_gradient_part1/cartpole",a.output,figures/"part1_vs_part2.png",
                    "CartPole: G_t weighting; only training baseline differs")
            diagnose(directory/"latest.pt",a.output/"gradient_diagnostics",figure=figures/"gradient_variance.png")
        diagnostic=evaluate(directory/"latest.pt",directory/"validation",episodes=100,seed=2000000,role="diagnostic")
        results.append(dict(seed=seed,training=summary,diagnostic=diagnostic))
        write_json(a.output/"seed_results.json",results)
