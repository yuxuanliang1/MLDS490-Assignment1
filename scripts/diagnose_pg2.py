"""Paired gradient variance on independent complete trajectories at a frozen policy."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from policy_gradient_part2.analysis import diagnose
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episodes",type=int,default=64);p.add_argument("--seed",type=int,default=4000000)
    p.add_argument("--device",default="cpu");p.add_argument("--figure",type=Path)
    a=p.parse_args();diagnose(a.checkpoint,a.output,a.episodes,a.seed,a.device,a.figure)
