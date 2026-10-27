# Part 2 completed ? 2026-10-26

Both environments are implemented, trained and independently evaluated. The
constant baseline is an EMA of previous discounted G0 returns, alpha 0.05,
frozen across each update batch and updated only after Adam.step(). Settings
match the corresponding Part 1 primary runs, including their distinct discount
weighting conventions. No new training or algorithm change was made while
preparing the final report.

## Completed primary results

| Environment | Training episodes | Final evaluation episodes | Mean | Sample SD |
|---|---:|---:|---:|---:|
| CartPole-v1 | 1500 | 500 | 497.282 | 19.689332534 |
| ALE/Pong-v5 | 1000 | 500 | -20.850 | 0.464684693 |

All evaluations are stochastic, training seed 11 and evaluation seed 1000000.
CartPole keeps its normal 500-step limit. All 1000 Pong training games and
500 evaluation games terminated with zero truncations. Pong's final-100
training mean is -20.59, versus Part 1's -20.65. Its final evaluation mean is
lower by 0.242 than Part 1's -20.608; no meaningful Pong learning is demonstrated.

All CartPole seeds 11,22,33 completed 1500 episodes. Their separate 100-episode
diagnostic means (sample SDs) are 498.22 (13.057), 324.76 (32.520), and 152.38
(9.800). Substantial seed dependence remains.

## Validation and gradient covariance

Thirty CPU correctness/edge/resume checks passed, including exact expected
gradient invariance and bitwise model/Adam/RNG/baseline/history continuation
within the tested CPU/software context. Exact gradient error is 2.165e-15;
chunk/full-gradient error is 2.384e-7. The GPU smoke passed 22 mathematical
checks, four complete training games, nonzero-baseline checkpoint resume and
two separately loaded diagnostic evaluation games.

Synthetic paired covariance trace decreases from 0.8452484862 to 0.2387437093
(71.7546% reduction), on 200000 one-step trajectories with an EMA fitted on
2000 disjoint prior trajectories. Fixed-policy held-out game diagnostics give:

| Environment | No-baseline trace | EMA trace | Reduction |
|---|---:|---:|---:|
| CartPole | 1920756.3666 | 44981.7877 | 97.6581% |
| Pong | 1679.6099 | 524.7806 | 68.7558% |

Each game diagnostic uses 32 held-out trajectories, after 32 disjoint
calibration trajectories for a secondary constant. The primary EMA is frozen
from training. Per-parameter means, variances and paired mean differences are
saved in each `gradient_diagnostics/gradient_statistics.npz`. These estimators
are compared at one fixed Part 2 policy, not across independently trained
Part 1/Part 2 policies. Lower gradient variance does not imply higher reward.

## Completed remote execution and retrieval

- Host: deepdish5; one RTX PRO 5000 Blackwell, physical GPU 1.
- UUID: `GPU-ec0c988c-3d23-d55e-7713-293719142e7d`, exposed alone as cuda:0.
- Python: `/nfs/home/fqu0882/.venvs/mlds490-pg-deepdish5/bin/python`.
- Release: `/nfs/home/fqu0882/mlds490/releases/part2-20261026-baseline`.
- Historical pipeline PID/SID: 1518229, verified PPID 1 after SSH disconnection.
- Log: `<release>/results/policy_gradient_part2/remote/formal.log`.
- Completion marker: `<release>/results/policy_gradient_part2/remote/completed.json`.
- Checkpoint: `<release>/results/policy_gradient_part2/pong/formal/latest.pt`.
- Archive: `/nfs/home/fqu0882/pg_part2_completed_artifacts.tar.gz`.

The one-time final status check confirmed `PART2 PIPELINE COMPLETE`. Training,
500-game evaluation, 64-game gradient diagnostics and figures all finished.
The archive was downloaded to `tmp/final_part2_import/`; 37 new formal,
evaluation, diagnostic and figure files were imported into matching local
paths without replacing existing files. Recomputed statistics, episode counts,
complete-game flags and the deployed source manifest agree. Checkpoint SHA256:

`85e0d38f4cf7697b29ba3af04f82c302a506e604de33036dc3768242d28d6190`

Existing launch-log snapshots, audit JSONs and aggregate experimental summaries
are preserved as historical evidence and may describe an earlier state. Current
per-environment final summaries and `dist/report_verification.json` establish
completion. Do not restart the completed pipeline to regenerate documentation.

## Final documentation and owner actions

`report/report.tex` includes the completed Part 2 results and all required
figures. `report/prepare_submission.py --refresh-tables` updates only report
macros, validates saved evidence and creates the Overleaf ZIP. It avoids the
older update script's side effects on aggregate results and valid figures.
The ZIP is `dist/MLDS490_Assignment1_Overleaf.zip`; main file `report/report.tex`.
See `report/OVERLEAF.md` and the top-level README for exact commands.

No numerical results remain pending. The owner must fill student name/ID and
GitHub username/repository name/URL, then compile and inspect the report in
Overleaf. The PDF was not compiled during documentation preparation.
