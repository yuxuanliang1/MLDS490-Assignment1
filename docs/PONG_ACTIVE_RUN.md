# Completed Part 1 formal Pong run

**Completion update (2026-10-26):** training and the chained evaluation finished.
Final-100 training mean: -20.65. Exactly 500 terminated stochastic evaluation
games: mean -20.608, sample SD 0.653598, range -21 to -17, zero truncations.
The archive was retrieved; reward statistics and checkpoint SHA256 were verified
before importing new artifacts locally. Existing artifacts were preserved.
[Part 2](PART2_STATUS.md) has also completed in its separate release; both final
evaluations are now integrated into the unified report.
The launch notes below describe the completed Part 1 execution.

Launched 2026-10-26 on **deepdish5, physical GPU 1: NVIDIA RTX PRO 5000
Blackwell**, UUID `GPU-ec0c988c-3d23-d55e-7713-293719142e7d`. It was idle before
both benchmark and formal launch. No occupied GPU or simultaneous GPU job was used.

| Deployment | Games measured | Seconds/game | Environment steps/second |
|---|---:|---:|---:|
| Earlier Deepdish3 RTX 2080 Ti pilots | 90 | 4.049 | 246.95 |
| Deepdish5 RTX PRO 5000 bounded benchmark | 10 | 1.298 | 748.06 |

The ten-game benchmark took 12.982 s for 9,711 steps: environment stepping
2.952 s, reset 0.925 s, forward host time 1.982 s, backpropagation 0.734 s, Adam
0.095 s, other rollout/Python/preprocessing/sampling/transfers 6.294 s. Forward
CUDA-event time was 1.984 s and overlaps host time; do not add them. Sampled
mean GPU utilization was 7.92%; system CPU utilization 58.42%; process CPU usage
about 96% of one core. Raw evidence is under
`results/policy_gradient_part1/benchmarks/deepdish5_blackwell_10ep/`.

The measured deployment is 3.12x faster by games/second (68% less time/game).
This short comparison does not isolate hardware: Python/PyTorch differ
(3.10/2.8 versus 3.12/2.14), the old pilots used relative G_t weighting, and
the new benchmark uses gamma^t G_t. Both use the same CNN, frame differences,
gamma 0.99, frameskip 4, stickiness 0.25, action mapping, batch five, one thread,
and seeded initial configuration. Steps/second also improves substantially.
No extra hardware or hyperparameter sweep was run. Readiness/benchmark/launch
finished within six minutes of the ten-minute investigation budget.

The existing compatible PyTorch 2.14.0+cu130 installation was reused in a user
overlay at `/nfs/home/fqu0882/.venvs/mlds490-pg-deepdish5`; only Gymnasium 1.4.0
and ALE 0.10.2 were added. Tiny CUDA forward/backward kernels, seven objective
checks, and ten complete-game GPU rollouts/updates passed. Shared software was
not modified; local reference PDFs were excluded from deployment.

## Formal configuration and paths

REINFORCE has no baseline/centering. Its weights are **gamma^t G_t**, with the
outer exponent starting at zero each episode, never at chunks or Pong points.
Adam uses the assignment's initial rate **0.01**, gamma **0.99**, batch **5**,
seed **11**, **1,000 episodes**, checkpoint interval **50**, log interval **10**.
Only the selected UUID is visible as `cuda:0`. Earlier validated runs remain intact.

Remote release:
`/nfs/home/fqu0882/mlds490/releases/part1-20261026-blackwell`

Relative to that release:

```text
results/policy_gradient_part1/remote/formal_independent.log
results/policy_gradient_part1/pong/formal/latest.pt
results/policy_gradient_part1/pong/formal/episode_000050.pt
results/policy_gradient_part1/pong/formal/history.csv
results/policy_gradient_part1/pong/formal/config.json
```

Persistence uses `nohup`, a new POSIX session, closed stdin, and file-backed
stdout/stderr. Parent job PID **1515156** was verified reparented to PID 1 after
SSH disconnected; initial training PID **1515165**. An account-wide GPU lock
prevents overlapping Pong tasks. No Codex connection or tmux installation is needed.

```powershell
ssh deepdish5 'tail -n 15 ~/mlds490/releases/part1-20261026-blackwell/results/policy_gradient_part1/remote/formal_independent.log'
ssh deepdish5 'ps -p 1515156 -o pid,ppid,sid,etime,stat,args'
```

After successful training, the same detached shell starts a separate **500-game
stochastic evaluation**, seed 1000000, and generates both required Pong figures.
It then writes `~/pg_absolute_completed_artifacts.tar.gz`. Training and evaluation
execute sequentially. No final Pong outcome is claimed before its rewards CSV
and completion summary exist. Games may become longer if the policy improves.

After completion, retrieve that archive into `remote_artifacts/`, inspect its
manifest/configuration, then import generated Pong results/figures into matching
local directories. Run `scripts/update_pg_report.py`, `scripts/audit_pg_artifacts.py`,
and compile the one unified report. Existing CartPole and Component 1 artifacts
remain preserved.
