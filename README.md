# MLDS 490 — Assignment 1

One project and one academic report cover discounted shuttle dispatching,
baseline-free REINFORCE, and REINFORCE with a running constant baseline.
All four primary policy-gradient evaluations are complete. Neither formal Pong
run demonstrates meaningful learning; variance reduction and policy performance
are reported separately.

| Component | Status |
|---|---|
| Value and Policy Iteration (30 points) | Complete; numerical and simulation validation saved |
| Policy Gradient Part 1 (40 points) | CartPole and Pong trained and evaluated |
| Policy Gradient Part 2 (20 points) | Constant EMA baseline; both environments trained and evaluated |
| Unified report (10 points) | Complete sources and Overleaf package; owner will compile the PDF |

## Verified results

Every primary evaluation below contains **exactly 500 stochastic episodes**,
evaluation seed 1000000, and a separately loaded fixed-budget checkpoint from
training seed 11. SD is sample standard deviation (`ddof=1`). All Pong games
terminated with zero truncations. CartPole retains its standard 500-step limit;
reaching that cap counts as a complete capped episode.

| Environment | Method | Training episodes | Evaluation mean | Sample SD |
|---|---|---:|---:|---:|
| CartPole-v1 | Part 1, no baseline | 1500 | 429.248 | 56.776 |
| CartPole-v1 | Part 2, EMA baseline | 1500 | 497.282 | 19.689 |
| ALE/Pong-v5 | Part 1, no baseline | 1000 | -20.608 | 0.654 |
| ALE/Pong-v5 | Part 2, EMA baseline | 1000 | -20.850 | 0.465 |

CartPole's primary mean increases by 68.034. Pong's Part 2 mean is lower by
0.242; its lower SD reflects concentration near -21, not successful play.
Separate 100-episode CartPole diagnostics at evaluation seed 2000000 show
substantial training-seed variation:

| Training seed | Part 1 mean | Part 2 mean | Part 2 SD |
|---:|---:|---:|---:|
| 11 | 430.14 | 498.22 | 13.057 |
| 22 | 101.50 | 324.76 | 32.520 |
| 33 | 25.31 | 152.38 | 9.800 |

These diagnostics do not replace or enlarge the primary 500 episodes.

For the shuttle MDP, VI and PI agree at all 201 states: wait at 0–12 and
dispatch at 13–200. VI takes 519 backups; PI takes six evaluation/improvement
passes including the stable final pass. Maximum value disagreement is
`1.865e-9`; independent Bellman residuals are `9.413e-11` and `9.095e-13`.
The 500-period error is `4.940e-9`, below the rigorous bound `6.838e-8`.

## Directory structure

```text
requirements.txt                     # NumPy, SciPy, Matplotlib
requirements-pg.txt                  # PyTorch, Gymnasium, ALE
requirements-pg-local-lock.txt       # recorded isolated Windows versions
configs/                            # JSON experiment configurations
value_policy_iteration/             # MDP, solvers, experiments, validation
policy_gradient_part1/              # shared networks, rollout, loss, training, evaluation
policy_gradient_part2/              # EMA baseline and paired gradient analysis
scripts/                            # execution and validation entry points
results/value_policy_iteration/
results/policy_gradient_part1/{cartpole,pong,validation}/
results/policy_gradient_part2/{cartpole,pong,validation}/
figures/                            # plots grouped by component and environment
report/report.tex                   # single report entry point
report/sections/                     # methods, results, discussion and bibliography
report/prepare_submission.py         # report-only verification and ZIP preparation
report/regenerate_figures.py         # required PG plots from saved CSVs
docs/                               # experimental protocols and compute records
dist/                               # Overleaf ZIP and verification inventory
```

## Requirements and isolated installation

Use Python 3.10 or newer. Component 1 needs only NumPy, SciPy and Matplotlib.
Policy gradients also need PyTorch, Gymnasium and ALE. ALE 0.10.2 is pinned;
the later 0.11.2 build failed Windows DLL initialization during development.
Pillow, used for packaging image checks, is installed with Matplotlib. Use an
isolated environment without inherited system/Anaconda site packages.

Windows PowerShell, matching the recorded CPU setup:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-pg.txt
```

For Component 1 and report packaging only, install `requirements.txt` instead.
The local PG runs used Python 3.11.5, NumPy 1.26.4, PyTorch 2.5.1+cpu,
Gymnasium 1.4.0 and Matplotlib 3.9.4. The complete record is
`requirements-pg-local-lock.txt`. Component 1's earlier saved run used NumPy
1.24.3, SciPy 1.11.1 and Matplotlib 3.7.2, recorded in its summary.

Linux CPU setup:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-pg.txt
```

GPU training requires a PyTorch build compatible with the allocated device;
the Windows CPU wheel does not provide CUDA. Formal Pong reused an existing
isolated Linux environment: Python 3.12.6, PyTorch 2.14.0 with CUDA 13.0 and
`sm_120` support, Gymnasium 1.4.0 and ALE 0.10.2, on one RTX PRO 5000 Blackwell.
Per-run metadata records actual versions and source hashes. Timings depend on
emulator/CPU throughput as well as the GPU; cross-device bitwise identity is
not promised.

All following commands run from the project root. `python` means the selected
isolated interpreter: substitute `.venv\Scripts\python.exe` on Windows or
`.venv/bin/python` on Linux, or activate that environment first.

## Short validation and saved-artifact checks

```sh
python report/prepare_submission.py --check-only
```

This reads all four evaluations, checks counts, statistics, seeds and checkpoint
hashes, and checks LaTeX dependencies without policy sampling or training.

To reproduce bounded mathematical and checkpoint tests, use a separate project
copy and fresh output directories:

```sh
python scripts/run_pg_validation.py --output tmp/pg1_validation_reproduction --absolute --pong
python scripts/validate_pg2.py --output tmp/pg2_validation_reproduction
python scripts/audit_pg_artifacts.py --output tmp/part1_audit_reproduction.json
```

The validators include a tiny synthetic bandit, numerical gradients, return
checks, checkpoint continuation and bounded emulator checks; they are not
formal training. The Part 2 validator also writes its canonical synthetic
variance figure, so run it in the reproduction copy. The legacy artifact
audit's explicit output preserves its historical audit JSON.

## Full reproduction: Value and Policy Iteration

The shuttle model has states 0–200, actions wait/dispatch, capacity 15, dispatch
cost 100, holding cost 2, uniform arrivals 1–5, gamma 0.95 and zero finite
terminal cost. Costs follow dispatch and precede arrivals; overflow is
discarded at the cap. Empty and partial dispatches are permitted.

```sh
python scripts/run_value_policy.py --episodes 20000 --simulation-horizon 400 --seed 20261026
```

This runs finite-horizon backward induction, VI, PI, experiments A–I and edge
cases, then regenerates numerical outputs, report tables and figures. It
**overwrites canonical generated outputs**; reproduce in a separate copy.
Saved outputs include all finite-horizon rows in `solutions.npz`, state-wise
CSVs, sensitivities, exhaustive policies, simulations and `summary.json`.

## Full reproduction: Policy Gradient Part 1

CartPole's primary settings are Adam 0.003, gamma 0.95, 1500 episodes, batch
ten and **relative `G_t` weighting**. The reference config has a different
initial rate and budget, so primary overrides are explicit:

```sh
python scripts/train_cartpole.py --config configs/cartpole_part1.json --learning-rate 0.003 --episodes 1500 --seed 11 --discount-weighting relative --log-interval 50 --output results/reproduction/pg1_cartpole/seed_11
```

The original pilot/three-seed driver is `scripts/run_cartpole_experiments.py`.
It repeats the existing training-only learning-rate selection. Suite drivers
write canonical figure destinations even with a fresh `--output`; run them in
a separate reproduction copy.

Formal Pong uses **absolute `gamma^t G_t` weighting**, gamma 0.99, Adam 0.01,
1000 games and batch five. Inside an allocated one-GPU Linux session:

```sh
export CUDA_VISIBLE_DEVICES=GPU_UUID_ASSIGNED_TO_YOU
python scripts/train_pong.py --config configs/pong_part1_absolute.json --device cuda:0 --episodes 1000 --learning-rate 0.01 --checkpoint-interval 50 --log-interval 10 --output results/reproduction/pg1_pong/formal
```

Replace the device placeholder with the permitted, available GPU UUID. The
absolute config retains a short diagnostic budget, so `--episodes 1000` is
essential. Use only the assigned idle device. The original workflow uses a
shared account lock, detached execution and sequential evaluation; see
[the completed Part 1 record](docs/PONG_ACTIVE_RUN.md).

## Full reproduction: Policy Gradient Part 2

The constant baseline is an EMA of **previous discounted episode returns**
`G_0 = sum_t gamma^t r_t`, alpha 0.05. The first batch uses zero. After the
optimizer step, the first completed return initializes the EMA directly;
later returns apply `b <- 0.95*b + 0.05*G_0` in episode order. One scalar is
frozen throughout each batch, so current-batch returns never influence their
own gradients. Its action-independent score contribution has zero conditional
expectation. No critic, entropy bonus, scaling or point-based reset is added.

Primary comparisons preserve each Part 1 model, optimizer, learning rate,
budget, environment and weighting convention: **`G_t-b` for CartPole**, and
**`gamma^t*(G_t-b)` for Pong**.

```sh
python scripts/train_pg2.py --config configs/cartpole_part2.json --baseline-alpha 0.05 --output results/reproduction/pg2_cartpole/seed_11
```

In the allocated one-GPU context described above:

```sh
python scripts/train_pg2.py --config configs/pong_part2.json --device cuda:0 --baseline-alpha 0.05 --output results/reproduction/pg2_pong/formal
```

The executed CartPole suite is `scripts/run_cartpole_part2.py`, with seeds
11,22,33 and 1500 episodes each. The completed remote Pong pipeline is
`scripts/pong_part2_pipeline.py`, launched by `scripts/launch_pong_part2.py`;
its fixed-release prerequisites and completion are documented in
[Part 2 status](docs/PART2_STATUS.md). Both formal Pong jobs have finished.

## Saved-checkpoint evaluation and resume

The shared evaluator loads any Part 1 or Part 2 checkpoint. These examples
read completed checkpoints but write new evaluations:

```sh
python scripts/evaluate_pg.py --checkpoint results/policy_gradient_part2/cartpole/seed_11/latest.pt --output results/reproduction/pg2_cartpole_evaluation --episodes 500 --seed 1000000
python scripts/evaluate_pg.py --checkpoint results/policy_gradient_part2/pong/formal/latest.pt --output results/reproduction/pg2_pong_evaluation --device cuda:0 --episodes 500 --seed 1000000
```

Use the analogous Part 1 path for those policies. CPU evaluation uses
`--device cpu`. Final mode requires exactly 500 episodes and rejects truncated
Pong games. `--role diagnostic --episodes 100` is a separate diagnostic.
Evaluation never subtracts a baseline. Existing evaluation directories are
refused, preventing accidental replacement.

Checkpoints save model, Adam, RNG, configuration and committed history; Part 2
adds baseline value/count/alpha. For an interrupted run, repeat its training
command with its original output, a larger total `--episodes` target and
`--resume <output>/latest.pt`. Alpha and algorithm/environment settings must
match. Recovery is at an update boundary, not a mid-game emulator state.
SIGINT/SIGTERM requests a save after the current update; a hard failure can
lose later work since the last checkpoint. Only load trusted checkpoints.
Extending a completed reported run would be a new, separately labeled experiment.

## Figure locations and regeneration from saved data

The report includes all 26 existing assignment figures:

```text
figures/value_policy_iteration/                         # 11 figures
figures/policy_gradient_part1/cartpole/                 # 3 figures
figures/policy_gradient_part1/pong/                     # 2 figures
figures/policy_gradient_part2/cartpole/                 # 5 figures
figures/policy_gradient_part2/pong/                     # 4 figures
figures/policy_gradient_part2/synthetic_gradient_variance.png
```

Required PG filenames are `training_rewards.png` (raw rewards and MA100) and
`evaluation_histogram.png` (500 episodes). Part 2 adds `part1_vs_part2.png`,
`gradient_variance.png`, and CartPole `training_seeds.png`. Packaging verification
lists each included image and SHA256. Rebuild the required PG plots from saved
CSVs into a **new** directory without running policies:

```sh
python report/regenerate_figures.py --part 1 --environment cartpole --output tmp/rebuilt_pg1_cartpole
python report/regenerate_figures.py --part 1 --environment pong --output tmp/rebuilt_pg1_pong
python report/regenerate_figures.py --part 2 --environment cartpole --output tmp/rebuilt_pg2_cartpole
python report/regenerate_figures.py --part 2 --environment pong --output tmp/rebuilt_pg2_pong
```

Existing output directories are refused. Component 1's full reproduction command
also regenerates its figures. No correct existing figure was regenerated during
final documentation preparation.

## Quantitative validation and interpretation

| Evidence | Actual procedure and measured outcome |
|---|---|
| MDP correctness | Independent Bellman residuals, VI/PI agreement, all 32 policies on a five-state MDP, valid horizon bounds |
| MDP simulation | 20 policy/state comparisons, 20000 episodes each; exact continuation at period 400; statistical consistency with exact evaluation |
| Part 1 gradient | Exact error 0; 100000-sample error 0.00116; finite differences agree; controlled bandit learns the better action |
| Estimator distinction | Exact two-step loss derivatives -0.42 (absolute) and -0.63 (relative) |
| Part 2 invariance/resume | 30 checks; exact discrepancy `2.165e-15`; chunk discrepancy `2.384e-7`; CPU model/Adam/RNG/baseline/history continuation identical |

Paired gradient covariance measurements, separate from episodic reward SD:

| Fixed-policy problem | Assessment trajectories | No-baseline trace | EMA trace | Reduction |
|---|---:|---:|---:|---:|
| Synthetic | 200000 | 0.845248 | 0.238744 | 71.75% |
| CartPole | 32 held out | 1920756.37 | 44981.79 | 97.66% |
| Pong | 32 held out | 1679.61 | 524.78 | 68.76% |

Game diagnostics also use 32 disjoint calibration trajectories for a secondary
constant; the primary frozen EMA uses none of those returns. Synthetic EMA
calibration uses 2000 disjoint prior trajectories. Both estimators use the same
assessment trajectories. These finite-sample measurements compare estimators
at one common policy, not the separately trained Part 1 and Part 2 policies.
Per-parameter statistics and paired mean differences are saved in
`gradient_statistics.npz`.

Pong uses actions 2/3, frameskip four, sticky probability 0.25 and signed
differences of binary 80x80 frames. A production-path audit found no confirmed
implementation defect: sprites survive preprocessing, no-FIRE play progresses,
and observations/actions/rewards align. It observed strong late-credit
attenuation from whole-game `gamma^t`, loss of stationary paddle information,
and weak visual dependence in the tested Part 1 checkpoints. These observations
do not identify a unique cause. One thousand games provide 200 optimizer updates.
The report discusses these limits and a hypothetical additional three-month
plan without presenting proposed studies as completed work.

## Artifacts and reproducibility records

Runs save `config.json`, `metadata.json`, `history.csv`, `summary.json`,
first-step traces, `latest.pt` and periodic checkpoints. Evaluations save
`rewards.csv`, ending flags, seeds, statistics and checkpoint hashes. Part 2
adds baseline histories, MA100 and `baseline_protocol.json`. Training reset
seeds equal training seed plus zero-based episode index.

`configs/cartpole_part2.json` and `configs/pong_part2.json` match the actual
primary settings. `configs/cartpole_part1.json` is a pre-selection reference,
and `configs/pong_part1_absolute.json` also serves short diagnostics; use the
explicit primary overrides above or actual saved configs for reproduction.

The final Pong import verified 1000 complete training rows, 500 complete
evaluation rows, summary statistics, checkpoint identity and the deployed source
manifest. Existing artifacts were not replaced. Historical status/aggregate JSON
files are retained as dated evidence and may describe an earlier pending state.
Current per-environment final summaries, this README, the report and
`dist/report_verification.json` supersede that status. The original
[REINFORCE protocol](docs/REINFORCE_PROTOCOL.md) records experimental conventions.

The documentation audit confirmed that all 541 protected pre-existing
source/configuration/artifact files and Component 1 report sources remain
byte-for-byte unchanged. Numerical solver error, evaluation uncertainty,
training-seed variability and cross-device reproducibility remain distinct limits.
