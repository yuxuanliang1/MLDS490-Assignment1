> Completion note: both parts now have 500-episode final evaluations. This
> document records the original Part 1 protocol; Part 2 adds the EMA baseline
> described in `PART2_STATUS.md` and the unified report.

# Policy Gradient Part 1 protocol

## Estimator and objective convention

The preserved CartPole/early Pong reference runs use `discount_weighting=relative`.
The completed formal Part 1 Pong run uses `discount_weighting=absolute`, which
multiplies G_t by gamma^t starting at episode timestep zero. Its configuration is
`configs/pong_part1_absolute.json`; operational details are in `PONG_ACTIVE_RUN.md`.
Existing results are preserved under their original convention.

For each complete trajectory, `G[t] = r[t] + gamma * G[t+1]` with zero continuation
at either termination or truncation. The implemented loss is

```
L = -(1 / B) sum_{episodes i=1..B} sum_t log pi_theta(a_it | s_it) w_t G_it
w_t = 1 (relative reference), or gamma**t (absolute initial-state objective)
```

Time steps are **summed**, not averaged within episodes, so a long trajectory
does not receive an implicit inverse-length weight. A partial last batch uses
its actual episode count B. Rewards and returns are not centered, standardized,
clipped, rescaled, or reset at Pong points. There is no baseline, critic, entropy
bonus, or learning-rate schedule. Optional gradient clipping defaults to null;
the executed reference experiments do not enable it. Adam's learning rate is
0.01 in both reference configs. CartPole's alternate LR is selected from recorded
training-only pilots, separately from its final evaluation.

The relative mode is the commonly used reward-to-go reference and omits an
**outer** `gamma**t`. For the strict initial-state objective
`J_gamma = E[sum_t gamma**t r_t]`, the likelihood-ratio gradient includes
`sum_t gamma**t grad log pi(a_t|s_t) G_t`. Those conventions are not mathematically
identical. Absolute mode implements that strict weighting, with the environment's
finite episode boundary. Unweighted reference results are not relabeled as
unbiased estimates for that strict objective. Seven extra exact two-step,
finite-difference, boundary and chunking checks validate the distinction.

Rollouts use `Categorical(logits=...)` and store the sampled action's log
probability. Gradient calculation recomputes those same actions under unchanged
weights and compares log probabilities numerically. The model has no dropout or
batch normalization, no update occurs until the entire batch is collected, and
each trajectory is used for exactly one update. Chunks accumulate gradients of
the summed loss before the Adam step; they do not introduce minibatch updates
or off-policy replay. The validation compares these gradients to a retained
full-graph calculation. Pong inputs are buffered as int8 binary frames/differences;
only a chunk's graph is retained on the GPU. Rollout inference remains serial,
so GPU launch/transfer and CPU emulator latency can dominate this small network.

## Environments and models

CartPole uses exactly `CartPole-v1`, four float32 observations, MLP
`4 -> 128 ReLU -> 128 ReLU -> 2 logits`, gamma 0.95, and Gymnasium's 500-step
limit. Termination and time-limit truncation are separate CSV columns. No
bootstrapping occurs at either boundary. A capped 500-step CartPole episode is
an ordinary complete evaluation episode under this task's time-limit convention.
The two ending flags can overlap: in the primary 500 episodes, 403 set
`terminated`, 99 set `truncated`, and two set both. Each episode is counted once;
these flag counts must not be treated as mutually exclusive categories.

Pong uses exactly `ALE/Pong-v5`, RGB observations, frameskip 4, sticky-action
probability 0.25, minimal six-action space, and the ALE default 108000-emulator-
frame episode cap. No life-based episode wrapper or reward clipping is used.
The complete ordered action meanings are checked at construction. Output indices
0 and 1 map exclusively to environment actions 2 (RIGHT) and 3 (LEFT).
The assignment crop `frame[35:195:2, ::2, 0]` maps backgrounds 144/109 to zero
and all other nonzero pixels to one, on a copy. Default inputs are consecutive
frame differences (additional modeling choice); the first difference is zero.
Raw binary frames are supported and compared in CPU pilots.

Pong CNN: `Conv(1,8,8,stride4) -> ReLU -> Conv(8,16,4,stride2) -> ReLU ->
Conv(16,32,3,stride1) -> ReLU -> Flatten(32*6*6) -> Linear(64) -> ReLU -> Linear(2)`.
All dimensions refer to the 80x80 input, with no padding. It has no critic.
Initialization/FIRE and score-return resetting are separate issues: the executed
startup inspection completed a full match with repeated scoring events using
only actions 2 and 3, so FIRE is disabled at reset and between points. An optional
reset-only FIRE flag remains explicit and must match in training/evaluation;
nonzero initialization rewards or an initialization termination raise errors.

## Selection and evaluation

`run_cartpole_experiments.py` writes its protocol before training. It compares
LR 0.01/0.003/0.001 for 300 episodes each at seed 101 and selects the largest
last-100 training mean. A 1500-episode budget (five pilot durations) tests sustained
learning and possible regression, using fresh seeds 11, 22, 33. Seed 11 is
predeclared primary, with its final-budget checkpoint selected before any final
evaluation. This intentionally exposes variance; it does not pick whichever
seed or intermediate checkpoint later looks best on evaluation.

The primary saved model is sampled stochastically for exactly 500 episodes with
evaluation reset seeds `1000000 + i`. Random-policy reference episodes use
`3000000 + i`; seed robustness uses 100 diagnostic episodes per separately
trained model with reset seeds `2000000 + i`. These diagnostic runs are not
additional hyperparameter selection. Normal-approximation 95% intervals are
reported with their limitations (bounded, potentially skewed rewards; only
three training seeds). All reported SDs use `ddof=1`. A random comparison policy
is an experimental reference, never a subtracted reward baseline.

`run_pong_pilots.py` declares its CPU budget and compares raw/difference inputs
and three LRs. Ten episodes at batch size five yield just two updates: evidence
about code, startup, gradients, and runtime, not a claim of convergence. The
diagnostic checkpoint (difference, LR 0.003) is predeclared, and 30 episodes each
compare it to uniform-random and fixed-RIGHT policies under identical settings.
None is the required final 500-game trained Pong evaluation. Remote GPU pilots
must separately establish runtime/VRAM and a defensible formal budget.

Every final evaluation loads a saved checkpoint in a fresh invocation, switches
to eval mode and `torch.no_grad`, preserves environment/preprocessing settings,
and records all raw undiscounted rewards. Pong final mode rejects time-limit
truncation rather than silently counting partial games as complete matches.
Incomplete evaluation CSVs have no completion summary. Final checkpoints must
be selected from training or separate validation, not repeated final-set tuning.

## Edge cases and reproduction

| Case | Handling and evidence |
|---|---|
| Empty/short/long trajectories | Actual rollouts require >=1 action; one-step truncation tested; summed time loss; compact Pong buffer |
| Gamma 0, 0.999, 1; sparse signs | Hand-computed recursive returns, including zeros, single step, negative/positive sparse rewards |
| Extreme log probabilities | Stable categorical logits; a logit gap of 2000 has finite loss and gradient |
| NaN/Inf | Reject invalid observations, logits, returns, losses, gradients, and updated parameters; no silent optimizer update |
| Failure vs time limit | Separate flags and raw rewards in each history/evaluation row; no critic bootstrap |
| Identical frames | Zero difference is valid; preprocessing shape/dtype/copy/binary behavior tested |
| Wrong Pong registration/actions | Fail explicitly; no version fallback; inspect all six action meanings |
| Startup stalls | Real no-FIRE vs FIRE experiment with frame changes and successive scoring events |
| Collapse/high variance | Entropy, both-action probabilities/frequencies, raw loss, norm, and independent seeds; report regressions |
| CUDA absent/OOM | Requested CUDA fails explicitly; exactly one visible GPU required; lower chunk size in a documented new config |
| Interruption/history mismatch | Atomic episode-boundary checkpoint includes committed history/optimizer/RNG; resume recovers it; no mid-game guarantee |
| Stale output or final-set reuse | Fresh output directory required; no silent rerun overwrite |

Run `python scripts/run_pg_validation.py --pong`, the two experiment suites, and
`python scripts/audit_pg_artifacts.py`. JSON evidence, complete CSV histories,
first-ten-step traces, source hashes, versions and configurations are stored
under `results/policy_gradient_part1/`. The shared checkpoint test reproduces
fixed-observation probabilities and the next episode/Adam step on the same CPU
software. Cross-platform/CUDA bitwise reproducibility is not claimed.

With three more months: increase independent training seeds and report
hierarchical uncertainty; compare architectures/LRs/budgets under equal frame
budgets; test raw/difference inputs and motion ambiguity; stress sticky actions,
initial states and physical parameter shifts; investigate sparse reward credit
assignment with exact short synthetic tasks; independently finite-difference
multi-step policy gradients under both discount conventions; and test recovery
under real preemptions. Each tests a specific estimator, representation,
robustness, or reliability claim. Baseline methods belong to Part 2 and are not
implemented here.
