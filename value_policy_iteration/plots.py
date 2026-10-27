"""Publication-readable, headless figures, all saved at 300 dpi."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


COLORS = ["#2364AA", "#D45113", "#23856D", "#8757A6", "#B28B16"]


def generate_plots(root: Path, mdp, finite, vi, pi, summary, parameter_arrays):
    output = root / "figures" / "value_policy_iteration"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                         "legend.fontsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=COLORS)})

    def save(fig, name):
        fig.savefig(output / name, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def values_figure(values, title, name, label):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.plot(mdp.states, values, linewidth=2, label=label)
        ax.set(xlabel="Passengers waiting at decision epoch", ylabel="Expected discounted cost (cost units)",
               title=title, xlim=(0, mdp.parameters.max_state))
        ax.grid(alpha=0.2)
        ax.legend()
        save(fig, name)

    values_figure(finite.values[0], "Finite-horizon optimal value at time 0 (T = 500)",
                  "finite_horizon_value.png", r"$V_0$, zero terminal cost")
    values_figure(vi.values, "Infinite-horizon optimal value (value iteration)",
                  "infinite_horizon_value.png", r"$V^*$, $\gamma=0.95$")

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.step(mdp.states, pi.policy, where="post", linewidth=2)
    threshold = summary["policy_iteration"]["structure"]["threshold"]
    if threshold is not None:
        ax.axvline(threshold, color=COLORS[1], linestyle="--", alpha=0.75,
                   label=f"First dispatch state: {threshold}")
    ax.set(xlabel="Passengers waiting at decision epoch", ylabel="Optimal action",
           title="Infinite-horizon optimal policy (policy iteration)",
           yticks=[0, 1], yticklabels=["0 = Wait", "1 = Dispatch"], ylim=(-0.15, 1.15),
           xlim=(0, mdp.parameters.max_state))
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, "optimal_policy.png")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.semilogy(np.arange(1, vi.iterations + 1), np.maximum(vi.history, np.finfo(float).tiny),
                label=r"$\|V_{k+1}-V_k\|_\infty$", linewidth=2)
    ax.axhline(1e-10, linestyle="--", color=COLORS[1], label="Stopping tolerance")
    ax.set(xlabel="Bellman backups", ylabel="Successive sup difference (cost units)",
           title="Value iteration convergence")
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    save(fig, "vi_convergence.png")

    horizons = summary["horizon_validation"]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.semilogy([r["horizon"] for r in horizons], [r["sup_error"] for r in horizons],
                "o-", label="Observed sup error against PI")
    ax.semilogy([r["horizon"] for r in horizons], [r["tail_bound"] for r in horizons],
                "s--", label=r"$\gamma^T C_{\max}/(1-\gamma)$")
    ax.set(xlabel="Finite horizon T (periods)", ylabel="Sup error / upper bound (cost units)",
           title="Finite-horizon convergence to the infinite-horizon value")
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    save(fig, "horizon_error.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(mdp.states, vi.values, label="VI", linewidth=2)
    axes[0].plot(mdp.states, pi.values, "--", label="PI", linewidth=1.5)
    axes[0].set(title="Independent solvers agree", ylabel="Expected discounted cost")
    axes[0].legend()
    axes[1].plot(mdp.states, vi.values - pi.values)
    difference_scale = max(float(np.max(np.abs(vi.values - pi.values))), np.finfo(float).eps)
    axes[1].axhline(0, color="#666666", linewidth=1)
    axes[1].set_ylim(-1.15 * difference_scale, 0.15 * difference_scale)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    axes[1].set(title="Numerical difference", ylabel="VI value minus PI value (cost units)")
    for ax in axes:
        ax.set_xlabel("Passengers waiting")
        ax.grid(alpha=0.2)
    save(fig, "vi_pi_comparison.png")

    q = mdp.action_values(pi.values)
    gap = q[1] - q[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, limit in zip(axes, (200, 25)):
        ax.plot(mdp.states, gap, linewidth=2)
        ax.axhline(0, color="#666666", linewidth=1)
        ax.axvline(13, color=COLORS[1], linestyle="--", label="First dispatch state = 13")
        ax.set(xlim=(0, limit), xlabel="Passengers waiting", ylabel=r"$Q(s,1)-Q(s,0)$ (cost units)")
        ax.grid(alpha=0.2)
    axes[0].set_title("Action gap across the state space")
    axes[1].set(title="Switching region", ylim=(-45, 110))
    axes[1].legend()
    save(fig, "action_value_gap.png")

    def sensitivity(group, labels, title, filename):
        scenarios = parameter_arrays[group]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        policies = []
        for (name, arrays), label in zip(scenarios.items(), labels):
            axes[0].plot(mdp.states, arrays["values"], label=label, linewidth=1.7)
            policies.append(arrays["policy"])
        axes[0].set(xlabel="Passengers waiting", ylabel="Expected discounted cost",
                    title="Optimal values")
        axes[0].grid(alpha=0.2)
        axes[0].legend()
        image = axes[1].imshow(policies, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
                               cmap=ListedColormap(["#E3ECF5", "#2364AA"]),
                               extent=(-0.5, mdp.n_states - 0.5, len(labels) - 0.5, -0.5))
        axes[1].set(xlabel="Passengers waiting", title="Optimal actions",
                    yticks=range(len(labels)), yticklabels=labels)
        fig.colorbar(image, ax=axes[1], ticks=[0, 1], shrink=0.65).ax.set_yticklabels(["Wait", "Dispatch"])
        fig.suptitle(title)
        save(fig, filename)

    sensitivity("cost", ["Baseline", "Free dispatch", "No holding", "Dispatch cost 1000", "Holding cost 10"],
                "Dispatch and holding cost sensitivity", "cost_sensitivity.png")
    sensitivity("discount", [r"$\gamma=0.50$", r"$\gamma=0.80$", r"$\gamma=0.95$", r"$\gamma=0.99$"],
                "Discount-factor sensitivity", "discount_sensitivity.png")
    sensitivity("deterministic", ["A = 1", "A = 3", "A = 5"],
                "Deterministic arrival experiments", "deterministic_arrivals.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, name in zip(axes.flat, ("optimal", "never_dispatch", "always_dispatch", "threshold_15")):
        rows = [r for r in summary["monte_carlo"]["rows"] if r["policy"] == name]
        ax.errorbar([r["initial_state"] for r in rows], [r["error"] for r in rows],
                    yerr=[1.959963984540054 * r["standard_error"] for r in rows],
                    fmt="o", capsize=4, color=COLORS[0])
        ax.axhline(0, color="#666666", linewidth=1)
        ax.set(xlabel="Initial passengers waiting", ylabel="MC mean minus exact cost",
               title=name.replace("_", " ").title())
        ax.grid(alpha=0.2)
    fig.suptitle(f"Monte Carlo error and individual 95% confidence intervals\n"
                 f"{summary['metadata']['episodes']:,} episodes; exact terminal continuation")
    save(fig, "monte_carlo_validation.png")
