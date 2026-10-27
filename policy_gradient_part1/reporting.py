"""LaTeX macros from executed Part 1 evidence; missing results remain explicit."""

import json
from pathlib import Path


def write_report_results(root):
    root = Path(root)
    results = root / "results/policy_gradient_part1"
    lines = ["% Generated from actual Part 1 JSON evidence. Missing experiments are labeled pending."]

    def macro(name, text):
        lines.append("\\newcommand{\\"+name+"}{"+str(text)+"}")

    def table(name, columns, header, rows):
        text = [r"\begin{tabular}{"+columns+"}", r"\toprule", header+r" \\", r"\midrule"]
        text.extend(" & ".join(row)+r" \\" for row in rows)
        text.extend([r"\bottomrule", r"\end{tabular}"])
        macro(name, "\n"+"\n".join(text)+"\n")

    validation = json.loads((results / "validation/summary.json").read_text(encoding="utf-8"))
    e = validation["evidence"]
    macro("PGValidationPasses", f"{sum(validation['checks'].values())}/{len(validation['checks'])}")
    macro("PGExactGradientError", f"{e['analytical_gradient']['exact_max_error']:.3g}")
    macro("PGEmpiricalGradientError", f"{e['analytical_gradient']['empirical_max_error']:.5f}")
    macro("PGChunkGradientError", f"{e['chunking']['max_gradient_error']:.3g}")
    macro("PGBanditFinal", f"{e['controlled_decision']['final_better_action_probability']:.5f}")
    cartpole_path = results / "cartpole/experiment_summary.json"
    if cartpole_path.exists():
        c = json.loads(cartpole_path.read_text(encoding="utf-8"))
        primary, random = c["primary"], c["random_reference"]
        macro("PGCartPoleStatus", "Completed: three training seeds, separate primary 500-episode evaluation.")
        macro("PGCartPoleLR", f"{c['selected_learning_rate']:g}")
        macro("PGCartPoleEpisodes", primary["training_episodes"])
        macro("PGCartPoleMean", f"{primary['mean']:.3f}")
        macro("PGCartPoleSD", f"{primary['sample_std']:.3f}")
        macro("PGCartPoleRange", f"{primary['minimum']:g}--{primary['maximum']:g}")
        macro("PGCartPoleDifference", f"{c['trained_minus_random']:.3f}")
        macro("PGCartPoleDifferenceCI", ", ".join(f"{n:.3f}" for n in c["difference_ci95_normal"]))
        table("PGCartPoleEvaluationTable", "lrrrr", r"Policy & $n$ & Mean & Sample SD & 95\% CI", [
            [label, str(r["episodes"]), f"{r['mean']:.3f}", f"{r['sample_std']:.3f}",
             "["+", ".join(f"{n:.2f}" for n in r["ci95_normal"])+"]"]
            for label, r in (("Trained, seed 11", primary), ("Uniform random", random))])
        table("PGCartPolePilotTable", "rr", r"Learning rate & Final 100 training episodes: mean reward", [
            [f"{r['learning_rate']:g}", f"{r['mean_last100']:.2f}"] for r in c["pilots"]])
        table("PGCartPoleSeedTable", "rrrr", r"Training seed & Diagnostic $n$ & Mean & Sample SD", [
            [str(r["training_seed"]), str(r["episodes"]), f"{r['mean']:.3f}", f"{r['sample_std']:.3f}"]
            for r in c["seed_robustness"]])
    else:
        macro("PGCartPoleStatus", "Pending completion of the recorded experiment suite.")
    pong_path = results / "pong/pilot_summary.json"
    if pong_path.exists():
        p = json.loads(pong_path.read_text(encoding="utf-8"))
        table("PGPongPilotTable", "lrrr", r"Input & LR & Episodes & Mean raw training reward", [
            [r["representation"].capitalize(), f"{r['learning_rate']:g}", str(r["completed_episodes"]),
             f"{r['mean_last100']:.2f}"] for r in p["pilots"]])
        table("PGPongReferenceTable", "lrrr", r"Policy & Diagnostic $n$ & Mean & Sample SD", [
            [name.replace("_", " ").capitalize(), str(r["episodes"]), f"{r['mean']:.3f}", f"{r['sample_std']:.3f}"]
            for name, r in p["references"].items()])
    final_path = results / "pong/evaluation/summary.json"
    if final_path.exists():
        p = json.loads(final_path.read_text(encoding="utf-8"))
        if p["role"] != "final" or p["episodes"] != 500 or p["truncated_count"]:
            raise ValueError("Pong final summary does not meet the 500 complete-game protocol")
        macro("PGPongFinalStatus", f"Completed 500 stochastic games: mean {p['mean']:.3f}, sample SD "
              f"{p['sample_std']:.3f}, range {p['minimum']:g}--{p['maximum']:g}. "
              f"Training seed {p['training_seed']}, evaluation seed {p['evaluation_seed']}.")
    else:
        macro("PGPongFinalStatus", r"\textbf{Pending: formal GPU training and its independent 500-game evaluation "
              r"are not yet complete. No final mean, SD, or histogram is reported.}")
    target = root / "report/sections/policy_gradient_1_results.tex"
    target.write_text("\n\n".join(lines)+"\n", encoding="utf-8")
