"""Check stored rewards/statistics, figures, checkpoints, and Component 1 preservation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy_gradient_part1.evaluator import reward_statistics
from policy_gradient_part1.runtime import load_checkpoint, write_json


def main(output=None):
    checks, details = {}, {}
    manifest_path = ROOT / "results/policy_gradient_part1/preservation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        changed = [name for name, expected in manifest.items()
                   if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected]
        checks["component1_unchanged"] = not changed
        details["component1_changed"] = changed
        details["preserved_files"] = len(manifest)
    evaluations = []
    for path in (ROOT / "results/policy_gradient_part1").rglob("rewards.csv"):
        summary_path = path.parent / "summary.json"
        if not summary_path.exists():
            details.setdefault("partial_evaluations", []).append(str(path.relative_to(ROOT)))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        stats = reward_statistics([float(row["reward"]) for row in rows])
        name = path.relative_to(ROOT).as_posix()
        matches = all(np.isclose(summary[key], stats[key], rtol=1e-12, atol=1e-12)
                      for key in ("mean", "sample_std", "minimum", "maximum", "episodes"))
        checks[name+"_statistics"] = bool(matches)
        checks[name+"_ended_episodes"] = all(row["terminated"] == "True" or row["truncated"] == "True"
                                                   for row in rows)
        checks[name+"_ending_counts"] = (summary["terminated_count"] == sum(row["terminated"] == "True" for row in rows)
                                         and summary["truncated_count"] == sum(row["truncated"] == "True" for row in rows))
        if summary["environment"]["id"] == "CartPole-v1":
            checks[name+"_cartpole_reward_equals_steps"] = all(float(row["reward"]) == int(row["steps"]) for row in rows)
        if summary["role"] == "final":
            checks[name+"_500_episodes"] = len(rows) == 500 and summary["reference"] is None
            if summary["environment"]["id"] == "ALE/Pong-v5":
                checks[name+"_complete_pong_games"] = all(row["truncated"] == "False" for row in rows)
        evaluations.append(dict(path=name, role=summary["role"], **stats))
    details["evaluations"] = evaluations
    for path in (ROOT / "results/policy_gradient_part1").rglob("latest.pt"):
        checkpoint = load_checkpoint(path)
        name = path.relative_to(ROOT).as_posix()
        history = checkpoint["history"]
        checks[name+"_count"] = checkpoint["completed_episodes"] == len(history)
        checks[name+"_finite"] = all(np.isfinite(float(row[key])) for row in history
                                     for key in ("reward", "loss_contribution", "gradient_norm_before_clip", "entropy"))
        checks[name+"_on_policy_recomputation"] = max(r["logprob_recompute_max_error"] for r in history) < 1e-4
    figures = []
    for path in (ROOT / "figures/policy_gradient_part1").rglob("*.png"):
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            checks[path.relative_to(ROOT).as_posix()+"_image"] = (
                min(dpi) >= 299.9 and min(image.size) >= 600 and np.asarray(image).std() > 1)
            figures.append(dict(path=path.relative_to(ROOT).as_posix(), size=image.size, dpi=dpi))
    details["figures"] = figures
    expected = [f"figures/policy_gradient_part1/{environment}/{name}.png"
                for environment in ("cartpole", "pong")
                for name in ("training_rewards", "evaluation_histogram")]
    details["missing_required_figures"] = [name for name in expected if not (ROOT/name).exists()]
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(all_available_checks_passed=all(checks.values()), checks=checks, details=details,
                  assignment_complete=not details["missing_required_figures"] and all(checks.values()))
    write_json(output or ROOT / "results/policy_gradient_part1/artifact_audit.json", result)
    print(f"Artifact checks: {sum(checks.values())}/{len(checks)} passed")
    print("Missing required figures:", details["missing_required_figures"])
    if not all(checks.values()):
        raise SystemExit([name for name, passed in checks.items() if not passed])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save a fresh audit without replacing historical evidence")
    main(parser.parse_args().output)
