"""Check LaTeX inputs/macros without claiming a compiler or rendered layout."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
report = ROOT / "report"
files = [report / "report.tex"] + list((report / "sections").glob("*.tex"))
text = "\n".join(path.read_text(encoding="utf-8") for path in files)
declared = set(re.findall(r"\\newcommand\{\\(PG[A-Za-z]+)\}", text))
used = set(re.findall(r"\\(PG[A-Za-z]+)", text))
assert used <= declared, f"Undefined Part 1 macros: {used-declared}"
for name in re.findall(r"\\input\{([^}]+)\}", text):
    if "..." not in name:
        assert (report / (name+".tex")).exists(), name
for path in files:
    # Ignore comments, escaped braces, and require balanced grouping.
    source = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text(encoding="utf-8"))
    source = source.replace(r"\{", "").replace(r"\}", "")
    level = 0
    for char in source:
        level += (char == "{") - (char == "}")
        assert level >= 0, f"Unbalanced closing brace: {path}"
    assert level == 0, f"Unbalanced braces: {path}"
print(f"LaTeX source checks passed: {len(files)} files, {len(declared)} Part 1 macros. No rendering claimed.")
