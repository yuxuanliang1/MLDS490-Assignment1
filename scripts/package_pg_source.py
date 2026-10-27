"""Create an allowlisted source snapshot for hosts without a Git remote."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/pg_source.tar.gz")
    args = parser.parse_args()
    files = [ROOT / p for p in ("README.md", "requirements.txt", "requirements-pg.txt")]
    for folder in ("policy_gradient_part1", "scripts", "configs", "docs"):
        files.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".sh", ".json", ".md", ".ps1"))
    files = sorted(set(files))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest_path = args.output.parent / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    with tarfile.open(args.output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix())
        archive.add(manifest_path, arcname="source_manifest.json")
    print(f"{args.output}: {len(files)} source files; SHA256={hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
