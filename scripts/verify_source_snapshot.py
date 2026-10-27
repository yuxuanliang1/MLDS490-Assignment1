"""Verify a deployed allowlisted source manifest before remote experiments."""
import hashlib
import json
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    differences = [name for name, expected in manifest.items()
                   if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest() != expected]
    if differences:
        raise SystemExit(f"Source manifest mismatch: {differences}")
    print(f"Verified {len(manifest)} deployed source hashes")
