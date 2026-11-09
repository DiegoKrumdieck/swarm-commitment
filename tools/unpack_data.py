"""Decompress data/logs/ back into logs/ so analysis.ipynb runs without any API calls.

    python tools/unpack_data.py                # -> logs/
    python tools/unpack_data.py --dest DIR     # -> DIR (e.g. a scratch copy)
    python tools/unpack_data.py --force        # overwrite existing files

Joins split files (<name>.gz.part000, ...) and checks every file against data/logs/MANIFEST.json.
"""
import argparse
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "logs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "logs"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dest = Path(a.dest)
    manifest = json.loads((SRC / "MANIFEST.json").read_text(encoding="utf-8"))
    n = skipped = 0
    for rel, info in manifest.items():
        out = dest / rel
        if out.exists() and not a.force:
            skipped += 1
            continue
        gz = SRC / (rel + ".gz")
        if info.get("parts"):
            blob = b"".join((gz.parent / p).read_bytes() for p in info["parts"])
        else:
            blob = gz.read_bytes()
        data = gzip.GzipFile(fileobj=io.BytesIO(blob)).read()
        if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
            sys.exit(f"checksum mismatch for {rel}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        n += 1
    print(f"unpacked {n} files into {dest} (verified against MANIFEST.json); skipped {skipped} existing "
          f"(use --force to overwrite)")


if __name__ == "__main__":
    main()
