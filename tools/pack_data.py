"""Compress logs/ into data/logs/ (same structure, each file gzipped as <name>.gz).

    python tools/pack_data.py

Any compressed file over MAX_BYTES is split into <name>.gz.part000, .part001, ... (tools/unpack_data.py
joins them). A manifest (data/logs/MANIFEST.json) records every file's size and SHA-256 before compression.
"""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "logs"
DST = ROOT / "data" / "logs"
MAX_BYTES = 45 * 1024 * 1024  # stay well under GitHub's 50 MB warning


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if DST.exists():
        shutil.rmtree(DST)
    manifest, split = {}, []
    for src in sorted(SRC.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(SRC).as_posix()
        out = DST / (rel + ".gz")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(src, "rb") as fi, gzip.GzipFile(out, "wb", compresslevel=9, mtime=0) as fo:
            shutil.copyfileobj(fi, fo)
        parts = None
        if out.stat().st_size > MAX_BYTES:
            data = out.read_bytes()
            out.unlink()
            parts = []
            for i in range(0, len(data), MAX_BYTES):
                pp = out.with_name(out.name + f".part{i // MAX_BYTES:03d}")
                pp.write_bytes(data[i:i + MAX_BYTES])
                parts.append(pp.name)
            split.append(rel)
        manifest[rel] = {"bytes": src.stat().st_size, "sha256": sha256(src), "parts": parts}
    (DST / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file())
    print(f"packed {len(manifest)} files into {DST} ({total / 1e6:.1f} MB); split: {split or 'none'}")


if __name__ == "__main__":
    main()
