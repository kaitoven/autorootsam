"""Utility to export the full repository source code into a single text file.

This is useful for paper supplements or code review.

Example:
  python tools/print_repo.py --root . --out full_source.txt
"""

import argparse
import os
from pathlib import Path


DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "checkpoints",
    "runs",
    "runs_tl",
    "outputs",
}


def is_text_file(p: Path) -> bool:
    if p.suffix.lower() in {".pt", ".pth", ".png", ".jpg", ".jpeg", ".npy", ".npz", ".zip", ".pdf"}:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="full_source.txt")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()

    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in DEFAULT_EXCLUDES for part in rel.parts):
            continue
        if not is_text_file(p):
            continue
        files.append(p)

    files = sorted(files)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in files:
            rel = p.relative_to(root)
            f.write("=" * 88 + "\n")
            f.write(f"FILE: {rel}\n")
            f.write("=" * 88 + "\n")
            try:
                f.write(p.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                f.write(p.read_text(encoding="latin-1"))
            f.write("\n\n")

    print(f"Wrote: {out} (files={len(files)})")


if __name__ == "__main__":
    main()
