#!/usr/bin/env python3
"""Fetch the planning datasets.

The Neural A* datasets are *committed* into the planning-datasets repository --
there is nothing to generate and no training run to wait for. This script
clones that repo at a pinned commit and copies the two ``.npz`` files used
into the working directory.

Run it once, before anything else::

    python fetch_data.py

Files produced:

``mazes_032_moore_c8.npz``
    32x32 mazes, 8-connected. This is the dataset the shipped checkpoint was
    trained on, so it is the only one where Neural A* results are meaningful.
    800 train / 100 valid / 100 test.

``all_064_moore_c16.npz``
    64x64 maps, a mix of all eight map families. The checkpoint was not trained
    on these, so treat the planning quality numbers with suspicion -- but it is
    a free map-size scaling axis for the latency and hardware-sizing analysis,
    which is what it is needed for.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/omron-sinicx/planning-datasets"
PINNED_COMMIT = "7f8953c4b0f511c2cc03410abfdb3687d54deafb"
CLONE_DIR = Path("planning-datasets")

WANTED = ["mazes_032_moore_c8.npz", "all_064_moore_c16.npz"]


def main() -> int:
    target_dir = Path.cwd()

    missing = [name for name in WANTED if not (target_dir / name).exists()]
    if not missing:
        print("All datasets already present:")
        for name in WANTED:
            size_kb = (target_dir / name).stat().st_size // 1024
            print(f"  {name}  ({size_kb:,} KB)")
        return 0

    print(f"Missing: {', '.join(missing)}")

    if not CLONE_DIR.exists():
        print(f"Cloning {REPO_URL} (~90 MB, one-off)...")
        try:
            subprocess.run(["git", "clone", "-q", REPO_URL, str(CLONE_DIR)], check=True)
        except subprocess.CalledProcessError:
            print(
                "\nClone failed. If this machine has no outbound network access, "
                "download the two .npz files locally from:\n"
                f"  {REPO_URL}/tree/{PINNED_COMMIT}/data/mpd\n"
                "and scp them into this directory instead.",
                file=sys.stderr,
            )
            return 1
        subprocess.run(
            ["git", "checkout", "-q", PINNED_COMMIT], cwd=CLONE_DIR, check=True
        )
    else:
        print(f"Reusing existing {CLONE_DIR}/")

    source_dir = CLONE_DIR / "data" / "mpd"
    for name in missing:
        src = source_dir / name
        if not src.exists():
            print(f"  ERROR: {src} not found in the clone", file=sys.stderr)
            return 1
        shutil.copy2(src, target_dir / name)
        print(f"  copied {name}  ({src.stat().st_size // 1024:,} KB)")

    print("\nDone. Next:")
    print("  pytest test_reference.py -q      # 18 tests, ~2 seconds")
    print("  jupyter lab 01_baseline.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
