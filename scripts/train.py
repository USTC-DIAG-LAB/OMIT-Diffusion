#!/usr/bin/env python3
"""Stable training entry point for OMIT-Diffusion."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import main  # noqa: E402


if __name__ == "__main__":
    os.chdir(ROOT)
    if "--train" not in sys.argv and "-t" not in sys.argv:
        sys.argv.append("--train")
    main()
