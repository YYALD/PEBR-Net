#!/usr/bin/env python3
# PEBR-Net -- MIT License, see LICENSE.
"""Train PEBR-Net on a paired dataset, test it on the held-out profiles and render the figure package.

usage: python scripts/train.py @configs/train.args            # the default training parameters
       python scripts/train.py --paired_dir data/paired [--mode train] [any pebrnet option]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from pebrnet.__main__ import run   # the entry point of `python -m pebrnet`, including its GPU pinning
    argv = sys.argv[1:]
    if not any(a == "--mode" or a.startswith("@") for a in argv):
        argv = ["--mode", "all"] + argv
    raise SystemExit(run(argv))
