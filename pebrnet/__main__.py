# PEBR-Net -- MIT License, see LICENSE.
"""``python -m pebrnet ...`` and the ``pebrnet`` console command: the command line of the program."""
from __future__ import annotations

import os
import sys


def run(argv=None) -> int:
    import pebrnet
    args = list(sys.argv[1:] if argv is None else argv)
    try:   # pin the GPU named by --device before CUDA initialises (it reads CUDA_VISIBLE_DEVICES once)
        args, notes = pebrnet.pin_visible_gpu(args, os.environ)
        for note in notes:
            print(note, file=sys.stderr)
    except Exception:   # noqa: BLE001 - pinning is a convenience; the run proceeds without it
        pass
    try:
        return int(pebrnet.main(args) or 0)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(run())
