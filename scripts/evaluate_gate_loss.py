#!/usr/bin/env python3
# PEBR-Net -- MIT License, see LICENSE.
"""The gate-loss cohort of the manuscript (Section 5.2) from a trained checkpoint.

The last 3, 6, 12, 18, 19 and 22 of 31 gates (9.7-71.0 %) of every station of the held-out test profiles are
withheld; the full network, the network without CDM-R and the network without E-GSR are scored against the clean
reference (amplitude NRMSE, late NRMSE of Eq. 7, decay-slope RMSE and temporal-structure NRMSE of Eq. A13).

usage: python scripts/evaluate_gate_loss.py --checkpoint checkpoints/pebrnet_checkpoint.pt --paired_dir data/paired
       [any further pebrnet option, e.g. --gate_loss_withheld 3,6,12,18 --gate_loss_profiles 0 --device cuda:0]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from pebrnet.__main__ import run   # the entry point of `python -m pebrnet`, including its GPU pinning
    argv = sys.argv[1:]
    if "--checkpoint" not in argv:
        raise SystemExit(__doc__)
    raise SystemExit(run(["--mode", "gateloss"] + argv))
