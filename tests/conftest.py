"""Test configuration: the repository root on sys.path, and a writer of small paired datasets in the layout of
the paired dataset (the tests do not need the dataset itself)."""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")

# the 31 gate times of the paired dataset in seconds, as its CSV headers give them
GATES_S = np.array([
    0.0, 0.0005236080773384, 0.001152607619101, 0.001776110458744, 0.002547856536238, 0.003353844383394,
    0.004147711721695, 0.004685442968505, 0.005274647955725, 0.005920253951327, 0.006384637233729, 0.006878201834764,
    0.00767730628632, 0.008252100744119, 0.008863014633551, 0.009852113137651, 0.01056356999072, 0.01171545231206,
    0.01254399928494, 0.01342461120963, 0.01388545605261, 0.01485036359872, 0.01535532231451, 0.01587590484037,
    0.01641259458899, 0.01696588993018, 0.01753630465374, 0.01812436844649, 0.01873062738415, 0.01935564443848, 0.02])


# a positive 31-gate axis (ms) for the synthetic power-law decays of the mechanism tests
GATES_POS_MS = np.geomspace(0.5, 20.0, 31)

def clean_profile(rng, t_s=GATES_S, n_stations=61):
    """A positive decaying profile [stations, gates]: two relaxations and a conductor-like late departure."""
    z = np.arange(n_stations, dtype=np.float64)
    a1, a2 = rng.uniform(800, 2000), rng.uniform(5, 40)
    tau1, tau2 = rng.uniform(0.002, 0.004), rng.uniform(0.008, 0.02)
    host = a1 * np.exp(-t_s[None, :] / tau1) + a2 * np.exp(-t_s[None, :] / tau2)
    c = rng.uniform(15, 45)
    body = 1.0 + 0.4 * np.exp(-0.5 * ((z - c) / 3.0) ** 2)[:, None] * (t_s[None, :] / t_s[-1])
    return host * (1.0 + 0.002 * z[:, None]) * body


def write_paired_dataset(root, splits, seed=0, copies=()):
    """Six paired CSV files in the layout of the paired dataset under `root`.

    splits: {"train": [sample ids], "val": [...], "test": [...]}; copies: (id_a, id_b) pairs whose clean profiles
    agree to 1e-5 relative, each with its own noise."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clean = {}
    for sp in ("train", "val", "test"):
        for sid in splits[sp]:
            clean[sid] = clean_profile(rng)
    for a, b in copies:
        clean[b] = clean[a] * (1.0 + 1e-5 * rng.standard_normal(clean[a].shape))
    hdr = "sample_id,depth_m," + ",".join("t_%.12e_s" % v for v in GATES_S)
    depth = np.arange(61) * 5.0
    for sp in ("train", "val", "test"):
        lines_c, lines_n = [hdr], [hdr]
        for sid in splits[sp]:
            y = clean[sid]
            rel = 0.01 + 0.4 * (np.arange(31) / 30.0) ** 3
            n = y + rng.standard_normal(y.shape) * rel[None, :] * np.sqrt(np.mean(y ** 2, axis=0))[None, :]
            for i in range(61):
                lines_c.append("%d,%g," % (sid, depth[i]) + ",".join("%.10e" % v for v in y[i]))
                lines_n.append("%d,%g," % (sid, depth[i]) + ",".join("%.10e" % v for v in n[i]))
        (root / ("%s_clean.csv" % sp)).write_text("\n".join(lines_c) + "\n", encoding="utf-8")
        (root / ("%s_noisy.csv" % sp)).write_text("\n".join(lines_n) + "\n", encoding="utf-8")
    return root
