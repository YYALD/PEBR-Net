"""The dataset check of scripts/download_data.py compares content, not bytes."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)


def _write_set(folder: Path, frames, **kw):
    folder.mkdir(parents=True, exist_ok=True)
    for name, fr in frames.items():
        fr.to_csv(folder / name, index=False, **kw)


def _frames(seed=0):
    rng = np.random.default_rng(seed)
    t = np.array([0.0] + list(np.geomspace(5e-4, 2e-2, 30)))
    out = {}
    for k, name in enumerate(dd.FILES):
        rows = [(100 * k + s, z) for s in range(3) for z in np.arange(0.0, 50.0, 5.0)]
        fr = pd.DataFrame(rng.uniform(1e-3, 1e3, size=(len(rows), t.size)), columns=["t_%.12e_s" % v for v in t])
        fr.insert(0, "depth_m", [z for _s, z in rows])
        fr.insert(0, "sample_id", [s for s, _z in rows])
        out[name] = fr
    return out


def test_content_check_accepts_another_number_format_and_rejects_other_values(tmp_path):
    frames = _frames()
    _write_set(tmp_path / "a", frames)
    content = {name: dd.describe_file(tmp_path / "a" / name) for name in dd.FILES}
    _write_set(tmp_path / "b", frames, float_format="%.15g", lineterminator="\r\n", encoding="utf-8-sig")
    assert dd.check_folder(tmp_path / "b", content, {}) == []
    changed = {k: v.copy() for k, v in frames.items()}
    col = changed["test_noisy.csv"].columns[20]
    changed["test_noisy.csv"].loc[5, col] *= 1.01
    _write_set(tmp_path / "c", changed)
    problems = dd.check_folder(tmp_path / "c", content, {})
    assert len(problems) == 1 and problems[0].startswith("test_noisy.csv differs")
    (tmp_path / "c" / "val_clean.csv").unlink()
    assert any(p.startswith("missing") for p in dd.check_folder(tmp_path / "c", content, {}))
