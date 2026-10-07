# PEBR-Net -- prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.
# MIT License, see LICENSE.
"""PEBR-Net: prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.

The package is one namespace written across topic files. This loader executes the files listed in FILES, in that
order, into the package namespace, so module-level state and every cross-reference behave as in a single module.
Each file is also exposed as a view module (``pebrnet.model``, ``pebrnet.gate_loss``, ...) holding the names it
defines.

Quick start::

    import pebrnet
    pebrnet.main(["--mode", "audit", "--paired_dir", "data/paired"])          # read and audit the data
    pebrnet.main(["--mode", "all", "--paired_dir", "data/paired"])            # train, test and figures
"""
from __future__ import annotations

import sys as _sys
import types as _types
from pathlib import Path as _Path

FILES = ('config', 'utils', 'noise', 'data', 'measured', 'dataset', 'model', 'losses', 'training', 'inference', 'evaluation', 'figures', 'gate_loss', 'cli', 'public')
__version__ = '1.0.0'

_HERE = _Path(__file__).resolve().parent


def _load() -> None:
    ns = globals()
    doc = ns.get("__doc__")
    for _name in FILES:
        _path = _HERE / (_name + ".py")
        _before = set(ns)
        exec(compile(_path.read_text(encoding="utf-8"), str(_path), "exec"), ns)
        _defined = sorted(set(ns) - _before - {"__doc__"})
        _view = _types.ModuleType(__name__ + "." + _name, "Names defined in pebrnet/%s.py." % _name)
        for _k in _defined:
            setattr(_view, _k, ns[_k])
        _view.__file__ = str(_path)
        _view.__all__ = [k for k in _defined if not k.startswith("_")]
        _sys.modules[__name__ + "." + _name] = _view
    ns["__doc__"] = doc


_load()
del _load
