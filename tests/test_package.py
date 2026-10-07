"""Package integrity: the namespace loader, the command line and the inactive measured-profile hooks."""
import ast
import builtins
import re
import symtable
from pathlib import Path

import pytest

import pebrnet as pn

PKG = Path(pn.__file__).resolve().parent


def _top_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        for sub in ast.walk(n):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store) and not isinstance(
                    n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(sub.id)
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                names |= {(a.asname or a.name).split(".")[0] for a in sub.names}
            if isinstance(sub, ast.Global):
                names |= set(sub.names)
            if isinstance(sub, ast.ExceptHandler) and sub.name:
                names.add(sub.name)
    return names


def test_every_global_read_is_defined_somewhere_in_the_package():
    texts = {f: (PKG / (f + ".py")).read_text(encoding="utf-8") for f in pn.FILES}
    top = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__package__", "__builtins__",
                                "__loader__", "__path__", "__conditional_annotations__"}
    for t in texts.values():
        top |= _top_names(ast.parse(t))
    missing = []

    def walk(tbl, f):
        for sym in tbl.get_symbols():
            if not sym.is_referenced():
                continue
            glob = sym.is_global() or (tbl.get_type() == "module" and not sym.is_assigned() and not sym.is_imported())
            if glob and sym.get_name() not in top:
                missing.append("%s: %s -> %s" % (f, tbl.get_name(), sym.get_name()))
        for ch in tbl.get_children():
            walk(ch, f)

    for f, t in texts.items():
        walk(symtable.symtable(t, f + ".py", "exec"), f)
    assert not missing, missing[:20]


def test_view_modules_and_names():
    import pebrnet.gate_loss as gl
    import pebrnet.model as md
    assert gl.run_gate_loss_suite is pn.run_gate_loss_suite
    assert md.PEBRNet is pn.PEBRNet
    assert pn.PROGRAM_NAME == "PEBR_NET" and pn.__version__ == pn.PROGRAM_VERSION


def test_measured_profile_hooks_are_inactive():
    cfg = pn.Config()
    assert cfg.paths.field_csv == "" and cfg.data.measured_noise_inject == 0.0
    for name in ("infer_field", "run_field_if_qualified", "field_preflight", "load_field_matrix"):
        obj = getattr(pn, name, None)          # absent, or an inactive hook
        assert obj is None or "inactive" in obj.__doc__, name
    with pytest.raises(RuntimeError):
        pn.load_field_matrix(Path("x.csv"), 31, None)
    src = "\n".join((PKG / (f + ".py")).read_text(encoding="utf-8") for f in pn.FILES)
    assert not re.search(r"[\u3000-\u9fff]|\bFIX-\d|/home/|\b[A-Z]:[\\/](?:Users|work|File)", src)
    assert pn.Config().runtime.deployment_gate is False        # the held-out test is recorded, not enforced


def test_help_lists_public_modes_only(capsys):
    with pytest.raises(SystemExit) as e:
        pn.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "gateloss" in out and "--paired_dir" in out
    assert "--field_csv" not in out and "evidence" not in out.split("--mode")[1].split("\n")[0]


def test_public_modes():
    p = pn.build_parser()
    mode = next(a for a in p._actions if a.dest == "mode")
    assert set(mode.choices) == {"train", "all", "figures", "gateloss", "diagnose", "audit", "ablation_report"}
    with pytest.raises(SystemExit):
        p.parse_args(["--mode", "synth"])
    ablation = next(a for a in p._actions if a.dest == "ablation")
    assert "full" in ablation.choices and not any(c.endswith("_ip") or "_ip_" in c for c in ablation.choices)


def test_figure_package_lists_the_public_figures():
    src = (PKG / "public.py").read_text(encoding="utf-8")
    body = src[src.index("def produce_figure_package"):]
    assert set(re.findall(r'"(fig_[a-z_]+)"', body)) == {
        "fig_test_profile_comparison", "fig_val_profile_comparison", "fig_gate_loss_reconstruction",
        "fig_gate_loss_mechanisms", "fig_gate_loss_levels"}
    assert pn.Config().runtime.profile_extras is False
