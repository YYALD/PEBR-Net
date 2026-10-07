# PEBR-Net, the prior- and evidence-bounded reconstruction network.
# MIT License, see LICENSE.
"""Run infrastructure and numerical helpers.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

_REAL_DFT_BASIS_CACHE: Dict[Tuple[int, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}
_AMP_DTYPE: Optional[torch.dtype] = None
_AMP_DTYPE_NAME: str = "fp16"


def set_amp_dtype(dtype: torch.dtype) -> None:
    global _AMP_DTYPE, _AMP_DTYPE_NAME
    _AMP_DTYPE = dtype
    _AMP_DTYPE_NAME = "bf16" if dtype is torch.bfloat16 else "fp16"


def amp_dtype_name() -> str:
    return _AMP_DTYPE_NAME


def amp_autocast_context(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager:
    """Return a non-deprecated autocast context with backward compatibility."""
    if not enabled:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type=device.type, enabled=False)
        return contextlib.nullcontext()
    dt = _AMP_DTYPE if _AMP_DTYPE is not None else torch.float16
    if device.type != "cuda":
        dt = torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, dtype=dt, enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def create_grad_scaler(device: torch.device, enabled: bool) -> Any:
    """Create GradScaler without the deprecated torch.cuda.amp API warning."""
    if enabled and _AMP_DTYPE is torch.bfloat16:
        enabled = False
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _real_dft_basis(length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if length < 2:
        raise ValueError(f"Spectrum length must be >=2, got {length}")
    device_index = -1 if device.index is None else int(device.index)
    key = (int(length), str(device.type), device_index)
    cached = _REAL_DFT_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    n = np.arange(length, dtype=np.float64)[None, :]
    k = np.arange(length // 2 + 1, dtype=np.float64)[:, None]
    angle = 2.0 * np.pi * k * n / float(length)
    cos_np = np.cos(angle).astype(np.float32, copy=False)
    sin_np = (-np.sin(angle)).astype(np.float32, copy=False)
    cos_basis = torch.from_numpy(cos_np).to(device=device, dtype=torch.float32)
    sin_basis = torch.from_numpy(sin_np).to(device=device, dtype=torch.float32)
    _REAL_DFT_BASIS_CACHE[key] = (cos_basis, sin_basis)
    return cos_basis, sin_basis


def log_spectrum_magnitude(
    x: torch.Tensor,
    eps: float,
    backend: str = "real_dft",
) -> torch.Tensor:
    """Differentiable log spectrum without CUDA complex ``abs``."""
    if x.ndim < 1:
        raise ValueError("Spectrum input must have at least one dimension")
    with amp_autocast_context(x.device, enabled=False):
        x32 = x.float()
        if backend == "real_dft":
            cos_basis, sin_basis = _real_dft_basis(x32.shape[-1], x32.device)
            real = torch.matmul(x32, cos_basis.transpose(0, 1))
            imag = torch.matmul(x32, sin_basis.transpose(0, 1))
        elif backend == "rfft_manual":
            spectrum = torch.fft.rfft(x32, dim=-1)
            parts = torch.view_as_real(spectrum)
            real = parts[..., 0]
            imag = parts[..., 1]
        else:
            raise ValueError(
                f"Unsupported spectrum backend {backend!r}; use 'real_dft' or 'rfft_manual'."
            )
        magnitude_sq = real.square() + imag.square()
        return 0.5 * torch.log(magnitude_sq.clamp_min(float(eps) ** 2))


def run_numeric_preflight(cfg: "Config", logger: logging.Logger) -> None:
    """Exercise AMP-sensitive numerical paths before loading the full dataset."""
    device = resolve_device(cfg.runtime.device)
    enabled = bool(cfg.train.amp and device.type == "cuda")
    probe = torch.linspace(-1.0, 1.0, cfg.data.target_gates, device=device, dtype=torch.float32)
    probe = probe.reshape(1, 1, -1).requires_grad_(True)
    logits = torch.linspace(-4.0, 4.0, cfg.data.target_gates, device=device, dtype=torch.float32)
    logits = logits.reshape(1, 1, -1).requires_grad_(True)
    target = torch.linspace(0.02, 0.98, cfg.data.target_gates, device=device, dtype=torch.float32)
    target = target.reshape(1, 1, -1)
    with amp_autocast_context(device, enabled):
        spectrum_value = log_spectrum_magnitude(
            probe, cfg.loss.eps, cfg.loss.spectrum_backend
        ).mean()
        reliability_value = F.binary_cross_entropy_with_logits(logits, target)
        value = spectrum_value + reliability_value
    value.backward()
    if probe.grad is None or not torch.isfinite(probe.grad).all():
        raise RuntimeError("Numerical preflight failed: spectrum gradient is missing or non-finite.")
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise RuntimeError("Numerical preflight failed: reliability BCE gradient is missing or non-finite.")
    logger.info(
        "Numerical preflight passed | spectrum_backend=%s | reliability_loss=bce_with_logits "
        "| AMP=%s | device=%s",
        cfg.loss.spectrum_backend,
        enabled,
        device,
    )


def atomic_json_dump(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


_RESOLVED_AUTO_DEVICE: Optional[torch.device] = None


def resolve_device(name: str) -> torch.device:
    """Resolve a device string, and fail with something actionable."""
    global _RESOLVED_AUTO_DEVICE
    if name == "auto" and _RESOLVED_AUTO_DEVICE is not None:
        return _RESOLVED_AUTO_DEVICE
    if name == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        _n = int(torch.cuda.device_count())
        if _n <= 1:
            return torch.device("cuda:0")
        _free = []
        for _i in range(_n):
            try:
                _f, _t = torch.cuda.mem_get_info(_i)
            except Exception:
                _f, _t = 0, 1
            _free.append((_f, _t, _i))
        _f, _t, _i = max(_free)
        logging.getLogger("btem").info(
            "--device auto: picked cuda:%d (freest). Cards: %s.",
            _i, " | ".join("cuda:%d %.1f/%.1f GB free" % (j, f / 2**30, t / 2**30)
                           for f, t, j in sorted(_free, key=lambda x: x[2])))
        _dev_auto = resolve_device("cuda:%d" % _i)
        _RESOLVED_AUTO_DEVICE = _dev_auto
        logging.getLogger("btem").info(
            "--device auto is now FROZEN at %s for this process; later resolutions return the same "
            "card so no stage can drift onto another.", _dev_auto)
        return _dev_auto
    _m = re.fullmatch(r"cuda:(\d+)", str(name).strip())
    if _m is not None and torch.cuda.is_available():
        _want = int(_m.group(1))
        if _want >= int(torch.cuda.device_count()):
            logging.getLogger("btem").warning(
                "requested %s but only %d CUDA device(s) visible -> falling back "
                "to --device auto (freest card).",
                name, int(torch.cuda.device_count()))
            return resolve_device("auto")
    try:
        dev = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Unrecognized device %r (%s). Use cpu, cuda, or cuda:<index>."
            % (name, exc)
        ) from exc
    if dev.type != "cuda":
        return dev
    if not torch.cuda.is_available():
        _alts = [b for b in ("mps", "xpu", "npu", "mlu", "musa")
                 if getattr(getattr(torch, b, None), "is_available", lambda: False)()]
        raise RuntimeError(
            "CUDA was requested (%s) but torch.cuda.is_available() is False. This build of PyTorch (%s) sees "
            "no CUDA-compatible device.%s Run with --device cpu to verify the pipeline, or install the "
            "vendor's PyTorch build for this accelerator."
            % (name, torch.__version__,
               (" Other backends this build does expose: %s." % ", ".join(_alts)) if _alts else "")
        )
    n = int(torch.cuda.device_count())
    idx = 0 if dev.index is None else int(dev.index)
    if idx >= n:
        _names = []
        for i in range(n):
            try:
                _names.append("%d=%s" % (i, torch.cuda.get_device_name(i)))
            except Exception:
                _names.append("%d=?" % i)
        raise RuntimeError(
            "Device %s was requested but only %d CUDA device(s) are visible: %s. If the card is physically "
            "present, CUDA_VISIBLE_DEVICES may be hiding it -- note that under CUDA_VISIBLE_DEVICES the "
            "indices are RENUMBERED from 0, so the second physical card is cuda:0 when only it is exposed."
            % (name, n, ", ".join(_names) if _names else "none")
        )
    try:
        torch.cuda.set_device(idx)
    except Exception as _sd_exc:
        if n > 1:
            raise RuntimeError(
                "torch.cuda.set_device(%d) failed on a %d-card system: %s. Untagged allocations would go to "
                "cuda:0 instead of the card you selected -- the exact way a run pinned to cuda:%d still fills "
                "cuda:0 and dies there. Set CUDA_VISIBLE_DEVICES=%d and use --device cuda:0."
                % (idx, n, _sd_exc, idx, idx)) from _sd_exc
        logging.getLogger("btem").warning(
            "torch.cuda.set_device(%d) failed: %s. Allocations that do not "
            "carry an explicit device may land on cuda:0.", idx, _sd_exc)
    return dev


def setup_logging(run_dir: Path, log_root: str = "", run_name: str = "",
                  retain: int = 60) -> logging.Logger:
    """Console + run-directory log + a managed copy under the log root."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(PROGRAM_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    root = str(log_root or "").strip()
    if root:
        try:
            rp = Path(root)
            rp.mkdir(parents=True, exist_ok=True)
            name = (run_name or run_dir.name).strip() or run_dir.name
            gh = logging.FileHandler(rp / ("%s.log" % name), encoding="utf-8")
            gh.setFormatter(fmt)
            logger.addHandler(gh)
            _prune_log_root(rp, int(retain), logger)
        except Exception as exc:
            logger.warning("Log root %s unusable (%s); run-directory log only.",
                           root, exc)
    return logger


def _prune_log_root(root: Path, retain: int, logger: logging.Logger) -> None:
    if retain <= 0:
        return
    try:
        runs = sorted((p for p in root.glob("*.log") if p.is_file()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        stale = runs[retain:]
        for p in stale:
            for suffix in ("", ".anomaly", ".metrics"):
                q = p.with_name(p.stem + suffix + p.suffix) if suffix else p
                if q.exists():
                    q.unlink()
            csv_p = root / (p.stem + "_epoch_metrics.csv")
            if csv_p.exists():
                csv_p.unlink()
        if stale:
            logger.info(
                "Log retention: kept the %d newest run logs under %s, pruned %d older one(s).",
                retain, root, len(stale))
    except Exception as exc:
        logger.warning("Log pruning skipped: %s", exc)


_LAST_MONITOR_SUMMARY: Dict[str, Any] = {}
_LAST_DEPLOYMENT: Dict[str, Any] = {}


def append_run_index(cfg: "Config", run_dir: Path, row: Mapping[str, Any],
                     logger: logging.Logger) -> None:
    """One line per run in ``<log_root>/runs_index.csv``."""
    root = str(getattr(cfg.paths, "log_root", "") or "").strip()
    if not root:
        return
    try:
        rp = Path(root)
        rp.mkdir(parents=True, exist_ok=True)
        idx = rp / "runs_index.csv"
        cols = ["run", "started", "finished", "mode", "supervision", "ablation",
                "device", "epochs", "global_percent", "early_mid_percent",
                "late_percent", "late_cvar_percent", "anomaly_recovery",
                "anomaly_auroc", "false_anomaly_upper95", "feasible", "exit_code",
                "run_dir", "log"]
        rec = {c: row.get(c, "") for c in cols}
        rec["run"] = rec["run"] or run_dir.name
        rec["run_dir"] = str(run_dir)
        rec["log"] = str(rp / ("%s.log" % run_dir.name))
        new = not idx.exists()
        with idx.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow(rec)
        logger.info("Run index updated: %s", idx)
    except Exception as exc:
        logger.warning("Run index not written: %s", exc)


_OUTPUT_TAG = ""


def set_output_tag(tag: str) -> None:
    global _OUTPUT_TAG
    _OUTPUT_TAG = tag


def output_tag() -> str:
    return _OUTPUT_TAG


def append_output_tag(tag: str) -> str:
    """Add a status tag without discarding the ablation tag already in place."""
    global _OUTPUT_TAG
    t = str(tag or "")
    if not t:
        return _OUTPUT_TAG
    if not t.startswith("__"):
        t = "__" + t
    if t not in _OUTPUT_TAG:
        _OUTPUT_TAG = f"{_OUTPUT_TAG}{t}"
    return _OUTPUT_TAG


def tagged_name(filename: str) -> str:
    """Insert the ablation tag before the extension so every artefact (figure, CSV, TorchScript) names which model
    variant produced it.
    """
    if not _OUTPUT_TAG:
        return filename
    stem, dot, ext = filename.rpartition(".")
    return f"{stem}{_OUTPUT_TAG}.{ext}" if dot else f"{filename}{_OUTPUT_TAG}"


def apply_ablation_preset(cfg: Config, name: str, logger: Optional[logging.Logger] = None) -> None:
    """Apply a named preset onto the config (after individual flags, so the preset is authoritative) and install the
    output tag.
    """
    if name not in ABLATION_PRESETS:
        raise ValueError(f"Unknown ablation preset '{name}'. Choices: {list(ABLATION_PRESETS)}")
    desc, overrides = ABLATION_PRESETS[name]
    for path, value in overrides.items():
        section, attr = path.split(".", 1)
        setattr(getattr(cfg, section), attr, value)
    cfg.runtime.ablation = name
    set_output_tag("" if name == "full" else f"__{name}")
    if logger is not None:
        logger.info("Ablation preset: %s | %s | overrides=%s", name, desc, overrides or "none")


def _link_subdir(run_dir: Path, sub: str, target_root: str,
                 run_name: str) -> Tuple[Path, str]:
    link = run_dir / sub
    root = str(target_root or "").strip()
    if not root:
        link.mkdir(exist_ok=True)
        return link, "local (no root configured)"
    target = Path(root) / run_name
    try:
        target.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            return link, "local (already present)"
        os.symlink(str(target.resolve()), str(link), target_is_directory=True)
        return target, "linked -> %s" % target
    except Exception as exc:
        link.mkdir(exist_ok=True)
        return link, "local (link failed: %s; mirrored at exit)" % exc


def mirror_run_outputs(cfg: Config, run_dir: Path, logger: logging.Logger) -> None:
    """Copy checkpoints/ and figures/ to their roots when they could not be linked, so the delivered artefacts are in
    the configured trees either way.
    """
    for sub, root in (("checkpoints", getattr(cfg.paths, "model_root", "")),
                      ("figures", getattr(cfg.paths, "plot_root", ""))):
        src = run_dir / sub
        if not root or src.is_symlink() or not src.is_dir():
            continue
        dst = Path(root) / run_dir.name
        try:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
            logger.info("Mirrored %s -> %s", src, dst)
        except Exception as exc:
            logger.warning("Could not mirror %s -> %s: %s", src, dst, exc)


def make_run_dir(cfg: Config) -> Path:
    root = Path(cfg.paths.output_root)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    default = f"PEBRNet_{stamp}"
    if cfg.runtime.ablation != "full":
        default = f"ABL_{cfg.runtime.ablation}_{stamp}"
    name = cfg.runtime.run_name.strip() or default
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ["cache", "metrics", "reports"]:
        (run_dir / sub).mkdir(exist_ok=True)
    _ck, _ckw = _link_subdir(run_dir, "checkpoints", getattr(cfg.paths, "model_root", ""), name)
    _fg, _fgw = _link_subdir(run_dir, "figures", getattr(cfg.paths, "plot_root", ""), name)
    cfg.runtime.resolved_checkpoint_dir = str(_ck)
    cfg.runtime.resolved_figure_dir = str(_fg)
    cfg.runtime.checkpoint_dir_note = _ckw
    cfg.runtime.figure_dir_note = _fgw
    return run_dir


def model_to_native_axis(matrix: np.ndarray, target_time: np.ndarray, native_axis: np.ndarray) -> np.ndarray:
    """Signed log-time resampling of a model-axis matrix [n, T] back onto the measured channels [n, n_field]."""
    plan = make_logtime_plan(np.asarray(target_time, dtype=np.float64), np.asarray(native_axis, dtype=np.float64))
    return apply_logtime_plan(np.asarray(matrix, dtype=np.float64), plan, uncovered_policy="hold_edge")


def native_presentation_axis(cfg: Config, target_time: np.ndarray) -> Optional[np.ndarray]:
    """The measured gate axis every figure and export is presented on: the file's channel count
    (cfg.data.field_native_gates, 31) log-spaced over the model span, i.e.
    """
    if not bool(getattr(cfg.runtime, "native_presentation", True)):
        return None
    tt = np.asarray(target_time, dtype=np.float64).reshape(-1)
    n = int(getattr(cfg.data, "field_native_gates", 0) or 0)
    if n < 2 or n >= tt.size or not bool(getattr(cfg.data, "unify_axis_to_measured", False)) or float(tt[0]) <= 0:
        return None
    return np.geomspace(float(tt[0]), float(tt[-1]), n)


def native_late_start(late_start: int, target_time: np.ndarray, native_axis: Optional[np.ndarray]) -> int:
    """The late-window start on the presentation axis: the number of measured gates that lie before the model's
    late-start time.
    """
    if native_axis is None:
        return int(late_start)
    tt = np.asarray(target_time, dtype=np.float64).reshape(-1)
    t0 = float(tt[min(max(int(late_start), 0), tt.size - 1)])
    return int(np.sum(np.asarray(native_axis, dtype=np.float64) < t0 * (1.0 - 1e-9)))


def to_presentation_axis(matrix: np.ndarray, target_time: np.ndarray, native_axis: Optional[np.ndarray]) -> np.ndarray:
    """A model-axis matrix [..., T] resampled onto the presentation axis (identity when off)."""
    m = np.asarray(matrix, dtype=np.float64)
    if native_axis is None or m.shape[-1] != np.asarray(target_time).size:
        return m
    if m.ndim == 1:
        return model_to_native_axis(m.reshape(1, -1), target_time, native_axis)[0]
    lead = m.shape[:-1]
    return model_to_native_axis(m.reshape(-1, m.shape[-1]), target_time, native_axis).reshape(lead + (int(native_axis.size),))


def write_native_axis_copies(out_dir: Path, cfg: Config, target_time: np.ndarray, logger: logging.Logger) -> None:
    """Re-issue an export directory on the measured axis."""
    try:
        nat = native_presentation_axis(cfg, target_time)
        if nat is None or not Path(out_dir).is_dir():
            return
        tt = np.asarray(target_time, dtype=np.float64).reshape(-1)
        T = int(tt.size)
        n_csv = n_npz = 0
        for p in sorted(Path(out_dir).glob("*.csv")):
            if p.stem.endswith("_modelaxis"):
                continue
            with open(p, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f))
            tcols = [i for i, h in enumerate(header) if str(h).startswith("t_")]
            if len(tcols) != T or tcols != list(range(tcols[0], tcols[0] + T)):
                continue
            df = pd.read_csv(p)
            block = to_presentation_axis(df.iloc[:, tcols].to_numpy(dtype=np.float64), tt, nat)
            keep = [h for i, h in enumerate(header) if i not in tcols]
            out = pd.DataFrame({h: df[h] for h in keep})
            for j, t_ in enumerate(nat):
                out["t_%.6g" % float(t_)] = block[:, j]
            p.rename(p.with_name(p.stem + "_modelaxis" + p.suffix))
            out.to_csv(p, index=False, float_format="%.9g")
            n_csv += 1
        for p in sorted(Path(out_dir).glob("*.npz")):
            if p.stem.endswith("_modelaxis"):
                continue
            with np.load(p, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            hit = False
            for k, v in list(arrays.items()):
                if k == "target_time" and np.asarray(v).size == T:
                    arrays[k] = np.asarray(nat, dtype=np.float64)
                    hit = True
                elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[-1] == T and v.dtype.kind == "f":
                    arrays[k] = to_presentation_axis(v, tt, nat)
                    hit = True
            if hit:
                p.rename(p.with_name(p.stem + "_modelaxis" + p.suffix))
                np.savez_compressed(p, **arrays)
                n_npz += 1
        logger.info("EXPORT ON THE MEASURED AXIS | %s | %d CSV + %d NPZ rewritten on %d gates (model-axis "
                    "originals kept as *_modelaxis.*).", str(out_dir), n_csv, n_npz, int(nat.size))
    except Exception as exc:
        logger.warning("native-axis export copies failed for %s: %r", str(out_dir), exc)


def write_final_report(
    cfg: Config,
    run_dir: Path,
    selected_checkpoint: Optional[Path],
    logger: logging.Logger,
) -> None:
    """Write reports/FINAL_REPORT.txt for a run."""
    synthetic_path = run_dir / "metrics" / "synthetic_test_metrics.json"
    field_path = run_dir / "metrics" / "field_no_reference_metrics.json"
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8")) if synthetic_path.exists() else None
    field_metrics = json.loads(field_path.read_text(encoding="utf-8")) if field_path.exists() else None
    lines = [
        f"{PROGRAM_NAME} v{PROGRAM_VERSION}",
        "=" * 100,
        f"Run directory: {run_dir}",
        f"Selected checkpoint: {selected_checkpoint}",
        "",
        ("Accuracy contract (the constraint targets of training):" if cfg.runtime.deployment_gate
         else "Constraint targets of training (augmented-Lagrangian inequality constraints, ContractConfig):"),
        f"  global <= {100 * cfg.contract.global_threshold:.2f}%",
        f"  early/middle <= {100 * cfg.contract.early_mid_threshold:.2f}%",
        f"  late <= {100 * cfg.contract.late_threshold:.2f}%",
        f"  late CVaR <= {100 * cfg.contract.late_cvar_threshold:.2f}%",
        "",
    ]
    if synthetic:
        lines.extend(
            [
                "Held-out paired test:",
                f"  global error: {synthetic['global_error_percent']:.6f}%",
                f"  early/middle error: {synthetic['early_mid_error_percent']:.6f}%",
                f"  late error: {synthetic['late_error_percent']:.6f}%",
                f"  late CVaR: {synthetic['late_cvar_percent']:.6f}%",
                *([f"  contract met: {synthetic['project_feasible']}"] if cfg.runtime.deployment_gate else []),
                "",
            ]
        )
        per_bin = synthetic.get("per_snr_bin")
        if isinstance(per_bin, dict):
            lines.append("Held-out per-SNR-bin errors (global / late):")
            for label, m in per_bin.items():
                if m.get("count"):
                    lines.append(
                        f"  {label:>9} dB: G={m['global_error_percent']:.4f}%  "
                        f"L={m['late_error_percent']:.4f}%  n={m['count']}"
                    )
            lines.append("")
    _ps = run_dir / "reports" / "test_profile_summary.json"
    if _ps.exists():
        _p = json.loads(_ps.read_text(encoding="utf-8"))
        _r, _w = _p["late_nrmse_percent_reconstruction"], _p["late_nrmse_percent_raw"]
        lines.extend([
            "Per-profile late NRMSE (Eq. 7), held-out test, %d profiles:" % int(_p["profiles"]),
            "  reconstruction: median %.2f%% (IQR %.2f-%.2f%%)" % (_r["median"], _r["p25"], _r["p75"]),
            "  raw record:     median %.2f%% (IQR %.2f-%.2f%%)" % (_w["median"], _w["p25"], _w["p75"]),
            "  profiles below the best raw profile: %d; better than their own raw record: %d"
            % (int(_p["profiles_below_the_best_raw_profile"]), int(_p["profiles_better_than_their_raw_record"])),
            "",
        ])
    if field_metrics:
        lines.extend(
            [
                "Field no-reference diagnostics:",
                f"  roughness reduction ratio: {field_metrics['roughness_reduction_ratio']:.6f}",
                f"  early/middle relative change: {field_metrics['early_mid_relative_change']:.6f}",
                f"  late relative change: {field_metrics['late_relative_change']:.6f}",
                f"  monotonic violation original: {field_metrics['monotonic_violation_original']:.6f}",
                f"  monotonic violation denoised: {field_metrics['monotonic_violation_denoised']:.6f}",
                "",
            ]
        )
    sim_summary_path = run_dir / "metrics" / "simulation_figure_summary.json"
    ingestion_path = run_dir / "reports" / "field_ingestion.json"
    fig_lines: List[str] = []
    if ingestion_path.exists():
        ing = json.loads(ingestion_path.read_text(encoding="utf-8"))
        fig_lines.append("Field ingestion:")
        fig_lines.append(f"  station column: {ing.get('station_column')!r} | rows: {ing.get('rows')} | gates: {ing.get('gates')}")
        fig_lines.append(f"  non-positive late samples kept: {ing.get('nonpositive_values')}")
        for act in ing.get("actions", []):
            fig_lines.append(f"    - {act}")
        fig_lines.append("")
    if sim_summary_path.exists():
        ss = json.loads(sim_summary_path.read_text(encoding="utf-8"))
        fig_lines.append("Simulation figure (held-out, with ground truth):")
        fig_lines.append(
            f"  late NRMSE median denoised {ss['late_nrmse_median_out_percent']:.3f}% "
            f"vs noisy {ss['late_nrmse_median_in_percent']:.3f}% over {ss['examples']} traces"
        )
        fig_lines.append("")
    fig_lines.append("Figures (vector PDF + 600-dpi PNG in the figures/ folder):")
    if field_profile_present(cfg):
        fig_lines.append("  fig_simulation_denoising  clean/noisy/denoised traces, per-gate error, NRMSE-vs-SNR, reliability")
        fig_lines.append("  fig_field_denoising       measured vs denoised decays and pseudo-sections with removed noise")
        fig_lines.append("  fig_field_profile_curves  per-station raw vs denoised decay-curve comparison grid across the profile")
        fig_lines.append("  fig_field_multigate_profile  multi-gate profile plot (one curve per gate across stations), raw vs denoised")
        fig_lines.append("  fig_field_tf_quantification  spatial-spectrum / roughness / preservation / removed-fraction quantification")
        fig_lines.append("  fig_field_reliability     per-gate reliability pseudo-section")
    else:
        _fd = run_dir / "figures"
        _st = sorted({p.stem.split("__")[0] for p in _fd.glob("*.pdf")}) if _fd.is_dir() else []
        fig_lines.extend("  %s" % s for s in _st)
        if not _st:
            fig_lines.append("  none yet: the figure package writes them (end of --mode all, or --mode figures)")
    fig_lines.append("")
    lines.extend(fig_lines)
    dq_path = run_dir / "reports" / "deployment_qualification.json"
    dq = json.loads(dq_path.read_text(encoding="utf-8")) if dq_path.exists() else None
    stress_path = run_dir / "metrics" / "stress_test_metrics.json"
    stress = json.loads(stress_path.read_text(encoding="utf-8")) if stress_path.exists() else None
    if stress:
        lines.extend(
            [
                "Stress test (%s; robustness only):" % stress.get("snr_protocol", "-8..6 dB"),
                f"  global error: {stress['global_error_percent']:.6f}%",
                f"  late error: {stress['late_error_percent']:.6f}%",
                f"  late CVaR: {stress['late_cvar_percent']:.6f}%",
                "",
            ]
        )
    if dq is not None and not cfg.runtime.deployment_gate:
        lines.extend(["The held-out test and the split audit are recorded in reports/deployment_qualification.json.",
                      ""])
    elif dq is not None:
        lines.extend(
            [
                ("Deployment qualification :" if field_profile_present(cfg)
                 else "Qualification (accuracy contract, anomaly preservation, split audit):"),
                f"  checkpoint feasible: {dq.get('checkpoint_feasible')}",
                f"  {'deployment allowed' if field_profile_present(cfg) else 'qualified'}: {dq.get('deployment_allowed')}",
                f"  test G/L/CVaR: {dq.get('test_global_nrmse'):.4f}% / "
                f"{dq.get('test_late_nrmse'):.4f}% / {dq.get('test_late_cvar'):.4f}%",
                f"  process exit code: {dq.get('exit_code')} "
                "(0 ok | 20 contract not met | 21 met on validation, not on the held-out test | 25 another "
                "qualification gate failed | 22 field alignment failed)",
                "",
            ]
        )
        if not dq.get("deployment_allowed", True) and field_profile_present(cfg):
            lines.append("  *** MODEL UNQUALIFIED: any field files present are DEBUG output, tagged")
            lines.append("      __UNQUALIFIED_DEBUG_ONLY, and must not be interpreted or published. ***")
            lines.append("")
    lines.extend(
        [
            "Interpretation rule:",
            "  The accuracy contract is met only when the held-out paired test satisfies global <=%g%%,"
            % (100 * cfg.contract.global_threshold),
            "  early/middle <=%g%%, late <=%g%% and late CVaR <=%g%% at the same time (the contract of this run)."
            % (100 * cfg.contract.early_mid_threshold, 100 * cfg.contract.late_threshold,
               100 * cfg.contract.late_cvar_threshold),
        ] if cfg.runtime.deployment_gate else []
    )
    if field_profile_present(cfg):
        lines.extend([
            "  Field data have no clean truth, so field diagnostics are reported separately and never substituted",
            "  for the paired project gates.",
        ])
    (run_dir / "reports" / "FINAL_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Final report written: %s", run_dir / "reports" / "FINAL_REPORT.txt")


def arg_file_line(line: str) -> List[str]:
    """One line of an argument file (@file): shell-like words; '#' starts a comment."""
    import shlex
    return shlex.split(line, comments=True)


def pin_visible_gpu(argv: Sequence[str],
                    environ: MutableMapping[str, str]) -> Tuple[List[str], List[str]]:
    """Make every other GPU invisible to this process, before CUDA starts."""
    notes: List[str] = []
    args = list(argv)
    prior = environ.get("CUDA_VISIBLE_DEVICES")
    if prior is not None and prior.strip() != "":
        notes.append(
            "CUDA_VISIBLE_DEVICES=%s was already set in the environment; left as is. Indices are "
            "renumbered from 0 within that set, so --device cuda:0 is its first entry." % (prior,))
        return args, notes
    want: Optional[int] = None
    pos: Optional[Tuple[int, bool]] = None
    for i, tok in enumerate(args):
        val = None
        if tok == "--device" and i + 1 < len(args):
            val, pos = args[i + 1], (i + 1, True)
        elif tok.startswith("--device="):
            val, pos = tok.split("=", 1)[1], (i, False)
        if val is None:
            continue
        v = val.strip().lower()
        if v.startswith("cuda:"):
            try:
                want = int(v.split(":", 1)[1])
            except ValueError:
                want = None
        break
    if want is None and pos is not None:
        _tok = args[pos[0]] if pos[1] else args[pos[0]].split("=", 1)[1]
        if _tok.strip().lower() == "auto":
            _free = _query_free_memory_mib()
            if _free:
                _best = max(range(len(_free)), key=lambda i: _free[i])
                environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environ["CUDA_VISIBLE_DEVICES"] = str(_best)
                args[pos[0]] = ("cuda:0" if pos[1] else "--device=cuda:0")
                notes.append(
                    "--device auto resolved BEFORE CUDA started: physical GPU %d is the freest (%s), so "
                    "CUDA_VISIBLE_DEVICES=%d is pinned and --device becomes cuda:0. auto now gets the same "
                    "containment as an explicit index -- no other card exists for this process."
                    % (_best, " | ".join("GPU %d %d MiB" % (i, m)
                                         for i, m in enumerate(_free)), _best))
            else:
                notes.append(
                    "--device auto: nvidia-smi free-memory query unavailable, "
                    "so the card is chosen after CUDA starts and NOT pinned.")
        return args, notes
    if want is None or want < 0:
        return args, notes
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"] = str(want)
    if pos is not None:
        idx, separate = pos
        args[idx] = "cuda:0" if separate else "--device=cuda:0"
    notes.append(
        "PINNED TO PHYSICAL GPU %d: CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=%d "
        "are set before the CUDA runtime starts, so this process sees exactly one card and --device is "
        "rewritten to cuda:0 (the renumbered form of GPU %d). No allocation from any layer -- including "
        "a vendor compatibility layer that ignores set_device -- can reach any other card, because no "
        "other card exists for this process."
        % (want, want, want))
    return args, notes
