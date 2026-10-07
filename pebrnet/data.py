# PEBR-Net -- prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.
# MIT License, see LICENSE.
"""Paired dataset ingestion, gate axes, caches, splits and split audits.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

def count_csv_data_rows(path: Path) -> int:
    with path.open("rb") as f:
        count = sum(1 for _ in f)
    return max(0, count - 1)


def parse_numeric_headers(columns: Sequence[Any]) -> Optional[np.ndarray]:
    values: List[float] = []
    for c in columns:
        try:
            values.append(float(str(c).strip()))
        except ValueError:
            return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2 or not np.all(np.diff(arr) > 0):
        return None
    return arr


def is_ch_headers(columns: Sequence[Any]) -> bool:
    parsed: List[int] = []
    for c in columns:
        s = str(c).strip().upper()
        if not s.startswith("CH"):
            return False
        try:
            parsed.append(int(s[2:]))
        except ValueError:
            return False
    return parsed == list(range(1, len(parsed) + 1))


@dataclass
class CleanCacheInfo:
    data_path: str
    station_path: str
    metadata_path: str
    rows: int
    gates: int
    target_time: np.ndarray
    global_scale: float
    gate_scale: np.ndarray
    descriptor_path: str = ""
    descriptors: Optional[np.ndarray] = None
    province_path: str = ""
    province_id: Optional[np.ndarray] = None
    province_names: Optional[List[str]] = None
    paired_noisy_path: str = ""
    paired_split_path: str = ""
    paired_background_path: str = ""
    paired_event_path: str = ""
    measured_noise_path: str = ""
    extra_clean_path: str = ""
    survey_profile_path: str = ""
    paired_split_id: Optional[np.ndarray] = None
    paired_gate_names: Optional[List[str]] = None
    paired_group_id: Optional[np.ndarray] = None


def robust_gate_scale(clean_matrix: np.ndarray, global_scale: float) -> np.ndarray:
    """Robust per-gate amplitude scale for the equalized network domain."""
    scale = np.median(np.abs(np.asarray(clean_matrix, dtype=np.float64)), axis=0)
    floor = max(float(global_scale) * 1.0e-12, float(np.finfo(np.float64).tiny))
    scale = np.maximum(scale, floor)
    if not np.all(np.isfinite(scale)):
        raise ValueError("Per-gate scale contains non-finite values.")
    return scale


@dataclass
class NoiseCacheInfo:
    data_path: str
    metadata_path: str
    rows: int
    gates: int
    target_time: np.ndarray
    block_rows: Optional[List[int]] = None


def _read_csv_smart(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    with open(path, "rb") as _fh:
        _bom = _fh.read(2)
    _encs = ("utf-16",) if _bom in (b"\xff\xfe", b"\xfe\xff") else ()
    for enc in _encs + ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            frame = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            frame = frame.dropna(axis=0, how="all").dropna(axis=1, how="all")
            frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
            return frame.reset_index(drop=True)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Unable to read field CSV {path}: {last_error}")


def inspect_csv_structure(path: Path, max_rows_for_stats: int = 5000) -> Dict[str, Any]:
    header = pd.read_csv(path, nrows=0)
    columns = header.columns.tolist()
    if len(columns) < 2:
        raise ValueError(f"CSV has fewer than two columns: {path}")
    sample = pd.read_csv(path, nrows=max_rows_for_stats)
    value_df = sample.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values = value_df.to_numpy(dtype=np.float64, copy=False)
    first_col = pd.to_numeric(sample.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
    times = parse_numeric_headers(columns[1:])
    return {
        "path": str(path),
        "rows_total": count_csv_data_rows(path),
        "columns_total": len(columns),
        "first_column_name": str(columns[0]),
        "data_columns": len(columns) - 1,
        "header_kind": "numeric_time" if times is not None else ("CH" if is_ch_headers(columns[1:]) else "other"),
        "time_min": None if times is None else float(times.min()),
        "time_max": None if times is None else float(times.max()),
        "time_strictly_increasing": None if times is None else bool(np.all(np.diff(times) > 0)),
        "sample_rows": int(len(sample)),
        "sample_finite_values": int(np.isfinite(values).sum()),
        "sample_nan_inf_values": int(values.size - np.isfinite(values).sum()),
        "sample_min": float(np.nanmin(values)),
        "sample_max": float(np.nanmax(values)),
        "sample_mean": float(np.nanmean(values)),
        "sample_median": float(np.nanmedian(values)),
        "first_axis_min": float(np.nanmin(first_col)),
        "first_axis_max": float(np.nanmax(first_col)),
    }


def _load_or_build_descriptors(
    data_path: Path,
    cache_dir: Path,
    rows: int,
    target_time: np.ndarray,
    cfg: "Config",
    logger: logging.Logger,
    force: bool = False,
) -> Optional[np.ndarray]:
    path = cache_dir / "clean_descriptors.npy"
    try:
        if path.exists() and not force and not cfg.runtime.overwrite_cache:
            d = np.load(path)
            if d.shape == (rows, len(CURVE_DESCRIPTOR_NAMES)):
                return d
        clean = np.load(data_path, mmap_mode="r")
        out = np.empty((rows, len(CURVE_DESCRIPTOR_NAMES)), dtype=np.float64)
        chunk = 8192
        for start in range(0, rows, chunk):
            block = np.asarray(clean[start : start + chunk], dtype=np.float64)
            out[start : start + len(block)] = compute_curve_descriptors(
                block, target_time, int(cfg.data.late_start_index)
            )
        np.save(path, out)
        with np.errstate(invalid="ignore"):
            logger.info(
                "CURVE DESCRIPTORS cached (%d curves) | log-amplitude %.2f+-%.2f | late/early %.2f+-%.2f | "
                "late log-log slope %.2f+-%.2f | curves outside the physics cone (log-convexity violated on "
                ">20%% of gates): %.1f%% | curves with non-positive samples: %.1f%%",
                rows,
                float(np.nanmean(out[:, 0])), float(np.nanstd(out[:, 0])),
                float(np.nanmean(out[:, 1])), float(np.nanstd(out[:, 1])),
                float(np.nanmean(out[:, 2])), float(np.nanstd(out[:, 2])),
                100.0 * float(np.mean(out[:, 4] > 0.20)),
                100.0 * float(np.mean(out[:, 5] > 0.0)),
            )
        return out
    except Exception as exc:
        logger.warning("Curve-descriptor computation skipped: %s", exc)
        return None


_TIME_UNIT_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "μs": 1e-6}


def axis_seconds_strict(values: Any, unit: str) -> np.ndarray:
    """Convert an axis to seconds with an explicit unit; validate it."""
    if unit not in _TIME_UNIT_SCALE:
        raise ValueError("an explicit time unit (s/ms/us) is required; guessing by magnitude is forbidden "
                         "(got %r)." % (unit,))
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 2 or not np.all(np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("a gate-time axis must be finite, positive and strictly increasing with >= 2 gates. ")
    return x * _TIME_UNIT_SCALE[unit]


def gate_axis_has_origin(t: Any) -> bool:
    """True when the first gate of a gate axis is at t = 0 and every later gate is positive."""
    x = np.asarray(t, dtype=np.float64).reshape(-1)
    return bool(x.size >= 3 and x[0] == 0.0 and np.all(x[1:] > 0.0))


def validate_gate_axis(t: Any) -> np.ndarray:
    """A gate axis is finite and strictly increasing with >= 2 gates, and every gate is positive except that the
    first may be the time origin (t = 0, then with >= 3 gates).
    """
    x = np.asarray(t, dtype=np.float64).reshape(-1)
    ok = (x.size >= 2 and bool(np.all(np.isfinite(x))) and bool(np.all(np.diff(x) > 0))
          and (bool(x[0] > 0.0) or gate_axis_has_origin(x)))
    if not ok:
        raise ValueError("a gate-time axis must be finite and strictly increasing with >= 2 gates, every gate "
                         "positive except that the first may be t = 0.")
    return x


def log_time_axis(t: Any) -> np.ndarray:
    """Ln t of a gate axis. ln t is undefined at a time-origin gate; that gate is placed one logarithmic step before
    the first positive gate, 2 ln t1 - ln t2.
    """
    x = np.asarray(t, dtype=np.float64).reshape(-1)
    if gate_axis_has_origin(x):
        lt = np.empty_like(x)
        lt[1:] = np.log(x[1:])
        lt[0] = 2.0 * lt[1] - lt[2]
        return lt
    return np.log(x)


def positive_gate_times(t: Any) -> np.ndarray:
    """The gate axis with a time-origin gate replaced by exp(log_time_axis) = t1^2 / t2, for power-law envelopes
    such as (t / t_L)^-q that are infinite at t = 0.
    """
    x = np.asarray(t, dtype=np.float64).reshape(-1)
    return np.exp(log_time_axis(x)) if gate_axis_has_origin(x) else x


def decay_tau_grid(t: Any, n: int) -> np.ndarray:
    """Log-spaced relaxation times of the decay dictionary: from a third of the earliest positive gate time to three
    times the last gate time (on a positive axis the earliest gate itself, as before).
    """
    x = validate_gate_axis(t)
    return np.geomspace(float(x[x > 0.0][0]) / 3.0, float(x[-1]) * 3.0, int(n))


def set_gate_time_xscale(ax: Any, t: Any) -> None:
    """A logarithmic gate-time axis; when the first gate is the time origin, a symmetric-logarithmic axis that is
    linear up to the first positive gate, so the origin gate is drawn at 0 instead of dropped.
    """
    x = np.asarray(t, dtype=np.float64).reshape(-1)
    if gate_axis_has_origin(x):
        from matplotlib.ticker import FixedLocator, NullLocator
        ax.set_xscale("symlog", linthresh=float(x[1]), linscale=0.4)
        lo, hi = int(math.ceil(math.log10(float(x[1])))), int(math.floor(math.log10(float(x[-1]))))
        ax.xaxis.set_major_locator(FixedLocator([0.0] + [10.0 ** p for p in range(lo, hi + 1)]))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.set_xscale("log")


def gate_time_xlim(t: Any) -> Tuple[float, float]:
    """X limits of a gate-time axis drawn by set_gate_time_xscale, with a margin on both ends."""
    x = np.asarray(t, dtype=np.float64).reshape(-1)
    if gate_axis_has_origin(x):
        return -0.08 * float(x[1]), float(x[-1]) * 1.06
    return float(x[0]) * 0.92, float(x[-1]) * 1.06


def axis_digest(times_s: np.ndarray) -> str:
    """SHA-256 of an axis in seconds (float64 bytes)."""
    x = np.asarray(times_s, dtype=np.float64).reshape(-1)
    return hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()


_HEADER_TIME_RE = re.compile(
    r"^(?:(?:time|t)_?)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)(?:_?(ms|us|µs|μs|s))?$", re.I)


def parse_time_headers_seconds(headers: Sequence[Any], declared_unit: Optional[str] = None) -> np.ndarray:
    """Physical gate times from headers such as t_1.6374E-05_s, time_0.054ms or numeric labels with an external unit
    declaration.
    """
    vals: List[float] = []
    for h in headers:
        m = _HEADER_TIME_RE.fullmatch(str(h).strip())
        if m is None:
            raise ValueError("no physical gate time in header %r; supply a verified time table. "
                              % (h,))
        u = (m.group(2) or "").lower() or declared_unit
        if u not in _TIME_UNIT_SCALE:
            raise ValueError("missing time unit in header %r." % (h,))
        if m.group(2) and declared_unit and _TIME_UNIT_SCALE[u] != _TIME_UNIT_SCALE.get(declared_unit, -1.0):
            raise ValueError("header unit conflicts with the declared unit.")
        vals.append(float(m.group(1).replace("D", "e").replace("d", "e")) * _TIME_UNIT_SCALE[u])
    return axis_seconds_strict(vals, "s")


def make_logtime_plan(source_s: np.ndarray, target_s: np.ndarray) -> Dict[str, Any]:
    """Piecewise-linear-in-log(t) interpolation plan with a coverage mask."""
    src = axis_seconds_strict(source_s, "s")
    tgt = axis_seconds_strict(target_s, "s")
    tol = 32.0 * np.finfo(np.float64).eps * max(abs(src[0]), abs(src[-1]))
    valid = (tgt >= src[0] - tol) & (tgt <= src[-1] + tol)
    tc = np.clip(tgt, src[0], src[-1])
    hi = np.clip(np.searchsorted(src, tc, side="left"), 1, src.size - 1)
    lo = hi - 1
    w = np.log(tc / src[lo]) / np.log(src[hi] / src[lo])
    w = np.clip(w, 0.0, 1.0)
    return {"source_s": src, "target_s": tgt, "lo": lo, "hi": hi, "w": w, "valid": valid,
            "source_sha": axis_digest(src), "target_sha": axis_digest(tgt),
            "method": "signed_linear_in_log_time", "full_coverage": bool(valid.all())}


def apply_logtime_plan(values: np.ndarray, plan: Mapping[str, Any], uncovered_policy: str = "refuse") -> np.ndarray:
    """Apply the plan along the last axis."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim < 1 or a.shape[-1] != int(plan["source_s"].size):
        raise ValueError("values' last axis (%d) must match the source gates (%d)." % (a.shape[-1], plan["source_s"].size))
    valid = np.asarray(plan["valid"], dtype=bool)
    if not valid.all() and uncovered_policy == "refuse":
        raise ValueError("%d/%d target gates have no source coverage (uncovered gates, 1-based: %s); "
                         "policy=refuse."
                         % (int((~valid).sum()), valid.size, ",".join(str(i + 1) for i in np.flatnonzero(~valid))))
    w = np.asarray(plan["w"], dtype=np.float64)
    out = (1.0 - w) * a[..., plan["lo"]] + w * a[..., plan["hi"]]
    if not valid.all() and uncovered_policy == "nan":
        out[..., ~valid] = np.nan
    return out


def require_same_axis(data_s: np.ndarray, checkpoint_s: np.ndarray, what: str = "checkpoint", rtol: float = 2e-7) -> None:
    """Same gate count and the same physical times (float32 rounding tolerated by default; an operator-typed gate
    table may pass a looser rtol such as 1e-4).
    """
    a = np.asarray(data_s, dtype=np.float64).reshape(-1)
    b = np.asarray(checkpoint_s, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not np.allclose(a, b, rtol=float(rtol), atol=0.0):
        raise ValueError("the %s gate axis differs from the data axis (%d vs %d gates; first/last %.4g..%.4g s "
                         "vs %.4g..%.4g s). Same tensor shapes do not make two gate tables the same physics; resume/reuse "
                         "is refused."
                         % (what, b.size, a.size, (b[0] if b.size else float('nan')), (b[-1] if b.size else float('nan')),
                            a[0], a[-1]))


def time_contract_self_test() -> Dict[str, Any]:
    """Pure-numpy self-test of the contract (run by pre-flight P8)."""
    out: Dict[str, Any] = {}
    t = np.array([1.0, 2.0, 4.0]) * 1e-3
    u = np.array([1.0, 1.5, 3.0, 4.0]) * 1e-3
    p = make_logtime_plan(t, u)
    a = np.array([[1.0, -1.0, 0.0], [10.0, 5.0, 2.5]])
    b = np.array([[2.0, 1.0, -3.0], [-1.0, 2.0, 4.0]])
    ar, br = apply_logtime_plan(a, p), apply_logtime_plan(b, p)
    out["pair_additivity_residual"] = float(np.max(np.abs(apply_logtime_plan(b - a, p) - (br - ar))))
    pc = make_logtime_plan(np.array([1e-3, 4e-3]), np.array([1e-3, 2e-3, 4e-3]))
    z = apply_logtime_plan(np.array([1.0, -1.0]), pc)
    out["zero_crossing_mid"] = float(z[1])
    out["identity_max_abs"] = float(np.max(np.abs(apply_logtime_plan(a, make_logtime_plan(t, t)) - a)))
    for name, fn in (("reject_CH", lambda: parse_time_headers_seconds(["CH1", "CH2"])),
                     ("reject_unitless", lambda: parse_time_headers_seconds(["0.1", "0.2"])),
                     ("reject_decreasing", lambda: make_logtime_plan(t, np.array([1e-3, 1e-1, 2e-3]))),
                     ("reject_uncovered", lambda: apply_logtime_plan(np.array([1.0, 2.0, 3.0]), make_logtime_plan(t, np.array([1e-3, 1e-1]))))):
        try:
            fn()
            out[name] = False
        except ValueError:
            out[name] = True
    out["explicit_units_equal"] = bool(np.allclose(parse_time_headers_seconds(["t_0.001_s", "t_0.002_s"]),
                                                   parse_time_headers_seconds(["time_1ms", "time_2ms"])))
    out["pass"] = bool(out["pair_additivity_residual"] < 1e-12 and abs(out["zero_crossing_mid"]) < 1e-12
                       and out["identity_max_abs"] < 1e-12 and all(out[k] for k in
                       ("reject_CH", "reject_unitless", "reject_decreasing", "reject_uncovered", "explicit_units_equal")))
    return out


def resample_noise_to_gate_axis(mat: np.ndarray, gate_ms: np.ndarray,
                                target_time: np.ndarray,
                                logger: logging.Logger,
                                source_unit: str = "ms", target_unit: str = "s",
                                uncovered_policy: str = "refuse",
                                plan: Optional[Mapping[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Put the library's gate axis onto the training gate axis."""
    _src_s = axis_seconds_strict(gate_ms, source_unit)
    _tgt_s = axis_seconds_strict(target_time, target_unit)
    _plan = dict(plan) if plan is not None else make_logtime_plan(_src_s, _tgt_s)
    if plan is not None and (_plan["source_sha"] != axis_digest(_src_s) or _plan["target_sha"] != axis_digest(_tgt_s)):
        raise ValueError("the shared plan was built for different axes.")
    _valid = np.asarray(_plan["valid"], dtype=bool)
    _overlap = float(np.mean(_valid))
    if _overlap <= 0.0:
        raise ValueError(
            "Measured-noise library has ZERO physical time overlap with the training gate axis "
            "(library %.4g..%.4g s vs axis %.4g..%.4g s)."
            % (float(_src_s[0]), float(_src_s[-1]), float(_tgt_s[0]), float(_tgt_s[-1])))
    _out = apply_logtime_plan(np.asarray(mat, dtype=np.float64), _plan, uncovered_policy=uncovered_policy)
    _report = {"mode": "physical-time", "method": _plan["method"], "source_unit": source_unit, "target_unit": target_unit,
               "overlap_fraction": _overlap, "held_edge_fraction": float(1.0 - _overlap) if uncovered_policy == "hold_edge" else 0.0,
               "uncovered_gates_1based": [int(i) + 1 for i in np.flatnonzero(~_valid)], "uncovered_policy": uncovered_policy,
               "source_range_s": [float(_src_s[0]), float(_src_s[-1])], "target_gates": int(_tgt_s.size),
               "source_sha": _plan["source_sha"], "target_sha": _plan["target_sha"], "full_coverage": bool(_valid.all())}
    logger.info("Noise gate axis: signed-linear in log(t), units %s->%s | %.1f%% of target "
                "gates inside the source support; uncovered gates %s -> policy %s.",
                source_unit, target_unit, 100.0 * _overlap, _report["uncovered_gates_1based"] or "none", uncovered_policy)
    return _out, _report
    tgt = np.asarray(target_time, dtype=np.float64)
    src = np.asarray(gate_ms, dtype=np.float64)
    best = None
    for f, label in ((1.0, "ms"), (1.0e3, "s->ms"), (1.0e-3, "us->ms")):
        t = tgt * f
        ov = float(np.mean((t >= src.min()) & (t <= src.max())))
        if best is None or ov > best[0]:
            best = (ov, f, label, t)
    overlap, factor, label, t_scaled = best
    mode = "physical-time"
    if overlap <= 0.0:
        lo0, hi0 = float(src.min()), float(src.max())
        raise ValueError(
            "Measured-noise library has ZERO physical time overlap with the training gate axis (library "
            "%.4g..%.4g ms). The injection arm is disabled rather than resampled by normalised position; "
            "paired observations and residual replay still supply this survey's measured noise."
            % (lo0, hi0))
    lo, hi = float(src.min()), float(src.max())
    tc = np.clip(t_scaled, lo, hi)
    held = float(np.mean((t_scaled < lo) | (t_scaled > hi)))
    ls, lt = np.log(src), np.log(np.maximum(tc, 1e-300))
    out = np.empty((mat.shape[0], tgt.size), dtype=np.float64)
    eps = 1e-30
    for i in range(mat.shape[0]):
        row = mat[i]
        lm = np.interp(lt, ls, np.log(np.abs(row) + eps))
        sg = np.sign(row)[np.clip(np.searchsorted(src, tc), 0, src.size - 1)]
        sg[sg == 0.0] = 1.0
        out[i] = np.exp(lm) * sg
    report = {"mode": mode, "unit_factor": factor, "unit_label": label,
              "overlap_fraction": overlap, "held_edge_fraction": held,
              "src_range_ms": [lo, hi], "target_gates": int(tgt.size)}
    logger.info(
        "Noise gate axis: %s (%s, factor %.3g) | %.1f%% of target gates lie inside the "
        "measured range; %.1f%% take the HELD EDGE ENVELOPE -- those gates carry extrapolated "
        "morphology, not measured truth, and the training objective treats them as augmentation only.",
        mode, label, factor, 100.0 * overlap, 100.0 * held)
    return out, report


def build_extra_clean_cache(paired_dir, target_time, cache_dir, cfg, logger):
    """Optional extra CLEAN-only dataset: <paired_dir>/clean.csv."""
    src = Path(paired_dir) / "clean.csv"
    if not src.exists():
        return None
    if not bool(getattr(cfg.data, "extra_clean_enabled", True)):
        logger.info("extra clean-only dataset DISABLED by configuration (extra_clean_enabled=False).")
        return None
    out = Path(cache_dir) / "extra_clean.npy"
    meta_p = Path(cache_dir) / "extra_clean.json"
    _pol = str(getattr(cfg.data, "uncovered_gate_policy", "refuse"))
    sig = "v672:%s:%s:%s" % (sha256_file(src)[:24], axis_digest(np.asarray(target_time, dtype=np.float64)), _pol)
    if out.exists() and meta_p.exists():
        try:
            _m = json.loads(meta_p.read_text(encoding="utf-8"))
            if _m.get("sig") == sig and _m.get("full_coverage", False) is not None:
                logger.info("Extra clean cache hit: %d rows (axis-bound signature).", int(_m.get("rows", 0)))
                return {"path": str(out), "rows": int(_m.get("rows", 0)), "full_coverage": bool(_m.get("full_coverage", False))}
        except Exception:
            pass
    G = int(np.asarray(target_time).size)
    raw = np.genfromtxt(src, delimiter=",", dtype=np.float64)
    hdr = None
    with open(src, "r", encoding="utf-8-sig") as _f:
        hdr = _f.readline().strip().split(",")
    gate_ms = None
    _full_cov = True
    if hdr and any(h.startswith("t_") for h in hdr):
        tcols = [i for i, h in enumerate(hdr) if h.startswith("t_")]
        gate_s = parse_time_headers_seconds([hdr[i] for i in tcols], declared_unit="s")
        X = raw[1:, tcols] if raw.ndim == 2 else raw[None, tcols]
        gate_ms = gate_s * 1000.0
    elif raw.ndim == 2 and raw.shape[1] >= 3 and np.all(np.diff(raw[0, 1:]) > 0) and \
            np.isnan(raw[1:, 0]).sum() == 0 and not np.allclose(
                raw[0, 1], round(raw[0, 1])):
        gate_ms = raw[0, 1:].astype(np.float64)
        X = raw[1:, 1:]
    else:
        X = raw if raw.ndim == 2 else raw[None, :]
        if X.shape[1] == G + 2 and np.isnan(X[0, :2]).any():
            X = X[:, 2:]
    X = np.asarray(X, dtype=np.float64)
    if gate_ms is not None:
        _src_s = np.asarray(gate_ms, dtype=np.float64) * 1e-3
        _tgt_s = np.asarray(target_time, dtype=np.float64)
        if _src_s.size == _tgt_s.size and np.allclose(_src_s, _tgt_s, rtol=2e-7, atol=0.0):
            pass
        else:
            try:
                Xr, _rep = resample_noise_to_gate_axis(X, gate_ms, target_time, logger, source_unit="ms",
                                                        target_unit="s", uncovered_policy=_pol)
            except ValueError as _e:
                logger.warning("extra clean-only dataset DISABLED for this run: %s", _e)
                return None
            X = Xr
            _full_cov = bool(_rep.get("full_coverage", False))
            if not _full_cov:
                logger.warning("extra clean rows carry HELD-EDGE values on gates %s (policy %s): augmentation "
                               "only; excluded from the physics dictionary.", _rep.get("uncovered_gates_1based"), _pol)
    if X.shape[1] != G:
        logger.warning("clean.csv gate count %d != %d; skipped.",
                       X.shape[1], G)
        return None
    ok = np.all(np.isfinite(X), axis=1) & (np.nanmax(X, axis=1) > 0)
    X = X[ok].astype(np.float32)
    if X.shape[0] == 0:
        logger.warning("clean.csv contained no usable rows; skipped.")
        return None
    np.save(out, X)
    meta_p.write_text(json.dumps({"sig": sig, "rows": int(X.shape[0]), "gates": G, "full_coverage": bool(_full_cov),
                                  "target_axis_sha": axis_digest(np.asarray(target_time, dtype=np.float64)),
                                  "label_origin": "extra clean.csv (pseudo-labels unless certified otherwise)"}),
                      encoding="utf-8")
    logger.info("Extra clean dataset: %d rows x %d gates (online-injection pool; full coverage %s).",
                X.shape[0], G, _full_cov)
    return {"path": str(out), "rows": int(X.shape[0]), "full_coverage": bool(_full_cov)}


PAIRED_SPLIT_NAMES = ("train", "val", "test")


def paired_split_files(paired_dir: Path) -> Optional[Dict[str, Tuple[Path, Path]]]:
    """The six paired files, or None when the directory is not a paired dataset."""
    if not paired_dir or not paired_dir.is_dir():
        return None
    found: Dict[str, Tuple[Path, Path]] = {}
    missing: List[str] = []
    for name in PAIRED_SPLIT_NAMES:
        c = paired_dir / ("%s_clean.csv" % name)
        n = paired_dir / ("%s_noisy.csv" % name)
        if c.is_file() and n.is_file():
            found[name] = (c, n)
        else:
            if not c.is_file():
                missing.append(c.name)
            if not n.is_file():
                missing.append(n.name)
    if not found:
        _stray = sorted(
            p.name for p in paired_dir.glob("*.csv")
            if p.name.endswith("_clean.csv") or p.name.endswith("_noisy.csv")
        )
        if _stray:
            raise FileNotFoundError(
                "Paired dataset directory %s holds %s but no COMPLETE split: %s is missing. The paired path "
                "needs both members of at least one split."
                % (paired_dir, ", ".join(_stray), ", ".join(sorted(set(missing))))
            )
        return None
    if missing:
        raise FileNotFoundError(
            "Paired dataset directory %s is incomplete: missing %s. The paired path needs all six files "
            "(train/val/test x clean/noisy)."
            % (paired_dir, ", ".join(missing))
        )
    return found


_GATE_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")


def parse_gate_time_headers(columns: Sequence[Any]) -> Optional[np.ndarray]:
    """Gate times from headers such as ``t_1.6374E-05``, ``t1.64e-5``, ``1.6374E-05`` or ``time_0.054ms``."""
    values: List[float] = []
    if all(re.fullmatch(r"\s*CH\s*\d+\s*", str(c), flags=re.I) for c in columns):
        return None
    for c in columns:
        s = str(c).strip()
        m = _GATE_NUMBER_RE.search(s)
        if m is None:
            return None
        try:
            v = float(m.group(0).replace("d", "e").replace("D", "E"))
        except ValueError:
            return None
        um = re.search(r"(ms|us|µs|μs|s)\s*$", s[m.end():], flags=re.I)
        if um is not None:
            v *= _TIME_UNIT_SCALE[um.group(1).lower()]
        values.append(v)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2 or not np.all(np.isfinite(arr)) or not np.all(np.diff(arr) > 0):
        return None
    return arr


_PAIRED_ID_HINTS = ("sample_id", "sampleid", "sample", "survey", "area", "line")
_PAIRED_DEPTH_HINTS = ("depth", "station", "elev", "z_m")


def _detect_paired_key_columns(frame: "pd.DataFrame") -> Tuple[int, int]:
    cols = [str(c).strip().lower() for c in frame.columns]
    if len(cols) < 3:
        raise ValueError(
            "A paired file needs at least sample_id, depth and one gate column; got %d."
            % len(cols)
        )
    sid_idx: Optional[int] = None
    dep_idx: Optional[int] = None
    for i, name in enumerate(cols[: min(6, len(cols))]):
        if sid_idx is None and any(h in name for h in _PAIRED_ID_HINTS):
            sid_idx = i
            continue
        if dep_idx is None and any(h in name for h in _PAIRED_DEPTH_HINTS):
            dep_idx = i
    if sid_idx is None:
        sid_idx = 0
    if dep_idx is None:
        dep_idx = 1 if sid_idx != 1 else 0
    if sid_idx == dep_idx:
        raise ValueError("sample_id and depth resolved to the same column index %d." % sid_idx)
    for idx, label in ((sid_idx, "sample_id"), (dep_idx, "depth")):
        frac = float(pd.to_numeric(frame.iloc[:, idx], errors="coerce").notna().mean())
        if frac < 0.999:
            raise ValueError(
                "Paired %s column %r is only %.1f%% numeric; the pairing key must be exact."
                % (label, str(frame.columns[idx]), 100.0 * frac)
            )
    return int(sid_idx), int(dep_idx)


def read_paired_csv(path: Path, logger: logging.Logger) -> Dict[str, Any]:
    """One paired file -> (sample_id, depth, matrix, gate header names)."""
    frame = _read_csv_smart(path)
    for _c in list(frame.columns[:2]):
        if re.fullmatch(r"(unnamed:\s*\d+|index|)", str(_c).strip().lower()):
            _v = pd.to_numeric(frame[_c], errors="coerce").to_numpy(dtype=np.float64)
            if _v.size and (np.array_equal(_v, np.arange(_v.size))
                               or np.array_equal(_v, np.arange(1, _v.size + 1))):
                frame = frame.drop(columns=[_c])
                logger.info("%s: row-counter column %r ignored.", path.name, str(_c))
    sid_idx, dep_idx = _detect_paired_key_columns(frame)
    gate_pos = [i for i in range(frame.shape[1]) if i not in (sid_idx, dep_idx)]
    if not gate_pos:
        raise ValueError("No gate columns in %s" % path)
    gate_names = [str(frame.columns[i]).strip() for i in gate_pos]
    sid = pd.to_numeric(frame.iloc[:, sid_idx], errors="coerce").to_numpy(dtype=np.float64)
    dep = pd.to_numeric(frame.iloc[:, dep_idx], errors="coerce").to_numpy(dtype=np.float64)
    mat = frame.iloc[:, gate_pos].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(sid) & np.isfinite(dep) & np.isfinite(mat).all(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        logger.warning(
            "%s: dropped %d of %d rows with non-finite entries.",
            path.name, dropped, int(keep.size),
        )
    sid, dep, mat = sid[keep], dep[keep], mat[keep]
    if sid.size == 0:
        raise ValueError("No usable rows in %s" % path)
    return {
        "sample_id": np.round(sid).astype(np.int64),
        "depth": dep,
        "matrix": mat,
        "gate_names": gate_names,
        "rows_dropped": dropped,
        "path": str(path),
    }


def _paired_join(clean: Dict[str, Any], noisy: Dict[str, Any], split: str,
                 logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    def _keys(d: Dict[str, Any]) -> np.ndarray:
        return np.stack([d["sample_id"].astype(np.float64),
                         np.round(d["depth"] / 1.0e-6) * 1.0e-6], axis=1)

    kc, kn = _keys(clean), _keys(noisy)
    tc = [tuple(r) for r in kc]
    tn = [tuple(r) for r in kn]
    for tbl, label in ((tc, "clean"), (tn, "noisy")):
        if len(set(tbl)) != len(tbl):
            from collections import Counter as _Counter
            dup = [k for k, v in _Counter(tbl).items() if v > 1][:5]
            raise ValueError(
                "%s_%s.csv has duplicate (sample_id, depth) keys, e.g. %s. The pairing would be ambiguous." % (split, label, dup)
            )
    pos_n = {k: i for i, k in enumerate(tn)}
    common = [(i, pos_n[k]) for i, k in enumerate(tc) if k in pos_n]
    if not common:
        raise ValueError(
            "%s: the clean and noisy files share no (sample_id, depth) key at all."
            % (split,)
        )
    lost_c = len(tc) - len(common)
    lost_n = len(tn) - len(common)
    worst = max(lost_c / max(len(tc), 1), lost_n / max(len(tn), 1))
    if worst > 0.01:
        raise ValueError(
            "%s: only %d of %d clean and %d noisy rows pair up (%.2f%% unmatched). A partly mis-paired set "
            "trains the network against the wrong station, which no metric in this program would report as "
            "an error."
            % (split, len(common), len(tc), len(tn), 100.0 * worst)
        )
    if lost_c or lost_n:
        logger.warning(
            "%s: %d clean-only and %d noisy-only rows dropped (%.3f%%).",
            split, lost_c, lost_n, 100.0 * worst,
        )
    ci = np.asarray([c for c, _ in common], dtype=np.int64)
    ni = np.asarray([n for _, n in common], dtype=np.int64)
    order = np.lexsort((clean["depth"][ci], clean["sample_id"][ci]))
    return ci[order], ni[order]


def _choose_background_window(matrix: np.ndarray, province_id: np.ndarray,
                              late_start: int, logger: logging.Logger,
                              requested: int = 0) -> int:
    if int(requested) > 0:
        return int(requested) | 1
    uniq = np.unique(province_id)
    sizes = [int((province_id == p).sum()) for p in uniq]
    n_min = max(min(sizes) if sizes else 0, 0)
    cap = max(5, (n_min // 2) | 1)
    ladder = [w for w in (9, 13, 17, 21, 27, 33, 41, 51, 65) if w <= cap]
    if not ladder:
        return max(5, cap)
    probe_rows: List[np.ndarray] = []
    taken = 0
    for p in uniq:
        r = np.nonzero(province_id == p)[0]
        probe_rows.append(r)
        taken += r.size
        if taken >= 20000:
            break
    sel = np.concatenate(probe_rows)
    m_probe = matrix[sel]
    p_probe = province_id[sel]
    ls = int(late_start)
    scores: List[Tuple[int, float]] = []
    for w in ladder:
        bg = _lateral_background(m_probe, p_probe, w)
        rel = np.abs(m_probe[:, ls:] - bg[:, ls:]) / np.maximum(np.abs(bg[:, ls:]), 1.0e-30)
        scores.append((w, float(np.percentile(rel, 99.9)) if rel.size else 0.0))
    best = max(s for _, s in scores)
    if best <= 0.0:
        chosen = ladder[len(ladder) // 2]
    else:
        chosen = next(w for w, s in scores if s >= 0.90 * best)
    logger.info(
        "Background window chosen from the data: %d stations (ladder %s -> recovered late "
        "departure p99.9 %s; knee at 90%% of %.1f%%). A window that is not several times the body width "
        "would absorb the anomaly into the background and silently disable the paired anomaly "
        "supervision.",
        chosen, [w for w, _ in scores],
        ["%.1f%%" % (100.0 * s) for _, s in scores], 100.0 * best)
    return int(chosen)


def _lateral_background(matrix: np.ndarray, province_id: np.ndarray,
                        window: int) -> np.ndarray:
    out = np.empty_like(matrix)
    half = max(int(window) // 2, 1)
    for p in np.unique(province_id):
        rows = np.nonzero(province_id == p)[0]
        block = matrix[rows]
        n = block.shape[0]
        if n <= 2:
            out[rows] = block
            continue
        w = min(2 * half + 1, n if n % 2 == 1 else n - 1)
        h = w // 2
        _pos = bool(np.all(block > 0)) and n >= 3
        _tr = None
        if _pos:
            _lb = np.log(block)
            _k = max(1, n // 3)
            _sl = ((np.median(_lb[-_k:], axis=0) - np.median(_lb[:_k], axis=0))
                      / max(float(n - _k), 1.0))
            _tr = _sl[None, :] * np.arange(n, dtype=np.float64)[:, None]
            _src = _lb - _tr
        else:
            _src = block
        pad = np.concatenate([_src[h:0:-1], _src, _src[-2:-2 - h:-1]], axis=0)
        if pad.shape[0] != n + 2 * h:
            pad = np.concatenate(
                [np.repeat(_src[:1], h, axis=0), _src, np.repeat(_src[-1:], h, axis=0)],
                axis=0,
            )
        step = max(1, int(4_000_000 // max(matrix.shape[1] * w, 1)))
        for a0 in range(0, n, step):
            b0 = min(a0 + step, n)
            win = np.lib.stride_tricks.sliding_window_view(
                pad[a0 : b0 + 2 * h], w, axis=0)
            _med = np.median(win, axis=-1)
            out[rows[a0:b0]] = (np.exp(_med + _tr[a0:b0]) if _tr is not None else _med)
    return out


def _apply_gate_count_to_cfg(cfg: "Config", gates: int, logger: logging.Logger) -> None:
    gates = int(gates)
    if int(cfg.data.target_gates) != gates:
        old_g = int(cfg.data.target_gates)
        old_ls = int(cfg.data.late_start_index)
        new_ls = int(round(old_ls * gates / max(old_g, 1)))
        new_ls = max(1, min(gates - 2, new_ls))
        logger.warning(
            "The paired dataset has %d gates, not %d. target_gates %d -> %d and late_start_index "
            "%d -> %d (late window = gates %d..%d).",
            gates, old_g, old_g, gates, old_ls, new_ls, new_ls + 1, gates)
        cfg.data.target_gates = gates
        cfg.data.late_start_index = new_ls


def group_paired_profiles(blocks: Sequence[Mapping[str, Any]], cfg: "Config",
                              logger: logging.Logger) -> Dict[str, Any]:
    """Groups of near-identical clean profiles across the splits of a paired dataset."""
    policy = str(getattr(cfg.data, "paired_split_policy", "files")).lower()
    if policy not in ("grouped", "files"):
        raise ValueError("--paired_split_policy must be 'grouped' or 'files', got %r" % policy)
    tol = float(getattr(cfg.data, "paired_copy_rel_tol", 2.0e-3))
    profs: List[Tuple[int, int, np.ndarray]] = []
    for bi, b in enumerate(blocks):
        sids = np.asarray(b["sample_id"])
        dep = np.asarray(b["depth"], dtype=np.float64)
        if sids.size and bool(np.all(np.diff(sids) >= 0)):
            u, start, cnt = np.unique(sids, return_index=True, return_counts=True)
            parts = [(int(s), np.arange(a, a + c)) for s, a, c in zip(u.tolist(), start.tolist(), cnt.tolist())]
        else:
            parts = [(int(s), np.nonzero(sids == s)[0]) for s in np.unique(sids).tolist()]
        for sid, sel in parts:
            profs.append((bi, sid, sel[np.argsort(dep[sel], kind="stable")]))
    n = len(profs)

    def _y(i: int) -> np.ndarray:
        bi, _s, sel = profs[i]
        return np.asarray(blocks[bi]["clean"], dtype=np.float64)[sel]

    def _d(i: int) -> np.ndarray:
        bi, _s, sel = profs[i]
        return np.asarray(blocks[bi]["depth"], dtype=np.float64)[sel]

    key = np.asarray([float(np.mean(np.log10(np.abs(_y(i)) + 1e-300))) for i in range(n)])
    order = np.argsort(key, kind="mergesort")
    parent = np.arange(n)

    def _root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    max_in, min_out, compared = 0.0, float("inf"), 0
    if tol > 0.0:
        for a in range(n):
            ia = int(order[a])
            ya = None
            for b in range(a + 1, n):
                ib = int(order[b])
                if key[ib] - key[ia] > tol:
                    break
                if profs[ia][2].size != profs[ib][2].size or not np.allclose(_d(ia), _d(ib), rtol=0.0, atol=1e-6):
                    continue
                ya = _y(ia) if ya is None else ya
                yb = _y(ib)
                scale = max(float(np.abs(ya).max()), float(np.abs(yb).max()), 1e-300)
                d = float(np.max(np.abs(ya - yb) / np.maximum(np.maximum(np.abs(ya), np.abs(yb)), 1e-9 * scale)))
                compared += 1
                if d <= tol:
                    parent[_root(ia)] = _root(ib)
                    max_in = max(max_in, d)
                else:
                    min_out = min(min_out, d)
    roots = np.asarray([_root(i) for i in range(n)], dtype=np.int64)
    _u, gid = np.unique(roots, return_inverse=True)
    src = np.asarray([int(blocks[bi]["split"]) for bi, _s, _sel in profs], dtype=np.int64)
    sid_arr = np.asarray([s for _b, s, _sel in profs], dtype=np.int64)
    eff = src.copy()
    members: Dict[int, List[int]] = {}
    for i, g in enumerate(gid.tolist()):
        members.setdefault(int(g), []).append(i)
    multi = [m for m in members.values() if len(m) > 1]
    straddling = sum(1 for m in multi if len(set(src[m].tolist())) > 1)
    if policy == "grouped":
        for m in multi:
            first = min(m, key=lambda i: (int(sid_arr[i]), int(src[i])))
            eff[m] = src[first]
    moves: Dict[str, int] = {}
    moved: List[List[Any]] = []
    for i in np.nonzero(eff != src)[0].tolist():
        k = "%s->%s" % (PAIRED_SPLIT_NAMES[int(src[i])], PAIRED_SPLIT_NAMES[int(eff[i])])
        moves[k] = moves.get(k, 0) + 1
        moved.append([int(sid_arr[i]), PAIRED_SPLIT_NAMES[int(src[i])], PAIRED_SPLIT_NAMES[int(eff[i])]])
    per_split = {PAIRED_SPLIT_NAMES[int(s)]: int((eff == s).sum()) for s in sorted(set(src.tolist()))}
    summary = {
        "policy": policy, "copy_rel_tol": tol, "profiles": int(n), "pairs_compared": int(compared),
        "groups_with_copies": len(multi), "profiles_in_groups": int(sum(len(m) for m in multi)),
        "groups_in_more_than_one_split_of_the_files": int(straddling),
        "max_rel_diff_within_groups": float(max_in),
        "min_rel_diff_between_compared_non_copies": (float(min_out) if math.isfinite(min_out) else None),
        "moves": moves, "moved_profiles": moved, "profiles_per_split": per_split,
    }
    logger.info(
        "PAIRED PROFILE COPIES | %d profiles | %d groups of near-identical clean profiles hold %d profiles "
        "(largest relative difference inside a group %.2e; smallest between compared profiles that are not copies %s; "
        "tolerance %.1e) | %d groups lie in more than one split of the files | policy %s: %s | profiles %s",
        n, len(multi), summary["profiles_in_groups"], max_in,
        ("%.2e" % min_out) if math.isfinite(min_out) else "n/a", tol, straddling, policy,
        (", ".join("%s %d" % kv for kv in sorted(moves.items())) if moves else "no profile moved"),
        " / ".join("%s %d" % kv for kv in per_split.items()),
    )
    if policy == "files" and straddling:
        logger.warning(
            "%d groups of copies lie in more than one split: held-out profiles have copies in training, so "
            "the held-out scores partly measure recall of training profiles. --paired_split_policy grouped puts every "
            "group into one split.", straddling)
    if multi and math.isfinite(min_out) and min_out < 10.0 * max(max_in, 1e-12):
        logger.warning(
            "The copy tolerance does not separate copies from distinct profiles by a factor of 10 (largest "
            "difference inside a group %.2e, smallest outside %.2e); check --paired_copy_rel_tol.", max_in, min_out)
    of = {(int(profs[i][0]), int(profs[i][1])): (int(gid[i]), int(eff[i])) for i in range(n)}
    return {"of": of, "summary": summary}


def build_paired_cache(
    paired_dir: Path,
    cache_dir: Path,
    cfg: "Config",
    logger: logging.Logger,
) -> CleanCacheInfo:
    """Ingest the six paired files into the single cache contract the rest of the program speaks, and additionally
    carry the measured observation, the split the data itself declares.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = paired_split_files(paired_dir)
    if not files:
        raise FileNotFoundError("Not a paired dataset directory: %s" % paired_dir)

    data_path = cache_dir / "paired_clean.npy"
    noisy_path = cache_dir / "paired_noisy.npy"
    station_path = cache_dir / "paired_station.npy"
    province_path = cache_dir / "paired_province.npy"
    split_path = cache_dir / "paired_split.npy"
    bg_path = cache_dir / "paired_background.npy"
    event_path = cache_dir / "paired_event.npy"
    metadata_path = cache_dir / "paired_metadata.json"
    group_path = cache_dir / "paired_group.npy"

    signature = {
        "dir": str(paired_dir),
        "files": {k: [str(v[0].name), str(v[1].name)] for k, v in files.items()},
        "sizes": {k: [int(v[0].stat().st_size), int(v[1].stat().st_size)] for k, v in files.items()},
        "mtimes": {k: [int(v[0].stat().st_mtime), int(v[1].stat().st_mtime)] for k, v in files.items()},
        "bg_window_requested": int(getattr(cfg.data, "paired_background_window", 0) or 0),
        "event_threshold": float(getattr(cfg.data, "paired_event_threshold", 0.15)),
        "unified_axis": {"on": bool(getattr(cfg.data, "unify_axis_to_measured", False)),
                         "gates": int(getattr(cfg.data, "unified_gates", 0) or 0),
                         "noise_library_dir": str(getattr(cfg.paths, "noise_library_dir", "")),
                         "field_gate_table": str(getattr(cfg.paths, "field_gate_table", ""))},
        "dtype": cfg.data.cache_dtype,
        "split_policy": [str(getattr(cfg.data, "paired_split_policy", "files")),
                         float(getattr(cfg.data, "paired_copy_rel_tol", 2.0e-3))],
        "schema": 4,
    }
    _all = [data_path, noisy_path, station_path, province_path, split_path, bg_path, event_path, group_path]
    if metadata_path.exists() and all(p.exists() for p in _all) and not cfg.runtime.overwrite_cache:
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("signature") == signature:
            logger.info("Using existing paired cache: %s", data_path)
            _apply_gate_count_to_cfg(cfg, int(meta["target_gates"]), logger)
            desc = _load_or_build_descriptors(
                data_path, cache_dir, int(meta["rows"]),
                np.asarray(meta["target_time"], dtype=np.float64), cfg, logger,
            )
            return CleanCacheInfo(
                data_path=str(data_path), station_path=str(station_path),
                metadata_path=str(metadata_path), rows=int(meta["rows"]),
                gates=int(meta["target_gates"]),
                target_time=np.asarray(meta["target_time"], dtype=np.float64),
                global_scale=float(meta["global_scale"]),
                gate_scale=np.asarray(meta["gate_scale"], dtype=np.float64),
                descriptor_path=str(cache_dir / "clean_descriptors.npy"), descriptors=desc,
                province_path=str(province_path), province_id=np.load(province_path),
                province_names=list(meta["province_names"]),
                paired_noisy_path=str(noisy_path), paired_split_path=str(split_path),
                paired_background_path=str(bg_path), paired_event_path=str(event_path),
                paired_split_id=np.load(split_path),
                paired_gate_names=list(meta["gate_names"]),
                paired_group_id=np.load(group_path),
            )

    ref_gate_names: Optional[List[str]] = None
    blocks: List[Dict[str, Any]] = []
    for si, name in enumerate(PAIRED_SPLIT_NAMES):
        if name not in files:
            continue
        c_path, n_path = files[name]
        c = read_paired_csv(c_path, logger)
        n = read_paired_csv(n_path, logger)
        if ref_gate_names is None:
            ref_gate_names = list(c["gate_names"])
        for d, lbl in ((c, "clean"), (n, "noisy")):
            if list(d["gate_names"]) != ref_gate_names:
                raise ValueError(
                    "Gate columns of %s_%s.csv differ from %s_clean.csv: %d vs %d columns, first mismatch at index "
                    "%d."
                    % (name, lbl, PAIRED_SPLIT_NAMES[0], len(d["gate_names"]),
                       len(ref_gate_names),
                       next((i for i, (a, b) in enumerate(
                           zip(d["gate_names"], ref_gate_names)) if a != b), -1))
                )
        ci, ni = _paired_join(c, n, name, logger)
        blocks.append({
            "split": si,
            "name": name,
            "sample_id": c["sample_id"][ci],
            "depth": c["depth"][ci],
            "clean": c["matrix"][ci],
            "noisy": n["matrix"][ni],
        })
        logger.info(
            "%-5s paired: %6d stations | %2d work areas | clean %s | noisy %s",
            name, int(ci.size), int(np.unique(c["sample_id"][ci]).size),
            c_path.name, n_path.name,
        )
    if not blocks:
        raise ValueError("No paired split could be read from %s" % paired_dir)
    assert ref_gate_names is not None
    gates = len(ref_gate_names)

    parsed = parse_gate_time_headers(ref_gate_names)
    if parsed is None:
        target_time = np.geomspace(1.0, float(gates), gates)
        logger.warning(
            "Gate headers are not a strictly increasing time sequence "
            "(first three: %s). This is what a spreadsheet produces when the "
            "scientific-notation exponent is truncated. Falling back to COLUMN ORDER "
            "as the gate axis, which is physically correct; only absolute time labels "
            "are unavailable.",
            ref_gate_names[:3],
        )
    else:
        target_time = parsed
        logger.info(
            "Gate axis from headers: %d gates, %.6g .. %.6g (%s .. %s)",
            gates, float(target_time[0]), float(target_time[-1]),
            ref_gate_names[0], ref_gate_names[-1],
        )

    if bool(getattr(cfg.data, "unify_axis_to_measured", False)):
        _old = np.asarray(target_time, dtype=np.float64).copy()
        _ax_g = unified_measured_axis(cfg, gates, logger)
        _N = int(getattr(cfg.data, "unified_gates", 0) or 0)
        if _N <= 0:
            _N = gates
            try:
                _xc = Path(paired_dir) / "clean.csv"
                if _xc.exists():
                    with _xc.open("r", encoding="utf-8-sig") as _f:
                        _hdr = _f.readline().strip().split(",")
                    _nx = sum(1 for tok in _hdr[1:] if tok.strip())
                    _N = max(_N, int(_nx))
            except Exception:
                pass
        if _N != gates:
            _axN = unified_measured_axis(cfg, _N, logger)
            _plan = make_logtime_plan(_ax_g, _axN)
            for b in blocks:
                b["clean"] = apply_logtime_plan(np.asarray(b["clean"], dtype=np.float64), _plan, uncovered_policy="refuse")
                b["noisy"] = apply_logtime_plan(np.asarray(b["noisy"], dtype=np.float64), _plan, uncovered_policy="refuse")
            target_time = _axN
            gates = int(_N)
        else:
            target_time = _ax_g
        logger.warning("UNIFIED AXIS IN FORCE | paired columns relabelled %.4g..%.4g s -> %.4g..%.4g s "
                       "and resampled to %d gates (file had %d) by operator instruction; data-driven, no physical-time "
                       "assertion.",
                       float(_old[0]), float(_old[-1]), float(target_time[0]), float(target_time[-1]), gates, len(ref_gate_names))
    _apply_gate_count_to_cfg(cfg, gates, logger)
    _g = group_paired_profiles(blocks, cfg, logger)

    total_rows = int(sum(b["clean"].shape[0] for b in blocks))
    clean_mm = np.lib.format.open_memmap(
        data_path, mode="w+", dtype=np.dtype(cfg.data.cache_dtype), shape=(total_rows, gates))
    noisy_mm = np.lib.format.open_memmap(
        noisy_path, mode="w+", dtype=np.dtype(cfg.data.cache_dtype), shape=(total_rows, gates))
    station_mm = np.lib.format.open_memmap(
        station_path, mode="w+", dtype=np.float64, shape=(total_rows,))
    province_mm = np.lib.format.open_memmap(
        province_path, mode="w+", dtype=np.int32, shape=(total_rows,))
    split_mm = np.lib.format.open_memmap(
        split_path, mode="w+", dtype=np.int8, shape=(total_rows,))
    group_mm = np.lib.format.open_memmap(
        group_path, mode="w+", dtype=np.int32, shape=(total_rows,))

    province_names: List[str] = []
    offset = 0
    for _bi, b in enumerate(blocks):
        for sid in np.unique(b["sample_id"]):
            sel = np.nonzero(b["sample_id"] == sid)[0]
            sel = sel[np.argsort(b["depth"][sel], kind="stable")]
            n = sel.size
            pid = len(province_names)
            _gid, _sp = _g["of"][(_bi, int(sid))]
            province_names.append("%s#sample_%d" % (PAIRED_SPLIT_NAMES[_sp], int(sid)))
            clean_mm[offset:offset + n] = b["clean"][sel].astype(cfg.data.cache_dtype, copy=False)
            noisy_mm[offset:offset + n] = b["noisy"][sel].astype(cfg.data.cache_dtype, copy=False)
            station_mm[offset:offset + n] = b["depth"][sel]
            province_mm[offset:offset + n] = pid
            split_mm[offset:offset + n] = _sp
            group_mm[offset:offset + n] = _gid
            offset += n
    if offset != total_rows:
        raise RuntimeError("Paired cache row mismatch: wrote %d, expected %d" % (offset, total_rows))
    for mm in (clean_mm, noisy_mm, station_mm, province_mm, split_mm, group_mm):
        mm.flush()
    del clean_mm, noisy_mm, station_mm, province_mm, split_mm, group_mm

    clean_ro = np.load(data_path, mmap_mode="r")
    noisy_ro = np.load(noisy_path, mmap_mode="r")
    province_id = np.load(province_path)
    split_id = np.load(split_path)
    group_id = np.load(group_path)

    train_rows = np.nonzero(split_id == 0)[0]
    if train_rows.size == 0:
        train_rows = np.arange(total_rows)
    sample_count = min(train_rows.size, int(cfg.data.scale_sample_rows))
    rng = np.random.default_rng(cfg.train.seed)
    sample_idx = np.sort(rng.choice(train_rows, size=sample_count, replace=False))
    sampled = np.asarray(clean_ro[sample_idx], dtype=np.float64)
    global_scale = float(np.percentile(np.abs(sampled).reshape(-1), cfg.data.global_scale_percentile))
    if not np.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("Invalid global scale from the paired library: %s" % global_scale)
    gate_scale = robust_gate_scale(sampled, global_scale)

    ls = int(cfg.data.late_start_index)
    _c = np.asarray(clean_ro[sample_idx], dtype=np.float64)
    _n = np.asarray(noisy_ro[sample_idx], dtype=np.float64) - _c
    _crms = np.sqrt(np.mean(_c ** 2, axis=1))
    _nrms = np.sqrt(np.mean(_n ** 2, axis=1))
    snr = 20.0 * np.log10(np.maximum(_crms, 1e-300) / np.maximum(_nrms, 1e-300))
    snr = snr[np.isfinite(snr)]
    _lc = np.sqrt(np.mean(_c[:, ls:] ** 2, axis=1))
    _ln = np.sqrt(np.mean(_n[:, ls:] ** 2, axis=1))
    snr_late = 20.0 * np.log10(np.maximum(_lc, 1e-300) / np.maximum(_ln, 1e-300))
    snr_late = snr_late[np.isfinite(snr_late)]

    _idn = int(np.sum(np.all(
        np.asarray(clean_ro[sample_idx], dtype=np.float64)
        == np.asarray(noisy_ro[sample_idx], dtype=np.float64), axis=1)))
    _idn_frac = _idn / max(sample_idx.size, 1)
    if _idn_frac > 0.01:
        raise ValueError(
            "%.1f%% of sampled stations are bit-identical between the clean and noisy files. They are almost "
            "certainly the same file, or one was copied over the other: the noise target would be zero and "
            "the identity map would score perfectly."
            % (100.0 * _idn_frac,)
        )

    def _roughness(m: np.ndarray) -> float:
        eq = m / np.maximum(gate_scale[None, :], 1.0e-300)
        d2 = np.diff(eq, n=2, axis=1)
        num = np.sqrt(np.mean(d2 ** 2, axis=1))
        den = np.maximum(np.sqrt(np.mean(eq ** 2, axis=1)), 1.0e-300)
        return float(np.median(num / den))

    _rc = _roughness(np.asarray(clean_ro[sample_idx], dtype=np.float64))
    _rn = _roughness(np.asarray(noisy_ro[sample_idx], dtype=np.float64))
    if _rc > _rn * 1.05:
        raise ValueError(
            "The file named 'clean' is ROUGHER than the file named 'noisy' (normalized second-difference "
            "%.4g vs %.4g). A transient decay is smooth in log-time and its measurement is not, so this "
            "ordering means the two roles are swapped: training would fit a model that ADDS this survey's "
            "noise, and every metric would improve while it did so. Swap the *_clean.csv and *_noisy.csv "
            "paths."
            % (_rc, _rn)
        )
    logger.info(
        "Pairing integrity: %d/%d sampled stations identical (%.3f%%), roughness clean %.4g < "
        "noisy %.4g. The roles are the right way round.",
        _idn, int(sample_idx.size), 100.0 * _idn_frac, _rc, _rn)

    _clean_full = np.asarray(clean_ro, dtype=np.float64)
    bg_window = _choose_background_window(
        _clean_full, province_id.astype(np.int64), int(cfg.data.late_start_index), logger,
        requested=int(getattr(cfg.data, "paired_background_window", 0) or 0),
    )
    cfg.data.paired_background_window = int(bg_window)
    bg = _lateral_background(_clean_full, province_id.astype(np.int64), bg_window)
    bg_mm = np.lib.format.open_memmap(
        bg_path, mode="w+", dtype=np.dtype(cfg.data.cache_dtype), shape=(total_rows, gates))
    bg_mm[:] = bg.astype(cfg.data.cache_dtype, copy=False)
    bg_mm.flush()
    del bg_mm

    thr = float(getattr(cfg.data, "paired_event_threshold", 0.15))
    rel = (np.asarray(clean_ro, dtype=np.float64) - bg) / np.maximum(np.abs(bg), 1.0e-30)
    ev = np.clip(np.abs(rel) / max(thr, 1.0e-9), 0.0, 1.0)
    ev[:, :ls] = 0.0
    ev_mm = np.lib.format.open_memmap(
        event_path, mode="w+", dtype=np.float32, shape=(total_rows, gates))
    ev_mm[:] = ev.astype(np.float32, copy=False)
    ev_mm.flush()
    del ev_mm
    anom_rows = int((ev.max(axis=1) >= 1.0).sum())

    meta = {
        "signature": signature,
        "rows": total_rows,
        "target_time": np.asarray(target_time, dtype=np.float64).tolist(),
        "target_gates": int(gates),
        "pairing_integrity": {
            "identical_row_fraction": _idn_frac,
            "roughness_clean": _rc,
            "roughness_noisy": _rn,
        },
        "global_scale": global_scale,
        "gate_scale": gate_scale.tolist(),
        "province_names": province_names,
        "gate_names": ref_gate_names,
        "split_rows": {n: int((split_id == i).sum()) for i, n in enumerate(PAIRED_SPLIT_NAMES)},
        "profile_groups": _g["summary"],
        "late_start_index": int(ls),
        "anomalous_rows": anom_rows,
        "realized_snr_db": {
            "p05": float(np.percentile(snr, 5)) if snr.size else None,
            "median": float(np.median(snr)) if snr.size else None,
            "p95": float(np.percentile(snr, 95)) if snr.size else None,
            "late_median": float(np.median(snr_late)) if snr_late.size else None,
        },
    }
    atomic_json_dump(meta, metadata_path)
    logger.info(
        "PAIRED CACHE COMPLETE | %d stations, %d gates, %d work areas "
        "(train %d / val %d / test %d rows) | global scale %.6g | gate dynamic range %.4g",
        total_rows, gates, len(province_names),
        meta["split_rows"]["train"], meta["split_rows"]["val"], meta["split_rows"]["test"],
        global_scale, float(gate_scale.max() / gate_scale.min()),
    )
    logger.info(
        "REALIZED PAIR SNR (the supervision the data actually delivers) | global p05 %.1f dB, "
        "median %.1f dB, p95 %.1f dB | LATE-window median %.1f dB. The late figure is the one that "
        "matters: it is the difficulty of the window the contract is written on.",
        meta["realized_snr_db"]["p05"] or float("nan"),
        meta["realized_snr_db"]["median"] or float("nan"),
        meta["realized_snr_db"]["p95"] or float("nan"),
        meta["realized_snr_db"]["late_median"] or float("nan"))
    if (meta["realized_snr_db"]["late_median"] is not None
            and meta["realized_snr_db"]["late_median"] > 40.0):
        logger.warning(
            "The LATE window of these pairs carries almost no noise (median %.1f dB). The "
            "late-window contract would then be met by any reasonable model, and the measured profile -- "
            "whose late window is genuinely buried -- would be far outside anything seen in training. Check "
            "that the noisy files are the intended noise level.",
            meta["realized_snr_db"]["late_median"])
    logger.info(
        "Lateral background: running median over %d stations within each work area; %d of %d "
        "stations (%.1f%%) carry a late-window departure of at least %.0f%% from it and are supervised "
        "as anomalies.",
        bg_window, anom_rows, total_rows, 100.0 * anom_rows / max(total_rows, 1),
        100.0 * thr)
    if anom_rows == 0:
        logger.warning(
            "NO station departs from its lateral background by %.0f%% in the "
            "late window. The paired anomaly supervision will contribute nothing. Lower "
            "--paired_event_threshold if the bodies in this dataset are subtler than "
            "that.",
            100.0 * thr,
        )

    desc = _load_or_build_descriptors(data_path, cache_dir, total_rows, target_time, cfg, logger, force=True)
    return CleanCacheInfo(
        data_path=str(data_path), station_path=str(station_path),
        metadata_path=str(metadata_path), rows=total_rows, gates=gates,
        target_time=np.asarray(target_time, dtype=np.float64),
        global_scale=global_scale, gate_scale=gate_scale,
        descriptor_path=str(cache_dir / "clean_descriptors.npy"), descriptors=desc,
        province_path=str(province_path), province_id=province_id,
        province_names=province_names,
        paired_noisy_path=str(noisy_path), paired_split_path=str(split_path),
        paired_background_path=str(bg_path), paired_event_path=str(event_path),
        paired_split_id=split_id, paired_gate_names=ref_gate_names,
        paired_group_id=group_id,
    )


def build_paired_noise_cache(
    clean_cache: CleanCacheInfo,
    cache_dir: Path,
    cfg: "Config",
    logger: logging.Logger,
) -> NoiseCacheInfo:
    """The noise bank of the paired path is the measured residual ``noisy - clean``, one row per station, blocked by
    work area.
    """
    data_path = cache_dir / "paired_noise.npy"
    metadata_path = cache_dir / "paired_noise_metadata.json"
    rows, gates = int(clean_cache.rows), int(clean_cache.gates)
    signature = {"from": clean_cache.data_path, "rows": rows, "gates": gates,
                 "dtype": cfg.data.cache_dtype, "schema": 3,
                 "noisy_digest": _content_digest(getattr(clean_cache, "paired_noisy_path", "")),
                 "clean_digest": _content_digest(clean_cache.data_path)}
    province_id = np.asarray(clean_cache.province_id, dtype=np.int64)
    block_rows = [int((province_id == p).sum()) for p in range(len(clean_cache.province_names or []))]
    block_rows = [b for b in block_rows if b > 0]
    if metadata_path.exists() and data_path.exists() and not cfg.runtime.overwrite_cache:
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("signature") == signature:
            logger.info("Using existing measured-noise cache: %s", data_path)
            return NoiseCacheInfo(
                data_path=str(data_path), metadata_path=str(metadata_path),
                rows=rows, gates=gates, target_time=clean_cache.target_time,
                block_rows=list(meta.get("block_rows") or block_rows),
            )
    clean_ro = np.load(clean_cache.data_path, mmap_mode="r")
    noisy_ro = np.load(clean_cache.paired_noisy_path, mmap_mode="r")
    mm = np.lib.format.open_memmap(
        data_path, mode="w+", dtype=np.dtype(cfg.data.cache_dtype), shape=(rows, gates))
    step = 8192
    for a in range(0, rows, step):
        b = min(a + step, rows)
        mm[a:b] = (np.asarray(noisy_ro[a:b], dtype=np.float64)
                   - np.asarray(clean_ro[a:b], dtype=np.float64)
                   ).astype(cfg.data.cache_dtype, copy=False)
    mm.flush()
    del mm
    atomic_json_dump(
        {"signature": signature, "rows": rows, "gates": gates,
         "block_rows": block_rows,
         "target_time": np.asarray(clean_cache.target_time, dtype=np.float64).tolist()},
        metadata_path,
    )
    logger.info(
        "Paired residual cache written (noisy - clean): %d residual traces in %d work-area "
        "blocks.",
        rows, len(block_rows))
    return NoiseCacheInfo(
        data_path=str(data_path), metadata_path=str(metadata_path),
        rows=rows, gates=gates, target_time=clean_cache.target_time,
        block_rows=block_rows,
    )


def describe_validated_fusion(d: Mapping[str, float]) -> str:
    """One log sentence for the validated fusion (training loop and field inference)."""
    _st = int(round(float(d.get("fusion_state", -1.0))))
    if _st == 2:
        return ("%d validation stations the network never trained | late error there: network alone %.1f%% (level x%.2f), "
                "family completion %.1f%% (level x%.2f) -> in a buried late gate of a survey-family row the family corrects %.2f of "
                "the laterally common part of (completion - network) (kappa, late mean; min %.2f, max %.2f); what one station "
                "holds against its neighbours stays the network's | delivered there: see the SURVEY PROBE line. "
                % (int(d.get("fusion_evidence_stations", 0)), float(d.get("fusion_network_alone_late", float("nan"))),
                   float(d.get("fusion_network_alone_level", float("nan"))), float(d.get("fusion_completion_late", float("nan"))),
                   float(d.get("fusion_completion_level", float("nan"))), float(d.get("fusion_share_late", 1.0)),
                   float(d.get("fusion_share_min", 1.0)), float(d.get("fusion_share_max", 1.0))))
    if _st == 1:
        return ("no held-out evidence (%d validation stations this network never trained; at least 4 are needed): kappa = 1, the "
                "family completion sets the laterally common level of the buried late gates of survey-family rows. Train with "
                "--survey_val_blocks >= 1 to let the validation stations decide."
                 % int(d.get("fusion_evidence_stations", 0)))
    return ("off (--no_family_validated_fusion or --no_family_gate_floor): kappa = 1.")


def audit_and_prepare_data(cfg: Config, run_dir: Path, logger: logging.Logger) -> Tuple[CleanCacheInfo, NoiseCacheInfo, Dict[str, Any]]:
    _paired_dir = Path(str(getattr(cfg.paths, "paired_dir", "") or "").strip() or ".")
    _paired_on = bool(getattr(cfg.data, "paired_supervision", True))
    _paired_files = paired_split_files(_paired_dir) if _paired_on else None
    if _paired_on and not _paired_files:
        raise FileNotFoundError(
            "Paired dataset root not found or incomplete: %s (PROJECT_ROOT=%s). Expected "
            "train/val/test_{clean,noisy}.csv there. Fix PROJECT_ROOT at the top of the program or pass "
            "--paired_dir; pass --no_paired only if you really mean the synthetic branch. "
            % (str(_paired_dir), PROJECT_ROOT))
    if _paired_files:
        field_path = Path(str(cfg.paths.field_csv)) if field_profile_present(cfg) else None
        if field_path is not None and not field_path.exists():
            raise FileNotFoundError(
                "Missing field CSV: %s (pass --no_field for a study without a measured profile)."
                % (field_path,))
        if field_path is None:
            logger.info("NO FIELD PROFILE (--no_field): known-truth paired study; the measured section, the "
                        "survey arm, the amplitude audit and every field stage are off.")
            if bool(getattr(cfg.data, "unify_axis_to_measured", False)):
                cfg.data.unify_axis_to_measured = False
                logger.info("unified measured axis off: there is no measured profile to unify to.")
        shared_cache_dir = Path(cfg.paths.output_root) / "_cache"
        shared_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "PAIRED SUPERVISION ACTIVE | dataset root %s | %d split(s): %s. The synthetic "
            "noise-injection path, the noise database and the forward/field domain mixer are OFF for this "
            "run; the paired training arms are listed in the SUPERVISION MODE line.",
            _paired_dir, len(_paired_files), ", ".join(sorted(_paired_files)))
        clean_cache = build_paired_cache(_paired_dir, shared_cache_dir, cfg, logger)
        try:
            _ua = (None if bool(getattr(cfg.data, "unify_axis_to_measured", False))
                      else unified_measured_axis(cfg, int(clean_cache.target_time.size), logger))
            if _ua is not None:
                _old = np.asarray(clean_cache.target_time, dtype=np.float64).copy()
                clean_cache.target_time = _ua
                clean_cache.unified_axis_from = [float(_old[0]), float(_old[-1])]
                logger.warning("UNIFIED AXIS IN FORCE | simulated gates relabelled %.4g..%.4g s -> measured span "
                               "%.4g..%.4g s (%d gates, %s) by operator instruction; data-driven, no physical-time assertion. ",
                               float(_old[0]), float(_old[-1]), float(_ua[0]), float(_ua[-1]), int(_ua.size),
                               str(getattr(cfg.data, "unified_axis_source", "noise_library")))
        except Exception as _e:
            raise RuntimeError("unified axis could not be built: %r" % (_e,))
        try:
            _mn = build_measured_noise_cache(
                Path(str(getattr(cfg.paths, "noise_library_dir", "") or ".")),
                clean_cache.target_time, shared_cache_dir, cfg, logger)
            if _mn:
                clean_cache.measured_noise_path = _mn["path"]
            _xc = build_extra_clean_cache(
                _paired_dir, clean_cache.target_time,
                shared_cache_dir, cfg, logger)
            if _xc:
                clean_cache.extra_clean_path = _xc["path"]
                clean_cache.extra_clean_full_coverage = bool(_xc.get("full_coverage", False))
            elif float(getattr(cfg.data, "measured_noise_inject", 0.0)) > 0.0 and _mn:
                logger.info(
                    "no extra clean-only set at %s/clean.csv: the measured-library injection arm (share %.2f) "
                    "takes its clean rows from the paired training set.",
                    str(_paired_dir), float(cfg.data.measured_noise_inject))
            elif float(getattr(cfg.data, "measured_noise_inject", 0.0)) > 0.0:
                logger.warning(
                    "measured_noise_inject is %.2f but no library was found at %s; the injection arm is "
                    "inactive.",
                    float(cfg.data.measured_noise_inject),
                    getattr(cfg.paths, "noise_library_dir", ""))
        except Exception as _mexc:
            logger.error("Measured-noise library rejected: %s", _mexc)
            raise
        noise_cache = build_paired_noise_cache(clean_cache, shared_cache_dir, cfg, logger)
        field_values = None
        field_report: Dict[str, Any] = {"present": False}
        if field_path is not None:
            field_station, field_values, _fh, field_report = load_field_matrix(
                field_path, int(getattr(cfg.data, "field_native_gates", 0) or cfg.data.target_gates), logger)
            _field_native = np.asarray(field_values, dtype=np.float64).copy()
            field_values = field_to_model_axis(field_values, _fh, cfg, np.asarray(clean_cache.target_time, dtype=np.float64), logger)
            try:
                _sv = build_survey_profile_cache(field_station, _field_native, _fh, field_values, clean_cache,
                                                        shared_cache_dir, cfg, logger)
                if _sv:
                    clean_cache.survey_profile_path = str(_sv["path"])
            except Exception as _e:
                logger.warning("survey profile cache not built: %r.", _e)
        audit: Dict[str, Any] = {
            "program": PROGRAM_NAME,
            "version": PROGRAM_VERSION,
            "mode": "paired_supervision",
            "paired_dir": str(_paired_dir),
            "paired_files": {k: [str(v[0]), str(v[1])] for k, v in _paired_files.items()},
            "field": ({**inspect_csv_structure(field_path), "ingestion": field_report} if field_path is not None
                      else {"present": False}),
            "clean_cache": dataclass_to_dict(clean_cache),
            "noise_cache": dataclass_to_dict(noise_cache),
            "resolved_target_time": clean_cache.target_time.tolist(),
        }
        with Path(clean_cache.metadata_path).open("r", encoding="utf-8") as _f:
            audit["paired_metadata"] = json.load(_f)
        try:
            if field_values is None:
                raise LookupError("no field profile (--no_field)")
            _em = max(int(cfg.data.late_start_index), 1)
            _lib = np.asarray(
                np.load(clean_cache.data_path, mmap_mode="r")[: min(clean_cache.rows, 20000)],
                dtype=np.float64)
            _r = float(
                np.median(np.sqrt(np.mean(field_values[:, :_em] ** 2, axis=1)))
                / max(float(np.median(np.sqrt(np.mean(_lib[:, :_em] ** 2, axis=1)))), 1e-300))
            audit["field_library_amplitude_ratio_early_mid"] = _r
            logger.info(
                "Field/library EARLY-MID amplitude ratio: %.3g | field %d stations x %d gates.",
                _r, int(field_values.shape[0]), int(field_values.shape[1]))
        except Exception as _exc:
            (logger.info if field_values is None else logger.warning)("Amplitude audit skipped: %s", _exc)
        atomic_json_dump(audit, run_dir / "reports" / "data_audit.json")
        return clean_cache, noise_cache, audit

    raise ValueError("Training needs a paired dataset (train/val/test_{clean,noisy}.csv in --paired_dir).")


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        z = (x + np.uint64(0x9E3779B97F4A7C15)) & mask
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
        return z ^ (z >> np.uint64(31))


CURVE_DESCRIPTOR_NAMES = (
    "log10_rms_amplitude",
    "log10_late_over_early",
    "late_loglog_slope",
    "log10_late_tau_ms",
    "logconvexity_violation_fraction",
    "nonpositive_fraction",
)


def compute_curve_descriptors(matrix: np.ndarray, times_ms: np.ndarray, late_start: int) -> np.ndarray:
    """Per-curve physical descriptors, vectorized over a chunk of the library."""
    v = np.asarray(matrix, dtype=np.float64)
    t = np.asarray(times_ms, dtype=np.float64).reshape(-1)
    n, g = v.shape
    late = int(np.clip(late_start, 1, g - 2))
    eps = 1e-300
    out = np.zeros((n, len(CURVE_DESCRIPTOR_NAMES)), dtype=np.float64)
    rms = np.sqrt(np.mean(v**2, axis=1))
    out[:, 0] = np.log10(np.maximum(rms, eps))
    early_rms = np.sqrt(np.mean(v[:, :late] ** 2, axis=1))
    late_rms = np.sqrt(np.mean(v[:, late:] ** 2, axis=1))
    out[:, 1] = np.log10(np.maximum(late_rms, eps)) - np.log10(np.maximum(early_rms, eps))
    lt = log_time_axis(t)
    pos = v > 0
    lv = np.log(np.maximum(np.abs(v), eps))
    for i in range(n):
        ok = pos[i, late:]
        if int(ok.sum()) >= 3:
            xs = lt[late:][ok]
            ys = lv[i, late:][ok]
            out[i, 2] = float(np.polyfit(xs, ys, 1)[0])
            a = np.polyfit(t[late:][ok], ys, 1)
            out[i, 3] = math.log10(max(-1.0 / a[0], 1e-6)) if a[0] < 0 else np.nan
        else:
            out[i, 2] = np.nan
            out[i, 3] = np.nan
    dt_l = t[1:-1] - t[:-2]
    dt_r = t[2:] - t[1:-1]
    slope_l = (lv[:, 1:-1] - lv[:, :-2]) / np.maximum(dt_l, 1e-30)[None, :]
    slope_r = (lv[:, 2:] - lv[:, 1:-1]) / np.maximum(dt_r, 1e-30)[None, :]
    curv = 2.0 * (slope_r - slope_l) / np.maximum(dt_l + dt_r, 1e-30)[None, :]
    scale = 0.5 * (slope_l**2 + slope_r**2) + 1e-30
    curv_norm = curv / scale
    out[:, 4] = np.mean(curv_norm < -0.05, axis=1)
    out[:, 5] = np.mean(v <= 0.0, axis=1)
    return out


def deterministic_group_split_indices(
    rows: int,
    group_size: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    descriptors: Optional[np.ndarray] = None,
    min_groups: int = 120,
    min_val_groups: int = 6,
    stratify: bool = True,
) -> Dict[str, np.ndarray]:
    """Contiguity-preserving, seed-respecting, population-matched split."""
    if rows < 3:
        raise ValueError("At least three clean rows are required for train/val/test splits.")
    requested = max(1, int(group_size))
    group_size = max(1, min(requested, max(1, rows // max(int(min_groups), 3))))
    idx = np.arange(rows, dtype=np.int64)
    groups = idx // group_size
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        groups = idx
        unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    keys = _splitmix64(unique_groups.astype(np.uint64) ^ _splitmix64(np.uint64(seed)))
    n_val = max(1, int(round(n_groups * val_fraction)))
    n_test = max(1, int(round(n_groups * max(0.0, 1.0 - train_fraction - val_fraction))))
    if n_val + n_test >= n_groups:
        n_val = n_test = 1

    strata: Optional[np.ndarray] = None
    if stratify and descriptors is not None and len(descriptors) == rows and n_groups >= 12:
        d = np.asarray(descriptors, dtype=np.float64)
        gm = np.zeros((n_groups, 2), dtype=np.float64)
        for j, gid in enumerate(unique_groups):
            rows_j = idx[groups == gid]
            block = d[rows_j][:, [0, 1]]
            with np.errstate(invalid="ignore"):
                gm[j] = np.nanmedian(block, axis=0)
        gm = np.nan_to_num(gm, nan=0.0, posinf=0.0, neginf=0.0)
        nq = int(np.clip(int(math.sqrt(n_groups / 6.0)), 2, 5))
        codes = np.zeros(n_groups, dtype=np.int64)
        for c in range(2):
            edges = np.quantile(gm[:, c], np.linspace(0.0, 1.0, nq + 1)[1:-1]) if nq > 1 else np.asarray([])
            codes = codes * nq + np.searchsorted(edges, gm[:, c], side="right")
        strata = codes

    def _assign(members: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = members[np.argsort(keys[members], kind="mergesort")]
        m = len(order)
        v = int(round(m * val_fraction))
        s = int(round(m * max(0.0, 1.0 - train_fraction - val_fraction)))
        v = min(max(v, 0), max(m - 2, 0))
        s = min(max(s, 0), max(m - 2 - v, 0))
        tr = m - v - s
        return order[:tr], order[tr : tr + v], order[tr + v :]

    if strata is None:
        order = np.argsort(keys, kind="mergesort")
        tr = unique_groups[order][: n_groups - n_val - n_test]
        va = unique_groups[order][n_groups - n_val - n_test : n_groups - n_test]
        te = unique_groups[order][n_groups - n_test :]
    else:
        tr_l, va_l, te_l = [], [], []
        for code in np.unique(strata):
            members = np.where(strata == code)[0]
            a, b, c = _assign(members)
            tr_l.append(unique_groups[a])
            va_l.append(unique_groups[b])
            te_l.append(unique_groups[c])
        tr = np.concatenate(tr_l) if tr_l else np.asarray([], dtype=np.int64)
        va = np.concatenate(va_l) if va_l else np.asarray([], dtype=np.int64)
        te = np.concatenate(te_l) if te_l else np.asarray([], dtype=np.int64)
        pool = list(tr[np.argsort(keys[np.searchsorted(unique_groups, tr)], kind="mergesort")])
        while len(va) < min(min_val_groups, max(n_groups - 2, 1)) and len(pool) > 2:
            va = np.append(va, pool.pop())
        while len(te) < min(min_val_groups, max(n_groups - 2, 1)) and len(pool) > 2:
            te = np.append(te, pool.pop())
        tr = np.asarray(pool, dtype=np.int64)
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        raise ValueError(
            "Split degenerated (train/val/test group counts %d/%d/%d). The clean library is too small "
            "or too uniform for a group split; reduce --split_group_size." % (len(tr), len(va), len(te))
        )
    return {
        "train": idx[np.isin(groups, tr)],
        "val": idx[np.isin(groups, va)],
        "test": idx[np.isin(groups, te)],
        "_groups": groups,
        "_group_size": np.asarray([group_size], dtype=np.int64),
        "_n_groups": np.asarray([n_groups], dtype=np.int64),
        "_val_groups": np.asarray(va, dtype=np.int64),
        "_test_groups": np.asarray(te, dtype=np.int64),
        "_train_groups": np.asarray(tr, dtype=np.int64),
    }


def province_split_indices(
    province_id: np.ndarray,
    province_names: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, np.ndarray]:
    """Split by province."""
    pid = np.asarray(province_id, dtype=np.int64)
    synth_lib = pid < 0
    real_pid = pid[~synth_lib]
    n_prov = int(real_pid.max()) + 1 if real_pid.size else 0
    if n_prov < 3:
        raise ValueError(
            "Province split needs at least 3 provinces (got %d). Use the single-file "
            "library, or point --clean_dir at a directory with more files." % n_prov
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_prov)
    n_test = max(1, int(round(n_prov * float(test_fraction))))
    n_val = max(1, int(round(n_prov * float(val_fraction))))
    if n_test + n_val >= n_prov:
        n_val = max(1, n_prov - n_test - 1)
    test_p = set(order[:n_test].tolist())
    val_p = set(order[n_test : n_test + n_val].tolist())
    train_p = set(order[n_test + n_val :].tolist())
    idx = np.arange(pid.size, dtype=np.int64)
    train_mask = np.isin(pid, list(train_p)) | synth_lib
    out = {
        "train": idx[train_mask],
        "val": idx[np.isin(pid, list(val_p))],
        "test": idx[np.isin(pid, list(test_p))],
        "train_provinces": np.array(sorted(train_p), dtype=np.int64),
        "val_provinces": np.array(sorted(val_p), dtype=np.int64),
        "test_provinces": np.array(sorted(test_p), dtype=np.int64),
        "_groups": pid.astype(np.int64),
        "_group_size": np.asarray(
            [int(np.bincount(real_pid).max()) if real_pid.size else 1], dtype=np.int64
        ),
        "_n_groups": np.asarray([n_prov], dtype=np.int64),
        "_train_groups": np.array(sorted(train_p), dtype=np.int64),
        "_val_groups": np.array(sorted(val_p), dtype=np.int64),
        "_test_groups": np.array(sorted(test_p), dtype=np.int64),
    }
    if logger is not None:
        names = list(province_names)
        logger.info(
            "PROVINCE SPLIT | %d provinces -> train %d (%d stations), val %d (%d stations), "
            "test %d (%d stations). No province is shared, so a val/test curve's geology was "
            "never seen in training.",
            n_prov, len(train_p), out["train"].size, len(val_p), out["val"].size,
            len(test_p), out["test"].size,
        )
        logger.info("  val provinces : %s", ", ".join(names[i] for i in sorted(val_p)))
        logger.info("  test provinces: %s", ", ".join(names[i] for i in sorted(test_p)))
    return out


def _shape_digest_rows(clean_cache: "CleanCacheInfo", decimals: int = 3) -> np.ndarray:
    lib = np.load(clean_cache.data_path, mmap_mode="r")
    n = int(clean_cache.rows)
    out = np.empty(n, dtype=np.int64)
    step = 20000
    for s in range(0, n, step):
        a = np.asarray(lib[s : min(s + step, n)], dtype=np.float64)
        rms = np.sqrt(np.maximum((a ** 2).mean(axis=1, keepdims=True), 1e-300))
        a = a / rms
        q = np.round(np.log10(np.abs(a) + 1e-12), decimals) * np.sign(a)
        for i, row in enumerate(q):
            out[s + i] = int.from_bytes(
                hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "big", signed=True)
    return out


def enforce_duplicate_free_split(
    splits: Dict[str, np.ndarray],
    clean_cache: "CleanCacheInfo",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, np.ndarray]:
    """Move every copy of A shape to one side of the split."""
    try:
        dig = _shape_digest_rows(clean_cache)
    except Exception as exc:
        if logger is not None:
            logger.error(
                "Could not compute shape digests to de-duplicate the split (%r). The "
                "split is left as it was and the duplicate probe will judge it.", exc)
        return splits
    owner = {"train": 0, "val": 1, "test": 2}
    missing = [k for k in owner if k not in splits]
    if missing:
        if logger is not None:
            logger.error("Split is missing %s; de-duplication skipped.", missing)
        return splits
    side = np.full(int(clean_cache.rows), -1, dtype=np.int8)
    for name, code in owner.items():
        side[np.asarray(splits[name], dtype=np.int64)] = code
    used = side >= 0
    order = np.argsort(dig[used], kind="mergesort")
    rows_used = np.nonzero(used)[0][order]
    d_sorted = dig[rows_used]
    bounds = np.nonzero(np.diff(d_sorted))[0] + 1
    moved = {"train": 0, "val": 0, "test": 0}
    dropped = [0]
    groups_merged = 0
    for a, b in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(rows_used)]])):
        if b - a < 2:
            continue
        rows = rows_used[a:b]
        sides = side[rows]
        if np.all(sides == sides[0]):
            continue
        groups_merged += 1
        counts = np.bincount(sides, minlength=3)
        top = int(counts.max())
        target = 2 if counts[2] == top else (1 if counts[1] == top else 0)
        keep_all = (target == 0)
        first_on_target = int(rows[np.nonzero(sides == target)[0][0]])
        for r, s in zip(rows, sides):
            r = int(r)
            if int(s) == target and (keep_all or r == first_on_target):
                continue
            moved[("train", "val", "test")[int(s)]] += 1
            side[r] = -1
            dropped[0] += 1
    out = {k: v for k, v in splits.items() if k not in owner}
    for name, code in owner.items():
        out[name] = np.nonzero(side == code)[0].astype(np.int64)
    for _n in ("val", "test"):
        _before, _after = int(np.asarray(splits[_n]).size), int(out[_n].size)
        _collapsed = _before > 0 and _after < _before and (
            _after < 0.25 * _before or (_before >= 32 and _after < 32))
        if _collapsed:
            raise RuntimeError(
                "De-duplicating the split leaves only %d of %d %s curves. The library "
                "itself repeats the same modelled shapes across provinces, so there is no "
                "held-out population large enough to support a generalization claim: this is a "
                "DATA problem and no split can fix it. Either de-duplicate the clean library at "
                "source (keep one copy of each parameter combination), or accept that this "
                "library can only support an in-distribution reconstruction claim. "
                "(--no_split_dedup_by_shape runs anyway; the duplicate probe will then block "
                "deployment, correctly.)" % (_after, _before, _n)
            )
    if logger is not None:
        if groups_merged:
            logger.warning(
                "DE-DUPLICATED SPLIT | %d shape groups straddled the split; %d "
                "redundant copies were DROPPED (never moved across the boundary, which would "
                "break the by-province guarantee) so "
                "no shape is weighted twice in the held-out statistics (the late CVaR is a "
                "worst-decile mean, where a duplicated hard curve counts twice). Held-out sizes: "
                "val %d -> %d, test %d -> %d. An identical curve on both sides is not a held-out "
                "test, so the smaller number is the honest one.",
                groups_merged, dropped[0],
                int(np.asarray(splits["val"]).size), int(out["val"].size),
                int(np.asarray(splits["test"]).size), int(out["test"].size),
            )
        else:
            logger.info("Split de-duplication: no shape group straddled the split.")
    return out


def shape_first_stratified_split(
    splits: Dict[str, np.ndarray],
    clean_cache: "CleanCacheInfo",
    cfg: "Config",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, np.ndarray]:
    """Shape-first stratified split."""
    owner = ("train", "val", "test")
    missing = [k for k in owner if k not in splits]
    groups_row = splits.get("_groups")
    if missing or groups_row is None:
        if logger is not None:
            logger.error("Split is missing %s; shape-first pass skipped, "
                         "falling back to drop-based de-duplication.", missing or "_groups")
        return enforce_duplicate_free_split(splits, clean_cache, logger)
    try:
        dig = _shape_digest_rows(clean_cache)
    except Exception as exc:
        if logger is not None:
            logger.error("Could not compute shape digests (%r); falling back "
                         "to drop-based de-duplication.", exc)
        return enforce_duplicate_free_split(splits, clean_cache, logger)
    n_rows = int(clean_cache.rows)
    gid_row = np.asarray(groups_row, dtype=np.int64)
    if gid_row.shape[0] != n_rows:
        if logger is not None:
            logger.error("_groups length %d != cache rows %d; falling back.",
                         gid_row.shape[0], n_rows)
        return enforce_duplicate_free_split(splits, clean_cache, logger)
    real_mask = gid_row >= 0
    n_real = int(real_mask.sum())
    if n_real < 3:
        return enforce_duplicate_free_split(splits, clean_cache, logger)
    side0 = np.full(n_rows, -1, dtype=np.int8)
    for c, name in enumerate(owner):
        side0[np.asarray(splits[name], dtype=np.int64)] = c
    frac = np.zeros(3, dtype=np.float64)
    for c in range(3):
        frac[c] = float(np.sum((side0 == c) & real_mask)) / max(n_real, 1)
    if frac[1] <= 0.0 or frac[2] <= 0.0:
        frac = np.asarray([0.90, 0.05, 0.05], dtype=np.float64)
    uniq_g = np.unique(gid_row[real_mask])
    g_index = {int(g): i for i, g in enumerate(uniq_g)}
    parent = np.arange(len(uniq_g), dtype=np.int64)

    def _find(a: int) -> int:
        r = a
        while parent[r] != r:
            r = int(parent[r])
        while parent[a] != r:
            parent[a], a = r, int(parent[a])
        return r

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    fwd_dig = np.unique(dig[~real_mask]) if bool((~real_mask).any()) else np.asarray([], dtype=dig.dtype)
    order = np.argsort(dig, kind="mergesort")
    d_sorted = dig[order]
    bounds = np.nonzero(np.diff(d_sorted))[0] + 1
    straddling = 0
    for a, b in zip(np.concatenate([[0], bounds]), np.concatenate([bounds, [len(order)]])):
        if b - a < 2:
            continue
        rows = order[a:b]
        gset = np.unique(gid_row[rows])
        reals = gset[gset >= 0]
        if reals.size <= 1:
            continue
        straddling += 1
        base = g_index[int(reals[0])]
        for g in reals[1:]:
            _union(base, g_index[int(g)])
    roots = np.asarray([_find(i) for i in range(len(uniq_g))], dtype=np.int64)
    comp_of_row = np.full(n_rows, -1, dtype=np.int64)
    comp_of_row[real_mask] = roots[np.searchsorted(uniq_g, gid_row[real_mask])]
    comp_ids = np.unique(roots)
    comp_pos = {int(c): i for i, c in enumerate(comp_ids)}
    K = len(comp_ids)
    comp_rows = np.zeros(K, dtype=np.int64)
    comp_uniq = np.zeros(K, dtype=np.int64)
    comp_desc = np.zeros((K, 2), dtype=np.float64)
    ridx = np.nonzero(real_mask)[0]
    cpos = np.asarray([comp_pos[int(c)] for c in comp_of_row[ridx]], dtype=np.int64)
    np.add.at(comp_rows, cpos, 1)
    pos_of_row = np.full(n_rows, -1, dtype=np.int64)
    pos_of_row[ridx] = cpos
    reps = order[np.concatenate([[0], bounds])]
    reps_real = reps[real_mask[reps]]
    np.add.at(comp_uniq, pos_of_row[reps_real], 1)
    n_unique_real = int(reps_real.size)
    desc = getattr(clean_cache, "descriptors", None)
    if desc is not None and len(desc) == n_rows:
        d2 = np.nan_to_num(np.asarray(desc, dtype=np.float64)[:, :2][ridx],
                           nan=0.0, posinf=0.0, neginf=0.0)
        np.add.at(comp_desc, cpos, d2)
        comp_desc /= np.maximum(comp_rows, 1)[:, None]
    nq = int(np.clip(int(math.sqrt(max(K, 1) / 6.0)), 1, 6))
    codes = np.zeros(K, dtype=np.int64)
    if desc is not None and nq > 1:
        for c in range(2):
            edges = np.quantile(comp_desc[:, c], np.linspace(0.0, 1.0, nq + 1)[1:-1])
            codes = codes * nq + np.searchsorted(edges, comp_desc[:, c], side="right")
    seed = int(getattr(cfg.data, "province_split_seed", 0) or 0) ^ int(cfg.train.seed)
    keys = _splitmix64(comp_ids.astype(np.uint64)
                       ^ _splitmix64(np.uint64(seed & 0xFFFFFFFFFFFFFFFF)))
    target = frac * float(n_unique_real)
    ord2 = np.lexsort((keys, codes))
    u_ord = comp_uniq[ord2].astype(np.float64)
    cum_r = np.cumsum(u_ord)
    assign = np.zeros(K, dtype=np.int8)
    got_v = 0.0
    got_t = 0.0
    for _i in range(len(ord2)):
        _u = float(u_ord[_i])
        _dv = float(frac[1]) * float(cum_r[_i]) - got_v
        _dt = float(frac[2]) * float(cum_r[_i]) - got_t
        if _dv >= 0.5 * _u and _dv >= _dt:
            assign[ord2[_i]] = 1
            got_v += _u
        elif _dt >= 0.5 * _u:
            assign[ord2[_i]] = 2
            got_t += _u
    side = np.full(n_rows, -1, dtype=np.int8)
    side[~real_mask] = 0
    side[ridx] = assign[cpos]
    moved_fwd = 0
    if fwd_dig.size:
        held = np.nonzero(side >= 1)[0]
        bad = held[np.isin(dig[held], fwd_dig)]
        moved_fwd = int(bad.size)
        side[bad] = 0
    pool_side = side.copy()
    collapsed = {"val": 0, "test": 0}
    for c, name in ((1, "val"), (2, "test")):
        rows_c = np.nonzero(side == c)[0]
        if rows_c.size == 0:
            continue
        _, first = np.unique(dig[rows_c], return_index=True)
        keep = np.zeros(rows_c.size, dtype=bool)
        keep[first] = True
        collapsed[name] = int(rows_c.size - keep.sum())
        side[rows_c[~keep]] = -1
    for c, name in ((1, "val"), (2, "test")):
        _rows_c = int(np.sum(side == c))
        _min_rows = max(0.10 * target[c], min(64.0, 0.5 * target[c]))
        if _rows_c < _min_rows:
            if logger is not None:
                logger.error(
                    "Shape-first allocation leaves %s with %d rows against "
                    "a target of %d -- the shape topology is too welded to stratify "
                    "(one component holds most of the library, or the forward pool "
                    "shares most field shapes). Falling back to drop-based "
                    "de-duplication for this run; de-duplicate the library at source "
                    "to unlock the representative split.",
                    name, _rows_c, int(target[c]))
            return enforce_duplicate_free_split(splits, clean_cache, logger)
    out = {k: v for k, v in splits.items() if k not in owner}
    for c, name in enumerate(owner):
        out[name] = np.nonzero(side == c)[0].astype(np.int64)
    out["val_pool"] = np.nonzero(pool_side == 1)[0].astype(np.int64)
    out["test_pool"] = np.nonzero(pool_side == 2)[0].astype(np.int64)
    for c, (gk, pk) in enumerate(
        (("_train_groups", "train_provinces"), ("_val_groups", "val_provinces"),
         ("_test_groups", "test_provinces"))):
        gvals = np.unique(gid_row[out[owner[c]]])
        gvals = gvals[gvals >= 0]
        if gk in out:
            out[gk] = gvals.astype(np.int64)
        if pk in out:
            out[pk] = gvals.astype(np.int64)
    if logger is not None:
        _dd = (np.nan_to_num(np.asarray(desc, dtype=np.float64)[:, 1],
                             nan=0.0, posinf=0.0, neginf=0.0)
               if desc is not None and len(desc) == n_rows
               else np.zeros(n_rows))

        def _side_desc(c: int) -> float:
            sel = (side == c) & real_mask
            return float(_dd[sel].mean()) if bool(sel.any()) else float("nan")

        logger.info(
            "SHAPE-FIRST STRATIFIED SPLIT | %d shape components over %d "
            "groups (%d digest groups straddled) | rows train/val/test = %d/%d/%d "
            "(+station-contiguous pools val %d / test %d for the anomaly probe and "
            "neighbour consumers ) "
            "(unique-shape targets %.0f/%.0f/%.0f of %d) | held-out rows moved to TRAIN because the "
            "forward library shares their shape: %d | held-out exact-duplicate rows "
            "collapsed: val %d, test %d | row-weighted late/early descriptor by "
            "side: train %.3f / val %.3f / test %.3f. "
            "\u4e2d\u6587\uff1a\u9a8c\u8bc1/\u6d4b\u8bd5\u96c6\u73b0\u5728\u662f\u66f2\u7ebf\u5f62\u72b6\u65cf\u7684\u5206\u5c42\u4ee3\u8868\u6027\u6837\u672c\uff0c"
            "\u540c\u5f62\u72b6\u66f2\u7ebf\u6c38\u4e0d\u8de8\u8d8a\u5212\u5206\u3002",
            K, len(uniq_g), straddling,
            int(out["train"].size), int(out["val"].size), int(out["test"].size),
            int(out["val_pool"].size), int(out["test_pool"].size),
            target[0], target[1], target[2], n_unique_real, moved_fwd,
            collapsed["val"], collapsed["test"],
            _side_desc(0), _side_desc(1), _side_desc(2),
        )
    return out


def build_splits(clean_cache: "CleanCacheInfo", cfg: "Config") -> Dict[str, np.ndarray]:
    """Single entry point so every call site produces the identical split."""
    _dedup = bool(getattr(cfg.data, "split_dedup_by_shape", True))
    _log = logging.getLogger(PROGRAM_NAME) if _dedup else None
    _psid = getattr(clean_cache, "paired_split_id", None)
    if _psid is not None:
        _sp_arr = np.asarray(_psid, dtype=np.int64)
        _out = {
            name: np.nonzero(_sp_arr == i)[0].astype(np.int64)
            for i, name in enumerate(PAIRED_SPLIT_NAMES)
        }
        for _n in ("val", "test"):
            if _out[_n].size == 0:
                raise ValueError(
                    "The paired dataset produced an empty %s split." % (_n,)
                )
        _out["val_pool"] = _out["val"].copy()
        _out["test_pool"] = _out["test"].copy()
        _pid_arr = np.asarray(clean_cache.province_id, dtype=np.int64)
        _grp_of = {n: np.unique(_pid_arr[_out[n]]) for n in PAIRED_SPLIT_NAMES}
        _sizes = np.bincount(_pid_arr) if _pid_arr.size else np.asarray([1])
        _out["_groups"] = _pid_arr
        _out["_group_size"] = np.asarray(
            [int(_sizes.max()) if _sizes.size else 1], dtype=np.int64)
        _out["_n_groups"] = np.asarray([int(np.unique(_pid_arr).size)], dtype=np.int64)
        _out["_train_groups"] = _grp_of["train"].astype(np.int64)
        _out["_val_groups"] = _grp_of["val"].astype(np.int64)
        _out["_test_groups"] = _grp_of["test"].astype(np.int64)
        _names = list(clean_cache.province_names or [])
        _out["train_provinces"] = _out["_train_groups"].copy()
        _out["val_provinces"] = _out["_val_groups"].copy()
        _out["test_provinces"] = _out["_test_groups"].copy()
        if _log is not None and _names:
            def _few(ix: np.ndarray) -> str:
                nm = [_names[i] for i in ix if i < len(_names)]
                return "%d (%s%s)" % (len(nm), ", ".join(nm[:5]), ", ..." if len(nm) > 5 else "")
            _log.info(
                "Work areas held out whole | train %s | val %s | test %s.",
                _few(_out["_train_groups"]), _few(_out["_val_groups"]), _few(_out["_test_groups"]))
        if _log is not None:
            _log.info(
                "Split of the dataset's own files%s: train %d / val %d / test %d stations. No "
                "re-partitioning.",
                (", every group of near-identical profiles in the split of its lowest sample_id (policy grouped)"
                 if str(getattr(cfg.data, "paired_split_policy", "files")) == "grouped" else " as given (policy files)"),
                _out["train"].size, _out["val"].size, _out["test"].size)
        return _out
    if getattr(clean_cache, "province_id", None) is not None:
        _sp = province_split_indices(
            clean_cache.province_id,
            clean_cache.province_names or [],
            cfg.data.province_val_fraction,
            cfg.data.province_test_fraction,
            int(cfg.data.province_split_seed),
        )
        if _dedup and bool(getattr(cfg.data, "split_shape_first", True)):
            return shape_first_stratified_split(_sp, clean_cache, cfg, _log)
        return enforce_duplicate_free_split(_sp, clean_cache, _log) if _dedup else _sp
    desc = clean_cache.descriptors
    _sp = deterministic_group_split_indices(
        clean_cache.rows,
        cfg.data.split_group_size,
        cfg.data.train_fraction,
        cfg.data.val_fraction,
        cfg.train.seed,
        descriptors=desc,
        min_groups=int(cfg.data.split_min_groups),
        min_val_groups=int(cfg.data.split_min_val_groups),
        stratify=bool(cfg.data.split_stratify),
    )
    if _dedup and bool(getattr(cfg.data, "split_shape_first", True)):
        return shape_first_stratified_split(_sp, clean_cache, cfg, _log)
    return enforce_duplicate_free_split(_sp, clean_cache, _log) if _dedup else _sp


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 3 or b.size < 3:
        return float("nan")
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / a.size
    cb = np.searchsorted(b, allv, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


_LAST_SPLIT_AUDIT: Dict[str, Any] = {"split_leakage_pass": None, "reason": "probe not run"}


def cross_split_duplicate_probe(
    clean_cache: "CleanCacheInfo",
    splits: Mapping[str, np.ndarray],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Duplicate probe."""
    out: Dict[str, Any] = {}
    try:
        _d = np.load(clean_cache.data_path, mmap_mode="r")
        def _keys(idx: np.ndarray, cap: int = 20000, decimals: int = 3) -> set:
            idx = np.asarray(idx, dtype=np.int64)
            sel = idx if idx.size <= cap else np.random.default_rng(7).choice(idx, cap, replace=False)
            a = np.asarray(_d[np.sort(sel)], dtype=np.float64)
            a = a / np.maximum(np.sqrt(np.mean(a**2, axis=1, keepdims=True)), 1e-30)
            q = np.round(np.log10(np.abs(a) + 1e-12), decimals) * np.sign(a)
            return {hashlib.blake2b(row.tobytes(), digest_size=16).hexdigest() for row in q}
        k_tr, k_te = _keys(splits["train"]), _keys(splits["test"])
        inter = len(k_tr & k_te)
        frac = inter / max(len(k_te), 1)
        n_tr, n_te = _keys(splits["train"], decimals=2), _keys(splits["test"], decimals=2)
        n_inter = len(n_tr & n_te)
        n_frac = n_inter / max(len(n_te), 1)
        out["cross_split_duplicate_shapes"] = int(inter)
        out["cross_split_duplicate_fraction"] = float(frac)
        out["cross_split_near_duplicate_fraction"] = float(n_frac)
        out["split_leakage_pass"] = bool(frac <= 0.0 and n_frac <= 0.001)
        out["split_leakage_criterion"] = "exact duplicate fraction == 0 and near-duplicate < 0.1%"
        out["sampled_train_rows"] = int(min(len(splits["train"]), 20000))
        out["sampled_test_rows"] = int(min(len(splits["test"]), 20000))
        out["sample_coverage_test"] = float(out["sampled_test_rows"] / max(len(splits["test"]), 1))
        out["provenance_limitation"] = (
            "content digests only: the cache carries no source profile/anchor/parent identifiers, "
            "so curves interpolated from a shared anchor are not detectable by this probe [P1-8]"
        )
        _g = getattr(clean_cache, "paired_group_id", None)
        if _g is not None:
            _g = np.asarray(_g, dtype=np.int64)
            _gs = {k: set(np.unique(_g[np.asarray(splits[k], dtype=np.int64)]).tolist())
                      for k in ("train", "val", "test")}
            _st = {"train_val": len(_gs["train"] & _gs["val"]),
                      "train_test": len(_gs["train"] & _gs["test"]),
                      "val_test": len(_gs["val"] & _gs["test"])}
            out["profile_groups_in_two_splits"] = _st
            out["station_curve_shape_overlap_test_in_train"] = float(frac)
            out["split_leakage_pass"] = bool(sum(_st.values()) == 0)
            out["split_leakage_criterion"] = ("paired data: no group of near-identical clean profiles in more than one "
                                              "split; the station-curve shape overlap is reported, not gated")
        global _LAST_SPLIT_AUDIT
        _LAST_SPLIT_AUDIT = dict(out)
        if _g is not None:
            (logger.info if out["split_leakage_pass"] else logger.error)(
                "SPLIT LEAKAGE BY PROFILE: %s | groups of near-identical clean profiles in two splits: "
                "train-val %d, train-test %d, val-test %d | station curves of the sampled test rows whose shape also "
                "occurs in training: %.2f%% (%.2f%% at the coarser quantisation). Different forward models can share a "
                "station curve, for example where no conductor reaches the station, so that overlap is reported and "
                "not gated.",
                "PASS" if out["split_leakage_pass"] else "FAIL", _st["train_val"], _st["train_test"],
                _st["val_test"], 100.0 * frac, 100.0 * n_frac,
            )
        elif frac > 0.005:
            logger.error(
                "CROSS-SPLIT DUPLICATE SHAPES : %d shapes (%.2f%% of the sampled test "
                "curves) appear identically in the training split. The held-out metrics are then "
                "partly a memorization score, not generalization, and the 'real neighbour' "
                "assumption behind the contiguous grouping is not supported by the data. Supply "
                "profile/station metadata, or raise --field_group_size / --split_group_size.",
                inter, 100.0 * frac,
            )
        else:
            logger.info(
                "Cross-split duplicate probe: %d identical shapes (%.3f%% of sampled "
                "test curves) -- the contiguous-block grouping holds up empirically.",
                inter, 100.0 * frac,
            )
    except Exception as _exc:
        logger.error(
            "Cross-split duplicate probe FAILED to run: %r. A held-out claim that "
            "cannot be audited is not a held-out claim: split_leakage_pass=False.", _exc,
        )
        out = {"split_leakage_pass": False, "reason": repr(_exc),
               "cross_split_duplicate_fraction": float("nan"),
               "cross_split_near_duplicate_fraction": float("nan")}
    globals()["_LAST_SPLIT_AUDIT"] = dict(out)
    return out


def audit_split_populations(
    clean_cache: "CleanCacheInfo",
    splits: Mapping[str, np.ndarray],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Covariate-shift audit between splits."""
    d = clean_cache.descriptors
    if d is None:
        return cross_split_duplicate_probe(clean_cache, splits, logger)
    report: Dict[str, Any] = {"n": {k: int(len(v)) for k, v in splits.items() if not k.startswith("_")}}
    report.update(cross_split_duplicate_probe(clean_cache, splits, logger))
    flagged: List[str] = []
    tr = d[splits["train"]]
    for name_i, name in enumerate(CURVE_DESCRIPTOR_NAMES):
        a = tr[:, name_i]
        a = a[np.isfinite(a)]
        entry: Dict[str, Any] = {
            "train_mean": float(np.mean(a)) if a.size else float("nan"),
            "train_std": float(np.std(a)) if a.size else float("nan"),
        }
        for split_name in ("val", "test"):
            b = d[splits[split_name]][:, name_i]
            b = b[np.isfinite(b)]
            if a.size < 3 or b.size < 3:
                continue
            pooled = math.sqrt(0.5 * (float(np.var(a)) + float(np.var(b)))) or 1e-12
            smd = (float(np.mean(b)) - float(np.mean(a))) / pooled
            ks = _ks_statistic(a, b)
            g_tr = max(len(splits["_train_groups"]), 1)
            g_ot = max(len(splits["_%s_groups" % split_name]), 1)
            ks_crit = 1.63 * math.sqrt(1.0 / g_tr + 1.0 / g_ot)
            entry["%s_mean" % split_name] = float(np.mean(b))
            entry["%s_smd" % split_name] = float(smd)
            entry["%s_ks" % split_name] = ks
            entry["%s_ks_critical_001" % split_name] = ks_crit
            if abs(smd) > 0.25 and np.isfinite(ks) and ks > ks_crit:
                flagged.append(
                    "%s/%s (SMD %+.2f, KS %.2f > crit %.2f)" % (name, split_name, smd, ks, ks_crit)
                )
        report[name] = entry
    report["flagged"] = flagged
    logger.info(
        "SPLIT POPULATION AUDIT | groups: train=%d val=%d test=%d (group size %d) | curves: %d/%d/%d",
        len(splits["_train_groups"]), len(splits["_val_groups"]), len(splits["_test_groups"]),
        int(splits["_group_size"][0]), len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    for name_i, name in enumerate(CURVE_DESCRIPTOR_NAMES):
        e = report.get(name, {})
        if "val_smd" in e:
            logger.info(
                "  %-34s train %+8.3f | val %+8.3f (SMD %+5.2f, KS %.2f) | test %+8.3f (SMD %+5.2f)",
                name, e["train_mean"], e["val_mean"], e["val_smd"], e["val_ks"],
                e.get("test_mean", float("nan")), e.get("test_smd", float("nan")),
            )
    if flagged:
        logger.warning(
            "SPLIT POPULATION MISMATCH on: %s. The validation curves are drawn from a different "
            "population than the training curves, so a raised validation error is a SPLIT artefact "
            "before it is anything else. Raise --split_min_groups, or keep --split_stratify on. "
            "When the drop-based de-duplication above dropped a large share of the "
            "held-out rows, THAT is the mechanism and no split parameter will fix it: dedup "
            "removes the held-out copies of shapes that also occur in train, so what survives "
            "in val/test is precisely the sparse corner of the library that train under-"
            "represents -- a harder population, scored against the same absolute gate. Compare "
            "the held-out error with the per-SNR stacked identification ceiling before reading "
            "a gap as a model defect.",
            "; ".join(flagged),
        )
    else:
        logger.info(
            "  Populations match across splits (no descriptor is both shifted by |SMD|>0.25 and "
            "KS-significant at the GROUP-level critical value), so a validation gap cannot be blamed "
            "on the split."
        )
    if len(splits["_val_groups"]) < 6:
        logger.warning(
            "Only %d validation GROUPS. Curves inside a group are near-duplicates, so the effective "
            "validation sample size is the GROUP count, not the curve count: this metric has very wide "
            "error bars whatever it says. Enlarge the library or lower --split_group_size.",
            len(splits["_val_groups"]),
        )
    return report


def compute_reliability_target_np(
    noisy: np.ndarray,
    clean: np.ndarray,
    kappa: float,
    window: int,
    eps: float = 1e-8,
) -> np.ndarray:
    half = window // 2
    padded = np.pad(clean**2, (half, half), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    local_energy = np.sqrt(np.convolve(padded, kernel, mode="valid"))
    rel_error = np.abs(noisy - clean) / np.maximum(local_energy, eps)
    q = np.exp(-kappa * rel_error)
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def impute_zero_gates_row(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replace EXACT-zero gates of one observation by log-linear interpolation of |v| between the nearest finite
    non-zero gates (sign of the nearer neighbour)
    """
    v = np.asarray(v, dtype=np.float64)
    zero = (v == 0.0)
    if not zero.any():
        return v, zero
    ok = np.isfinite(v) & (~zero)
    if ok.sum() < 2:
        return v, zero
    idx = np.arange(v.shape[0], dtype=np.float64)
    lg = np.interp(idx, idx[ok], np.log(np.abs(v[ok])))
    _ki = np.flatnonzero(ok)
    if _ki.size >= 2:
        _a0, _a1 = _ki[0], _ki[1]
        _sl0 = (np.log(abs(v[_a1])) - np.log(abs(v[_a0]))) / max(float(_a1 - _a0), 1.0)
        left = idx < _a0
        lg[left] = np.log(abs(v[_a0])) + _sl0 * (idx[left] - _a0)
        _b1, _b0 = _ki[-1], _ki[-2]
        _sl1 = (np.log(abs(v[_b1])) - np.log(abs(v[_b0]))) / max(float(_b1 - _b0), 1.0)
        right = idx > _b1
        lg[right] = np.log(abs(v[_b1])) + _sl1 * (idx[right] - _b1)
    sgn = np.sign(np.interp(idx, idx[ok], np.sign(v[ok])))
    sgn[sgn == 0] = 1.0
    out = v.copy()
    out[zero] = sgn[zero] * np.exp(lg[zero])
    return out, zero


def audit_gate_columns(clean_mm: np.ndarray, noisy_mm: Optional[np.ndarray], times: Optional[np.ndarray],
                       province_id: Optional[np.ndarray], late_start: int, logger: logging.Logger,
                       max_rows: int = 120000) -> Dict[str, Any]:
    """Per-gate plausibility of the reference (and observation) table."""
    out: Dict[str, Any] = {}
    try:
        n = int(clean_mm.shape[0])
        step = max(1, n // max_rows)
        rows = np.arange(0, n, step)
        c = np.asarray(clean_mm[rows], dtype=np.float64)
        T = c.shape[1]
        ls = int(late_start)
        pid = np.asarray(province_id)[rows] if province_id is not None else np.zeros(rows.size, dtype=np.int64)
        if times is not None:
            t = np.asarray(times, dtype=np.float64).reshape(-1)
            mono = bool(np.all(np.diff(t) > 0))
            tp = t[t > 0]
            ratio = (tp[1:] / tp[:-1]) if tp.size > 1 else np.array([1.0])
            logger.info("GATE-TIME AXIS | %d gates | %.3e .. %.3e s%s | monotone %s | step ratio of the positive "
                        "gates median %.3f min %.3f max %.3f | %s",
                        t.size, float(t.min()), float(t.max()),
                        " (first gate at the time origin)" if gate_axis_has_origin(t) else "", mono,
                        float(np.median(ratio)), float(ratio.min()), float(ratio.max()), " ".join("%.2e" % v for v in t))
            out["time_monotone"] = mono
        pos = c > 0
        viol = np.zeros(T)
        dist = np.zeros(T, dtype=np.int64)
        plateau = np.zeros(T)
        areas = np.unique(pid)
        n_areas = max(1, areas.size)
        for g in range(T):
            col = c[:, g]
            dist[g] = int(np.unique(np.round(col, 12)).size)
            if g > 0:
                m = pos[:, g] & pos[:, g - 1]
                viol[g] = float(np.mean(col[m] > c[m, g - 1])) if m.any() else 0.0
        for a in areas:
            idx = np.flatnonzero(pid == a)
            if idx.size < 5:
                continue
            blk = c[idx]
            d = np.abs(np.diff(blk, axis=0)) <= 1e-9 * np.maximum(np.abs(blk[:-1]), 1e-30)
            for g in range(T):
                run = 0
                hit = False
                for v in d[:, g]:
                    run = run + 1 if v else 0
                    if run >= 4:
                        hit = True
                        break
                if hit:
                    plateau[g] += 1.0
        plateau /= n_areas
        sus = [g for g in range(T) if (plateau[g] > 0.02 or viol[g] > 0.40 or dist[g] < 0.001 * rows.size)]
        logger.info("GATE COLUMNS (clean reference, %d rows, %d areas) | late window gates %d..%d | "
                    "plateau(>=5 stations constant) share by gate: %s | decay-violation share (gate>previous): %s | distinct values: %s",
                    int(rows.size), int(n_areas), ls, T - 1,
                    " ".join("%d:%.1f%%" % (g, 100 * plateau[g]) for g in range(T) if plateau[g] > 0) or "none",
                    " ".join("%d:%.0f%%" % (g, 100 * viol[g]) for g in range(1, T) if viol[g] > 0.05) or "none",
                    " ".join("%d:%d" % (g, dist[g]) for g in range(T)))
        if sus:
            logger.warning("SUSPICIOUS GATE COLUMN(S) %s in the CLEAN reference: constant plateaus across stations, "
                           "decay violations or too few distinct values. A column that is not a physical gate (or a floored "
                           "reference) makes every late error against it meaningless -- check the source table header for "
                           "these columns.", sus)
        out.update({"suspicious_gates": sus, "plateau": plateau.tolist(), "decay_violation": viol.tolist()})
        if noisy_mm is not None:
            x = np.asarray(noisy_mm[rows], dtype=np.float64)
            same = np.mean(np.abs(x[:, ls:] - c[:, ls:]) <= 1e-9 * np.maximum(np.abs(c[:, ls:]), 1e-30))
            logger.info("observation == reference on %.2f%% of late gate values (a large share means the 'noisy' late "
                        "window was copied from the reference, i.e. no measurement there)", 100 * float(same))
            out["late_copied_share"] = float(same)
            e0 = max(1, ls // 2)
            num = np.sqrt(np.sum((x[:, :e0] - c[:, :e0]) ** 2, axis=1))
            den = np.sqrt(np.sum(c[:, :e0] ** 2, axis=1))
            ee = num / np.maximum(den, 1e-300)
            bad = float(np.mean(ee > 1.0))
            out["early_misaligned_share"] = bad
            logger.info("PAIRING ALIGNMENT | early-band relative error observation vs reference: P50 %.2f%% P90 %.2f%% "
                        "P99 %.2f%% | rows > 100%% (misaligned or garbage): %.3f%%", 100 * float(np.percentile(ee, 50)),
                        100 * float(np.percentile(ee, 90)), 100 * float(np.percentile(ee, 99)), 100 * bad)
            if bad > 0.01:
                logger.warning("%.2f%% of rows disagree with their reference already in the EARLY band: the pairing "
                               "(sample_id, depth) is wrong for those rows -- this is a data-pipeline fault, not noise. ", 100 * bad)
            drift_rows = 0
            tot_rows = 0
            for a in areas:
                idx = np.flatnonzero(pid == a)
                if idx.size < 8:
                    continue
                nz_a = x[idx][:, ls:] - c[idx][:, ls:]
                sidx = np.arange(idx.size, dtype=np.float64)
                sidx = (sidx - sidx.mean()) / max(sidx.std(), 1e-9)
                nzc = nz_a - nz_a.mean(axis=0, keepdims=True)
                sd = np.maximum(nzc.std(axis=0), 1e-30)
                r = (nzc * sidx[:, None]).mean(axis=0) / sd
                if int(np.sum(np.abs(r) > 0.7)) >= 5:
                    drift_rows += int(idx.size)
                tot_rows += int(idx.size)
            share = drift_rows / max(tot_rows, 1)
            out["lateral_drift_area_share"] = float(share)
            logger.info("LATERAL DRIFT | %.1f%% of rows sit in areas whose late-band NOISE is linearly correlated "
                        "with station (|r| > 0.7 on >= 5 late gates); a data statistic only.", 100 * share)
    except Exception as exc:
        logger.warning("gate-column audit skipped: %r", exc)
    return out


def audit_paired_zero_gates(noisy_mm: np.ndarray, clean_mm: Optional[np.ndarray], late_start: int,
                            province_id: Optional[np.ndarray], logger: logging.Logger,
                            max_rows: int = 200000) -> Dict[str, float]:
    """Startup integrity audit of the measured pair (see header)."""
    try:
        n = int(noisy_mm.shape[0])
        step = max(1, n // max_rows)
        rows = np.arange(0, n, step)
        x = np.asarray(noisy_mm[rows], dtype=np.float64)
        T = x.shape[1]
        zero = (x == 0.0)
        med = np.median(np.abs(x[np.isfinite(x) & (x != 0.0)])) if np.isfinite(x).any() else 1.0
        gmed = np.array([np.median(np.abs(x[:, g][(x[:, g] != 0) & np.isfinite(x[:, g])])) if ((x[:, g] != 0) & np.isfinite(x[:, g])).any() else med for g in range(T)])
        near = (np.abs(x) < 1e-6 * gmed[None, :]) & (~zero)
        ls = int(late_start)
        stats = {"rows_scanned": float(x.shape[0]),
                 "zero_frac_early": float(zero[:, :ls // 2].mean()), "zero_frac_mid": float(zero[:, ls // 2:ls].mean()),
                 "zero_frac_late": float(zero[:, ls:].mean()), "near_zero_frac_late": float(near[:, ls:].mean()),
                 "rows_with_any_zero_late": float((zero[:, ls:].any(axis=1)).mean())}
        if clean_mm is not None:
            c = np.asarray(clean_mm[rows], dtype=np.float64)
            stats["late_sign_flip_frac"] = float(((np.sign(x[:, ls:]) * np.sign(c[:, ls:])) < 0).mean())
        worst = ""
        if province_id is not None and zero[:, ls:].any():
            pid = np.asarray(province_id)[rows]
            zr = zero[:, ls:].mean(axis=1)
            ids = np.unique(pid)
            fr = np.array([zr[pid == a].mean() for a in ids])
            top = np.argsort(-fr)[:5]
            worst = " | worst areas (late-gate zero fraction): " + ", ".join("%d:%.1f%%" % (int(ids[i]), 100 * fr[i]) for i in top)
        msg = ("PAIR INTEGRITY | %d rows scanned | exact-zero gates early/mid/late = %.3f%% / %.3f%% / %.3f%% "
               "| rows with any exact-zero late gate = %.2f%% | near-zero (<1e-6 x gate median) late = %.3f%%%s%s. ") % (
            int(x.shape[0]), 100 * stats["zero_frac_early"], 100 * stats["zero_frac_mid"], 100 * stats["zero_frac_late"],
            100 * stats["rows_with_any_zero_late"], 100 * stats["near_zero_frac_late"],
            (" | late sign disagreement with clean = %.2f%%" % (100 * stats["late_sign_flip_frac"])) if "late_sign_flip_frac" in stats else "",
            worst)
        (logger.warning if stats["rows_with_any_zero_late"] > 0.005 else logger.info)(msg)
        if stats["rows_with_any_zero_late"] > 0.005:
            logger.warning("Exact zeros are treated as GAPS: imputed row-locally (log-linear between "
                           "non-zero gates) and marked unobserved (reliability target 0) in every observation the "
                           "model receives (impute_zero_gates=True). If the raw table used 0 for 'no reading', this "
                           "is the data-reading defect behind the fan-to-zero curves in the section figures. ")
        return stats
    except Exception as exc:
        logger.warning("pair integrity audit skipped: %r", exc)
        return {}


def _draw_anchor(rng: np.random.Generator, a: int, b: int, P: int, p_edge: float) -> int:
    hi = int(b - P + 2)
    if hi - int(a) <= 1:
        return int(a)
    if p_edge > 0.0 and float(rng.random()) < float(p_edge):
        return int(a) if float(rng.random()) < 0.5 else int(b - P + 1)
    return int(rng.integers(a, hi))


@torch.no_grad()
def build_neighbor_indices(n: int, k: int) -> np.ndarray:
    """For each of n position-sorted stations, return the k nearest other station indices (boundary windows shift
    inward).
    """
    if n <= 1 or k <= 0:
        return np.zeros((max(n, 0), max(k, 0)), dtype=np.int64)
    idx = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        lo = i - (k // 2)
        lo = max(0, min(lo, n - 1 - k)) if n - 1 - k >= 0 else 0
        picks: List[int] = []
        offset = 0
        while len(picks) < k:
            for cand in (i - offset, i + offset):
                if offset == 0:
                    continue
                if 0 <= cand < n and cand != i and cand not in picks:
                    picks.append(cand)
                    if len(picks) == k:
                        break
            offset += 1
            if offset > n:
                break
        while len(picks) < k:
            picks.append(picks[-1] if picks else (0 if i != 0 else min(1, n - 1)))
        idx[i] = np.asarray(picks[:k], dtype=np.int64)
    return idx


def build_province_neighbor_rows(
    province_id: np.ndarray,
    station: np.ndarray,
    k: int,
) -> np.ndarray:
    """For every cache row, the absolute cache rows of its k nearest stations within the same province, by station
    order, boundaries shifting inward.
    """
    pid = np.asarray(province_id, dtype=np.int64)
    st = np.asarray(station, dtype=np.float64)
    n = pid.size
    out = np.empty((n, max(k, 0)), dtype=np.int64)
    if k <= 0 or n == 0:
        return out
    for p in np.unique(pid):
        rows = np.nonzero(pid == p)[0]
        if int(p) < 0:
            out[rows] = rows[:, None]
            continue
        order = rows[np.argsort(st[rows], kind="stable")]
        local = build_neighbor_indices(order.size, k)
        out[order] = order[local]
    return out


NEIGHBOR_GEOMETRY_FEATURES = 4


def station_median_spacing(station: np.ndarray) -> float:
    """The work area's own length unit: median |gap| between consecutive position-sorted stations."""
    st = np.sort(np.asarray(station, dtype=np.float64).reshape(-1))
    d = np.abs(np.diff(st))
    d = d[d > 0]
    return float(np.median(d)) if d.size else 1.0


def neighbor_geometry_from_stations(
    station: np.ndarray,
    nb_idx: np.ndarray,
    add_stack: bool,
    spacing: Optional[float] = None,
) -> np.ndarray:
    """Per-slot geometry features [n, K(+1), 4] from true station coordinates: signed offset in median-spacing units
    (clipped to +-16), |offset|, is_real=1.
    """
    st = np.asarray(station, dtype=np.float64).reshape(-1)
    idx = np.asarray(nb_idx, dtype=np.int64)
    m = float(spacing) if spacing else station_median_spacing(st)
    m = m if m > 0 else 1.0
    off = np.clip((st[idx] - st[:, None]) / m, -16.0, 16.0)
    n, k = idx.shape
    kt = k + (1 if add_stack else 0)
    geo = np.zeros((n, kt, NEIGHBOR_GEOMETRY_FEATURES), dtype=np.float32)
    geo[:, :k, 0] = off
    geo[:, :k, 1] = np.abs(off)
    geo[:, :k, 2] = 1.0
    if add_stack:
        geo[:, k, 3] = 1.0
    return geo


def surrogate_ring_geometry(k: int, add_stack: bool, is_real: float = 0.0) -> np.ndarray:
    """Rank-surrogate geometry (+1, -1, +2, -2, ... spacing units) for slots with no metric coordinates: the
    synthetic shared-shape scheme and any caller that supplies neighbors without stations.
    """
    kt = k + (1 if add_stack else 0)
    geo = np.zeros((kt, NEIGHBOR_GEOMETRY_FEATURES), dtype=np.float32)
    for j in range(k):
        off = float(((j // 2) + 1) * (1 if j % 2 == 0 else -1))
        geo[j, 0] = off
        geo[j, 1] = abs(off)
        geo[j, 2] = float(is_real)
    if add_stack:
        geo[k, 3] = 1.0
    return geo


def province_neighbor_geometry(clean_cache: "CleanCacheInfo", cfg: "Config") -> Optional[np.ndarray]:
    """Per-row geometry table [rows, K, 2] = (signed offset, is_real) for the province library, computed with each
    PROVINCE'S own median spacing and cached like the neighbor-row table.
    """
    if getattr(clean_cache, "province_id", None) is None:
        return None
    k = int(cfg.data.num_neighbors)
    if k <= 0:
        return None
    cached = getattr(clean_cache, "_real_nb_geo", None)
    if cached is not None and cached.shape == (int(clean_cache.rows), k, 2):
        return cached
    rows_tbl = province_real_neighbor_rows(clean_cache, cfg)
    if rows_tbl is None:
        return None
    station = np.load(clean_cache.station_path)
    pid = np.asarray(clean_cache.province_id, dtype=np.int64)
    st = np.asarray(station, dtype=np.float64).reshape(-1)
    out = np.zeros((pid.size, k, 2), dtype=np.float32)
    ring = surrogate_ring_geometry(k, add_stack=False, is_real=0.0)
    for p in np.unique(pid):
        rows = np.nonzero(pid == p)[0]
        if int(p) < 0:
            out[rows, :, 0] = ring[None, :, 0]
            out[rows, :, 1] = 0.0
            continue
        m = station_median_spacing(st[rows])
        off = np.clip((st[rows_tbl[rows]] - st[rows, None]) / (m if m > 0 else 1.0), -16.0, 16.0)
        out[rows, :, 0] = off.astype(np.float32)
        out[rows, :, 1] = 1.0
    try:
        clean_cache._real_nb_geo = out
    except Exception:
        pass
    return out


def province_real_neighbor_rows(clean_cache: "CleanCacheInfo", cfg: "Config") -> Optional[np.ndarray]:
    """The per-row real-neighbour table for province libraries, built once and cached on the CleanCacheInfo object;
    None for the single-file library.
    """
    if getattr(clean_cache, "province_id", None) is None:
        return None
    k = int(cfg.data.num_neighbors)
    if k <= 0:
        return None
    cached = getattr(clean_cache, "_real_nb_rows", None)
    if cached is not None and cached.shape == (int(clean_cache.rows), k):
        return cached
    station = np.load(clean_cache.station_path)
    table = build_province_neighbor_rows(clean_cache.province_id, station, k)
    try:
        clean_cache._real_nb_rows = table
    except Exception:
        pass
    return table
