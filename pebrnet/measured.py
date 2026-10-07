# PEBR-Net, the prior- and evidence-bounded reconstruction network.
# MIT License, see LICENSE.
"""Hooks for a measured profile, inactive while no measured profile is configured.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

def load_field_matrix(
    path: Path,
    target_gates: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """Measured-profile hook; inactive while no measured profile is configured."""
    raise RuntimeError("no measured profile is configured (paths.field_csv)")


def read_noise_library(noise_dir: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Measured-profile hook; inactive while no measured profile is configured."""
    return None


def build_measured_noise_cache(noise_dir: Path, target_time: np.ndarray,
                               cache_dir: Path, cfg: "Config",
                               logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Measured-profile hook; inactive while no measured profile is configured."""
    return None


def build_survey_profile_cache(field_station: np.ndarray, field_native: np.ndarray, field_headers: Sequence[Any],
                                   field_model: np.ndarray, clean_cache: "CleanCacheInfo", cache_dir: Path,
                                   cfg: "Config", logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Measured-profile hook; inactive while no measured profile is configured."""
    return None


def adopt_survey_record(model: Optional[nn.Module], state: Mapping[str, Any]) -> None:
    """Measured-profile hook; inactive while no measured profile is configured."""
    return None


def begin_survey_training_record(model: nn.Module, cfg: "Config", clean_cache: "CleanCacheInfo",
                                     logger: Optional[logging.Logger] = None) -> None:
    """Measured-profile hook; inactive while no measured profile is configured."""
    return None


def survey_profile_probe(model: nn.Module, cfg: "Config", clean_cache: "CleanCacheInfo",
                             device: torch.device) -> Optional[Dict[str, float]]:
    """Measured-profile hook; inactive while no measured profile is configured."""
    return None


def ensure_family_completion(model: nn.Module, cfg: Config, clean_cache: CleanCacheInfo,
                                 logger: Optional[logging.Logger] = None) -> bool:
    """Measured-profile hook; inactive while no measured profile is configured."""
    return None


_LAST_FIELD_PREFLIGHT: Dict[str, Any] = {"amplitude_alignment_pass": None,
                                         "reason": "field preflight not run"}


def field_profile_present(cfg: "Config") -> bool:
    """True when the run has a measured profile (paths.field_csv non-empty); --no_field empties it."""
    return bool(str(getattr(getattr(cfg, "paths", None), "field_csv", "") or "").strip())


def unified_measured_axis(cfg: Config, gates: int, logger: logging.Logger) -> Optional[np.ndarray]:
    """The measured gate axis with `gates` entries, in seconds, or None when the mode is off."""
    if not bool(getattr(cfg.data, "unify_axis_to_measured", False)):
        return None
    tbl = str(getattr(cfg.paths, "field_gate_table", "") or "")
    if tbl and Path(tbl).exists():
        vals = []
        for tok in re.split(r"[\s,;]+", Path(tbl).read_text(encoding="utf-8-sig").strip()):
            if tok:
                try:
                    vals.append(float(tok))
                except ValueError:
                    continue
        ax = axis_seconds_strict(vals, str(getattr(cfg.data, "field_gate_time_unit", "ms")))
        if ax.size != int(gates):
            ax2 = np.geomspace(float(ax[0]), float(ax[-1]), int(gates))
            logger.info("unified axis = %d log-spaced gates over the instrument table's span %.4g..%.4g s (table has %d entries).",
                        int(gates), float(ax2[0]), float(ax2[-1]), ax.size)
            return ax2
        logger.info("unified axis = instrument gate table %s (%d gates).", tbl, ax.size)
        return ax
    lib = read_noise_library(Path(str(getattr(cfg.paths, "noise_library_dir", "") or ".")), logger)
    if not lib or "gate_ms" not in lib:
        raise ValueError("no instrument gate table and the measured-noise library could not be read: "
                         "the measured span is unknown.")
    src = axis_seconds_strict(np.asarray(lib["gate_ms"], dtype=np.float64), "ms")
    lo_s, hi_s = float(src[0]), float(src[-1])
    try:
        _xc = Path(str(getattr(cfg.paths, "paired_dir", ""))) / "clean.csv"
        if _xc.exists():
            with _xc.open("r", encoding="utf-8-sig") as f:
                _hdr = f.readline().strip().split(",")
            _tv = []
            for tok in _hdr[1:]:
                try:
                    _tv.append(float(tok))
                except ValueError:
                    pass
            if len(_tv) >= 2 and all(np.isfinite(_tv)) and _tv[0] > 0 and _tv[-1] > _tv[0]:
                lo_s = max(lo_s, _tv[0] * 1e-3)
                hi_s = min(hi_s, _tv[-1] * 1e-3)
                logger.info("extra clean.csv span %.4g..%.4g s narrows the unified span.", _tv[0] * 1e-3, _tv[-1] * 1e-3)
    except Exception as _e:
        logger.warning("extra clean.csv header not readable for the span: %r", _e)
    ax = np.geomspace(lo_s, hi_s, int(gates))
    logger.info("unified axis = %d log-spaced gates over the measured span %.4g..%.4g s "
                "(no instrument table given; pass --field_gate_table to use the exact instrument centres).",
                int(gates), float(ax[0]), float(ax[-1]))
    return ax


def field_to_model_axis(original: np.ndarray, headers: Sequence[Any], cfg: Config, target_time: np.ndarray,
                            logger: logging.Logger) -> np.ndarray:
    """Measured-profile hook; inactive while no measured profile is configured."""
    raise RuntimeError("no measured profile is configured (paths.field_csv)")


def _unit_declaration(cfg: Config, logger: logging.Logger) -> Dict[str, Any]:
    return {"pass": True, "missing_fields": [], "reason": "no measured profile is configured (paths.field_csv)"}


def field_preflight(
    cfg: Config,
    run_dir: Path,
    clean_cache: CleanCacheInfo,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Measured-profile hook; inactive while no measured profile is configured."""
    global _LAST_FIELD_PREFLIGHT
    out = {"amplitude_alignment_pass": None, "gate_ratio_tilt_pass": None, "segment_gain_pass": None,
           "unit_declaration_pass": True, "field_present": False, "reason": "no measured profile is configured (paths.field_csv)"}
    _LAST_FIELD_PREFLIGHT = out
    return out


def run_field_if_qualified(
    model: "PEBRNet",
    checkpoint_path: Optional[Path],
    cfg: Config,
    run_dir: Path,
    clean_cache: CleanCacheInfo,
    logger: logging.Logger,
    deployment: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Measured-profile hook; inactive while no measured profile is configured."""
    return None


def section_autopsy_pass(model: "PEBRNet", cfg: "Config", run_dir: Path, clean_cache: "CleanCacheInfo",
                             noise_cache: "NoiseCacheInfo", checkpoint_path: Path, logger: logging.Logger) -> int:
    """Measured-profile hook; inactive while no measured profile is configured."""
    raise RuntimeError("no measured profile is configured (paths.field_csv)")


def maybe_run_interactive_menu(args: argparse.Namespace) -> argparse.Namespace:
    """Unset options take their defaults (mode all, ablation full); a training run of the full network starts from
    random initialisation.
    """
    if args.mode is None:
        args.mode = "all"
    if args.ablation is None:
        args.ablation = "full"
    if args.mode in {"all", "train"} and args.ablation == "full" and not getattr(args, "init_from", ""):
        args.confirm_full_from_scratch = True
    return args
