# PEBR-Net, the prior- and evidence-bounded reconstruction network.
# MIT License, see LICENSE.
"""The gate-loss cohort.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

GATE_LOSS_OPERATORS: Tuple[str, ...] = ("full", "no_cdmr", "no_egsr")


GATE_LOSS_LABELS: Dict[str, str] = {
    "full": "Full PEBR-Net",
    "no_cdmr": "CDM-R removed",
    "no_egsr": "E-GSR removed",
    "noise_tail": "No continuation (noise-only tail)",
}


GATE_LOSS_COLORS: Dict[str, str] = {
    "full": OKABE_ITO["blue"],
    "no_cdmr": OKABE_ITO["green"],
    "no_egsr": OKABE_ITO["purple"],
    "noise_tail": OKABE_ITO["grey"],
}


def _core_model(model: Any) -> Any:
    m = model
    for _attr in ("module", "_orig_mod"):
        m = getattr(m, _attr, m)
    return m


def gate_times_ms(clean_cache: "CleanCacheInfo") -> np.ndarray:
    """The cache's gate times in milliseconds (the cache may hold seconds or milliseconds)."""
    t = np.asarray(clean_cache.target_time, dtype=np.float64).reshape(-1)
    return t * 1000.0 if (t.size and float(t[-1]) < 1.0) else t


def decay_law_descriptors(y: np.ndarray, t_ms: np.ndarray,
                              floor: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Eq. A13 per row: local decay exponent p(t) = -d ln y / d ln t and curvature kappa(t) = d p / d ln t,
    differences on the true (non-uniform) log-time axis.
    """
    yy = np.atleast_2d(np.asarray(y, dtype=np.float64))
    tt = np.asarray(t_ms, dtype=np.float64).reshape(-1)
    if floor is None:
        floor = 1e-6 * np.maximum(np.max(np.abs(yy), axis=1, keepdims=True), 1e-300)
    if gate_axis_has_origin(tt):
        p1, k1 = decay_law_descriptors(yy[:, 1:], tt[1:], floor)
        pad = np.full((yy.shape[0], 1), np.nan)
        return np.concatenate([pad, p1], axis=1), np.concatenate([pad, k1], axis=1)
    lt = np.log(tt)
    ly = np.log(np.maximum(yy, floor))
    p = -np.gradient(ly, lt, axis=1)
    return p, np.gradient(p, lt, axis=1)


def gate_loss_endpoints(pred: np.ndarray, ref: np.ndarray, t_ms: np.ndarray, cut: int, late_start: int,
                            kappa_floor: float = 0.05, keep_samples: bool = False) -> Dict[str, Any]:
    """Endpoints of one profile (rows = stations) whose gates from `cut` on were withheld."""
    yp = np.atleast_2d(np.asarray(pred, dtype=np.float64))
    yr = np.atleast_2d(np.asarray(ref, dtype=np.float64))
    T = int(yr.shape[1])
    W = np.arange(int(np.clip(cut, 1, T - 1)), T)
    L = np.arange(int(np.clip(late_start, 0, T - 1)), T)

    def _fro(cols: np.ndarray) -> float:
        return 100.0 * float(np.linalg.norm(yp[:, cols] - yr[:, cols]) / (np.linalg.norm(yr[:, cols]) + 1e-30))

    fl = 1e-6 * np.maximum(np.max(np.abs(yr), axis=1, keepdims=True), 1e-300)
    p_r, k_r = decay_law_descriptors(yr, t_ms, fl)
    p_h, k_h = decay_law_descriptors(yp, t_ms, fl)
    pos = yr > fl
    vp = pos.copy()
    vp[:, 1:] &= pos[:, :-1]
    vp[:, :-1] &= pos[:, 1:]
    vk = vp.copy()
    vk[:, 1:] &= vp[:, :-1]
    vk[:, :-1] &= vp[:, 1:]
    mp = vp[:, W] & np.isfinite(p_r[:, W]) & np.isfinite(p_h[:, W])
    mk = vk[:, W] & np.isfinite(k_r[:, W]) & np.isfinite(k_h[:, W])
    dp = (p_h - p_r)[:, W][mp]
    dk = (k_h - k_r)[:, W][mk]
    kr = k_r[:, W][mk]
    out: Dict[str, Any] = {
        "amplitude_nrmse": _fro(W),
        "late_nrmse": _fro(L),
        "global_nrmse": _fro(np.arange(T)),
        "slope_rmse": float(np.sqrt(np.mean(dp ** 2))) if dp.size else float("nan"),
        "structure_nrmse": (100.0 * float(np.linalg.norm(dk) / (np.linalg.norm(kr) + kappa_floor * math.sqrt(dk.size)))
                            if dk.size else float("nan")),
        "n_slope_samples": int(dp.size),
    }
    if keep_samples:
        out["dp"] = dp.astype(np.float32)
        out["p_ref"] = p_r[:, W][mp].astype(np.float32)
        out["p_hat"] = p_h[:, W][mp].astype(np.float32)
        out["kappa_ref"] = kr.astype(np.float32)
        out["kappa_hat"] = k_h[:, W][mk].astype(np.float32)
    return out


def _gate_loss_levels(cfg: "Config", T: int, t_ms: np.ndarray) -> List[Dict[str, Any]]:
    ref_g = max(1, int(getattr(cfg.runtime, "gate_loss_reference_gates", 31) or 31))
    levels: List[Dict[str, Any]] = []
    for k in sorted({int(v) for v in getattr(cfg.runtime, "gate_loss_withheld", (3, 6, 12, 18, 19, 22))}):
        n_w = int(np.clip(int(round(k * T / float(ref_g))), 1, T - 3))
        cut = T - n_w
        if any(int(lv["cut"]) == cut for lv in levels):
            continue
        levels.append({
            "key": "W%02d" % k, "withheld_of_reference": int(k), "reference_gates": ref_g,
            "withheld_gates": n_w, "cut": int(cut), "withheld_percent": 100.0 * n_w / float(T),
            "last_recorded_ms": float(t_ms[cut - 1]), "first_withheld_ms": float(t_ms[cut]),
        })
    return levels


def _gate_loss_dataset_meta(cfg: "Config") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        mp = Path(str(cfg.paths.paired_dir)) / "dataset_meta.json"
        if mp.is_file():
            with open(mp, "r", encoding="utf-8") as fh:
                for pr in json.load(fh).get("profiles", []):
                    out[_norm_sid(pr.get("sample_id"))] = pr
    except Exception:
        out = {}
    return out


def _norm_sid(v: Any) -> str:
    s = str(v).strip()
    try:
        f = float(s)
        if math.isfinite(f) and float(int(f)) == f:
            return str(int(f))
    except Exception:
        pass
    return s


def _area_sample_id(name: str) -> str:
    s = str(name)
    if "#sample_" in s:
        s = s.split("#sample_", 1)[1]
    elif "#" in s:
        s = s.split("#", 1)[1]
    return _norm_sid(s)


def run_gate_loss_suite(model: "PEBRNet", cfg: "Config", run_dir: Path, clean_cache: "CleanCacheInfo",
                        noise_cache: "NoiseCacheInfo", logger: logging.Logger, split: str = "test") -> Dict[str, Any]:
    """The gate-loss cohort of the manuscript (Sec. 5.2) on the held-out paired profiles."""
    if not getattr(clean_cache, "paired_noisy_path", ""):
        logger.info("Gate-loss suite skipped: not a paired run (no clean reference exists).")
        return {}
    t_start = time.time()
    core = _core_model(model)
    ensure_cdmr_calibration(model, cfg, clean_cache, logger)
    T = int(cfg.data.target_gates)
    ls = int(cfg.data.late_start_index)
    t_ms = gate_times_ms(clean_cache)
    if t_ms.size != T:
        raise ValueError("gate axis length %d != target_gates %d" % (t_ms.size, T))
    levels = _gate_loss_levels(cfg, T, t_ms)
    if not levels:
        logger.warning("Gate-loss suite: no valid level on a %d-gate axis.", T)
        return {}
    splits = build_splits(clean_cache, cfg)
    rows_all = np.asarray(splits.get(split, splits["test"]), dtype=np.int64)
    pid_all = np.asarray(clean_cache.province_id)
    areas = np.unique(pid_all[rows_all])
    if areas.size == 0:
        logger.warning("Gate-loss suite: the %s split is empty.", split)
        return {}
    cap = int(getattr(cfg.runtime, "gate_loss_profiles", 0) or 0)
    n_areas_total = int(areas.size)
    if 0 < cap < areas.size:
        pick = np.unique(np.round(np.linspace(0, areas.size - 1, cap)).astype(np.int64))
        areas = areas[pick]
        logger.info("Gate-loss suite scores %d of %d %s profiles (evenly spaced over the sorted profile "
                    "ids; --gate_loss_profiles 0 scores all).",
                    int(areas.size), n_areas_total, split)
    rows = np.sort(rows_all[np.isin(pid_all[rows_all], areas)])
    clean_mm = np.load(clean_cache.data_path, mmap_mode="r")
    noisy_mm = np.load(clean_cache.paired_noisy_path, mmap_mode="r")
    clean = np.asarray(clean_mm[rows], dtype=np.float64)
    noisy = np.asarray(noisy_mm[rows], dtype=np.float64)
    pid = pid_all[rows]
    names = list(clean_cache.province_names or [])
    meta = _gate_loss_dataset_meta(cfg)
    area_pos: Dict[int, np.ndarray] = {int(a): np.flatnonzero(pid == a) for a in areas}
    area_name = {int(a): (names[int(a)] if int(a) < len(names) else str(int(a))) for a in areas}
    Lc = np.arange(ls, T)
    area_info: Dict[int, Dict[str, Any]] = {}
    for a in areas:
        ia = area_pos[int(a)]
        c_l, n_l = clean[ia][:, Lc], noisy[ia][:, Lc]
        snr_real = 20.0 * math.log10(max(float(np.linalg.norm(c_l)), 1e-300)
                                     / max(float(np.linalg.norm(n_l - c_l)), 1e-300))
        rec = meta.get(_area_sample_id(area_name[int(a)]), {})
        dsnr = rec.get("design_late_snr_db", None)
        area_info[int(a)] = {
            "name": area_name[int(a)], "n_stations": int(ia.size), "input_late_snr_db": float(snr_real),
            "design_late_snr_db": (float(dsnr) if dsnr is not None else float("nan")),
            "regime": str(rec.get("regime", "")),
        }
    logger.info("GATE-LOSS SUITE | %d %s profiles, %d stations, %d gates (late window from gate %d) | levels: %s "
                "| operators: %s + no-continuation baseline.",
                int(areas.size), split, int(rows.size), T, ls + 1,
                ", ".join("%d/%d gates = %.1f%%" % (lv["withheld_gates"], T, lv["withheld_percent"]) for lv in levels),
                ", ".join(GATE_LOSS_LABELS[o] for o in GATE_LOSS_OPERATORS))

    def _fwd(cut: Optional[int], protocol: str = "mask") -> np.ndarray:
        return _paired_test_forward(model, cfg, clean_cache, noise_cache, rows, logger,
                                    tail_cut=cut, tail_protocol=protocol)

    sw = (bool(getattr(core, "_ablate_cdmr", False)), bool(getattr(core, "_ablate_egsr", False)))
    try:
        egsr_scale = float(getattr(core, "event_expression_scale", torch.zeros(())).item())
    except Exception:
        egsr_scale = float("nan")
    if not (egsr_scale > 0.0):
        logger.warning("E-GSR is not expressed by this checkpoint (event_expression_scale = %.3f: the "
                       "structural residual writes into the output only once the event detector has cleared its AUROC "
                       "qualification), so the 'E-GSR removed' operator is the full operator bit for bit.",
                       egsr_scale)
    dep_share: List[float] = []
    if getattr(core, "gl_departure_logit", None) is not None and bool(getattr(core.cfg, "gate_loss_departure", True)):
        with torch.no_grad():
            dep_share = [float(v) for v in torch.sigmoid(core.gl_departure_logit.detach().float()).cpu().numpy()]
        _kq = [k for k in (0, int(T) // 10, int(T) // 5, int(2 * T) // 5, int(3 * T) // 5, int(7 * T) // 10) if k < len(dep_share)]
        logger.info("WITHHELD-GATE DEPARTURE SHARE g(k) (k = position in the withheld window) | %s | mean "
                    "over each level's window: %s | E-GSR is added after the departure at unit weight. ",
                    " | ".join("k=%d %.3f" % (k, dep_share[k]) for k in _kq),
                    ", ".join("%s %.3f" % (str(lv["key"]), float(np.mean(dep_share[:max(1, int(lv["withheld_gates"]))])))
                              for lv in levels))
    ops: List[str] = list(GATE_LOSS_OPERATORS)
    if bool(getattr(cfg.runtime, "gate_loss_noise_tail_baseline", True)):
        ops.append("noise_tail")
    per_area: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    full_preds: Dict[str, np.ndarray] = {}
    full_record: Dict[int, Dict[str, float]] = {}
    ic_mode = str(getattr(core.cfg, "informative_continuation", "off")).lower()
    full_record_alt: Dict[int, Dict[str, float]] = {}
    regen_alone: Dict[str, float] = {}
    try:
        core._ablate_cdmr = False
        core._ablate_egsr = False
        pred0 = _fwd(None)
        for a in areas:
            ia = area_pos[int(a)]
            full_record[int(a)] = gate_loss_endpoints(pred0[ia], clean[ia], t_ms, T - 1, ls)
        core.cfg.informative_continuation = "off" if ic_mode == "on" else "on"
        try:
            pred0_alt = _fwd(None)
        finally:
            core.cfg.informative_continuation = ic_mode
        for a in areas:
            ia = area_pos[int(a)]
            full_record_alt[int(a)] = gate_loss_endpoints(pred0_alt[ia], clean[ia], t_ms, T - 1, ls)
        _fr_on = [(full_record if ic_mode == "on" else full_record_alt)[int(a)]["late_nrmse"] for a in areas]
        _fr_off = [(full_record_alt if ic_mode == "on" else full_record)[int(a)]["late_nrmse"] for a in areas]
        logger.info("FULL RECORD (0%% withheld) | median per-profile late NRMSE: informative continuation ON "
                    "%.2f%% | OFF %.2f%% | profiles better ON: %d of %d (in force: %s).",
                    float(np.median(_fr_on)), float(np.median(_fr_off)),
                    int(np.sum(np.asarray(_fr_on) < np.asarray(_fr_off))), len(_fr_on), ic_mode)
        for li, lv in enumerate(levels):
            key, cut = str(lv["key"]), int(lv["cut"])
            per_area[key] = {}
            t_lv = time.time()
            for op in ops:
                core._ablate_cdmr = (op == "no_cdmr")
                core._ablate_egsr = (op == "no_egsr")
                pred = _fwd(cut, "noise_only" if op == "noise_tail" else "mask")
                core._ablate_cdmr = False
                core._ablate_egsr = False
                if op == "full":
                    full_preds[key] = pred.astype(np.float32)
                per_area[key][op] = {}
                for a in areas:
                    ia = area_pos[int(a)]
                    ep = gate_loss_endpoints(pred[ia], clean[ia], t_ms, cut, ls)
                    per_area[key][op][int(a)] = ep
            if core.cdmr_calibrated_ready():
                _rg = cdmr_regeneration_records(core, noisy, cut, float(clean_cache.global_scale))
                regen_alone[key] = float(np.median([gate_loss_endpoints(_rg[area_pos[int(a)]],
                                                                            clean[area_pos[int(a)]], t_ms, cut, ls)["late_nrmse"]
                                                    for a in areas]))
            med = {op: float(np.median([per_area[key][op][int(a)]["late_nrmse"] for a in areas])) for op in ops}
            medw = {op: float(np.median([per_area[key][op][int(a)]["amplitude_nrmse"] for a in areas])) for op in ops}
            logger.info("GATE-LOSS %s | withheld %d/%d gates (%.1f%%; last recorded %.3f ms) | median late "
                        "NRMSE: %s | median withheld-gate NRMSE: %s | %.1f s.",
                        key, int(lv["withheld_gates"]), T, float(lv["withheld_percent"]), float(lv["last_recorded_ms"]),
                        " | ".join("%s %.2f%%" % (GATE_LOSS_LABELS[o], med[o]) for o in ops),
                        " | ".join("%s %.2f%%" % (GATE_LOSS_LABELS[o], medw[o]) for o in ops),
                        time.time() - t_lv)
            if key in regen_alone:
                logger.info("GATE-LOSS %s | CDM-R regeneration alone (the trunk's input on the withheld gates): "
                            "median late NRMSE %.2f%% | Full PEBR-Net %.2f%%.",
                            key, regen_alone[key], med["full"])
    finally:
        core._ablate_cdmr, core._ablate_egsr = sw

    alist = [int(a) for a in areas]

    samples: List[Dict[str, Any]] = []
    try:
        ref_li = int(np.argmin([abs(float(lv["withheld_percent"]) - 38.7) for lv in levels]))
        ref_key = str(levels[ref_li]["key"])
        fp = full_preds[ref_key].astype(np.float64)
        late_tr = 100.0 * np.sqrt(np.sum((fp[:, Lc] - clean[:, Lc]) ** 2, axis=1)
                                  / np.maximum(np.sum(clean[:, Lc] ** 2, axis=1), 1e-300))
        nl = noisy[:, Lc]
        flips = np.mean(np.signbit(nl[:, 1:]) != np.signbit(nl[:, :-1]), axis=1)
        negs = np.mean(nl < 0, axis=1)
        morph = negs + flips
        clean_pos = np.all(clean > 0, axis=1)
        regime_row = np.array([area_info[int(p)]["regime"] for p in pid])
        rules = (("smooth", "smoothly biased input", "smooth_bias", morph <= 0.0),
                 ("spiky", "spiky, sign-alternating input", "spiky", morph >= 0.25))
        used: set = set()
        for tag, desc, regime, cond in rules:
            cand = (regime_row == regime) if np.any(regime_row == regime) else cond
            if tag == "smooth":
                cand = cand & (negs <= 0.0) if np.any(cand & (negs <= 0.0)) else cand
            else:
                cand = cand & (negs > 0.0) if np.any(cand & (negs > 0.0)) else cand
            if not np.any(cand):
                q = np.quantile(morph, 0.1 if tag == "smooth" else 0.9)
                cand = (morph <= q) if tag == "smooth" else (morph >= q)
            if np.any(cand & clean_pos):
                cand = cand & clean_pos
            idx = np.flatnonzero(cand & ~np.isin(np.arange(rows.size), list(used)))
            if idx.size == 0:
                continue
            j = int(idx[np.argsort(late_tr[idx], kind="mergesort")[idx.size // 2]])
            used.add(j)
            a = int(pid[j])
            st = int(j - int(area_pos[a][0]))
            samples.append({
                "tag": tag, "description": desc, "area": area_info[a]["name"], "station_index": st,
                "rule": ("noise regime '%s' of the design record" % regime) if np.any(regime_row == regime) else
                        ("late-window input morphology: share of negative samples + share of sign flips %s"
                         % ("= 0" if tag == "smooth" else ">= 0.25")),
                "selection": "median full-operator late NRMSE at %.1f%% withheld among %d candidates"
                             % (float(levels[ref_li]["withheld_percent"]), int(idx.size)),
                "clean": clean[j], "noisy": noisy[j],
                "pred": [full_preds[str(lv["key"])][j].astype(np.float64) for lv in levels],
                "late_nrmse": [float(100.0 * np.linalg.norm(full_preds[str(lv["key"])][j][Lc] - clean[j][Lc])
                                     / max(float(np.linalg.norm(clean[j][Lc])), 1e-300)) for lv in levels],
            })
    except Exception as exc:
        logger.warning("Figure-9 sample selection skipped: %r", exc)

    rep = run_dir / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    summary_levels = []
    for li, lv in enumerate(levels):
        key = str(lv["key"])
        ent: Dict[str, Any] = dict(lv)
        for op in ops:
            d = {}
            for ek in ("late_nrmse", "amplitude_nrmse", "global_nrmse", "slope_rmse", "structure_nrmse"):
                v = np.array([per_area[key][op][a][ek] for a in alist], dtype=np.float64)
                v = v[np.isfinite(v)]
                d[ek] = ({"median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                          "q75": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90))} if v.size else {})
            ent[op] = d
        summary_levels.append(ent)
    full_rec_late = np.array([full_record[a]["late_nrmse"] for a in alist], dtype=np.float64)
    payload = {
        "program": PROGRAM_NAME, "version": PROGRAM_VERSION, "split": split,
        "profiles_scored": int(len(alist)), "profiles_in_split": n_areas_total, "stations": int(rows.size), "gates": T,
        "late_window_first_gate": ls + 1, "gate_times_ms": [float(v) for v in t_ms],
        "protocol": ("the last gates of every station are withheld; retained gates keep their noise; the network is told "
                     "which gates are withheld (obs_mask) and CDM-R regenerates them before the trunk"),
        "operators": {op: GATE_LOSS_LABELS[op] for op in ops},
        "egsr_expression_scale": egsr_scale,
        "withheld_departure_share_by_position": dep_share,
        "cdmr_calibrated": bool(core.cdmr_calibrated_ready()),
        "cdmr_regeneration_alone_late_nrmse_median": regen_alone,
        "full_record_late_nrmse_median": float(np.median(full_rec_late)),
        "informative_continuation": ic_mode,
        "full_record_late_nrmse_median_by_informative_continuation": {
            ic_mode: float(np.median(full_rec_late)),
            ("off" if ic_mode == "on" else "on"): float(np.median([full_record_alt[a]["late_nrmse"] for a in alist])),
        },
        "levels": summary_levels,
        "figure9_samples": [{k2: v2 for k2, v2 in s.items() if k2 not in ("clean", "noisy", "pred")} for s in samples],
        "elapsed_s": round(time.time() - t_start, 1),
    }
    atomic_json_dump(payload, rep / tagged_name("gate_loss_suite.json"))
    with open(rep / tagged_name("gate_loss_per_profile.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "withheld_gates", "withheld_percent", "operator", "profile", "n_stations",
                    "design_late_snr_db", "input_late_snr_db", "regime", "late_nrmse_percent",
                    "amplitude_nrmse_percent", "global_nrmse_percent", "slope_rmse", "structure_nrmse_percent"])
        for lv in levels:
            key = str(lv["key"])
            for op in ops:
                for i, a in enumerate(alist):
                    e = per_area[key][op][a]
                    w.writerow([key, int(lv["withheld_gates"]), "%.2f" % float(lv["withheld_percent"]), op,
                                area_info[a]["name"], area_info[a]["n_stations"],
                                "%.3f" % area_info[a]["design_late_snr_db"], "%.3f" % area_info[a]["input_late_snr_db"],
                                area_info[a]["regime"],
                                "%.5f" % e["late_nrmse"], "%.5f" % e["amplitude_nrmse"], "%.5f" % e["global_nrmse"],
                                "%.5f" % e["slope_rmse"], "%.5f" % e["structure_nrmse"]])
    if bool(getattr(cfg.runtime, "plot", True)):
        fig_dir = run_dir / "figures"
        for _nm, _fn in (
            ("reconstruction", lambda: render_gate_loss_reconstruction(
                fig_dir, t_ms, levels, samples, logger, unit=str(getattr(cfg.data, "paired_unit", "nT/s") or "nT/s"))),
            ("levels", lambda: render_gate_loss_levels(fig_dir, levels, per_area, alist, ops, full_rec_late, logger,
                                                           full_record_alt=np.array([full_record_alt[a]["late_nrmse"]
                                                                                     for a in alist]),
                                                           ic_mode=ic_mode)),
        ):
            try:
                _fn()
            except Exception as exc:
                logger.exception("gate-loss figure '%s' failed: %s", _nm, exc)
    logger.info("GATE-LOSS SUITE DONE in %.1f s | full record late NRMSE median %.2f%% | E-GSR expression scale "
                "%.2f | reports/gate_loss_suite.json.",
                time.time() - t_start, float(np.median(full_rec_late)), egsr_scale)
    return payload


def render_gate_loss_reconstruction(fig_dir: Path, t_ms: np.ndarray, levels: Sequence[Mapping[str, Any]],
                                        samples: Sequence[Mapping[str, Any]], logger: logging.Logger,
                                        stem: str = "fig_gate_loss_reconstruction", unit: str = "nT/s") -> None:
    """Reconstruction figure: one row per withheld level, one column per sample trace."""
    if not samples:
        logger.warning("%s not rendered: no sample trace satisfied the selection rules.", stem)
        return
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    nL, nS = len(levels), len(samples)
    orange, blue = OKABE_ITO["vermillion"], OKABE_ITO["blue"]
    letters = "abcdefghijklmnopqrstuvwxyz"
    t = np.asarray(t_ms, dtype=np.float64)
    with plt.rc_context(nature_rc()):
        fig, axes = plt.subplots(nL, nS, figsize=_mm(62.0 * nS + 22.0, 30.0 * nL + 18.0), squeeze=False)
        for j, smp in enumerate(samples):
            yc = np.asarray(smp["clean"], dtype=np.float64)
            yn = np.asarray(smp["noisy"], dtype=np.float64)
            mags = np.concatenate([np.abs(yc[yc != 0]), np.abs(yn[yn != 0])])
            hi = float(np.max(mags)) * 2.5 if mags.size else 1.0
            lo_c = float(np.min(np.abs(yc[yc != 0]))) if np.any(yc != 0) else hi * 1e-6
            lo_n = float(np.percentile(np.abs(yn[yn != 0]), 5)) if np.any(yn != 0) else lo_c
            _pp = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in smp["pred"]])
            _pp = _pp[_pp > 0]
            lo_p = float(np.percentile(_pp, 1)) if _pp.size else lo_c
            lo = max(min(lo_c, lo_n, lo_p) * 0.4, hi * 1e-9)
            llo, lhi = math.log10(lo), math.log10(hi)
            for i, lv in enumerate(levels):
                ax = axes[i, j]
                cut = int(lv["cut"])
                tr, yr = t[:cut], yn[:cut]
                ax.plot(tr, np.log10(np.maximum(np.abs(yr), lo)), color=orange, lw=0.6, zorder=2)
                pos = yr > 0
                ax.plot(tr[pos], np.log10(yr[pos]), ls="none", marker="o", mfc="white", mec=orange, ms=2.2, mew=0.5,
                        zorder=3)
                ax.plot(tr[~pos], np.log10(np.maximum(np.abs(yr[~pos]), lo)), ls="none", marker="v", mfc="white",
                        mec=orange, ms=2.4, mew=0.5, zorder=3)
                yp = np.asarray(smp["pred"][i], dtype=np.float64)
                ax.plot(t, log10_amplitude(yp), color=blue, lw=1.1, zorder=4)
                ax.plot(t, log10_amplitude(yc), color="black", lw=0.8, ls=(0, (3.0, 2.0)), zorder=5)
                set_gate_time_xscale(ax, t)
                ax.set_ylim(llo, lhi)
                ax.set_xlim(*gate_time_xlim(t))
                x0 = math.sqrt(float(t[cut - 1]) * float(t[cut])) if float(t[cut - 1]) > 0 else 0.5 * float(t[cut])
                ax.add_patch(Rectangle((x0, llo), t[-1] * 1.06 - x0, lhi - llo, fill=False, ls=":", lw=0.8,
                                       ec="black", zorder=6))
                ax.text(0.05, 0.08, "NRMSE %.2f%%" % float(smp["late_nrmse"][i]), transform=ax.transAxes,
                        fontsize=6, ha="left", va="bottom")
                ttl = "(%s)" % letters[(j * nL + i) % 26]
                if i == 0:
                    ttl = "Sample %d: %s\n%s" % (j + 1, str(smp.get("description", "")), ttl)
                ax.set_title(ttl, fontsize=6.5)
                if j == 0:
                    ax.set_ylabel("%.1f%% missing" % float(lv["withheld_percent"]), fontweight="bold")
                if i == nL - 1:
                    ax.set_xlabel("Gate time (ms)")
                else:
                    ax.tick_params(labelbottom=False)
        handles = [Line2D([0], [0], color=orange, lw=0.6, marker="o", mfc="white", mec=orange, ms=2.5, label="Noisy input"),
                   Line2D([0], [0], color=orange, lw=0, marker="v", mfc="white", mec=orange, ms=2.6,
                          label="Negative input (|amplitude|)"),
                   Line2D([0], [0], color=blue, lw=1.1, label="Full PEBR-Net"),
                   Line2D([0], [0], color="black", lw=0.8, ls=(0, (3.0, 2.0)), label="Clean reference"),
                   Line2D([0], [0], color="black", lw=0.8, ls=":", label="Withheld interval")]
        fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.supylabel("Amplitude (log10(%s))" % unit, fontsize=7)
        _save_figure(fig, fig_dir, stem)
        plt.close(fig)
    logger.info("%s rendered | samples: %s.", stem,
                "; ".join("%s station %d (%s; %s)" % (s["area"], int(s["station_index"]), s["rule"], s["selection"])
                          for s in samples))


def render_gate_loss_levels(fig_dir: Path, levels: Sequence[Mapping[str, Any]],
                                per_area: Mapping[str, Mapping[str, Mapping[int, Mapping[str, float]]]],
                                alist: Sequence[int], ops: Sequence[str], full_record_late: np.ndarray,
                                logger: logging.Logger, stem: str = "fig_gate_loss_levels",
                                full_record_alt: Optional[np.ndarray] = None, ic_mode: str = "on") -> None:
    """Every level of the suite (9.7-71.0% withheld by default): median over profiles with the inter-quartile band of
    the late NRMSE, the withheld-gate amplitude NRMSE.
    """
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array([float(lv["withheld_percent"]) for lv in levels])
    panels = (("late_nrmse", "Late-window NRMSE (%)"), ("amplitude_nrmse", "Withheld-gate amplitude NRMSE (%)"),
              ("slope_rmse", "Decay-slope RMSE, Δp"), ("structure_nrmse", "Temporal-structure NRMSE (%)"))
    mk = {"full": "o", "no_cdmr": "^", "no_egsr": "D", "noise_tail": "s"}
    with plt.rc_context(nature_rc()):
        fig, axes = plt.subplots(2, 2, figsize=_mm(183.0, 120.0))
        _draw = [o for o in ops if o != "full"] + (["full"] if "full" in ops else [])
        for pi, (ek, lab) in enumerate(panels):
            ax = axes.flat[pi]
            for op in _draw:
                med, q1, q3 = [], [], []
                for lv in levels:
                    v = np.array([per_area[str(lv["key"])][op][a][ek] for a in alist], dtype=np.float64)
                    v = v[np.isfinite(v)]
                    med.append(np.median(v) if v.size else np.nan)
                    q1.append(np.percentile(v, 25) if v.size else np.nan)
                    q3.append(np.percentile(v, 75) if v.size else np.nan)
                ax.fill_between(x, q1, q3, color=GATE_LOSS_COLORS[op], alpha=0.13, lw=0)
                ax.plot(x, med, color=GATE_LOSS_COLORS[op], lw=(1.4 if op == "full" else 0.9),
                        marker=mk.get(op, "o"), ms=(3.4 if op == "full" else 2.8), label=GATE_LOSS_LABELS[op],
                        zorder=(5 if op == "full" else 3), mfc=("white" if op == "no_egsr" else None))
            if ek == "late_nrmse":
                ax.plot([0.0], [float(np.median(full_record_late))], ls="none", marker="*", ms=6, color="black",
                        label="Full record, informative continuation %s" % ic_mode)
                if full_record_alt is not None and np.asarray(full_record_alt).size:
                    ax.plot([0.0], [float(np.median(full_record_alt))], ls="none", marker="*", ms=6, mfc="white",
                            mec="black", label="Full record, informative continuation %s"
                            % ("off" if ic_mode == "on" else "on"))
            ax.set_yscale("log")
            ax.set_xlim(-3.0 if ek == "late_nrmse" else x[0] - 3.0, x[-1] + 3.0)
            ax.set_xticks(([0.0] if ek == "late_nrmse" else []) + list(x))
            ax.set_xticklabels((["0"] if ek == "late_nrmse" else []) + ["%.1f" % v for v in x], fontsize=5.5,
                               rotation=55, ha="right", rotation_mode="anchor")
            ax.set_xlabel("Gates withheld from the end of the decay (%)")
            ax.set_ylabel(lab)
            ax.set_title("(%s)" % "abcd"[pi], loc="left", fontsize=7.5)
            if pi == 0:
                ax.legend(loc="upper left", fontsize=5.5)
        _save_figure(fig, fig_dir, stem)
        plt.close(fig)
    logger.info("%s rendered (%d levels, %d operators).", stem, len(levels), len(ops))
