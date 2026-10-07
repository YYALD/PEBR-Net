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


GATE_LOSS_ENDPOINTS: Tuple[Tuple[str, str], ...] = (
    ("amplitude_nrmse", "Amplitude NRMSE"),
    ("slope_rmse", "Decay-slope RMSE"),
    ("structure_nrmse", "Temporal-structure NRMSE"),
)


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


def _rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return np.bincount(inv, weights=ranks)[inv] / cnt[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties); NaN below three finite pairs."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    ra, rb = _rank(a[ok]), _rank(b[ok])
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_median_ci(v: np.ndarray, n_boot: int = 2000, seed: int = 765) -> Tuple[float, float, float]:
    """Median with its percentile-bootstrap 95% interval."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    meds = np.empty(int(n_boot), dtype=np.float64)
    for b0 in range(0, int(n_boot), 256):
        nb = min(256, int(n_boot) - b0)
        meds[b0:b0 + nb] = np.median(v[rng.integers(0, v.size, size=(nb, v.size))], axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def input_quality_clusters(snr_db: np.ndarray, difficulty: np.ndarray, k: int = 5,
                               iters: int = 60) -> np.ndarray:
    """Input-quality clusters of profiles from (late SNR, benchmark difficulty): k-means on the standardised pair,
    initialised deterministically at the SNR quantiles.
    """
    s = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    d = np.asarray(difficulty, dtype=np.float64).reshape(-1)
    n = s.size
    lab = np.zeros(n, dtype=np.int64)
    ok = np.isfinite(s) & np.isfinite(d)
    k = int(max(1, min(int(k), int(ok.sum()))))
    if k <= 1:
        return lab
    X = np.stack([s[ok], d[ok]], axis=1)
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    order = np.argsort(X[:, 0], kind="mergesort")
    cent = np.stack([X[order[int(round((j + 0.5) * X.shape[0] / k - 0.5))]] for j in range(k)])
    for _ in range(int(iters)):
        dist = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=-1)
        a = np.argmin(dist, axis=1)
        new = np.stack([X[a == j].mean(axis=0) if np.any(a == j) else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    mean_snr = np.array([s[ok][a == j].mean() if np.any(a == j) else np.inf for j in range(k)])
    remap = np.empty(k, dtype=np.int64)
    remap[np.argsort(mean_snr, kind="mergesort")] = np.arange(k)
    lab[ok] = remap[a]
    lab[~ok] = -1
    return lab


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
    have_design = bool(meta) and all(np.isfinite(area_info[int(a)]["design_late_snr_db"]) for a in areas)
    snr_name = "Designed late-SNR (dB)" if have_design else "Input late-SNR (dB)"
    snr_key = "design_late_snr_db" if have_design else "input_late_snr_db"
    logger.info("GATE-LOSS SUITE | %d %s profiles, %d stations, %d gates (late window from gate %d) | levels: %s "
                "| operators: %s + no-continuation baseline | cluster axis: %s.",
                int(areas.size), split, int(rows.size), T, ls + 1,
                ", ".join("%d/%d gates = %.1f%%" % (lv["withheld_gates"], T, lv["withheld_percent"]) for lv in levels),
                ", ".join(GATE_LOSS_LABELS[o] for o in GATE_LOSS_OPERATORS), snr_name)

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
                       "qualification), so the 'E-GSR removed' operator is the full operator bit for bit "
                       "and its ratios read 0.",
                       egsr_scale)
    ops: List[str] = list(GATE_LOSS_OPERATORS)
    if bool(getattr(cfg.runtime, "gate_loss_noise_tail_baseline", True)):
        ops.append("noise_tail")
    fan_k = int(getattr(cfg.runtime, "gate_loss_fan_withheld", 18) or 18)
    fan_li = int(np.argmin([abs(int(lv["withheld_of_reference"]) - fan_k) for lv in levels]))
    rng = np.random.default_rng(int(cfg.train.seed) + 765)
    per_area: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = {}
    full_preds: Dict[str, np.ndarray] = {}
    fan: Dict[str, Any] = {}
    fan_by_level: Dict[str, Dict[str, float]] = {}
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
            lvl_samples: Dict[str, Dict[str, np.ndarray]] = {}
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
                smp: Dict[str, List[np.ndarray]] = {"dp": [], "p_ref": [], "p_hat": [], "kappa_ref": [], "kappa_hat": []}
                for a in areas:
                    ia = area_pos[int(a)]
                    ep = gate_loss_endpoints(pred[ia], clean[ia], t_ms, cut, ls, keep_samples=True)
                    for kk in smp:
                        smp[kk].append(ep.pop(kk))
                    per_area[key][op][int(a)] = ep
                lvl_samples[op] = {kk: (np.concatenate(v) if v else np.zeros(0, np.float32)) for kk, v in smp.items()}
            fs = lvl_samples["full"]
            if fs["dp"].size >= 10:
                p0 = float(np.median(np.abs(fs["p_ref"])))
                s90 = float(np.quantile(np.abs(fs["dp"]) / (p0 + np.abs(fs["p_ref"])), 0.90))
                fan_by_level[key] = {"p0": p0, "scale": s90}
                for op in ops:
                    o = lvl_samples[op]
                    fan_by_level[key]["share_" + op] = (
                        100.0 * float(np.mean(np.abs(o["dp"]) <= s90 * (p0 + np.abs(o["p_ref"])))) if o["dp"].size else float("nan"))
                if li == fan_li:
                    fan = {"key": key, "withheld_percent": float(lv["withheld_percent"]), "p0": p0, "scale": s90, "ops": {}}
                    for op in GATE_LOSS_OPERATORS:
                        o = lvl_samples[op]
                        n_o = int(o["dp"].size)
                        sel = np.sort(rng.choice(n_o, size=200000, replace=False)) if n_o > 200000 else np.arange(n_o)
                        selk = (np.sort(rng.choice(o["kappa_ref"].size, size=200000, replace=False))
                                if o["kappa_ref"].size > 200000 else np.arange(o["kappa_ref"].size))
                        fan["ops"][op] = {"dp": o["dp"][sel], "p_ref": o["p_ref"][sel], "p_hat": o["p_hat"][sel],
                                          "kappa_ref": o["kappa_ref"][selk], "kappa_hat": o["kappa_hat"][selk],
                                          "share": fan_by_level[key]["share_" + op]}
            if core.cdmr_calibrated_ready():
                _rg = cdmr_regeneration_records(core, noisy, cut, float(clean_cache.global_scale))
                regen_alone[key] = float(np.median([gate_loss_endpoints(_rg[area_pos[int(a)]],
                                                                            clean[area_pos[int(a)]], t_ms, cut, ls)["late_nrmse"]
                                                    for a in areas]))
            med = {op: float(np.median([per_area[key][op][int(a)]["late_nrmse"] for a in areas])) for op in ops}
            medw = {op: float(np.median([per_area[key][op][int(a)]["amplitude_nrmse"] for a in areas])) for op in ops}
            logger.info("GATE-LOSS %s | withheld %d/%d gates (%.1f%%; last recorded %.3f ms) | median late "
                        "NRMSE: %s | median withheld-gate NRMSE: %s | fan share (full = 90%% by construction): %s | %.1f "
                        "s.",
                        key, int(lv["withheld_gates"]), T, float(lv["withheld_percent"]), float(lv["last_recorded_ms"]),
                        " | ".join("%s %.2f%%" % (GATE_LOSS_LABELS[o], med[o]) for o in ops),
                        " | ".join("%s %.2f%%" % (GATE_LOSS_LABELS[o], medw[o]) for o in ops),
                        " | ".join("%s %.0f%%" % (GATE_LOSS_LABELS[o], fan_by_level.get(key, {}).get("share_" + o, float("nan")))
                                   for o in ops),
                        time.time() - t_lv)
            if key in regen_alone:
                logger.info("GATE-LOSS %s | CDM-R regeneration alone (the trunk's input on the withheld gates): "
                            "median late NRMSE %.2f%% | Full PEBR-Net %.2f%%.",
                            key, regen_alone[key], med["full"])
    finally:
        core._ablate_cdmr, core._ablate_egsr = sw

    alist = [int(a) for a in areas]
    snr_vec = np.array([area_info[a][snr_key] for a in alist], dtype=np.float64)
    diff_vec = np.log10(np.maximum([full_record[a]["late_nrmse"] for a in alist], 1e-6))
    clusters = input_quality_clusters(snr_vec, diff_vec, k=min(5, len(alist)))
    ratios: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for op in ("no_cdmr", "no_egsr"):
        ratios[op] = {}
        for ek, _lab in GATE_LOSS_ENDPOINTS:
            ratios[op][ek] = []
            for lv in levels:
                key = str(lv["key"])
                num = np.array([per_area[key][op][a][ek] for a in alist], dtype=np.float64)
                den = np.array([per_area[key]["full"][a][ek] for a in alist], dtype=np.float64)
                r = np.log2(np.maximum(num, 1e-12) / np.maximum(den, 1e-12))
                r[~(np.isfinite(num) & np.isfinite(den))] = np.nan
                ratios[op][ek].append(r)
    fan_key = str(levels[fan_li]["key"])
    cl_stats: Dict[str, Any] = {}
    for op in ("no_cdmr", "no_egsr"):
        r = ratios[op]["structure_nrmse"][fan_li]
        xs, ms, p10, p90, ns = [], [], [], [], []
        for c in sorted(set(int(v) for v in clusters if v >= 0)):
            sel = (clusters == c) & np.isfinite(r)
            if not np.any(sel):
                continue
            xs.append(float(np.median(snr_vec[sel])))
            ms.append(float(np.median(r[sel])))
            p10.append(float(np.percentile(r[sel], 10)))
            p90.append(float(np.percentile(r[sel], 90)))
            ns.append(int(sel.sum()))
        fin = np.isfinite(r)
        cl_stats[op] = {"snr": xs, "median_log2_ratio": ms, "p10": p10, "p90": p90, "n": ns,
                        "median_factor": float(2.0 ** np.median(r[fin])) if np.any(fin) else float("nan"),
                        "spearman_rho": spearman(r, snr_vec)}
    area_delta: Dict[str, List[Dict[str, Any]]] = {}
    for op in ("no_cdmr", "no_egsr"):
        area_delta[op] = []
        for li, lv in enumerate(levels):
            key = str(lv["key"])
            dv = np.array([per_area[key][op][a]["late_nrmse"] - per_area[key]["full"][a]["late_nrmse"] for a in alist])
            m, lo, hi = bootstrap_median_ci(dv, n_boot=2000, seed=765 + li)
            cx, cm, cn = [], [], []
            for c in sorted(set(int(v) for v in clusters if v >= 0)):
                sel = (clusters == c) & np.isfinite(dv)
                if np.any(sel):
                    cx.append(int(c))
                    cm.append(float(np.median(dv[sel])))
                    cn.append(int(sel.sum()))
            area_delta[op].append({"median_pp": m, "ci_lo": lo, "ci_hi": hi,
                                   "share_positive_percent": 100.0 * float(np.mean(dv[np.isfinite(dv)] > 0)) if np.any(np.isfinite(dv)) else float("nan"),
                                   "cluster": cx, "cluster_median_pp": cm, "cluster_n": cn})

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
        ent["common_fan"] = fan_by_level.get(key, {})
        ent["log2_ratio_median"] = {op: {ek: float(np.nanmedian(ratios[op][ek][li])) if np.any(np.isfinite(ratios[op][ek][li])) else float("nan")
                                         for ek, _l in GATE_LOSS_ENDPOINTS} for op in ("no_cdmr", "no_egsr")}
        ent["work_area_delta_late_pp"] = {op: {k2: v2 for k2, v2 in area_delta[op][li].items()} for op in ("no_cdmr", "no_egsr")}
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
        "cdmr_calibrated": bool(core.cdmr_calibrated_ready()),
        "cdmr_regeneration_alone_late_nrmse_median": regen_alone,
        "full_record_late_nrmse_median": float(np.median(full_rec_late)),
        "informative_continuation": ic_mode,
        "full_record_late_nrmse_median_by_informative_continuation": {
            ic_mode: float(np.median(full_rec_late)),
            ("off" if ic_mode == "on" else "on"): float(np.median([full_record_alt[a]["late_nrmse"] for a in alist])),
        },
        "cluster_axis": snr_name, "clusters": "k-means (k<=5) on standardised (late SNR, log10 full-record late NRMSE)",
        "fan_level": fan_key, "fan_definition": "|dp| <= s (p0 + |p_ref|); p0 = median |p_ref| and s = 90% quantile "
                                                "of |dp| / (p0 + |p_ref|) of the full operator at the same level",
        "levels": summary_levels,
        "input_quality_clusters_at_fan_level": cl_stats,
        "figure9_samples": [{k2: v2 for k2, v2 in s.items() if k2 not in ("clean", "noisy", "pred")} for s in samples],
        "elapsed_s": round(time.time() - t_start, 1),
    }
    atomic_json_dump(payload, rep / tagged_name("gate_loss_suite.json"))
    with open(rep / tagged_name("gate_loss_per_profile.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "withheld_gates", "withheld_percent", "operator", "profile", "n_stations",
                    "design_late_snr_db", "input_late_snr_db", "regime", "cluster", "late_nrmse_percent",
                    "amplitude_nrmse_percent", "global_nrmse_percent", "slope_rmse", "structure_nrmse_percent"])
        for lv in levels:
            key = str(lv["key"])
            for op in ops:
                for i, a in enumerate(alist):
                    e = per_area[key][op][a]
                    w.writerow([key, int(lv["withheld_gates"]), "%.2f" % float(lv["withheld_percent"]), op,
                                area_info[a]["name"], area_info[a]["n_stations"],
                                "%.3f" % area_info[a]["design_late_snr_db"], "%.3f" % area_info[a]["input_late_snr_db"],
                                area_info[a]["regime"], int(clusters[i]),
                                "%.5f" % e["late_nrmse"], "%.5f" % e["amplitude_nrmse"], "%.5f" % e["global_nrmse"],
                                "%.5f" % e["slope_rmse"], "%.5f" % e["structure_nrmse"]])
    if bool(getattr(cfg.runtime, "plot", True)):
        fig_dir = run_dir / "figures"
        for _nm, _fn in (
            ("reconstruction", lambda: render_gate_loss_reconstruction(
                fig_dir, t_ms, levels, samples, logger, unit=str(getattr(cfg.data, "paired_unit", "nT/s") or "nT/s"))),
            ("mechanisms", lambda: render_gate_loss_mechanisms(fig_dir, levels, ratios, cl_stats, area_delta, fan,
                                                                     fan_li, snr_name, clusters, logger)),
            ("levels", lambda: render_gate_loss_levels(fig_dir, levels, per_area, alist, ops, full_rec_late, logger,
                                                           full_record_alt=np.array([full_record_alt[a]["late_nrmse"]
                                                                                     for a in alist]),
                                                           ic_mode=ic_mode)),
        ):
            try:
                _fn()
            except Exception as exc:
                logger.exception("gate-loss figure '%s' failed: %s", _nm, exc)
    logger.info("GATE-LOSS SUITE DONE in %.1f s | full record late NRMSE median %.2f%% | at the fan level (%s, "
                "%.1f%% withheld) the structure penalty is %.1fx without CDM-R (rho %+.2f with %s) and %.1fx without E-GSR "
                "(rho %+.2f; E-GSR expression scale %.2f) | reports/gate_loss_suite.json.",
                time.time() - t_start, float(np.median(full_rec_late)), fan_key, float(levels[fan_li]["withheld_percent"]),
                cl_stats["no_cdmr"]["median_factor"], cl_stats["no_cdmr"]["spearman_rho"], snr_name,
                cl_stats["no_egsr"]["median_factor"], cl_stats["no_egsr"]["spearman_rho"], egsr_scale)
    return payload


def _kde(v: np.ndarray, grid: np.ndarray, max_n: int = 4000, seed: int = 765) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.zeros_like(grid, dtype=np.float64)
    if v.size > max_n:
        v = np.random.default_rng(seed).choice(v, size=max_n, replace=False)
    sd = float(np.std(v))
    iqr = float(np.subtract(*np.percentile(v, [75, 25]))) / 1.349
    sig = min(sd, iqr) if iqr > 0 else sd
    bw = max(0.9 * sig * v.size ** (-0.2), 1e-6)
    z = (np.asarray(grid, dtype=np.float64)[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (v.size * bw * math.sqrt(2.0 * math.pi))


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


def render_gate_loss_mechanisms(fig_dir: Path, levels: Sequence[Mapping[str, Any]],
                                    ratios: Mapping[str, Mapping[str, Sequence[np.ndarray]]],
                                    cl_stats: Mapping[str, Mapping[str, Any]],
                                    area_delta: Mapping[str, Sequence[Mapping[str, Any]]], fan: Mapping[str, Any],
                                    fan_li: int, snr_name: str, clusters: np.ndarray, logger: logging.Logger,
                                    stem: str = "fig_gate_loss_mechanisms") -> None:
    """Mechanism figure."""
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.lines import Line2D
    nL = len(levels)
    col = GATE_LOSS_COLORS
    mk = {"no_cdmr": "^", "no_egsr": "D"}
    ls_op = {"full": "-", "no_cdmr": (0, (4.0, 2.0)), "no_egsr": (0, (4.0, 1.5, 1.0, 1.5))}
    rng = np.random.default_rng(765)
    with plt.rc_context(nature_rc()):
        fig = plt.figure(figsize=_mm(183.0, 196.0))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
        left = outer[0].subgridspec(3, 1, height_ratios=[1.2, 0.95, 0.95])
        gA = left[0].subgridspec(1, 2, wspace=0.05)
        axA = [fig.add_subplot(gA[0, 0]), None]
        axA[1] = fig.add_subplot(gA[0, 1], sharey=axA[0])
        axB = fig.add_subplot(left[1])
        axC = fig.add_subplot(left[2])
        right = outer[1].subgridspec(3, 1)
        shades = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, nL))
        n_ep = len(GATE_LOSS_ENDPOINTS)
        for c, op in enumerate(("no_cdmr", "no_egsr")):
            ax = axA[c]
            for r, (ek, _lab) in enumerate(GATE_LOSS_ENDPOINTS):
                y0 = float(n_ep - 1 - r)
                meds = []
                for li in range(nL):
                    v = np.asarray(ratios[op][ek][li], dtype=np.float64)
                    v = v[np.isfinite(v)]
                    meds.append(float(np.median(v)) if v.size else float("nan"))
                    if v.size == 0:
                        continue
                    sub = v if v.size <= 160 else rng.choice(v, 160, replace=False)
                    ax.scatter(sub, y0 + rng.uniform(-0.22, 0.22, sub.size), s=3.0, color=shades[li], alpha=0.6,
                               lw=0, zorder=2)
                offs = np.linspace(-0.15, 0.15, nL)
                ax.scatter(meds, y0 + offs, s=13, marker=mk[op], facecolor="white", edgecolor=col[op], lw=0.8, zorder=4)
                ax.text(0.98, y0 + 0.30, "%+.2f" % meds[fan_li] if np.isfinite(meds[fan_li]) else "n/a",
                        transform=ax.get_yaxis_transform(), ha="right", va="center", color=col[op], fontsize=6.5,
                        fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.85), zorder=6)
            _all = np.concatenate([np.asarray(ratios[op][ek][li], dtype=np.float64).reshape(-1)
                                   for ek, _l in GATE_LOSS_ENDPOINTS for li in range(nL)])
            _all = _all[np.isfinite(_all)]
            if _all.size and float(np.ptp(np.percentile(_all, [1, 99]))) > 0.05:
                _lo, _hi = np.percentile(_all, [1, 99])
                _sp = float(_hi - _lo)
                ax.set_xlim(min(float(_lo) - 0.08 * _sp, -0.25), max(float(_hi) + 0.45 * _sp, 0.5))
            else:
                ax.set_xlim(-1.0, 1.5)
            ax.axvline(0.0, color="0.55", lw=0.4, ls=":", zorder=1)
            ax.set_ylim(-0.6, n_ep - 0.4)
            ax.set_title(GATE_LOSS_LABELS[op], color=col[op], fontsize=6.5, fontweight="bold")
            ax.set_xlabel("log2 (ablation / Full)")
        axA[0].set_yticks(range(n_ep))
        axA[0].set_yticklabels([lab.replace(" ", "\n", 1) for _ek, lab in reversed(GATE_LOSS_ENDPOINTS)])
        axA[1].tick_params(labelleft=False)
        lv_handles = [Line2D([0], [0], ls="none", marker="o", ms=3.5, mfc=shades[li], mec="none",
                             label="%.0f%%" % float(levels[li]["withheld_percent"])) for li in range(nL)]
        axA[0].legend(handles=lv_handles, ncol=min(nL, 6), loc="lower left", bbox_to_anchor=(0.0, 1.07),
                      fontsize=5.5, handletextpad=0.1, columnspacing=0.6, frameon=False)
        axA[0].text(0.0, 1.30, "(a) Multi-endpoint effect atlas", transform=axA[0].transAxes, fontsize=7.5)
        for op in ("no_cdmr", "no_egsr"):
            s = cl_stats.get(op, {})
            if not s or not s.get("snr"):
                continue
            x = np.asarray(s["snr"], dtype=np.float64)
            o = np.argsort(x)
            axB.fill_between(x[o], np.asarray(s["p10"])[o], np.asarray(s["p90"])[o], color=col[op], alpha=0.13, lw=0)
            axB.plot(x[o], np.asarray(s["median_log2_ratio"])[o], color=col[op], ls=ls_op[op], lw=0.9, marker=mk[op],
                     ms=4.0, mfc=col[op], mec="white", mew=0.4)
        def _fmt_rho(v: float) -> str:
            return ("rho=%+.2f" % v) if np.isfinite(v) else "rho n/a"

        txt = ["%s%s %.1fx | %s" % ("\u2212", "CDM-R" if op == "no_cdmr" else "E-GSR",
                                     cl_stats.get(op, {}).get("median_factor", float("nan")),
                                     _fmt_rho(float(cl_stats.get(op, {}).get("spearman_rho", float("nan")))))
               for op in ("no_cdmr", "no_egsr")]
        axB.text(0.98, 0.97, txt[0], transform=axB.transAxes, ha="right", va="top", color=col["no_cdmr"], fontsize=6)
        axB.text(0.98, 0.86, txt[1], transform=axB.transAxes, ha="right", va="top", color=col["no_egsr"], fontsize=6)
        axB.axhline(0.0, color="0.55", lw=0.4, ls=":")
        axB.set_xlabel(snr_name)
        axB.set_ylabel("log2(structure error / Full)")
        axB.set_title("(b) Input-quality clusters at %.0f%% missing" % float(levels[fan_li]["withheld_percent"]),
                      loc="left", fontsize=7.5)
        w = 0.36
        for c, op in enumerate(("no_cdmr", "no_egsr")):
            for li in range(nL):
                d = area_delta[op][li]
                x = li + (-0.2 if c == 0 else 0.2)
                m = float(d["median_pp"])
                if not np.isfinite(m):
                    continue
                axC.bar(x, m, width=w, color=col[op], alpha=0.18, edgecolor=col[op], lw=0.7, zorder=2)
                if np.isfinite(d["ci_lo"]) and np.isfinite(d["ci_hi"]):
                    axC.errorbar(x, m, yerr=[[max(m - d["ci_lo"], 0.0)], [max(d["ci_hi"] - m, 0.0)]], color=col[op],
                                 lw=0.7, capsize=1.5, zorder=3)
                cm = np.asarray(d.get("cluster_median_pp", []), dtype=np.float64)
                cn = np.asarray(d.get("cluster_n", []), dtype=np.float64)
                if cm.size:
                    axC.scatter(x + rng.uniform(-0.08, 0.08, cm.size), cm, s=4.0 + 2.5 * np.sqrt(cn), facecolor=col[op],
                                edgecolor="white", lw=0.3, alpha=0.75, zorder=4)
                top = max(m, float(d["ci_hi"]) if np.isfinite(d["ci_hi"]) else m, float(np.max(cm)) if cm.size else m)
                axC.text(x, top, "%.0f%%" % float(d["share_positive_percent"]), ha="center", va="bottom",
                         color=col[op], fontsize=5.5, fontweight="bold")
        axC.axhline(0.0, color="0.4", lw=0.5)
        axC.set_xticks(range(nL))
        axC.set_xticklabels(["%.0f%%" % float(lv["withheld_percent"]) for lv in levels])
        axC.set_xlabel("Late-tail missing")
        axC.set_ylabel("Work-area ΔNRMSE (pp)")
        axC.set_title("(c) Cross-work-area stability, P(Δ>0)", loc="left", fontsize=7.5)
        p0, s90 = float(fan.get("p0", float("nan"))), float(fan.get("scale", float("nan")))
        _of = (fan.get("ops") or {}).get("full")
        if _of is not None and np.asarray(_of["dp"]).size >= 10:
            _dpf = np.asarray(_of["dp"], dtype=np.float64)
            _prf = np.asarray(_of["p_ref"], dtype=np.float64)
            _okf = np.isfinite(_dpf) & np.isfinite(_prf)
            lim_common = float(np.percentile(np.abs(_dpf[_okf]), 99.0)) * 1.25 + 1e-6
            if np.isfinite(s90) and np.isfinite(p0):
                _pmax = float(np.percentile(np.abs(_prf[_okf]), 99.5))
                lim_common = max(lim_common, 1.8 * s90 * (p0 + _pmax))
        else:
            lim_common = float("nan")
        for k, op in enumerate(GATE_LOSS_OPERATORS):
            sub = right[k].subgridspec(2, 2, width_ratios=[4.2, 1.0], height_ratios=[1.0, 4.0], wspace=0.04, hspace=0.04)
            axm = fig.add_subplot(sub[1, 0])
            axt = fig.add_subplot(sub[0, 0], sharex=axm)
            axr = fig.add_subplot(sub[1, 1])
            o = (fan.get("ops") or {}).get(op)
            ttl = "(%s) %s %.0f%% missing" % ("def"[k], "Full PEBR-Net" if op == "full" else
                                               ("w/o CDM-R" if op == "no_cdmr" else "w/o E-GSR"),
                                               float(fan.get("withheld_percent", float("nan"))))
            axt.set_title(ttl, loc="left", fontsize=7.5)
            if not o or np.asarray(o["dp"]).size < 10:
                axm.text(0.5, 0.5, "no valid decay-law samples", transform=axm.transAxes, ha="center")
                continue
            dp = np.asarray(o["dp"], dtype=np.float64)
            pr = np.asarray(o["p_ref"], dtype=np.float64)
            ok = np.isfinite(dp) & np.isfinite(pr)
            dp, pr = dp[ok], pr[ok]
            lo_p, hi_p = np.percentile(pr, [0.5, 99.5])
            lim = lim_common if np.isfinite(lim_common) else float(np.percentile(np.abs(dp), 99.0)) * 1.15 + 1e-6
            sel = rng.choice(dp.size, size=min(600, dp.size), replace=False)
            _in = np.abs(dp[sel]) <= lim
            axm.scatter(pr[sel][_in], dp[sel][_in], s=2.5, color=col[op], alpha=0.55, lw=0, zorder=2)
            _up = dp[sel] > lim
            _dn = dp[sel] < -lim
            axm.scatter(pr[sel][_up], np.full(int(_up.sum()), lim * 0.97), s=6, marker="^", color=col[op], alpha=0.7,
                        lw=0, zorder=2, clip_on=False)
            axm.scatter(pr[sel][_dn], np.full(int(_dn.sum()), -lim * 0.97), s=6, marker="v", color=col[op], alpha=0.7,
                        lw=0, zorder=2, clip_on=False)
            _out = 100.0 * float(np.mean(np.abs(dp) > lim))
            if _out >= 0.5:
                axm.text(0.03, 0.04, "%.0f%% beyond the axis (\u25b2 decays too fast)" % _out, transform=axm.transAxes,
                         ha="left", va="bottom", color=col[op], fontsize=5.5)
            edges = np.quantile(pr, np.linspace(0.0, 1.0, 9))
            cx, q10, q25, q50, q75, q90 = [], [], [], [], [], []
            for b in range(8):
                m_ = (pr >= edges[b]) & (pr <= edges[b + 1])
                if int(m_.sum()) < 5:
                    continue
                cx.append(float(np.median(pr[m_])))
                qq = np.percentile(dp[m_], [10, 25, 50, 75, 90])
                q10.append(qq[0])
                q25.append(qq[1])
                q50.append(qq[2])
                q75.append(qq[3])
                q90.append(qq[4])
            if cx:
                axm.fill_between(cx, q10, q90, color=col[op], alpha=0.10, lw=0, zorder=1)
                axm.fill_between(cx, q25, q75, color=col[op], alpha=0.22, lw=0, zorder=1)
                axm.plot(cx, q50, color=col[op], lw=1.1, zorder=3)
            if np.isfinite(s90) and np.isfinite(p0):
                pp = np.linspace(lo_p, hi_p, 50)
                h = s90 * (p0 + np.abs(pp))
                axm.plot(pp, h, color="0.25", lw=0.7, ls=(0, (3.0, 2.0)), zorder=3)
                axm.plot(pp, -h, color="0.25", lw=0.7, ls=(0, (3.0, 2.0)), zorder=3)
            axm.text(0.97, 0.95, "%.0f%% in common fan" % float(o.get("share", float("nan"))), transform=axm.transAxes,
                     ha="right", va="top", color=col[op], fontsize=6.5, fontweight="bold")
            axm.set_ylim(-lim, lim)
            axm.set_xlim(lo_p, hi_p)
            axm.set_ylabel("Signed local-\nexponent error, Δp(t)")
            if k == 2:
                axm.set_xlabel("Reference local decay exponent, p(t)$_{ref}$")
            _ph = np.asarray(o["p_hat"], dtype=np.float64)
            _ph = _ph[np.isfinite(_ph)]
            gx = np.linspace(min(lo_p, float(np.percentile(_ph, 1)) if _ph.size else lo_p),
                             max(hi_p, float(np.percentile(_ph, 99)) if _ph.size else hi_p), 200)
            fh = _kde(o["p_hat"], gx)
            fr = _kde(o["p_ref"], gx)
            axt.fill_between(gx, 0, fh, color=col[op], alpha=0.25, lw=0)
            axt.plot(gx, fh, color=col[op], lw=0.9)
            axt.plot(gx, fr, color="0.2", lw=0.8, ls=(0, (3.0, 2.0)))
            axt.set_yticks([])
            axt.set_xlim(lo_p, hi_p)
            axm.set_xlim(lo_p, hi_p)
            axt.tick_params(labelbottom=False)
            _pin = 100.0 * float(np.mean((_ph >= lo_p) & (_ph <= hi_p))) if _ph.size else float("nan")
            if np.isfinite(_pin) and _pin < 95.0:
                axt.text(0.99, 0.95, "%.0f%% of p(t) in range" % _pin, transform=axt.transAxes, ha="right", va="top",
                         color=col[op], fontsize=5.5)
            axt.spines["left"].set_visible(False)
            axt.text(0.01, 0.95, "p(t)", transform=axt.transAxes, ha="left", va="top", fontsize=6, fontweight="bold")
            kh = np.asarray(o["kappa_hat"], dtype=np.float64)
            kr = np.asarray(o["kappa_ref"], dtype=np.float64)
            k0 = max(float(np.percentile(np.abs(kr[np.isfinite(kr)]), 25)) if np.any(np.isfinite(kr)) else 1e-3, 1e-3)
            uh = np.sign(kh) * np.log10(1.0 + np.abs(kh) / k0)
            ur = np.sign(kr) * np.log10(1.0 + np.abs(kr) / k0)
            ulim = float(np.nanpercentile(np.abs(np.concatenate([uh, ur])), 99.5)) + 1e-6
            gu = np.linspace(-ulim, ulim, 160)
            dh, dr = _kde(uh, gu), _kde(ur, gu)
            axr.fill_betweenx(gu, 0, dh, color=col[op], alpha=0.25, lw=0)
            axr.plot(dh, gu, color=col[op], lw=0.9)
            axr.plot(dr, gu, color="0.2", lw=0.8, ls=(0, (3.0, 2.0)))
            axr.set_xscale("symlog", linthresh=max(float(np.max(np.concatenate([dh, dr]))) * 1e-2, 1e-6))
            axr.set_xticks([])
            axr.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            axr.set_yticks([])
            axr.set_ylim(-ulim, ulim)
            axr.axhline(0.0, color="0.6", lw=0.3)
            axr.text(0.97, 0.98, "\u03ba>0", transform=axr.transAxes, ha="right", va="top", fontsize=5)
            axr.text(0.97, 0.02, "\u03ba<0", transform=axr.transAxes, ha="right", va="bottom", fontsize=5)
            axr.spines["left"].set_visible(False)
            axr.spines["bottom"].set_visible(False)
            axr.text(0.5, 1.0, "κ(t)", transform=axr.transAxes, ha="center", va="bottom", fontsize=6, fontweight="bold")
        handles = [Line2D([0], [0], color=col["full"], lw=1.1, label="Full PEBR-Net"),
                   Line2D([0], [0], color=col["no_cdmr"], lw=1.1, ls=ls_op["no_cdmr"], label="CDM-R removed"),
                   Line2D([0], [0], color=col["no_egsr"], lw=1.1, ls=ls_op["no_egsr"], label="E-GSR removed"),
                   Line2D([0], [0], color="0.2", lw=0.8, ls=(0, (3.0, 2.0)), label="Clean reference / common fan")]
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False)
        _save_figure(fig, fig_dir, stem)
        plt.close(fig)
    logger.info("%s rendered (fan level %.1f%% withheld; %d input-quality clusters).", stem,
                float(levels[fan_li]["withheld_percent"]), int(len(set(int(v) for v in clusters if v >= 0))))


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
