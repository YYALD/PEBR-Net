# PEBR-Net -- prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.
# MIT License, see LICENSE.
"""Evaluation and diagnostics on known-truth data.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

_LAST_ANOMALY_SUITE: Dict[str, Any] = {"pass": None, "reason": "not run"}


def _ood_anomaly_profile(
    kind: str,
    stations: np.ndarray,
    gates: int,
    late0: int,
    rng: np.random.Generator,
    gate_times: Optional[np.ndarray] = None,
    contrast: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    g = np.arange(gates, dtype=np.float64)
    s = np.asarray(stations, dtype=np.float64)
    c = float(np.median(s)) + float(rng.uniform(-3.0, 3.0))
    pol = "positive"
    if contrast is None:
        contrast = float(rng.uniform(0.5, 1.6))
    if kind == "trapezoid":
        half, ramp = float(rng.uniform(1.5, 3.5)), float(rng.uniform(1.0, 2.5))
        env = np.clip((half + ramp - np.abs(s - c)) / max(ramp, 1e-6), 0.0, 1.0)
        u = np.clip((g - (late0 - 3)) / max(gates - 1 - (late0 - 3), 1e-9), 0.0, 1.0)
        prof = u * u * (3.0 - 2.0 * u)
    elif kind == "double_body":
        c2 = c + float(rng.uniform(4.0, 8.0)) * (1.0 if rng.random() < 0.5 else -1.0)
        w1, w2 = float(rng.uniform(1.0, 2.2)), float(rng.uniform(1.0, 2.2))
        env = np.maximum(np.exp(-0.5 * ((s - c) / w1) ** 2),
                         0.75 * np.exp(-0.5 * ((s - c2) / w2) ** 2))
        u = np.clip((g - (late0 - 3)) / max(gates - 1 - (late0 - 3), 1e-9), 0.0, 1.0)
        prof = u * u * (3.0 - 2.0 * u)
    elif kind == "bump":
        w = float(rng.uniform(1.2, 3.0))
        env = np.exp(-0.5 * ((s - c) / w) ** 2)
        mu = float(late0 + 0.45 * (gates - late0))
        prof = np.exp(-0.5 * ((g - mu) / max(0.18 * (gates - late0), 1.0)) ** 2)
    elif kind == "suppression":
        w = float(rng.uniform(1.2, 3.2))
        env = np.exp(-0.5 * ((s - c) / w) ** 2)
        u = np.clip((g - (late0 - 2)) / max(gates - 1 - (late0 - 2), 1e-9), 0.0, 1.0)
        prof = u * u * (3.0 - 2.0 * u)
        pol = "negative"
    elif kind == "late_onset":
        w = float(rng.uniform(1.2, 3.0))
        env = np.exp(-0.5 * ((s - c) / w) ** 2)
        onset = int(late0 + 0.5 * (gates - late0))
        u = np.clip((g - onset) / max(gates - 1 - onset, 1e-9), 0.0, 1.0)
        prof = u * u * (3.0 - 2.0 * u)
    else:
        w = float(rng.uniform(1.5, 4.0))
        env = np.exp(-0.5 * ((s - c) / w) ** 2)
        u = np.clip((g - (late0 - 3)) / max(gates - 1 - (late0 - 3), 1e-9), 0.0, 1.0)
        prof = u * u * (3.0 - 2.0 * u)
        if rng.random() < 0.25:
            pol = "negative"
    sign = -1.0 if pol == "negative" else 1.0
    lift = 1.0 + sign * float(contrast) * env[:, None] * prof[None, :]
    return lift, env, pol


def _tau_shift_profile(
    background: np.ndarray,
    stations: np.ndarray,
    gate_times: np.ndarray,
    late0: int,
    rng: np.random.Generator,
    contrast: float,
) -> Tuple[np.ndarray, np.ndarray, str]:
    t = np.asarray(gate_times, dtype=np.float64)
    lt = np.log(np.maximum(t, 1e-30))
    lam = 1.0 + max(float(contrast), 1e-6) * 1.5
    w = float(rng.uniform(1.2, 3.0))
    c = float(np.median(stations)) + float(rng.uniform(-3.0, 3.0))
    env = np.exp(-0.5 * ((np.asarray(stations, dtype=np.float64) - c) / w) ** 2)
    out = np.empty((len(stations), background.shape[1]), dtype=np.float64)
    for j in range(len(stations)):
        for _row in (0,):
            b = background[j]
            shifted = np.interp(lt - math.log(lam), lt, b)
            a = np.zeros_like(b)
            a[late0:] = env[j]
            out[j] = b * (1.0 - a) + shifted * a
    return out, env, "positive"


def _ip_reversal_profile(
    background: np.ndarray,
    stations: np.ndarray,
    gate_times: np.ndarray,
    late0: int,
    rng: np.random.Generator,
    contrast: float,
) -> Tuple[np.ndarray, np.ndarray, str]:
    t = np.asarray(gate_times, dtype=np.float64)
    tau_ip = float(t[late0] * rng.uniform(1.5, 4.0))
    v_ip = np.exp(-t / max(tau_ip, 1e-30))
    v_ip = v_ip / max(float(np.max(np.abs(v_ip[late0:]))), 1e-30)
    w = float(rng.uniform(1.2, 3.0))
    c = float(np.median(stations)) + float(rng.uniform(-3.0, 3.0))
    env = np.exp(-0.5 * ((np.asarray(stations, dtype=np.float64) - c) / w) ** 2)
    ref = np.median(np.abs(background[:, late0:]), axis=1)
    strength = max(float(contrast), 1e-4)
    out = background.copy()
    out[:, late0:] = (background[:, late0:]
                      - (env[:, None] * strength * ref[:, None]) * v_ip[None, late0:])
    return out, env, "negative"


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Lower 95% confidence bound of a binomial rate."""
    n = max(int(trials), 1)
    p = float(successes) / n
    den = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0))
    return float(max(0.0, (centre - half) / den))


def wilson_upper_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Upper 95% confidence bound of a binomial rate."""
    n = max(int(trials), 1)
    p = float(successes) / n
    den = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0))
    return float(min(1.0, (centre + half) / den))


SUITE_SNR_STRATA: Tuple[Tuple[str, float, float], ...] = (
    ("snr_-5_0", -5.0, 0.0), ("snr_0_5", 0.0, 5.0),
    ("snr_5_10", 5.0, 10.0), ("snr_10_20", 10.0, 20.0),
)


SUITE_OOD_FAMILIES: Tuple[str, ...] = (
    "trapezoid", "double_body", "tau_shift", "bump", "suppression", "late_onset", "ip_reversal",
)


SUITE_WEAK_STRATA: Tuple[Tuple[str, float, float], ...] = (
    ("very_weak", 0.02, 0.05), ("weak", 0.05, 0.10),
    ("mid_weak", 0.10, 0.20), ("moderate", 0.20, 0.60),
)


def run_anomaly_preservation_suite(
    model: PEBRNet,
    cfg: Config,
    run_dir: Path,
    clean_cache: CleanCacheInfo,
    noise_cache: Optional["NoiseCacheInfo"],
    logger: logging.Logger,
    n_per_family: Optional[int] = None,
    n_per_weak: Optional[int] = None,
    n_quiet: Optional[int] = None,
    n_stations: Optional[int] = None,
) -> Dict[str, Any]:
    """Ground-truth anomaly-qualification suite (E11)."""
    global _LAST_ANOMALY_SUITE_PASS, _LAST_ANOMALY_SUITE
    rt = cfg.runtime
    n_per_family = int(rt.anomaly_suite_per_family if n_per_family is None else n_per_family)
    n_per_weak = int(rt.anomaly_suite_per_weak if n_per_weak is None else n_per_weak)
    n_quiet = int(rt.anomaly_suite_quiet if n_quiet is None else n_quiet)
    n_stations = int(rt.anomaly_suite_stations if n_stations is None else n_stations)
    device = resolve_device(cfg.runtime.device)
    model.eval()
    rng = np.random.default_rng(20260718)
    lib = np.load(clean_cache.data_path, mmap_mode="r")
    gates = int(lib.shape[1])
    late0 = int(cfg.data.late_start_index)
    gate_times = np.asarray(clean_cache.target_time, dtype=np.float64).reshape(-1)
    gate_times_s = gate_times if float(gate_times[-1]) < 1.0 else gate_times / 1000.0

    splits = build_splits(clean_cache, cfg)
    pool = np.asarray(splits.get("test_pool", splits["test"]), dtype=np.int64)
    if "test_pool" in splits and len(splits["test_pool"]) > len(splits["test"]):
        logger.info(
            "Anomaly suite backgrounds read the station-contiguous test "
            "POOL (%d rows) instead of the de-duplicated split (%d): the suite needs "
            "a real station axis, and the pool holds no train-side shape.",
            int(pool.size), int(len(splits["test"])))
    if pool.size < int(rt.anomaly_suite_min_backgrounds):
        raise RuntimeError(
            "anomaly qualification requires a held-out test split with at least %d curves; "
            "only %d are available (a whole-cache fallback would grade "
            "anomaly recovery on curves the model had trained on). Fix the split "
            "(--split_group_size / --split_min_groups) instead of weakening the test."
            % (int(rt.anomaly_suite_min_backgrounds), int(pool.size))
        )

    if noise_cache is None:
        raise RuntimeError(
            "anomaly qualification requires the noise cache: the suite composes its "
            "observations from the real noise library, not from white Gaussian noise."
        )
    noise_mm = np.load(noise_cache.data_path, mmap_mode="r")
    n_noise = int(noise_mm.shape[0])
    blocks = list(getattr(noise_cache, "block_rows", None) or [n_noise])
    bounds = np.concatenate([[0], np.cumsum(np.asarray(blocks, dtype=np.int64))])
    k = int(cfg.data.num_neighbors)
    add_stack = bool(getattr(cfg.data, "coherent_stack_slot", False))
    scale = float(clean_cache.global_scale)
    thr = float(rt.anomaly_detection_contrast)

    def _stations() -> np.ndarray:
        base = np.arange(n_stations, dtype=np.float64)
        return base + rng.normal(0.0, float(rt.anomaly_suite_spacing_jitter), size=n_stations)

    def _observe(clean_prof: np.ndarray, snr_db: float) -> np.ndarray:
        s_n = clean_prof.shape[0]
        bi = int(rng.integers(0, len(bounds) - 1))
        bstart, blen = int(bounds[bi]), int(bounds[bi + 1] - bounds[bi])
        if blen < 1:
            bstart, blen = 0, n_noise
        start_local = int(rng.integers(0, blen))
        sgn = -1.0 if (cfg.data.random_noise_sign_flip and rng.random() < 0.5) else 1.0
        roll_k = (int(rng.integers(1, gates))
                  if rng.random() < cfg.data.noise_circular_shift_probability else 0)
        mix_delta = (int(rng.integers(1, blen))
                     if (rng.random() < cfg.data.noise_mixup_probability and blen > 1) else None)
        mix_sgn = -1.0 if (cfg.data.random_noise_sign_flip and rng.random() < 0.5) else 1.0
        mix_beta = float(rng.uniform(*cfg.data.noise_mixup_beta)) if mix_delta is not None else 0.0
        white_ratio = (float(rng.uniform(*cfg.data.white_noise_relative_rms))
                       if rng.random() < cfg.data.white_noise_probability else None)
        noise = np.empty_like(clean_prof)
        for j in range(s_n):
            loc = (start_local + j) % blen
            _mix = noise_mm[bstart + ((loc + mix_delta) % blen)] if mix_delta is not None else None
            noise[j] = compose_noise_trace(
                noise_mm[bstart + loc], _mix, cfg.data, rng, sgn=sgn, roll_k=roll_k,
                mix_sgn=mix_sgn, mix_beta=mix_beta, white_ratio=white_ratio,
                gate_times_s=gate_times_s,
            )
        ref_rms = float(np.median(np.sqrt(np.mean(clean_prof**2, axis=1))))
        n_rms = max(float(np.sqrt(np.mean(noise**2))), cfg.data.minimum_noise_rms)
        gain = ref_rms / ((10.0 ** (snr_db / 20.0)) * n_rms)
        return clean_prof + noise * gain

    def _forward(obs: np.ndarray, stations: np.ndarray) -> np.ndarray:
        nb_idx = build_neighbor_indices(len(stations), k)
        geo = neighbor_geometry_from_stations(
            stations, nb_idx, add_stack=add_stack,
            spacing=float(station_median_spacing(stations)) or 1.0)
        xb = torch.from_numpy((obs / scale).astype(np.float32))[:, None, :].to(device)
        nbt = torch.from_numpy((obs[nb_idx] / scale).astype(np.float32)).to(device)
        if add_stack:
            stack = nbt.sum(dim=1, keepdim=True) / float(nbt.shape[1])
            nbt = torch.cat([nbt, stack], dim=1)
        gt = torch.from_numpy(np.asarray(geo, dtype=np.float32)).to(device)
        _dn_s = torch.from_numpy(
            (np.arange(len(stations), dtype=np.float32)
             / float(max(1, len(stations) - 1)))[:, None]).to(device)
        with torch.no_grad():
            out = model(xb, nbt, gt, depth_norm=_dn_s,
                        profile_len=(int(len(stations)) if len(stations) > 2 else None))
        return out["denoised"].detach().cpu().numpy()[:, 0, :] * scale

    def _late_contrast(prof: np.ndarray, background: np.ndarray) -> np.ndarray:
        r = prof[:, late0:] / np.maximum(np.abs(background[:, late0:]), 1e-30)
        return np.mean(r, axis=1) - 1.0

    BOUNDS = {"peak_ratio": (0.8, 1.2), "area_ratio": (0.8, 1.2), "width_ratio": (0.75, 1.25)}

    def _score(den: np.ndarray, clean_prof: np.ndarray, background: np.ndarray,
               polarity: str, family: str) -> Dict[str, float]:
        L, Lt = _late_contrast(den, background), _late_contrast(clean_prof, background)
        sgn = -1.0 if polarity == "negative" else 1.0
        Ls, Lts = sgn * L, sgn * Lt
        peak_t = float(np.max(Lts))
        c_true, c_hat = int(np.argmax(Lts)), int(np.argmax(Ls))
        w_true = float(np.sum(Lts >= 0.5 * peak_t))
        w_hat = float(np.sum(Ls >= 0.5 * max(float(np.max(Ls)), 1e-9)))
        mid = late0 + max((gates - late0) // 2, 1)
        La = np.mean(den[:, late0:mid] / np.maximum(np.abs(background[:, late0:mid]), 1e-30), axis=1)
        Lb = np.mean(den[:, mid:] / np.maximum(np.abs(background[:, mid:]), 1e-30), axis=1)
        cons = float(np.corrcoef(La, Lb)[0, 1]) if np.std(La) > 0 and np.std(Lb) > 0 else 0.0
        i_t, i_p = int(np.argmax(np.abs(Lt))), int(np.argmax(np.abs(L)))
        sign_ok = float(np.sign(Lt[i_t]) == np.sign(L[i_p]))
        peak_ratio = float(np.max(Ls) / max(peak_t, 1e-9))
        area_ratio = float(np.sum(np.clip(Ls, 0, None)) / max(np.sum(np.clip(Lts, 0, None)), 1e-9))
        width_ratio = w_hat / max(w_true, 1.0)
        centre_shift = float(abs(c_hat - c_true))
        detected = float(np.max(np.abs(L)) >= thr and centre_shift <= 1.0)
        fidelity_ok = float(
            detected > 0.0 and sign_ok > 0.0
            and BOUNDS["peak_ratio"][0] <= peak_ratio <= BOUNDS["peak_ratio"][1]
            and BOUNDS["area_ratio"][0] <= area_ratio <= BOUNDS["area_ratio"][1]
            and BOUNDS["width_ratio"][0] <= width_ratio <= BOUNDS["width_ratio"][1]
            and cons >= 0.90
        )
        def _n_extrema(v: np.ndarray) -> int:
            if v.size < 3:
                return 0
            amp = max(float(np.max(np.abs(v))), 1e-12)
            d = np.diff(v)
            sig = np.abs(v[1:-1]) >= 0.25 * amp
            return int(np.sum((d[:-1] > 0) & (d[1:] < 0) & sig))
        new_extrema = float(max(_n_extrema(Ls) - _n_extrema(Lts), 0))
        out = {"peak_ratio": peak_ratio, "area_ratio": area_ratio, "center_shift": centre_shift,
               "width_ratio": width_ratio, "consistency": cons, "sign_agreement": sign_ok,
               "detected": detected, "recovered": fidelity_ok, "new_extrema": new_extrema,
               "realized_contrast": float(np.max(np.abs(Lt)))}
        if family == "double_body":
            def _peaks(v: np.ndarray) -> List[int]:
                amp = max(float(np.max(np.abs(v))), 1e-12)
                return [int(k) for k in range(1, len(v) - 1)
                        if v[k] > v[k - 1] and v[k] >= v[k + 1] and v[k] >= 0.25 * amp]
            pt, ph = _peaks(Lts), _peaks(Ls)
            matched, errs = 0, []
            for c_t in pt:
                if ph:
                    d = min(abs(c_t - c_p) for c_p in ph)
                    if d <= 1:
                        matched += 1
                        errs.append(float(d))
            out["bodies_true"] = float(len(pt))
            out["bodies_matched"] = float(matched)
            out["bodies_recovered"] = float(len(pt) > 0 and matched == len(pt)
                                            and len(ph) <= len(pt))
            out["body_center_error"] = float(np.mean(errs)) if errs else float("nan")
            out["recovered"] = float(out["recovered"] > 0.0 and out["bodies_recovered"] > 0.0)
        return out

    pool_sorted = np.sort(pool)
    runs: List[Tuple[int, int]] = []
    if pool_sorted.size:
        start = prev = int(pool_sorted[0])
        for v in pool_sorted[1:]:
            v = int(v)
            if v == prev + 1:
                prev = v
                continue
            runs.append((start, prev))
            start = prev = v
        runs.append((start, prev))
    usable = [(a, b) for a, b in runs if (b - a + 1) >= n_stations]
    if not usable:
        raise RuntimeError(
            "anomaly qualification needs CONTIGUOUS held-out station segments of at least %d "
            "rows: the longest run in the test split is %d. "
            "Reduce --anomaly_suite_stations or raise --split_group_size."
            % (n_stations, max((b - a + 1) for a, b in runs) if runs else 0)
        )
    seg_rows = int(sum(b - a + 1 for a, b in usable))

    def _background(s_n: int) -> np.ndarray:
        a, b = usable[int(rng.integers(0, len(usable)))]
        s0 = int(rng.integers(a, b - s_n + 2))
        seg = np.asarray(lib[s0 : s0 + s_n], dtype=np.float64).copy()
        return seg * np.exp(rng.normal(0.0, 0.02, size=(s_n, 1)))

    rows: List[Dict[str, Any]] = []
    strata = SUITE_SNR_STRATA
    _case_counter = {"i": 0}

    unplaceable: List[str] = []

    def _contrast_bin(c: float) -> str:
        for name, lo_c, hi_c in SUITE_WEAK_STRATA:
            if lo_c - 1e-12 <= c < hi_c:
                return name
        return "strong" if c >= SUITE_WEAK_STRATA[-1][2] else "sub_threshold"

    def _make_profile(family: str, bg: np.ndarray, st: np.ndarray, con: float):
        if family == "tau_shift":
            return _tau_shift_profile(bg, st, gate_times, late0, rng, con)
        if family == "ip_reversal":
            return _ip_reversal_profile(bg, st, gate_times, late0, rng, con)
        lift, env, pol = _ood_anomaly_profile(family, st, gates, late0, rng,
                                              gate_times=gate_times, contrast=con)
        return bg * lift, env, pol

    def _realized(clean_prof: np.ndarray, bg: np.ndarray) -> float:
        return float(np.max(np.abs(_late_contrast(clean_prof, bg))))

    def _calibrate(family: str, bg: np.ndarray, st: np.ndarray,
                   target_lo: float, target_hi: float) -> Tuple[np.ndarray, str, float, bool]:
        target = math.sqrt(max(target_lo, 1e-6) * target_hi)
        con = target
        prof, _e, pol = _make_profile(family, bg, st, con)
        for _ in range(6):
            r = _realized(prof, bg)
            if target_lo <= r < target_hi:
                return prof, pol, r, True
            con = float(np.clip(con * (target / max(r, 1e-6)), 1e-4, 50.0))
            prof, _e, pol = _make_profile(family, bg, st, con)
        return prof, pol, _realized(prof, bg), False

    def _run_case(family: str, contrast: Optional[float], weak_label: Optional[str] = None,
                  target_bin: Optional[Tuple[str, float, float]] = None,
                  snr_stratum: Optional[Tuple[str, float, float]] = None) -> None:
        st = _stations()
        bg = _background(n_stations)
        if snr_stratum is None:
            name, lo, hi = strata[_case_counter["i"] % len(strata)]
            _case_counter["i"] += 1
        else:
            name, lo, hi = snr_stratum
        snr = float(rng.uniform(lo, hi))
        placed = True
        if target_bin is not None:
            _bn, _blo, _bhi = target_bin
            clean_prof, pol, _r, placed = _calibrate(family, bg, st, _blo, _bhi)
            con = float(_r)
        else:
            con = float(rng.uniform(0.5, 1.6)) if contrast is None else float(contrast)
            clean_prof, _env, pol = _make_profile(family, bg, st, con)
        obs = _observe(clean_prof, snr)
        if rng.random() < float(rt.anomaly_suite_dead_trace_prob):
            obs[int(rng.integers(0, n_stations))] = 0.0
        m_ = _score(_forward(obs, st), clean_prof, bg, pol, family)
        m_.update({"family": family, "snr_db": snr, "snr_stratum": name,
                   "requested_contrast": con, "polarity": pol, "placed_in_target_bin": placed,
                   "requested_stratum": weak_label or "native",
                   "contrast_bin": _contrast_bin(float(m_["realized_contrast"]))})
        if target_bin is not None and not placed:
            unplaceable.append("%s|%s|%s (realized %.3f)" % (family, target_bin[0], name,
                                                            float(m_["realized_contrast"])))
        rows.append(m_)

    for fam in ("id_gauss",) + SUITE_OOD_FAMILIES:
        for _r in range(n_per_family):
            _run_case(fam, None)
    per_cell = int(rt.anomaly_suite_per_cell)
    per_diag = int(rt.anomaly_suite_per_diag_cell)
    _req_c = float(rt.weak_anomaly_required_contrast)
    for wname, lo_c, hi_c in SUITE_WEAK_STRATA:
        gated = lo_c >= _req_c - 1e-9
        count = per_cell if gated else per_diag
        for fam in SUITE_OOD_FAMILIES:
            for snr_s in strata:
                for _r in range(count):
                    _run_case(fam, None, weak_label=wname,
                              target_bin=(wname, lo_c, hi_c), snr_stratum=snr_s)

    quiet_rows: List[Dict[str, Any]] = []
    quiet_hits_station, quiet_n_station = 0, 0
    for qi in range(n_quiet):
        st = _stations()
        bg = _background(n_stations)
        name, lo, hi = strata[qi % len(strata)]
        L = _late_contrast(_forward(_observe(bg, float(rng.uniform(lo, hi))), st), bg)
        hits = int(np.sum(np.abs(L) > thr))
        quiet_hits_station += hits
        quiet_n_station += int(L.size)
        quiet_rows.append({"snr_stratum": name, "any_hit": float(hits > 0),
                           "worst_contrast": float(np.max(np.abs(L)))})

    KEYS = ("peak_ratio", "area_ratio", "center_shift", "width_ratio", "consistency")

    def _agg(sel: List[Dict[str, Any]]) -> Dict[str, float]:
        if not sel:
            return {kk: float("nan") for kk in KEYS} | {
                "n": 0, "sign_agreement": float("nan"), "detection_rate": float("nan"),
                "recovery_rate": float("nan"), "recovery_lower95": float("nan"),
                "sign_lower95": float("nan"), "new_extrema_rate": float("nan")}
        out = {kk: float(np.median([r[kk] for r in sel])) for kk in KEYS}
        n = len(sel)
        rec = int(sum(r["recovered"] for r in sel))
        sgn = int(sum(r["sign_agreement"] for r in sel))
        out["n"] = n
        out["sign_agreement"] = float(sgn) / n
        out["detection_rate"] = float(np.mean([r["detected"] for r in sel]))
        out["recovery_rate"] = float(rec) / n
        out["recovery_lower95"] = wilson_lower_bound(rec, n)
        out["sign_lower95"] = wilson_lower_bound(sgn, n)
        out["new_extrema_rate"] = float(np.mean([r["new_extrema"] > 0 for r in sel]))
        if any("bodies_recovered" in r for r in sel):
            bb = [r["bodies_recovered"] for r in sel if "bodies_recovered" in r]
            out["both_bodies_rate"] = float(np.mean(bb))
            out["both_bodies_lower95"] = wilson_lower_bound(int(sum(bb)), len(bb))
        return out

    def _gate(med: Dict[str, float], require_recovery: bool = True) -> Dict[str, bool]:
        if int(med.get("n", 0)) < int(rt.anomaly_suite_min_per_stratum):
            return {"sufficient_samples": False}
        g = {"peak": BOUNDS["peak_ratio"][0] <= med["peak_ratio"] <= BOUNDS["peak_ratio"][1],
             "area": BOUNDS["area_ratio"][0] <= med["area_ratio"] <= BOUNDS["area_ratio"][1],
             "center": med["center_shift"] <= 1.0,
             "width": BOUNDS["width_ratio"][0] <= med["width_ratio"] <= BOUNDS["width_ratio"][1],
             "consistency": med["consistency"] >= 0.90,
             "sign": med["sign_lower95"] >= float(rt.anomaly_sign_agreement_min),
             "new_extrema": med["new_extrema_rate"] <= float(rt.anomaly_new_extrema_max)}
        if require_recovery:
            g["recovery"] = med["recovery_lower95"] >= float(rt.weak_anomaly_recall_min)
        if "both_bodies_lower95" in med:
            g["both_bodies"] = med["both_bodies_lower95"] >= float(rt.weak_anomaly_recall_min)
        return g

    native = [r for r in rows if r["requested_stratum"] == "native"]
    per_family = {f: _agg([r for r in native if r["family"] == f])
                  for f in ("id_gauss",) + SUITE_OOD_FAMILIES}
    family_gates = {f: _gate(per_family[f]) for f in SUITE_OOD_FAMILIES}
    per_snr = {nm: _agg([r for r in native if r["snr_stratum"] == nm and r["family"] != "id_gauss"])
               for nm, _, _ in strata}
    snr_gates = {nm: _gate(per_snr[nm]) for nm in per_snr}

    req_c = float(rt.weak_anomaly_required_contrast)
    gated_bins = [w for w, lo_c, _hi in SUITE_WEAK_STRATA if lo_c >= req_c - 1e-9]
    cells: Dict[str, Dict[str, float]] = {}
    cell_gates: Dict[str, Dict[str, bool]] = {}
    for fam in SUITE_OOD_FAMILIES + ("id_gauss",):
        for wname, _lo, _hi in SUITE_WEAK_STRATA:
            for nm, _a, _b in strata:
                sel = [r for r in rows if r["family"] == fam and r["contrast_bin"] == wname
                       and r["snr_stratum"] == nm]
                if not sel:
                    continue
                key = f"{fam}|{wname}|{nm}"
                cells[key] = _agg(sel)
    required_cells = [_ck for _ck in cells
                      if _ck.split("|")[1] in gated_bins and not _ck.startswith("id_gauss|")]
    for _ck in required_cells:
        cell_gates[_ck] = _gate(cells[_ck])
    missing_cells = [f"{fam}|{w}|{nm}" for fam in SUITE_OOD_FAMILIES for w in gated_bins
                     for nm, _a, _b in strata if f"{fam}|{w}|{nm}" not in cells]
    _unplaceable = sorted(set(unplaceable))

    per_weak = {w: _agg([r for r in rows if r["contrast_bin"] == w])
                for w, _, _ in SUITE_WEAK_STRATA}
    weak_by_snr = {f"{w}|{nm}": _agg([r for r in rows if r["contrast_bin"] == w
                                      and r["snr_stratum"] == nm])["recovery_lower95"]
                   for w, _, _ in SUITE_WEAK_STRATA for nm, _a, _b in strata}
    weak_gates = {}
    for w in gated_bins:
        ks = [k for k in required_cells if k.split("|")[1] == w]
        weak_gates[w] = bool(ks) and all(all(cell_gates[k].values()) for k in ks)
    required_floor = req_c
    empirical_floor = float("nan")
    for w, lo_c, _hi in SUITE_WEAK_STRATA:
        med = per_weak[w]
        if (int(med.get("n", 0)) >= int(rt.anomaly_suite_min_per_stratum)
                and med["recovery_lower95"] >= float(rt.weak_anomaly_recall_min)
                and not math.isfinite(empirical_floor)):
            empirical_floor = lo_c

    ip_rows = [r for r in rows if r["family"] == "ip_reversal"]
    ip_sign = (wilson_lower_bound(int(sum(r["sign_agreement"] for r in ip_rows)), len(ip_rows))
               if ip_rows else float("nan"))
    q_prof_hits = int(sum(r["any_hit"] for r in quiet_rows))
    false_rate_profile = float(q_prof_hits) / max(len(quiet_rows), 1)
    false_upper = wilson_upper_bound(q_prof_hits, len(quiet_rows))
    false_rate_station = float(quiet_hits_station) / max(quiet_n_station, 1)

    ood_pass = all(all(g.values()) for g in family_gates.values())
    snr_pass = all(all(g.values()) for g in snr_gates.values())
    cell_pass = bool(cell_gates) and not missing_cells and all(
        all(g.values()) for g in cell_gates.values())
    weak_pass = bool(weak_gates) and all(weak_gates.values()) and cell_pass
    ip_pass = bool(math.isfinite(ip_sign) and ip_sign >= float(rt.ip_sign_preservation_min))
    quiet_pass = false_upper <= float(rt.quiet_false_rate_max)
    passed = bool(ood_pass and snr_pass and weak_pass and ip_pass and quiet_pass)

    payload: Dict[str, Any] = {
        "pass": passed,
        "anomaly_id_pass": bool(all(_gate(per_family["id_gauss"]).values())),
        "anomaly_ood_pass": bool(ood_pass),
        "anomaly_snr_stratum_pass": bool(snr_pass),
        "anomaly_cell_grid_pass": bool(cell_pass),
        "weak_anomaly_pass": bool(weak_pass),
        "ip_sign_preservation_pass": bool(ip_pass),
        "quiet_false_rate_pass": bool(quiet_pass),
        "per_family": per_family, "family_gates": family_gates,
        "per_snr_stratum": per_snr, "snr_gates": snr_gates,
        "cells": cells, "cell_gates": cell_gates,
        "required_cells": required_cells, "missing_required_cells": missing_cells,
        "unplaceable_cases": _unplaceable,
        "per_contrast_bin": per_weak, "weak_gates": weak_gates,
        "weak_recovery_lower95_by_snr": weak_by_snr,
        "required_contrast_floor": required_floor,
        "empirically_qualified_contrast_floor": empirical_floor,
        "ip_sign_agreement_lower95": ip_sign,
        "false_anomaly_rate_profile": false_rate_profile,
        "false_anomaly_rate_wilson_upper95": false_upper,
        "false_anomaly_rate_station_diagnostic": false_rate_station,
        "quiet_profiles": len(quiet_rows), "quiet_profile_hits": q_prof_hits,
        "quiet_station_tests": int(quiet_n_station),
        "quiet_worst_contrast": float(max((r["worst_contrast"] for r in quiet_rows), default=0.0)),
        "statistical_unit": "profile (stations within a profile share background, noise block "
                            "and neighbour attention, so they are not independent)",
        "detection_threshold_contrast": thr,
        "detection_threshold_source": "cfg.runtime.anomaly_detection_contrast (fixed)",
        "contrast_binning": "REALIZED max|dV/V| in the late window, not the requested "
                            "generator parameter",
        "thresholds": {"peak": list(BOUNDS["peak_ratio"]), "area": list(BOUNDS["area_ratio"]),
                       "center_max": 1.0, "width": list(BOUNDS["width_ratio"]),
                       "consistency_min": 0.90,
                       "sign_agreement_lower95_min": float(rt.anomaly_sign_agreement_min),
                       "recovery_lower95_min": float(rt.weak_anomaly_recall_min),
                       "new_extrema_rate_max": float(rt.anomaly_new_extrema_max),
                       "required_contrast_floor": req_c,
                       "ip_sign_lower95_min": float(rt.ip_sign_preservation_min),
                       "quiet_false_rate_upper_max": float(rt.quiet_false_rate_max),
                       "min_samples_per_cell": int(rt.anomaly_suite_min_per_stratum)},
        "background_pool": "contiguous held-out test segments (%d rows in %d usable runs)"
                           % (seg_rows, len(usable)),
        "noise_source": str(noise_cache.data_path),
        "snr_strata": [list(s) for s in strata],
        "num_neighbors": k, "coherent_stack_slot": add_stack, "n_stations": int(n_stations),
        "n_per_family": n_per_family, "n_per_weak": n_per_weak, "n_quiet": n_quiet,
        "profiles": rows,
    }
    atomic_json_dump(payload, run_dir / "reports" / "anomaly_preservation_metrics.json")
    atomic_json_dump({kk: payload[kk] for kk in
                      ("anomaly_ood_pass", "per_family", "family_gates", "per_snr_stratum",
                       "snr_gates", "thresholds", "background_pool", "noise_source")},
                     run_dir / "reports" / "anomaly_ood_metrics.json")
    atomic_json_dump({kk: payload[kk] for kk in
                      ("weak_anomaly_pass", "anomaly_cell_grid_pass", "per_contrast_bin",
                       "weak_gates", "cells", "cell_gates", "required_cells",
                       "missing_required_cells", "weak_recovery_lower95_by_snr",
                       "required_contrast_floor", "empirically_qualified_contrast_floor",
                       "contrast_binning", "detection_threshold_contrast", "thresholds")},
                     run_dir / "reports" / "weak_anomaly_metrics.json")
    atomic_json_dump({kk: payload[kk] for kk in
                      ("quiet_false_rate_pass", "false_anomaly_rate_profile",
                       "false_anomaly_rate_wilson_upper95", "false_anomaly_rate_station_diagnostic",
                       "quiet_profiles", "quiet_profile_hits", "quiet_station_tests",
                       "quiet_worst_contrast", "statistical_unit",
                       "detection_threshold_contrast", "detection_threshold_source")},
                     run_dir / "reports" / "quiet_false_positive_metrics.json")
    logger.info(
        "ANOMALY QUALIFICATION SUITE | %s | backgrounds: %s | noise: real library "
        "| K=%d stack=%s | SNR strata %s | binning: realized contrast | unit: profile",
        "PASS" if passed else "FAIL", payload["background_pool"], k, add_stack,
        ", ".join(nm for nm, _, _ in strata),
    )
    for f in SUITE_OOD_FAMILIES:
        med, g = per_family[f], family_gates[f]
        logger.info(
            "  OOD %-13s n=%3d peak %.3f area %.3f centre %.2f width %.3f cons %.3f "
            "sign>=%.2f recov>=%.2f newext %.2f -> %s",
            f, med["n"], med["peak_ratio"], med["area_ratio"], med["center_shift"],
            med["width_ratio"], med["consistency"], med["sign_lower95"],
            med["recovery_lower95"], med["new_extrema_rate"],
            "pass" if all(g.values()) else "FAIL(%s)" % ",".join(kk for kk, v in g.items() if not v),
        )
    for nm, _, _ in strata:
        med = per_snr[nm]
        logger.info("  SNR %-9s n=%3d peak %.3f centre %.2f cons %.3f recov>=%.2f -> %s", nm,
                    med["n"], med["peak_ratio"], med["center_shift"], med["consistency"],
                    med["recovery_lower95"],
                    "pass" if all(snr_gates[nm].values()) else "FAIL")
    for w, lo_c, hi_c in SUITE_WEAK_STRATA:
        med = per_weak[w]
        logger.info(
            "  CONTRAST %-10s (realized %.0f-%.0f%%) n=%3d detect %.2f recover %.2f "
            "(95%% lower %.2f) centre %.2f -> %s",
            w, 100 * lo_c, 100 * hi_c, med["n"], med["detection_rate"], med["recovery_rate"],
            med["recovery_lower95"], med["center_shift"],
            ("GATED:%s" % ("pass" if weak_gates.get(w) else "FAIL")) if w in weak_gates
            else "diagnostic (below the required floor)",
        )
    _failed_cells = [k for k, g in cell_gates.items() if not all(g.values())]
    logger.info(
        "  CELL GRID | %d required cells (family x realized contrast x SNR), "
        "%d passed, %d failed, %d never populated -> %s",
        len(required_cells), len(required_cells) - len(_failed_cells), len(_failed_cells),
        len(missing_cells), "pass" if cell_pass else "FAIL",
    )
    for k in _failed_cells[:12]:
        g = cell_gates[k]
        logger.info("    cell %-38s n=%3d FAIL(%s)", k, cells[k]["n"],
                    ",".join(kk for kk, v in g.items() if not v))
    if missing_cells:
        logger.warning(
            "    %d required cell(s) had NO samples: %s. The grid cannot certify a cell it "
            "never populated -- raise --anomaly_suite_per_cell, or accept that the family "
            "cannot physically produce that contrast and say so in the paper.",
            len(missing_cells), ", ".join(missing_cells[:6]))
    if _unplaceable:
        logger.info("    %d case(s) could not be calibrated into their target bin (e.g. %s)",
                    len(_unplaceable), "; ".join(_unplaceable[:3]))
    logger.info(
        "  IP sign preservation (95%% lower) %.2f >= %.2f | quiet false-anomaly %.1f%% of "
        "PROFILES (Wilson upper %.1f%% over %d profiles; station-level %.2f%% is a diagnostic "
        "only) | required contrast floor %.0f%% | empirically qualified floor: %s",
        ip_sign, float(rt.ip_sign_preservation_min), 100.0 * false_rate_profile,
        100.0 * false_upper, len(quiet_rows), 100.0 * false_rate_station,
        100.0 * required_floor,
        ("%.0f%%" % (100.0 * empirical_floor)) if math.isfinite(empirical_floor)
        else "NONE of the tested contrast bins reached the recall bound",
    )
    if not passed:
        logger.warning(
            "ANOMALY PRESERVATION NOT CERTIFIED : deployment requires EVERY OOD "
            "family, EVERY SNR stratum, EVERY required family x contrast x SNR cell at or above "
            "%.0f%% realized contrast, IP sign preservation and the profile-level quiet upper "
            "bound to pass. Field deployment is blocked (exit 25).", 100.0 * req_c,
        )
    _LAST_ANOMALY_SUITE_PASS = bool(passed)
    _LAST_ANOMALY_SUITE = payload
    return payload


def certify_anomaly_preservation(
    model: PEBRNet,
    cfg: Config,
    run_dir: Path,
    clean_cache: CleanCacheInfo,
    noise_cache: Optional["NoiseCacheInfo"],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run the anomaly-preservation suite; any outcome other than a pass is recorded as a failure."""
    global _LAST_ANOMALY_SUITE_PASS, _LAST_ANOMALY_SUITE
    if not bool(getattr(cfg.runtime, "deployment_gate", True)):
        _LAST_ANOMALY_SUITE_PASS = None
        _LAST_ANOMALY_SUITE = {"pass": None, "applicable": False,
                               "reason": "deployment gate off: the certification gates field products only"}
        logger.info("Anomaly-preservation certification not run: it gates field products, and the "
                    "deployment gate is off.")
        return _LAST_ANOMALY_SUITE
    if not bool(cfg.model.use_manifold_branch):
        _LAST_ANOMALY_SUITE_PASS = None
        _LAST_ANOMALY_SUITE = {"pass": None, "applicable": False,
                               "reason": "no manifold branch in this variant"}
        logger.info("Anomaly-preservation suite not applicable (manifold branch disabled).")
        return _LAST_ANOMALY_SUITE
    try:
        payload = run_anomaly_preservation_suite(model, cfg, run_dir, clean_cache,
                                                 noise_cache, logger)
        payload["applicable"] = True
        return payload
    except BaseException as exc:
        logger.exception("ANOMALY PRESERVATION SUITE CRASHED: %r", exc)
        _LAST_ANOMALY_SUITE_PASS = False
        _LAST_ANOMALY_SUITE = {"pass": False, "applicable": True,
                               "reason": "suite raised %r" % (exc,)}
        try:
            atomic_json_dump(_LAST_ANOMALY_SUITE, run_dir / "reports" / "anomaly_preservation_metrics.json")
        except Exception:
            pass
        return _LAST_ANOMALY_SUITE


def _np_signed_symlog(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def _per_trace_nrmse(pred: np.ndarray, ref: np.ndarray, cols: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)[:, cols]
    r = np.asarray(ref, dtype=np.float64)[:, cols]
    num = np.sum((p - r) ** 2, axis=1)
    den = np.maximum(np.sum(r**2, axis=1), eps)
    return 100.0 * np.sqrt(num / den)


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(v) + 1, dtype=np.float64) / max(len(v), 1)
    return v, y


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _nn_search(query: np.ndarray, bank: np.ndarray, chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.float32)
    best_d = np.full(q.shape[0], np.inf, dtype=np.float64)
    best_i = np.zeros(q.shape[0], dtype=np.int64)
    q_sq = np.sum(q.astype(np.float64) ** 2, axis=1)
    for start in range(0, bank.shape[0], chunk):
        blk = np.asarray(bank[start : start + chunk], dtype=np.float32)
        b_sq = np.sum(blk.astype(np.float64) ** 2, axis=1)
        cross = q.astype(np.float64) @ blk.astype(np.float64).T
        d = q_sq[:, None] + b_sq[None, :] - 2.0 * cross
        j = np.argmin(d, axis=1)
        dj = d[np.arange(q.shape[0]), j]
        upd = dj < best_d
        best_d[upd] = dj[upd]
        best_i[upd] = start + j[upd]
    return np.sqrt(np.maximum(best_d, 0.0)), best_i


def _eval_split_metrics(
    model: "PEBRNet",
    cfg: "Config",
    clean_cache: "CleanCacheInfo",
    noise_cache: "NoiseCacheInfo",
    indices: np.ndarray,
    seed: int,
    samples: int,
    loss_fn: "ProjectLoss",
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    n_take = int(min(samples, len(idx)))
    if n_take < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=n_take, replace=False))
    ds = BTEMDenoisingDataset(
        clean_cache.data_path,
        noise_cache.data_path,
        idx,
        clean_cache.global_scale,
        cfg.data,
        seed,
        n_take,
        False,
        noise_block_rows=noise_cache.block_rows,
        gate_times=noise_cache.target_time,
        real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
        real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
        province_id=getattr(clean_cache, "province_id", None),
        paired_noisy_path=getattr(clean_cache, "paired_noisy_path", ""),
        paired_background_path=getattr(clean_cache, "paired_background_path", ""),
        paired_event_path=getattr(clean_cache, "paired_event_path", ""),
        measured_noise_path=getattr(clean_cache, "measured_noise_path", ""),
        extra_clean_path=getattr(clean_cache, "extra_clean_path", ""),
    )
    ds.set_epoch(0)
    loader = DataLoader(ds, batch_size=block_aligned_batch_size(ds, min(cfg.train.batch_size, 256)),
                        shuffle=False, num_workers=0)
    late = int(cfg.data.late_start_index)
    gates = int(cfg.data.target_gates)
    late_cols = np.arange(late, gates)
    all_cols = np.arange(gates)
    preds: List[np.ndarray] = []
    cleans: List[np.ndarray] = []
    idxs: List[np.ndarray] = []
    _mans: List[np.ndarray] = []
    _dens: List[np.ndarray] = []
    _bls: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            out = model(batch["noisy"], batch.get("neighbors"), batch.get("neighbor_geometry"),
                  depth_norm=batch.get("depth_norm"), profile_len=batch_profile_len(batch))
            try:
                if "manifold_z" in out and "z_denoise" in out:
                    _gsn = model.gate_scale_norm.float()
                    _inv = (1.0 / out["trace_scale"].float()) if "trace_scale" in out else 1.0
                    _mans.append((signed_symexp(out["manifold_z"].float()) * _gsn * _inv).cpu().numpy()[:, 0, :])
                    _dens.append((signed_symexp(out["z_denoise"].float()) * _gsn * _inv).cpu().numpy()[:, 0, :])
                    if "manifold_blend" in out:
                        _bls.append(out["manifold_blend"].float().cpu().numpy()[:, 0, :])
            except Exception:
                pass
            preds.append(out["denoised"].float().cpu().numpy()[:, 0, :])
            cleans.append(batch["clean"].float().cpu().numpy()[:, 0, :])
            idxs.append(raw["clean_index"].numpy().ravel())
    pred = np.concatenate(preds, 0)
    clean = np.concatenate(cleans, 0)
    cidx = np.concatenate(idxs, 0)
    e2 = (pred - clean) ** 2
    y2 = clean**2
    metrics = {
        "global_error_percent": 100.0 * math.sqrt(e2.sum() / max(y2.sum(), 1e-30)),
        "early_mid_error_percent": 100.0 * math.sqrt(
            e2[:, :late].sum() / max(y2[:, :late].sum(), 1e-30)
        ),
        "late_error_percent": 100.0 * math.sqrt(
            e2[:, late:].sum() / max(y2[:, late:].sum(), 1e-30)
        ),
        "samples": int(len(pred)),
    }
    per_late = _per_trace_nrmse(pred, clean, late_cols)
    per_glob = _per_trace_nrmse(pred, clean, all_cols)
    metrics["late_macro_median_percent"] = float(np.median(per_late))
    metrics["late_macro_p90_percent"] = float(np.percentile(per_late, 90))
    metrics["global_macro_median_percent"] = float(np.median(per_glob))
    try:
        if _mans and _dens:
            _mp = np.concatenate(_mans, 0)
            _dp = np.concatenate(_dens, 0)
            _den = np.abs(clean) + 1e-30
            metrics["per_gate_mean_error_prior_percent"] = (100.0 * np.mean(np.abs(_mp - clean) / _den, axis=0)).tolist()
            metrics["per_gate_mean_error_denoise_percent"] = (100.0 * np.mean(np.abs(_dp - clean) / _den, axis=0)).tolist()
            if _bls:
                metrics["per_gate_mean_blend"] = np.mean(np.concatenate(_bls, 0), axis=0).tolist()
    except Exception:
        pass
    return metrics, per_late, per_glob, cidx


def _cluster_bootstrap_ci(
    values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 5
) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    c = np.asarray(clusters)
    uc = np.unique(c)
    if len(uc) < 2:
        m = float(np.median(v))
        return m, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    members = [np.where(c == g)[0] for g in uc]
    stats = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        pick = rng.integers(0, len(uc), size=len(uc))
        sel = np.concatenate([members[i] for i in pick])
        stats[b] = np.median(v[sel])
    return float(np.median(v)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def run_late_window_diagnosis(
    model: "PEBRNet",
    cfg: "Config",
    run_dir: Path,
    clean_cache: "CleanCacheInfo",
    noise_cache: "NoiseCacheInfo",
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Why will the late error not come down?"""
    device = resolve_device(cfg.runtime.device)
    model.to(device).eval()
    gates = int(cfg.data.target_gates)
    late = int(cfg.data.late_start_index)
    late_cols = np.arange(late, gates)
    t_ms = gate_times_ms(clean_cache)
    gsn = np.clip(
        np.asarray(clean_cache.gate_scale, dtype=np.float64) / float(clean_cache.global_scale), 1e-30, None
    )
    splits = build_splits(clean_cache, cfg)
    n = int(min(cfg.data.val_samples, len(splits["val"])))
    ds = BTEMDenoisingDataset(
        clean_cache.data_path, noise_cache.data_path, splits["val"], clean_cache.global_scale,
        cfg.data, cfg.train.seed + 23, n, False,
        noise_block_rows=noise_cache.block_rows, gate_times=noise_cache.target_time,
        real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
        real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
        province_id=getattr(clean_cache, "province_id", None),
        paired_noisy_path=getattr(clean_cache, "paired_noisy_path", ""),
        paired_background_path=getattr(clean_cache, "paired_background_path", ""),
        paired_event_path=getattr(clean_cache, "paired_event_path", ""),
        measured_noise_path=getattr(clean_cache, "measured_noise_path", ""),
        extra_clean_path=getattr(clean_cache, "extra_clean_path", ""),
    )
    ds.set_epoch(0)
    loader = DataLoader(ds, batch_size=block_aligned_batch_size(ds, min(cfg.train.batch_size, 256)),
                        shuffle=False, num_workers=0)
    edges = np.asarray([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    labels = ["%.0f-%.0f" % (edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    acc: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("snr", "late_err", "man_err", "blend", "innov", "rel", "denoise_share",
                        "prior_share", "innov_share", "meas_snr_late")
    }
    with torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            out = model(batch["noisy"], batch.get("neighbors"), batch.get("neighbor_geometry"),
                  depth_norm=batch.get("depth_norm"), profile_len=batch_profile_len(batch))
            pred = out["denoised"].float().cpu().numpy()[:, 0, :]
            clean = batch["clean"].float().cpu().numpy()[:, 0, :]
            acc["snr"].append(raw["snr_db"].numpy().ravel().astype(np.float64))
            acc["late_err"].append(_per_trace_nrmse(pred, clean, late_cols))
            if "clean_row" in raw:
                acc.setdefault("clean_row", []).append(raw["clean_row"].numpy().ravel().astype(np.int64))
            if "manifold_z" in out:
                ts = out.get("trace_scale")
                man = signed_symexp(out["manifold_z"].float()) * model.gate_scale_norm.float()
                if ts is not None:
                    man = man / ts.float()
                acc["man_err"].append(
                    _per_trace_nrmse(man.cpu().numpy()[:, 0, :], clean, late_cols)
                )
                a = out["manifold_blend"].float().cpu().numpy()[:, 0, late:]
                acc["blend"].append(a.mean(axis=1))
                zdn = out["z_denoise"].float().cpu().numpy()[:, 0, late:]
                prior = out["manifold_prior_z"].float().cpu().numpy()[:, 0, late:]
                post = out["manifold_z"].float().cpu().numpy()[:, 0, late:]
                c_dn = np.abs((1.0 - a) * zdn)
                c_pr = np.abs(a * prior)
                c_in = np.abs(a * (post - prior))
                tot = np.maximum(c_dn + c_pr + c_in, 1e-12)
                acc["denoise_share"].append((c_dn / tot).mean(axis=1))
                acc["prior_share"].append((c_pr / tot).mean(axis=1))
                acc["innov_share"].append((c_in / tot).mean(axis=1))
            if "manifold_innovation_w" in out:
                acc["innov"].append(
                    out["manifold_innovation_w"].float().cpu().numpy()[:, 0, late:].mean(axis=1)
                )
            if "meas_reliability" in out:
                acc["rel"].append(
                    out["meas_reliability"].float().cpu().numpy()[:, 0, late:].mean(axis=1)
                )
            if "meas_sigma_z" in out:
                sm = out["meas_sigma_z"].float().cpu().numpy()[:, 0, late:]
                _ts = out.get("trace_scale")
                _tsn = _ts.float().cpu().numpy().reshape(-1, 1) if _ts is not None else 1.0
                zc = _np_signed_symlog(clean[:, late:] * _tsn / gsn[None, late:])
                acc["meas_snr_late"].append(
                    20.0 * np.log10(np.maximum(np.abs(zc).mean(axis=1), 1e-12) / np.maximum(sm.mean(axis=1), 1e-12))
                )
    data = {k: (np.concatenate(v) if v else np.asarray([])) for k, v in acc.items()}
    snr = data["snr"]
    bins = np.digitize(snr, edges[1:-1])
    tau = 100.0 * float(cfg.contract.late_threshold)
    rows: List[Dict[str, Any]] = []
    for b, lab in enumerate(labels):
        sel = np.where(bins == b)[0]
        if sel.size < 5:
            continue
        row: Dict[str, Any] = {"snr_bin": lab, "n": int(sel.size),
                               "late_nrmse_percent": float(np.median(data["late_err"][sel]))}
        for key, name in (("man_err", "manifold_late_nrmse_percent"), ("blend", "blend_late"),
                          ("innov", "innovation_weight"), ("rel", "measured_reliability"),
                          ("denoise_share", "share_denoise"), ("prior_share", "share_prior"),
                          ("innov_share", "share_innovation"), ("meas_snr_late", "late_measurement_snr_db")):
            if data[key].size:
                row[name] = float(np.median(data[key][sel]))
        rows.append(row)
    table = pd.DataFrame(rows)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    table.to_csv(metrics_dir / tagged_name("late_window_diagnosis.csv"), index=False, encoding="utf-8-sig")

    lines = ["LATE-WINDOW DIAGNOSIS (validation, by SNR stratum)",
             "  %-8s %6s %9s %11s %8s %9s %8s %8s %8s"
             % ("SNR(dB)", "n", "late%", "prior-own%", "blend", "innov-w", "denoise", "prior", "innov")]
    for r in rows:
        lines.append(
            "  %-8s %6d %9.3f %11s %8s %9s %7s%% %7s%% %7s%%"
            % (r["snr_bin"], r["n"], r["late_nrmse_percent"],
               "%.3f" % r.get("manifold_late_nrmse_percent", float("nan")),
               "%.3f" % r.get("blend_late", float("nan")),
               "%.4f" % r.get("innovation_weight", float("nan")),
               "%.0f" % (100 * r.get("share_denoise", float("nan"))),
               "%.0f" % (100 * r.get("share_prior", float("nan"))),
               "%.0f" % (100 * r.get("share_innovation", float("nan")))))
    if "clean_row" in acc and getattr(clean_cache, "province_id", None) is not None:
        rows_abs = np.concatenate(acc["clean_row"])
        le_all = np.concatenate(acc["late_err"])
        pid_all = np.asarray(clean_cache.province_id, dtype=np.int64)[rows_abs]
        names = list(clean_cache.province_names or [])
        per: List[Tuple[float, int, int]] = []
        for p in np.unique(pid_all):
            if int(p) < 0:
                continue
            m = pid_all == p
            per.append((float(np.mean(le_all[m])), int(m.sum()), int(p)))
        per.sort(reverse=True)
        wl = ["WORST HELD-OUT PROVINCES (validation, by mean late NRMSE) -- the geologies the",
              "prior fails on; densify or flag these families:",
              "  %-44s %-11s %6s %10s" % ("group", "source", "n", "late%")]
        for err_p, n_p, p in per[:8]:
            nm = names[p] if p < len(names) else str(p)
            src_tag = "clean-block" if str(nm).startswith("__field_block_") else "province"
            wl.append("  %-44s %-11s %6d %9.2f%%" % (nm[:44], src_tag, n_p, err_p))
        med = float(np.median([e for e, _, _ in per])) if per else float("nan")
        wl.append("  median per-group late error = %.2f%% over %d groups" % (med, len(per)))
        logger.info("\n".join(wl))
        try:
            k_tail = max(1, int(math.ceil(0.20 * len(le_all))))
            tail_ix = np.argpartition(le_all, len(le_all) - k_tail)[-k_tail:]
            tail_pid = pid_all[tail_ix]
            is_blk_all = np.array([str(names[p]).startswith("__field_block_")
                                   if 0 <= p < len(names) else False for p in pid_all])
            comp: Dict[str, Any] = {
                "definition": "worst 20% of per-curve late NRMSE on the validation stream "
                              "(same fraction the late-CVaR deployment gate reads)",
                "n_curves": int(len(le_all)), "n_tail": int(k_tail),
                "tail_mean_percent": float(np.mean(le_all[tail_ix])),
                "body_mean_percent": float(np.mean(np.delete(le_all, tail_ix))),
                "tail_by_source": {},
                "top_tail_groups": [],
            }
            for lab in ("clean-block", "province"):
                m_src = is_blk_all[tail_ix] if lab == "clean-block" else ~is_blk_all[tail_ix]
                m_all = is_blk_all if lab == "clean-block" else ~is_blk_all
                comp["tail_by_source"][lab] = {
                    "tail_count": int(m_src.sum()),
                    "tail_share_percent": float(100.0 * m_src.mean()) if len(tail_ix) else 0.0,
                    "population_share_percent": float(100.0 * m_all.mean()),
                    "median_late_percent": float(np.median(le_all[m_all])) if m_all.any() else None,
                }
            for p in np.unique(tail_pid):
                cnt = int((tail_pid == p).sum())
                comp["top_tail_groups"].append({
                    "group": names[p] if 0 <= p < len(names) else str(int(p)),
                    "source": "clean-block" if (0 <= p < len(names)
                              and str(names[p]).startswith("__field_block_")) else "province",
                    "tail_curves": cnt,
                    "share_of_tail_percent": float(100.0 * cnt / k_tail),
                    "group_mean_late_percent": float(np.mean(le_all[pid_all == p])),
                })
            comp["top_tail_groups"].sort(key=lambda d: -d["tail_curves"])
            comp["top_tail_groups"] = comp["top_tail_groups"][:40]
            atomic_json_dump(comp, run_dir / "reports" / "late_tail_composition.json")
            _blk = comp["tail_by_source"].get("clean-block", {})
            logger.info(
                "LATE-CVaR TAIL COMPOSITION | worst 20%% = %d curves, mean %.2f%% "
                "(body mean %.2f%%) | clean-block rows are %.0f%% of the tail vs %.0f%% of the "
                "population | full membership -> reports/late_tail_composition.json",
                k_tail, comp["tail_mean_percent"], comp["body_mean_percent"],
                _blk.get("tail_share_percent", float("nan")),
                _blk.get("population_share_percent", float("nan")),
            )
        except Exception as _exc:
            logger.warning("Tail composition report skipped: %r", _exc)
    logger.info("\n".join(lines))
    hi = [r for r in rows if r["snr_bin"] in ("30-35", "25-30")]
    if hi:
        m = float(np.mean([r["late_nrmse_percent"] for r in hi]))
        iw = float(np.mean([r.get("innovation_weight", 0.0) for r in hi]))
        bl = float(np.mean([r.get("blend_late", 1.0) for r in hi]))
        if m > tau and iw < 0.05:
            _prior_hi = float(np.mean([r.get("manifold_late_nrmse_percent", float("nan"))
                                       for r in hi]))
            _dn_hi = float(np.mean([r.get("share_denoise", 0.0) for r in hi]))
            logger.warning(
                "At 25-35 dB the late error is %.2f%% (contract %.1f%%) with innovation gain "
                "%.4f and blend %.3f. Read this against the LATE-WINDOW INFORMATION "
                "BUDGET above: the late measurement is short at EVERY SNR, so a near-zero gain "
                "is the budget-OPTIMAL estimator here, not a defect -- do not open it. The gap "
                "to the prior's own error (%.2f%%) is carried by the residual (1-blend) weight "
                "on the denoising path (share ~%.0f%%): if that path's late error is large "
                "(see den_path_late in the training log), THAT is the term to fix: its supervision "
                "is restored by capping the constraint block's total "
                "weight.",
                m, tau, iw, bl, _prior_hi, 100.0 * _dn_hi,
            )
        elif iw < 0.01 and (1.0 - bl) < 0.01:
            logger.warning(
                "PRIOR_MONOPOLY | late error %.2f%%, innovation gain %.4f, blend %.3f: the late "
                "window is carried ~entirely by the manifold/physics prior; the denoising and "
                "innovation channels contribute <1%%. Field anomalies the library cannot represent "
                "would be rewritten by the prior -- deployment requires the "
                "anomaly-preservation suite to PASS (the event residual is the pathway "
                "that carries them).", m, iw, bl,
            )
        else:
            logger.info(
                "At 25-35 dB: late error %.2f%%, innovation gain %.3f, blend %.3f -- the measurement channel "
                "is alive and carrying the late window where the data supports it.", m, iw, bl,
            )
    if cfg.runtime.plot:
        render_late_window_figure(
            {"rows": rows, "labels": labels, "tau": tau, "raw": data, "t_ms": t_ms, "late": late},
            cfg, run_dir, logger,
        )
    return {"per_snr": rows}


def run_generalization_diagnosis(
    model: "PEBRNet",
    cfg: "Config",
    run_dir: Path,
    clean_cache: "CleanCacheInfo",
    noise_cache: "NoiseCacheInfo",
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Answer, from the user's own data, the only question that matters when the validation error jumps after a
    library change: why.
    """
    device = resolve_device(cfg.runtime.device)
    model.to(device).eval()
    splits = build_splits(clean_cache, cfg)
    pop = audit_split_populations(clean_cache, splits, logger)
    gate_scale_norm = np.asarray(clean_cache.gate_scale, dtype=np.float64) / float(clean_cache.global_scale)
    controller = ConstraintController(cfg.contract)
    loss_fn = ProjectLoss(
        controller, cfg.contract, cfg.loss, cfg.data.late_start_index,
        torch.as_tensor(gate_scale_norm, dtype=torch.float32),
        late_terms_all_stages=cfg.train.late_terms_all_stages,
        ip_chargeability_max=cfg.model.ip_chargeability_max,
        relaxation_blocks=len(tuple(getattr(cfg.model, "relaxation_stretch_exponents", (1.0,)) or (1.0,))),
        profile_patch_len=int(getattr(cfg.data, "profile_patch_len", 0) or 0),
    ).to(device)
    n_eval = int(min(cfg.data.val_samples, len(splits["val"])))
    results: Dict[str, Any] = {"program": PROGRAM_NAME, "version": PROGRAM_VERSION}

    _train_rows = np.asarray(splits["train"], dtype=np.int64)
    _tr_label = "TRAIN(eval)"
    _fwd_rows = None
    _pid_all = getattr(clean_cache, "province_id", None)
    if _pid_all is not None:
        _pid_tr = np.asarray(_pid_all, dtype=np.int64)[_train_rows]
        _fmask = _pid_tr >= 0
        _p_fwd = float(getattr(cfg.data, "mix_forward_fraction", 0.0) or 0.0)
        if _p_fwd <= 0.0 and bool(_fmask.any()) and bool((~_fmask).any()):
            _fwd_rows = _train_rows[~_fmask]
            _train_rows = _train_rows[_fmask]
            _tr_label = "TRAIN(eval,draw)"
            logger.info(
                "TRAIN(eval) restricted to the EFFECTIVE training draw: %d "
                "field rows kept, %d forward-simulation rows excluded (draw fraction "
                "0.00 -- the optimizer never sees them; they are reported separately "
                "below as out-of-training-distribution).",
                int(_train_rows.size), int(_fwd_rows.size),
            )
    m_tr, late_tr, _, cidx_tr = _eval_split_metrics(
        model, cfg, clean_cache, noise_cache, _train_rows, cfg.train.seed + 23, n_eval, loss_fn, device
    )
    m_va, late_va, glob_va, cidx_va = _eval_split_metrics(
        model, cfg, clean_cache, noise_cache, splits["val"], cfg.train.seed + 23, n_eval, loss_fn, device
    )
    m_te, late_te, _, _ = _eval_split_metrics(
        model, cfg, clean_cache, noise_cache, splits["test"], cfg.train.seed + 37, n_eval, loss_fn, device
    )
    try:
        _ls = int(cfg.data.late_start_index)
        _g = max(0, _ls - 8)
        def _row(_m, _key):
            _v = list(_m.get(_key) or [])
            return " ".join("%.1f" % float(_v[g]) for g in range(_g, min(_ls + 3, len(_v)))) if _v else "n/a"
        logger.info("SEAM ARMS gates %d-%d (mean |err|/|clean| %%) | TRAIN(eval): prior %s | denoise %s | blend %s || "
                    "VAL: prior %s | denoise %s | blend %s. A denoise arm far better on TRAIN than on VAL is memorised seam "
                    "noise; the blend is then calibrated on held-out arms.",
                    _g + 1, min(_ls + 3, int(cfg.data.target_gates)),
                    _row(m_tr, "per_gate_mean_error_prior_percent"), _row(m_tr, "per_gate_mean_error_denoise_percent"),
                    _row(m_tr, "per_gate_mean_blend"),
                    _row(m_va, "per_gate_mean_error_prior_percent"), _row(m_va, "per_gate_mean_error_denoise_percent"),
                    _row(m_va, "per_gate_mean_blend"))
    except Exception as _e:
        logger.warning("seam arm comparison skipped: %r", _e)
    groups = splits["_groups"]
    val_clusters = groups[cidx_va]
    late_med, late_lo, late_hi = _cluster_bootstrap_ci(late_va, val_clusters)
    results["generalization_gap"] = {
        "train_eval": m_tr, "val": m_va, "test": m_te,
        "late_gap_pp": m_va["late_error_percent"] - m_tr["late_error_percent"],
        "global_gap_pp": m_va["global_error_percent"] - m_tr["global_error_percent"],
        "val_late_macro_median_percent": late_med,
        "val_late_macro_ci_lo": late_lo,
        "val_late_macro_ci_hi": late_hi,
        "val_groups": int(len(splits["_val_groups"])),
    }
    logger.info(
        "GENERALIZATION GAP (identical protocol: eval mode, deterministic pairs, representative subsample, "
        "no augmentation)\n"
        "  %-12s %8s %8s %8s %14s\n"
        "  %-12s %7.3f%% %7.3f%% %7.3f%% %13.3f%%\n"
        "  %-12s %7.3f%% %7.3f%% %7.3f%% %13.3f%%\n"
        "  %-12s %7.3f%% %7.3f%% %7.3f%% %13.3f%%\n"
        "  val late per-curve median %.3f%% [95%% CI %.3f, %.3f] over %d validation GROUPS "
        "(the group count, not the curve count, is the effective sample size)",
        "split", "G", "EM", "L", "late median",
        _tr_label, m_tr["global_error_percent"], m_tr["early_mid_error_percent"],
        m_tr["late_error_percent"], m_tr["late_macro_median_percent"],
        "VAL", m_va["global_error_percent"], m_va["early_mid_error_percent"],
        m_va["late_error_percent"], m_va["late_macro_median_percent"],
        "TEST", m_te["global_error_percent"], m_te["early_mid_error_percent"],
        m_te["late_error_percent"], m_te["late_macro_median_percent"],
        late_med, late_lo, late_hi, len(splits["_val_groups"]),
    )

    if _fwd_rows is not None and _fwd_rows.size:
        try:
            m_fw, _, _, _ = _eval_split_metrics(
                model, cfg, clean_cache, noise_cache, _fwd_rows, cfg.train.seed + 29,
                min(n_eval, int(_fwd_rows.size)), loss_fn, device)
            results["forward_pool_out_of_draw"] = m_fw
            logger.info(
                "  %-12s %7.3f%% %7.3f%% %7.3f%% %13.3f%%   <- FORWARD pool, draw "
                "fraction 0.00: the model never trains on these; this row is an "
                "out-of-distribution reference, NOT a generalization gap.",
                "FWD(no-draw)", m_fw["global_error_percent"], m_fw["early_mid_error_percent"],
                m_fw["late_error_percent"], m_fw["late_macro_median_percent"],
            )
        except Exception as _fexc:
            logger.warning("forward-pool reference row skipped: %r", _fexc)
    mem: Dict[str, Any] = {}
    try:
        lib = np.load(clean_cache.data_path, mmap_mode="r")
        gsn = np.clip(gate_scale_norm, 1e-30, None)

        def _shape(rows: np.ndarray) -> np.ndarray:
            blk = np.asarray(lib[np.sort(rows)], dtype=np.float64) / float(clean_cache.global_scale)
            z = _np_signed_symlog(blk / gsn[None, :])
            return (z - z.mean(axis=1, keepdims=True)).astype(np.float32)

        tr_rows = splits["train"]
        if len(tr_rows) > 120_000:
            tr_rows = np.sort(np.random.default_rng(0).choice(tr_rows, 120_000, replace=False))
        bank = _shape(tr_rows)
        uniq_val = np.unique(cidx_va)
        q = _shape(uniq_val)
        d_nn, _ = _nn_search(q, bank)
        lut = {int(c): float(d) for c, d in zip(np.sort(uniq_val), d_nn)}
        d_per_sample = np.asarray([lut[int(c)] for c in cidx_va], dtype=np.float64)
        order = np.argsort(d_per_sample)
        k = max(1, len(order) // 5)
        quint = [order[i * k : (i + 1) * k] for i in range(5)]
        curve = [
            {
                "quintile": i + 1,
                "nn_distance_median": float(np.median(d_per_sample[q_i])),
                "late_error_median_percent": float(np.median(late_va[q_i])),
            }
            for i, q_i in enumerate(quint) if len(q_i)
        ]
        rho = float(np.corrcoef(_rankdata(d_per_sample), _rankdata(late_va))[0, 1])
        ratio = (curve[-1]["late_error_median_percent"] / max(curve[0]["late_error_median_percent"], 1e-9)) if len(curve) >= 2 else float("nan")
        mem = {
            "spearman_error_vs_nn_distance": rho,
            "late_error_ratio_far_over_near": ratio,
            "quintiles": curve,
            "nn_distance_median": float(np.median(d_per_sample)),
            "_arrays": {"d": d_per_sample, "err": late_va},
        }
        logger.info(
            "MEMORIZATION PROBE | validation late error against distance to the NEAREST TRAINING curve "
            "(shape space): Spearman rho=%.3f | error in the farthest quintile is %.2fx the nearest "
            "(%s). A strong positive trend means the late window is being INTERPOLATED inside the training "
            "curve family, i.e. the library is too sparse -- not that the optimizer overfit the noise.",
            rho, ratio,
            " | ".join("Q%d: d=%.2f err=%.2f%%" % (c["quintile"], c["nn_distance_median"],
                                                   c["late_error_median_percent"]) for c in curve),
        )
    except Exception as exc:
        logger.warning("Memorization probe skipped: %s", exc)
    results["memorization_probe"] = {k: v for k, v in mem.items() if k != "_arrays"}

    attr: Dict[str, Any] = {}
    d = clean_cache.descriptors
    if d is not None:
        per_curve_err: Dict[int, List[float]] = {}
        for c, e in zip(cidx_va, late_va):
            per_curve_err.setdefault(int(c), []).append(float(e))
        cids = np.asarray(sorted(per_curve_err))
        err = np.asarray([np.median(per_curve_err[int(c)]) for c in cids])
        for i, name in enumerate(CURVE_DESCRIPTOR_NAMES):
            x = d[cids][:, i]
            ok = np.isfinite(x) & np.isfinite(err)
            if int(ok.sum()) < 10:
                continue
            if float(np.std(x[ok])) < 1e-12 or float(np.std(err[ok])) < 1e-12:
                continue
            attr[name] = float(np.corrcoef(_rankdata(x[ok]), _rankdata(err[ok]))[0, 1])
        cone_bad = d[cids][:, 4] > 0.20
        if cone_bad.any() and (~cone_bad).any():
            attr["late_error_inside_cone_percent"] = float(np.median(err[~cone_bad]))
            attr["late_error_outside_cone_percent"] = float(np.median(err[cone_bad]))
            attr["fraction_outside_cone_percent"] = float(100.0 * np.mean(cone_bad))
        logger.info(
            "PER-CURVE ATTRIBUTION (Spearman of the validation late error against each curve property) | %s",
            " | ".join("%s %+.2f" % (k, v) for k, v in attr.items() if isinstance(v, float) and abs(v) <= 1.0),
        )
        if "late_error_outside_cone_percent" in attr:
            logger.info(
                "  Curves OUTSIDE the physics cone (%.1f%% of validation): late error %.2f%% vs %.2f%% inside. "
                "If the outside group dominates, the positive-exponential head is the wrong prior for this "
                "library and --disable_physics_atoms (or the signed IP head) is the fix, not regularization.",
                attr["fraction_outside_cone_percent"],
                attr["late_error_outside_cone_percent"], attr["late_error_inside_cone_percent"],
            )
    results["per_curve_attribution"] = attr

    ceiling = log_identification_ceiling(clean_cache, cfg, logger)
    results["identification_ceiling"] = ceiling or {}

    try:
        results["late_window"] = run_late_window_diagnosis(
            model, cfg, run_dir, clean_cache, noise_cache, logger
        )
    except Exception as exc:
        logger.warning("Late-window diagnosis skipped: %s", exc)

    tau = 100.0 * cfg.contract.late_threshold
    val_l = m_va["late_macro_median_percent"]
    tr_l = m_tr["late_macro_median_percent"]
    gap = val_l - tr_l
    results["generalization_gap"]["late_gap_macro_pp"] = gap
    for nm, mm in (("TRAIN", m_tr), ("VAL", m_va), ("TEST", m_te)):
        if mm["late_error_percent"] > 3.0 * max(mm["late_macro_median_percent"], 1e-9):
            logger.warning(
                "%s: the POOLED late NRMSE (%.2f%%) is %.1fx the per-curve MEDIAN (%.2f%%). The contract "
                "metric is energy-weighted, so it is being set by a few curves whose late window carries "
                "almost no energy. Their relative error explodes for an absolute error that is tiny. This "
                "alone can make the headline number lurch when the library changes, with nothing wrong in "
                "the model.",
                nm, mm["late_error_percent"],
                mm["late_error_percent"] / max(mm["late_macro_median_percent"], 1e-9),
                mm["late_macro_median_percent"],
            )
    ceil_pooled = float(ceiling["pooled"]) if ceiling else float("nan")
    verdict: List[str] = []
    if len(splits["_val_groups"]) < 6:
        verdict.append(
            "SPLIT ARTEFACT: only %d validation groups, so this number has error bars wider than the effect "
            "you are chasing. Fix the split before concluding anything." % len(splits["_val_groups"])
        )
    if pop.get("flagged"):
        verdict.append(
            "SPLIT ARTEFACT: the validation curves are a different population from the training curves (%s). "
            "Keep --split_stratify on and raise --split_min_groups." % "; ".join(pop["flagged"][:3])
        )
    if np.isfinite(ceil_pooled) and ceil_pooled > tau:
        verdict.append(
            "IDENTIFICATION CEILING: even a ridge regression from the CLEAN early/mid window to the CLEAN late "
            "window, fitted on train and evaluated on val, only reaches %.2f%% -- above the %.1f%% contract. "
            "The library's own conditional spread binds; no architecture beats it from identification alone. "
            "Densify the forward library along tau and geometry (the augmentation does part of this "
            "for free)." % (ceil_pooled, tau)
        )
    if attr.get("late_error_outside_cone_percent", 0.0) > 2.0 * max(attr.get("late_error_inside_cone_percent", 1e-9), 1e-9):
        verdict.append(
            "PHYSICS-CONE MISMATCH: the validation error is concentrated on curves the positive-exponential "
            "head cannot represent (%.2f%% vs %.2f%%). Use --disable_physics_atoms, or enable the signed IP "
            "head, or restrict the library to induction-regime responses."
            % (attr["late_error_outside_cone_percent"], attr["late_error_inside_cone_percent"])
        )
    if gap > max(0.5 * tr_l, 1.0) and tr_l < tau:
        msg = ("OVERFITTING / LIBRARY SPARSITY: train %.2f%% vs val %.2f%% under the identical protocol "
               "(gap %+.2f pp)." % (tr_l, val_l, gap))
        rho = mem.get("spearman_error_vs_nn_distance")
        if rho is not None and np.isfinite(rho) and rho > 0.25:
            msg += (" The memorization probe confirms it (rho=%.2f): the error grows with distance to the "
                    "nearest training curve, so the network is interpolating the curve FAMILY. Enlarge or "
                    "densify the library and keep --clean_augment on; regularization alone will not fix a "
                    "library that does not cover the validation curves." % rho)
        else:
            msg += (" The memorization probe does NOT show a family-interpolation signature, so this is a "
                    "classic variance gap: raise dropout/weight decay, or shorten training.")
        verdict.append(msg)
    if tr_l >= tau and val_l >= tau and abs(gap) < max(0.5 * tr_l, 1.0) and val_l >= tr_l - 1.0:
        verdict.append(
            "BIAS, NOT VARIANCE: train %.2f%% and val %.2f%% (per-curve medians) are both above the %.1f%% "
            "contract with almost no gap. The model is not overfitting -- it cannot represent this library. "
            "Regularizing harder will make it worse. Look at the identification ceiling and the physics-cone "
            "fit above." % (tr_l, val_l, tau)
        )
    if val_l + 1.0 < tr_l and tr_l > tau:
        _src_note = ""
        try:
            _pid = np.asarray(clean_cache.province_id, dtype=np.int64)
            _nm = list(clean_cache.province_names or [])
            def _by_src(per_arr, cidx_arr):
                is_blk = np.array([str(_nm[_pid[i]]).startswith("__field_block_")
                                   if 0 <= _pid[i] < len(_nm) else False for i in cidx_arr])
                out = {}
                for lab, msk in (("clean-block", is_blk), ("province", ~is_blk)):
                    if msk.any():
                        out[lab] = (float(np.median(per_arr[msk])), int(msk.sum()))
                return out
            _tr_b = _by_src(late_tr, cidx_tr)
            _va_b = _by_src(late_va, cidx_va)
            _src_note = " Per-source per-curve late medians -- TRAIN: %s | VAL: %s." % (
                ", ".join("%s %.2f%% (n=%d)" % (k, v[0], v[1]) for k, v in _tr_b.items()),
                ", ".join("%s %.2f%% (n=%d)" % (k, v[0], v[1]) for k, v in _va_b.items()),
            )
        except Exception:
            pass
        verdict.append(
            "TRAIN-EVAL ANOMALY (val %.2f%% << train %.2f%% under the identical protocol): a model "
            "cannot generalize better than it fits. Either the train split holds a population the "
            "val split lacks (see the amplitude-KS audit above -- after shape de-duplication the "
            "train side keeps the hard originals while easier copies were dropped), or the two "
            "evaluations are not actually identical. This is a DATA/PROTOCOL finding, not an "
            "optimization one; do NOT respond by training longer.%s" % (val_l, tr_l, _src_note)
        )
    if not verdict:
        verdict.append(
            "HEALTHY: train %.2f%% vs val %.2f%% (gap %+.2f pp), val within the %.1f%% contract, populations "
            "matched, no memorization signature." % (tr_l, val_l, gap, tau)
        )
    results["verdict"] = verdict
    logger.info("=" * 78)
    for v in verdict:
        logger.info("VERDICT | %s", v)
    logger.info("=" * 78)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    atomic_json_dump({**results, "split_population_audit": pop}, metrics_dir / tagged_name("generalization_diagnosis.json"))
    if cfg.runtime.plot:
        render_diagnosis_figure(
            {
                "m_tr": m_tr, "m_va": m_va, "m_te": m_te,
                "late_tr": late_tr, "late_va": late_va, "late_te": late_te,
                "mem": mem, "attr": attr, "pop": pop,
                "descriptors": d, "cids_val": cidx_va,
                "tau": tau, "ceiling": ceiling,
            },
            cfg, run_dir, logger,
        )
    return results


FIGURE_MANIFEST: Tuple[Dict[str, str], ...] = (
    {"stem": "fig_evidence_m1_missing_signal_robustness", "group": "Missing late signal",
     "claim": "Late-window NRMSE, contract-pass rate, detectability index and the significance of the "
              "model's advantage, as 10-50% of the late signal is removed; both scopes."},
    {"stem": "fig_evidence_m2_pergate_error_global", "group": "Missing late signal",
     "claim": "Per-gate error profile with bootstrap CI bands across the missing-tail levels."},
    {"stem": "fig_evidence_m2_pergate_error_late", "group": "Missing late signal",
     "claim": "Per-gate error profile when the loss is confined to the late window."},
    {"stem": "fig_evidence_m3_case_decays", "group": "Missing late signal",
     "claim": "Individual decays at low/mid/high SNR with the ablated zone shaded: what the "
              "reconstruction actually looks like."},
    {"stem": "fig_evidence_m4_distribution_and_snr", "group": "Missing late signal",
     "claim": "Full error distributions (ECDF) and SNR-stratified boxes, not just medians."},
    {"stem": "fig_evidence_m5_target_parameter_recovery", "group": "Concealed target",
     "claim": "Recovery of the parameters an interpreter reads: late time constant (target "
              "conductance), log-amplitude error in dex, sign agreement."},
    {"stem": "fig_evidence_m6_attainable_depth", "group": "Concealed target",
     "claim": "Reconstruction-limited depth of investigation in metres, versus missing fraction and "
              "versus SNR. The 'gong shen tan mang' quantity."},
    {"stem": "fig_evidence_e2_conditioning", "group": "Evidence conditioning",
     "claim": "The late output follows THIS trace's evidence: amplitude transfer, decay-rate "
              "following, same-noise curve swap."},
    {"stem": "fig_evidence_e3_memorization", "group": "Anti-memorization",
     "claim": "Reconstructions are closer to their own truth than to any training curve; the model "
              "beats a retrieval lookup."},
    {"stem": "fig_evidence_e4_spectral_tracking", "group": "Anti-memorization",
     "claim": "The physics spectrum follows tau continuously; a prototype lookup would jump."},
    {"stem": "fig_evidence_e5_misfit_qc", "group": "Inversion QC",
     "claim": "The reconstruction is consistent with the trace's own early/mid measurements."},
    {"stem": "fig_evidence_e6_source_decomposition", "group": "Inversion QC",
     "claim": "The late window decomposed into denoising path, physics cone, bounded residual and "
              "measurement innovation."},
    {"stem": "fig_ip_p1_identification_and_limit", "group": "Polarization",
     "claim": "ROC of the decay-domain reference, the profile-domain reference and their fusion; the "
              "detector against its matched-filter bound; the minimum detectable chargeability."},
    {"stem": "fig_ip_p2_preservation", "group": "Polarization",
     "claim": "The induction-only network erases the sign reversal; the dual-reference guard restores "
              "it, and leaves non-polarizable traces bit-identical."},
    {"stem": "fig_ip_p3_profile_localization", "group": "Polarization",
     "claim": "Wiggle section of the whitened late residual along the borehole, the detection score "
              "profile, and the body-localization error."},
    {"stem": "fig_master_concealed_target", "group": "Integrative",
     "claim": "One figure for the science question: how deep can a concealed, possibly polarizable "
              "target be located, as a function of acquisition quality and of how much late signal "
              "is missing."},
    {"stem": "fig_late_window_diagnosis", "group": "Model behaviour",
     "claim": "Why the late error is what it is: prior versus measurement, the innovation gain, the "
              "source decomposition, and the information the data actually carries."},
    {"stem": "fig_generalization_diagnosis", "group": "Model behaviour",
     "claim": "Train/val/test under one protocol, the memorization probe, and per-curve attribution."},
    {"stem": "fig_simulation_denoising", "group": "Model behaviour",
     "claim": "Simulated denoising cases across the SNR range."},
    {"stem": "fig_test_profile_comparison", "group": "Model behaviour",
     "claim": "Held-out test work areas (2 best, 2 worst, strongest and weakest body): noisy "
              "pair, clean reference, denoised and the late-window anomaly response."},
    {"stem": "fig_val_profile_comparison", "group": "Model behaviour",
     "claim": "Validation work areas selected by merit, same layout as the test sections."},
    {"stem": "fig_test_profile_best5", "group": "Model behaviour", "claim": "Test: 5 best areas by late error."},
    {"stem": "fig_test_profile_worst5", "group": "Model behaviour", "claim": "Test: 5 worst areas by late error."},
    {"stem": "fig_test_profile_bodies", "group": "Model behaviour", "claim": "Test: 3 strongest and 3 weakest bodies."},
    {"stem": "fig_test_weak_anomaly_quantification", "group": "Model behaviour",
     "claim": "Test: late error and recovery vs body contrast, quiet false level, recovery by stratum."},
    {"stem": "fig_val_profile_best5", "group": "Model behaviour", "claim": "Validation: 5 best areas by late error."},
    {"stem": "fig_val_profile_worst5", "group": "Model behaviour", "claim": "Validation: 5 worst areas by late error."},
    {"stem": "fig_val_profile_bodies", "group": "Model behaviour", "claim": "Validation: 3 strongest and 3 weakest bodies."},
    {"stem": "fig_val_weak_anomaly_quantification", "group": "Model behaviour",
     "claim": "Validation: late error and recovery vs body contrast, quiet false level, recovery by stratum."},
    {"stem": "fig_test_anomaly_quantification", "group": "Model behaviour",
     "claim": "Time, lateral-frequency, phase-space, energy and statistics evidence "
              "that the weak anomaly is recovered against the clean reference."},
    {"stem": "fig_field_denoising", "group": "Field",
     "claim": "Measured profile: raw, denoised, predicted noise, reliability."},
    {"stem": "fig_field_profile_curves", "group": "Field",
     "claim": "Every measured station's decay before and after reconstruction."},
    {"stem": "fig_field_reliability", "group": "Field",
     "claim": "Reliability and uncertainty along the measured profile."},
    {"stem": "fig_field_multigate_profile", "group": "Field",
     "claim": "Gate-by-gate profile sections along the borehole."},
    {"stem": "fig_field_tf_quantification", "group": "Field",
     "claim": "Time-frequency quantification of what was removed."},
)


def project_gate_monotonic(mat: np.ndarray) -> Tuple[np.ndarray, float]:
    """Per-station projection onto the non-increasing decay cone, pav in log space, plus the pre-projection violation
    rate.
    """
    a = np.asarray(mat, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2:
        return a.copy(), 0.0
    viol = float((a[:, 1:] > a[:, :-1]).sum())
    pairs = float(a.shape[0] * (a.shape[1] - 1))
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        row = a[i]
        floor = max(float(np.nanmax(np.abs(row))) * 1e-9, 1e-300)
        v = np.log(np.clip(row, floor, None))
        means: List[float] = []
        widths: List[int] = []
        for x in v:
            m, w = float(x), 1
            while means and m > means[-1]:
                pm, pw = means.pop(), widths.pop()
                m = (m * w + pm * pw) / (w + pw)
                w += pw
            means.append(m)
            widths.append(w)
        j = 0
        for m, w in zip(means, widths):
            out[i, j:j + w] = math.exp(m)
            j += w
    return out, (100.0 * viol / pairs if pairs > 0 else 0.0)


def deliverable_contract_eval(model: "PEBRNet", cfg: Config, run_dir: Path, clean_cache: CleanCacheInfo,
                              noise_cache: NoiseCacheInfo, rows: np.ndarray, logger: logging.Logger,
                              label: str) -> Dict[str, Any]:
    """The contract measured on the deliverable inference path (whole-area sliding window, the edge mode in force)
    over all given held-out rows: pooled G/EM/L.
    """
    t0 = time.time()
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    if rows.size == 0:
        return {}
    ls = int(cfg.data.late_start_index)
    K = int(max(1, cfg.data.num_neighbors))
    clean_mm = np.load(clean_cache.data_path, mmap_mode="r")
    noisy_mm = np.load(str(clean_cache.paired_noisy_path), mmap_mode="r") if getattr(clean_cache, "paired_noisy_path", "") else None
    pid = np.asarray(getattr(clean_cache, "province_id"))
    pred = np.asarray(_paired_test_forward(model, cfg, clean_cache, noise_cache, rows, logger), dtype=np.float64)
    cl = np.asarray(clean_mm[rows], dtype=np.float64)
    e2 = (pred - cl) ** 2
    y2 = cl ** 2
    def _nr(mask_rows, g0=0, g1=None):
        g1 = cl.shape[1] if g1 is None else g1
        num = float(e2[mask_rows, g0:g1].sum())
        den = float(y2[mask_rows, g0:g1].sum())
        return 100.0 * math.sqrt(num / max(den, 1e-300)) if den > 0 else float("nan")
    all_rows = np.ones(rows.size, dtype=bool)
    G = _nr(all_rows)
    EM = _nr(all_rows, 0, ls)
    L = _nr(all_rows, ls)
    r_late = 100.0 * np.sqrt(e2[:, ls:].sum(axis=1) / np.maximum(y2[:, ls:].sum(axis=1), 1e-300))
    k20 = max(1, int(round(0.2 * r_late.size)))
    cvar = float(np.sort(r_late)[-k20:].mean())
    med = float(np.median(r_late))
    pass3 = 100.0 * float(np.mean(r_late <= 3.0))
    pass4 = 100.0 * float(np.mean(r_late <= 4.0))
    gate_med = 100.0 * np.median(np.abs(pred - cl) / (np.abs(cl) + 1e-30), axis=0)
    dist = np.zeros(rows.size, dtype=np.int64)
    pos = {int(r): i for i, r in enumerate(rows)}
    for a in np.unique(pid[rows]):
        r = rows[pid[rows] == a]
        nn_ = int(r.size)
        for i, rr in enumerate(r):
            dist[pos[int(rr)]] = min(i, nn_ - 1 - i)
    edge = dist < K
    L_edge = _nr(edge, ls)
    L_int = _nr(~edge, ls)
    e_share = float(y2[edge, ls:].sum() / max(float(y2[:, ls:].sum()), 1e-300))
    amp_ratio = float(np.sqrt(y2[edge, ls:].sum(axis=1).mean() / max(y2[~edge, ls:].sum(axis=1).mean(), 1e-300))) if (~edge).any() and edge.any() else float("nan")
    amp: Dict[str, Any] = {}
    try:
        _gsv = np.asarray(clean_cache.gate_scale, dtype=np.float64).reshape(1, -1)
        _v = np.median(np.abs(cl[:, ls:]) / np.maximum(_gsv[:, ls:], 1e-300), axis=1)
        _ok = np.isfinite(_v) & (_v > 0) & np.isfinite(r_late)
        if int(_ok.sum()) >= 10:
            _q2 = np.quantile(_v[_ok], [0.2, 0.4, 0.6, 0.8])
            _bin = np.digitize(_v, _q2)
            amp["quintiles"] = []
            for _b in range(5):
                _m = _ok & (_bin == _b)
                if _m.any():
                    amp["quintiles"].append({"v_median": float(np.median(_v[_m])),
                                                "row_late_median_percent": float(np.median(r_late[_m])),
                                                "pass3_percent": 100.0 * float(np.mean(r_late[_m] <= 3.0)),
                                                "n": int(_m.sum())})
            _rv = pd.Series(np.log(_v[_ok])).rank().to_numpy()
            _rr = pd.Series(r_late[_ok]).rank().to_numpy()
            amp["spearman_logv_rowlate"] = float(np.corrcoef(_rv, _rr)[0, 1])
            _qs = amp["quintiles"]
            amp["low_high_quintile_median_ratio"] = (float(_qs[0]["row_late_median_percent"]
                                                              / max(_qs[-1]["row_late_median_percent"], 1e-12))
                                                        if len(_qs) >= 2 else float("nan"))
            amp["edge_interior_amplitude_ratio"] = (float(np.median(_v[edge & _ok]) / max(np.median(_v[~edge & _ok]), 1e-300))
                                                       if (edge & _ok).any() and (~edge & _ok).any() else float("nan"))
            _bd = {}
            for _q in range(4):
                if bool(np.any(dist == _q)):
                    _bd["d=%d" % _q] = float(np.median(r_late[dist == _q]))
            if bool(np.any((dist >= 4) & (dist < K))):
                _bd["d=4-7"] = float(np.median(r_late[(dist >= 4) & (dist < K)]))
            if bool(np.any(~edge)):
                _bd["interior"] = float(np.median(r_late[~edge]))
            amp["row_late_median_by_distance"] = _bd
    except Exception as _e:
        amp = {"error": repr(_e)}
    by_d = {}
    for q in range(4):
        by_d["d=%d" % q] = _nr(dist == q, ls)
    by_d["d=4-7"] = _nr((dist >= 4) & (dist < K), ls)
    by_d["interior"] = L_int
    tail = np.argsort(r_late)[-k20:]
    tail_edge = 100.0 * float(np.mean(edge[tail]))
    per_bin: Dict[str, Dict[str, float]] = {}
    tail_deep = float("nan")
    out_matched: Dict[str, Any] = {}
    if noisy_mm is not None:
        ny = np.asarray(noisy_mm[rows], dtype=np.float64)
        snr = 10.0 * np.log10(y2[:, ls:].sum(axis=1) / np.maximum(((ny - cl) ** 2)[:, ls:].sum(axis=1), 1e-300))
        edges = np.arange(-40.0, 40.0, 5.0)
        idx = np.clip(np.searchsorted(edges, snr, side="right") - 1, 0, edges.size - 2)
        for i in range(edges.size - 1):
            m = idx == i
            if m.any():
                per_bin["[%g,%g)" % (edges[i], edges[i + 1])] = {
                    "global_error_percent": _nr(m), "late_error_percent": _nr(m, ls),
                    "count": int(m.sum()), "row_share_percent": 100.0 * float(m.mean())}
        tail_deep = 100.0 * float(np.mean(snr[tail] < -10.0))
        _mb: Dict[str, Dict[str, float]] = {}
        _num = 0.0
        _den = 0.0
        _shared = np.zeros(rows.size, dtype=bool)
        for i in range(edges.size - 1):
            _mi = idx == i
            _me = _mi & edge
            _mn = _mi & (~edge)
            if _me.any() and _mn.any():
                _le = _nr(_me, ls)
                _li = _nr(_mn, ls)
                _mb["[%g,%g)" % (edges[i], edges[i + 1])] = {"edge": _le, "interior": _li, "n_edge": int(_me.sum()), "n_int": int(_mn.sum())}
                _num += float(y2[_me, ls:].sum()) * (_li / 100.0) ** 2
                _den += float(y2[_me, ls:].sum())
                _shared |= _me
        matched_interior = 100.0 * math.sqrt(_num / _den) if _den > 0 else float("nan")
        _edge_shared = _nr(_shared, ls) if _shared.any() else float("nan")
        _cov = 100.0 * float(_shared.sum()) / max(1.0, float(edge.sum()))
        out_matched = {"per_bin": _mb, "interior_at_edge_snr_mix_percent": matched_interior,
                          "edge_percent_all": L_edge, "edge_percent_shared_support": _edge_shared,
                          "edge_rows_in_shared_support_percent": _cov,
                          "edge_over_matched_interior": (_edge_shared / matched_interior
                                                         if (matched_interior > 0 and math.isfinite(_edge_shared)) else float("nan"))}
    rough_out = []
    rough_ref = []
    for a in np.unique(pid[rows]):
        m = pid[rows] == a
        if m.sum() < 3:
            continue
        lo = np.log10(np.abs(pred[m][:, ls:]) + 1e-30)
        lr = np.log10(np.abs(cl[m][:, ls:]) + 1e-30)
        rough_out.append(float(np.abs(np.diff(lo - lr, n=2, axis=0)).mean()))
        rough_ref.append(float(np.abs(np.diff(lr, n=2, axis=0)).mean()))
    rough_out_v = float(np.mean(rough_out)) if rough_out else float("nan")
    rough_ref_v = float(np.mean(rough_ref)) if rough_ref else float("nan")
    thr_g = 100.0 * float(getattr(cfg.contract, "global_threshold", 0.03))
    thr_em = 100.0 * float(getattr(cfg.contract, "early_mid_threshold", 0.03))
    thr_l = 100.0 * float(getattr(cfg.contract, "late_threshold", 0.03))
    thr_cv = 100.0 * float(effective_late_cvar_threshold(cfg.contract))
    out = {"rows": int(rows.size), "areas": int(np.unique(pid[rows]).size), "global_error_percent": G, "early_mid_error_percent": EM,
           "late_error_percent": L, "late_cvar_percent": cvar, "late_median_percent": med, "pass_at_3_percent": pass3,
           "pass_at_4_percent": pass4, "late_edge_percent": L_edge, "late_interior_percent": L_int, "edge_energy_share": e_share,
           "edge_amplitude_ratio": amp_ratio, "tail_edge_share_percent": tail_edge, "tail_deep_snr_share_percent": tail_deep,
           "lateral_roughness_out_dex": rough_out_v, "lateral_roughness_ref_dex": rough_ref_v,
           "edge_mode": resolve_edge_mode(cfg), "per_gate_median_error_percent": gate_med.tolist(), "by_distance": by_d, "per_snr_bin": per_bin,
           "edge_vs_interior_matched_snr": out_matched,
           "amplitude_audit": amp}
    _verd = qualification_verdict(out, cfg.contract, label)
    out["four_gate_pass"] = bool(G <= thr_g and EM <= thr_em and L <= thr_l and cvar <= thr_cv)
    out["contract_status"] = _verd["status"]
    out["contract_pass"] = bool(_verd["status"] == "PASS")
    out["qualification"] = _verd
    passed = out["contract_pass"]
    try:
        _DELIVERABLE_PRED_TABLE[label] = {"rows": rows.copy(), "pred": pred, "edge_mode": str(resolve_edge_mode(cfg)),
                                          "state_digest": model_state_digest(model.state_dict())}
    except Exception:
        pass
    logger.info("DELIVERABLE-PATH CONTRACT (%s, ALL %d rows / %d areas, whole-area sliding window, edge mode %s) | "
                "G %.3f%% | EM %.3f%% | L %.3f%% | late CVaR-20 %.2f%% | late median %.2f%% | pass@3%% %.1f%% of rows | pass@4%% %.1f%% | "
                "gates G<=%.0f EM<=%.0f L<=%.0f CVaR<=%.0f -> %s.",
                label, int(rows.size), out["areas"], out["edge_mode"], G, EM, L, cvar, med, pass3, pass4, thr_g, thr_em, thr_l, thr_cv,
                "PASS" if passed else "FAIL")
    logger.info("  %s QUALIFICATION VERDICT = %s | %s.", label, _verd["status"],
                " | ".join(_verd["reasons"]))
    logger.info("  %s per-gate median error %%: %s | worst gate %d (%.2f%%)", label,
                " ".join("%.1f" % v for v in gate_med), int(np.argmax(gate_med)) + 1, float(gate_med.max()))
    logger.info("  %s edge(<=%d from the end) late %.2f%% vs interior %.2f%% | edge rows carry %.1f%% of the late ENERGY at "
                "%.2fx the interior amplitude | by distance %s | worst-20%% tail: %.0f%% edge rows, %.0f%% rows below -10 dB | "
                "lateral roughness of the late error %.4f dex vs reference structure %.4f dex.", label, K, L_edge, L_int,
                100.0 * e_share, amp_ratio, " ".join("%s:%.2f%%" % (k, v) for k, v in by_d.items()), tail_edge, tail_deep,
                rough_out_v, rough_ref_v)
    try:
        if amp.get("quintiles"):
            logger.info("  %s AMPLITUDE AUDIT (per row) | row late median / pass@3 by amplitude quintile: %s | "
                        "low/high quintile median ratio %.2f | Spearman(log v, row late) %+.2f | edge/interior amplitude %.2f | "
                        "row late median by distance %s. ", label,
                        " ".join("v%.2f:%.2f%%/%.0f%%(n=%d)" % (q["v_median"], q["row_late_median_percent"], q["pass3_percent"], q["n"])
                                 for q in amp["quintiles"]),
                        float(amp.get("low_high_quintile_median_ratio", float("nan"))),
                        float(amp.get("spearman_logv_rowlate", float("nan"))),
                        float(amp.get("edge_interior_amplitude_ratio", float("nan"))),
                        " ".join("%s:%.2f%%" % (k, v) for k, v in (amp.get("row_late_median_by_distance") or {}).items()))
    except Exception:
        pass
    if per_bin:
        logger.info("  %s per late-local-SNR bin (population shares): %s", label,
                    " | ".join("%s G=%.2f%% L=%.2f%% n=%d (%.1f%%)" % (k, v["global_error_percent"], v["late_error_percent"],
                                                                     v["count"], v["row_share_percent"]) for k, v in per_bin.items()))
    if out_matched:
        logger.info("  %s EDGE vs INTERIOR at MATCHED late-local SNR (SHARED-support bins only; %.0f%% of edge rows covered) | "
                    "edge %.2f%% (all edge rows %.2f%%) vs interior re-weighted to the same SNR mix %.2f%% (ratio %.2f) | per bin (edge/interior): %s. "
                    "A descriptive comparison: SNR is one covariate; amplitude, curvature, coherent noise and area composition are not "
                    "controlled here.", label,
                    out_matched["edge_rows_in_shared_support_percent"], out_matched["edge_percent_shared_support"],
                    out_matched["edge_percent_all"], out_matched["interior_at_edge_snr_mix_percent"], out_matched["edge_over_matched_interior"],
                    " | ".join("%s %.1f/%.1f%% (n %d/%d)" % (k, v["edge"], v["interior"], v["n_edge"], v["n_int"]) for k, v in out_matched["per_bin"].items()))
    logger.info("  %s took %.1fs", label, time.time() - t0)
    try:
        (Path(run_dir) / "reports").mkdir(parents=True, exist_ok=True)
        atomic_json_dump(out, Path(run_dir) / "reports" / ("deliverable_contract_%s.json" % label))
    except Exception:
        pass
    return out


_DELIVERABLE_PRED_TABLE: Dict[str, Dict[str, Any]] = {}


def qualification_verdict(m: Mapping[str, Any], contract_cfg: ContractConfig, label: str = "") -> Dict[str, Any]:
    """The one acceptance function of the deliverable path."""
    checks: List[Dict[str, Any]] = []
    def _add(name: str, value: Any, thr: Optional[float], sense: str, required: bool = True) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        ok: Optional[bool]
        if thr is None or not math.isfinite(v):
            ok = None
        else:
            ok = (v <= thr) if sense == "<=" else (v >= thr)
        checks.append({"name": name, "value": v, "threshold": thr, "sense": sense, "required": required, "pass": ok})
    _add("global_nrmse_percent", m.get("global_error_percent"), 100.0 * float(contract_cfg.global_threshold), "<=")
    _add("early_mid_nrmse_percent", m.get("early_mid_error_percent"), 100.0 * float(contract_cfg.early_mid_threshold), "<=")
    _add("late_nrmse_percent", m.get("late_error_percent"), 100.0 * float(contract_cfg.late_threshold), "<=")
    _add("late_cvar20_percent", m.get("late_cvar_percent"), 100.0 * float(contract_cfg.late_cvar_threshold), "<=")
    _minc = int(getattr(contract_cfg, "late_bin_min_count", 8))
    _wb = float("nan")
    _wbn = ""
    _bins = m.get("per_snr_bin") or {}
    _bad_bins: List[str] = []
    for k, v in _bins.items():
        try:
            _cnt = float(v.get("count", 0))
            if not math.isfinite(_cnt) or _cnt < 0 or not float(_cnt).is_integer():
                _bad_bins.append(str(k))
                continue
            if int(_cnt) >= _minc:
                _le = float(v.get("late_error_percent", float("nan")))
                if not math.isfinite(_le):
                    _bad_bins.append(str(k))
                    continue
                if not math.isfinite(_wb) or _le > _wb:
                    _wb = _le
                    _wbn = str(k)
        except Exception:
            _bad_bins.append(str(k))
    if not math.isfinite(_wb) and m.get("late_bin_worst_percent") is not None:
        _wb = float(m.get("late_bin_worst_percent"))
    _add("worst_populated_late_bin_percent" + (" %s" % _wbn if _wbn else ""), _wb,
         100.0 * float(getattr(contract_cfg, "late_bin_threshold", contract_cfg.late_threshold)), "<=")
    _add("row_pass_rate_at_3_percent", m.get("pass_at_3_percent"),
         100.0 * float(getattr(contract_cfg, "row_pass_rate_target", 0.95)), ">=")
    _add("lateral_roughness_residual_over_reference", (float(m.get("lateral_roughness_out_dex", float("nan")))
                                                        / max(float(m.get("lateral_roughness_ref_dex", float("nan"))), 1e-12)),
         None, "<=", required=False)
    _add("quiet_false_anomaly_level", m.get("quiet_false_level_percent"), None, "<=", required=False)
    if _bad_bins:
        checks.append({"name": "populated_bin_completeness", "value": float("nan"), "threshold": None, "sense": "<=",
                       "required": True, "pass": None, "invalid_bins": _bad_bins})
    req = [c for c in checks if c["required"]]
    if any(c["pass"] is None for c in req):
        status = "INCOMPLETE"
    elif all(bool(c["pass"]) for c in req):
        status = "PASS"
    else:
        status = "FAIL"
    reasons = ["%s=%.3f %s %s -> %s" % (c["name"], c["value"], c["sense"],
                                        ("%.3f" % c["threshold"]) if c["threshold"] is not None else "n/a",
                                        "ok" if c["pass"] else ("MISSING" if c["pass"] is None else "FAIL"))
               + ((" [invalid bins: %s]" % ", ".join(c["invalid_bins"])) if c.get("invalid_bins") else "")
               for c in req]
    return {"status": status, "label": label, "checks": checks, "reasons": reasons}


def model_state_digest(sd: Mapping[str, Any]) -> str:
    """Full-content digest of a state dict: every key, dtype, shape and all bytes of every tensor (SHA-256)."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(("%s|%s|%s|" % (k, str(t.dtype), tuple(t.shape))).encode("utf-8"))
            h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
        else:
            h.update(("%s|%r|" % (k, v)).encode("utf-8", "replace"))
    return h.hexdigest()


def export_deliverable_arrays(model: "PEBRNet", cfg: Config, clean_cache: CleanCacheInfo, noise_cache: NoiseCacheInfo,
                              rows: np.ndarray, out_dir: Path, tag: str, logger: logging.Logger,
                              max_rows: int = 40000) -> Dict[str, Any]:
    """The deliverable export: complete work areas, predictions read from the prediction table of this label (the
    same array the contract verdict and the section figures use) or, when absent.
    """
    out: Dict[str, Any] = {"status": "INCOMPLETE"}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = np.sort(np.asarray(rows, dtype=np.int64))
        pid = np.asarray(getattr(clean_cache, "province_id"))
        keep_rows: List[int] = []
        seen: set = set()
        for a in pid[rows]:
            a = int(a)
            if a in seen:
                continue
            seen.add(a)
            ar = np.nonzero(pid == a)[0]
            if keep_rows and len(keep_rows) + int(ar.size) > int(max_rows):
                break
            keep_rows.extend(int(r) for r in ar)
        rows_full = np.sort(np.asarray(keep_rows, dtype=np.int64))
        table = _DELIVERABLE_PRED_TABLE.get(tag)
        source = ""
        pred = None
        if table is not None:
            pos = {int(r): i for i, r in enumerate(table["rows"])}
            if all(int(r) in pos for r in rows_full):
                pred = np.asarray(table["pred"], dtype=np.float64)[[pos[int(r)] for r in rows_full]]
                source = "deliverable_contract_eval prediction table (edge mode %s)" % table.get("edge_mode")
        if pred is None:
            pred = np.asarray(_paired_test_forward(model, cfg, clean_cache, noise_cache, rows_full, logger), dtype=np.float64)
            source = "whole-area sliding window on the COMPLETE areas (edge mode %s)" % str(resolve_edge_mode(cfg))
        clean = np.asarray(np.load(clean_cache.data_path, mmap_mode="r")[rows_full], dtype=np.float64)
        noisy = (np.asarray(np.load(str(clean_cache.paired_noisy_path), mmap_mode="r")[rows_full], dtype=np.float64)
                 if getattr(clean_cache, "paired_noisy_path", "") else np.full_like(clean, np.nan))
        bgp = getattr(clean_cache, "paired_background_path", "")
        clean_bg = (np.asarray(np.load(str(bgp), mmap_mode="r")[rows_full], dtype=np.float64) if bgp else clean.copy())
        ls = int(cfg.data.late_start_index)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = 10.0 * np.log10((clean[:, ls:] ** 2).sum(axis=1) / np.maximum(((noisy - clean) ** 2)[:, ls:].sum(axis=1), 1e-300))
            r_late = 100.0 * np.sqrt(((pred - clean) ** 2)[:, ls:].sum(axis=1) / np.maximum((clean ** 2)[:, ls:].sum(axis=1), 1e-300))
        posn = np.zeros(rows_full.size, dtype=np.int64)
        nsta = np.zeros(rows_full.size, dtype=np.int64)
        for a in np.unique(pid[rows_full]):
            m = pid[rows_full] == a
            posn[m] = np.arange(int(m.sum()))
            nsta[m] = int(m.sum())
        hdr = ["t_%.6g" % float(t_) for t_ in np.asarray(clean_cache.target_time).reshape(-1)]
        arrays = {"noisy": noisy, "clean": clean, "clean_bg": clean_bg, "denoised": pred}
        for k, a in arrays.items():
            _export_write_csv(out_dir / ("%s_%s.csv" % (tag, k)), hdr, a)
        meta_keys = ("clean_row", "area_id", "station_pos", "stations_in_area", "late_local_snr_db", "late_nrmse_percent")
        meta = np.stack([rows_full.astype(np.float64), pid[rows_full].astype(np.float64), posn.astype(np.float64),
                         nsta.astype(np.float64), snr, r_late], 1)
        _export_write_csv(out_dir / ("%s_meta.csv" % tag), list(meta_keys), meta)
        np.savez_compressed(out_dir / ("%s_arrays.npz" % tag), **arrays, meta=meta, meta_keys=np.asarray(meta_keys),
                            target_time=np.asarray(clean_cache.target_time))
        digest = model_state_digest(model.state_dict())
        out = {"status": "COMPLETE", "tag": tag, "rows": int(rows_full.size), "areas": int(np.unique(pid[rows_full]).size),
               "inference_protocol": "whole_area_sliding_window", "prediction_source": source,
               "edge_mode": str(resolve_edge_mode(cfg)), "model_state_digest": digest, "program_digest": _program_digest(),
               "units": "physical (global scale applied; same units as the paired caches)",
               "global_scale": float(clean_cache.global_scale), "late_start_index": ls,
               "gate_time_s": [float(t_) for t_ in np.asarray(clean_cache.target_time).reshape(-1)],
               "columns": {"clean_row": "row index into the paired clean cache", "area_id": "work area id",
                           "station_pos": "0-based station position inside the area", "late_local_snr_db": "from the pair",
                           "late_nrmse_percent": "per-row late NRMSE of `denoised` vs `clean`"}}
        atomic_json_dump(out, out_dir / ("%s_manifest.json" % tag))
        logger.info("DELIVERABLE EXPORT %s | %d rows / %d complete areas -> %s | source: %s | state %s. ", tag, out["rows"], out["areas"], str(out_dir), source, digest[:16])
    except Exception as _e:
        logger.error("deliverable export %s FAILED (status INCOMPLETE; no block-path substitute is written): %r", tag, _e)
        out = {"status": "INCOMPLETE", "tag": tag, "error": repr(_e)}
    return out


def _program_digest() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return "unavailable"


def _deliverable_contract_stage(model: "PEBRNet", cfg: Config, run_dir: Path, clean_cache: CleanCacheInfo,
                                   noise_cache: NoiseCacheInfo, logger: logging.Logger) -> None:
    if bool(getattr(cfg.data, "_deliverable_done", False)):
        logger.info("deliverable-path contract already evaluated earlier in this run.")
        return
    splits = build_splits(clean_cache, cfg)
    for lab in ("val", "test"):
        if lab in splits:
            deliverable_contract_eval(model, cfg, run_dir, clean_cache, noise_cache,
                                      np.asarray(splits[lab], dtype=np.int64), logger, lab)
    cfg.data._deliverable_done = True


_AUTOPSY_STAGES = (("z_noisy", "noisy input"), ("z_denoise", "denoise arm"), ("manifold_prior_z", "prior arm (before innovation)"),
                       ("manifold_z", "prior arm"), ("z_hat_model", "arm blend"), ("library_bg_z", "library anchor alone"),
                       ("ridge_bg_z", "continuation anchor alone"), ("z_hat_pre_event", "after the anchors"),
                       ("z_hat_pre_lateral", "after the event head"), ("z_hat", "delivered"))


_AUTOPSY_AUX = ("manifold_blend", "library_gate", "ridge_gate", "event_prob", "noise_head_reliability", "reliability",
                    "lateral_strength", "lateral_lambda_factor")


def worst_section_autopsy(model: "PEBRNet", cfg: Config, run_dir: Path, clean_cache: CleanCacheInfo,
                              noise_cache: NoiseCacheInfo, rows: np.ndarray, split: str, area_name: str,
                              logger: logging.Logger, quiet: bool = False,
                              batch: Optional[Dict[str, torch.Tensor]] = None) -> Optional[Dict[str, Any]]:
    """Stage autopsy of one section."""
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    n = int(rows.size)
    if n < 3:
        return None
    device = resolve_device(cfg.runtime.device)
    if batch is None:
        ds = BTEMDenoisingDataset(
            clean_cache.data_path, noise_cache.data_path, rows, clean_cache.global_scale, cfg.data,
            seed=cfg.train.seed, samples=n, training=False,
            noise_block_rows=noise_cache.block_rows, gate_times=clean_cache.target_time,
            real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
            real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
            province_id=np.asarray(clean_cache.province_id),
            paired_noisy_path=clean_cache.paired_noisy_path,
            paired_background_path=clean_cache.paired_background_path,
            paired_event_path=clean_cache.paired_event_path,
        )
        ds.inject_weak_bodies = False
        _it = torch.utils.data.default_collate([ds[i] for i in range(n)])
        batch = {k: _it[k] for k in ("noisy", "clean", "neighbors", "neighbor_geometry", "depth_norm") if k in _it}
    net = getattr(model, "module", model)
    xb = batch["noisy"].to(device)
    nb = batch["neighbors"].to(device) if "neighbors" in batch else None
    geo = batch["neighbor_geometry"].to(device) if "neighbor_geometry" in batch else None
    dn = batch["depth_norm"].to(device) if "depth_norm" in batch else None
    W = resolve_inference_window(cfg)
    keys = ("denoised",) + tuple(k for k, _ in _AUTOPSY_STAGES) + _AUTOPSY_AUX
    _was = bool(net.training)
    net.eval()
    try:
        with torch.no_grad():
            def _fwd(_keys):
                if W > 0 and n > W:
                    return sliding_window_forward(net, xb, nb, geo, dn, W, keys=_keys, edge_mode=resolve_edge_mode(cfg),
                                                  reflect_k=int(getattr(cfg.data, "edge_reflect_k", 0) or 0))
                _o = net(xb, nb, geo, depth_norm=dn, profile_len=n)
                return {k: _o[k] for k in _keys if k in _o}
            try:
                out = _fwd(keys)
            except Exception:
                out = _fwd(("denoised",) + tuple(k for k, _ in _AUTOPSY_STAGES))
    finally:
        if _was:
            net.train()
    G = float(clean_cache.global_scale)
    gs = net.gate_scale_norm.detach().float().reshape(1, -1).cpu().numpy().astype(np.float64)
    clean = batch["clean"][:, 0, :].numpy().astype(np.float64) * G
    noisy = batch["noisy"][:, 0, :].numpy().astype(np.float64) * G
    T = int(clean.shape[1])
    ls = int(max(1, min(int(cfg.data.late_start_index), T - 1)))

    def _amp(z):
        zz = z.detach().float().cpu().numpy().astype(np.float64).reshape(n, -1)
        if zz.shape[1] != T:
            return None
        return np.sign(zz) * np.expm1(np.abs(zz)) * gs * G

    stages: Dict[str, np.ndarray] = {}
    for k, _lab in _AUTOPSY_STAGES:
        if k in out:
            a = _amp(out[k])
            if a is not None:
                stages[k] = a
    if "denoised" in out:
        stages["denoised"] = out["denoised"].detach().float().cpu().numpy().astype(np.float64).reshape(n, -1) * G
    le = np.sum(clean[:, ls:] ** 2, axis=1)
    order = np.argsort(le)
    k3 = max(1, n // 3)
    strong = np.zeros(n, dtype=bool)
    strong[order[-k3:]] = True
    weak = np.zeros(n, dtype=bool)
    weak[order[:k3]] = True

    def _late(a, msk):
        e = a[msk][:, ls:] - clean[msk][:, ls:]
        return float(100.0 * math.sqrt(float(np.sum(e ** 2)) / max(float(np.sum(clean[msk][:, ls:] ** 2)), 1e-300)))

    def _level(a, msk):
        return float(np.median(np.sum(np.abs(a[msk][:, ls:]), axis=1) / np.maximum(np.sum(np.abs(clean[msk][:, ls:]), axis=1), 1e-300)))

    allm = np.ones(n, dtype=bool)
    table = []
    for k, lab in _AUTOPSY_STAGES + (("denoised", "delivered (physical output)"),):
        if k in stages:
            table.append((lab, _late(stages[k], allm), _late(stages[k], strong), _level(stages[k], strong),
                          _late(stages[k], weak), _level(stages[k], weak)))
    aux: Dict[str, np.ndarray] = {}
    for k in _AUTOPSY_AUX:
        if k in out:
            v = out[k].detach().float().cpu().numpy().astype(np.float64).reshape(n, -1)
            aux[k] = v
    _aux_txt = []
    for k in ("manifold_blend", "library_gate", "ridge_gate", "event_prob", "noise_head_reliability", "lateral_strength",
              "lateral_lambda_factor"):
        if k in aux:
            v = aux[k]
            vl = v[:, ls:] if v.shape[1] == T else v
            _aux_txt.append("%s strong %.3f / weak %.3f" % (k, float(np.mean(vl[strong])), float(np.mean(vl[weak]))))
    if quiet:
        return {"table": table, "stations": n, "batch": batch}
    logger.info(
        "SECTION STAGE AUTOPSY %s %s (%d stations; late NRMSE %% vs the clean reference: whole section | strongest "
        "third of the stations [late level stage/clean] | weakest third [level]) | %s | late-window means: %s. ",
        split, area_name, n,
        " ; ".join("%s %.1f | %.1f [x%.2f] | %.1f [x%.2f]" % t for t in table),
        " ; ".join(_aux_txt) if _aux_txt else "n/a")
    try:
        _rep = run_dir / "reports"
        _rep.mkdir(parents=True, exist_ok=True)
        _safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(area_name))
        np.savez_compressed(_rep / ("section_autopsy_%s_%s.npz" % (split, _safe)), rows=rows, clean=clean, noisy=noisy,
                            late_start=np.int64(ls), gate_time=np.asarray(clean_cache.target_time, dtype=np.float64),
                            **{"stage_" + k: v for k, v in stages.items()}, **{"aux_" + k: v for k, v in aux.items()})
    except Exception as _e:
        logger.warning("autopsy arrays not written: %r", _e)
    return {"table": table, "stations": n, "batch": batch}


def select_tail_panel(clean_cache: CleanCacheInfo, cfg: Config, count: int,
                          logger: logging.Logger) -> List[Dict[str, Any]]:
    """The tail panel: the validation sections where the late contract is hardest, chosen from the data alone (before
    any training, so the panel is the same for every epoch and every run on this data)
    """
    if int(count) <= 0 or not getattr(clean_cache, "paired_noisy_path", "") or getattr(clean_cache, "province_id", None) is None:
        return []
    splits = build_splits(clean_cache, cfg)
    rows = np.sort(np.asarray(splits["val" if "val" in splits else "test"], dtype=np.int64))
    if int(rows.size) == 0:
        return []
    pid = np.asarray(clean_cache.province_id)[rows]
    ls = int(cfg.data.late_start_index)
    c = np.asarray(np.load(clean_cache.data_path, mmap_mode="r")[rows][:, ls:], dtype=np.float64)
    x = np.asarray(np.load(clean_cache.paired_noisy_path, mmap_mode="r")[rows][:, ls:], dtype=np.float64)
    areas, inv = np.unique(pid, return_inverse=True)
    sc = np.bincount(inv, weights=np.sum(c ** 2, axis=1))
    sn = np.bincount(inv, weights=np.sum((x - c) ** 2, axis=1))
    cnt = np.bincount(inv)
    contrast = np.ones(int(areas.size), dtype=np.float64)
    _bp = str(getattr(clean_cache, "paired_background_path", "") or "")
    if _bp:
        b = np.asarray(np.load(_bp, mmap_mode="r")[rows][:, ls:], dtype=np.float64)
        contrast = np.sqrt(np.bincount(inv, weights=np.sum((c - b) ** 2, axis=1))
                           / np.maximum(np.bincount(inv, weights=np.sum(b ** 2, axis=1)), 1e-300))
    nsr = np.sqrt(sn / np.maximum(sc, 1e-300))
    score = np.where(cnt >= 9, contrast * nsr, -1.0)
    order = [int(i) for i in np.argsort(-score)[:int(count)] if score[i] > 0.0]
    names = list(getattr(clean_cache, "province_names", None) or [])
    panel = [{"area": int(areas[i]), "name": (names[int(areas[i])] if int(areas[i]) < len(names) else str(int(areas[i]))),
              "rows": rows[inv == i], "snr_db": float(-20.0 * math.log10(max(float(nsr[i]), 1e-12))),
              "contrast": float(contrast[i])} for i in order]
    if panel:
        logger.info("TAIL PANEL selected from the data (late body contrast x late noise/signal, %d of %d validation "
                    "sections; fixed for the run): %s.",
                    len(panel), int(areas.size),
                    "; ".join("%s (late SNR %.1f dB, body contrast %.0f%%)" % (q["name"], q["snr_db"], 100.0 * q["contrast"])
                              for q in panel))
    return panel


def tail_panel_monitor(model: "PEBRNet", cfg: Config, run_dir: Path, clean_cache: CleanCacheInfo,
                           noise_cache: NoiseCacheInfo, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Every validation: the tail panel through the deliverable path with every intermediate estimate (the autopsy,
    quiet), one line with the median over the panel -- so the log shows, epoch by epoch.
    """
    n = int(getattr(cfg.runtime, "tail_panel_areas", 0) or 0)
    if n <= 0:
        return None
    st = getattr(clean_cache, "_tail_panel", None)
    if st is None:
        st = {"panel": select_tail_panel(clean_cache, cfg, n, logger), "batch": {}}
        try:
            clean_cache._tail_panel = st
        except Exception:
            pass
    if not st["panel"]:
        return None
    tabs: Dict[str, List[Tuple[float, float, float]]] = {}
    worst = ("", -1.0)
    for q in st["panel"]:
        r = worst_section_autopsy(model, cfg, run_dir, clean_cache, noise_cache, q["rows"], "val", q["name"], logger,
                                      quiet=True, batch=st["batch"].get(q["area"]))
        if not r:
            continue
        st["batch"][q["area"]] = r["batch"]
        for lab, whole, strong, slvl, _weak, _wlvl in r["table"]:
            tabs.setdefault(lab, []).append((float(whole), float(strong), float(slvl)))
            if lab == "delivered" and float(whole) > worst[1]:
                worst = (str(q["name"]), float(whole))
    if "delivered" not in tabs:
        return None
    labs = [lab for _, lab in _AUTOPSY_STAGES
            if lab in tabs and lab != "prior arm (before innovation)"]
    med = {lab: tuple(float(np.median([t[j] for t in tabs[lab]])) for j in range(3)) for lab in labs}
    logger.info(
        "TAIL PANEL (%d hardest validation sections, deliverable path; late NRMSE %% vs the clean reference, median "
        "over the sections: whole section | strongest third of the stations [late level stage/clean]) | %s | worst section "
        "%s: delivered %.1f%%.",
        len(tabs["delivered"]), " ; ".join("%s %.1f | %.1f [x%.2f]" % ((lab,) + med[lab]) for lab in labs), worst[0], worst[1])
    return {"sections": len(tabs["delivered"]), "worst_section": worst[0], "worst_delivered_late": worst[1],
            **{("%s_late_median" % lab.replace(" ", "_")): med[lab][0] for lab in labs},
            **{("%s_strong_late_median" % lab.replace(" ", "_")): med[lab][1] for lab in labs}}


PROFILE_ROLES: Tuple[str, ...] = ("25th percentile", "median", "75th percentile")


def representative_profiles(late_err: Sequence[float], q: Sequence[float] = (25.0, 50.0, 75.0)) -> List[int]:
    """Indices of the profiles nearest to the given percentiles of the per-profile late NRMSE (distinct, in
    percentile order).
    """
    v = np.asarray(late_err, dtype=np.float64)
    ok = np.flatnonzero(np.isfinite(v))
    out: List[int] = []
    if ok.size == 0:
        return out
    for p in q:
        target = float(np.percentile(v[ok], p))
        for i in ok[np.argsort(np.abs(v[ok] - target), kind="mergesort")]:
            if int(i) not in out:
                out.append(int(i))
                break
    return out


def log10_amplitude(y: np.ndarray) -> np.ndarray:
    """Log10 of the amplitude; non-positive samples NaN (left out of a line)."""
    y = np.asarray(y, dtype=np.float64)
    return np.where(y > 0, np.log10(np.where(y > 0, y, 1.0)), np.nan)


def audit_late_window_information(
    clean_cache: "CleanCacheInfo",
    noise_cache: "NoiseCacheInfo",
    cfg: "Config",
    logger: logging.Logger,
) -> Dict[str, Any]:
    """How much information does the late measurement actually carry?"""
    late = int(cfg.data.late_start_index)
    try:
        clean_all = np.load(clean_cache.data_path, mmap_mode="r")
        noise = np.asarray(np.load(noise_cache.data_path, mmap_mode="r"), dtype=np.float64)
    except Exception as exc:
        logger.warning("Late-window information budget skipped: %s", exc)
        return {}
    if clean_all.shape[0] == 0 or noise.shape[0] == 0:
        return {}
    rng = np.random.default_rng(12345)
    clean = np.asarray(clean_all[rng.integers(0, clean_all.shape[0], 256)], dtype=np.float64)
    k = int(cfg.data.num_neighbors) + 1
    need_db = 20.0 * math.log10(1.0 / max(float(cfg.contract.late_threshold), 1e-9))
    need_1pct_db = 40.0
    stack_gain: List[float] = []
    n_noise = int(noise.shape[0])
    for _ in range(256):
        a = int(rng.integers(0, max(1, n_noise - k)))
        blk = noise[a:a + k, late:]
        if blk.shape[0] < 2:
            continue
        single = float(np.sqrt(np.mean(blk ** 2)))
        stacked = float(np.sqrt(np.mean(blk.mean(axis=0) ** 2)))
        if single > 0 and stacked > 0:
            stack_gain.append(20.0 * math.log10(single / stacked))
    gain_db = float(np.median(stack_gain)) if stack_gain else 0.0
    n_eff_meas = 10.0 ** (gain_db / 10.0)
    rows: List[Dict[str, float]] = []
    for snr_db in (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        st = snr_db + gain_db
        rows.append({"snr_db": snr_db, "late_stacked_snr_db": st,
                     "shortfall_db": need_db - st, "shortfall_1pct_db": need_1pct_db - st})
    lines = [
        "LATE-WINDOW INFORMATION BUDGET | "
        "stacking K+1=%d ADJACENT library traces gains %.1f dB (measured N_eff %.2f, real "
        "correlation included) | the late contract (%.0f%%) needs %.1f dB; 1%% needs %.1f dB"
        % (k, gain_db, n_eff_meas, 100.0 * cfg.contract.late_threshold, need_db, need_1pct_db),
        "  late-local SNR   single-trace   after stacking   shortfall(4%)   shortfall(1%)   "
        "can the MEASUREMENT carry the late window?",
    ]
    for r in rows:
        lines.append("  %+8.0f dB     %+8.1f dB     %+8.1f dB       %+7.1f dB       %+7.1f dB     %s" % (
            r["snr_db"], r["snr_db"], r["late_stacked_snr_db"], r["shortfall_db"],
            r["shortfall_1pct_db"],
            "yes (4%)" if r["shortfall_db"] <= 0 else "NO -> infer from early/mid + profile",
        ))
    first_yes = next((r["snr_db"] for r in rows if r["shortfall_db"] <= 0), None)
    lines.append(
        "  READING: the stacked measurement carries the 4%% late window from late-local %s dB "
        "upward and never below; the fusion must therefore be SNR-ADAPTIVE -- prior-dominated "
        "in the low bins, measurement-permitted in the high bins (measured-noise late floor, evidence-gated innovation) "
        "-- and the sawtooth in every bin is removable only by inference along the profile "
        ". A near-zero innovation gain everywhere is NOT the correct estimator on this "
        "axis; it was the correct answer only on the retired full-band axis."
        % ("%+.0f" % first_yes if first_yes is not None else "no tested"))
    rows_fb: List[Dict[str, float]] = []
    for snr_db in (-5.0, 0.0, 10.0, 20.0, 30.0, 35.0):
        got: List[float] = []
        for i in range(clean.shape[0]):
            c = clean[i]
            rms = float(np.sqrt(np.mean(c ** 2)))
            if rms <= 0:
                continue
            nstd = rms / (10.0 ** (snr_db / 20.0))
            nb = noise[rng.integers(0, noise.shape[0], k)]
            nrm = float(np.sqrt(np.mean(nb ** 2)))
            if nrm <= 0:
                continue
            nb = nb / nrm * nstd
            s_l = float(np.sqrt(np.mean(c[late:] ** 2)))
            n_st = float(np.sqrt(np.mean(nb[:, late:].mean(axis=0) ** 2)))
            if s_l > 0 and n_st > 0:
                got.append(20.0 * math.log10(s_l / n_st))
        if got:
            rows_fb.append({"snr_db": snr_db, "late_stacked_snr_db": float(np.median(got)),
                            "shortfall_db": need_db - float(np.median(got))})
    if rows_fb:
        lines.append("  (retired FULL-BAND axis, historical: " + "; ".join(
            "%+.0f dB full-band -> %+.1f dB late" % (r["snr_db"], r["late_stacked_snr_db"])
            for r in rows_fb) + " -- not the contract's axis)")
    logger.info("\n".join(lines))
    return {"need_db": need_db, "rows": rows, "stacking": k, "stack_gain_db": gain_db,
            "n_eff_measured": n_eff_meas, "full_band_rows": rows_fb}


def verify_frozen_contract(
    model: "PEBRNet",
    cfg: "Config",
    clean_cache: "CleanCacheInfo",
    noise_cache: "NoiseCacheInfo",
    state: Mapping[str, Any],
    logger: logging.Logger,
) -> Dict[str, float]:
    """Re-evaluate the loaded checkpoint on the paper's own held-out test stream and compare against the metrics
    stored inside the checkpoint.
    """
    device = resolve_device(cfg.runtime.device)
    gate_scale_norm = np.asarray(clean_cache.gate_scale, dtype=np.float64) / float(clean_cache.global_scale)
    controller = ConstraintController(cfg.contract)
    loss_fn = ProjectLoss(
        controller, cfg.contract, cfg.loss, cfg.data.late_start_index,
        torch.as_tensor(gate_scale_norm, dtype=torch.float32),
        late_terms_all_stages=cfg.train.late_terms_all_stages,
        ip_chargeability_max=cfg.model.ip_chargeability_max,
        relaxation_blocks=len(tuple(getattr(cfg.model, "relaxation_stretch_exponents", (1.0,)) or (1.0,))),
        profile_patch_len=int(getattr(cfg.data, "profile_patch_len", 0) or 0),
    ).to(device)
    _, _, test_loader, _, _ = make_loaders(clean_cache, noise_cache, cfg)
    metrics = evaluate_model(model, test_loader, loss_fn, device, cfg, "C")
    metrics["project_feasible"] = is_project_feasible(metrics, cfg.contract)
    stored = state.get("metrics", {}) or {}
    logger.info(
        "FROZEN CONTRACT REPRODUCED | TEST G=%.4f%% EM=%.4f%% L=%.4f%% CVaR=%.4f%% feasible=%s",
        metrics["global_error_percent"], metrics["early_mid_error_percent"],
        metrics["late_error_percent"], metrics["late_cvar_percent"], metrics["project_feasible"],
    )
    for key, label in (("global_error_percent", "G"), ("early_mid_error_percent", "EM"),
                       ("late_error_percent", "L")):
        if key in stored and np.isfinite(float(stored[key])):
            delta = float(metrics[key]) - float(stored[key])
            if abs(delta) > 0.25:
                logger.warning(
                    "  %s differs from the value recorded in the checkpoint by %+.3f pp "
                    "(checkpoint stored the VAL metric at selection time, so a small gap is "
                    "expected; a large one means the evaluation stream or the scales moved).",
                    label, delta,
                )
    return metrics
