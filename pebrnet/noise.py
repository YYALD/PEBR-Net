# PEBR-Net -- prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.
# MIT License, see LICENSE.
"""Noise models of the training augmentation.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

def late_scale_with_early_guard(clean: np.ndarray, noise: np.ndarray,
                                late_start: int, target_db: float,
                                early_floor_db: float) -> Tuple[float, float]:
    """Noise scale for a late-local SNR target, capped so the early window keeps at least ``early_floor_db`` of local
    SNR.
    """
    ls = int(max(1, late_start))
    cl = float(np.sqrt(np.mean(np.asarray(clean, dtype=np.float64)[ls:] ** 2)))
    nl = float(np.sqrt(np.mean(np.asarray(noise, dtype=np.float64)[ls:] ** 2)))
    if not (cl > 1e-300 and nl > 1e-300):
        return 0.0, float(target_db)
    s = (cl / (10.0 ** (float(target_db) / 20.0))) / nl
    if early_floor_db and early_floor_db > 0.0:
        ce = float(np.sqrt(np.mean(np.asarray(clean, dtype=np.float64)[:ls] ** 2)))
        ne = float(np.sqrt(np.mean(np.asarray(noise, dtype=np.float64)[:ls] ** 2)))
        if ce > 1e-300 and ne > 1e-300:
            s_cap = (ce / (10.0 ** (float(early_floor_db) / 20.0))) / ne
            s = min(s, s_cap)
    realized = 20.0 * math.log10(cl / max(s * nl, 1e-300)) if s > 0.0 else float(target_db)
    return float(s), float(realized)


def sample_snr_db(
    rng: np.random.Generator,
    cfg: DataConfig,
    override_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Contract SNR sampler."""
    if override_range is not None:
        lo, hi = float(override_range[0]), float(override_range[1])
        return float(rng.uniform(min(lo, hi), max(lo, hi)))
    if rng.random() < cfg.snr_below_zero_probability:
        return float(rng.uniform(cfg.snr_min_db, 0.0))
    return float(rng.uniform(0.0, cfg.snr_max_db))


def loglog_dilation_matrix(times: np.ndarray, lam: float) -> np.ndarray:
    """Linear operator that maps log|V| sampled at the gates onto log|V(t/lambda)| at the same gates."""
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    x = log_time_axis(t)
    q = x - math.log(max(float(lam), 1e-12))
    n = len(x)
    m = np.zeros((n, n), dtype=np.float64)
    for i, qi in enumerate(q):
        if qi <= x[0]:
            denom = x[1] - x[0]
            w = (qi - x[0]) / denom
            m[i, 0] = 1.0 - w
            m[i, 1] = w
        elif qi >= x[-1]:
            denom = x[-1] - x[-2]
            w = (qi - x[-2]) / denom
            m[i, -2] = 1.0 - w
            m[i, -1] = w
        else:
            j = int(np.searchsorted(x, qi, side="right") - 1)
            j = int(np.clip(j, 0, n - 2))
            w = (qi - x[j]) / (x[j + 1] - x[j])
            m[i, j] = 1.0 - w
            m[i, j + 1] = w
    return m


def compose_noise_trace(
    row: np.ndarray,
    mix_row: Optional[np.ndarray],
    cfg: "DataConfig",
    rng: np.random.Generator,
    sgn: float = 1.0,
    roll_k: int = 0,
    mix_sgn: float = 1.0,
    mix_beta: float = 0.0,
    white_ratio: Optional[float] = None,
    gate_times_s: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The training noise generator, callable from outside the dataset."""
    v = np.asarray(row, dtype=np.float64).copy()
    v *= sgn
    if roll_k:
        v = np.roll(v, roll_k)
    if mix_row is not None:
        other = np.asarray(mix_row, dtype=np.float64).copy() * mix_sgn
        if roll_k:
            other = np.roll(other, roll_k)
        v = mix_beta * v + (1.0 - mix_beta) * other
    v = v - float(np.mean(v))
    rms_struct = max(float(np.sqrt(np.mean(v**2))), cfg.minimum_noise_rms)
    if cfg.colored_noise_probability > 0.0 and rng.random() < cfg.colored_noise_probability:
        rho = float(rng.uniform(*cfg.colored_noise_ar_coeff))
        e = rng.standard_normal(len(v))
        ar = np.empty_like(e)
        acc = 0.0
        gain_ar = math.sqrt(max(1.0 - rho * rho, 1e-12))
        for i in range(len(e)):
            acc = rho * acc + gain_ar * e[i]
            ar[i] = acc
        ar -= float(np.mean(ar))
        ar_rms = max(float(np.sqrt(np.mean(ar**2))), 1e-30)
        rel = float(rng.uniform(*cfg.colored_noise_relative_rms))
        v = v + ar * (rms_struct * rel / ar_rms)
    if cfg.spike_noise_probability > 0.0 and rng.random() < cfg.spike_noise_probability:
        k = int(rng.integers(cfg.spike_count_range[0], cfg.spike_count_range[1] + 1))
        pos = rng.choice(len(v), size=min(k, len(v)), replace=False)
        amp = rng.uniform(
            cfg.spike_relative_amplitude[0], cfg.spike_relative_amplitude[1], size=len(pos)
        ) * rms_struct
        sgn_s = np.where(rng.random(len(pos)) < 0.5, -1.0, 1.0)
        v[pos] = v[pos] + amp * sgn_s
    if (gate_times_s is not None and cfg.harmonic_noise_probability > 0.0
            and rng.random() < cfg.harmonic_noise_probability):
        f = float(rng.uniform(*cfg.harmonic_freq_hz))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        h = np.sin(2.0 * math.pi * f * np.asarray(gate_times_s, dtype=np.float64) + phase)
        h -= float(np.mean(h))
        h_rms = max(float(np.sqrt(np.mean(h**2))), 1e-30)
        v = v + h * (rms_struct * float(rng.uniform(*cfg.harmonic_relative_rms)) / h_rms)
    if cfg.drift_noise_probability > 0.0 and rng.random() < cfg.drift_noise_probability:
        u = np.linspace(-1.0, 1.0, len(v))
        order = int(rng.integers(1, 3))
        d = rng.normal() * u + (rng.normal() * (u**2) if order == 2 else 0.0)
        d -= float(np.mean(d))
        d_rms = max(float(np.sqrt(np.mean(d**2))), 1e-30)
        v = v + d * (rms_struct * float(rng.uniform(*cfg.drift_relative_rms)) / d_rms)
    if cfg.burst_noise_probability > 0.0 and rng.random() < cfg.burst_noise_probability:
        wlen = int(rng.integers(cfg.burst_width_range[0], cfg.burst_width_range[1] + 1))
        start = int(rng.integers(0, max(1, len(v) - wlen)))
        b = rng.standard_normal(wlen)
        b -= float(np.mean(b))
        b_rms = max(float(np.sqrt(np.mean(b**2))), 1e-30)
        v[start : start + wlen] += b * (rms_struct * float(rng.uniform(*cfg.burst_relative_rms)) / b_rms)
    if white_ratio is not None:
        w = rng.standard_normal(len(v))
        v = v + w * (
            max(float(np.sqrt(np.mean(v**2))), cfg.minimum_noise_rms)
            * white_ratio
            / max(float(np.sqrt(np.mean(w**2))), 1e-300)
        )
    return v


SIM_NOISE_REGIMES: Tuple[str, ...] = ("smooth_bias", "spiky", "mixed")


_SIM_REGIME_WEIGHTS: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "spiky":       (1.00, 1.0, 1.0, 0.25, 0.35, 0.0),
    "smooth_bias": (0.08, 0.1, 0.4, 1.00, 1.00, 1.0),
    "mixed":       (0.60, 0.6, 0.7, 0.60, 0.70, 0.4),
}


def snr_gate_columns(n_gates: int) -> np.ndarray:
    """Gates of the designed late SNR: gates 24-31 of 31 in the manuscript, i.e."""
    k = max(1, int(round(8.0 * int(n_gates) / 31.0)))
    return np.arange(int(n_gates) - k, int(n_gates))


def smooth_random_series(rng: np.random.Generator, n: int, corr: float, size: int = 1) -> np.ndarray:
    """Zero-mean, unit-variance smooth random series of length n (Gaussian kernel of width `corr` samples)."""
    corr = max(float(corr), 0.5)
    pad = int(math.ceil(3.0 * corr))
    w = rng.standard_normal((int(size), int(n) + 2 * pad))
    k = np.exp(-0.5 * (np.arange(-pad, pad + 1) / corr) ** 2)
    k /= np.sqrt(np.sum(k ** 2))
    out = np.stack([np.convolve(w[i], k, mode="valid") for i in range(int(size))])
    if out.shape[1] < n:
        out = np.pad(out, ((0, 0), (0, int(n) - out.shape[1])), mode="edge")
    return out[:, :int(n)]


SIM_EARLY_CAP = 0.25


def simulate_late_noise(clean: np.ndarray, t_ms: np.ndarray, rng: np.random.Generator, snr_cols: np.ndarray,
                        design_snr_db: float, regime: str = "mixed", return_parts: bool = False):
    """The online simulated noise: late-time noise of a profile `clean` [S stations, T gates] at a designed late SNR,
    20 log10(rms(clean[:, snr_cols]) / rms(noise[:, snr_cols]))
    """
    clean = np.asarray(clean, dtype=np.float64)
    S, T = clean.shape
    t = np.asarray(t_ms, dtype=np.float64)
    if regime not in _SIM_REGIME_WEIGHTS:
        raise ValueError("regime must be one of %s" % (SIM_NOISE_REGIMES,))
    wf, ws, wbe, wbl, wd, wp = _SIM_REGIME_WEIGHTS[regime]
    cols = np.asarray(snr_cols, dtype=np.int64)
    late_rms = float(np.sqrt(np.mean(clean[:, cols] ** 2)))
    target = late_rms * 10.0 ** (-float(design_snr_db) / 20.0)
    tL = float(t[int(cols[0])])
    if gate_axis_has_origin(t):
        _lt = log_time_axis(t)
        u = (_lt - _lt[0]) / max(float(_lt[-1] - _lt[0]), 1e-9)
    else:
        u = (np.log(t) - math.log(t[0])) / max(math.log(t[-1]) - math.log(t[0]), 1e-9)
    tp = positive_gate_times(t)
    lvl = np.exp(0.3 * smooth_random_series(rng, S, corr=rng.uniform(4, 15))[0])
    q = rng.uniform(0.2, 0.8)
    env = np.power(tp / tL, -q)
    p_flip = rng.uniform(0.35, 0.8)
    signs = np.empty((S, T))
    signs[:, 0] = rng.choice([-1.0, 1.0], size=S)
    flips = rng.random((S, max(T - 1, 0))) < p_flip
    for j in range(1, T):
        signs[:, j] = np.where(flips[:, j - 1], -signs[:, j - 1], signs[:, j - 1])
    floor = wf * lvl[:, None] * env[None, :] * np.exp(rng.normal(0.0, 0.3, size=(S, T))) * signs
    p_sp = rng.uniform(0.01, 0.06)
    sp = (rng.random((S, T)) < p_sp) * rng.uniform(2.0, 4.0, size=(S, T)) * rng.choice([-1.0, 1.0], size=(S, T))
    spikes = ws * lvl[:, None] * env[None, :] * sp
    nseg = int(rng.integers(1, 4)) if S > 3 else 1
    cuts = np.sort(rng.choice(np.arange(1, S), size=nseg - 1, replace=False)) if nseg > 1 else np.zeros(0, dtype=int)
    seg = np.searchsorted(cuts, np.arange(S), side="right")
    drift = np.zeros((S, T))
    q_d = rng.uniform(0.3, 1.0)
    for kk in range(nseg):
        idx = seg == kk
        shp = np.power(tp / tL, -q_d) * (1.0 + 0.3 * smooth_random_series(rng, T, corr=rng.uniform(3, 8))[0])
        amp = float(rng.choice([-1.0, -1.0, 1.0])) * rng.uniform(0.5, 1.5)
        drift[idx] = amp * shp[None, :] * (1.0 + 0.15 * rng.standard_normal((int(idx.sum()), 1)))
    drift = wd * drift * lvl[:, None]
    rest = floor + spikes + drift
    ce = smooth_random_series(rng, S, corr=rng.uniform(6, 20))[0]
    cl = smooth_random_series(rng, S, corr=rng.uniform(6, 20))[0]
    b_early = wbe * (rng.normal(0.0, 0.12) + 0.05 * ce)
    b_late = wbl * (rng.uniform(-1.15, 0.45) + 0.15 * cl)
    bias_rel = (b_early[:, None] * (1.0 - u[None, :]) ** 2
                + b_late[:, None] * np.clip((u[None, :] - 0.35) / 0.65, 0.0, 1.0) ** 2)
    bias = clean * bias_rel
    tg2 = target * target
    b2 = float(np.mean(bias[:, cols] ** 2))
    if b2 >= 0.9 * tg2:
        bias = bias * math.sqrt(0.5 * tg2 / max(b2, 1e-300))
    pol = np.zeros((S, T), dtype=bool)
    if wp > 0.0:
        pol = (rng.random((S, T)) < wp * rng.uniform(0.0, 0.03)) & (u[None, :] > 0.5)
    Bw = np.where(pol[:, cols], -2.0 * clean[:, cols] - bias[:, cols], bias[:, cols])
    Rw = np.where(pol[:, cols], -rest[:, cols], rest[:, cols])
    if float(np.mean(Bw ** 2)) >= 0.9 * tg2:
        pol[:, cols] = False
        Bw, Rw = bias[:, cols], rest[:, cols]
    b2 = float(np.mean(Bw ** 2))
    r2 = float(np.mean(Rw ** 2))
    br = float(np.mean(Bw * Rw))
    disc = br * br + r2 * (tg2 - b2)
    sc = (-br + math.sqrt(max(disc, 0.0))) / max(r2, 1e-300)
    add = sc * rest
    e_end = int(cols[0])
    f_early = np.ones(T)
    if e_end > 0:
        c_rms = np.sqrt(np.mean(clean[:, :e_end] ** 2, axis=0))
        a_rms = np.sqrt(np.mean(add[:, :e_end] ** 2, axis=0))
        f_early[:e_end] = np.minimum(1.0, SIM_EARLY_CAP * c_rms / np.maximum(a_rms, 1e-300))
        f_early = np.minimum.accumulate(f_early[::-1])[::-1]
    add = add * f_early[None, :]
    noise = bias + add
    if bool(pol.any()):
        noise = np.where(pol, -2.0 * clean - noise, noise)
    if return_parts:
        return noise, {"floor": sc * floor * f_early, "spikes": sc * spikes * f_early, "bias": bias,
                       "drift": sc * drift * f_early, "regime": regime, "p_flip": float(p_flip), "q": float(q),
                       "scale": float(sc), "early_cap_factor": f_early}
    return noise


def realised_late_snr_db(clean: np.ndarray, noise: np.ndarray, snr_cols: np.ndarray) -> float:
    """20 log10(rms(clean[:, snr_cols]) / rms(noise[:, snr_cols]))."""
    c = np.asarray(clean, dtype=np.float64)[..., snr_cols]
    n = np.asarray(noise, dtype=np.float64)[..., snr_cols]
    return float(20.0 * np.log10(np.sqrt(np.mean(c ** 2)) / max(float(np.sqrt(np.mean(n ** 2))), 1e-300)))
