"""The online simulated late-time noise: exact designed SNR, bounded early band, regime morphology."""
import numpy as np
import pytest

import pebrnet as pn
from conftest import GATES_POS_MS, GATES_S, clean_profile


def _profile(seed: int):
    rng = np.random.default_rng(seed)
    t = GATES_POS_MS
    return clean_profile(rng, t / 1000.0), t, pn.snr_gate_columns(31), rng


def test_gate_axis_and_snr_window():
    t = GATES_S * 1000.0
    assert t.shape == (31,) and t[0] == 0.0 and abs(t[-1] - 20.0) < 1e-12 and np.all(np.diff(t) > 0)
    assert list(pn.snr_gate_columns(31)) == list(range(23, 31))      # gates 24-31 of the manuscript


@pytest.mark.parametrize("regime", ["spiky", "smooth_bias", "mixed"])
@pytest.mark.parametrize("snr_db", [-30.0, -12.5, 0.0, 10.0])
def test_realised_late_snr_equals_designed(regime, snr_db):
    for seed in range(3):
        y, t, cols, rng = _profile(100 + seed)
        n = pn.simulate_late_noise(y, t, rng, cols, snr_db, regime)
        assert abs(pn.realised_late_snr_db(y, n, cols) - snr_db) < 1e-6


def test_early_band_is_bounded():
    for seed in range(5):
        y, t, cols, rng = _profile(200 + seed)
        _n, parts = pn.simulate_late_noise(y, t, rng, cols, -30.0, "spiky", return_parts=True)
        add = parts["floor"] + parts["spikes"] + parts["drift"]
        e = slice(0, int(cols[0]))
        ratio = np.sqrt(np.mean(add[:, e] ** 2, axis=0)) / np.sqrt(np.mean(y[:, e] ** 2, axis=0))
        assert np.all(ratio <= pn.SIM_EARLY_CAP + 1e-9)
        assert np.all(np.diff(parts["early_cap_factor"]) >= -1e-12)        # non-decreasing in time


def test_regime_morphology():
    """Spiky records alternate in sign across the late gates; smoothly biased records rarely do."""
    rate = {}
    for regime in ("spiky", "smooth_bias"):
        r = []
        for seed in range(8):
            y, t, cols, rng = _profile(300 + seed)
            x = (y + pn.simulate_late_noise(y, t, rng, cols, -15.0, regime))[:, cols]
            r.append(np.mean(np.signbit(x[:, 1:]) != np.signbit(x[:, :-1])))
        rate[regime] = float(np.mean(r))
    assert rate["spiky"] > 2.0 * rate["smooth_bias"]


@pytest.mark.parametrize("snr_db", [-10.0, 0.0, 10.0])
def test_axis_that_starts_at_the_time_origin(snr_db):
    """A gate axis whose first gate is t = 0, where (t / t_L)^-q and ln t are not defined."""
    rng = np.random.default_rng(7)
    t_ms = GATES_S * 1000.0
    y = clean_profile(rng)
    cols = pn.snr_gate_columns(31)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        n = pn.simulate_late_noise(y, t_ms, rng, cols, snr_db, "mixed")
    assert np.all(np.isfinite(n))
    assert abs(pn.realised_late_snr_db(y, n, cols) - snr_db) < 1e-6
