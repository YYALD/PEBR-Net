"""Endpoints of the gate-loss cohort: Eq. 7 NRMSE and the decay-law descriptors of Eq. A13."""
import numpy as np
import pytest

import pebrnet as pn
from conftest import GATES_POS_MS, clean_profile


def _clean(seed=0):
    t = GATES_POS_MS
    return clean_profile(np.random.default_rng(seed), t / 1000.0), t


def test_power_law_has_constant_exponent_and_zero_curvature():
    t = GATES_POS_MS
    y = 7.0 * t[None, :] ** -2.5
    p, k = pn.decay_law_descriptors(y, t)
    assert np.allclose(p, 2.5, atol=1e-9)
    assert np.allclose(k, 0.0, atol=1e-9)


def test_endpoints_identity_level_error_and_collapse():
    y, t = _clean(1)
    e = pn.gate_loss_endpoints(y, y, t, cut=13, late_start=20)
    assert e["amplitude_nrmse"] == 0 and e["late_nrmse"] == 0 and e["slope_rmse"] == 0
    e = pn.gate_loss_endpoints(1.1 * y, y, t, cut=13, late_start=20)       # a pure level error
    assert e["amplitude_nrmse"] == pytest.approx(10.0, abs=1e-9) and e["slope_rmse"] < 1e-9
    yc = y.copy()
    yc[:, 25:] = -1e-3 * np.abs(y[:, 25:])                                 # collapses through zero
    assert pn.gate_loss_endpoints(yc, y, t, cut=13, late_start=20)["slope_rmse"] > 1.0


def test_levels_of_the_manuscript_and_extension():
    cfg = pn.Config()
    t = GATES_POS_MS
    levels = pn._gate_loss_levels(cfg, 31, t)
    assert [lv["withheld_gates"] for lv in levels] == [3, 6, 12, 18, 19, 22]
    assert [round(lv["withheld_percent"], 1) for lv in levels] == [9.7, 19.4, 38.7, 58.1, 61.3, 71.0]


def test_rank_statistics_and_clusters():
    a = np.arange(20.0)
    assert pn.spearman(a, a ** 3) == pytest.approx(1.0)
    assert np.isnan(pn.spearman(a, np.ones_like(a)))
    lab = pn.input_quality_clusters(np.linspace(-30.0, 10.0, 25), np.full(25, 0.1), k=5)
    assert sorted(set(lab.tolist())) == [0, 1, 2, 3, 4] and np.all(np.diff(lab) >= 0)
    m, lo, hi = pn.bootstrap_median_ci(np.arange(101.0), n_boot=500)
    assert m == 50.0 and lo < 50.0 < hi


def test_descriptors_on_an_axis_that_starts_at_the_time_origin():
    """p and kappa of Eq. A13 are undefined at t = 0 (NaN there) and finite on every positive gate; the endpoints
    over the withheld gates stay finite."""
    from conftest import GATES_S
    t_ms = GATES_S * 1000.0
    y = clean_profile(np.random.default_rng(3))
    with np.errstate(divide="raise", invalid="raise"):
        p, k = pn.decay_law_descriptors(y, t_ms)
    assert np.isnan(p[:, 0]).all() and np.isnan(k[:, 0]).all()
    assert np.isfinite(p[:, 1:]).all() and np.isfinite(k[:, 1:]).all()
    e = pn.gate_loss_endpoints(y * 1.02, y, t_ms, cut=9, late_start=23)
    assert all(np.isfinite(e[key]) for key in ("amplitude_nrmse", "late_nrmse", "slope_rmse", "structure_nrmse"))
    assert e["amplitude_nrmse"] == pytest.approx(2.0, rel=1e-9)
