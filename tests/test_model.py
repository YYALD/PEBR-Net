"""PEBR-Net forward pass and the withheld-gate (gate-loss) path: CDM-R continuation and its ablation."""
import numpy as np
import pytest
import torch

import pebrnet as pn
from conftest import GATES_POS_MS


@pytest.fixture(scope="module")
def model():
    cfg = pn.Config()
    # a narrower trunk keeps the test fast; every mechanism of the network is still present
    cfg.model.base_channels, cfg.model.feature_channels = 32, 64
    cfg.model.branch_channels, cfg.model.residual_blocks = 16, 6
    pn.sync_model_gates(cfg, None)
    cfg.model.informative_continuation = "off"      # single-pass reading; the two-pass reading is tested below
    torch.manual_seed(0)
    m = pn.PEBRNet(cfg.model)
    m.set_gate_times(GATES_POS_MS)
    m.eval()
    return m


def _batch(n=4, seed=0):
    rng = np.random.default_rng(seed)
    t = GATES_POS_MS
    y = (rng.uniform(0.5, 2.0, size=(n, 1)) * t[None, :] ** -rng.uniform(1.5, 3.0, size=(n, 1)))
    y = y + 0.02 * y * rng.standard_normal(y.shape)
    return torch.as_tensor(y, dtype=torch.float32).unsqueeze(1), t


@torch.no_grad()
def test_forward_shapes_and_mask_of_ones_is_identity(model):
    x, _t = _batch()
    out = model(x)
    assert out["denoised"].shape == x.shape and torch.isfinite(out["denoised"]).all()
    out1 = model(x, obs_mask=torch.ones_like(x))
    assert torch.equal(out["denoised"], out1["denoised"])          # every gate recorded: bit for bit


@torch.no_grad()
def test_withheld_gates_are_regenerated_and_recorded_gates_untouched(model):
    x, t = _batch(seed=1)
    cut = 19                                                       # 12 of 31 gates withheld (38.7 %)
    m = torch.ones_like(x)
    m[..., cut:] = 0.0
    xin = x.clone()
    xin[..., cut:] = 1e6                                           # whatever a withheld gate holds is ignored
    gs = model.gate_scale_norm.to(dtype=x.dtype)
    x_fill, _nb, info = model._gate_loss_fill(xin, None, m, gs)
    assert info is not None
    assert torch.equal(x_fill[..., :cut], x[..., :cut])
    assert torch.isfinite(x_fill).all() and float(x_fill[..., cut:].abs().max()) < 1e3
    out = model(xin, obs_mask=m)
    assert torch.isfinite(out["denoised"]).all()


@torch.no_grad()
def test_power_law_is_continued_exactly_without_a_manifold(model):
    """Before a decay manifold is installed, CDM-R continues the last recorded log-slope (a power law)."""
    t = GATES_POS_MS
    y = torch.as_tensor(3.0 * t ** -2.2, dtype=torch.float32).view(1, 1, -1)
    m = torch.ones_like(y)
    m[..., 20:] = 0.0
    gs = torch.ones_like(model.gate_scale_norm, dtype=y.dtype)
    x_fill, _nb, _info = model._gate_loss_fill(y, None, m, gs)
    rel = (x_fill[..., 20:] - y[..., 20:]).abs() / y[..., 20:]
    assert float(rel.max()) < 1e-3


def test_informative_mask_rule(model):
    """The window is removed from the first two consecutive uninformative gates on, never before the minimum."""
    thr = np.exp(-model.cfg.informative_kappa)
    q = torch.full((3, 1, 31), 0.9)
    q[0, 0, 20] = 0.1                       # one isolated dip: no cut
    q[1, 0, 20:] = 0.1                      # uninformative from gate 21 on: cut at index 20
    q[2, 0, 3:] = 0.1                       # uninformative from gate 4 on: cut held at the minimum
    m = model.informative_mask(q)
    assert thr > 0.1 and m.shape == q.shape
    assert float(m[0].min()) == 1.0
    assert float(m[1, 0, :20].min()) == 1.0 and float(m[1, 0, 20:].max()) == 0.0
    k = int(model.cfg.informative_min_gates)
    assert float(m[2, 0, :k].min()) == 1.0 and float(m[2, 0, k:].max()) == 0.0


@torch.no_grad()
def test_two_pass_informative_reading(model):
    x, _t = _batch(seed=3)
    model.cfg.informative_continuation = "on"
    try:
        out = model(x)
    finally:
        model.cfg.informative_continuation = "off"
    assert "informative_mask" in out and torch.isfinite(out["denoised"]).all()
    if float(out["informative_mask"].min()) == 1.0:          # every gate informative: the single-pass result
        assert torch.equal(out["denoised"], model(x)["denoised"])


@torch.no_grad()
def test_cdmr_ablation_withholds_everything(model):
    x, _t = _batch(seed=2)
    m = torch.ones_like(x)
    m[..., 13:] = 0.0
    gs = model.gate_scale_norm.to(dtype=x.dtype)
    model._ablate_cdmr = True
    try:
        x_fill, _nb, _info = model._gate_loss_fill(x, None, m, gs)
    finally:
        model._ablate_cdmr = False
    assert float(x_fill[..., 13:].abs().max()) == 0.0
    assert torch.equal(x_fill[..., :13], x[..., :13])
