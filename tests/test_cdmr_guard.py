"""CDM-R inside training: a regeneration that is not finite, or leaves the network's z-domain on a withheld gate, is
replaced by the uncalibrated continuation; the withheld-gate departure keeps the gradient finite; the factorisation
retries with a diagonal jitter; the start-up check's retries after a failure on the gradients."""
import pytest
import torch

import pebrnet as pn
from conftest import GATES_POS_MS


@pytest.fixture(scope="module")
def net():
    cfg = pn.Config()
    cfg.model.base_channels, cfg.model.feature_channels = 32, 64
    cfg.model.branch_channels, cfg.model.residual_blocks = 16, 6
    pn.sync_model_gates(cfg, None)
    cfg.model.informative_continuation = "off"
    torch.manual_seed(0)
    m = pn.PEBRNet(cfg.model)
    m.set_gate_times(GATES_POS_MS)
    m.eval()
    return m


def test_guard_replaces_an_implausible_regeneration(net, monkeypatch):
    T = int(net.gate_scale_norm.numel())
    torch.manual_seed(2)
    za = 0.5 * torch.randn(6, 3, T)
    mask = torch.ones(6, 1, T)
    mask[:4, :, T // 2:] = 0.0
    bad = (20.0, float("nan"), float("inf"), -40.0)

    def corrupted(z, mk):
        out = z.clone()
        for r, v in enumerate(bad):
            out[r, 0] = torch.where(mk[r, 0] < 0.5, torch.full_like(out[r, 0], v), out[r, 0])
        return out
    monkeypatch.setattr(net, "cdmr_calibrated_ready", lambda: True)
    monkeypatch.setattr(net, "_cdmr_mixture_posterior", corrupted)
    with torch.no_grad():
        out = net.cdmr_continuation(za, mask)
        alt = net._cdmr_uncalibrated(za, mask)
    assert torch.isfinite(out).all() and float(out.abs().max()) <= 16.0
    assert torch.equal(out[:4, 0], alt[:4, 0])                       # the implausible traces
    assert torch.equal(out[:4, 1:], za[:4, 1:]) and torch.equal(out[4:], za[4:])   # every other trace as regenerated


def test_departure_keeps_the_gradient_finite(net):
    T = int(net.gate_scale_norm.numel())
    torch.manual_seed(3)
    mask = torch.ones(4, 1, T)
    mask[:2, :, T // 2:] = 0.0
    zh = (0.3 * torch.randn(4, 1, T)).requires_grad_(True)
    fill = torch.where(mask < 0.5, torch.full_like(zh, 120.0), zh.detach())
    out = net._withheld_departure(zh, {"mask": mask, "z_fill": fill}, None)
    y = pn.signed_symexp(out.float())
    loss = torch.where(mask > 0.5, y, torch.zeros_like(y)).sum()     # a term over the recorded gates only
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(zh.grad).all()
    assert float(out.detach().abs().max()) <= 16.0 + 1e-4


def test_cholesky_with_jitter():
    torch.manual_seed(4)
    x = torch.randn(10, 3, dtype=torch.float64)
    singular = x @ x.T
    with pytest.raises(Exception):
        torch.linalg.cholesky(singular)
    L = pn.chol_jitter(singular)
    assert torch.isfinite(L).all()
    assert float((L @ L.T - singular).abs().max()) < 1e-3 * float(singular.diagonal().mean())
    spd = singular + torch.eye(10, dtype=torch.float64)
    assert torch.equal(pn.chol_jitter(spd), torch.linalg.cholesky(spd))


def test_start_up_check_retries():
    assert [p[0] for p in pn.p3_recovery_plan(True, True, True, True)] == [
        "float32 without TF32 on the two cards", "one card, float32 without TF32"]
    assert pn.p3_recovery_plan(False, False, False, True) == []
    assert pn.is_gradient_failure(RuntimeError("gradient-share audit FAILED (it is part of P3, not optional)"))
    assert not pn.is_gradient_failure(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
