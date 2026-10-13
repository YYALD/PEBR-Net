"""E-GSR composes with the complete baseline (Eq. 6, z = z0 + x (.) r_struct): with the residual expressed, the output
minus the output without E-GSR is the increment E tc p r on every gate, recorded and withheld, also with the lateral
joint prior and the withheld-gate departure in force; without E-GSR the output is the baseline z_hat_pre_event; with the
expression scale at 0 the output is the baseline bit for bit."""
import numpy as np
import pytest
import torch

import pebrnet as pn
from conftest import GATES_POS_MS

P = 9


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
    gen = torch.Generator().manual_seed(796)                         # a detector that has found bodies
    with torch.no_grad():
        for head, bias in ((m.event_gate_head, 3.0), (m.struct_residual_head, 0.0)):
            head[-1].weight.copy_(0.5 * torch.randn(head[-1].weight.shape, generator=gen))
            head[-1].bias.fill_(bias)
    return m


def _records(T, n_prof=2, seed=5):
    rng = np.random.default_rng(seed)
    t = np.asarray(GATES_POS_MS, dtype=np.float64)
    rows = []
    for _ in range(n_prof * P):
        y = rng.uniform(0.5, 2.0) * (t / t[0]) ** (-rng.uniform(1.5, 3.5))
        rows.append(y * (1.0 + 0.05 * rng.standard_normal(T)))
    x = torch.as_tensor(np.asarray(rows), dtype=torch.float32)[:, None, :]
    K = 8
    nb = torch.stack([torch.stack([x[(i // P) * P + int(np.clip(i % P + d, 0, P - 1)), 0]
                                   for d in (-4, -3, -2, -1, 1, 2, 3, 4)][:K]) for i in range(x.shape[0])])
    return x, nb


def _forward(net, x, nb, mask, ablate):
    net._ablate_egsr = bool(ablate)
    try:
        with torch.no_grad():
            return net(x, nb, None, profile_len=P, obs_mask=mask)
    finally:
        net._ablate_egsr = False


def _increment(o, net):
    tc = torch.sigmoid(o["event_logits"].float().masked_fill(net.nonevidence_gate_mask, -1.0e4)
                       .amax(dim=-1, keepdim=True))
    return float(net.event_expression_scale) * tc * o["event_prob"].float() * o["struct_residual"].float()


def test_egsr_is_added_to_the_complete_baseline(net):
    T = int(net.gate_scale_norm.numel())
    x, nb = _records(T)
    mask = torch.ones(x.shape[0], 1, T)
    mask[..., T - 12:] = 0.0                                          # 12 of 31 gates withheld
    net.event_expression_scale.fill_(1.0)
    try:
        full, base = _forward(net, x, nb, mask, False), _forward(net, x, nb, mask, True)
        inc = _increment(full, net)                                   # at the scale the forward used
    finally:
        net.event_expression_scale.fill_(0.0)
    assert float(inc.abs().max()) > 1e-2
    assert torch.allclose(full["z_hat"].float() - base["z_hat"].float(), inc, atol=2e-6, rtol=0.0)
    assert torch.equal(base["z_hat"], full["z_hat_pre_event"])
    assert torch.equal(base["z_hat_pre_event"], full["z_hat_pre_event"])


def test_unexpressed_egsr_leaves_the_baseline(net):
    T = int(net.gate_scale_norm.numel())
    x, nb = _records(T, seed=6)
    mask = torch.ones(x.shape[0], 1, T)
    mask[..., T - 6:] = 0.0
    net.event_expression_scale.fill_(0.0)
    with torch.no_grad():
        o = net(x, nb, None, profile_len=P, obs_mask=mask)
    assert torch.equal(o["z_hat"], o["z_hat_pre_event"])
