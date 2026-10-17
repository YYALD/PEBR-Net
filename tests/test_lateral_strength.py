"""The lateral joint prior's strength is learned for each station and gate: the head's output m = 2 sigmoid(h) scales
the per-gate strength, starts weak, has no fixed lower bound and is trained through the solve; the learned per-gate
form remains available. The class-balanced lateral loss weighs the second-difference rows that touch a body and the
rows that do not equally."""
import numpy as np
import pytest
import torch

import pebrnet as pn
from conftest import GATES_POS_MS

P = 9


def _cfg(form="station"):
    cfg = pn.Config()
    cfg.model.base_channels, cfg.model.feature_channels = 32, 64
    cfg.model.branch_channels, cfg.model.residual_blocks = 16, 6
    pn.sync_model_gates(cfg, None)
    cfg.model.informative_continuation = "off"
    cfg.model.lateral_joint_form = form
    return cfg


@pytest.fixture(scope="module")
def net():
    cfg = _cfg()
    torch.manual_seed(0)
    m = pn.PEBRNet(cfg.model)
    m.set_gate_times(GATES_POS_MS)
    m.eval()
    return m, cfg


def _block(T, n_prof=2, seed=5):
    rng = np.random.default_rng(seed)
    t = np.asarray(GATES_POS_MS, dtype=np.float64)
    rows = []
    for _ in range(n_prof * P):
        y = rng.uniform(0.5, 2.0) * (t / t[0]) ** (-rng.uniform(1.5, 3.5))
        rows.append(y * (1.0 + 0.05 * rng.standard_normal(T)))
    x = torch.as_tensor(np.asarray(rows), dtype=torch.float32)[:, None, :]
    nb = torch.stack([torch.stack([x[(i // P) * P + int(np.clip(i % P + d, 0, P - 1)), 0]
                                   for d in (-4, -3, -2, -1, 1, 2, 3, 4)]) for i in range(x.shape[0])])
    return x, nb


def test_strength_is_learned_per_station(net):
    m, cfg = net
    assert cfg.model.lateral_joint_form == "station" and m.lateral_strength_head is not None
    assert m.lateral_joint.lam_floor_late == 0.0
    T = int(m.gate_scale_norm.numel())
    x, nb = _block(T)
    with torch.no_grad():
        o = m(x, nb, None, profile_len=P)
    f = o["lateral_lambda_factor"]
    assert f.shape == (x.shape[0], 1, T) and float(f.min()) > 0.0 and float(f.max()) < 2.0
    assert abs(float(f.mean()) - 2.0 / (1.0 + np.exp(2.0))) < 0.05             # starts weak (bias -2)
    assert float((o["z_hat_pre_event"] - o["z_hat_pre_lateral"]).abs().max()) > 0.0


def test_strength_head_trains_through_the_solve():
    cfg = _cfg()
    torch.manual_seed(1)
    m = pn.PEBRNet(cfg.model)
    m.set_gate_times(GATES_POS_MS)
    m.train()
    T = int(m.gate_scale_norm.numel())
    x, nb = _block(T, seed=6)
    o = m(x, nb, None, profile_len=P)
    o["z_hat"].float().pow(2).mean().backward()
    g = [p.grad for p in m.lateral_strength_head.parameters()]
    assert all(v is not None and bool(torch.isfinite(v).all()) for v in g)
    assert float(sum(v.abs().sum() for v in g)) > 0.0


def test_learned_form_has_no_head():
    cfg = _cfg("learned")
    m = pn.PEBRNet(cfg.model)
    assert m.lateral_strength_head is None
    keys = list(m.state_dict())
    assert not any(k.startswith("lateral_strength_head") for k in keys)


def test_balanced_lateral_loss_weighs_body_rows_like_quiet_rows():
    cfg = _cfg()
    torch.manual_seed(2)
    m = pn.PEBRNet(cfg.model)
    m.set_gate_times(GATES_POS_MS)
    m.train()
    T = int(m.gate_scale_norm.numel())
    x, nb = _block(T, seed=7)
    o = m(x, nb, None, profile_len=P)
    late = int(cfg.data.late_start_index)
    loss = pn.ProjectLoss(pn.ConstraintController(cfg.contract), cfg.contract, cfg.loss, late,
                          m.gate_scale_norm.detach().reshape(-1).clone(), profile_patch_len=P)
    tgt = x * 0.98
    ha = torch.zeros(x.shape[0])
    ha[4] = 1.0                                                               # one body station in the first block
    batch = {"clean": tgt, "noisy": x, "q_target": torch.ones_like(x), "has_anomaly": ha,
             "snr_db": torch.full((x.shape[0],), 10.0)}
    terms = loss(o, batch, "A")
    n = x.shape[0] // P
    g0 = late // 2
    ts = o.get("trace_scale")
    tz = pn.signed_symlog((tgt if ts is None else tgt * ts) / m.gate_scale_norm.reshape(1, 1, -1))
    yh, yt = o["z_hat_pre_event"].detach().reshape(n, P, -1)[..., g0:], tz.reshape(n, P, -1)[..., g0:]
    e = ((yh[:, 2:] - 2 * yh[:, 1:-1] + yh[:, :-2]) - (yt[:, 2:] - 2 * yt[:, 1:-1] + yt[:, :-2])).abs().mean(dim=-1)
    h = ha.reshape(n, P)
    b = torch.maximum(torch.maximum(h[:, 2:], h[:, 1:-1]), h[:, :-2]) > 0.5
    expected = 0.5 * float(e[b].mean()) + 0.5 * float(e[~b].mean())
    assert int(b.sum()) == 3 and (0.5 / 3) / (0.5 / int((~b).sum())) > 3.0      # a body row weighs > 3x a quiet row
    assert float(terms["l_lateral_balanced"].detach()) == pytest.approx(expected, rel=1e-5)
    assert cfg.loss.alpha_lateral_balanced == 1.0


def test_cli_options():
    args = pn.build_parser().parse_args(["--lateral_joint_form", "learned", "--alpha_lateral_balanced", "0.5"])
    args = pn.maybe_run_interactive_menu(args)
    c = pn.config_from_args(args)
    assert c.model.lateral_joint_form == "learned" and c.loss.alpha_lateral_balanced == 0.5
