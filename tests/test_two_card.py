"""Two-card training on the CPU: the wrapper's own split (on patch boundaries) and gather, with the network's forward on
each card's share. A gate-loss batch whose withheld gates fall on one card's share only gives what one card gives."""
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
    return m, cfg


class CpuTwoCard(pn.PatchAlignedDataParallel):
    """The two-card wrapper without the device copies: split_sizes, the module on each share, gather."""

    def forward(self, *inputs, **kwargs):
        B = next(int(a.shape[0]) for a in list(inputs) + list(kwargs.values()) if torch.is_tensor(a) and a.dim() >= 1)
        per = self.split_sizes(B, 2)
        offs = np.cumsum([0] + per).tolist()

        def cut(x, i):
            return x[offs[i]:offs[i + 1]] if (torch.is_tensor(x) and x.dim() >= 1 and int(x.shape[0]) == B) else x
        outs = [self.module(*[cut(a, i) for a in inputs], **{k: cut(v, i) for k, v in kwargs.items()})
                for i in range(len(per))]
        self._chunk_sizes = per
        return self.gather(outs, torch.device("cpu"))


def _batch(m, cfg):
    torch.manual_seed(1)
    return pn.make_probe_batch(m, cfg, 4 * P, torch.device("cpu"))


@pytest.mark.parametrize("patch", [0, 3])
def test_gate_loss_forward_on_two_cards_equals_one_card(net, patch):
    m, cfg = net
    b = _batch(m, cfg)
    om = torch.ones_like(b["noisy"])
    om[patch * P:(patch + 1) * P, ..., -6:] = 0.0                     # withheld gates on one card's share only
    args = (b["noisy"], b["neighbors"], b["neighbor_geometry"])
    kw = {"depth_norm": b["depth_norm"], "profile_len": P, "obs_mask": om}
    with torch.no_grad():
        one = m(*args, **kw)
        two = CpuTwoCard(m, [0, 1], P)(*args, **kw)
    assert set(one) == set(two) and "gate_loss_fill_z" in one
    rows = int(b["noisy"].shape[0])
    for k, v in one.items():
        if torch.is_tensor(v) and v.dim() >= 1 and int(v.shape[0]) == rows and v.is_floating_point():
            assert float((two[k] - v).abs().max()) <= 1e-4 * float(v.abs().max()) + 1e-12, k


def test_pre_flight_two_card_check(net):
    m, cfg = net
    b = _batch(m, cfg)
    rel_plain, rel_gl = pn.two_card_check(m, CpuTwoCard(m, [0, 1], P), b, P)
    assert rel_plain < 1e-4 and rel_gl < 1e-4


def test_gather_names_an_output_one_card_lacks(net):
    w = CpuTwoCard(net[0], [0, 1], P)
    w._chunk_sizes = [2, 2]
    with pytest.raises(RuntimeError, match=r"card 1 lacks \['b'\]"):
        w.gather([{"a": torch.zeros(2), "b": torch.zeros(2)}, {"a": torch.zeros(2)}], torch.device("cpu"))
