"""Reading a paired dataset in the layout of the release: a gate axis that starts at t = 0, the grouping of
near-identical profiles into one split, and the leakage audit by profile."""
import json
import logging

import numpy as np
import pytest
import torch

import pebrnet as pn
from conftest import GATES_S, write_paired_dataset

SPLITS = {"train": [1, 2, 3, 4, 5, 6], "val": [7, 2508], "test": [9, 2503]}
COPIES = [(3, 2503), (2, 2508)]        # profiles 3 and 2 stored twice, in different splits of the files


def _cache(tmp_path, policy):
    root = write_paired_dataset(tmp_path / "data", SPLITS, seed=5, copies=COPIES)
    cfg = pn.Config()
    cfg.data.paired_split_policy = policy
    cfg.runtime.overwrite_cache = True
    cache = pn.build_paired_cache(root, tmp_path / ("cache_" + policy), cfg, logging.getLogger("test"))
    return cfg, cache


def _split_of(cache, sid):
    names = list(cache.province_names)
    rows = [i for i, p in enumerate(np.asarray(cache.province_id)) if names[p].endswith("#sample_%d" % sid)]
    sp = set(np.asarray(cache.paired_split_id)[rows].tolist())
    assert len(rows) == 61 and len(sp) == 1
    return pn.PAIRED_SPLIT_NAMES[sp.pop()]


@pytest.fixture(scope="module")
def grouped(tmp_path_factory):
    return _cache(tmp_path_factory.mktemp("g"), "grouped")


@pytest.fixture(scope="module")
def as_files(tmp_path_factory):
    return _cache(tmp_path_factory.mktemp("f"), "files")


def test_gate_axis_starts_at_the_time_origin(grouped):
    _cfg, cache = grouped
    t = np.asarray(cache.target_time)
    assert np.allclose(t, GATES_S, rtol=1e-12, atol=0.0) and t[0] == 0.0
    assert pn.gate_axis_has_origin(t)
    lt = pn.log_time_axis(t)
    assert np.all(np.isfinite(lt)) and np.all(np.diff(lt) > 0)
    assert lt[0] == pytest.approx(2.0 * np.log(t[1]) - np.log(t[2]))
    assert np.array_equal(pn.log_time_axis(t[1:]), np.log(t[1:]))          # a positive axis is unchanged


def test_grouped_policy_puts_every_copy_in_the_split_of_its_lowest_sample_id(grouped):
    _cfg, cache = grouped
    assert _split_of(cache, 3) == "train" and _split_of(cache, 2503) == "train"
    assert _split_of(cache, 2) == "train" and _split_of(cache, 2508) == "train"
    assert _split_of(cache, 9) == "test" and _split_of(cache, 7) == "val"
    meta = json.loads(open(cache.metadata_path, encoding="utf-8").read())["profile_groups"]
    assert meta["policy"] == "grouped" and meta["groups_with_copies"] == 2
    assert sorted(map(tuple, meta["moved_profiles"])) == [(2503, "test", "train"), (2508, "val", "train")]
    assert meta["max_rel_diff_within_groups"] < 1e-3


def test_files_policy_keeps_the_split_of_the_files(as_files):
    _cfg, cache = as_files
    assert _split_of(cache, 2503) == "test" and _split_of(cache, 2508) == "val"


@pytest.mark.parametrize("policy, passes", [("grouped", True), ("files", False)])
def test_leakage_audit_by_profile(grouped, as_files, policy, passes):
    cfg, cache = grouped if policy == "grouped" else as_files
    splits = pn.build_splits(cache, cfg)
    out = pn.cross_split_duplicate_probe(cache, splits, logging.getLogger("test"))
    assert out["split_leakage_pass"] is passes
    assert set(out["profile_groups_in_two_splits"]) == {"train_val", "train_test", "val_test"}


def test_network_reads_the_axis(grouped):
    _cfg, cache = grouped
    cfg = pn.Config()
    cfg.model.base_channels, cfg.model.feature_channels = 32, 64
    cfg.model.branch_channels, cfg.model.residual_blocks = 16, 6
    pn.sync_model_gates(cfg, None)
    m = pn.PEBRNet(cfg.model)
    m.set_gate_times(np.asarray(cache.target_time))
    assert torch.isfinite(m.gate_position).all() and torch.isfinite(m.decay_atoms).all()
    x = torch.as_tensor(np.array(np.load(cache.paired_noisy_path, mmap_mode="r")[:8]), dtype=torch.float32).unsqueeze(1)
    m.eval()
    with torch.no_grad():
        mask = torch.ones_like(x)
        mask[..., 19:] = 0.0                                                   # 12 of 31 gates withheld
        assert torch.isfinite(m(x)["denoised"]).all() and torch.isfinite(m(x, obs_mask=mask)["denoised"]).all()


def test_a_negative_or_unordered_axis_is_refused():
    for bad in (np.array([-1e-4, 1e-3, 2e-3]), np.array([0.0, 0.0, 1e-3]), np.array([1e-3, 5e-4, 2e-3])):
        with pytest.raises(ValueError):
            pn.validate_gate_axis(bad)
