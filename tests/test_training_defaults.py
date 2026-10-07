"""docs/training.md, checked against the code: every training default and rule that page states."""
import logging
import shlex
from pathlib import Path

import pytest
import torch

import pebrnet as pn

ROOT = Path(__file__).resolve().parents[1]


def _cfg_from(argv):
    args = pn.build_parser().parse_args(argv)
    args = pn.maybe_run_interactive_menu(args)
    return args, pn.config_from_args(args)


def test_optimiser_schedule_and_seed():
    t = pn.TrainConfig()
    assert (t.base_lr, t.weight_decay, t.grad_clip_norm) == (1e-3, 1e-4, 1.0)
    assert t.use_ema and t.ema_decay == 0.999
    assert t.seed == 20260710 and t.validate_every == 1
    assert (t.lr_warmup_fraction, t.lr_min_ratio) == (0.03, 0.02)


def test_stages_hand_over_and_remediation():
    t = pn.TrainConfig()
    assert (t.stage_a_max_epochs, t.stage_b_max_epochs, t.stage_c_epochs) == (60, 80, 120)
    assert t.stage_a_min_epochs_before_transition == 50 and t.stage_switch_plateau_patience == 8
    assert (t.stage_a_switch_patience, t.stage_a_min_plateau) == (5, 4)
    assert (t.stage_b_switch_global, t.stage_b_switch_late, t.stage_b_switch_patience) == (0.035, 0.05, 5)
    assert t.early_stop_patience == 60
    assert (t.stage_regression_ratio, t.stage_regression_patience, t.stage_regression_min_epochs) == (1.5, 12, 50)
    assert (t.remediation_rounds, t.remediation_epochs, t.remediation_lr_scale) == (2, 12, 0.25)
    assert t.stage_init_from_previous_best
    budget = t.stage_a_max_epochs + t.stage_b_max_epochs + t.stage_c_epochs
    assert budget + t.remediation_rounds * t.remediation_epochs == 284


def test_batch_epoch_and_contract():
    t, d, c = pn.TrainConfig(), pn.DataConfig(), pn.ContractConfig()
    assert (t.batch_size, t.patch_batch_cap, d.profile_patch_len) == (512, 2048, 9)
    # the conversion of main(): whole patches, capped at patch_batch_cap rows
    rows = min(t.batch_size * d.profile_patch_len, t.patch_batch_cap) // d.profile_patch_len * d.profile_patch_len
    assert rows == 2043 and rows // d.profile_patch_len == 227
    assert d.train_samples_per_epoch == 200_000
    assert t.optimizer_batch_policy == "micro" and t.min_effective_rows == 288 and t.cpu_micro_batch_rows == 144
    thresholds = (c.global_threshold, c.early_mid_threshold, c.late_threshold, c.late_cvar_threshold)
    assert thresholds == (0.03, 0.03, 0.03, 0.08)
    assert d.paired_split_policy == "grouped" and d.paired_copy_rel_tol == 2e-3


@pytest.mark.parametrize("micro, accum, expected", [(621, 4, (621, 1, 621)), (117, 18, (117, 3, 351)),
                                                    (2043, 1, (2043, 1, 2043))])
def test_rows_per_optimiser_update(micro, accum, expected):
    cfg = pn.Config()
    cfg.train.batch_size = 2043
    log = logging.getLogger("test")
    assert pn.apply_optimizer_batch_policy(micro, accum, cfg, log) == expected
    cfg.train.optimizer_batch_policy = "legacy"
    m, a, _eff = pn.apply_optimizer_batch_policy(micro, accum, cfg, log)
    assert (m, a) == (micro, accum) and m * a >= 2043


@pytest.fixture(scope="module")
def model():
    cfg = pn.Config()
    cfg.model.base_channels, cfg.model.feature_channels = 32, 64
    cfg.model.branch_channels, cfg.model.residual_blocks = 16, 6
    pn.sync_model_gates(cfg, None)
    torch.manual_seed(0)
    return pn.PEBRNet(cfg.model)


@pytest.mark.parametrize("stage, factors", [("A", (1.0, 1.0, 1.0, 0.35)), ("B", (0.10, 0.20, 0.50, 1.00)),
                                            ("C", (0.075, 0.075, 0.075, 0.075))])
def test_learning_rate_per_stage_and_group(model, stage, factors):
    t = pn.TrainConfig()
    opt = pn.build_optimizer(model, t, stage)
    assert opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["weight_decay"] == 1e-4
    groups = {g["name"]: g for g in opt.param_groups}
    lr = {k: g["lr"] / t.base_lr for k, g in groups.items()}
    for name, f in zip(("shared", "noise", "router", "refine"), factors):
        assert lr[name] == pytest.approx(f)
    if "prior_sigma" in groups:
        assert lr["prior_sigma"] == pytest.approx(20.0) and groups["prior_sigma"]["weight_decay"] == 0.0
    if "lateral" in groups:
        assert lr["lateral"] == pytest.approx(lr["shared"]) and groups["lateral"]["weight_decay"] == 0.0
    if "uncertainty" in groups:
        assert lr["uncertainty"] == pytest.approx(lr["router"])
    covered = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) in covered for p in model.parameters() if p.requires_grad)


def test_schedule_within_a_stage():
    t = pn.TrainConfig()
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    f = pn.build_scheduler(opt, t, 1000).lr_lambdas[0]
    assert f(0) == pytest.approx(0.02 + 0.98 / 30) and f(29) == pytest.approx(1.0)       # 3 % of 1000 steps
    assert f(30) == pytest.approx(1.0) and f(1000) == pytest.approx(0.02)
    assert f(30 + 485) == pytest.approx(0.02 + 0.98 * 0.5, abs=2e-3)                       # half way down the cosine


def test_command_line_defaults_reach_the_configuration():
    args, cfg = _cfg_from([])
    assert args.mode == "all" and args.ablation == "full" and args.confirm_full_from_scratch
    assert (cfg.train.stage_a_max_epochs, cfg.train.stage_b_max_epochs, cfg.train.stage_c_epochs) == (60, 80, 120)
    assert (cfg.train.batch_size, cfg.train.base_lr, cfg.train.seed) == (512, 1e-3, 20260710)
    assert cfg.train.large_vram_preset is None and cfg.train.optimizer_batch_policy == "micro"


@pytest.mark.parametrize("name", ["train.args", "evaluate.args", "gate_loss.args", "audit.args"])
def test_argument_files_leave_training_parameters_at_their_defaults(name):
    path = ROOT / "configs" / name
    flags = {tok for line in path.read_text(encoding="utf-8").splitlines()
             for tok in shlex.split(line, comments=True) if tok.startswith("--")}
    allowed = {"--mode", "--paired_dir", "--device", "--gate_loss_profiles", "--gate_loss_withheld",
               "--gate_loss_fan_withheld", "--confirm_full_from_scratch", "--output_root", "--model_root",
               "--plot_root", "--log_root"}
    assert flags <= allowed, flags - allowed
    _args, cfg = _cfg_from(["@" + str(path)])
    ref = pn.TrainConfig()
    for field in ("base_lr", "weight_decay", "batch_size", "seed", "stage_a_max_epochs", "stage_b_max_epochs",
                  "stage_c_epochs", "early_stop_patience", "ema_decay", "grad_clip_norm", "optimizer_batch_policy"):
        assert getattr(cfg.train, field) == getattr(ref, field), field
    assert cfg.paths.paired_dir == "data/paired"
