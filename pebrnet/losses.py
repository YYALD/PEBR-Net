# PEBR-Net -- prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.
# MIT License, see LICENSE.
"""Losses and the constraint controller.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

def _huber_z(e: torch.Tensor, delta: float) -> torch.Tensor:
    if not (delta > 0.0):
        return torch.abs(e)
    a = torch.abs(e)
    return torch.where(a >= delta, a, e * e / (2.0 * delta) + 0.5 * delta)


def completion_observation_weight(num_gates: int, late_start: int) -> np.ndarray:
    """The one observation operator of the coefficient completion A: unit weight on the early/mid gates, 1e-3 on the
    late gates.
    """
    w = np.ones(int(num_gates), dtype=np.float64)
    w[int(late_start):] = 1e-3
    return w


def split_masks(length: int, late_start: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    late = torch.zeros(1, 1, length, device=device, dtype=dtype)
    late[..., late_start:] = 1.0
    early_mid = 1.0 - late
    return early_mid, late


def relative_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    error2 = (pred - target).pow(2)
    target2 = target.pow(2)
    if mask is not None:
        error2 = error2 * mask
        target2 = target2 * mask
    return error2.sum() / (target2.sum() + eps)


def per_sample_relative_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    error2 = (pred - target).pow(2)
    target2 = target.pow(2)
    if mask is not None:
        error2 = error2 * mask
        target2 = target2 * mask
    dims = tuple(range(1, pred.ndim))
    return torch.sqrt(error2.sum(dim=dims) / (target2.sum(dim=dims) + ROW_DENOM_EPS) + eps)


def first_difference(x: torch.Tensor) -> torch.Tensor:
    return x[..., 1:] - x[..., :-1]


def second_difference(x: torch.Tensor) -> torch.Tensor:
    return first_difference(first_difference(x))


class ConstraintController:
    """Multipliers and penalties of the augmented-Lagrangian constraints."""
    def __init__(self, cfg: ContractConfig) -> None:
        self.cfg = cfg
        self.lambdas = {
            "global": float(cfg.initial_lambda_global),
            "early_mid": float(cfg.initial_lambda_early_mid),
            "late": float(cfg.initial_lambda_late),
            "late_cvar": float(cfg.initial_lambda_late_cvar),
            "anomaly_late": float(cfg.initial_lambda_anomaly_late),
        }
        self.rhos = {
            "global": float(cfg.initial_rho_global),
            "early_mid": float(cfg.initial_rho_early_mid),
            "late": float(cfg.initial_rho_late),
            "late_cvar": float(cfg.initial_rho_late_cvar),
            "anomaly_late": float(cfg.initial_rho_anomaly_late),
        }
        self.stagnation_counts = {k: 0 for k in self.lambdas}
        self.previous_positive_violation = {k: None for k in self.lambdas}
        self.smoothed_violation = {k: None for k in self.lambdas}
        self.stall_counts = {k: 0 for k in self.lambdas}
        self.frozen = {k: False for k in self.lambdas}
        self.frozen_val = {k: False for k in self.lambdas}
        self.val_best = {k: None for k in self.lambdas}
        self.val_stall = {k: 0 for k in self.lambdas}
        self.val_met = {k: True for k in self.lambdas}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "lambdas": self.lambdas.copy(),
            "rhos": self.rhos.copy(),
            "stagnation_counts": self.stagnation_counts.copy(),
            "previous_positive_violation": self.previous_positive_violation.copy(),
            "smoothed_violation": self.smoothed_violation.copy(),
            "stall_counts": self.stall_counts.copy(),
            "val_best": self.val_best.copy(),
            "frozen_val": self.frozen_val.copy(),
            "val_stall": self.val_stall.copy(),
            "val_met": self.val_met.copy(),
            "frozen": self.frozen.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.lambdas.update({k: float(v) for k, v in state.get("lambdas", {}).items()})
        self.rhos.update({k: float(v) for k, v in state.get("rhos", {}).items()})
        self.stagnation_counts.update({k: int(v) for k, v in state.get("stagnation_counts", {}).items()})
        self.previous_positive_violation.update(state.get("previous_positive_violation", {}))
        self.smoothed_violation.update(state.get("smoothed_violation", {}))
        self.stall_counts.update({k: int(v) for k, v in state.get("stall_counts", {}).items()})
        self.val_best.update(state.get("val_best", {}))
        self.val_stall.update({k: int(v) for k, v in state.get("val_stall", {}).items()})
        self.val_met.update({k: bool(v) for k, v in state.get("val_met", {}).items()})
        self.frozen_val.update({k: bool(v) for k, v in state.get("frozen_val", {}).items()})
        self.frozen.update({k: bool(v) for k, v in state.get("frozen", {}).items()})

    def threshold(self, name: str) -> float:
        return {
            "global": self.cfg.global_threshold,
            "early_mid": self.cfg.early_mid_threshold,
            "late": self.cfg.late_threshold,
            "late_cvar": effective_late_cvar_threshold(self.cfg),
            "anomaly_late": self.cfg.late_threshold * self.cfg.anomaly_late_multiple,
        }[name]

    ROOT_STATISTICS: Tuple[str, ...] = ("late_cvar",)

    def violation(self, relative_mse_value: torch.Tensor, name: str) -> torch.Tensor:
        tau = self.threshold(name)
        if name in self.ROOT_STATISTICS:
            return relative_mse_value / tau - 1.0
        return torch.sqrt(relative_mse_value + 1.0e-12) / tau - 1.0

    def term(self, relative_mse_value: torch.Tensor, name: str, active: bool = True) -> torch.Tensor:
        if not active:
            return relative_mse_value * 0.0
        v = self.violation(relative_mse_value, name)
        positive = F.relu(v)
        cap = max(float(self.cfg.violation_soft_cap), 1.0)
        if name in self.ROOT_STATISTICS:
            cap = min(cap, max(float(self.cfg.cvar_violation_cap), 1.0))
        stable_positive = cap * torch.tanh(positive / cap)
        return self.lambdas[name] * stable_positive + 0.5 * self.rhos[name] * stable_positive.pow(2)

    def note_validation(self, val_metrics: Mapping[str, float], logger: Optional[logging.Logger] = None,
                        constraint_scale: float = 1.0) -> None:
        """Freeze (and gently decay) a multiplier whose validation metric has stopped improving."""
        keys = {
            "global": "global_error_percent",
            "early_mid": "early_mid_error_percent",
            "late": "late_error_percent",
            "late_cvar": "late_cvar_percent",
            "anomaly_late": "anomaly_late_error_percent",
        }
        for name, mkey in keys.items():
            v = val_metrics.get(mkey)
            if v is None or not math.isfinite(float(v)):
                continue
            v = float(v)
            self.val_met[name] = bool(v <= 100.0 * float(self.threshold(name)))
            best = self.val_best[name]
            if best is None or v < float(best) - 1e-4:
                self.val_best[name] = v
                self.val_stall[name] = 0
                self.frozen_val[name] = False
                continue
            self.val_stall[name] += 1
            if float(constraint_scale) < 0.95:
                if (self.val_stall[name] == int(self.cfg.lambda_stall_patience)
                        and logger is not None):
                    logger.info(
                        "%s validation stalled %d epochs, but the constraint "
                        "block is rescaled to x%.3f: the effective weight has been "
                        "diluted, not proven futile. Freeze withheld.",
                        name, self.val_stall[name], float(constraint_scale),
                    )
                continue
            if self.val_stall[name] >= int(self.cfg.lambda_stall_patience) and not self.frozen_val[name]:
                self.frozen_val[name] = True
                if logger is not None:
                    logger.warning(
                        "LAMBDA FROZEN (%s): the VALIDATION metric has not improved for %d epochs "
                        "(best %.4f%%, now %.4f%%) while lambda kept ramping to %.1f. Pushing a "
                        "penalty harder on a constraint that is regressing on held-out data only "
                        "distorts the loss; the multiplier is frozen and %s from here.",
                        name, self.val_stall[name], float(best), v, self.lambdas[name],
                        ("decayed (validation gate met)" if self.val_met[name] else
                         "HELD, not decayed (validation gate still violated)"),
                    )

    def update(self, relative_mse_values: Mapping[str, float], active_names: Iterable[str]) -> Dict[str, float]:
        active_set = set(active_names)
        violations: Dict[str, float] = {}
        if not hasattr(self, "last_constraint_scale"):
            self.last_constraint_scale = 1.0
        for name, r in relative_mse_values.items():
            if name not in active_set:
                continue
            if name in self.ROOT_STATISTICS:
                v = float(max(r, 0.0) / self.threshold(name) - 1.0)
            else:
                v = float(math.sqrt(max(r, 0.0) + 1.0e-12) / self.threshold(name) - 1.0)
            violations[name] = v
            beta = float(self.cfg.violation_ema)
            prev_s = self.smoothed_violation[name]
            smoothed = v if prev_s is None else beta * float(prev_s) + (1.0 - beta) * v
            if prev_s is not None and float(prev_s) > 0.0 and smoothed > 0.0:
                improvement = (float(prev_s) - smoothed) / max(float(prev_s), 1e-9)
                if improvement < float(self.cfg.lambda_stall_tolerance):
                    self.stall_counts[name] += 1
                else:
                    self.stall_counts[name] = 0
                    self.frozen[name] = False
            else:
                self.stall_counts[name] = 0
                self.frozen[name] = False
            self.smoothed_violation[name] = smoothed
            if (
                self.stall_counts[name] >= int(self.cfg.lambda_stall_patience)
                and self.lambdas[name] >= float(self.cfg.lambda_stall_min)
            ):
                self.frozen[name] = True
            if self.frozen[name] or self.frozen_val[name]:
                if (float(getattr(self, "last_constraint_scale", 1.0)) >= 0.95
                        and bool(getattr(self, "val_met", {}).get(name, True))):
                    self.lambdas[name] *= float(self.cfg.lambda_stall_decay)
                self.previous_positive_violation[name] = max(0.0, v)
                continue
            update_v = max(
                -float(self.cfg.violation_soft_cap),
                min(float(self.cfg.violation_soft_cap), smoothed),
            )
            self.lambdas[name] = min(
                self.cfg.lambda_max,
                max(0.0, self.lambdas[name] + self.cfg.lambda_update_scale * self.rhos[name] * update_v),
            )
            pos = max(0.0, v)
            prev = self.previous_positive_violation[name]
            if prev is not None and pos > 0 and pos >= self.cfg.stagnation_ratio * float(prev):
                self.stagnation_counts[name] += 1
            else:
                self.stagnation_counts[name] = 0
            if self.stagnation_counts[name] >= self.cfg.stagnation_patience:
                self.rhos[name] = min(self.cfg.rho_max, self.rhos[name] * self.cfg.rho_growth)
                self.stagnation_counts[name] = 0
            self.previous_positive_violation[name] = pos
        _vg2 = violations.get("global")
        _vem2 = violations.get("early_mid")
        if _vg2 is not None and _vem2 is not None:
            if abs(float(_vg2) - float(_vem2)) <= 0.005:
                self._em_redundant_streak = int(getattr(self, "_em_redundant_streak", 0)) + 1
            else:
                self._em_redundant_streak = 0
                self._em_zeroed_logged = False
            if (self._em_redundant_streak >= 5
                    and float(self.lambdas.get("early_mid", 0.0)) > 0.0):
                if not bool(getattr(self, "_em_zeroed_logged", False)):
                    logging.getLogger(PROGRAM_NAME).info(
                        "EARLY_MID is numerically the SAME statistic as GLOBAL "
                        "on this library (|v_G - v_EM| <= 0.005 for %d epochs): "
                        "lambda(early_mid) %.1f -> 0 so one gate stops being charged "
                        "twice against the total cap. It re-ramps automatically if the "
                        "two ever diverge.",
                        int(self._em_redundant_streak),
                        float(self.lambdas["early_mid"]))
                    self._em_zeroed_logged = True
                self.lambdas["early_mid"] = 0.0
        _lg = float(self.lambdas.get("global", 0.0))
        _lem = float(self.lambdas.get("early_mid", 0.0))
        _peers = max(_lg, _lem)
        _vlate = float(violations.get("late", 0.0))
        if "late" in self.lambdas and _vlate > 0.0:
            if ((self.frozen.get("late") or self.frozen_val.get("late"))
                    and _peers > 2.0 * float(self.lambdas["late"])):
                self.frozen["late"] = False
                self.frozen_val["late"] = False
                self.stall_counts["late"] = 0
                logging.getLogger(PROGRAM_NAME).warning(
                    "LATE FREEZE RELEASED: the gate is still violated "
                    "(v=%.3f) while lambda(G/EM)=%.1f/%.1f carries >2x lambda(late)"
                    "=%.1f -- the stall was a budget transfer, not an information "
                    "limit.",
                    _vlate, _lg, _lem, float(self.lambdas["late"]))
            _ratio = float(getattr(self.cfg, "late_priority_ratio", 0.0))
            _floor = min(self.cfg.lambda_max, _ratio * _peers)
            if _ratio > 0.0 and float(self.lambdas["late"]) < _floor:
                if _floor > 1.05 * float(self.lambdas["late"]):
                    logging.getLogger(PROGRAM_NAME).info(
                        "LATE PRIORITY FLOOR | lambda(late) %.1f -> %.1f "
                        "(= %.2f x max(lambda_G, lambda_EM) while the late gate is "
                        "violated).",
                        float(self.lambdas["late"]), _floor, _ratio)
                self.lambdas["late"] = _floor
        if ("anomaly_late" in self.lambdas and "late" in self.lambdas
                and float(violations.get("anomaly_late", 0.0)) > 0.0):
            _warm = 0.5 * float(self.lambdas["late"])
            if float(self.lambdas["anomaly_late"]) < _warm:
                self.lambdas["anomaly_late"] = _warm
        if "anomaly_late" in self.lambdas and "late" in self.lambdas:
            _cap = (float(getattr(self.cfg, "anomaly_lambda_cap_ratio", 1.0))
                    * float(self.lambdas["late"]))
            if float(self.lambdas["anomaly_late"]) > _cap:
                self.lambdas["anomaly_late"] = _cap
        return violations


class ProjectLoss(nn.Module):
    """The training loss: reconstruction terms and the augmented-Lagrangian constraint terms."""
    def __init__(
        self,
        contract: ConstraintController,
        contract_cfg: ContractConfig,
        loss_cfg: LossConfig,
        late_start: int,
        gate_scale_norm: torch.Tensor,
        late_terms_all_stages: bool = True,
        ip_chargeability_max: float = 0.9,
        relaxation_blocks: int = 1,
        profile_patch_len: int = 0,
    ) -> None:
        super().__init__()
        self.profile_patch_len = int(max(0, profile_patch_len))
        self.cvar_armed: bool = False
        self.anomaly_armed: bool = False
        self.last_constraint_scale: float = 1.0
        self.relaxation_blocks = int(max(1, relaxation_blocks))
        self.contract = contract
        self.contract_cfg = contract_cfg
        self.cfg = loss_cfg
        self.late_start = late_start
        self.late_terms_all_stages = bool(late_terms_all_stages)
        self.contract_cfg_ip_eta = float(ip_chargeability_max)
        self.acceptance_scale: float = float(getattr(loss_cfg, "acceptance_scale_init", 10.0))
        self.last_row_late_mean: float = float("inf")
        self.progress = 0.0
        self.register_buffer(
            "gate_scale_norm",
            gate_scale_norm.detach().clone().float().reshape(1, 1, -1),
            persistent=False,
        )

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        pred = outputs["denoised"]
        target = batch["clean"]
        q_target = batch["q_target"]
        eps = self.cfg.eps
        em_mask, late_mask = split_masks(pred.shape[-1], self.late_start, pred.device, pred.dtype)
        l_blend_oracle = pred.new_zeros(())
        l_fused_band_z = pred.new_zeros(())
        gs = self.gate_scale_norm

        with amp_autocast_context(pred.device, enabled=False):
            pred32 = pred.float()
            target32 = target.float()
            em32 = em_mask.float()
            late32 = late_mask.float()
            _isf = batch.get("is_forward")
            _isf_fwd = _isf
            _cex = batch.get("contract_exempt")
            if _cex is not None:
                _cex = _cex.float().reshape(-1).to(pred32.device)
                _cand = (_cex if _isf is None else torch.maximum(_isf.float().reshape(-1).to(pred32.device), _cex))
                if bool((_cand < 0.5).any()):
                    _isf = _cand
            _wlib = float(getattr(self.cfg, "library_row_weight", 1.0))
            if _isf is not None:
                _wdc = (1.0 - _isf.float().reshape(-1)).view(-1, 1, 1).to(pred32.device)
                em32c = em32 * _wdc
                late32c = late32 * _wdc
                _wd_full_c = _wdc.expand_as(pred32)
            else:
                em32c, late32c, _wd_full_c = em32, late32, None
            _wsup = None
            if _isf_fwd is not None:
                _wsup = 1.0 - _isf_fwd.float().reshape(-1).to(pred32.device)
            if _cex is not None and _wlib < 1.0:
                _wsup = (torch.ones(pred32.shape[0], device=pred32.device) if _wsup is None else _wsup) \
                    * (1.0 - (1.0 - _wlib) * _cex)
            _keep_sup = ((_wsup > 1e-6) if _wsup is not None
                            else torch.ones(pred32.shape[0], dtype=torch.bool, device=pred32.device))
            if _wsup is not None:
                _wd = _wsup.view(-1, 1, 1)
                em32 = em32 * _wd
                late32 = late32 * _wd
                _wd_full = _wd.expand_as(pred32)
            else:
                _wd_full = None
            rg = relative_mse(pred32, target32, _wd_full_c, eps)
            rem = relative_mse(pred32, target32, em32c, eps)
            rl = relative_mse(pred32, target32, late32c, eps)
            _n3 = ((pred32 - target32) ** 2 * late32).sum(dim=(-2, -1))
            _d = ((target32 ** 2) * late32).sum(dim=(-2, -1)).clamp_min(ROW_DENOM_EPS)
            def _tail(_v: torch.Tensor, _c: float) -> torch.Tensor:
                _c = max(float(_c), 1e-6)
                return torch.where(_v <= _c, _v, _c * (1.0 + torch.log(_v.clamp_min(_c) / _c)))
            _r2 = _tail(torch.sqrt(_n3 / _d + eps), 3.0)
            _w6 = (late32.amax(dim=(-2, -1)) > 0).float()
            if _w6.shape[0] != _r2.shape[0]:
                _w6 = _w6.expand(_r2.shape[0])
            l_late_ps = ((_r2 * _w6).sum() / _w6.sum().clamp_min(1.0))
            _cap = float(getattr(self.cfg, "acceptance_hinge_cap", 0.07))
            l_pass_hinge = ((_tail(torch.relu(_r2 - 0.03), _cap) * _w6).sum()
                            / _w6.sum().clamp_min(1.0))
            _m4 = _w6 > 0
            late_row_median = (_r2.detach()[_m4].median() if bool(_m4.any())
                               else torch.zeros((), device=pred.device))
            _cl = torch.zeros((), device=pred.device)
            _cn2 = torch.zeros((), device=pred.device)
            _sn4 = batch.get("snr_db")
            if _sn4 is not None:
                _m3 = (_sn4.float().view(-1) >= 75.0).float() * _w6
                _cl = (_r2.detach() * _m3).sum()
                _cn2 = _m3.sum()
            l_no_harm = torch.zeros((), device=pred.device)
            _sn3 = batch.get("snr_db")
            _nz3 = batch.get("noisy")
            if (_sn3 is not None and _nz3 is not None
                    and float(getattr(self.cfg, "alpha_no_harm", 1.0)) > 0.0):
                _nzf = _nz3.float()
                if _nzf.dim() == 2:
                    _nzf = _nzf.unsqueeze(1)
                _nN = ((_nzf - target32) ** 2 * late32).sum(dim=(-2, -1))
                _rN = torch.sqrt(_nN / _d + eps).clamp(max=3.0).detach()
                _m2 = ((_sn3.float().view(-1)
                          >= float(getattr(self.cfg, "no_harm_snr_db", 25.0))).float()
                         * _w6)
                l_no_harm = ((_tail(torch.relu(_r2 - _rN), _cap) * _m2).sum()
                             / _m2.sum().clamp_min(1.0))
            l_quiet_var = torch.zeros((), device=pred.device)
            _Pq = int(self.profile_patch_len)
            _haq = batch.get("has_anomaly")
            if (_Pq > 2 and pred32.shape[0] % _Pq == 0 and _haq is not None
                    and float(getattr(self.cfg, "alpha_quiet_var", 0.5)) > 0.0):
                _nq = pred32.shape[0] // _Pq
                _eq = ((pred32 - target32) * late32).reshape(_nq, _Pq, -1)
                _sq = torch.sqrt(_d).reshape(_nq, _Pq, 1)
                _eq = _eq / _sq
                _qm = (1.0 - _haq.float().view(-1)).reshape(_nq, _Pq, 1)
                _cn = _qm.sum(dim=1, keepdim=True).clamp_min(1.0)
                _cm2 = (_eq * _qm).sum(dim=1, keepdim=True) / _cn
                _ev3 = (_eq - _cm2) * _qm
                _rv = torch.sqrt((_ev3 ** 2).sum(dim=-1) + eps)
                _ok = ((_qm[..., 0] > 0.5)
                          & (_cn[..., 0].expand(-1, _Pq) >= 3.0))
                if bool(_ok.any()):
                    l_quiet_var = _tail(torch.relu(
                        _rv[_ok]
                        - float(getattr(self.cfg, "quiet_var_target", 0.01))), _cap).mean()
            late_sample = per_sample_relative_rmse(pred32, target32, late_mask.float(), eps)
            em_sample = per_sample_relative_rmse(pred32, target32, em_mask.float(), eps)
            if _wd_full_c is not None:
                late_sample = late_sample.reshape(-1)[_isf.float().reshape(-1) < 0.5]
            l_nrmse_early = torch.sqrt(rem + eps)
            l_nrmse_late = torch.sqrt(rl + eps)
            l_nrmse_global = torch.sqrt(rg + eps)
            _ones_mask = torch.ones_like(late_mask.float()).reshape(1, 1, -1)
            _em_mask = (1.0 - late_mask.float()).reshape(1, 1, -1)
            _rw4 = (_wd.reshape(-1) if _wd_full is not None else torch.ones(pred32.shape[0], device=pred32.device))
            _rwn = _rw4.sum().clamp_min(1.0)
            _capG = float(getattr(self.cfg, "library_row_cap_global", 0.0) or 0.0)
            _capL = float(getattr(self.cfg, "library_row_cap_late", 0.0) or 0.0)

            def _lib(_r: torch.Tensor, _c: float) -> torch.Tensor:
                if _cex is None or _c <= 0.0:
                    return _r
                return torch.where(_cex > 0.5, _tail(_r, _c), _r)

            l_ps_global = (_lib(per_sample_relative_rmse(
                pred32, target32, _ones_mask, eps).reshape(-1), _capG).pow(2) * _rw4).sum() / _rwn
            l_ps_early = (_lib(per_sample_relative_rmse(
                pred32, target32, _em_mask, eps).reshape(-1), _capG).pow(2) * _rw4).sum() / _rwn
            _ls3 = per_sample_relative_rmse(pred32, target32, late_mask.float(), eps).reshape(-1)
            l_ps_late = (_lib(_ls3, _capL).pow(2) * _rw4).sum() / _rwn
            _gw = _wd_full if _wd_full is not None else torch.ones_like(target32)
            _num = ((pred32 - target32).pow(2) * _gw).sum(dim=(0, 1))
            _den = (target32.pow(2) * _gw).sum(dim=(0, 1))
            _cnt = _gw.sum(dim=(0, 1))
            _valid = (_cnt > 0) & (_den > 0)
            _pg_ratio = (torch.sqrt((_num[_valid] / (_den[_valid] + eps)) + eps)
                         if bool(_valid.any())
                         else torch.zeros(1, device=pred32.device))
            l_gate_balanced = _pg_ratio.mean()
            l_gate_worst = _pg_ratio.topk(min(4, _pg_ratio.numel())).values.mean()
            _k = float(getattr(self.cfg, "gate_eq_row_kappa", 0.03))
            _t3 = max(float(getattr(self.cfg, "gate_eq_row_tau", 0.03)), 1e-6)
            _gs = gs.reshape(1, 1, -1).to(device=pred32.device, dtype=pred32.dtype).clamp_min(1e-30)
            _q = (pred32 - target32) / torch.maximum(target32.abs(), _k * _gs)
            _r3 = torch.sqrt((_q.pow(2) * _gw).sum(dim=(-2, -1)) / _gw.sum(dim=(-2, -1)).clamp_min(1.0) + eps)
            _w7 = (_gw.amax(dim=(-2, -1)) > 0).float()
            l_gate_eq_row = ((_tail(torch.relu(_r3 / _t3 - 1.0),
                                       float(getattr(self.cfg, "gate_eq_row_cap", 0.12)) / _t3) * _w7).sum()
                             / _w7.sum().clamp_min(1.0))
            _noisy32 = batch["noisy"].float()[:, 0, :] if batch["noisy"].dim() == 3 \
                else batch["noisy"].float()
            _p2 = pred32[:, 0, :] if pred32.dim() == 3 else pred32
            _t2 = target32[:, 0, :] if target32.dim() == 3 else target32
            _gw2 = _gw[:, 0, :] if _gw.dim() == 3 else _gw
            _gs2 = self.gate_scale_norm.reshape(1, -1).to(_t2.device).clamp_min(1e-30)
            _rel_den = _gs2
            _e_out = (_p2 - _t2).abs() / _rel_den
            _e_in = (_noisy32 - _t2).abs() / _rel_den
            _harm = torch.tanh(torch.relu(_e_out - _e_in)) * _gw2
            l_do_no_harm = _harm.sum() / _gw2.sum().clamp_min(1.0)
            _nz = _noisy32 - _t2
            _rs = _p2 - _t2
            _em2 = em_mask.float()
            _em2 = _em2[:, 0, :] if _em2.dim() == 3 else _em2
            _gsn = gs.reshape(1, -1).to(_p2.dtype) if gs is not None else 1.0
            _nzm, _rsm = (_nz / _gsn) * _em2, (_rs / _gsn) * _em2
            _num_o = (_rsm * _nzm).sum(dim=1)
            _den_o = (_rsm.pow(2).sum(dim=1).sqrt()
                      * _nzm.pow(2).sum(dim=1).sqrt() + eps)
            _cos = _num_o / _den_o
            if _wd_full is not None:
                _keep = _keep_sup
                _cos = _cos.reshape(-1)[_keep] if bool(_keep.any()) else _cos.reshape(-1)
            l_noise_orthogonal = _cos.pow(2).mean()
            _alt = batch.get("noisy_alt")
            _hasalt = outputs.get("has_alt_effective", batch.get("has_alt"))
            if _alt is not None and "denoised_alt" in outputs:
                _pa = outputs["denoised_alt"].float()
                _pa = _pa[:, 0, :] if _pa.dim() == 3 else _pa
                _gsi = gs.reshape(1, -1).to(_p2.dtype) if gs is not None else 1.0
                _dif = ((_p2 - _pa) / _gsi).pow(2).mean(dim=1)
                if _hasalt is not None:
                    _mk = _hasalt.float().reshape(-1)
                    l_real_noise_inv = ((_dif * _mk).sum()
                                        / _mk.sum().clamp_min(1.0) + eps).sqrt()
                else:
                    l_real_noise_inv = (_dif.mean() + eps).sqrt()
            else:
                l_real_noise_inv = torch.zeros((), device=pred32.device)
            _lp = _p2.abs().clamp_min(1.0e-8).log()
            _lt = _t2.abs().clamp_min(1.0e-8).log()
            _rp = (_lp[:, 2:] - 2.0 * _lp[:, 1:-1] + _lp[:, :-2]).abs()
            _rt = (_lt[:, 2:] - 2.0 * _lt[:, 1:-1] + _lt[:, :-2]).abs()
            _rw = _gw2[:, 2:] if _gw2.shape[-1] == _p2.shape[-1] else torch.ones_like(_rp)
            l_roughness_cap = ((torch.relu(_rp - _rt) * _rw).sum()
                               / _rw.sum().clamp_min(1.0))
            with torch.no_grad():
                _gb_dom = {}
                if _isf_fwd is not None:
                    _m = _isf_fwd.float().reshape(-1, 1, 1).to(pred32.device)
                    for _nm, _sel in (("field", 1.0 - _m), ("forward", _m)):
                        _n2 = ((pred32 - target32).pow(2) * _sel).sum(dim=(0, 1))
                        _d2 = (target32.pow(2) * _sel).sum(dim=(0, 1))
                        _v2 = _d2 > 0
                        _gb_dom[_nm] = (torch.sqrt((_n2[_v2] / (_d2[_v2] + eps)) + eps).mean()
                                        if bool(_v2.any()) else
                                        torch.full((), float("nan"), device=pred32.device))
                    _gb_dom["forward_row_fraction"] = _m.mean()
        active_late = self.late_terms_all_stages or stage in {"B", "C"}
        _lsum = sum(max(float(v), 0.0) for v in self.contract.lambdas.values())
        _lcap = float(getattr(self.contract_cfg, "lambda_total_cap", 0.0))
        _cscale = 1.0 if (_lcap <= 0.0 or _lsum <= _lcap) else _lcap / max(_lsum, 1e-9)
        if self.training:
            _prev = float(getattr(self, "_cscale_ema", _cscale))
            _cscale = 0.9 * _prev + 0.1 * float(_cscale)
            self._cscale_ema = _cscale
        self.last_constraint_scale = float(_cscale)
        term_g = self.contract.term(rg, "global", active=True) * _cscale
        term_em = self.contract.term(rem, "early_mid", active=True) * _cscale
        term_l = self.contract.term(rl, "late", active=active_late) * _cscale
        _elab = batch.get("event_label")
        _z0 = torch.zeros((), device=pred.device, dtype=torch.float32)
        rl_anom, _num, _den = _z0, _z0, _z0
        if _elab is not None:
            with amp_autocast_context(pred.device, enabled=False):
                _w = _elab.float().reshape(-1, 1, 1)
                _num = (((pred32 - target32) ** 2) * late32c * _w).sum()
                _den = ((target32 ** 2) * late32c * _w).sum()
                rl_anom = _num / torch.clamp(_den, min=eps)
        _anom_on = bool(self.anomaly_armed) and active_late and _elab is not None
        term_anom = self.contract.term(rl_anom, "anomaly_late", active=_anom_on) * _cscale
        _quiet_w = float(self.cfg.alpha_event_quiet)
        if bool(getattr(self.cfg, "quiet_guard_follow", True)):
            _quiet_w *= max(1.0, float(self.contract.lambdas.get("anomaly_late", 0.0))
                            * float(_cscale) / 10.0)
        contract_global = term_g + term_em
        contract_late = term_l + term_anom

        with amp_autocast_context(pred.device, enabled=False):
            ts = outputs.get("trace_scale")
            target_phys = target.float() if ts is None else target.float() * ts.float()
            target_eq = target_phys / gs
            target_z = signed_symlog(target_eq)
            z_noisy = outputs["z_noisy"].float()
            z_hat = outputs["z_hat"].float()
            base_z = outputs["base_z"].float()
            noise_z_pred = outputs["noise_z"].float()
            noise_z_true = z_noisy - target_z
            _flw = float(getattr(self.cfg, "forward_late_recon_weight", 1.0))
            if _isf_fwd is not None and _flw < 1.0:
                _zw = torch.ones_like(target_z)
                _zw[..., self.late_start:] = (
                    1.0 - (1.0 - _flw) * _isf_fwd.float().reshape(-1, 1, 1))
            else:
                _zw = None

            def _wmean(t: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
                if w is None:
                    return torch.mean(t)
                return (t * w).sum() / w.sum().clamp_min(1.0)

            l_rec = _wmean(torch.abs(z_hat - target_z), _zw)
            l_base = _wmean(torch.abs(base_z - target_z), _zw)
            l_noise = _wmean(torch.abs(noise_z_pred - noise_z_true), _zw)
            d1_target = first_difference(target_z)
            d1_pred = first_difference(z_hat)
            l_d1 = _wmean(torch.abs(d1_pred - d1_target),
                          _zw[..., 1:] if _zw is not None else None)
            d2_target = second_difference(target_z)
            d2_pred = second_difference(z_hat)
            l_d2 = _wmean(torch.abs(d2_pred - d2_target),
                          _zw[..., 2:] if _zw is not None else None)
            spec_pred = log_spectrum_magnitude(z_hat, eps, self.cfg.spectrum_backend)
            spec_target = log_spectrum_magnitude(target_z, eps, self.cfg.spectrum_backend)
            l_spec = torch.mean(torch.abs(spec_pred - spec_target))
            l_fusion_adv = torch.zeros((), device=pred.device)
            l_noise_leak = torch.zeros((), device=pred.device)
            l_seam_floor_soft = torch.zeros((), device=pred.device)
            l_sigma_nll = torch.zeros((), device=pred.device)
            l_patch_late = torch.zeros((), device=pred.device)
            if "log_sigma_prior" in outputs and "manifold_z" in outputs and "denoise_arm_z" in outputs:
                _gl = int(outputs["completion_gate_end"].item()) if "completion_gate_end" in outputs else int(self.late_start)
                _m6 = torch.zeros(1, 1, target_z.shape[-1], device=pred.device, dtype=torch.float32)
                _m6[..., _gl:] = 1.0
                _ep2 = (outputs["manifold_z"].float() - target_z.float()).pow(2).detach()
                _em2 = (outputs["denoise_arm_z"].float() - target_z.float()).pow(2).detach()
                _lsp = outputs["log_sigma_prior"].float()
                _lsm = outputs["log_sigma_meas"].float()
                _den = _m6.sum() * float(pred.shape[0])
                l_sigma_nll = (0.5 * ((_ep2 * torch.exp(-_lsp) + _lsp) * _m6).sum() / _den
                                   + 0.5 * ((_em2 * torch.exp(-_lsm) + _lsm) * _m6).sum() / _den)
            _P2 = int(getattr(self, "profile_patch_len", 0) or 0)
            if _P2 > 1 and pred.shape[0] % _P2 == 0 and pred.shape[0] >= _P2:
                _e2p = (pred.float() - target.float()).pow(2)[..., self.late_start:].sum(dim=(1, 2)).reshape(-1, _P2).sum(dim=1)
                _y2p = target.float().pow(2)[..., self.late_start:].sum(dim=(1, 2)).reshape(-1, _P2).sum(dim=1)
                _pw2 = _rw4.reshape(-1, _P2).amax(dim=1)
                l_patch_late = (torch.sqrt(_e2p / (_y2p + 1e-12)) * _pw2).sum() / _pw2.sum().clamp_min(1.0)
            if "manifold_z" in outputs:
                mz = outputs.get("manifold_prior_z", outputs["manifold_z"]).float()
                _mt = target_z
                _bg_m = batch.get("clean_bg")
                _has_m = batch.get("has_anomaly")
                if (bool(getattr(self.cfg,
                                 "manifold_prior_target_background", False))
                        and _bg_m is not None and _has_m is not None):
                    _wm = _has_m.float().reshape(-1, 1, 1)
                    _mt = (1.0 - _wm) * target_z + _wm * signed_symlog(
                        _bg_m.float() / gs)
                l_prior_direct = ((mz - _mt) ** 2).mean(
                    dim=tuple(range(1, mz.ndim))).mean()
                m_abs = _huber_z(mz - _mt, float(getattr(self.cfg, "z_band_huber_delta", 0.02)))
                _gm2 = int(getattr(self, "manifold_late_start", -1))
                if 0 < _gm2 < int(self.late_start):
                    _lm = split_masks(pred.shape[-1], _gm2, pred.device, pred.dtype)[1]
                    _em32m = em32 * (1.0 - _lm)
                    _lt32m = late32 + em32 * _lm
                else:
                    _em32m, _lt32m = em32, late32
                def _bmean(_v: torch.Tensor, _w: torch.Tensor) -> torch.Tensor:
                    _wb = torch.broadcast_to(_w.to(dtype=_v.dtype), _v.shape)
                    return (_v * _wb).sum() / _wb.sum().clamp_min(1.0)
                _g2 = float(getattr(self.cfg, "z_band_gain", 1.0))
                l_manifold_early = _g2 * _bmean(m_abs, _em32m)
                l_manifold_late = _g2 * _bmean(m_abs, _lt32m)
                try:
                    _a = float(getattr(self.cfg, "alpha_blend_oracle", 0.0))
                    _zd = outputs.get("z_denoise")
                    _bl = outputs.get("manifold_blend_head", outputs.get("manifold_blend"))
                    if _a > 0.0 and _zd is not None and _bl is not None:
                        _mzf = outputs["manifold_z"].float()
                        _ed = (_zd.float() - target_z).detach()
                        _ep = (_mzf - target_z).detach()
                        _dd2 = _ed - _ep
                        _idm = (_dd2.abs() > float(getattr(self.cfg, "blend_oracle_min_gap_z", 1e-3))).float()
                        _msk = (_lt32m.expand_as(_dd2) if _lt32m.dim() == 3 else _lt32m) * _idm
                        _aid = batch.get("arm_id")
                        if _aid is not None and not bool(getattr(self.cfg, "blend_oracle_all_arms", True)):
                            _msk = _msk * (_aid.float().reshape(-1, 1, 1) < 0.5).float()
                        _fz = (1.0 - _bl.float()) * _ed + _bl.float() * _ep
                        l_blend_oracle = (torch.sum(_fz * _fz * _msk)
                                          / (torch.sum(_dd2 * _dd2 * _msk) + eps))
                except Exception as _e678a:
                    if bool(getattr(self.cfg, "strict_loss_terms", True)):
                        raise RuntimeError("l_blend_oracle failed: %r" % (_e678a,))
                    l_blend_oracle = pred.new_zeros(())
                try:
                    _fl2 = outputs.get("seam_blend_floor")
                    _hd = outputs.get("manifold_blend_head")
                    if (_fl2 is not None and _hd is not None
                            and float(getattr(self.cfg, "alpha_seam_floor_soft", 0.0)) > 0.0):
                        _fl2 = _fl2.float().reshape(-1, _fl2.shape[-1])[:1].view(1, 1, -1)
                        _gap = torch.relu(_fl2 - _hd.float()) * (_fl2 > 0).float()
                        _wf = (_wd if _wd_full is not None else torch.ones(1, 1, 1, device=pred.device))
                        l_seam_floor_soft = (_gap * _wf).sum() / (torch.broadcast_to((_fl2.float() > 0).float() * _wf, _gap.shape).sum() + eps)
                except Exception as _e:
                    if bool(getattr(self.cfg, "strict_loss_terms", True)):
                        raise RuntimeError("l_seam_floor_soft failed: %r" % (_e,))
                try:
                    _a3 = float(getattr(self.cfg, "alpha_fused_band_z", 0.0))
                    _zf = outputs.get("z_hat_pre_lateral")
                    if _a3 > 0.0 and _zf is not None:
                        _m5 = _lt32m.expand_as(_zf) if _lt32m.dim() == 3 else _lt32m
                        l_fused_band_z = (torch.sum(((_zf.float() - target_z) ** 2) * _m5)
                                          / (torch.sum(_m5) + eps))
                except Exception as _e678b:
                    if bool(getattr(self.cfg, "strict_loss_terms", True)):
                        raise RuntimeError("l_fused_band_z failed: %r" % (_e678b,))
                    l_fused_band_z = pred.new_zeros(())
                l_manifold_late_rel = _bmean(torch.tanh(m_abs), late32)
                l_denoise_late = torch.zeros((), device=pred.device)
                l_denoise_late_acc = torch.zeros((), device=pred.device)
                l_denoise_seam = torch.zeros((), device=pred.device)
                l_denoise_early = torch.zeros((), device=pred.device)
                l_denoise_contract = torch.zeros((), device=pred.device)
                l_denoise_ps_late = torch.zeros((), device=pred.device)
                l_lib_coef = torch.zeros((), device=pred.device)
                l_lib_late = torch.zeros((), device=pred.device)
                l_lib_coef_cont = torch.zeros((), device=pred.device)
                l_ridge_late = torch.zeros((), device=pred.device)
                l_order = torch.zeros((), device=pred.device)
                l_gate_deep = torch.zeros((), device=pred.device)
                l_lib_late_ps = torch.zeros((), device=pred.device)
                l_ridge_late_ps = torch.zeros((), device=pred.device)
                l_recovery = torch.zeros((), device=pred.device)
                l_diff_orth = torch.zeros((), device=pred.device)
                if "z_denoise" in outputs:
                    _dz_abs = torch.abs(outputs["z_denoise"].float() - target_z)
                    _dz = _huber_z(outputs["z_denoise"].float() - target_z, float(getattr(self.cfg, "z_band_huber_delta", 0.02)))
                    _dz_leash = F.relu(_dz_abs - float(self.cfg.denoise_late_leash_z))
                    l_denoise_late = _g2 * _bmean(_dz_leash, late32)
                    l_denoise_late_acc = _g2 * _bmean(_dz, late32)
                    _b = int(getattr(self, "manifold_late_start", -1))
                    if 0 < _b < int(self.late_start):
                        _sm = split_masks(pred.shape[-1], _b, pred.device, pred.dtype)[1] * em_mask
                    else:
                        _sm = torch.zeros_like(em_mask)
                    _ea = em_mask - _sm
                    _rw2 = (_wd if _wd_full is not None
                              else torch.ones(1, 1, 1, device=pred.device, dtype=pred.dtype))
                    l_denoise_seam = _g2 * _bmean(_dz, _sm * _rw2)
                    l_denoise_early = _g2 * _bmean(_dz, _ea * _rw2)
                man_pred32 = (signed_symexp(mz) * gs).float()
                if ts is not None:
                    man_pred32 = man_pred32 / ts.float()
                _mt32 = (signed_symexp(_mt) * gs).float()
                if ts is not None:
                    _mt32 = _mt32 / ts.float()
                rl_man = relative_mse(man_pred32, _mt32, late32c, eps)
                l_manifold_nrmse_late = torch.sqrt(rl_man + eps)
                if "z_denoise" in outputs:
                    _den32_523 = (signed_symexp(outputs["z_denoise"].float()) * gs).float()
                    if ts is not None:
                        _den32_523 = _den32_523 / ts.float()
                    l_denoise_contract = torch.sqrt(
                        relative_mse(_den32_523, _mt32, late32c, eps) + eps)
                    l_denoise_ps_late = (torch.clamp(
                        per_sample_relative_rmse(_den32_523, _mt32,
                                                 late_mask.float(), eps),
                        max=3.0).reshape(-1) * _rw4).sum() / _rwn
                _anch_z = target_z
                _bg = batch.get("clean_bg")
                _ha = batch.get("has_anomaly")
                if _bg is not None and _ha is not None:
                    with torch.no_grad():
                        _w4 = _ha.float().view(-1, 1, 1)
                        _bgp = (_bg.float() if ts is None
                                   else _bg.float() * ts.float())
                        _anch_z = signed_symlog(
                            ((1.0 - _w4) * target_phys + _w4 * _bgp) / gs)
                _lco = outputs.get("library_coef")
                if (_lco is not None and "library_basis" in outputs
                        and "library_mu" in outputs):
                    _lbz = outputs["library_bg_z"]
                    _bas = outputs["library_basis"].float()
                    _mu = outputs["library_mu"].float()
                    _cstar = torch.einsum(
                        "bg,rg->br",
                        (_anch_z.float() - _mu).squeeze(1), _bas).detach()
                    _ce = (_lco.float() - _cstar).pow(2).mean(dim=-1)
                    _sn2 = batch.get("snr_db")
                    if _sn2 is not None:
                        _wc = 1.0 + ((-10.0 - _sn2.float().view(-1))
                                        / 15.0).clamp(0.0, 1.0)
                        l_lib_coef = ((_ce * _wc).sum()
                                      / _wc.sum().clamp_min(1.0))
                    else:
                        l_lib_coef = _ce.mean()
                    _bg32_532 = (signed_symexp(_lbz.float()) * gs).float()
                    _amt = (signed_symexp(_anch_z.float()) * gs).float()
                    if ts is not None:
                        _bg32_532 = _bg32_532 / ts.float()
                        _amt = _amt / ts.float()
                    l_lib_late = torch.sqrt(
                        relative_mse(_bg32_532, _amt, late32c, eps) + eps)
                    _nA = ((_bg32_532 - _amt) ** 2 * late32).sum(dim=(-2, -1))
                    _dA = ((_amt ** 2) * late32).sum(dim=(-2, -1)).clamp_min(eps)
                    _rA = torch.sqrt(_nA / _dA.clamp_min(ROW_DENOM_EPS) + eps).clamp(max=3.0)
                    _wA = (late32.amax(dim=(-2, -1)) > 0).float()
                    if _wA.shape[0] != _rA.shape[0]:
                        _wA = _wA.expand(_rA.shape[0])
                    l_lib_late_ps = ((_rA * _wA).sum()
                                     / _wA.sum().clamp_min(1.0))
                    _lcc = outputs.get("library_coef_cont")
                    if _lcc is not None:
                        l_lib_coef_cont = F.mse_loss(
                            _lcc.float(),
                            _cstar[: int(_lcc.shape[0])])
                _rbg = outputs.get("ridge_bg")
                if _rbg is not None:
                    _rb32 = _rbg.float().clone()
                    _amtR = _amt if "_amt" in dir() else None
                    if _amtR is None:
                        _amtR = _mt32
                    l_ridge_late = torch.sqrt(
                        relative_mse(_rb32, _amtR, late32c, eps) + eps)
                    _nR = ((_rb32 - _amtR) ** 2 * late32).sum(dim=(-2, -1))
                    _dR = ((_amtR ** 2) * late32).sum(dim=(-2, -1)).clamp_min(eps)
                    _rR = torch.sqrt(_nR / _dR.clamp_min(ROW_DENOM_EPS) + eps).clamp(max=3.0)
                    _wR = (late32.amax(dim=(-2, -1)) > 0).float()
                    if _wR.shape[0] != _rR.shape[0]:
                        _wR = _wR.expand(_rR.shape[0])
                    l_ridge_late_ps = ((_rR * _wR).sum()
                                       / _wR.sum().clamp_min(1.0))
                with amp_autocast_context(pred.device, enabled=False):
                    _po = pred.float().clone().abs()
                    _tg = target.float().abs()
                    _ev2 = batch.get("event_mask")
                    _ls2 = self.late_start
                    _a1 = _po[..., _ls2:-1]
                    _a2 = _po[..., _ls2 + 1:]
                    _sc = _tg[..., _ls2:-1].clamp_min(
                        1e-6 * _tg.amax(dim=-1, keepdim=True))
                    _v3 = torch.relu(_a2 - _a1) / _sc
                    if _ev2 is not None:
                        _em = _ev2.float()
                        if _em.dim() == 2:
                            _em = _em.unsqueeze(1)
                        _wb2 = ((1.0 - _em[..., _ls2:-1])
                                  * (1.0 - _em[..., _ls2 + 1:]))
                        l_order = ((_v3 * _wb2).sum()
                                   / (_wb2.sum() + eps))
                    else:
                        l_order = _v3.mean()
                    l_order = torch.clamp(l_order, max=3.0)
                if _bg is not None and _ha is not None:
                    with amp_autocast_context(pred.device, enabled=False):
                        _selA8 = _ha.float().view(-1) > 0.5
                        if bool(_selA8.any()):
                            _lm8 = late32 if late32.shape[0] == pred.shape[0] \
                                else late32.expand(pred.shape[0], -1, -1)
                            _dt8 = ((target.float() - _bgp) * _lm8)[_selA8]
                            _dh8 = ((pred.float() - _bgp) * _lm8)[_selA8]
                            _den8 = (_dt8 * _dt8).sum(dim=(-2, -1)).clamp_min(eps)
                            _rec8 = ((_dh8 * _dt8).sum(dim=(-2, -1))
                                     / _den8).clamp(-2.0, 3.0)
                            _bl8 = ((_bgp * _lm8) ** 2)[_selA8].sum(
                                dim=(-2, -1)).clamp_min(eps)
                            _ct8 = torch.sqrt(_den8 / _bl8)
                            _wt8 = 1.0 + 2.0 * (_ct8 < 0.05).float()
                            l_recovery = (( _wt8 * (_rec8 - 1.0) ** 2).sum()
                                          / _wt8.sum().clamp_min(1.0))
                            _res8 = _dh8 - _rec8.view(-1, 1, 1) * _dt8
                            l_diff_orth = (((_res8 ** 2).sum(dim=(-2, -1))
                                            / _den8).clamp(max=9.0)).mean()
                _sn = batch.get("snr_db")
                _ha2 = batch.get("has_anomaly")
                _lg2 = outputs.get("library_gate")
                _rg = outputs.get("ridge_gate")
                if (_sn is not None
                        and (_lg2 is not None or _rg is not None)):
                    with amp_autocast_context(pred.device, enabled=False):
                        _thr = float(getattr(self.cfg,
                                                "deep_gate_snr_db", -10.0))
                        _w5 = ((_thr - _sn.float().view(-1)) / 15.0
                                 ).clamp(0.0, 1.0)
                        if _ha2 is not None:
                            _w5 = _w5 * (_ha2.float().view(-1) < 0.5).float()
                        _nbg = (_w5 > 0).float().sum().clamp_min(1.0)
                        if float(_w5.sum()) > 0.0:
                            _terms = []
                            for _g in (_lg2, _rg):
                                if _g is not None:
                                    _gm = _g.float()[..., self.late_start:]
                                    _gm = _gm.mean(dim=(-2, -1))
                                    _terms.append(
                                        ((1.0 - _gm) * _w5).sum()
                                        / _nbg)
                            if _terms:
                                l_gate_deep = torch.stack(
                                    _terms).mean()
                floor = float(self.cfg.ps_late_energy_floor)
                if floor > 0.0:
                    lt32 = late_mask.float().reshape(1, 1, -1)
                    num = torch.sum((man_pred32 - _mt32) ** 2 * lt32, dim=(1, 2))
                    den = torch.sum(_mt32 ** 2 * lt32, dim=(1, 2)) + floor * torch.sum(
                        _mt32 ** 2, dim=(1, 2)
                    )
                    l_manifold_ps_late = torch.sqrt(num / (den + eps)).mean()
                else:
                    _ps = per_sample_relative_rmse(
                        man_pred32, _mt32, late_mask.float(), eps)
                    if _wd_full is not None:
                        _ps = _ps.reshape(-1)[_keep_sup]
                    l_manifold_ps_late = _ps.mean()
                err_fused = per_sample_relative_rmse(pred32, target32, late_mask.float(), eps).clamp(max=2.0)
                err_prior = per_sample_relative_rmse(man_pred32, target32, late_mask.float(), eps).clamp(max=2.0)
                if "z_denoise" in outputs:
                    den32 = (signed_symexp(outputs["z_denoise"].float()) * gs).float()
                    if ts is not None:
                        den32 = den32 / ts.float()
                    err_den = per_sample_relative_rmse(den32, target32, late_mask.float(), eps).clamp(max=2.0)
                if _wd_full is not None:
                    _fm = _keep_sup
                    err_fused = err_fused.reshape(-1)[_fm]
                    err_prior = err_prior.reshape(-1)[_fm]
                    err_den = err_den.reshape(-1)[_fm]
                else:
                    err_den = err_prior
                best_branch = torch.minimum(err_prior, err_den).detach()
                l_fusion_adv = F.relu(
                    err_fused - best_branch - float(self.cfg.fusion_advantage_margin)
                ).mean()
                noisy32 = batch["noisy"].float()
                _lg = late_mask.float()
                noise32 = (noisy32 - target32) * _lg
                res32 = (pred32 - target32) * _lg
                num = (res32 * noise32).sum(dim=(1, 2))
                den = res32.pow(2).sum(dim=(1, 2)).sqrt() * noise32.pow(2).sum(dim=(1, 2)).sqrt() + eps
                _nl = (num / den).pow(2)
                if _wd_full is not None:
                    _nl = _nl.reshape(-1)[_keep_sup]
                l_noise_leak = _nl.mean()
                if "prior_log_sigma_sp" in outputs and "manifold_prior_z" in outputs:
                    pz = outputs["manifold_prior_z"].float()
                    perr_rms = torch.sqrt(
                        torch.mean((pz - target_z).pow(2), dim=(0, 1)) + 1e-12
                    ).detach().clamp(1e-4, 1e2)
                    sp_pred = outputs["prior_log_sigma_sp"].float().reshape(-1)
                    l_prior_cal = torch.mean(
                        torch.abs(torch.log(sp_pred.clamp_min(1e-6))
                                  - torch.log(perr_rms))
                    )
                else:
                    l_prior_cal = torch.zeros((), device=pred.device)
                if "manifold_residual" in outputs:
                    l_manifold_res = outputs["manifold_residual"].float().abs().mean()
                else:
                    l_manifold_res = torch.zeros((), device=pred.device)
                if "manifold_atom_w" in outputs:
                    lw = torch.log(outputs["manifold_atom_w"].float() + 1e-3)
                    _nb = max(1, int(getattr(self, "relaxation_blocks", 1)))
                    if _nb > 1 and lw.shape[1] % _nb == 0:
                        lwb = lw.reshape(lw.shape[0], _nb, -1)
                        d2 = lwb[:, :, 2:] - 2.0 * lwb[:, :, 1:-1] + lwb[:, :, :-2]
                    else:
                        d2 = lw[:, 2:] - 2.0 * lw[:, 1:-1] + lw[:, :-2]
                    l_spectrum_smooth = d2.pow(2).mean()
                else:
                    l_spectrum_smooth = torch.zeros((), device=pred.device)
                if "ip_atom_w" in outputs and "ip_gate" in outputs:
                    a_plus = outputs["manifold_atom_w"].float().sum(dim=1)
                    a_minus = outputs["ip_atom_w"].float().sum(dim=1)
                    g_ip = outputs["ip_gate"].float().squeeze(-1)
                    eta = float(self.contract_cfg_ip_eta)
                    l_ip_charge = F.relu(
                        (a_minus * g_ip) / (a_plus + eps) - eta
                    ).mean()
                    phys_lin = outputs["phys_linear"].float()
                    ref = torch.abs(phys_lin[:, :1]).detach() + eps
                    l_ip_pos = (
                        F.relu(-phys_lin[:, : self.late_start] / ref).pow(2).mean()
                    )
                    lwi = torch.log(outputs["ip_atom_w"].float() + 1e-3)
                    d2i = lwi[:, 2:] - 2.0 * lwi[:, 1:-1] + lwi[:, :-2]
                    l_ip_smooth = d2i.pow(2).mean()
                    l_ip_sparse = g_ip.abs().mean()
                else:
                    l_ip_charge = torch.zeros((), device=pred.device)
                    l_ip_pos = torch.zeros((), device=pred.device)
                    l_ip_smooth = torch.zeros((), device=pred.device)
                    l_ip_sparse = torch.zeros((), device=pred.device)
            else:
                l_manifold_early = torch.zeros((), device=pred.device)
                l_manifold_late = torch.zeros((), device=pred.device)
                l_denoise_late = torch.zeros((), device=pred.device)
                l_denoise_late_acc = torch.zeros((), device=pred.device)
                l_denoise_seam = torch.zeros((), device=pred.device)
                l_prior_direct = torch.zeros((), device=pred.device)
                l_manifold_late_rel = torch.zeros((), device=pred.device)
                l_denoise_early = torch.zeros((), device=pred.device)
                if "z_denoise" in outputs:
                    _dz2 = _huber_z(outputs["z_denoise"].float() - target_z, float(getattr(self.cfg, "z_band_huber_delta", 0.02)))
                    _g3 = float(getattr(self.cfg, "z_band_gain", 1.0))
                    def _bm(_v: torch.Tensor, _w: torch.Tensor) -> torch.Tensor:
                        _wb = torch.broadcast_to(_w.to(dtype=_v.dtype), _v.shape)
                        return (_v * _wb).sum() / _wb.sum().clamp_min(1.0)
                    _b2 = int(getattr(self, "manifold_late_start", -1))
                    _sm2 = (split_masks(pred.shape[-1], _b2, pred.device, pred.dtype)[1] * em_mask
                              if 0 < _b2 < int(self.late_start) else torch.zeros_like(em_mask))
                    _rw3 = (_wd if _wd_full is not None
                              else torch.ones(1, 1, 1, device=pred.device, dtype=pred.dtype))
                    l_denoise_late_acc = _g3 * _bm(_dz2, late32)
                    l_denoise_seam = _g3 * _bm(_dz2, _sm2 * _rw3)
                    l_denoise_early = _g3 * _bm(_dz2, (em_mask - _sm2) * _rw3)
                l_denoise_contract = torch.zeros((), device=pred.device)
                l_denoise_ps_late = torch.zeros((), device=pred.device)
                l_lib_coef = torch.zeros((), device=pred.device)
                l_lib_late = torch.zeros((), device=pred.device)
                l_lib_coef_cont = torch.zeros((), device=pred.device)
                l_ridge_late = torch.zeros((), device=pred.device)
                l_order = torch.zeros((), device=pred.device)
                l_gate_deep = torch.zeros((), device=pred.device)
                l_lib_late_ps = torch.zeros((), device=pred.device)
                l_ridge_late_ps = torch.zeros((), device=pred.device)
                l_recovery = torch.zeros((), device=pred.device)
                l_diff_orth = torch.zeros((), device=pred.device)
                l_manifold_nrmse_late = torch.zeros((), device=pred.device)
                l_manifold_ps_late = torch.zeros((), device=pred.device)
                l_manifold_res = torch.zeros((), device=pred.device)
                l_prior_cal = torch.zeros((), device=pred.device)
                l_spectrum_smooth = torch.zeros((), device=pred.device)
                l_ip_charge = torch.zeros((), device=pred.device)
                l_ip_pos = torch.zeros((), device=pred.device)
                l_ip_smooth = torch.zeros((), device=pred.device)
                l_ip_sparse = torch.zeros((), device=pred.device)
                l_fusion_adv = torch.zeros((), device=pred.device)
                l_noise_leak = torch.zeros((), device=pred.device)

        q_logits = outputs["reliability_logits"]
        with amp_autocast_context(pred.device, enabled=False):
            q_logits32 = q_logits.float()
            q_target32 = q_target.float().clamp_(0.0, 1.0)
            q32 = torch.sigmoid(q_logits32)
            l_q = F.binary_cross_entropy_with_logits(q_logits32, q_target32)
            l_q = l_q + torch.mean(torch.abs(q32 - q_target32))
            refinement_z32 = outputs["refinement_z"].float()
            l_corr = torch.sum(q32 * torch.abs(refinement_z32)) / (torch.sum(q32) + eps)
            l_corr_global = torch.mean(torch.abs(refinement_z32))

        late_sample_clamped = torch.clamp(late_sample, max=self.cfg.cvar_clip)
        k = max(1, int(math.ceil(len(late_sample_clamped) * self.cfg.cvar_fraction)))
        l_cvar = torch.topk(late_sample_clamped, k=k, largest=True).values.mean()
        l_late_percurve = torch.clamp(late_sample, max=1.0).mean()
        l_em_percurve = (torch.clamp(em_sample, max=1.0).reshape(-1) * _rw4).sum() / _rwn
        if not active_late:
            l_cvar = l_cvar * 0.0
            l_late_percurve = l_late_percurve * 0.0
        _cvar_on = bool(active_late and getattr(self, "cvar_armed", False))
        term_cvar = self.contract.term(l_cvar, "late_cvar", active=_cvar_on) * _cscale

        l_unc = pred.new_zeros(()).float()
        if "log_variance" in outputs:
            with amp_autocast_context(pred.device, enabled=False):
                log_var = outputs["log_variance"].float()
                diff_z = outputs.get("z_hat_model", outputs["z_hat"]).float() - target_z
                l_unc = torch.mean(0.5 * torch.exp(-log_var) * diff_z.pow(2) + 0.5 * log_var)

        p = float(min(1.0, max(0.0, self.progress)))
        z_scale = 1.0 - (1.0 - float(TrainConfig.aux_anneal_final)) * p
        linear_scale = 0.2 + 0.8 * p
        z_shape = (
            self.cfg.alpha_reconstruction * l_rec
            + self.cfg.alpha_base_reconstruction * l_base
            + self.cfg.alpha_noise * l_noise
            + self.cfg.alpha_d1 * l_d1
            + self.cfg.alpha_d2 * l_d2
            + self.cfg.alpha_spectrum * l_spec
        )
        l_gate_bal_term = (float(self.cfg.alpha_gate_balanced) * linear_scale
                           * l_gate_balanced)
        l_gate_bal_term = (l_gate_bal_term
                           + float(self.cfg.alpha_gate_worst) * linear_scale
                           * l_gate_worst
                           + float(self.cfg.alpha_do_no_harm) * linear_scale
                           * l_do_no_harm
                           + float(self.cfg.alpha_noise_orthogonal) * linear_scale
                           * l_noise_orthogonal
                           + float(self.cfg.alpha_real_noise_invariance) * linear_scale
                           * l_real_noise_inv
                           + float(self.cfg.alpha_roughness_cap) * linear_scale
                           * l_roughness_cap)
        l_metric_early = self.cfg.alpha_metric_early * linear_scale * l_nrmse_early
        l_metric_late = self.cfg.alpha_metric_late * linear_scale * l_nrmse_late
        if not active_late:
            l_metric_late = l_metric_late * 0.0
        l_man_e = (self.cfg.alpha_manifold_early * l_manifold_early
                   + float(getattr(self.cfg, "alpha_denoise_early", 0.10)) * l_denoise_early)
        lam_l = float(self.contract.lambdas.get("late", 0.0))
        man_gain = min(max(lam_l / max(float(self.cfg.manifold_lambda_follow_ref), 1e-6), 1.0),
                       float(self.cfg.manifold_lambda_follow_max))
        l_man_l = man_gain * (self.cfg.alpha_manifold_late * l_manifold_late
                              + float(getattr(self.cfg, "alpha_manifold_late_rel", 0.8))
                              * l_manifold_late_rel
                              + self.cfg.alpha_manifold_nrmse_late * l_manifold_nrmse_late
                              + self.cfg.alpha_manifold_ps_late * l_manifold_ps_late
                              + float(self.cfg.alpha_denoise_late)
                              * float(getattr(self, "denoise_late_boost", 1.0))
                              * l_denoise_late
                              + float(getattr(self.cfg, "alpha_denoise_late_accuracy", 0.30))
                              * float(getattr(self, "denoise_late_boost", 1.0))
                              * l_denoise_late_acc
                              + float(getattr(self.cfg, "alpha_denoise_contract", 1.0))
                              * float(getattr(self, "denoise_late_boost", 1.0))
                              * l_denoise_contract
                              + float(getattr(self.cfg, "alpha_denoise_ps_late", 0.35))
                              * l_denoise_ps_late
                              + float(getattr(self.cfg, "alpha_library_coef", 1.0))
                              * l_lib_coef
                              + float(getattr(self.cfg, "alpha_library_coef_cont", 0.5))
                              * l_lib_coef_cont
                              + float(getattr(self.cfg, "alpha_ridge_late", 0.5))
                              * l_ridge_late
                              + float(getattr(self.cfg, "alpha_order", 0.2))
                              * l_order
                              + float(getattr(self.cfg, "alpha_gate_deep", 0.3))
                              * l_gate_deep
                              + float(getattr(self.cfg,
                                              "alpha_late_per_sample", 0.5))
                              * l_late_ps
                              + float(getattr(self.cfg,
                                              "alpha_anchor_late_ps", 0.5))
                              * l_ridge_late_ps
                              + float(getattr(self.cfg, "alpha_recovery", 0.3))
                              * l_recovery
                              + float(getattr(self.cfg, "alpha_diff_orth", 0.15))
                              * l_diff_orth)
        _acc_group = (float(getattr(self.cfg, "alpha_pass_hinge", 1.0)) * l_pass_hinge
                         + float(getattr(self.cfg, "alpha_no_harm", 1.0)) * l_no_harm
                         + float(getattr(self.cfg, "alpha_quiet_var", 0.5)) * l_quiet_var)
        _anchor_static = (float(getattr(self.cfg, "alpha_anchor_late_ps", 0.5)) * l_lib_late_ps
                             + float(getattr(self.cfg, "alpha_library_late", 0.5)) * l_lib_late)
        _ev_dev = outputs["denoised"].device
        l_event_gate = torch.zeros((), device=_ev_dev)
        l_event_struct = torch.zeros((), device=_ev_dev)
        l_event_quiet = torch.zeros((), device=_ev_dev)
        l_residual_d2 = torch.zeros((), device=_ev_dev)
        l_event_station = torch.zeros((), device=_ev_dev)
        if "event_prob" in outputs:
            ev_r = outputs["struct_residual"].float()
            ev_mask = batch.get("event_mask")
            ev_mask = ev_mask.to(ev_r) if ev_mask is not None else torch.zeros_like(ev_r)
            _lab0 = batch.get("event_label")
            _isf_e = batch.get("is_forward")
            if _isf_e is not None and _lab0 is not None:
                _known = (1.0 - _isf_e.float().reshape(-1)
                          * (1.0 - _lab0.float().reshape(-1))).to(ev_r.device)
            else:
                _known = torch.ones(ev_r.shape[0], device=ev_r.device)
            _kn3 = _known.view(-1, 1, 1)
            l_event_quiet = ((ev_r.abs() * (1.0 - ev_mask) * _kn3).sum()
                             / ((1.0 - ev_mask) * _kn3).sum().clamp_min(1.0))
            _pos = ev_mask.mean()
            _pw = torch.clamp((1.0 - _pos) / torch.clamp(_pos, min=1e-4),
                              min=1.0, max=float(self.cfg.event_pos_weight_max)).detach()
            _logits = outputs["event_logits"].float()
            _lab3 = (_lab0.float().reshape(-1, 1, 1).to(ev_r.device)
                     if _lab0 is not None else torch.zeros(ev_r.shape[0], 1, 1,
                                                           device=ev_r.device))
            _cellw = _kn3 * (1.0 - _lab3 * (1.0 - ev_mask))
            l_event_gate = F.binary_cross_entropy_with_logits(
                _logits, ev_mask, pos_weight=_pw, weight=_cellw.expand_as(_logits))
            _lab = _lab0
            if _lab is not None:
                _pooled = _logits[..., self.late_start // 2:].amax(dim=-1).reshape(-1)
                _lab = _lab.to(_pooled).reshape(-1)
                _pw_s = torch.clamp((1.0 - _lab.mean()) / torch.clamp(_lab.mean(), min=1e-4),
                                    min=1.0, max=float(self.cfg.event_pos_weight_max)).detach()
                l_event_station = F.binary_cross_entropy_with_logits(
                    _pooled, _lab, pos_weight=_pw_s, weight=_known)
            m_sum = ev_mask.sum()
            if float(m_sum) > 0:
                gap = target_z.float() - outputs["z_hat_pre_event"].float().detach()
                l_event_struct = ((ev_r - gap).abs() * ev_mask).sum() / m_sum
        if "manifold_residual" in outputs:
            _mr = outputs["manifold_residual"].float()
            l_residual_d2 = (_mr[..., 2:] - 2.0 * _mr[..., 1:-1] + _mr[..., :-2]).abs().mean()
        _P = int(self.profile_patch_len)
        _zdev = z_hat.device if isinstance(z_hat, torch.Tensor) else target.device
        l_prof_d1 = torch.zeros((), device=_zdev)
        l_prof_d2 = torch.zeros((), device=_zdev)
        l_prof_newext = torch.zeros((), device=_zdev)
        l_prof_mg = torch.zeros((), device=_zdev)
        if _P > 2 and z_hat.shape[0] % _P == 0 and z_hat.shape[0] >= _P:
            _n = z_hat.shape[0] // _P
            _yh = z_hat.reshape(_n, _P, -1)
            _yt = target_z.reshape(_n, _P, -1)
            d1h, d1t = _yh[:, 1:] - _yh[:, :-1], _yt[:, 1:] - _yt[:, :-1]
            l_prof_d1 = (d1h - d1t).abs().mean()
            if _P > 3:
                d2h, d2t = d1h[:, 1:] - d1h[:, :-1], d1t[:, 1:] - d1t[:, :-1]
                l_prof_d2 = (d2h - d2t).abs().mean()
                tv_h = d2h.abs().sum(dim=1)
                tv_t = d2t.abs().sum(dim=1)
                l_prof_newext = F.relu(tv_h - tv_t).mean()
            _ls = int(self.late_start)
            _mid = _ls + max((z_hat.shape[-1] - _ls) // 2, 1)
            if _mid < z_hat.shape[-1]:
                def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    a = a - a.mean(dim=1, keepdim=True)
                    b = b - b.mean(dim=1, keepdim=True)
                    return (a * b).sum(dim=1) / (
                        a.pow(2).sum(dim=1).sqrt() * b.pow(2).sum(dim=1).sqrt() + 1e-8)
                ch = _corr(_yh[:, :, _ls:_mid].mean(-1), _yh[:, :, _mid:].mean(-1))
                ct = _corr(_yt[:, :, _ls:_mid].mean(-1), _yt[:, :, _mid:].mean(-1))
                l_prof_mg = (ch - ct).abs().mean()

        aux = (
            z_scale * z_shape
            + l_gate_bal_term
            + l_metric_early
            + l_metric_late
            + l_man_e
            + l_man_l
            + self.cfg.alpha_manifold_residual * l_manifold_res
            + self.cfg.alpha_prior_sigma_calibration * l_prior_cal
            + self.cfg.alpha_spectrum_smooth * l_spectrum_smooth
            + self.cfg.alpha_fusion_advantage * l_fusion_adv
            + self.cfg.alpha_noise_leak * l_noise_leak
            + self.cfg.alpha_ip_chargeability * l_ip_charge
            + self.cfg.alpha_ip_early_positivity * l_ip_pos
            + self.cfg.alpha_ip_spectrum_smooth * l_ip_smooth
            + self.cfg.alpha_ip_sparsity * l_ip_sparse
            + self.cfg.alpha_reliability * l_q
            + self.cfg.alpha_correction * l_corr
            + self.cfg.alpha_correction_global * l_corr_global
            + self.cfg.alpha_cvar * l_cvar
            + term_cvar
            + self.cfg.alpha_uncertainty * l_unc
            + self.cfg.alpha_event_gate * l_event_gate
            + self.cfg.alpha_event_struct * l_event_struct
            + _quiet_w * l_event_quiet
            + self.cfg.alpha_event_station * l_event_station
            + self.cfg.alpha_residual_d2 * l_residual_d2
            + self.cfg.alpha_profile_d1 * l_prof_d1
            + self.cfg.alpha_profile_d2 * l_prof_d2
            + self.cfg.alpha_profile_newext * l_prof_newext
            + self.cfg.alpha_profile_multigate * l_prof_mg
        )
        l_anom_diff = pred.new_zeros(())
        l_anom_amp = pred.new_zeros(())
        l_anom_sign = pred.new_zeros(())
        l_anom_total = pred.new_zeros(())
        l_anom_patch = pred.new_zeros(())
        _abg = batch.get("clean_bg")
        _ahas = outputs.get("has_anomaly_effective", batch.get("has_anomaly"))
        _apred_bg = outputs.get("denoised_bg")
        if (_abg is not None and _ahas is not None
                and float(_ahas.sum().item()) > 0.0):
            with amp_autocast_context(pred.device, enabled=False):
                _aw = _ahas.float().reshape(-1, 1, 1)
                _alm = late_mask.float().reshape(1, 1, -1)
                if bool(getattr(self.cfg, "anomaly_union_window", True)):
                    _alm = _alm.clone()
                    _alm[..., int(self.late_start) // 2:] = 1.0
                _d_true = ((target.float() - _abg.float()) * _alm) * _aw
                _base = _apred_bg.float() if _apred_bg is not None else _abg.float()
                _d_hat = ((pred.float() - _base) * _alm) * _aw
                _an = torch.linalg.vector_norm(_d_true, dim=-1)
                _aok = (_an > eps).float()
                _nz2 = ((batch["noisy"].float() - target.float()) * _alm) * _aw
                _nn = torch.linalg.vector_norm(_nz2, dim=-1)
                _y = target.float().abs() * _alm
                _nall = (batch["noisy"].float() - target.float()) * _alm
                _rr2 = (_nall / target.float().abs().clamp_min(eps)).reshape(_nall.shape[0], -1)
                _rr2 = _rr2[torch.linalg.vector_norm(_rr2, dim=-1) > eps]
                _Pk = int(self.profile_patch_len)
                _rrp = _rr2
                if _Pk > 1 and int(pred.shape[0]) % _Pk == 0:
                    _rrp = (_nall.reshape(-1, _Pk, _nall.shape[-1]).sum(dim=1)
                               / _y.reshape(-1, _Pk, _y.shape[-1]).sum(dim=1).clamp_min(eps))
                    _rrp = _rrp[torch.linalg.vector_norm(_rrp, dim=-1) > eps]

                def _kappa(_dd: torch.Tensor, _yy: torch.Tensor, _rr: torch.Tensor) -> torch.Tensor:
                    if _rr.shape[0] < 2:
                        return torch.full(_dd.shape[:1], 1.0 / math.sqrt(max(int(_dd.shape[-1]), 1)),
                                          device=_dd.device, dtype=_dd.dtype)
                    _u = _dd / torch.linalg.vector_norm(_dd, dim=-1, keepdim=True).clamp_min(eps)
                    _num = (_u * _yy) @ _rr.t()
                    _den = (_yy * _yy) @ (_rr * _rr).t()
                    return torch.sqrt((_num * _num / _den.clamp_min(eps)).mean(dim=1)).clamp_min(1e-6)
                _z = (_an / (_nn + eps)) / _kappa(
                    _d_true.reshape(_d_true.shape[0], -1), _y.reshape(_y.shape[0], -1), _rr2).view_as(_an)
                _Pv = int(self.profile_patch_len)
                if _Pv > 1 and int(pred.shape[0]) % _Pv == 0:
                    _npv = int(pred.shape[0]) // _Pv
                    _sup = (_an.view(_npv, _Pv, 1) > eps).float()
                    _dts = (_d_true.reshape(_npv, _Pv, -1) * _sup).sum(dim=1)
                    _dtv = torch.linalg.vector_norm(_dts, dim=-1)
                    _nzv = torch.linalg.vector_norm(
                        (_nz2.reshape(_npv, _Pv, -1) * _sup).sum(dim=1), dim=-1)
                    _zpv = (_dtv / (_nzv + eps)) / _kappa(
                        _dts, (_y.reshape(_npv, _Pv, -1) * _sup).sum(dim=1), _rrp)
                    _zpv = _zpv.view(_npv, 1).expand(_npv, _Pv).reshape(-1)
                    _z = torch.maximum(_z.view(-1), _zpv).view_as(_z)
                _z0 = float(getattr(self.cfg, "anomaly_visibility_z0", 1.0))
                _zw = float(getattr(self.cfg, "anomaly_visibility_width", 0.75))
                _fl = float(getattr(self.cfg, "anomaly_visibility_floor", 0.0))
                _t = torch.clamp((_z - (_z0 - _zw)) / (2.0 * _zw), 0.0, 1.0)
                _w3 = _fl + (1.0 - _fl) * _t * _t * (3.0 - 2.0 * _t)
                _wv = _aok * _w3.squeeze(-1) if _w3.dim() > _aok.dim() else _aok * _w3
                _acnt = _aok.sum().clamp_min(1.0)
                _g4 = (_z * _z / (1.0 + _z * _z)).detach().unsqueeze(-1)
                l_anom_diff = ((torch.linalg.vector_norm(_d_hat - _g4 * _d_true, dim=-1)
                                / (_an + eps)) * _wv).sum() / _acnt
                _ar = torch.linalg.vector_norm(_d_hat, dim=-1) / (_an + eps)
                l_anom_amp = (((_ar - 1.0).abs()) * _wv).sum() / _acnt
                _acos = (_d_hat * _d_true).sum(dim=-1) / (
                    torch.linalg.vector_norm(_d_hat, dim=-1) * _an + eps)
                l_anom_sign = ((1.0 - _acos) * _wv).sum() / _acnt
            _Pp = int(self.profile_patch_len)
            if _Pp > 1 and int(pred.shape[0]) % _Pp == 0:
                with amp_autocast_context(pred.device, enabled=False):
                    _np = int(pred.shape[0]) // _Pp
                    _dt_p = _d_true.reshape(_np, _Pp, -1).sum(dim=1)
                    _dh_p = _d_hat.reshape(_np, _Pp, -1).sum(dim=1)
                    _an_p = torch.linalg.vector_norm(_dt_p, dim=-1)
                    _bg_p = torch.linalg.vector_norm(
                        (_abg.float() * _alm).reshape(_np, _Pp, -1).sum(dim=1), dim=-1)
                    _ok_p = ((_an_p > eps) & (_an_p >= 0.002 * _bg_p)).float()
                    _nn_p = torch.linalg.vector_norm(
                        _nz2.reshape(_np, _Pp, -1).sum(dim=1), dim=-1)
                    _z_p = (_an_p / (_nn_p + eps)) / _kappa(
                        _dt_p, _y.reshape(_np, _Pp, -1).sum(dim=1), _rrp)
                    _t_p = torch.clamp((_z_p - (_z0 - _zw)) / (2.0 * _zw), 0.0, 1.0)
                    _w_p = _fl + (1.0 - _fl) * _t_p * _t_p * (3.0 - 2.0 * _t_p)
                    _wv_p = _ok_p * _w_p
                    _cnt_p = _ok_p.sum().clamp_min(1.0)
                    _gp = (_z_p * _z_p / (1.0 + _z_p * _z_p)).detach().unsqueeze(-1)
                    _rel_p = torch.linalg.vector_norm(_dh_p - _gp * _dt_p, dim=-1) / (_an_p + eps)
                    _amp_p = (torch.linalg.vector_norm(_dh_p, dim=-1) / (_an_p + eps) - 1.0).abs()
                    l_anom_patch = ((_rel_p + float(self.cfg.alpha_anomaly_amp) * _amp_p) * _wv_p).sum() / _cnt_p
            l_anom_total = (float(self.cfg.alpha_anomaly_diff) * l_anom_diff
                            + float(self.cfg.alpha_anomaly_amp) * l_anom_amp
                            + float(self.cfg.alpha_anomaly_sign) * l_anom_sign
                            + float(getattr(self.cfg, "alpha_anomaly_patch", 1.0)) * l_anom_patch)
        l_cont = pred.new_zeros(())
        _dc = outputs.get("denoised_cont")
        if _dc is not None and "cont_gate_mask" in outputs:
            with amp_autocast_context(pred.device, enabled=False):
                _cm = outputs["cont_gate_mask"].float()
                _mB = int(_dc.shape[0])
                _tc = target[:_mB].float()
                _ec2_525 = ((_dc.float() - _tc).pow(2) * _cm).sum(dim=(1, 2))
                _tc2_525 = (_tc.pow(2) * _cm).sum(dim=(1, 2))
                l_cont = torch.clamp(
                    torch.sqrt(_ec2_525 / (_tc2_525 + ROW_DENOM_EPS) + eps), max=3.0).mean()
        _pm = pred32[:, 0, :] if pred32.dim() == 3 else pred32
        _pv = _pm.clamp_min(1.0e-8).log()
        _tv = (target32[:, 0, :] if target32.dim() == 3 else target32
               ).clamp_min(1.0e-8).log()
        _mono_ok = (_tv[..., 1:] <= _tv[..., :-1] + 1.0e-6).float()
        _evm = batch.get("event_mask")
        if _evm is not None:
            _ev = _evm.float()
            _ev = _ev[:, 0, :] if _ev.dim() == 3 else _ev
            _mono_ok = _mono_ok * (1.0 - torch.maximum(_ev[..., 1:], _ev[..., :-1]))
        _mono_viol = torch.relu(_pv[..., 1:] - _pv[..., :-1]) * _mono_ok
        l_gate_mono = _mono_viol.sum() / _mono_ok.sum().clamp_min(1.0)
        _w2 = None
        _pm2 = pred32[:, 0, :] if pred32.dim() == 3 else pred32
        _tm = target32[:, 0, :] if target32.dim() == 3 else target32
        _nsrc = batch.get("noise")
        if _nsrc is not None:
            _nm2 = _nsrc.float()
            _nm2 = _nm2[:, 0, :] if _nm2.dim() == 3 else _nm2
        else:
            _xm = batch["noisy"].float()
            _xm = _xm[:, 0, :] if _xm.dim() == 3 else _xm
            _nm2 = _xm - _tm
        _s = self.gate_scale_norm.reshape(1, -1).to(_pm2.device).clamp_min(1e-30)
        _e_l = (_pm2 - _tm) / _s
        _n_l = _nm2 / _s
        _t_l = _tm / _s
        _w2 = (_n_l * _n_l) / (_t_l * _t_l + _n_l * _n_l + 1e-20)
        _num_c = (_w2 * _n_l * _n_l).sum(dim=1).clamp_min(1e-12)
        _beta = ((_w2 * _e_l * _n_l).sum(dim=1) / _num_c).clamp(-2.0, 2.0)
        _snr_l = ((_w2 * _t_l * _t_l).sum(dim=1) / _num_c).clamp_min(1e-6)
        _allow = (1.5 * _snr_l / (1.0 + _snr_l)).clamp(max=1.0)
        _excess = torch.relu(_beta - _allow)
        l_noise_carryover = _excess.mean()
        _nbm = batch.get("neighbor_noise_mean")
        _nbcl = batch.get("neighbors_clean")
        _nbno = batch.get("neighbors")
        _nbar_l = None
        if _nbm is not None:
            _nbar = _nbm.float()
            _nbar_l = _nbar[:, 0, :] if _nbar.dim() == 3 else _nbar
        elif _nbcl is not None and _nbno is not None and _nbno.shape[1] > 0:
            _nbar_l = (_nbno.float() - _nbcl.float()).mean(dim=1)
        if _nbar_l is not None:
            _nbar_l = _nbar_l / _s
            _num_nb = (_w2 * _nbar_l * _nbar_l).sum(dim=1).clamp_min(1e-12)
            _beta_nb = ((_w2 * _e_l * _nbar_l).sum(dim=1)
                        / _num_nb).clamp(-2.0, 2.0)
            _rho_hat = float(getattr(self.cfg, "neighbor_carryover_rho", 0.5))
            _allow_nb = (1.5 * _rho_hat * _snr_l / (1.0 + _snr_l)).clamp(max=1.0)
            _excess_nb = torch.relu(_beta_nb - _allow_nb)
            l_neighbor_carryover = _excess_nb.mean()
            _beta_nb_mean = _beta_nb.mean().detach()
        else:
            l_neighbor_carryover = torch.zeros((), device=_e_l.device)
            _beta_nb_mean = torch.zeros((), device=_e_l.device)
        del _nbcl, _nbno
        _fpc = outputs.get("fp_curves")
        if _fpc is not None and _fpc.shape[-1] >= 3:
            l_fp_smooth = ((_fpc[:, 2:] - 2.0 * _fpc[:, 1:-1]
                            + _fpc[:, :-2]) ** 2).mean()
        else:
            l_fp_smooth = torch.zeros((), device=z_hat.device, dtype=z_hat.dtype)
        total = (contract_global + contract_late + aux + l_anom_total
                 + float(getattr(self.cfg, "alpha_blend_oracle", 0.0)) * l_blend_oracle
                 + float(getattr(self.cfg, "alpha_fused_band_z", 0.0)) * l_fused_band_z
                 + float(self.acceptance_scale) * _acc_group
                 + _anchor_static
                 + float(getattr(self.cfg, "alpha_denoise_seam", 0.30)) * l_denoise_seam
                 + float(getattr(self.cfg, "alpha_seam_floor_soft", 1.0)) * l_seam_floor_soft
                 + float(getattr(self.cfg, "alpha_continuation", 0.30)) * l_cont
                 + float(getattr(self.cfg, "alpha_late_percurve", 0.5)) * l_late_percurve
                 + float(getattr(self.cfg, "alpha_em_percurve", 0.5)) * l_em_percurve
                 + float(getattr(self.cfg, "alpha_gate_eq_row", 0.0)) * l_gate_eq_row
                 + 1e-2 * l_fp_smooth
                 + float(getattr(self.cfg, "alpha_sigma_nll", 1.0)) * l_sigma_nll
                 + float(getattr(self.cfg, "alpha_patch_late", 1.0)) * l_patch_late
                 + float(getattr(self.cfg, "alpha_prior_direct", 15.0))
                 * l_prior_direct
                 + float(getattr(self.cfg, "alpha_contract_global", 25.0))
                 * l_ps_global
                 + float(getattr(self.cfg, "alpha_contract_early_mid", 25.0))
                 * l_ps_early
                 + float(getattr(self.cfg, "alpha_contract_late", 35.0))
                 * (l_ps_late if active_late else l_ps_late.detach() * 0.0)
                 + float(self.cfg.alpha_gate_monotonic) * l_gate_mono
                 + float(getattr(self.cfg, "alpha_noise_carryover", 0.10))
                 * l_noise_carryover
                 + float(getattr(self.cfg, "alpha_neighbor_carryover", 0.30))
                 * l_neighbor_carryover)
        return {
            "total": total,
            "l_noise_carryover": l_noise_carryover,
            "carryover_beta_mean": _beta.mean().detach(),
            "carryover_beta_excess": _excess.mean().detach(),
            "l_neighbor_carryover": l_neighbor_carryover,
            "carryover_beta_nb": _beta_nb_mean,
            "l_anomaly_diff": l_anom_diff,
            "l_anomaly_amp": l_anom_amp,
            "l_anomaly_patch": l_anom_patch,
            "l_anomaly_sign": l_anom_sign,
            "l_gate_monotonic": l_gate_mono,
            "l_gate_balanced": l_gate_balanced,
            "l_gate_worst": l_gate_worst,
            "l_gate_eq_row": l_gate_eq_row,
            "l_do_no_harm": l_do_no_harm,
            "l_noise_orthogonal": l_noise_orthogonal,
            "l_real_noise_invariance": l_real_noise_inv,
            "l_roughness_cap": l_roughness_cap,
            "l_gate_balanced_field": _gb_dom.get(
                "field", torch.full((), float("nan"), device=pred32.device)),
            "l_gate_balanced_forward": _gb_dom.get(
                "forward", torch.full((), float("nan"), device=pred32.device)),
            "forward_row_fraction": _gb_dom.get(
                "forward_row_fraction", torch.zeros((), device=pred32.device)),
            "l_profile_d1": l_prof_d1,
            "l_profile_d2": l_prof_d2,
            "l_profile_newext": l_prof_newext,
            "l_profile_multigate": l_prof_mg,
            "global_part": contract_global + l_metric_early + l_man_e,
            "l_event_gate": l_event_gate,
            "l_event_struct": l_event_struct,
            "l_event_quiet": l_event_quiet,
            "l_event_station": l_event_station,
            "l_residual_d2": l_residual_d2,
            "late_part": l_anom_total + contract_late + l_metric_late + l_man_l
                         + self.cfg.alpha_cvar * l_cvar + term_cvar,
            "aux": aux,
            "rg": rg,
            "rem": rem,
            "rl": rl,
            "rl_anom": rl_anom,
            "rl_anom_num": _num,
            "rl_anom_den": _den,
            "l_rec": l_rec,
            "l_base": l_base,
            "l_noise": l_noise,
            "l_late_percurve": l_late_percurve,
            "l_em_percurve": l_em_percurve,
            "l_d1": l_d1,
            "l_d2": l_d2,
            "l_spec": l_spec,
            "l_q": l_q,
            "l_corr": l_corr,
            "l_corr_global": l_corr_global,
            "l_cvar": l_cvar,
            "l_unc": l_unc,
            "l_nrmse_early": l_nrmse_early,
            "l_nrmse_global": l_nrmse_global,
            "l_ps_global": l_ps_global,
            "l_prior_direct": l_prior_direct,
            "l_ps_early_mid": l_ps_early,
            "l_ps_late": l_ps_late,
            "l_nrmse_late": l_nrmse_late,
            "l_manifold_early": l_manifold_early,
            "l_blend_oracle": l_blend_oracle,
            "l_fused_band_z": l_fused_band_z,
            "l_manifold_late": l_manifold_late,
            "l_sigma_nll": l_sigma_nll,
            "l_patch_late": l_patch_late,
            "l_denoise_late": l_denoise_late,
            "l_denoise_late_accuracy": l_denoise_late_acc,
            "l_denoise_seam": l_denoise_seam,
            "l_seam_floor_soft": l_seam_floor_soft,
            "l_denoise_early": l_denoise_early,
            "l_denoise_contract": l_denoise_contract,
            "l_denoise_ps_late": l_denoise_ps_late,
            "l_continuation": l_cont,
            "l_library_coef": l_lib_coef,
            "l_library_late": l_lib_late,
            "l_library_coef_cont": l_lib_coef_cont,
            "l_ridge_late": l_ridge_late,
            "l_order": l_order,
            "l_gate_deep": l_gate_deep,
            "l_late_per_sample": l_late_ps,
            "l_pass_hinge": l_pass_hinge,
            "l_no_harm": l_no_harm,
            "l_quiet_var": l_quiet_var,
            "clean_arm_late_sum": _cl,
            "late_row_median": late_row_median,
            "clean_arm_rows": _cn2,
            "acceptance_group": _acc_group,
            "acceptance_scale": torch.tensor(float(self.acceptance_scale),
                                             device=l_no_harm.device),
            "l_library_late_ps": l_lib_late_ps,
            "l_ridge_late_ps": l_ridge_late_ps,
            "l_recovery": l_recovery,
            "l_diff_orth": l_diff_orth,
            "l_manifold_nrmse_late": l_manifold_nrmse_late,
            "l_manifold_late_rel": l_manifold_late_rel,
            "l_manifold_ps_late": l_manifold_ps_late,
            "l_manifold_res": l_manifold_res,
            "l_prior_sigma_cal": l_prior_cal,
            "l_spectrum_smooth": l_spectrum_smooth,
            "l_fusion_advantage": l_fusion_adv,
            "l_noise_leak": l_noise_leak,
            "l_ip_charge": l_ip_charge,
            "l_ip_pos": l_ip_pos,
            "l_ip_smooth": l_ip_smooth,
            "l_ip_sparse": l_ip_sparse,
        }
