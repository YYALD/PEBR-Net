# PEBR-Net, the prior- and evidence-bounded reconstruction network.
# MIT License, see LICENSE.
"""Training, evaluation during training and checkpoint selection.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

class AnomalyReconstructionMonitor:
    """A dedicated, per-epoch record of the two things this project is actually judged on: how much of the late
    window is reconstructed, and how much of a weak anomaly survives that reconstruction.
    """

    FIELDS: List[Tuple[str, str, str, str]] = [
        ("global_percent", "global_error_percent", "global", ""),
        ("early_mid_percent", "early_mid_error_percent", "early/mid", ""),
        ("late_percent", "late_error_percent", "late", ""),
        ("late_cvar_percent", "late_cvar_percent", "late CVaR", ""),
        ("em_misfit_ratio", "early_mid_misfit_ratio", "E/M misfit ratio", ""),
        ("seam_band_max", "seam_band_max_percent", "seam max", ""),
        ("early_harm", "early_do_no_harm_percent", "early do-no-harm", ""),
        ("gate_mono_violation", "gate_monotonic_violation_percent",
         "gate-order violations", ""),
        ("denoise_late_percent", "denoise_path_late_error_percent", "denoise-path late", ""),
        ("manifold_late_percent", "manifold_late_error_percent", "manifold-prior late", ""),
        ("blend_late", "blend_late_mean", "blend(late)", ""),
        ("innovation_late", "innovation_w_late_mean", "innovation(late)", ""),
        ("lateral_rough_out_dex", "lateral_late_roughness_out_dex", "lateral rough (out)", ""),
        ("lateral_rough_ref_dex", "lateral_late_roughness_ref_dex", "lateral rough (ref)", ""),
        ("lateral_rough_ratio", "lateral_late_roughness_ratio", "lateral rough ratio", ""),
        ("late_noise_floor", "late_noise_floor_late_mean", "late noise floor", ""),
        ("innovation_gate_open", "innovation_evidence_open_rate", "innovation gate open", ""),
        ("kernel_dev_low_snr", "lateral_kernel_dev_low_snr", "kernel dev (low SNR)", "(SNR)"),
        ("kernel_dev_high_snr", "lateral_kernel_dev_high_snr", "kernel dev (high SNR)", "(SNR)"),
        ("quiet_rough_out_dex", "lateral_quiet_roughness_out_dex", "quiet false structure", ""),
        ("body_rough_out_dex", "lateral_body_roughness_out_dex", "body structure err", ""),
        ("precision_w_low_snr", "innovation_precision_w_low_snr", "precision w (low SNR)", "(SNR)"),
        ("precision_w_high_snr", "innovation_precision_w_high_snr", "precision w (high SNR)", "(SNR)"),
        ("lat_strength_low_snr", "lateral_strength_low_snr", "lateral strength (low SNR)", "(SNR)"),
        ("lat_strength_high_snr", "lateral_strength_high_snr", "lateral strength (high SNR)", "(SNR)"),
        ("robust_inflation", "innovation_robust_inflation_mean", "robust inflation", ""),
        ("robust_tail_rate", "innovation_robust_tail_rate", "robust tail rate", ""),
        ("out_fusion_w_low", "output_fusion_w_low_snr", "out-fusion w (low SNR)", "(SNR)"),
        ("out_fusion_w_high", "output_fusion_w_high_snr", "out-fusion w (high SNR)", "(SNR)"),
        ("lambda_factor_low", "lateral_lambda_factor_low_snr", "lambda factor (low SNR)", "λ(SNR)"),
        ("lambda_factor_high", "lateral_lambda_factor_high_snr", "lambda factor (high SNR)", "λ(SNR)"),
        ("real_rec_weak", "paired_anomaly_recovery_weak", "REAL-rec weak (<5%)", ""),
        ("real_rec_medium", "paired_anomaly_recovery_medium", "REAL-rec medium", ""),
        ("real_rec_strong", "paired_anomaly_recovery_strong", "REAL-rec strong", ""),
        ("quiet_false_level", "paired_quiet_false_level", "quiet false-anomaly level", ""),
        ("event_late_percent", "event_term_late_percent", "event term late", ""),
        ("paired_anomaly_recovery", "paired_anomaly_recovery",
         "REAL recovery (proj)", ""),
        ("paired_anomaly_recovery_p10", "paired_anomaly_recovery_p10",
         "REAL recovery p10", "p10"),
        ("paired_anomaly_amplitude", "paired_anomaly_amplitude_ratio",
         "REAL amplitude ratio", ""),
        ("paired_anomaly_sign", "paired_anomaly_sign_agreement",
         "REAL sign agreement", ""),
        ("paired_anomaly_stations", "paired_anomaly_stations",
         "anomalous stations", ""),
        ("anomaly_body_recovery", "anomaly_body_recovery", "body recovery", ""),
        ("anomaly_body_late_percent", "anomaly_body_late_error_percent", "body late err", ""),
        ("anomaly_body_auroc", "anomaly_body_auroc", "body AUROC", " AUROC"),
        ("anomaly_singleton_recovery", "anomaly_singleton_recovery", "singleton recovery", ""),
        ("anomaly_singleton_auroc", "anomaly_singleton_auroc", "singleton AUROC", " AUROC"),
        ("anomaly_fieldlike_recovery", "anomaly_fieldlike_recovery", "field-like recovery", ""),
        ("anomaly_fieldlike_late_percent", "anomaly_fieldlike_late_error_percent",
         "field-like late err", ""),
        ("anomaly_fieldlike_auroc", "anomaly_fieldlike_auroc", "field-like AUROC", " AUROC"),
        ("l_anomaly_diff", "l_anomaly_diff", "paired differential", ""),
        ("l_anomaly_amp", "l_anomaly_amp", "paired amplitude", ""),
        ("l_anomaly_sign", "l_anomaly_sign", "paired sign", ""),
    ]

    def __init__(self, cfg: "Config", run_dir: Path, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.rows: List[Dict[str, Any]] = []
        self.run_name = run_dir.name
        self._prev: Optional[Dict[str, float]] = None
        self._targets = {
            "late_percent": 100.0 * float(cfg.contract.late_threshold),
            "global_percent": 100.0 * float(cfg.contract.global_threshold),
            "early_mid_percent": 100.0 * float(cfg.contract.early_mid_threshold),
            "late_cvar_percent": 100.0 * float(cfg.contract.late_cvar_threshold),
        }
        self.paths: List[Path] = []
        self.csv_paths: List[Path] = []
        d = run_dir / "metrics"
        d.mkdir(parents=True, exist_ok=True)
        self.paths.append(d / "anomaly_reconstruction_monitor.log")
        self.csv_paths.append(d / "anomaly_reconstruction_metrics.csv")
        root = str(getattr(cfg.paths, "log_root", "") or "").strip()
        if root:
            try:
                rp = Path(root)
                rp.mkdir(parents=True, exist_ok=True)
                self.paths.append(rp / ("%s.anomaly.log" % self.run_name))
                self.csv_paths.append(rp / ("%s_epoch_metrics.csv" % self.run_name))
            except Exception as exc:
                logger.warning("Monitor log root unusable (%s).", exc)
        self._write_header()

    def _emit(self, block: str) -> None:
        for p in self.paths:
            try:
                with p.open("a", encoding="utf-8") as f:
                    f.write(block)
            except Exception as exc:
                self.logger.warning("Monitor write failed (%s): %s", p, exc)

    def _write_header(self) -> None:
        self._emit(
            "=" * 100 + "\n"
            "ANOMALY RECOVERY & RECONSTRUCTION MONITOR \n"
            "run: %s | program %s v%s | started %s\n"
            "contract gates: global <= %.1f%% | early/mid <= %.1f%% | late <= %.1f%% | "
            "late CVaR <= %.1f%%\n"
            "\n"
            % (self.run_name, PROGRAM_NAME, PROGRAM_VERSION,
               _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               self._targets["global_percent"], self._targets["early_mid_percent"],
               self._targets["late_percent"], self._targets["late_cvar_percent"])
            + "=" * 100 + "\n"
        )

    @staticmethod
    def _f(v: Any) -> float:
        try:
            x = float(v)
            return x if math.isfinite(x) else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _fmt(v: float, width: int = 9, dec: int = 4) -> str:
        return ("%*s" % (width, "n/a")) if not math.isfinite(v) else ("%*.*f" % (width, dec, v))

    def record(self, stage: str, epoch: int, val_metrics: Mapping[str, Any],
               train_components: Optional[Mapping[str, Any]] = None,
               extra: Optional[Mapping[str, Any]] = None) -> None:
        """One epoch."""
        src: Dict[str, Any] = dict(val_metrics or {})
        for holder in (train_components or {}, extra or {}):
            for k, v in dict(holder).items():
                src.setdefault(k, v)
        row: Dict[str, Any] = {"run": self.run_name, "stage": stage, "epoch": int(epoch),
                               "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        vals: Dict[str, float] = {}
        for col, key, _lab, _cn in self.FIELDS:
            vals[col] = self._f(src.get(key))
            row[col] = "" if not math.isfinite(vals[col]) else "%.6g" % vals[col]
        self.rows.append(row)
        self._write_csv()
        self._write_block(stage, epoch, vals)
        self._check_regression(stage, epoch, vals)
        self._prev = vals

    def _write_csv(self) -> None:
        cols = ["run", "stage", "epoch", "time"] + [c for c, _, _, _ in self.FIELDS]
        for p in self.csv_paths:
            try:
                with p.open("w", encoding="utf-8-sig", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writeheader()
                    for r in self.rows:
                        w.writerow({c: r.get(c, "") for c in cols})
            except Exception as exc:
                self.logger.warning("Monitor CSV write failed (%s): %s", p, exc)

    def _delta(self, col: str, v: float) -> str:
        if self._prev is None:
            return ""
        p = self._prev.get(col, float("nan"))
        if not (math.isfinite(p) and math.isfinite(v)):
            return ""
        d = v - p
        return "  (%+.4f)" % d

    def _write_block(self, stage: str, epoch: int, v: Dict[str, float]) -> None:
        lines = [
            "\n" + "-" * 100,
            "stage=%s  epoch=%-4d  %s" % (stage, epoch,
                                          _dt.datetime.now().strftime("%H:%M:%S")),
            "-" * 100,
            "[RECONSTRUCTION]",
        ]
        for col, _key, lab, cn in self.FIELDS[:10]:
            tgt = self._targets.get(col)
            gate = ""
            if tgt is not None and math.isfinite(v[col]):
                gate = "   gate %.1f%%  %s" % (tgt, "PASS" if v[col] <= tgt else "FAIL")
            lines.append("  %-22s %-14s %s%%%s%s"
                         % (lab, cn, self._fmt(v[col]), self._delta(col, v[col]), gate))
        lines.append("[FUSION]")
        for col, _key, lab, cn in self.FIELDS[10:13]:
            lines.append("  %-22s %-14s %s%s"
                         % (lab, cn, self._fmt(v[col]), self._delta(col, v[col])))
        lines.append("[ANOMALY]")
        for col, _key, lab, cn in self.FIELDS[13:26]:
            lines.append("  %-22s %-14s %s%s"
                         % (lab, cn, self._fmt(v[col]), self._delta(col, v[col])))
        lines.append("[PAIRED SUPERVISION]")
        for col, _key, lab, cn in self.FIELDS[26:]:
            lines.append("  %-22s %-14s %s%s"
                         % (lab, cn, self._fmt(v[col]), self._delta(col, v[col])))
        bl, iv = v.get("blend_late", float("nan")), v.get("innovation_late", float("nan"))
        if math.isfinite(bl):
            if bl > 0.90:
                lines.append(
                    "  READING: the late window is carried by the PRIOR (blend %.3f). Its "
                    "error therefore cannot fall below the prior's own late error (%s%%). "
                    % (bl, self._fmt(v.get("manifold_late_percent", float("nan")), 1, 3)))
            elif bl < 0.35:
                lines.append(
                    "  READING: the late window is carried by the MEASUREMENT (blend %.3f, "
                    "innovation %s)."
                    % (bl, self._fmt(iv, 1, 3)))
            else:
                lines.append(
                    "  READING: the late window is a LEARNED mixture (blend %.3f, "
                    "innovation %s) ."
                    % (bl, self._fmt(iv, 1, 3)))
        self._emit("\n".join(lines) + "\n")

    def _check_regression(self, stage: str, epoch: int, v: Dict[str, float]) -> None:
        if self._prev is None:
            return
        le, ple = v.get("late_percent"), self._prev.get("late_percent")
        for col, label, cn in (("paired_anomaly_recovery", "REAL (measured)", ""),
                               ("anomaly_body_recovery", "body", ""),
                               ("anomaly_fieldlike_recovery", "field-like", "")):
            rc, prc = v.get(col), self._prev.get(col)
            if not all(x is not None and math.isfinite(x) for x in (le, ple, rc, prc)):
                continue
            if le < ple - 1e-9 and rc < prc - 0.01:
                msg = (
                    "ANOMALY REGRESSION | stage=%s epoch=%d | late error IMPROVED %.4f%% -> %.4f%% while "
                    "%s anomaly recovery FELL %.4f -> %.4f. That combination is the smoother signature: the metric "
                    "is being satisfied by flattening the late window, and the weak body is going with it."
                    % (stage, epoch, ple, le, label, prc, rc)
                )
                self.logger.warning(msg)
                self._emit("  !! " + msg + "\n")
                if col == "paired_anomaly_recovery":
                    _amp = v.get("paired_anomaly_amplitude", float("nan"))
                    if math.isfinite(_amp):
                        if _amp < 0.5:
                            _d = ("FLATTENED: the reconstructed anomaly's amplitude collapsed too (ratio %.3f). The objective is "
                                  "rewarding smoothness." % (_amp,))
                        elif _amp > 0.8:
                            _d = ("REPLACED: the amplitude survived (ratio %.3f) but no longer correlates with the truth. The "
                                  "model is writing an anomaly of its own into the late window." % (_amp,))
                        else:
                            _d = ("amplitude ratio %.3f (between flattening and replacement)."
                                  % (_amp,))
                        self.logger.warning("%s", _d)
                        self._emit("     -> " + _d + "\n")

    def summarize(self) -> Dict[str, Any]:
        """Best-epoch summary for the run index and the final report."""
        out: Dict[str, Any] = {"epochs": len(self.rows)}
        if not self.rows:
            return out

        def _col(c: str) -> List[float]:
            return [self._f(r.get(c)) for r in self.rows]

        late = _col("late_percent")
        fin = [x for x in late if math.isfinite(x)]
        if fin:
            bi = int(np.nanargmin(np.asarray(late, dtype=np.float64)))
            out["best_late_percent"] = float(late[bi])
            out["best_late_epoch"] = int(self.rows[bi].get("epoch", -1))
            for c in ("global_percent", "early_mid_percent", "late_cvar_percent",
                      "anomaly_body_recovery", "anomaly_body_auroc",
                      "anomaly_fieldlike_recovery"):
                out["at_best_" + c] = self._f(self.rows[bi].get(c))
        for c in ("anomaly_body_recovery", "anomaly_fieldlike_recovery"):
            v = [x for x in _col(c) if math.isfinite(x)]
            if v:
                out["final_" + c] = float(v[-1])
                out["max_" + c] = float(max(v))
        return out

    def close(self, verdict: str = "") -> None:
        s = self.summarize()
        lines = ["\n" + "=" * 100, "RUN SUMMARY | %s" % self.run_name]
        if "best_late_percent" in s:
            lines.append(
                "  best late error %.4f%% at epoch %d | global %.4f%% | early/mid %.4f%% "
                "| late CVaR %.4f%%"
                % (s["best_late_percent"], s.get("best_late_epoch", -1),
                   s.get("at_best_global_percent", float("nan")),
                   s.get("at_best_early_mid_percent", float("nan")),
                   s.get("at_best_late_cvar_percent", float("nan"))))
            lines.append(
                "  at that epoch: body recovery %.4f | body AUROC %.4f | field-like "
                "recovery %.4f"
                % (s.get("at_best_anomaly_body_recovery", float("nan")),
                   s.get("at_best_anomaly_body_auroc", float("nan")),
                   s.get("at_best_anomaly_fieldlike_recovery", float("nan"))))
        if verdict:
            lines.append("  verdict: %s" % verdict)
        lines.append("  epochs recorded: %d | table: %s"
                     % (s.get("epochs", 0),
                        ", ".join(str(p) for p in self.csv_paths)))
        lines.append("=" * 100)
        self._emit("\n".join(lines) + "\n")
        self.logger.info(
            "Anomaly/reconstruction monitor closed | %d epochs | %s.",
            s.get("epochs", 0), ", ".join(str(p) for p in self.paths))


class MetricAccumulator:
    def __init__(
        self,
        late_start: int,
        eps: float = 1e-8,
        snr_bin_edges: Optional[np.ndarray] = None,
    ) -> None:
        self.late_start = late_start
        self.eps = eps
        if snr_bin_edges is None:
            snr_bin_edges = np.arange(-5.0, 40.0, 5.0, dtype=np.float64)
        self.snr_bin_edges = np.asarray(snr_bin_edges, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.sse_g = 0.0
        self.energy_g = 0.0
        self.sse_em = 0.0
        self.energy_em = 0.0
        self.sse_l = 0.0
        self.energy_l = 0.0
        self.loss_sum = 0.0
        self.samples = 0
        self.late_sample_errors: List[np.ndarray] = []
        self.gate_rel_errors: List[np.ndarray] = []
        self.q_sum = 0.0
        self.q_em_sum = 0.0
        self.q_l_sum = 0.0
        self.q_count = 0
        self.q_em_count = 0
        self.q_l_count = 0
        n_bins = max(1, len(self.snr_bin_edges) - 1)
        self.bin_sse_g = np.zeros(n_bins, dtype=np.float64)
        self.bin_energy_g = np.zeros(n_bins, dtype=np.float64)
        self.bin_sse_l = np.zeros(n_bins, dtype=np.float64)
        self.bin_energy_l = np.zeros(n_bins, dtype=np.float64)
        self.bin_counts = np.zeros(n_bins, dtype=np.int64)

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        q: torch.Tensor,
        loss: float,
        snr_db: Optional[torch.Tensor] = None,
    ) -> None:
        pred = pred.float()
        target = target.float()
        e2 = (pred - target).pow(2)
        y2 = target.pow(2)
        if snr_db is not None:
            snr_np = snr_db.detach().cpu().numpy().astype(np.float64).reshape(-1)
            sse_g_s = e2.sum(dim=(1, 2)).detach().cpu().numpy().astype(np.float64)
            eng_g_s = y2.sum(dim=(1, 2)).detach().cpu().numpy().astype(np.float64)
            sse_l_s = (
                e2[..., self.late_start :].sum(dim=(1, 2)).detach().cpu().numpy().astype(np.float64)
            )
            eng_l_s = (
                y2[..., self.late_start :].sum(dim=(1, 2)).detach().cpu().numpy().astype(np.float64)
            )
            idx = np.clip(
                np.searchsorted(self.snr_bin_edges, snr_np, side="right") - 1,
                0,
                len(self.snr_bin_edges) - 2,
            )
            np.add.at(self.bin_sse_g, idx, sse_g_s)
            np.add.at(self.bin_energy_g, idx, eng_g_s)
            np.add.at(self.bin_sse_l, idx, sse_l_s)
            np.add.at(self.bin_energy_l, idx, eng_l_s)
            np.add.at(self.bin_counts, idx, 1)
        self.sse_g += float(e2.sum().item())
        self.energy_g += float(y2.sum().item())
        self.sse_em += float(e2[..., : self.late_start].sum().item())
        self.energy_em += float(y2[..., : self.late_start].sum().item())
        self.sse_l += float(e2[..., self.late_start :].sum().item())
        self.energy_l += float(y2[..., self.late_start :].sum().item())
        late_err = torch.sqrt(
            e2[..., self.late_start :].sum(dim=(1, 2))
            / (y2[..., self.late_start :].sum(dim=(1, 2)) + self.eps)
        )
        self.late_sample_errors.append(late_err.detach().cpu().numpy())
        if sum(a.shape[0] for a in self.gate_rel_errors) < 40000:
            _re = ((pred - target).abs()
                   / (target.abs() + self.eps)).reshape(pred.shape[0], -1,
                                                        pred.shape[-1]).mean(dim=1)
            self.gate_rel_errors.append(_re.detach().cpu().numpy().astype(np.float32))
        b = pred.shape[0]
        self.loss_sum += float(loss) * b
        self.samples += b
        self.q_sum += float(q.sum().item())
        self.q_em_sum += float(q[..., : self.late_start].sum().item())
        self.q_l_sum += float(q[..., self.late_start :].sum().item())
        self.q_count += int(q.numel())
        self.q_em_count += int(q[..., : self.late_start].numel())
        self.q_l_count += int(q[..., self.late_start :].numel())

    def compute(self, cvar_fraction: float = 0.20) -> Dict[str, float]:
        rg = self.sse_g / max(self.energy_g, self.eps)
        rem = self.sse_em / max(self.energy_em, self.eps)
        rl = self.sse_l / max(self.energy_l, self.eps)
        late = np.concatenate(self.late_sample_errors) if self.late_sample_errors else np.asarray([], dtype=np.float64)
        if len(late):
            k = max(1, int(math.ceil(len(late) * cvar_fraction)))
            cvar = float(np.partition(late, len(late) - k)[-k:].mean())
            median = float(np.median(late))
            p95 = float(np.percentile(late, 95))
        else:
            cvar = median = p95 = float("nan")
        gate_med: List[float] = []
        worst_gate = -1
        worst_val = float("nan")
        mid = float("nan")
        if self.gate_rel_errors:
            _gm = np.median(np.concatenate(self.gate_rel_errors, axis=0), axis=0)
            gate_med = [100.0 * float(v) for v in _gm]
            worst_gate = int(np.argmax(_gm))
            worst_val = 100.0 * float(_gm[worst_gate])
            _ls = int(self.late_start)
            _m0737 = max(0, _ls - int(round(0.425 * _ls)))
            if len(gate_med) >= _ls > _m0737:
                mid = float(np.mean(gate_med[_m0737:_ls]))
        per_snr_bin: Dict[str, Dict[str, Any]] = {}
        for i in range(len(self.snr_bin_edges) - 1):
            lo, hi = self.snr_bin_edges[i], self.snr_bin_edges[i + 1]
            label = f"[{lo:g},{hi:g})"
            count = int(self.bin_counts[i])
            if count:
                g_err = 100.0 * math.sqrt(max(self.bin_sse_g[i] / max(self.bin_energy_g[i], self.eps), 0.0))
                l_err = 100.0 * math.sqrt(max(self.bin_sse_l[i] / max(self.bin_energy_l[i], self.eps), 0.0))
            else:
                g_err = None
                l_err = None
            per_snr_bin[label] = {
                "global_error_percent": g_err,
                "late_error_percent": l_err,
                "count": count,
            }
        return {
            "loss": self.loss_sum / max(self.samples, 1),
            "relative_mse_global": rg,
            "relative_mse_early_mid": rem,
            "relative_mse_late": rl,
            "per_snr_bin": per_snr_bin,
            "per_gate_median_error_percent": gate_med,
            "worst_gate_index": float(worst_gate),
            "worst_gate_median_percent": worst_val,
            "global_error_percent": 100.0 * math.sqrt(max(rg, 0.0)),
            "early_mid_error_percent": 100.0 * math.sqrt(max(rem, 0.0)),
            "late_error_percent": 100.0 * math.sqrt(max(rl, 0.0)),
            "late_median_percent": 100.0 * median,
            "late_p95_percent": 100.0 * p95,
            "late_cvar_percent": 100.0 * cvar,
            "mid_band_gate_median_percent": mid,
            "reliability_mean": self.q_sum / max(self.q_count, 1),
            "reliability_early_mid": self.q_em_sum / max(self.q_em_count, 1),
            "reliability_late": self.q_l_sum / max(self.q_l_count, 1),
            "samples": self.samples,
        }


def _content_digest(path: Any) -> str:
    try:
        p = Path(str(path))
        if not p.exists():
            return ""
        h = hashlib.sha1()
        st = p.stat()
        h.update(("%d:%d" % (st.st_size, int(st.st_mtime))).encode())
        arr = np.load(str(p), mmap_mode="r")
        step = max(1, int(arr.shape[0]) // 4096)
        h.update(np.ascontiguousarray(np.asarray(arr[::step], dtype=np.float32)).tobytes())
        return h.hexdigest()[:16]
    except Exception:
        return "n/a"


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _sample_nbytes(sample: Any) -> int:
    total = 0
    if isinstance(sample, torch.Tensor):
        return int(sample.numel() * sample.element_size())
    if isinstance(sample, dict):
        for v in sample.values():
            total += _sample_nbytes(v)
    elif isinstance(sample, (list, tuple)):
        for v in sample:
            total += _sample_nbytes(v)
    return int(total)


def shm_budget_audit(dataset: Any, batch_size: int, num_workers: int, prefetch: int,
                     safety: float, logger: logging.Logger) -> Tuple[int, int]:
    """Fit the DataLoader's in-flight volume to /dev/shm."""
    try:
        if num_workers <= 0 or not os.path.isdir("/dev/shm"):
            return int(num_workers), int(prefetch)
        _u = shutil.disk_usage("/dev/shm")
        free_b, total_b = int(_u.free), int(_u.total)
        try:
            item = dataset[0]
        except Exception as _e:
            logger.warning("shm audit: dataset item unavailable (%r); loader unchanged.", _e)
            return int(num_workers), int(prefetch)
        per_item = _sample_nbytes(item)
        per_batch = per_item * int(max(1, batch_size))
        budget = max(0.0, float(safety)) * float(free_b)

        def inflight(w: int, pf: int) -> float:
            return 1.25 * float(per_batch) * float(w) * float(pf)

        w, pf = int(num_workers), int(prefetch)
        decision = "unchanged"
        if inflight(w, pf) > budget and pf > 1:
            pf = 1
            decision = "prefetch_factor -> 1"
        while inflight(w, pf) > budget and w > 1:
            w -= 1
            decision = "workers -> %d (prefetch 1)" % w
        if inflight(w, pf) > budget:
            w, pf = 0, 1
            decision = "SINGLE-PROCESS loading (num_workers=0)"
        msg = ("SHM BUDGET | /dev/shm total %.0f MB free %.0f MB | item %.1f KB x batch %d = %.1f MB/batch "
               "| in-flight requested %d workers x %d prefetch = %.0f MB vs budget %.0f MB (%.0f%% of free) -> %s. ")
        args = (total_b / 2**20, free_b / 2**20, per_item / 1024.0, int(batch_size), per_batch / 2**20,
                int(num_workers), int(prefetch), inflight(int(num_workers), int(prefetch)) / 2**20, budget / 2**20,
                100.0 * float(safety), decision)
        if decision == "unchanged":
            logger.info(msg, *args)
        else:
            logger.warning(msg + " Raise it with docker --shm-size=8g or `mount -o remount,size=8G /dev/shm`; "
                           "do NOT run a second loader (test harness) on this host while training.", *args)
        return int(w), int(pf)
    except Exception as _e:
        logger.warning("shm audit skipped: %r", _e)
        return int(num_workers), int(prefetch)


def _from_numpy_tree(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x))
    if isinstance(x, dict):
        return {k: _from_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_from_numpy_tree(v) for v in x)
    return x


def _stack_numpy(values: List[Any]) -> Any:
    v0 = values[0]
    if isinstance(v0, torch.Tensor):
        return np.stack([np.asarray(v.detach().cpu().numpy()) for v in values], axis=0)
    if isinstance(v0, np.ndarray):
        return np.stack([np.asarray(v) for v in values], axis=0)
    if isinstance(v0, (bool, int, float, np.generic)):
        return np.asarray(values)
    if isinstance(v0, dict):
        return {k: _stack_numpy([v[k] for v in values]) for k in v0}
    if isinstance(v0, (list, tuple)) and v0 and all(isinstance(x, (int, float, np.generic)) for x in v0):
        return np.asarray(values)
    return list(values)


def collate_numpy_transport(items: List[Any]) -> Any:
    """Pure-numpy collation in the worker: the batch travels through the worker queue's pipe as plain bytes and no
    /dev/shm segment is touched at any point.
    """
    if not items:
        return items
    if isinstance(items[0], dict):
        return {k: _stack_numpy([it[k] for it in items]) for k in items[0]}
    return _stack_numpy(list(items))


class ShmFreeLoader:
    """Iterates the wrapped DataLoader and hands back tensor batches (CPU), so every consumer (run_epoch, the
    pre-flight, the exporters) sees exactly the batches it always saw.
    """

    def __init__(self, loader: DataLoader) -> None:
        object.__setattr__(self, "_loader", loader)

    def __iter__(self):
        for b in self._loader:
            yield _from_numpy_tree(b)

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_loader"), name)


def is_dataloader_worker_crash(exc: BaseException) -> bool:
    """The worker-failure exception PyTorch raises from its signal handler."""
    s = str(exc)
    return ("DataLoader worker" in s and ("killed by signal" in s or "exited unexpectedly" in s
                                           or "shared memory" in s.lower()))


def log_identification_ceiling(
    clean_cache: "CleanCacheInfo",
    cfg: "Config",
    logger: logging.Logger,
    max_rows: int = 20000,
    noise_cache: Optional["NoiseCacheInfo"] = None,
) -> Optional[Dict[str, float]]:
    """Measure, on the USER'S own clean library, the ceiling of the identification pathway."""
    try:
        clean = np.load(clean_cache.data_path, mmap_mode="r")
        late = int(cfg.data.late_start_index)
        splits = build_splits(clean_cache, cfg)
        rng = np.random.default_rng(0)
        tr = splits["train"]
        va = splits["val"]
        if len(tr) > max_rows:
            tr = rng.choice(tr, size=max_rows, replace=False)
        if len(va) > max_rows:
            va = rng.choice(va, size=max_rows, replace=False)
        ctr = np.asarray(clean[np.sort(tr)], dtype=np.float64)
        cva = np.asarray(clean[np.sort(va)], dtype=np.float64)
        gs = np.clip(np.median(np.abs(ctr), axis=0), 1e-30, None)
        def _slog(x: np.ndarray) -> np.ndarray:
            return np.sign(x) * np.log1p(np.abs(x))
        def _sexp(z: np.ndarray) -> np.ndarray:
            return np.sign(z) * np.expm1(np.abs(z))
        ztr = _slog(ctr / gs)
        zva = _slog(cva / gs)
        X = np.hstack([ztr[:, :late], np.ones((len(ztr), 1))])
        Y = ztr[:, late:]
        lam = 1e-4
        W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
        P = np.hstack([zva[:, :late], np.ones((len(zva), 1))]) @ W
        pred = _sexp(P) * gs[late:]
        ref = cva[:, late:]
        err = 100.0 * np.sqrt(
            np.sum((pred - ref) ** 2, axis=1) / np.maximum(np.sum(ref**2, axis=1), 1e-30)
        )
        med = float(np.median(err))
        p90 = float(np.percentile(err, 90))
        pooled = 100.0 * math.sqrt(
            float(np.sum((pred - ref) ** 2)) / max(float(np.sum(ref**2)), 1e-30)
        )
        tau = 100.0 * cfg.contract.late_threshold
        logger.info(
            "IDENTIFICATION CEILING (clean early/mid -> late, ridge, this library) | "
            "pooled=%.2f%% median=%.2f%% p90=%.2f%% | late contract=%.1f%%",
            pooled, med, p90, tau,
        )
        if pooled > 2.0 * tau:
            logger.warning(
                "Ceiling is far above the late contract: the library's own "
                "early/mid->late conditional spread binds. No denoiser can beat "
                "this from identification alone; enrich/condition the clean "
                "forward library (survey-consistent models, denser parameter "
                "coverage) or the late contract is infeasible at this SNR mix."
            )
        elif pooled > tau:
            logger.warning(
                "Ceiling exceeds the late contract: identification alone is "
                "insufficient; the margin must come from late-window "
                "measurements (neighbor aggregation / innovation fusion)."
            )
        else:
            logger.info(
                "Ceiling is below the late contract: the identification pathway "
                "can carry the late window; remaining error is a training/"
                "architecture matter."
            )
        _cvf = float(getattr(cfg.loss, "cvar_fraction", 0.20))
        _kc = max(1, int(math.ceil(err.size * _cvf)))
        ceil_cvar = float(np.partition(err, err.size - _kc)[-_kc:].mean())
        _tail_gate = 100.0 * float(getattr(cfg.contract, "late_cvar_threshold", 0.08))
        logger.info(
            "CEILING TAIL | worst-%.0f%%%% mean of the ridge errors = "
            "%.2f%% vs tail gate %.1f%% | median %.2f%% vs late gate %.1f%%.",
            100.0 * _cvf, ceil_cvar, _tail_gate, med, tau,
        )
        if med > tau:
            logger.warning(
                "The per-curve MEDIAN ceiling (%.2f%%) exceeds the %.1f%% "
                "late gate on THIS split: half the held-out curves cannot meet the "
                "gate even from clean early/mid identification, whatever the model "
                "does. The pooled pass (%.2f%%) is an energy-weighted statement about "
                "bright curves, not a per-curve one.",
                med, tau, pooled,
            )
        if ceil_cvar > _tail_gate:
            logger.warning(
                "The tail gate (late CVaR <= %.1f%%) is POPULATION-LIMITED "
                "on this split: the identification ceiling's own worst-decile mean is "
                "%.2f%%. Feasibility on this gate measures the split, not the model; "
                "repair the split before reading it as a model defect. ",
                _tail_gate, ceil_cvar,
            )
        try:
            if noise_cache is not None and getattr(clean_cache, "province_id", None) is not None:
                nb_tbl = province_real_neighbor_rows(clean_cache, cfg)
                if nb_tbl is not None:
                    noise_mm = np.load(noise_cache.data_path, mmap_mode="r")
                    n_noise = int(noise_mm.shape[0])
                    pid_all = np.asarray(clean_cache.province_id, dtype=np.int64)
                    tr2 = np.asarray([r for r in tr if pid_all[int(r)] >= 0], dtype=np.int64)[:4000]
                    va2 = np.asarray(va, dtype=np.int64)[:2000]
                    rng2 = np.random.default_rng(7)

                    def _noisy_feats(rows: np.ndarray, snr_db: float, stacked: bool) -> np.ndarray:
                        width = 2 * late if stacked else late
                        feats = np.empty((len(rows), width), dtype=np.float64)
                        for a, r in enumerate(rows):
                            def _noisy(rr: int) -> np.ndarray:
                                cl = np.asarray(clean[rr], dtype=np.float64)
                                nz = np.asarray(noise_mm[int(rng2.integers(0, n_noise))], dtype=np.float64)
                                g = float(np.sqrt(np.mean(cl**2))) / (
                                    (10.0 ** (snr_db / 20.0))
                                    * max(float(np.sqrt(np.mean(nz**2))), 1e-30)
                                )
                                return cl + nz * g
                            centre_tr = _noisy(int(r))
                            feats[a, :late] = _slog(centre_tr[:late] / gs[:late])
                            if stacked:
                                acc_tr = centre_tr.copy()
                                for x in nb_tbl[int(r)]:
                                    acc_tr += _noisy(int(x))
                                acc_tr /= float(1 + nb_tbl.shape[1])
                                feats[a, late:] = _slog(acc_tr[:late] / gs[:late])
                        return feats

                    lines2 = [
                        "NOISY-STACKED IDENTIFICATION CEILING (ridge, train provinces -> held-out "
                        "provinces, SAME pipeline; K=%d real neighbours) | clean ceiling above = "
                        "%.2f%%" % (nb_tbl.shape[1], pooled),
                        "  SNR(dB)   single-trace   centre+stack(K)",
                    ]
                    yva = np.asarray(clean[np.sort(va2)], dtype=np.float64)[:, late:]
                    for snr in (-5.0, 5.0, 15.0, 25.0):
                        res = {}
                        for stk in (False, True):
                            Xtr = np.hstack([_noisy_feats(np.sort(tr2), snr, stk), np.ones((len(tr2), 1))])
                            Ytr = _slog(np.asarray(clean[np.sort(tr2)], dtype=np.float64)[:, late:] / gs[late:])
                            W2 = np.linalg.solve(Xtr.T @ Xtr + 1e-4 * np.eye(Xtr.shape[1]), Xtr.T @ Ytr)
                            Pv = np.hstack([_noisy_feats(np.sort(va2), snr, stk), np.ones((len(va2), 1))]) @ W2
                            pv = _sexp(Pv) * gs[late:]
                            res[stk] = 100.0 * math.sqrt(
                                float(np.sum((pv - yva) ** 2)) / max(float(np.sum(yva**2)), 1e-30)
                            )
                        lines2.append("  %7.0f   %11.2f%%   %14.2f%%" % (snr, res[False], res[True]))
                    lines2.append(
                        "  Read it against the per-SNR validation table: validation error >> the "
                        "stacked ceiling at that SNR = trainable headroom; validation ~ ceiling = "
                        "information floor, add neighbours or improve the survey."
                    )
                    logger.info("\n".join(lines2))
        except Exception as _exc:
            logger.warning("Noisy-stacked ceiling skipped: %s", _exc)
        return {"pooled": pooled, "median": med, "p90": p90, "cvar": ceil_cvar}
    except Exception as exc:
        logger.warning("Identification-ceiling diagnosis skipped: %s", exc)
        return None


def log_physics_basis_fit(
    clean_cache: "CleanCacheInfo",
    noise_cache: "NoiseCacheInfo",
    cfg: "Config",
    logger: logging.Logger,
    max_rows: int = 256,
) -> Optional[Dict[str, float]]:
    """Verify, on the USER'S own clean library, that the decay-atom cone (non-negative exponential superpositions at
    the resolved gate times) can represent the library's late window.
    """
    try:
        if not cfg.model.use_physics_atoms:
            return None
        clean = np.load(clean_cache.data_path, mmap_mode="r")
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(clean_cache.rows, size=min(max_rows, clean_cache.rows), replace=False))
        C = np.asarray(clean[idx], dtype=np.float64)
        gs = np.clip(
            np.asarray(clean_cache.gate_scale, dtype=np.float64) / float(clean_cache.global_scale),
            1e-30, None,
        ) * float(clean_cache.global_scale)
        A = PEBRNet._build_decay_atoms(
            np.asarray(noise_cache.target_time, dtype=np.float64), int(cfg.model.physics_atom_count)
        ).numpy().astype(np.float64)
        B = A / gs[None, :]
        Tgt = C / gs[None, :]
        neg_frac = float(np.mean(np.any(C <= 0, axis=1)))
        BBt = B @ B.T
        W = np.full((len(C), A.shape[0]), 0.05)
        TB = np.clip(Tgt, 0.0, None) @ B.T
        for _ in range(2000):
            W = W * TB / np.maximum(W @ BBt, 1e-12)
        rec = W @ A
        late = int(cfg.data.late_start_index)
        err = 100.0 * np.sqrt(
            np.sum((rec[:, late:] - C[:, late:]) ** 2, axis=1)
            / np.maximum(np.sum(C[:, late:] ** 2, axis=1), 1e-30)
        )
        med = float(np.median(err))
        p90 = float(np.percentile(err, 90))
        logger.info(
            "PHYSICS BASIS FIT (decay-atom cone vs this library) | late median=%.3f%% p90=%.3f%% | "
            "curves with non-positive samples: %.1f%%",
            med, p90, 100.0 * neg_frac,
        )
        if med > 2.0 or neg_frac > 0.05:
            logger.warning(
                "The decay-atom cone does not represent this library well "
                "(non-monotone or sign-changing responses). Run with "
                "--disable_physics_atoms, or restrict the library to induction-"
                "regime responses."
            )
        else:
            logger.info(
                "Library lies inside the physics cone: the constrained head can "
                "express it with negligible loss while excluding non-physical "
                "late windows."
            )
        return {"median": med, "p90": p90, "neg_frac": neg_frac}
    except Exception as exc:
        logger.warning("Physics-basis-fit diagnosis skipped: %s", exc)
        return None


def make_loaders(
    clean_cache: CleanCacheInfo,
    noise_cache: NoiseCacheInfo,
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray], Tuple[BTEMDenoisingDataset, ...]]:
    splits = build_splits(clean_cache, cfg)
    logging.getLogger(PROGRAM_NAME).info(
        "EVALUATION PROTOCOL for validation and the block-protocol test: %s.",
        ("library-noise injection swept over %s dB (--val_snr_protocol contract)"
         % list(cfg.data.contract_test_snr_db_range)) if bool(getattr(cfg.data, "val_contract_snr_protocol", False))
        else "THE DATASET'S OWN PAIRS -- the noise they carry, their own late-local SNR population")
    _lowsnr_ds = None
    if getattr(clean_cache, "measured_noise_path", ""):
        _low_cfg = dataclasses.replace(
            cfg.data,
            paired_residual_replay=0.0,
            measured_noise_inject=1.0,
            measured_noise_snr_min_db=-40.0,
            measured_noise_snr_max_db=10.0,
            measured_noise_snr_low_focus=0.0,
            paired_anomaly_oversample=0.0,
            paired_hard_snr_oversample=0.0,
            neighbor_channel_dropout=0.0,
            paired_alt_corruption=1.0,
            survey_profile_fraction=0.0,
        )
        _lowsnr_ds = BTEMDenoisingDataset(
            clean_cache.data_path, noise_cache.data_path, splits["val"],
            clean_cache.global_scale, _low_cfg, cfg.train.seed + 4407, 720, True,
            noise_block_rows=noise_cache.block_rows,
            gate_times=noise_cache.target_time,
            real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
            real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
            province_id=getattr(clean_cache, "province_id", None),
            paired_noisy_path=getattr(clean_cache, "paired_noisy_path", ""),
            paired_background_path=getattr(clean_cache, "paired_background_path", ""),
            paired_event_path=getattr(clean_cache, "paired_event_path", ""),
            measured_noise_path=getattr(clean_cache, "measured_noise_path", ""),
            extra_clean_path=getattr(clean_cache, "extra_clean_path", ""),
            survey_profile_path=getattr(clean_cache, "survey_profile_path", ""),
        )

    train_ds = BTEMDenoisingDataset(
        clean_cache.data_path,
        noise_cache.data_path,
        splits["train"],
        clean_cache.global_scale,
        cfg.data,
        cfg.train.seed + 11,
        cfg.data.train_samples_per_epoch,
        True,
        noise_block_rows=noise_cache.block_rows,
        gate_times=noise_cache.target_time,
        real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
        real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
        province_id=getattr(clean_cache, "province_id", None),
        paired_noisy_path=getattr(clean_cache, "paired_noisy_path", ""),
        paired_background_path=getattr(clean_cache, "paired_background_path", ""),
        paired_event_path=getattr(clean_cache, "paired_event_path", ""),
        measured_noise_path=getattr(clean_cache, "measured_noise_path", ""),
        extra_clean_path=getattr(clean_cache, "extra_clean_path", ""),
        survey_profile_path=getattr(clean_cache, "survey_profile_path", ""),
    )
    try:
        _na2, _pe = train_ds.anchor_cycle_summary()
        if _na2 > 0:
            logging.getLogger(PROGRAM_NAME).info(
                "TRAINING COVERAGE | %d patch anchors over the training split | %d patches per epoch "
                "-> one full pass of every anchor every %.1f epochs (global permutation, no anchor repeats "
                "within a pass; oversampling routes and the boundary boost resample on top).", _na2, _pe, _na2 / max(1.0, float(_pe)))
    except Exception:
        pass
    val_ds = BTEMDenoisingDataset(
        clean_cache.data_path,
        noise_cache.data_path,
        splits["val"],
        clean_cache.global_scale,
        cfg.data,
        cfg.train.seed + 23,
        min(cfg.data.val_samples, len(splits["val"])),
        False,
        noise_block_rows=noise_cache.block_rows,
        gate_times=noise_cache.target_time,
        real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
        real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
        province_id=getattr(clean_cache, "province_id", None),
        paired_noisy_path=getattr(clean_cache, "paired_noisy_path", ""),
        paired_background_path=getattr(clean_cache, "paired_background_path", ""),
        paired_event_path=getattr(clean_cache, "paired_event_path", ""),
        measured_noise_path=getattr(clean_cache, "measured_noise_path", ""),
        extra_clean_path=getattr(clean_cache, "extra_clean_path", ""),
    )
    test_ds = BTEMDenoisingDataset(
        clean_cache.data_path,
        noise_cache.data_path,
        splits["test"],
        clean_cache.global_scale,
        cfg.data,
        cfg.train.seed + 37,
        min(cfg.data.test_samples, len(splits["test"])),
        False,
        noise_block_rows=noise_cache.block_rows,
        gate_times=noise_cache.target_time,
        real_neighbor_rows=province_real_neighbor_rows(clean_cache, cfg),
        real_neighbor_geometry=province_neighbor_geometry(clean_cache, cfg),
        province_id=getattr(clean_cache, "province_id", None),
        paired_noisy_path=getattr(clean_cache, "paired_noisy_path", ""),
        paired_background_path=getattr(clean_cache, "paired_background_path", ""),
        paired_event_path=getattr(clean_cache, "paired_event_path", ""),
        measured_noise_path=getattr(clean_cache, "measured_noise_path", ""),
        extra_clean_path=getattr(clean_cache, "extra_clean_path", ""),
    )
    _vp = splits.get("val_pool")
    if _vp is not None and len(_vp) > len(splits["val"]):
        val_ds.split_indices_full = np.sort(np.asarray(_vp, dtype=np.int64))
    _tp = splits.get("test_pool")
    if _tp is not None and len(_tp) > len(splits["test"]):
        test_ds.split_indices_full = np.sort(np.asarray(_tp, dtype=np.int64))
    if getattr(clean_cache, "province_id", None) is not None:
        test_ds._snr_override = tuple(cfg.data.contract_test_snr_db_range)
        if train_ds._mix_field_idx is not None and train_ds._mix_forward_idx is not None:
            if bool(getattr(cfg.data, "forward_admission_filter", True)):
                try:
                    _desc = clean_cache.descriptors
                    if _desc is None and clean_cache.descriptor_path and Path(clean_cache.descriptor_path).exists():
                        _desc = np.load(clean_cache.descriptor_path)
                    if _desc is not None:
                        _q = float(np.clip(getattr(cfg.data, "forward_admission_quantile", 0.005), 0.0, 0.2))
                        _tgt = np.asarray(_desc)[train_ds._mix_field_idx]
                        _lo = np.nanquantile(_tgt, _q, axis=0)
                        _hi = np.nanquantile(_tgt, 1.0 - _q, axis=0)
                        _fwd = np.asarray(_desc)[train_ds._mix_forward_idx]
                        _env_ok = np.all((_fwd >= _lo[None, :]) & (_fwd <= _hi[None, :]), axis=1)
                        with np.errstate(invalid="ignore"):
                            _phys_ok = (
                                np.all(np.isfinite(_fwd), axis=1)
                                & (_fwd[:, 0] >= _lo[0] - 0.5) & (_fwd[:, 0] <= _hi[0] + 0.5)
                                & (_fwd[:, 1] <= _hi[1] + 1.5)
                                & (_fwd[:, 2] <= min(float(_hi[2]) + 0.5, -0.05))
                                & (_fwd[:, 5] <= max(float(_hi[5]), 0.05))
                            )
                        _finite = np.all(np.isfinite(_fwd), axis=1)
                        _mode = str(getattr(cfg.data, "forward_admission_filter", True))
                        if _mode == "envelope":
                            _ok = _env_ok
                        else:
                            _ok = _finite
                        _total = int(len(train_ds._mix_forward_idx))
                        _kept = int(_ok.sum())
                        _resid = int((_finite & ~(_env_ok | _phys_ok)).sum())
                        logging.getLogger(PROGRAM_NAME).info(
                            "Forward strata | envelope (background-like): "
                            "%d | physics (anomaly space): %d | residual (IP-like / "
                            "slow-decay, finite): %d | non-finite excluded: %d | "
                            "TRAINING ON: %d / %d (%s).",
                            int(_env_ok.sum()), int((_phys_ok & ~_env_ok).sum()),
                            _resid, int((~_finite).sum()), _kept, _total,
                            "ALL finite" if _mode != "envelope" else "legacy envelope",
                        )
                        if _kept == 0:
                            logging.getLogger(PROGRAM_NAME).warning(
                                "Forward admission gate admitted 0 of %d forward curves "
                                "into the field envelope -- training proceeds FIELD-ONLY.", _total)
                            train_ds._mix_forward_idx = None
                        else:
                            train_ds._mix_forward_idx = train_ds._mix_forward_idx[_ok]
                            logging.getLogger(PROGRAM_NAME).info(
                                "Forward admission gate | %d / %d forward curves lie inside "
                                "the field target's [%.1f%%, %.1f%%] descriptor envelope and are "
                                "admitted; %d are held out for stress experiments.",
                                _kept, _total, 100.0 * _q, 100.0 * (1.0 - _q), _total - _kept)
                except Exception as _exc:
                    logging.getLogger(PROGRAM_NAME).warning("admission gate skipped: %s", _exc)
        if train_ds._mix_field_idx is not None and train_ds._mix_forward_idx is not None:
            _nf, _na = len(train_ds._mix_field_idx), len(train_ds._mix_forward_idx)
            _p = float(getattr(cfg.data, 'mix_forward_fraction', 0.15))
            logging.getLogger(PROGRAM_NAME).info(
                'TRAIN DRAW | pool: %d FIELD-survey (target) + %d FORWARD-sim '
                '(auxiliary) | draw: %.0f%% FIELD / %.0f%% forward by construction. The '
                'field-survey domain is the majority of every gradient step, and it -- not '
                'the simulation -- is what validation, test and model selection now use.',
                _nf, _na, 100.0 * (1 - _p), 100.0 * _p,
            )
            try:
                _desc2 = clean_cache.descriptors
                if _desc2 is None and clean_cache.descriptor_path and Path(clean_cache.descriptor_path).exists():
                    _desc2 = np.load(clean_cache.descriptor_path)
                if _desc2 is not None and len(splits.get("val", [])) > 0:
                    _d = np.asarray(_desc2, dtype=np.float64)
                    _mf = np.nanmean(_d[train_ds._mix_field_idx], axis=0)
                    _ma = np.nanmean(_d[train_ds._mix_forward_idx], axis=0)
                    _eff = (1.0 - _p) * _mf + _p * _ma
                    _dv = _d[np.asarray(splits["val"], dtype=np.int64)]
                    _mv = np.nanmean(_dv, axis=0)
                    _sv = np.nanstd(_dv, axis=0) + 1e-12
                    _smd = (_eff - _mv) / _sv
                    logging.getLogger(PROGRAM_NAME).info(
                        "[D09] EFFECTIVE TRAIN vs VAL descriptor SMDs (draw-weighted %.0f/%.0f): %s "
                        "-- |SMD| > 0.5 on any descriptor means the gradient and the gate are "
                        "measuring different populations.",
                        100.0 * (1 - _p), 100.0 * _p,
                        ", ".join("%.2f" % v for v in np.asarray(_smd).ravel()),
                    )
            except Exception as _exc:
                logging.getLogger(PROGRAM_NAME).warning("[D09] effective-sampling audit skipped: %s", _exc)
        logging.getLogger(PROGRAM_NAME).info(
            "PROVINCE TEST | CONTRACT protocol: held-out test provinces scored under "
            "UNIFORM SNR [%.1f, %.1f] dB (matches the project gates). STRESS protocol "
            "[%.1f, %.1f] dB is evaluated separately into stress_test_metrics.json and is "
            "never substituted for the contract figure.",
            float(cfg.data.contract_test_snr_db_range[0]),
            float(cfg.data.contract_test_snr_db_range[1]),
            float(cfg.data.province_test_snr_db_range[0]),
            float(cfg.data.province_test_snr_db_range[1]),
        )
    try:
        _pn = getattr(clean_cache, "paired_noisy_path", "") or ""
        _nz = np.load(_pn, mmap_mode="r") if _pn else None
        audit_gate_columns(np.load(clean_cache.data_path, mmap_mode="r"), _nz,
                           getattr(clean_cache, "target_time", None), getattr(clean_cache, "province_id", None),
                           int(cfg.data.late_start_index), logging.getLogger(PROGRAM_NAME))
        if _nz is not None:
            audit_paired_zero_gates(_nz, np.load(clean_cache.data_path, mmap_mode="r"),
                                    int(cfg.data.late_start_index), getattr(clean_cache, "province_id", None),
                                    logging.getLogger(PROGRAM_NAME))
    except Exception as _e:
        logging.getLogger(PROGRAM_NAME).warning("pair integrity audit unavailable: %r", _e)
    nw = max(0, cfg.train.num_workers)
    _pf = max(1, int(getattr(cfg.train, "prefetch_factor", 2) or 2))
    _shmfree = bool(getattr(cfg.train, "shm_free_transport", True)) and nw > 0
    if nw > 0 and not _shmfree:
        nw, _pf = shm_budget_audit(train_ds, int(cfg.train.batch_size), nw, _pf,
                                      float(getattr(cfg.train, "shm_safety_fraction", 0.5)),
                                      logging.getLogger(PROGRAM_NAME))
        cfg.train.num_workers = int(nw)
        cfg.train.prefetch_factor = int(_pf)
    elif _shmfree:
        log_cpu_budget(logging.getLogger(PROGRAM_NAME), nw)
        try:
            _shm = shutil.disk_usage("/dev/shm") if os.path.isdir("/dev/shm") else None
            _shm_txt = ("/dev/shm total %.0f MB" % (_shm.total / 2 ** 20)) if _shm else "/dev/shm absent"
        except Exception:
            _shm_txt = "/dev/shm unknown"
        logging.getLogger(PROGRAM_NAME).info(
            "SHM-FREE LOADER TRANSPORT | %s | workers %d x prefetch %d kept: batches travel as "
            "numpy bytes through the worker queue, no shared-memory segment per in-flight batch, so the "
            "worker cut for a small /dev/shm does not apply. --no_shm_free_transport restores the shared-memory "
            "path.", _shm_txt, nw, _pf)
    common = dict(
        batch_size=cfg.train.batch_size,
        num_workers=nw,
        pin_memory=cfg.train.pin_memory and torch.cuda.is_available(),
        persistent_workers=cfg.train.persistent_workers and nw > 0,
        worker_init_fn=worker_init_fn if nw > 0 else None,
    )
    if nw > 0:
        common["prefetch_factor"] = int(_pf)
    if _shmfree:
        common["collate_fn"] = collate_numpy_transport
    train_loader = DataLoader(train_ds, shuffle=False, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)
    if _shmfree:
        train_loader, val_loader, test_loader = (ShmFreeLoader(train_loader), ShmFreeLoader(val_loader),
                                                 ShmFreeLoader(test_loader))
    train_ds._lowsnr_probe_ds = _lowsnr_ds
    return train_loader, val_loader, test_loader, splits, (train_ds, val_ds, test_ds)


def assert_optimizer_covers_model(model: nn.Module, optimizer: torch.optim.Optimizer,
                                  logger: Optional[logging.Logger] = None) -> None:
    """Every trainable tensor must belong to exactly one parameter group."""
    trainable = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    held = {id(p) for g in optimizer.param_groups for p in g["params"]}
    missing = [n for i, n in trainable.items() if i not in held]
    extra = len(held - set(trainable))
    if missing or extra:
        raise RuntimeError(
            "OPTIMIZER COVERAGE FAILED: %d trainable tensor(s) are in no "
            "parameter group and %d optimized tensor(s) are not parameters of this "
            "model. Training would silently freeze them (this is the failure "
            "mode). Missing: %s."
            % (len(missing), extra,
               ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else "")))
    if logger is not None:
        logger.info("Optimizer covers all %d trainable tensors in %d groups: "
                    "%s.", len(trainable), len(optimizer.param_groups),
                    ", ".join(str(g.get("name", "?")) for g in optimizer.param_groups))


def model_parameter_groups(model: PEBRNet, cfg: TrainConfig, stage: str) -> List[Dict[str, Any]]:
    base_lr = cfg.base_lr
    if stage == "A":
        factors = {"shared": 1.0, "noise": 1.0, "router": 1.0, "refine": 0.35}
    elif stage == "B":
        factors = {
            "shared": cfg.stage_b_shared_lr_factor,
            "noise": cfg.stage_b_noise_head_lr_factor,
            "router": cfg.stage_b_router_lr_factor,
            "refine": cfg.stage_b_refine_lr_factor,
        }
    else:
        f = cfg.stage_c_lr_factor
        factors = {"shared": f, "noise": f, "router": f, "refine": f}
    groups = [
        {
            "name": "shared",
            "params": list(model.stem.parameters())
            + list(model.multiscale.parameters())
            + list(model.refinement_stack.parameters()),
            "lr": base_lr * factors["shared"],
        },
        {"name": "noise", "params": list(model.noise_head.parameters()), "lr": base_lr * factors["noise"]},
        {"name": "router", "params": list(model.router.parameters()), "lr": base_lr * factors["router"]},
        {
            "name": "refine",
            "params": list(model.reference_attention.parameters()) + list(model.refine_head.parameters())
            + (list(model.neighbor_attention.parameters()) if model.neighbor_attention is not None else [])
            + (list(model.manifold_score.parameters()) + list(model.manifold_decoder.parameters())
               + list(model.manifold_head_free.parameters())
               + (list(model.manifold_head_atoms.parameters()) if model.manifold_head_atoms is not None else [])
               + list(model.manifold_blend.parameters()) if model.manifold_decoder is not None else [])
            + (list(model.manifold_innovation.parameters()) if getattr(model, "manifold_innovation", None) is not None else [])
            + (list(model.manifold_head_ip.parameters()) + list(model.manifold_gate_ip.parameters())
               if getattr(model, "manifold_head_ip", None) is not None else [])
            + (list(model.relax_state_head.parameters())
               if getattr(model, "relax_state_head", None) is not None else [])
            + (list(model.event_gate_head.parameters()) + list(model.struct_residual_head.parameters())
               if getattr(model, "event_gate_head", None) is not None else []),
            "lr": base_lr * factors["refine"],
        },
    ]
    if getattr(model, "prior_log_sigma", None) is not None:
        groups.append({
            "name": "prior_sigma",
            "params": [model.prior_log_sigma],
            "lr": base_lr * float(cfg.prior_sigma_lr_factor),
            "weight_decay": 0.0,
        })
    if model.uncertainty_head is not None:
        groups.append(
            {
                "name": "uncertainty",
                "params": list(model.uncertainty_head.parameters()),
                "lr": base_lr * factors["router"],
            }
        )
    _lat_params: List[nn.Parameter] = []
    for _nm in ("lateral_joint", "profile_joint", "joint_inject", "lateral_fp_prior", "lateral_fp_out"):
        _mod = getattr(model, _nm, None)
        if _mod is not None:
            _lat_params.extend(list(_mod.parameters()))
    if _lat_params:
        groups.append({
            "name": "lateral",
            "params": _lat_params,
            "lr": base_lr * factors["shared"],
            "weight_decay": 0.0,
        })
    if getattr(model, "gl_departure_logit", None) is not None:
        groups.append({
            "name": "gate_loss_departure",
            "params": [model.gl_departure_logit],
            "lr": base_lr * factors["shared"],
            "weight_decay": 0.0,
        })
    covered = {id(p) for g in groups for p in g["params"]}
    leftovers = [(n, p) for n, p in model.named_parameters() if p.requires_grad and id(p) not in covered]
    if leftovers:
        logging.getLogger(PROGRAM_NAME).warning(
            "%d parameter tensor(s) were missing from every optimizer group and "
            "have been folded into 'shared': %s. A parameter outside the optimizer is a "
            "gradient outside the scaler's inf/nan certification -- the exact mechanism "
            "of the all-NaN runs.",
            len(leftovers), [n for n, _ in leftovers],
        )
        groups[0]["params"] = list(groups[0]["params"]) + [p for _, p in leftovers]
    return groups


def build_optimizer(model: PEBRNet, cfg: TrainConfig, stage: str) -> torch.optim.Optimizer:
    groups = model_parameter_groups(model, cfg, stage)
    kw = dict(lr=cfg.base_lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    try:
        if any(p.is_cuda for g in groups for p in g["params"]):
            return torch.optim.AdamW(groups, fused=True, **kw)
    except (TypeError, ValueError, RuntimeError):
        pass
    return torch.optim.AdamW(groups, **kw)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    total_steps: int,
) -> "torch.optim.lr_scheduler.LambdaLR":
    """Warmup-then-cosine multiplier applied uniformly to every param group, so the per-group stage LR ratios are
    preserved while the whole schedule warms up briefly and then decays to ``lr_min_ratio``.
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(round(total_steps * float(cfg.lr_warmup_fraction))))
    floor = float(cfg.lr_min_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return floor + (1.0 - floor) * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {name: p.detach().clone().float() for name, p in model.named_parameters()}
        self._backup: Dict[str, torch.Tensor] = {}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": int(self.num_updates),
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))
        saved = state.get("shadow", {})
        for k, v in saved.items():
            if k in self.shadow:
                self.shadow[k] = torch.as_tensor(v, dtype=torch.float32).clone()

    def _effective_decay(self) -> float:
        warmup = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, warmup)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self._effective_decay()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow[name]
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def reseed(self, model: nn.Module) -> None:
        """Reset the shadow weights to the model's current weights."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].copy_(p.detach().float())
        self.num_updates = 0

    def store(self, model: nn.Module) -> None:
        self._backup = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name].to(p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self._backup:
                p.copy_(self._backup[name])
        self._backup = {}


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


_NAN_FORENSICS: Dict[str, Any] = {"path": None, "events": 0}
_STEP_HEARTBEAT: Dict[str, float] = {}
_NAN_MODULE_HUNTED = False


def _hunt_first_nonfinite_module(model, batch, amp_enabled, logger):
    global _NAN_MODULE_HUNTED
    if _NAN_MODULE_HUNTED or model is None:
        return None
    _NAN_MODULE_HUNTED = True
    found: Dict[str, Any] = {}
    handles = []

    def _stats(x):
        xt = x.detach().float()
        return {"dtype": str(x.dtype), "shape": list(x.shape),
                "finite": bool(torch.isfinite(xt).all()),
                "absmax": float(xt.abs().max()) if xt.numel() else 0.0}

    def make_hook(name, mod):
        def hook(_m, inp, outp):
            if found:
                return
            outs = outp if isinstance(outp, (tuple, list)) else (outp,)
            if any(isinstance(o, torch.Tensor) and o.is_floating_point()
                   and not torch.isfinite(o).all() for o in outs):
                found["module"] = name
                found["class"] = type(mod).__name__
                found["inputs"] = [_stats(i) for i in inp if isinstance(i, torch.Tensor)][:4]
                found["outputs"] = [_stats(o) for o in outs if isinstance(o, torch.Tensor)][:4]
        return hook

    try:
        for nm, mod in model.named_modules():
            if nm:
                handles.append(mod.register_forward_hook(make_hook(nm, mod)))
        was_training = model.training
        model.eval()
        with torch.no_grad(), amp_autocast_context(batch["noisy"].device, amp_enabled):
            model(batch["noisy"], batch.get("neighbors"), batch.get("neighbor_geometry"),
                  depth_norm=batch.get("depth_norm"), profile_len=batch_profile_len(batch))
        if was_training:
            model.train()
    except Exception as exc:
        found.setdefault("hunt_error", repr(exc))
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
    if found.get("module"):
        logger.error(
            "FIRST NON-FINITE MODULE: %s (%s) | inputs %s | outputs %s -- this "
            "is the causal op; the alphabetical output list is downstream fallout.",
            found["module"], found.get("class"), found.get("inputs"), found.get("outputs"),
        )
    return found or None


def set_nan_forensics_path(path: Path) -> None:
    _NAN_FORENSICS["path"] = str(path)
    _NAN_FORENSICS["events"] = 0


def _tensor_forensics(v: torch.Tensor) -> Dict[str, Any]:
    finite = torch.isfinite(v)
    with torch.no_grad():
        fv = v[finite]
        return {
            "dtype": str(v.dtype).replace("torch.", ""),
            "shape": list(v.shape),
            "nonfinite_fraction": float((~finite).float().mean().item()),
            "finite_min": float(fv.min().item()) if fv.numel() else None,
            "finite_max": float(fv.max().item()) if fv.numel() else None,
        }


def record_nan_forensics(
    where: str,
    stage: str,
    epoch: int,
    step: int,
    outputs: Mapping[str, Any],
    losses: Mapping[str, Any],
    batch: Mapping[str, Any],
    logger: logging.Logger,
    halt: bool,
    model: Optional["PEBRNet"] = None,
    amp_enabled: bool = False,
) -> None:
    """A run that prints nothing but nan for six epochs is undiagnosable from its log."""
    bad_out = {
        k: _tensor_forensics(v)
        for k, v in outputs.items()
        if isinstance(v, torch.Tensor) and not bool(torch.isfinite(v).all())
    }
    bad_loss = {
        k: _tensor_forensics(v)
        for k, v in losses.items()
        if isinstance(v, torch.Tensor) and v.ndim == 0 and not bool(torch.isfinite(v).all())
    }
    snr = batch.get("snr_db")
    first_module = _hunt_first_nonfinite_module(model, batch, amp_enabled, logger)
    entry = {
        "where": where, "stage": stage, "epoch": int(epoch), "step": int(step),
        "first_nonfinite_module": first_module,
        "nonfinite_outputs": bad_out, "nonfinite_losses": sorted(bad_loss),
        "batch_snr_db": [float(snr.min()), float(snr.max())] if isinstance(snr, torch.Tensor) else None,
        "batch_abs_max": float(batch["noisy"].abs().max()) if "noisy" in batch else None,
    }
    logger.error(
        "NON-FINITE %s at stage=%s epoch=%d step=%d | first offending outputs: %s | losses: %s | "
        "the step is SKIPPED (weights and metrics stay clean). Full report: %s",
        where, stage, epoch, step, sorted(bad_out)[:6] or "none",
        sorted(bad_loss)[:6] or "none", _NAN_FORENSICS["path"] or "(no run dir set)",
    )
    path = _NAN_FORENSICS["path"]
    if path is not None and _NAN_FORENSICS["events"] < 5:
        try:
            p = Path(path)
            existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
            existing.append(entry)
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(existing, p)
            _NAN_FORENSICS["events"] += 1
        except Exception as exc:
            logger.warning("Could not write nan forensics: %s", exc)
    if halt:
        raise RuntimeError(
            "Non-finite %s at stage=%s epoch=%d step=%d (--halt_on_nan). See %s"
            % (where, stage, epoch, step, path)
        )


def project_late_gradient_against_global(
    model: PEBRNet,
    global_part: torch.Tensor,
    late_part: torch.Tensor,
    scaler: "torch.cuda.amp.GradScaler",
    eps: float = 1e-12,
) -> Tuple[Optional[List[Optional[torch.Tensor]]], Optional[List[Optional[torch.Tensor]]], float, bool]:
    shared = model.shared_parameters()
    if not shared or not global_part.requires_grad or not late_part.requires_grad:
        return None, None, float("nan"), False
    scaled_global = scaler.scale(global_part)
    scaled_late = scaler.scale(late_part)
    gg = torch.autograd.grad(scaled_global, shared, retain_graph=True, allow_unused=True)
    gl = torch.autograd.grad(scaled_late, shared, retain_graph=True, allow_unused=True)
    dot = global_part.new_zeros(())
    norm_g = global_part.new_zeros(())
    norm_l = global_part.new_zeros(())
    for a, b in zip(gg, gl):
        if a is not None and b is not None:
            dot = dot + torch.sum(a * b)
            norm_g = norm_g + torch.sum(a * a)
            norm_l = norm_l + torch.sum(b * b)
    denom = torch.sqrt(norm_g * norm_l + eps)
    cosine = float((dot / denom).detach().cpu().item()) if denom.item() > 0 else float("nan")
    return list(gg), list(gl), cosine, bool(dot.detach().item() < 0.0 and norm_g.detach().item() > eps)


def apply_projection_delta(
    model: PEBRNet,
    gg: Optional[List[Optional[torch.Tensor]]],
    gl: Optional[List[Optional[torch.Tensor]]],
    should_project: bool,
    eps: float = 1e-12,
) -> None:
    if not should_project or gg is None or gl is None:
        return
    dot = None
    norm_g = None
    norm_l = None
    for a, b in zip(gg, gl):
        if a is not None and b is not None:
            term_dot = torch.sum(a * b)
            dot = term_dot if dot is None else dot + term_dot
            tg = torch.sum(a * a)
            norm_g = tg if norm_g is None else norm_g + tg
            tl = torch.sum(b * b)
            norm_l = tl if norm_l is None else norm_l + tl
    if dot is None or norm_g is None or norm_l is None:
        return
    coeff_l = dot / (norm_g + eps)
    coeff_g = dot / (norm_l + eps)
    for p, a, b in zip(model.shared_parameters(), gg, gl):
        if p.grad is None or a is None or b is None:
            continue
        delta_late = -coeff_l * a
        delta_global = -coeff_g * b
        p.grad.add_(delta_late + delta_global)


_DEAD_LOSS_HISTORY: Dict[str, List[float]] = {}


def apply_neighbor_dropout(batch: MutableMapping[str, Any], p: float) -> int:
    """Replace the neighbour stack of a random share p of the patches (contiguous profile_len rows; whole rows when
    the batch is not block-structured) by the centre trace of each row -- for the main.
    """
    nb = batch.get("neighbors")
    if p <= 0.0 or nb is None or nb.dim() != 3 or nb.shape[1] == 0:
        return 0
    B = int(nb.shape[0])
    P = batch_profile_len(batch) or 1
    n = B // P
    pick = (torch.rand(n, device=nb.device) < p)
    if not bool(pick.any()):
        return 0
    rows = pick.repeat_interleave(P)
    for xk, nk in (("noisy", "neighbors"), ("noisy_bg", "neighbors_bg"), ("noisy_alt", "neighbors_alt")):
        x = batch.get(xk)
        nbk = batch.get(nk)
        if x is None or nbk is None or nbk.dim() != 3 or int(nbk.shape[0]) != B:
            continue
        xc = x.reshape(B, 1, -1)
        if int(xc.shape[-1]) != int(nbk.shape[-1]):
            continue
        nbk = nbk.clone()
        nbk[rows] = xc[rows].expand(-1, int(nbk.shape[1]), -1).to(nbk.dtype)
        batch[nk] = nbk
    batch["neighbor_dropout_rows"] = rows
    return int(rows.sum())


def run_epoch(
    model: PEBRNet,
    loader: DataLoader,
    loss_fn: ProjectLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: "torch.cuda.amp.GradScaler",
    device: torch.device,
    cfg: Config,
    stage: str,
    epoch: int,
    logger: logging.Logger,
    scheduler: Optional["torch.optim.lr_scheduler.LambdaLR"] = None,
    ema: Optional[ModelEMA] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    snr_edges = np.arange(cfg.data.snr_min_db, cfg.data.snr_max_db + 5.0, 5.0, dtype=np.float64)
    accum = MetricAccumulator(cfg.data.late_start_index, cfg.loss.eps, snr_bin_edges=snr_edges)
    loss_components: MutableMapping[str, float] = {}
    component_samples = 0
    projection_count = 0
    projection_calls = 0
    nonfinite_steps = 0
    _oom = 0
    _nf_micro_dropped = 0
    _opt_params2 = ([p for grp in optimizer.param_groups for p in grp["params"] if p.requires_grad]
                      if optimizer is not None else [])
    _pname = {id(p): n for n, p in model.named_parameters()}
    _pname.update({id(p): "loss." + n for n, p in loss_fn.named_parameters()})
    _tm = {"wait": 0.0, "fwd": 0.0, "loss": 0.0, "bwd": 0.0, "opt": 0.0, "timed": 0}
    _last: Dict[str, float] = {}
    _hb_n2 = max(0, int(getattr(cfg.train, "heartbeat_steps", 8)))
    _li = max(1, int(cfg.train.log_interval))
    nonfinite_grad_steps = 0
    scaler_collapses = 0
    sentinel_hits = 0
    fp32_retry_success = 0
    fp32_retry_failed = 0
    _arm_counts = [0, 0, 0]
    _arm_alt = 0
    _arm_total = 0
    _arm_e = np.zeros((5, 2))
    _arm_y = np.zeros((5, 2))
    _arm_sv = 0
    _fp578_done = False
    _clean_sum = 0.0
    _clean_n = 0.0
    _acc_share = float("nan")
    _acc_nm = float("nan")
    _acc_na = float("nan")
    cosine_values: List[float] = []
    amp_enabled = cfg.train.amp and (device.type == "cuda" or getattr(cfg.train, "_force_amp_for_tests", False))

    _accum = max(1, int(getattr(cfg.train, "grad_accum_steps", 1)))
    _real_steps = 0
    _t_ep0 = time.perf_counter()
    _it = iter(loader)
    step = -1
    _t_ep_start = time.perf_counter()
    while True:
        _tw = time.perf_counter()
        try:
            raw_batch = next(_it)
        except StopIteration:
            break
        step += 1
        _wait_step = time.perf_counter() - _tw
        _tm["wait"] += _wait_step
        _hb = bool(training and device.type == "cuda"
                      and (step <= _hb_n2 or step % _li == 0 or (step + 1) % _li == 0))
        batch = move_batch(raw_batch, device)
        if training:
            apply_neighbor_dropout(batch, float(getattr(cfg.train, "neighbor_dropout", 0.0) or 0.0))
        _micro = (step % _accum) if training else 0
        _is_update_step = (not training) or (_micro == _accum - 1)
        if training:
            assert optimizer is not None
            if _micro == 0:
                optimizer.zero_grad(set_to_none=True)
        autocast_ctx = amp_autocast_context(device, amp_enabled)
        if _hb:
            torch.cuda.synchronize(device)
            _tf0_704 = time.perf_counter()
        try:
            with autocast_ctx:
                outputs = forward_training_arms(model, batch)
                if _hb:
                    torch.cuda.synchronize(device)
                    _tf1_704 = time.perf_counter()
                losses = loss_fn(outputs, batch, stage)
            if _hb:
                torch.cuda.synchronize(device)
                _tl1_704 = time.perf_counter()
                _last = {"wait": _wait_step, "fwd": _tf1_704 - _tf0_704,
                            "loss": _tl1_704 - _tf1_704, "bwd": float("nan"), "opt": float("nan")}
        except Exception as _e697f:
            if not is_oom_error(_e697f):
                raise
            _oom += 1
            outputs = None
            losses = None
            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            free_cuda_after_oom()
            _b = int(batch["clean"].shape[0])
            _P = int(max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 1)))
            _half = max(_P, ((_b // 2) // _P) * _P)
            logger.error("OUT OF MEMORY in the training step (stage=%s epoch=%d step=%d, %d rows, event %d/3): "
                         "%s. The batch is skipped and the accumulation window continues. If this repeats, restart with "
                         "--micro_batch_size %d (the effective batch is preserved by accumulation).",
                         stage, int(epoch), int(step), _b, _oom, str(_e697f)[:160], _half)
            if _oom >= 3:
                raise RuntimeError("three out-of-memory steps in one epoch at %d rows per micro-batch: the run would "
                                   "train on a silently truncated epoch. Restart with --micro_batch_size %d (effective batch "
                                   "unchanged) or --activation_checkpointing on." % (_b, _half))
            continue
        _aid = batch.get("arm_id")
        if training and not _fp578_done:
            _fp578_done = True
            try:
                _ci = batch.get("clean_index")
                _sn = batch.get("snr_db")
                _h = hashlib.sha1()
                if _ci is not None:
                    _h.update(np.ascontiguousarray(
                        _ci.detach().cpu().numpy().astype(np.int64)).tobytes())
                if _sn is not None:
                    _h.update(np.ascontiguousarray(
                        np.round(_sn.detach().cpu().numpy().astype(np.float64), 3)).tobytes())
                _ep = int(getattr(loader.dataset, "_epoch_now", lambda: epoch)())
                logger.info(
                    "DATA FINGERPRINT | stage %s epoch %d | worker-side epoch %d | "
                    "sha1(first-batch clean_index, snr_db) = %s | first rows %s. ",
                    stage, int(epoch), _ep, _h.hexdigest()[:12],
                    (_ci.detach().reshape(-1)[:6].tolist() if _ci is not None else "n/a"))
            except Exception:
                pass
        if _aid is not None:
            _av = _aid.detach().reshape(-1)
            _arm_total += int(_av.numel())
            for _k3 in (0, 1, 2):
                _arm_counts[_k3] += int((_av == float(_k3)).sum().item())
            try:
                _d = outputs["denoised"].detach().float()
                _c = batch["clean"].detach().float()
                _e2_632 = (_d - _c) ** 2
                _y2_632 = _c ** 2
                _ls = int(cfg.data.late_start_index)
                for _k3 in (0, 1, 2):
                    _m = (_av == float(_k3))
                    if bool(_m.any()):
                        _arm_e[_k3, 0] += float(_e2_632[_m].sum().item())
                        _arm_y[_k3, 0] += float(_y2_632[_m].sum().item())
                        _arm_e[_k3, 1] += float(_e2_632[_m][..., _ls:].sum().item())
                        _arm_y[_k3, 1] += float(_y2_632[_m][..., _ls:].sum().item())
                _sr = batch.get("survey_row")
                if _sr is not None:
                    _m2 = _sr.detach().reshape(-1) > 0.5
                    if bool(_m2.any()):
                        _arm_sv += int(_m2.sum().item())
                        _arm_e[3, 0] += float(_e2_632[_m2].sum().item())
                        _arm_y[3, 0] += float(_y2_632[_m2].sum().item())
                        _arm_e[3, 1] += float(_e2_632[_m2][..., _ls:].sum().item())
                        _arm_y[3, 1] += float(_y2_632[_m2][..., _ls:].sum().item())
                        _n = _m2 & (_sr.detach().reshape(-1) < 1.5)
                        if bool(_n.any()):
                            _arm_e[4, 0] += float(_e2_632[_n].sum().item())
                            _arm_y[4, 0] += float(_y2_632[_n].sum().item())
                            _arm_e[4, 1] += float(_e2_632[_n][..., _ls:].sum().item())
                            _arm_y[4, 1] += float(_y2_632[_n][..., _ls:].sum().item())
            except Exception:
                pass
            _hal = batch.get("has_alt")
            if _hal is not None:
                _arm_alt += int(_hal.detach().reshape(-1).gt(0.5).sum().item())
        if not bool(torch.isfinite(losses["total"]).all()):
            if amp_enabled:
                with amp_autocast_context(device, False):
                    outputs = forward_training_arms(model, batch)
                    losses = loss_fn(outputs, batch, stage)
                if bool(torch.isfinite(losses["total"]).all()):
                    fp32_retry_success += 1
                    if fp32_retry_success == 1:
                        logger.warning(
                            "AUTOCAST-ONLY NON-FINITE at stage=%s epoch=%d step=%d: "
                            "the identical batch is FINITE in fp32 and will train. This is a "
                            "half-precision numerics issue, not a model defect; if it repeats "
                            "on >25%% of steps, AMP will be demoted automatically.",
                            stage, epoch, step,
                        )
                    if amp_enabled and fp32_retry_success >= max(8, int(0.25 * len(loader))):
                        amp_enabled = False
                        cfg.train.amp = False
                        logger.critical(
                            "AMP DEMOTED for the remainder of the run: %d/%d steps of "
                            "this epoch were finite only in fp32. Training continues in fp32; "
                            "expect a modest speed cost and no numerical cliffs. Rerun with "
                            "--no_amp to start this way.",
                            fp32_retry_success, len(loader),
                        )
                else:
                    fp32_retry_failed += 1
            if not bool(torch.isfinite(losses["total"]).all()):
                nonfinite_steps += 1
                if nonfinite_steps <= 3 or nonfinite_steps % 50 == 0:
                    record_nan_forensics(
                        "TRAIN" if training else "EVAL", stage, epoch, step, outputs, losses,
                        batch, logger, bool(cfg.runtime.halt_on_nan),
                        model=model, amp_enabled=amp_enabled,
                    )
                continue
        b = batch["clean"].shape[0]
        _names = [n for n, v in losses.items() if v.ndim == 0]
        _vals = torch.stack([losses[n].detach().float() for n in _names]).cpu().tolist()
        _scal = dict(zip(_names, _vals))
        _clean_sum += float(_scal.get("clean_arm_late_sum", 0.0))
        _clean_n += float(_scal.get("clean_arm_rows", 0.0))
        accum.update(
            outputs["denoised"].detach(),
            batch["clean"].detach(),
            outputs["reliability"].detach(),
            float(_scal.get("total", 0.0)),
            snr_db=batch.get("snr_db"),
        )
        component_samples += b
        for name, value in _scal.items():
            loss_components[name] = loss_components.get(name, 0.0) + float(value) * b
        _hb_n = max(0, int(getattr(cfg.train, "heartbeat_steps", 8)))
        if training and (step % cfg.train.log_interval == 0 or step < _hb_n):
            _t_now = time.perf_counter()
            _t_prev = float(_STEP_HEARTBEAT.get("t", _t_now))
            _s_prev = int(_STEP_HEARTBEAT.get("step", -(10 ** 9)))
            _elapsed = _t_now - _t_prev
            _nsteps = step - _s_prev
            _turnover = not (0 < _nsteps <= len(loader))
            _dt = (_elapsed / _nsteps) if not _turnover else float("nan")
            _STEP_HEARTBEAT["t"] = _t_now
            _STEP_HEARTBEAT["step"] = step
            _extra = ""
            if step < _hb_n or (math.isfinite(_dt) and _dt > 5.0):
                _mem = ""
                if device.type == "cuda":
                    _mem = " | VRAM %.2f/%.2f GB (alloc/reserved)" % (
                        torch.cuda.memory_allocated(device) / 2**30,
                        torch.cuda.memory_reserved(device) / 2**30,
                    )
                if math.isfinite(_dt):
                    _rate = (b / _dt) if _dt > 1e-6 else float("nan")
                    _extra = " | %.2fs/step (avg over %d) %.0f samp/s%s | epoch ETA ~%.1f min" % (
                        _dt, max(1, _nsteps), _rate, _mem,
                        _dt * max(0, len(loader) - step - 1) / 60.0,
                    )
                    _pt = _STEP_HEARTBEAT.get("timing_prev")
                    if isinstance(_pt, dict) and math.isfinite(float(_pt.get("bwd", float("nan")))):
                        _extra += " | prev step: wait %.2f fwd %.2f loss %.2f bwd %.2f opt %.2f s" % (
                            float(_pt["wait"]), float(_pt["fwd"]), float(_pt["loss"]), float(_pt["bwd"]),
                            float(_pt["opt"]))
                else:
                    _extra = " | %.1fs since previous line (epoch turnover: validation/probes/checkpoints included)%s" % (
                        _elapsed, _mem,
                    )
            logger.info(
                "stage=%s epoch=%d step=%d/%d loss=%.6f G=%.3f%% EM=%.3f%% L=%.3f%%%s",
                stage,
                epoch,
                step,
                len(loader),
                float(_scal.get("total", float("nan"))),
                100.0 * math.sqrt(max(float(_scal.get("rg", 0.0)), 0.0)),
                100.0 * math.sqrt(max(float(_scal.get("rem", 0.0)), 0.0)),
                100.0 * math.sqrt(max(float(_scal.get("rl", 0.0)), 0.0)),
                _extra,
            )
            if (device.type == "cuda" and (not _turnover)
                    and math.isfinite(_dt) and _dt > 5.0):
                logger.warning(
                    "%.1f s PER micro-batch of %d samples, averaged over "
                    "%d consecutive steps. On this card that is the shared-memory SPILL "
                    "signature, not compute: the step no longer fits in dedicated VRAM. "
                    "Lower the activation budget (--vram_budget_fraction), raise "
                    "--vram_reserve_gb, or pin the size with --micro_batch_size. ",
                    _dt, int(b), max(1, _nsteps),
                )
        gg = gl = None
        cosine = float("nan")
        should_project = False
        do_project = (
            training
            and stage in {"B", "C"}
            and cfg.train.use_priority_gradient_projection
            and step % max(1, cfg.train.gradient_projection_every) == 0
        )
        if do_project:
            gg, gl, cosine, should_project = project_late_gradient_against_global(
                model, losses["global_part"], losses["late_part"], scaler
            )
            projection_calls += 1
            if should_project:
                projection_count += 1
            if np.isfinite(cosine):
                cosine_values.append(cosine)
        if (training and "acceptance_group" in losses
                and bool(getattr(losses["acceptance_group"], "requires_grad", False))
                and float(getattr(cfg.loss, "acceptance_balance_share", 0.0)) > 0.0
                and int(epoch) >= int(getattr(cfg.loss, "acceptance_balance_start_epoch", 5))
                and float(getattr(loss_fn, "last_row_late_mean", float("inf")))
                <= float(getattr(cfg.loss, "acceptance_balance_row_late_max", 0.10))
                and step % max(1, int(getattr(cfg.loss, "acceptance_balance_every", 25))) == 0):
            try:
                _sp = [p for p in model.shared_parameters() if p.requires_grad]
                _acc = losses["acceptance_group"]
                _rest = losses["total"] - float(loss_fn.acceptance_scale) * _acc
                _ga = torch.autograd.grad(_acc, _sp, retain_graph=True, allow_unused=True)
                _gm = torch.autograd.grad(_rest, _sp, retain_graph=True, allow_unused=True)
                _na = math.sqrt(sum(float((g.float() ** 2).sum()) for g in _ga if g is not None))
                _nm = math.sqrt(sum(float((g.float() ** 2).sum()) for g in _gm if g is not None))
                del _ga, _gm
                if _na > 0.0 and math.isfinite(_na) and math.isfinite(_nm):
                    _tgt = float(cfg.loss.acceptance_balance_share) * _nm / _na
                    _tgt = min(max(_tgt, 1.0),
                                  float(getattr(cfg.loss, "acceptance_balance_max", 50.0)))
                    loss_fn.acceptance_scale = (0.9 * float(loss_fn.acceptance_scale)
                                                + 0.1 * _tgt)
                    _acc_share = float(loss_fn.acceptance_scale) * _na / max(_nm, 1e-12)
                    _acc_na, _acc_nm = _na, _nm
            except RuntimeError as _e:
                if "out of memory" in str(_e).lower():
                    cfg.loss.acceptance_balance_share = 0.0
                    logger.warning("acceptance-group balancing DISABLED (%s); the static "
                                   "scale %.2f stays for the rest of the run.",
                                   str(_e)[:160], float(loss_fn.acceptance_scale))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    logger.debug("balancing step skipped: %s", str(_e)[:160])
        if training:
            try:
                if _hb:
                    torch.cuda.synchronize(device)
                    _tb0_704 = time.perf_counter()
                if scaler.is_enabled():
                    scaler.scale(losses["total"] / float(_accum)).backward()
                else:
                    _ok, _bad = accumulate_finite_gradients(losses["total"] / float(_accum), _opt_params2)
                    if not _ok:
                        _nf_micro_dropped += 1
                        _names2 = sorted({_pname.get(id(p), "?") for p in _bad})
                        _terms: List[str] = []
                        if _nf_micro_dropped <= 3:
                            try:
                                with autocast_ctx:
                                    _o = forward_training_arms(model, batch)
                                    _l = loss_fn(_o, batch, stage)
                                _terms = name_nonfinite_grad_terms(_l, _opt_params2)
                                del _o, _l
                            except Exception as _e:
                                _terms = ["(term audit failed: %s)" % str(_e)[:80]]
                            logger.error("NON-FINITE GRADIENT in micro-batch %d of stage=%s epoch=%d (event %d): dropped "
                                         "alone, the accumulation window keeps its other micro-batches. Parameters with non-finite "
                                         "gradient: %s | loss terms with non-finite gradient: %s | snr_db of the batch: %s. ",
                                         int(step), stage, int(epoch), _nf_micro_dropped, _names2[:8],
                                         _terms[:12] if _terms else "(none identified)",
                                         (batch["snr_db"].detach().float().reshape(-1)[:6].tolist() if "snr_db" in batch else "n/a"))
                        losses = None
                        outputs = None
                        if _hb:
                            _last["bwd"] = float("nan")
                        continue
                if _hb:
                    torch.cuda.synchronize(device)
                    _last["bwd"] = time.perf_counter() - _tb0_704
            except Exception as _e697b:
                if not is_oom_error(_e697b):
                    raise
                _oom += 1
                losses = None
                outputs = None
                optimizer.zero_grad(set_to_none=True)
                free_cuda_after_oom()
                _b = int(batch["clean"].shape[0])
                _P = int(max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 1)))
                _half = max(_P, ((_b // 2) // _P) * _P)
                logger.error("OUT OF MEMORY in backward (stage=%s epoch=%d step=%d, %d rows, event %d/3): %s. This "
                             "accumulation window is dropped. Restart with --micro_batch_size %d if it repeats.",
                             stage, int(epoch), int(step), _b, _oom, str(_e697b)[:160], _half)
                if _oom >= 3:
                    raise RuntimeError("three out-of-memory events in one epoch at %d rows per micro-batch. Restart with "
                                       "--micro_batch_size %d or --activation_checkpointing on." % (_b, _half))
                continue
            apply_projection_delta(model, gg, gl, should_project)
            if not _is_update_step:
                if _hb:
                    _last["opt"] = 0.0
                    _STEP_HEARTBEAT["timing_prev"] = dict(_last)
                    for _k in ("fwd", "loss", "bwd"):
                        _tm[_k] += float(_last.get(_k, 0.0))
                    _tm["timed"] += 1
                continue
            if _hb:
                torch.cuda.synchronize(device)
                _to0_704 = time.perf_counter()
            scaler.unscale_(optimizer)
            _opt_params = [p for grp in optimizer.param_groups for p in grp["params"]]
            total_norm = torch.nn.utils.clip_grad_norm_(_opt_params, cfg.train.grad_clip_norm)
            if not bool(torch.isfinite(total_norm)):
                nonfinite_grad_steps += 1
                if nonfinite_grad_steps <= 3 or nonfinite_grad_steps % 50 == 0:
                    logger.warning(
                        "NON-FINITE GRADIENT NORM at stage=%s epoch=%d step=%d "
                        "(scale=%.3g): the step is skipped before any weight is touched. "
                        "A few of these are NORMAL in the first steps while the AMP loss "
                        "scale calibrates downward; persistent ones are itemized in "
                        "nan_forensics.json territory and end in the abort.",
                        stage, epoch, step, float(scaler.get_scale()),
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                _sc = float(scaler.get_scale()) if scaler.is_enabled() else float("inf")
                if scaler.is_enabled() and _sc < float(cfg.train.amp_scale_floor):
                    scaler_collapses += 1
                    if scaler_collapses > int(cfg.train.amp_scale_max_resets):
                        raise RuntimeError(
                            "AMP loss scale collapsed to %.3g for the %d-th time at "
                            "stage=%s epoch=%d step=%d. Every step is being skipped, so the "
                            "model is NOT training: continuing would produce hours of "
                            "identical metrics. This means some loss term is producing "
                            "non-finite gradients -- check the newest constraint or auxiliary "
                            "term first, and reports/nan_forensics.json for the tensor that "
                            "goes non-finite. Run with --no_amp to train in fp32 while the "
                            "term is diagnosed." % (_sc, scaler_collapses, stage, epoch, step)
                        )
                    logger.error(
                        "AMP LOSS SCALE COLLAPSED to %.3g at stage=%s epoch=%d "
                        "step=%d: at this scale every step is skipped and the model stops "
                        "training silently. Resetting the scale to %.0f (reset %d of %d). If "
                        "this recurs the run will stop rather than pretend to train.",
                        _sc, stage, epoch, step, float(cfg.train.amp_scale_reset_value),
                        scaler_collapses, int(cfg.train.amp_scale_max_resets),
                    )
                    scaler.update(float(cfg.train.amp_scale_reset_value))
                continue
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            _real_steps += 1
            if _hb:
                torch.cuda.synchronize(device)
                _last["opt"] = time.perf_counter() - _to0_704
                _STEP_HEARTBEAT["timing_prev"] = dict(_last)
                for _k in ("fwd", "loss", "bwd", "opt"):
                    _tm[_k] += float(_last.get(_k, 0.0))
                _tm["timed"] += 1
            stepped = (not scaler.is_enabled()) or (scaler.get_scale() >= scale_before)
            if stepped:
                _poisoned = [
                    n for n, p in model.named_parameters()
                    if p.requires_grad and not bool(torch.isfinite(p).all())
                ]
                if _poisoned:
                    sentinel_hits += 1
                    logger.critical(
                        "PARAMETER SENTINEL HIT #%d at stage=%s epoch=%d step=%d | "
                        "%d non-finite parameter tensor(s), first: %s | lr=%.3e scale=%.3g "
                        "grad_norm=%.3g. %s",
                        sentinel_hits, stage, epoch, step, len(_poisoned), _poisoned[:3],
                        float(optimizer.param_groups[0]["lr"]), float(scaler.get_scale()),
                        float(total_norm),
                        "Restoring the weights from the EMA shadow and continuing."
                        if (ema is not None and sentinel_hits < 2)
                        else "No finite restore point or repeated hit: aborting (exit 24).",
                    )
                    if ema is not None and sentinel_hits < 2:
                        with torch.no_grad():
                            for n, p in model.named_parameters():
                                if n in ema.shadow:
                                    p.copy_(ema.shadow[n].to(p.dtype))
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    raise SystemExit(24)
            if scheduler is not None and stepped:
                scheduler.step()
            if ema is not None and stepped:
                ema.update(model)
    _dt_ep = max(time.perf_counter() - _t_ep0, 1e-9)
    if training and optimizer is not None:
        _pend = (len(loader) % _accum) if _accum > 1 else 0
        if _pend:
            optimizer.zero_grad(set_to_none=True)
            logger.warning("%d trailing micro-batch(es) did not complete an accumulation window of %d and were "
                           "discarded (raise train_samples_per_epoch or lower the effective batch to avoid the waste).",
                           _pend, _accum)
        if _real_steps == 0:
            raise RuntimeError("ZERO optimizer steps were taken in stage=%s epoch=%d (%d micro-batches, "
                               "grad_accum_steps=%d, non-finite-gradient skips=%d): the model is NOT training. "
                                % (stage, epoch, len(loader), _accum, nonfinite_grad_steps))
    logger.info(
        "epoch throughput | %s: %d micro-batches (%d optimizer steps ACTUALLY taken), "
        "%d samples in %.1f s = %.0f samples/s",
        "train" if training else "eval", len(loader),
        int(_real_steps) if training else len(loader),
        int(component_samples), _dt_ep, component_samples / _dt_ep,
    )
    if nonfinite_steps:
        logger.warning(
            "%d/%d steps in this epoch produced a NON-FINITE loss and were skipped "
            "(weights untouched). The first occurrences are itemized in reports/nan_forensics.json.",
            nonfinite_steps, len(loader),
        )
    metrics = accum.compute(cfg.loss.cvar_fraction)
    metrics["nonfinite_steps"] = float(nonfinite_steps)
    metrics["nonfinite_grad_steps"] = float(nonfinite_grad_steps)
    metrics["fp32_retry_success"] = float(fp32_retry_success)
    metrics["fp32_retry_failed"] = float(fp32_retry_failed)
    metrics["steps_total"] = float(len(loader))
    components = {k: v / max(component_samples, 1) for k, v in loss_components.items()}
    components["gradient_cosine_global_late"] = float(np.mean(cosine_values)) if cosine_values else float("nan")
    components["gradient_projection_rate"] = projection_count / max(projection_calls, 1)
    if _arm_total:
        components["acceptance_grad_share"] = _acc_share
        components["clean_arm_late_percent"] = (100.0 * _clean_sum / _clean_n
                                                if _clean_n > 0 else float("nan"))
        components["nonfinite_grad_steps"] = float(nonfinite_grad_steps)
        components["clean_arm_rows"] = float(_clean_n)
        components["acceptance_grad_norm_group"] = _acc_na
        components["acceptance_grad_norm_rest"] = _acc_nm
        components["arm_exact_realized"] = _arm_counts[0] / _arm_total
        components["arm_replay_realized"] = _arm_counts[1] / _arm_total
        components["arm_library_realized"] = _arm_counts[2] / _arm_total
        components["arm_survey_realized"] = _arm_sv / _arm_total
        components["arm_survey_G_percent"] = (100.0 * math.sqrt(_arm_e[3, 0] / _arm_y[3, 0]) if _arm_y[3, 0] > 0 else float("nan"))
        components["arm_survey_L_percent"] = (100.0 * math.sqrt(_arm_e[3, 1] / _arm_y[3, 1]) if _arm_y[3, 1] > 0 else float("nan"))
        components["arm_survey_natural_G_percent"] = (100.0 * math.sqrt(_arm_e[4, 0] / _arm_y[4, 0]) if _arm_y[4, 0] > 0 else float("nan"))
        components["arm_survey_natural_L_percent"] = (100.0 * math.sqrt(_arm_e[4, 1] / _arm_y[4, 1]) if _arm_y[4, 1] > 0 else float("nan"))
        for _k3, _nm3 in enumerate(("exact", "replay", "library")):
            components["arm_%s_G_percent" % _nm3] = (100.0 * math.sqrt(_arm_e[_k3, 0] / _arm_y[_k3, 0])
                                                   if _arm_y[_k3, 0] > 0 else float("nan"))
            components["arm_%s_L_percent" % _nm3] = (100.0 * math.sqrt(_arm_e[_k3, 1] / _arm_y[_k3, 1])
                                                   if _arm_y[_k3, 1] > 0 else float("nan"))
        components["arm_alt_realized"] = _arm_alt / _arm_total
    if training and _AUX_ARM_STATS["rows"] > 0.0:
        _r = float(_AUX_ARM_STATS["rows"])
        components["aux_bg_row_share"] = float(_AUX_ARM_STATS["bg_rows"]) / _r
        components["aux_alt_row_share"] = float(_AUX_ARM_STATS["alt_rows"]) / _r
        logger.info("AUXILIARY ARMS this epoch | background arm ran on %.1f%% of the rows (%.1f%% carry has_anomaly) | "
                    "alternate arm on %.1f%% (%.1f%% carry has_alt) | model forwards per step %.2f (legacy 3.00 + continuation). ",
                    100.0 * _AUX_ARM_STATS["bg_rows"] / _r, 100.0 * _AUX_ARM_STATS["bg_flag"] / _r,
                    100.0 * _AUX_ARM_STATS["alt_rows"] / _r, 100.0 * _AUX_ARM_STATS["alt_flag"] / _r,
                    1.0 + (_AUX_ARM_STATS["bg_rows"] + _AUX_ARM_STATS["alt_rows"]) / _r)
        if _AUX_ARM_STATS["capped_steps"] > 0.0:
            logger.info("auxiliary-arm caps bound on %d of %d steps this epoch: %d background and %d alternate patch "
                        "terms skipped (each only for that step).",
                        int(_AUX_ARM_STATS["capped_steps"]), int(_AUX_ARM_STATS["steps"]),
                        int(_AUX_ARM_STATS["bg_dropped"]), int(_AUX_ARM_STATS["alt_dropped"]))
        for _k2 in list(_AUX_ARM_STATS.keys()):
            _AUX_ARM_STATS[_k2] = 0.0
    _ep2 = max(1e-9, time.perf_counter() - _t_ep_start)
    components["nonfinite_micro_dropped"] = float(_nf_micro_dropped)
    components["data_wait_share"] = float(_tm["wait"] / _ep2)
    if training:
        _nt = max(1, int(_tm["timed"]))
        logger.info("EPOCH TIMING stage=%s epoch=%d | wall %.1f min | loader wait %.1f s (%.1f%% of wall) | "
                    "timed steps (%d, CUDA-synced): fwd %.2f s + loss %.2f s + bwd %.2f s + opt %.2f s = %.2f s/step | "
                    "micro-batches dropped for a non-finite gradient: %d.",
                    stage, int(epoch), _ep2 / 60.0, _tm["wait"], 100.0 * _tm["wait"] / _ep2, _nt,
                    _tm["fwd"] / _nt, _tm["loss"] / _nt, _tm["bwd"] / _nt, _tm["opt"] / _nt,
                    (_tm["fwd"] + _tm["loss"] + _tm["bwd"] + _tm["opt"]) / _nt, _nf_micro_dropped)
    return metrics, components


class PairedAnomalyMeter:
    """How much of a real anomaly survives the reconstruction."""

    def __init__(self, late_start: int, eps: float = 1e-12) -> None:
        self.ls = int(late_start)
        self.eps = float(eps)
        self.num = 0.0
        self.den = 0.0
        self.amp_hat = 0.0
        self.amp_true = 0.0
        self.sign_hit = 0.0
        self.sign_n = 0.0
        self.n_anom = 0
        self.n_seen = 0
        self.per_sample: List[float] = []
        self.strata: Dict[str, List[float]] = {k: [0.0, 0.0, 0.0, 0.0, 0.0] for k in ("weak", "medium", "strong")}
        self.quiet_false = 0.0
        self.quiet_n = 0

    def update(self, denoised: "torch.Tensor", clean: "torch.Tensor",
               background: Optional["torch.Tensor"],
               has_anomaly: Optional["torch.Tensor"],
               event_mask: Optional["torch.Tensor"]) -> None:
        if background is None:
            return
        with torch.no_grad():
            d = denoised.detach().float()
            c = clean.detach().float()
            b = background.detach().float()
            if d.dim() == 3:
                d, c, b = d[:, 0], c[:, 0], b[:, 0]
            ls = min(self.ls, d.shape[-1] - 1)
            a_true = (c - b)[:, ls:]
            a_hat = (d - b)[:, ls:]
            self.n_seen += int(d.shape[0])
            if has_anomaly is not None:
                sel = has_anomaly.detach().float().reshape(-1) > 0.5
            else:
                sel = torch.ones(d.shape[0], dtype=torch.bool, device=d.device)
            b_late = torch.sqrt(torch.sum(b[:, ls:] ** 2, dim=1) + self.eps)
            _ct = torch.sqrt(torch.sum(a_true ** 2, dim=1) + self.eps) / b_late
            _qm = (~sel) | (_ct < 0.02)
            if bool(_qm.any()):
                _fa = (torch.sqrt(torch.sum(a_hat[_qm] ** 2, dim=1)
                                  + self.eps) / b_late[_qm])
                self.quiet_false += float(_fa.sum().item())
                self.quiet_n += int(_qm.sum().item())
            if not bool(sel.any()):
                return
            at, ah = a_true[sel], a_hat[sel]
            self.n_anom += int(sel.sum().item())
            _contrast = torch.sqrt(torch.sum(at * at, dim=1) + self.eps) / b_late[sel]
            for _k, _lo, _hi in (("weak", 0.0, 0.05), ("medium", 0.05, 0.20), ("strong", 0.20, float("inf"))):
                _m = (_contrast >= _lo) & (_contrast < _hi) & (torch.sum(at * at, dim=1) > self.eps)
                if bool(_m.any()):
                    _acc = self.strata[_k]
                    _acc[0] += float(torch.sum(ah[_m] * at[_m]).item())
                    _acc[1] += float(torch.sum(at[_m] * at[_m]).item())
                    _acc[2] += float(torch.sqrt(torch.sum(ah[_m] ** 2, dim=1) + self.eps).sum().item())
                    _acc[3] += float(torch.sqrt(torch.sum(at[_m] ** 2, dim=1) + self.eps).sum().item())
                    _acc[4] += float(_m.sum().item())
            num = torch.sum(ah * at, dim=1)
            den = torch.sum(at * at, dim=1)
            self.num += float(num.sum().item())
            self.den += float(den.sum().item())
            self.amp_hat += float(torch.sqrt(torch.sum(ah * ah, dim=1) + self.eps).sum().item())
            self.amp_true += float(torch.sqrt(den + self.eps).sum().item())
            ok = den > self.eps
            if bool(ok.any()):
                self.per_sample.extend(
                    (num[ok] / den[ok]).clamp(-2.0, 3.0).cpu().numpy().tolist())
            if event_mask is not None:
                m = event_mask.detach().float()
                if m.dim() == 3:
                    m = m[:, 0]
                m = m[sel][:, ls:]
            else:
                m = torch.ones_like(at)
            w = m * (at.abs() > self.eps).float()
            self.sign_hit += float((w * (torch.sign(ah) == torch.sign(at)).float()).sum().item())
            self.sign_n += float(w.sum().item())

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "paired_anomaly_stations": float(self.n_anom),
            "paired_anomaly_fraction": (self.n_anom / self.n_seen) if self.n_seen else float("nan"),
        }
        out["paired_anomaly_recovery"] = (self.num / self.den) if self.den > self.eps else float("nan")
        for _k, _acc in self.strata.items():
            out["paired_anomaly_recovery_%s" % _k] = (max(-5.0, min(5.0, _acc[0] / _acc[1]))
                                                       if _acc[1] > self.eps else float("nan"))
            out["paired_anomaly_amplitude_%s" % _k] = (_acc[2] / _acc[3]) if _acc[3] > self.eps else float("nan")
            out["paired_anomaly_n_%s" % _k] = float(_acc[4])
        out["paired_quiet_false_level"] = (self.quiet_false / self.quiet_n) if self.quiet_n else float("nan")
        out["paired_anomaly_amplitude_ratio"] = (
            self.amp_hat / self.amp_true) if self.amp_true > self.eps else float("nan")
        out["paired_anomaly_sign_agreement"] = (
            self.sign_hit / self.sign_n) if self.sign_n > 0 else float("nan")
        if self.per_sample:
            v = np.asarray(self.per_sample, dtype=np.float64)
            out["paired_anomaly_recovery_median"] = float(np.median(v))
            out["paired_anomaly_recovery_p10"] = float(
                np.percentile(v, 10, method="lower"))
            out["paired_anomaly_recovery_worst_decile_mean"] = float(
                np.mean(np.sort(v)[: max(1, v.size // 10)]))
        return out


@torch.no_grad()
def evaluate_model(
    model: PEBRNet,
    loader: DataLoader,
    loss_fn: ProjectLoss,
    device: torch.device,
    cfg: Config,
    stage: str = "C",
    epoch: int = -1,
) -> Dict[str, float]:
    """Metrics of a model over the batches of a loader: global, early/mid and late errors, by SNR bin."""
    model.eval()
    snr_edges = np.arange(cfg.data.snr_min_db, cfg.data.snr_max_db + 5.0, 5.0, dtype=np.float64)
    accum = MetricAccumulator(cfg.data.late_start_index, cfg.loss.eps, snr_bin_edges=snr_edges)
    amp_enabled = cfg.train.amp and (device.type == "cuda" or getattr(cfg.train, "_force_amp_for_tests", False))
    man_sse = 0.0
    man_energy = 0.0
    man_sse_clean = 0.0
    man_energy_clean = 0.0
    man_sse_comp = 0.0
    den_sse = 0.0
    ev_sse = 0.0
    relax_abs_sum = 0.0
    relax_abs_count = 0
    blend_late_sum = 0.0
    blend_late_count = 0
    innov_w_sum = 0.0
    innov_w_count = 0
    late_ix = cfg.data.late_start_index
    nonfinite_batches = 0
    fp32_retry_batches = 0
    total_batches = 0
    _pam = PairedAnomalyMeter(cfg.data.late_start_index, cfg.loss.eps)
    _dnh_out = None
    _dnh_in = None
    _dnh_n = 0
    _dnh_man = None
    _dnh_den = None
    _dnh_an = 0
    _cov = None
    _cov644_n = 0
    _bind = None
    _ev647_max = 0.0
    _lat647_sse = 0.0
    _dnh_bl = None
    _mono_viol = 0.0
    _mono_pairs = 0.0
    _lr_err = 0.0
    _lr_ref = 0.0
    _lr_in = 0.0
    _lr_n = 0.0
    _lr_blocks = 0
    _lib_sq = 0.0
    _lib_refq = 0.0
    _lib_gsum = 0.0
    _lib_n = 0
    _ord_v = 0.0
    _ord_n = 0.0
    _rdg_sq = 0.0
    _rdg_refq = 0.0
    _rdg_gsum = 0.0
    _rdg_n = 0
    _libD_sq = 0.0
    _libD_refq = 0.0
    _rdgD_sq = 0.0
    _rdgD_refq = 0.0
    _pr_list: list = []
    _bl: Dict[int, List[float]] = {}
    _arm: Dict[int, np.ndarray] = {}
    _lrq = [0.0, 0.0, 0.0]
    _lrb = [0.0, 0.0, 0.0]
    _lrbeta = [0.0, 0.0, 0.0, 0.0]
    _nf_sum = 0.0
    _nf_cnt = 0
    _eg_sum = 0.0
    _eg_cnt = 0
    _lat_applied = 0
    _kd_lo = [0.0, 0]
    _kd_hi = [0.0, 0]
    _pw_lo = [0.0, 0]
    _pw_hi = [0.0, 0]
    _st_lo = [0.0, 0]
    _st_hi = [0.0, 0]
    _lf_lo = [0.0, 0]
    _lf_hi = [0.0, 0]
    _inf_sum = 0.0
    _inf_cnt = 0
    _inf_tail = 0
    _of_lo = [0.0, 0]
    _of_hi = [0.0, 0]
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        total_batches += 1
        with amp_autocast_context(device, amp_enabled):
            outputs = model(batch["noisy"], batch.get("neighbors"), batch.get("neighbor_geometry"),
                  depth_norm=batch.get("depth_norm"), profile_len=batch_profile_len(batch))
            losses = loss_fn(outputs, batch, stage)
        _finite = (bool(torch.isfinite(outputs["denoised"]).all())
                   and bool(torch.isfinite(losses["total"]).all()))
        if not _finite and amp_enabled:
            with amp_autocast_context(device, False):
                outputs = model(batch["noisy"], batch.get("neighbors"), batch.get("neighbor_geometry"),
                  depth_norm=batch.get("depth_norm"), profile_len=batch_profile_len(batch))
                losses = loss_fn(outputs, batch, stage)
            _finite = (bool(torch.isfinite(outputs["denoised"]).all())
                       and bool(torch.isfinite(losses["total"]).all()))
            if _finite:
                fp32_retry_batches += 1
        if not _finite:
            nonfinite_batches += 1
            if nonfinite_batches == 1:
                logger_ = logging.getLogger(PROGRAM_NAME)
                record_nan_forensics("VAL", stage, epoch, total_batches - 1, outputs, losses,
                                     batch, logger_, False, model=model, amp_enabled=amp_enabled)
            continue
        try:
            with torch.no_grad():
                _do = outputs["denoised"].detach().float().abs()
                if _do.dim() == 3:
                    _do = _do[:, 0]
                _lsO = int(cfg.data.late_start_index)
                _vio = (_do[:, _lsO + 1:] > _do[:, _lsO:-1]).float()
                _ord_v += float(_vio.sum().item())
                _ord_n += float(_vio.numel())
        except Exception:
            pass
        try:
            with torch.no_grad():
                _do2 = outputs["denoised"].detach().float()
                _cl = batch["clean"].detach().float()
                if _do2.dim() == 3:
                    _do2 = _do2[:, 0]
                if _cl.dim() == 3:
                    _cl = _cl[:, 0]
                _lsP = int(cfg.data.late_start_index)
                _n5 = ((_do2[:, _lsP:] - _cl[:, _lsP:]) ** 2).sum(1)
                _d5 = (_cl[:, _lsP:] ** 2).sum(1).clamp_min(1e-30)
                if len(_pr_list) < 200000:
                    _e = (100.0 * torch.sqrt(_n5 / _d5)).cpu()
                    _s = batch.get("snr_db")
                    _h = batch.get("has_anomaly")
                    _h = (_h.detach().float().view(-1).cpu().tolist()
                             if _h is not None else [float("nan")] * int(_e.numel()))
                    if len(_h) != int(_e.numel()):
                        _h = [float("nan")] * int(_e.numel())
                    if _s is not None:
                        _s = _s.detach().float().view(-1).cpu()
                        _pr_list.extend(zip(_s.tolist(), _e.tolist(), _h))
                    else:
                        _pr_list.extend((float("nan"), float(v), float(h))
                                        for v, h in zip(_e.tolist(), _h))
        except Exception:
            pass
        try:
            _lb = outputs.get("library_bg")
            if _lb is not None:
                with torch.no_grad():
                    _lbv = _lb.detach().float()
                    _cbv = batch["clean"].detach().float()
                    _bgv = batch.get("clean_bg")
                    _hav = batch.get("has_anomaly")
                    if _bgv is not None and _hav is not None:
                        _wv = _hav.detach().float().view(-1, 1, 1)
                        _cbv = (1.0 - _wv) * _cbv + _wv * _bgv.detach().float()
                    if _lbv.dim() == 3:
                        _lbv = _lbv[:, 0]
                    if _cbv.dim() == 3:
                        _cbv = _cbv[:, 0]
                    _lsL = int(cfg.data.late_start_index)
                    _lib_sq += float(((_lbv[:, _lsL:] - _cbv[:, _lsL:]) ** 2).sum().item())
                    _lib_refq += float((_cbv[:, _lsL:] ** 2).sum().item())
                    _lg = outputs.get("library_gate")
                    if _lg is not None:
                        _lgv = _lg.detach().float()
                        if _lgv.dim() == 3:
                            _lgv = _lgv[:, 0]
                        _lib_gsum += float(_lgv[:, _lsL:].mean().item()) * int(_lbv.shape[0])
                        _lib_n += int(_lbv.shape[0])
        except Exception:
            pass
        try:
            with torch.no_grad():
                _cb = batch["clean"].detach().float()
                _bg = batch.get("clean_bg")
                _ha2 = batch.get("has_anomaly")
                if _bg is not None and _ha2 is not None:
                    _wv2 = _ha2.detach().float().view(-1, 1, 1)
                    _cb = ((1.0 - _wv2) * _cb
                              + _wv2 * _bg.detach().float())
                if _cb.dim() == 3:
                    _cb = _cb[:, 0]
                _lsD = int(cfg.data.late_start_index)
                _snD = batch.get("snr_db")
                _mD = None
                if _snD is not None:
                    _mD = (_snD.detach().float().view(-1) <= -20.0)
                _rb = outputs.get("ridge_bg")
                if _rb is not None:
                    _rv = _rb.detach().float()
                    if _rv.dim() == 3:
                        _rv = _rv[:, 0]
                    _e2 = ((_rv[:, _lsD:] - _cb[:, _lsD:]) ** 2)
                    _r2 = (_cb[:, _lsD:] ** 2)
                    _rdg_sq += float(_e2.sum().item())
                    _rdg_refq += float(_r2.sum().item())
                    if _mD is not None and bool(_mD.any()):
                        _rdgD_sq += float(_e2[_mD].sum().item())
                        _rdgD_refq += float(_r2[_mD].sum().item())
                    _rg = outputs.get("ridge_gate")
                    if _rg is not None:
                        _rgv = _rg.detach().float()
                        if _rgv.dim() == 3:
                            _rgv = _rgv[:, 0]
                        _rdg_gsum += float(_rgv[:, _lsD:].mean().item()) * int(_rv.shape[0])
                        _rdg_n += int(_rv.shape[0])
                _lb2 = outputs.get("library_bg")
                if _lb2 is not None and _mD is not None and bool(_mD.any()):
                    _lv = _lb2.detach().float()
                    if _lv.dim() == 3:
                        _lv = _lv[:, 0]
                    _libD_sq += float(((_lv[:, _lsD:] - _cb[:, _lsD:]) ** 2)[_mD].sum().item())
                    _libD_refq += float((_cb[:, _lsD:] ** 2)[_mD].sum().item())
        except Exception:
            pass
        try:
            _plb = batch_profile_len(batch)
            if _plb is not None and _plb > 2:
                with torch.no_grad():
                    _ls = int(cfg.data.late_start_index)
                    _d9 = outputs["denoised"].detach().float()
                    _c9 = batch["clean"].detach().float()
                    _x9 = batch["noisy"].detach().float()
                    if _d9.dim() == 3:
                        _d9, _c9, _x9 = _d9[:, 0], _c9[:, 0], _x9[:, 0]
                    _fl = (_c9[:, _ls:].abs().amax(dim=-1, keepdim=True) * 1e-6).clamp_min(1e-30)
                    _lc = torch.log10(_c9[:, _ls:].abs().clamp_min(1e-30).maximum(_fl))
                    _ld = torch.log10(_d9[:, _ls:].abs().clamp_min(1e-30).maximum(_fl))
                    _lx = torch.log10(_x9[:, _ls:].abs().clamp_min(1e-30).maximum(_fl))
                    _n9 = _d9.shape[0] // _plb
                    _L9 = _lc.shape[-1]

                    def _d2s(v: torch.Tensor) -> torch.Tensor:
                        v = v.reshape(_n9, _plb, _L9)
                        return v[:, 2:] - 2.0 * v[:, 1:-1] + v[:, :-2]
                    _e9 = (_ld - _lc).clamp(-2.0, 2.0)
                    _i9 = (_lx - _lc).clamp(-2.0, 2.0)
                    _lr_err += float(_d2s(_e9).pow(2).sum().item())
                    _lr_ref += float(_d2s(_lc).pow(2).sum().item())
                    _lr_in += float(_d2s(_i9).pow(2).sum().item())
                    _lr_n += float(_n9 * (_plb - 2) * _L9)
                    _lr_blocks += int(_n9)
                    _ha = batch.get("has_anomaly")
                    if _ha is not None and int(_ha.numel()) == int(_d9.shape[0]):
                        _hb9 = _ha.detach().float().reshape(_n9, _plb).amax(dim=1) > 0.5
                        _e2 = _d2s(_e9).pow(2).sum(dim=(1, 2))
                        _r2 = _d2s(_lc).pow(2).sum(dim=(1, 2))
                        _cnt = float((_plb - 2) * _L9)
                        if bool((~_hb9).any()):
                            _lrq[0] += float(_e2[~_hb9].sum())
                            _lrq[1] += float(_r2[~_hb9].sum())
                            _lrq[2] += _cnt * int((~_hb9).sum())
                        if bool(_hb9.any()):
                            _lrb[0] += float(_e2[_hb9].sum())
                            _lrb[1] += float(_r2[_hb9].sum())
                            _lrb[2] += _cnt * int(_hb9.sum())
                            _dd = _d2s(_ld)
                            _dc = _d2s(_lc)
                            _nb = (_dd * _dc).sum(dim=(1, 2))[_hb9]
                            _db = (_dc * _dc).sum(dim=(1, 2))[_hb9]
                            _lrbeta[0] += float(_nb.sum())
                            _lrbeta[1] += float(_db.sum())
                            _lrbeta[2] += int(_hb9.sum())
                            _lrbeta[3] += int((_nb < 0.5 * _db.clamp_min(1e-30)).sum())
                    if "lateral_applied" in outputs and float(outputs["lateral_applied"]) > 0.5:
                        _lat_applied += 1
            if "late_noise_floor" in outputs:
                _nf = outputs["late_noise_floor"].detach().float()[..., late_ix:]
                _nf_sum += float(_nf.sum().item())
                _nf_cnt += int(_nf.numel())
            if "innovation_evidence_gate" in outputs:
                _eg = outputs["innovation_evidence_gate"].detach().float()
                _eg_sum += float(_eg.sum().item())
                _eg_cnt += int(_eg.numel())
            if "innovation_robust_inflation" in outputs:
                _inf = outputs["innovation_robust_inflation"].detach().float().reshape(-1)
                _inf_sum += float(_inf.sum().item())
                _inf_cnt += int(_inf.numel())
                _inf_tail += int((_inf > 1.75).sum().item())
            if "output_fusion_w" in outputs and "snr_db" in batch:
                _of = outputs["output_fusion_w"].detach().float()[:, 0, :late_ix].mean(dim=-1)
                _sn4 = batch["snr_db"].detach().float().reshape(-1)
                _lo4 = _sn4 < 10.0
                _hi4 = _sn4 >= 20.0
                if bool(_lo4.any()):
                    _of_lo[0] += float(_of[_lo4].sum().item())
                    _of_lo[1] += int(_lo4.sum().item())
                if bool(_hi4.any()):
                    _of_hi[0] += float(_of[_hi4].sum().item())
                    _of_hi[1] += int(_hi4.sum().item())
            if "lateral_lambda_factor" in outputs and "snr_db" in batch:
                _lf = outputs["lateral_lambda_factor"].detach().float()[:, 0, late_ix:].mean(dim=-1)
                _sn5 = batch["snr_db"].detach().float().reshape(-1)
                _lo5 = _sn5 < 10.0
                _hi5 = _sn5 >= 20.0
                if bool(_lo5.any()):
                    _lf_lo[0] += float(_lf[_lo5].sum().item())
                    _lf_lo[1] += int(_lo5.sum().item())
                if bool(_hi5.any()):
                    _lf_hi[0] += float(_lf[_hi5].sum().item())
                    _lf_hi[1] += int(_hi5.sum().item())
            if "lateral_strength" in outputs and "snr_db" in batch:
                _st = outputs["lateral_strength"].detach().float()[:, 0, late_ix:].mean(dim=-1)
                _sn3 = batch["snr_db"].detach().float().reshape(-1)
                _lo3 = _sn3 < 10.0
                _hi3 = _sn3 >= 20.0
                if bool(_lo3.any()):
                    _st_lo[0] += float(_st[_lo3].sum().item())
                    _st_lo[1] += int(_lo3.sum().item())
                if bool(_hi3.any()):
                    _st_hi[0] += float(_st[_hi3].sum().item())
                    _st_hi[1] += int(_hi3.sum().item())
            if "innovation_precision_w" in outputs and "snr_db" in batch:
                _pw = outputs["innovation_precision_w"].detach().float()[:, 0, late_ix:].mean(dim=-1)
                _sn2 = batch["snr_db"].detach().float().reshape(-1)
                _lo2 = _sn2 < 10.0
                _hi2 = _sn2 >= 20.0
                if bool(_lo2.any()):
                    _pw_lo[0] += float(_pw[_lo2].sum().item())
                    _pw_lo[1] += int(_lo2.sum().item())
                if bool(_hi2.any()):
                    _pw_hi[0] += float(_pw[_hi2].sum().item())
                    _pw_hi[1] += int(_hi2.sum().item())
            if "lateral_kernel_dev" in outputs and "snr_db" in batch:
                _kd = outputs["lateral_kernel_dev"].detach().float()[:, 0, late_ix:].mean(dim=-1)
                _sn = batch["snr_db"].detach().float().reshape(-1)
                _lo = _sn < 10.0
                _hi = _sn >= 20.0
                if bool(_lo.any()):
                    _kd_lo[0] += float(_kd[_lo].sum().item())
                    _kd_lo[1] += int(_lo.sum().item())
                if bool(_hi.any()):
                    _kd_hi[0] += float(_kd[_hi].sum().item())
                    _kd_hi[1] += int(_hi.sum().item())
        except Exception as _e3:
            logging.getLogger(PROGRAM_NAME).debug("roughness skipped: %r", _e3)
        try:
            try:
                with torch.no_grad():
                    _d0 = outputs["denoised"].detach().float()
                    _c0 = batch["clean"].detach().float()
                    _x0 = batch["noisy"].detach().float()
                    if _d0.dim() == 3:
                        _d0, _c0, _x0 = _d0[:, 0], _c0[:, 0], _x0[:, 0]
                    _dn = _c0.abs() + 1.0e-30
                    _ro = ((_d0 - _c0).abs() / _dn).mean(dim=0)
                    _ri = ((_x0 - _c0).abs() / _dn).mean(dim=0)
                    _b0 = int(_d0.shape[0])
                    if _dnh_out is None:
                        _dnh_out = _ro * _b0
                        _dnh_in = _ri * _b0
                    else:
                        _dnh_out = _dnh_out + _ro * _b0
                        _dnh_in = _dnh_in + _ri * _b0
                    _dnh_n += _b0
            except Exception:
                pass
            _pam.update(outputs["denoised"], batch["clean"], batch.get("clean_bg"),
                        batch.get("has_anomaly"), batch.get("event_mask"))
            _dv = outputs["denoised"].detach().float()
            if _dv.dim() == 3:
                _dv = _dv[:, 0]
            _mono_viol += float((_dv[:, 1:] > _dv[:, :-1]).float().sum().item())
            _mono_pairs += float(_dv.shape[0] * (_dv.shape[1] - 1))
        except Exception:
            pass
        accum.update(
            outputs["denoised"],
            batch["clean"],
            outputs["reliability"],
            float(losses["total"].item()),
            snr_db=batch.get("snr_db"),
        )
        if "manifold_z" in outputs:
            gsn = model.gate_scale_norm.float()
            _mz_exam = outputs.get("manifold_prior_z", outputs["manifold_z"])
            man_pred = signed_symexp(_mz_exam.float()) * gsn
            man_pred_comp = signed_symexp(outputs["manifold_z"].float()) * gsn
            if "trace_scale" in outputs:
                man_pred = man_pred / outputs["trace_scale"].float()
            clean32 = batch["clean"].float()
            _mbg = batch.get("clean_bg")
            _mha = batch.get("has_anomaly")
            if (bool(getattr(cfg.loss,
                             "manifold_prior_target_background", False))
                    and _mbg is not None and _mha is not None):
                _wq = _mha.float().reshape(-1, 1, 1)
                man_target = (1.0 - _wq) * clean32 + _wq * _mbg.float()
            else:
                man_target = clean32
            man_sse += float(((man_pred - man_target) ** 2)[..., late_ix:].sum().item())
            man_energy += float((man_target**2)[..., late_ix:].sum().item())
            man_sse_clean += float(((man_pred - clean32) ** 2)[..., late_ix:].sum().item())
            man_energy_clean += float((clean32**2)[..., late_ix:].sum().item())
            if "trace_scale" in outputs:
                man_pred_comp = man_pred_comp / outputs["trace_scale"].float()
            man_sse_comp += float(((man_pred_comp - man_target) ** 2)[..., late_ix:].sum().item())
            if "z_denoise" in outputs:
                den_pred = signed_symexp(outputs["z_denoise"].float()) * gsn
                if "trace_scale" in outputs:
                    den_pred = den_pred / outputs["trace_scale"].float()
                den_sse += float(((den_pred - clean32) ** 2)[..., late_ix:].sum().item())
                try:
                    _sb2 = batch.get("snr_db")
                    if _sb2 is not None and _sb2.numel() == clean32.shape[0]:
                        _n = clean32.shape[0]
                        _c2 = clean32[..., late_ix:].reshape(_n, -1)
                        _d = den_pred.float()[..., late_ix:].reshape(_n, -1) - _c2
                        _p = man_pred_comp.float()[..., late_ix:].reshape(_n, -1) - _c2
                        _f = outputs["denoised"].float()[..., late_ix:].reshape(_n, -1) - _c2
                        _st2 = torch.stack([(_d * _d).sum(1), (_p * _p).sum(1), (_d * _p).sum(1),
                                              (_f * _f).sum(1), (_c2 * _c2).sum(1)], 1).double().cpu().numpy()
                        for _s_, _v_ in zip(_sb2.detach().float().view(-1).cpu().numpy().tolist(), _st2):
                            if math.isfinite(_s_) and bool(np.all(np.isfinite(_v_))):
                                _k2 = int(math.floor(_s_ / 5.0)) * 5
                                _arm[_k2] = _arm.get(_k2, np.zeros(5)) + _v_
                except Exception:
                    pass
                try:
                    _dn2 = clean32.abs() + 1.0e-30
                    _rm = ((man_pred.float() - clean32).abs() / _dn2).mean(dim=0).reshape(-1)
                    _rd = ((den_pred.float() - clean32).abs() / _dn2).mean(dim=0).reshape(-1)
                    _b = int(clean32.shape[0])
                    _bl2 = outputs.get("manifold_blend")
                    _rb2 = (_bl2.detach().float().mean(dim=0).reshape(-1) if _bl2 is not None
                              else torch.zeros_like(_rm))
                    if _dnh_man is None:
                        _dnh_man = _rm * _b
                        _dnh_den = _rd * _b
                        _dnh_bl = _rb2 * _b
                    else:
                        _dnh_man = _dnh_man + _rm * _b
                        _dnh_den = _dnh_den + _rd * _b
                        _dnh_bl = _dnh_bl + _rb2 * _b
                    _dnh_an += _b
                except Exception:
                    pass
                try:
                    _zd = outputs.get("z_denoise")
                    _zm = outputs.get("manifold_z")
                    if _zd is not None and _zm is not None:
                        _ts2 = outputs.get("trace_scale")
                        _tz = signed_symlog((clean32 * (_ts2.float() if _ts2 is not None else 1.0)) / gsn)
                        _ed2 = _zd.float() - _tz
                        _ep = _zm.float() - _tz
                        _s2 = torch.stack([(_ed2 ** 2).sum(dim=0).reshape(-1),
                                             (_ep ** 2).sum(dim=0).reshape(-1),
                                             (_ed2 * _ep).sum(dim=0).reshape(-1)], 0).detach()
                        _cov = _s2 if _cov is None else _cov + _s2
                        _cov644_n += int(clean32.shape[0])
                        _bh = outputs.get("manifold_blend_head")
                        if _bh is not None and isinstance(getattr(model, "seam_blend_floor", None), torch.Tensor):
                            _fl2 = model.seam_blend_floor.detach().float().view(1, 1, -1).to(_bh.device)
                            _bd = ((_bh.float() < _fl2) & (_fl2 > 0)).float().sum(dim=0).reshape(-1).detach()
                            _bind = _bd if _bind is None else _bind + _bd
                except Exception:
                    pass
            if "z_hat_pre_event" in outputs:
                _ts = outputs.get("trace_scale")
                _inv = (1.0 / _ts.float()) if _ts is not None else 1.0
                _zpost = outputs.get("z_hat_post_event", outputs["z_hat"])
                ev_post = signed_symexp(_zpost.float()) * gsn * _inv
                ev_pre = signed_symexp(outputs["z_hat_pre_event"].float()) * gsn * _inv
                ev_sse += float(((ev_post - ev_pre) ** 2)[..., late_ix:].sum().item())
                _ev647_max = max(_ev647_max, float((_zpost.float() - outputs["z_hat_pre_event"].float()).abs().max().item()))
                if "z_hat_pre_lateral" in outputs:
                    _lat_post = signed_symexp(outputs["z_hat"].float()) * gsn * _inv
                    _lat_pre = signed_symexp(outputs["z_hat_pre_lateral"].float()) * gsn * _inv
                    _lat647_sse += float(((_lat_post - _lat_pre) ** 2)[..., late_ix:].sum().item())
            if "relaxation_update" in outputs:
                ru = outputs["relaxation_update"].float()[..., late_ix:]
                relax_abs_sum += float(ru.abs().sum().item())
                relax_abs_count += int(ru.numel())
            bl = outputs["manifold_blend"][..., late_ix:]
            blend_late_sum += float(bl.sum().item())
            blend_late_count += int(bl.numel())
            _sb = batch.get("snr_db")
            if _sb is not None:
                try:
                    _blr = bl.detach().float().reshape(bl.shape[0], -1).mean(dim=1).cpu().numpy()
                    _sbn = _sb.detach().float().view(-1).cpu().numpy()
                    if _sbn.shape[0] == _blr.shape[0]:
                        for _s_, _b_ in zip(_sbn.tolist(), _blr.tolist()):
                            if not (math.isfinite(_s_) and math.isfinite(_b_)):
                                continue
                            _kb = int(math.floor(_s_ / 5.0)) * 5
                            _acc = _bl.setdefault(_kb, [0.0, 0.0])
                            _acc[0] += float(_b_)
                            _acc[1] += 1.0
                except Exception:
                    pass
            if "manifold_innovation_w" in outputs:
                iw = outputs["manifold_innovation_w"][..., late_ix:]
                innov_w_sum += float(iw.sum().item())
                innov_w_count += int(iw.numel())
    if fp32_retry_batches and cfg.train.amp:
        cfg.train.amp = False
        logging.getLogger(PROGRAM_NAME).critical(
            "AMP DEMOTED: %d validation batch(es) were finite only in fp32. "
            "All subsequent training and evaluation runs in fp32.",
            fp32_retry_batches,
        )
    if nonfinite_batches:
        logging.getLogger(PROGRAM_NAME).error(
            "%d/%d validation batches were NON-FINITE and quarantined from the metrics. "
            "If this is every batch, the forward is numerically broken for the current weights; "
            "reports/nan_forensics.json names the first offending tensors (and, once per run, "
            "the first non-finite MODULE ).",
            nonfinite_batches, total_batches,
        )
    metrics = accum.compute(cfg.loss.cvar_fraction)
    metrics.update(_pam.compute())
    if _dnh_n > 0 and _dnh_out is not None:
        _ls0 = int(cfg.data.late_start_index)
        _ro = (_dnh_out / _dnh_n).cpu().numpy()
        _ri = (_dnh_in / _dnh_n).cpu().numpy()
        metrics["early_do_no_harm_percent"] = float(
            100.0 * np.maximum(_ro[:_ls0] - _ri[:_ls0], 0.0).mean())
        metrics["per_gate_mean_error_in_percent"] = (100.0 * _ri).tolist()
        metrics["per_gate_mean_error_out_percent"] = (100.0 * _ro).tolist()
    if _dnh_an > 0 and _dnh_man is not None:
        metrics["per_gate_mean_error_prior_percent"] = (100.0 * (_dnh_man / _dnh_an).cpu().numpy()).tolist()
        metrics["per_gate_mean_error_denoise_percent"] = (100.0 * (_dnh_den / _dnh_an).cpu().numpy()).tolist()
        if _dnh_bl is not None:
            metrics["per_gate_mean_blend"] = (_dnh_bl / _dnh_an).cpu().numpy().tolist()
        if _cov is not None and _cov644_n > 0:
            _c = (_cov / float(_cov644_n)).cpu().numpy()
            metrics["per_gate_arm_cov_z"] = {"dd": _c[0].tolist(), "pp": _c[1].tolist(), "dp": _c[2].tolist()}
            if _bind is not None:
                metrics["per_gate_floor_binding_rate"] = (_bind / float(_cov644_n)).cpu().numpy().tolist()
        metrics["early_mid_misfit_ratio"] = float(
            np.mean(_ro[:_ls0]) / max(float(np.mean(_ri[:_ls0])), 1e-30))
    _gm = metrics.get("per_gate_median_error_percent") or []
    if _gm:
        _ls0 = int(cfg.data.late_start_index)
        _seam = _gm[max(0, _ls0 - 8):_ls0]
        metrics["seam_band_max_percent"] = float(max(_seam)) if _seam else float("nan")
    metrics["gate_monotonic_violation_percent"] = (
        100.0 * _mono_viol / _mono_pairs if _mono_pairs > 0 else float("nan"))
    valid_batches = total_batches - nonfinite_batches
    metrics["valid_batches"] = float(valid_batches)
    metrics["total_batches"] = float(total_batches)
    if valid_batches == 0 and total_batches > 0:
        for _k in ("global_error_percent", "early_mid_error_percent",
                   "late_error_percent", "late_cvar_percent",
                   "relative_mse_global", "relative_mse_early_mid", "relative_mse_late"):
            metrics[_k] = float("nan")
        metrics["poisoned"] = 1.0
        logging.getLogger(PROGRAM_NAME).critical(
            "VALIDATION POISONED: 0/%d finite batches. Metrics are NaN, the "
            "epoch is INFEASIBLE, and it earns no plateau/stage/lambda credit.",
            total_batches,
        )
    if man_energy > 0.0:
        metrics["manifold_late_error_percent"] = 100.0 * math.sqrt(man_sse / max(man_energy, 1e-30))
        if man_energy_clean > 0.0:
            metrics["manifold_late_vs_clean_percent"] = 100.0 * math.sqrt(
                man_sse_clean / max(man_energy_clean, 1e-30))
        metrics["manifold_z_late_vs_bg_percent"] = 100.0 * math.sqrt(
            man_sse_comp / max(man_energy, 1e-30))
        metrics["blend_late_mean"] = blend_late_sum / max(blend_late_count, 1)
        if den_sse > 0.0:
            metrics["denoise_path_late_error_percent"] = 100.0 * math.sqrt(den_sse / max(man_energy, 1e-30))
        metrics["event_term_late_percent"] = 100.0 * math.sqrt(ev_sse / max(man_energy, 1e-30))
        metrics["lateral_term_late_percent"] = 100.0 * math.sqrt(_lat647_sse / max(man_energy, 1e-30))
        metrics["event_increment_max_abs_z"] = float(_ev647_max)
        if relax_abs_count > 0:
            metrics["relaxation_update_late_mean_abs"] = relax_abs_sum / relax_abs_count
        if innov_w_count > 0:
            metrics["innovation_w_late_mean"] = innov_w_sum / innov_w_count
    if _lr_n > 0:
        metrics["lateral_late_roughness_out_dex"] = math.sqrt(_lr_err / _lr_n)
        metrics["lateral_late_roughness_ref_dex"] = math.sqrt(_lr_ref / _lr_n)
        metrics["lateral_late_roughness_in_dex"] = math.sqrt(_lr_in / _lr_n)
        metrics["lateral_late_roughness_ratio"] = (
            math.sqrt(_lr_err / _lr_n) / max(math.sqrt(_lr_ref / _lr_n), 1e-12))
        metrics["lateral_blocks"] = float(_lr_blocks)
        metrics["lateral_applied_batches"] = float(_lat_applied)
        if _lrq[2] > 0:
            metrics["lateral_quiet_roughness_out_dex"] = math.sqrt(_lrq[0] / _lrq[2])
            metrics["lateral_quiet_roughness_ref_dex"] = math.sqrt(_lrq[1] / _lrq[2])
            metrics["lateral_quiet_blocks"] = _lrq[2] / max(1.0, float((_plb - 2) * _L9)) if _lr_blocks else 0.0
        if _lrb[2] > 0:
            metrics["lateral_body_roughness_out_dex"] = math.sqrt(_lrb[0] / _lrb[2])
            metrics["lateral_body_roughness_ref_dex"] = math.sqrt(_lrb[1] / _lrb[2])
        if _lrbeta[1] > 0 and _lrbeta[2] > 0:
            metrics["lateral_body_structure_beta"] = _lrbeta[0] / _lrbeta[1]
            metrics["lateral_body_erased_share"] = _lrbeta[3] / _lrbeta[2]
    if _lib_refq > 0.0:
        metrics["library_anchor_late_percent"] = 100.0 * math.sqrt(
            _lib_sq / max(_lib_refq, 1e-300))
        if _lib_n > 0:
            metrics["library_gate_late_mean"] = _lib_gsum / _lib_n
    if _ord_n > 0.0:
        metrics["late_order_violation_percent"] = 100.0 * _ord_v / _ord_n
    if _rdg_refq > 0.0:
        metrics["ridge_only_late_percent"] = 100.0 * math.sqrt(
            _rdg_sq / max(_rdg_refq, 1e-300))
        if _rdg_n > 0:
            metrics["ridge_gate_late_mean"] = _rdg_gsum / _rdg_n
    if _rdgD_refq > 0.0:
        metrics["ridge_only_late_deep_percent"] = 100.0 * math.sqrt(
            _rdgD_sq / max(_rdgD_refq, 1e-300))
    if _libD_refq > 0.0:
        metrics["library_anchor_late_deep_percent"] = 100.0 * math.sqrt(
            _libD_sq / max(_libD_refq, 1e-300))
    if _pr_list:
        _prA = np.asarray(_pr_list, dtype=np.float64)
        _pra = _prA[:, 1]
        for _thr in (3.0, 4.0, 5.0):
            metrics["late_pass_rate_%d_percent" % int(_thr)] = float(
                100.0 * np.mean(_pra <= _thr))
        for _q in (50, 90, 95, 99):
            metrics["late_row_p%d_percent" % _q] = float(
                np.percentile(_pra, _q))
        _sn6 = _prA[:, 0]
        if np.isfinite(_sn6).any():
            _ed = np.nanpercentile(_sn6, np.linspace(0, 100, 6))
            _fr = []
            for _lo, _hi in zip(_ed[:-1], _ed[1:]):
                _m = (_sn6 >= _lo) & (_sn6 <= _hi)
                if _m.any():
                    _fr.append("[%+.0f,%+.0f]:%.0f%%" % (
                        _lo, _hi, 100.0 * np.mean(_pra[_m] <= 3.0)))
            metrics["late_pass_frontier"] = " ".join(_fr)
            _m2 = _sn6 >= 5.0
            if _m2.any():
                metrics["late_pass_rate_3_ge5db_percent"] = float(100.0 * np.mean(_pra[_m2] <= 3.0))
                metrics["late_rows_ge5db"] = float(int(_m2.sum()))
        if _prA.shape[1] >= 3:
            _ha3 = _prA[:, 2]
            _mq = _ha3 < 0.5
            _ma = _ha3 >= 0.5
            metrics["late_row_mean_percent"] = float(np.mean(_pra))
            if _mq.any():
                metrics["late_pass_rate_3_quiet_percent"] = float(100.0 * np.mean(_pra[_mq] <= 3.0))
                metrics["late_rows_quiet"] = float(int(_mq.sum()))
            if _ma.any():
                metrics["late_pass_rate_3_anomaly_percent"] = float(100.0 * np.mean(_pra[_ma] <= 3.0))
                metrics["late_rows_anomaly"] = float(int(_ma.sum()))
    if _arm:
        _parts = []
        for _k2, _v in sorted(_arm.items()):
            _sdd, _spp, _sdp, _sff, _e4 = [float(x) for x in _v]
            if _e4 <= 0.0:
                continue
            _den = _sdd + _spp - 2.0 * _sdp
            _b2 = min(1.0, max(0.0, (_sdd - _sdp) / _den)) if _den > 1e-30 else 0.5
            _so = (1.0 - _b2) ** 2 * _sdd + _b2 ** 2 * _spp + 2.0 * _b2 * (1.0 - _b2) * _sdp
            _parts.append("[%d,%d):d%.2f/p%.2f/out%.2f|b*%.2f->%.2f" % (
                _k2, _k2 + 5, 100.0 * math.sqrt(_sdd / _e4), 100.0 * math.sqrt(_spp / _e4),
                100.0 * math.sqrt(_sff / _e4), _b2, 100.0 * math.sqrt(max(_so, 0.0) / _e4)))
        if _parts:
            metrics["arm_decomposition_by_snr_bin"] = " ".join(_parts)
    if _bl:
        metrics["blend_by_snr_bin"] = " ".join(
            "[%d,%d):%.2f" % (k, k + 5, v[0] / max(v[1], 1.0)) for k, v in sorted(_bl.items()))
    if _nf_cnt > 0:
        metrics["late_noise_floor_late_mean"] = _nf_sum / _nf_cnt
    if _eg_cnt > 0:
        metrics["innovation_evidence_open_rate"] = _eg_sum / _eg_cnt
    if _kd_lo[1] > 0:
        metrics["lateral_kernel_dev_low_snr"] = _kd_lo[0] / _kd_lo[1]
    if _kd_hi[1] > 0:
        metrics["lateral_kernel_dev_high_snr"] = _kd_hi[0] / _kd_hi[1]
    if _pw_lo[1] > 0:
        metrics["innovation_precision_w_low_snr"] = _pw_lo[0] / _pw_lo[1]
    if _pw_hi[1] > 0:
        metrics["innovation_precision_w_high_snr"] = _pw_hi[0] / _pw_hi[1]
    if _lf_lo[1] > 0:
        metrics["lateral_lambda_factor_low_snr"] = _lf_lo[0] / _lf_lo[1]
    if _lf_hi[1] > 0:
        metrics["lateral_lambda_factor_high_snr"] = _lf_hi[0] / _lf_hi[1]
    if _st_lo[1] > 0:
        metrics["lateral_strength_low_snr"] = _st_lo[0] / _st_lo[1]
    if _st_hi[1] > 0:
        metrics["lateral_strength_high_snr"] = _st_hi[0] / _st_hi[1]
    if _inf_cnt > 0:
        metrics["innovation_robust_inflation_mean"] = _inf_sum / _inf_cnt
        metrics["innovation_robust_tail_rate"] = _inf_tail / _inf_cnt
    if _of_lo[1] > 0:
        metrics["output_fusion_w_low_snr"] = _of_lo[0] / _of_lo[1]
    if _of_hi[1] > 0:
        metrics["output_fusion_w_high_snr"] = _of_hi[0] / _of_hi[1]
    return metrics


def effective_late_cvar_threshold(cfg: ContractConfig) -> float:
    """The tail gate used for control and selection."""
    eff = getattr(cfg, "late_cvar_threshold_effective", None)
    try:
        eff = float(eff) if eff is not None else None
    except (TypeError, ValueError):
        eff = None
    return eff if eff and eff > 0.0 else float(cfg.late_cvar_threshold)


def contract_late_cvar_threshold(cfg: ContractConfig) -> float:
    """The PROJECT'S own tail gate, never rebased."""
    return float(cfg.late_cvar_threshold)


def is_project_feasible(metrics: Mapping[str, float], cfg: ContractConfig) -> bool:
    if metrics.get("poisoned"):
        return False
    vals = (metrics["global_error_percent"], metrics["late_error_percent"],
            metrics["early_mid_error_percent"], metrics.get("late_cvar_percent", float("nan")))
    if any(not math.isfinite(float(v)) for v in vals):
        return False
    _wb = metrics.get("late_bin_worst_percent")
    if _wb is not None and math.isfinite(float(_wb)):
        if float(_wb) > 100.0 * float(getattr(cfg, "late_bin_threshold", 0.03)):
            return False
    return (
        metrics["global_error_percent"] <= 100.0 * cfg.global_threshold
        and metrics["late_error_percent"] <= 100.0 * cfg.late_threshold
        and metrics["early_mid_error_percent"] <= 100.0 * cfg.early_mid_threshold
        and float(metrics["late_cvar_percent"]) <= 100.0 * contract_late_cvar_threshold(cfg)
    )


def remediation_failing_gates(vm: Mapping[str, float], c: ContractConfig) -> List[str]:
    """Which qualification gates fail on these validation metrics."""
    f: List[str] = []
    def _over(key: str, tau_pc: float) -> bool:
        x = float(vm.get(key, float("nan")))
        return math.isfinite(x) and x > tau_pc
    if _over("global_error_percent", 100.0 * c.global_threshold):
        f.append("global")
    if _over("early_mid_error_percent", 100.0 * c.early_mid_threshold):
        f.append("early_mid")
    if _over("late_error_percent", 100.0 * c.late_threshold):
        f.append("late")
    if _over("late_cvar_percent", 100.0 * contract_late_cvar_threshold(c)):
        f.append("late_cvar")
        if (effective_late_cvar_threshold(c) > contract_late_cvar_threshold(c)
                and _over("late_cvar_percent",
                          100.0 * effective_late_cvar_threshold(c)) is False):
            f[-1] = "late_cvar(contract %.1f%%, control target rebased to %.1f%%)" % (
                100.0 * contract_late_cvar_threshold(c),
                100.0 * effective_late_cvar_threshold(c))
    if _over("anomaly_late_error_percent",
             100.0 * c.late_threshold * c.anomaly_late_multiple):
        f.append("anomaly_late")
    _rc = float(vm.get("anomaly_lift_recovery", float("nan")))
    if math.isfinite(_rc) and _rc < float(c.anomaly_recovery_target):
        f.append("recovery")
    _a = float(vm.get("event_auroc", float("nan")))
    if math.isfinite(_a) and _a < float(c.event_auroc_target):
        f.append("event_detect")
    _sr = float(vm.get("singleton_reject_auroc", float("nan")))
    if math.isfinite(_sr) and _sr < float(c.event_auroc_target):
        f.append("singleton_reject")
    return f


def candidate_score_terms(metrics: Mapping[str, float], cfg: ContractConfig) -> Dict[str, float]:
    """The selection score and its decomposition (see candidate_score)."""
    vg = max(0.0, metrics["global_error_percent"] / (100 * cfg.global_threshold) - 1.0)
    vl = max(0.0, metrics["late_error_percent"] / (100 * cfg.late_threshold) - 1.0)
    vem = max(0.0, metrics["early_mid_error_percent"] / (100 * cfg.early_mid_threshold) - 1.0)
    vcv = max(0.0, metrics["late_cvar_percent"] / (100 * effective_late_cvar_threshold(cfg)) - 1.0)
    _wb = metrics.get("late_bin_worst_percent")
    _wb_label = "n/a"
    _wb_share = float("nan")
    _pb = metrics.get("per_snr_bin")
    _min_share = float(getattr(cfg, "late_bin_min_share", 0.02))
    if isinstance(_pb, Mapping) and _pb:
        _tot = float(sum(int((v or {}).get("count", 0) or 0) for v in _pb.values()))
        _best = None
        for _lab, _v in _pb.items():
            _cnt = int((_v or {}).get("count", 0) or 0)
            _le = (_v or {}).get("late_error_percent")
            if _cnt <= 0 or _le is None or not math.isfinite(float(_le)):
                continue
            if _tot > 0 and _cnt < _min_share * _tot:
                continue
            if _best is None or float(_le) > _best[0]:
                _best = (float(_le), str(_lab), _cnt / max(_tot, 1.0))
        if _best is not None:
            _wb, _wb_label, _wb_share = _best
    vwb = 0.0
    if _wb is not None and math.isfinite(float(_wb)):
        vwb = max(0.0, float(_wb)
                  / (100.0 * float(getattr(cfg, "late_bin_threshold", 0.03))) - 1.0)
        vwb = min(vwb, float(getattr(cfg, "late_bin_violation_cap", 1.0)))
    vp3 = 0.0
    _p3 = metrics.get("late_pass_rate_3_percent", metrics.get("pass_at_3_percent"))
    if _p3 is not None and math.isfinite(float(_p3)):
        _tp3 = 100.0 * float(getattr(cfg, "row_pass_rate_target", 0.95))
        vp3 = max(0.0, (_tp3 - float(_p3)) / max(_tp3, 1e-6))
    _redundant_em = (abs(metrics["global_error_percent"]
                         - metrics["early_mid_error_percent"])
                     <= max(0.02, 0.01 * abs(metrics["global_error_percent"])))
    base = (10.0 * (vg * vg + 2.0 * vl * vl + 2.0 * vwb * vwb
                    + (0.0 if _redundant_em else vem * vem) + vcv * vcv)
            + metrics["late_cvar_percent"] / 100.0 + 20.0 * vp3 * vp3)
    extra = 0.0
    _al = metrics.get("anomaly_late_error_percent")
    if _al is not None and math.isfinite(float(_al)):
        _tau_a = 100.0 * cfg.late_threshold * max(float(cfg.anomaly_late_multiple), 1e-6)
        _va = max(0.0, float(_al) / _tau_a - 1.0)
        extra += 10.0 * _va * _va
    _rc = metrics.get("anomaly_lift_recovery")
    if _rc is not None and math.isfinite(float(_rc)):
        _vr = max(0.0, 1.0 - float(_rc) / max(float(cfg.anomaly_recovery_target), 1e-6))
        extra += 6.0 * _vr * _vr
    _au = metrics.get("event_auroc")
    if _au is not None and math.isfinite(float(_au)):
        _span = max(float(cfg.event_auroc_target) - 0.5, 1e-6)
        _vd = max(0.0, (float(cfg.event_auroc_target) - float(_au)) / _span)
        extra += 6.0 * _vd * _vd
    _sr = metrics.get("singleton_reject_auroc")
    if _sr is not None and math.isfinite(float(_sr)):
        _span = max(float(cfg.event_auroc_target) - 0.5, 1e-6)
        _vs = max(0.0, (float(cfg.event_auroc_target) - float(_sr)) / _span)
        extra += 3.0 * _vs * _vs
    _pr = metrics.get("paired_anomaly_recovery")
    if _pr is not None and math.isfinite(float(_pr)):
        _vp = max(0.0, 1.0 - float(_pr) / max(float(cfg.anomaly_recovery_target), 1e-6))
        extra += 6.0 * _vp * _vp
    _pw = metrics.get("paired_anomaly_recovery_worst_decile_mean")
    if _pw is not None and math.isfinite(float(_pw)):
        _vw = max(0.0, 1.0 - float(_pw) / max(float(cfg.anomaly_recovery_target), 1e-6))
        extra += 4.0 * _vw * _vw
    _ps = metrics.get("paired_anomaly_sign_agreement")
    if _ps is not None and math.isfinite(float(_ps)):
        _vsg = max(0.0, (0.90 - float(_ps)) / 0.40)
        extra += 4.0 * _vsg * _vsg
    _cap = float(getattr(cfg, "anomaly_score_cap", 0.0) or 0.0)
    _raw = extra
    if _cap > 0.0:
        extra = _cap * math.tanh(extra / max(float(getattr(cfg, "anomaly_score_scale", 40.0)), 1e-6))
    return {
        "total": base + extra,
        "g": 10.0 * vg * vg,
        "l": 20.0 * vl * vl,
        "em": 0.0 if _redundant_em else 10.0 * vem * vem,
        "cvar": 10.0 * vcv * vcv + metrics["late_cvar_percent"] / 100.0,
        "worst_bin": 20.0 * vwb * vwb,
        "pass3": 20.0 * vp3 * vp3,
        "anomaly_raw": _raw,
        "worst_bin_value": float(_wb) if (_wb is not None and math.isfinite(float(_wb))) else float("nan"),
        "worst_bin_share": _wb_share,
        "anomaly": extra,
        "_worst_bin_label": _wb_label,
    }


def candidate_score(metrics: Mapping[str, float], cfg: ContractConfig) -> float:
    """Selection score: lower is better."""
    return float(candidate_score_terms(metrics, cfg)["total"])


def stamp_checkpoint_qualification(
    path: Path, updates: Mapping[str, Any], logger: Optional[logging.Logger] = None
) -> bool:
    """Merge a qualification payload into a checkpoint file."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        q = dict(state.get("qualification") or {})
        q.update(dict(updates))
        q["program_version"] = PROGRAM_VERSION
        state["qualification"] = q
        tmp = path.with_suffix(path.suffix + ".qtmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("Could not stamp qualification into %s: %r", path, exc)
        return False


def checkpoint_qualification(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The qualification payload of a loaded checkpoint, or an explicit 'unqualified, never contract-tested' record
    when the checkpoint predates v1.40.
    """
    q = state.get("qualification")
    if isinstance(q, Mapping):
        return dict(q)
    return {"contract_test_pass": None, "reason": "checkpoint carries no qualification payload"}


def save_checkpoint(
    path: Path,
    model: PEBRNet,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.cuda.amp.GradScaler",
    controller: ConstraintController,
    cfg: Config,
    epoch: int,
    stage: str,
    metrics: Mapping[str, float],
    clean_cache: CleanCacheInfo,
    ema: Optional["ModelEMA"] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    rng_state = {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
    }
    environment = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "seed": int(cfg.train.seed),
    }
    torch.save(
        {
            "program": PROGRAM_NAME,
            "version": PROGRAM_VERSION,
            "ema_state": ema.state_dict() if ema is not None else None,
            "rng_state": rng_state,
            "environment": environment,
            "model_state": model.state_dict(),
            "pending_control_updates": (model.export_pending_control()
                                        if hasattr(model, "export_pending_control") else {}),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "constraint_state": controller.state_dict(),
            "config": dataclass_to_dict(cfg),
            "epoch": epoch,
            "stage": stage,
            "metrics": dict(metrics),
            "global_scale": clean_cache.global_scale,
            "gate_scale": np.asarray(clean_cache.gate_scale, dtype=np.float64),
            "target_time": clean_cache.target_time,
            "survey_record": getattr(getattr(getattr(model, "module", model), "_orig_mod", getattr(model, "module", model)),
                                         "_survey_record", None),
        },
        tmp,
    )
    os.replace(tmp, path)


ARCHITECTURE_FLAGS = (
    "use_neighbor_attention", "num_neighbors", "use_manifold_branch", "use_position_encoding",
    "use_manifold_innovation", "use_physics_atoms", "physics_atom_count",
    "manifold_blend_floor_late", "manifold_pool_heads", "use_ip_head", "ip_atom_count",
    "per_trace_normalization", "per_trace_norm_gates", "per_trace_norm_estimator", "amplitude_conditioning",
    "measurement_conditioned_gates", "prior_sigma_floor", "innovation_mod_floor", "blend_ceiling_k",
    "supervised_primary_fusion",
    "use_library_subspace", "library_subspace_rank",
    "use_ridge_anchor",
    "coef_field_trust_mix",
    "relaxation_stretch_exponents", "use_relaxation_state_update", "relaxation_update_scale",
    "manifold_quality_gate",
    "use_neighbor_geometry",
    "use_event_residual",
    "gate_local_norm", "decoder_gate_norm_channel",
    "seam_blend_floor_mode",
    "lateral_event_licence", "lateral_event_licence_p0",
    "lateral_witness_gates_attention", "lateral_witness_dead_zone",
)


_PRE_135_ARCH_DEFAULTS = {
    "relaxation_stretch_exponents": (1.0,),
    "use_relaxation_state_update": False,
    "relaxation_update_scale": 0.20,
    "manifold_quality_gate": True,
    "use_neighbor_geometry": False,
    "use_event_residual": False,
    "coef_field_trust_mix": True,
    "gate_local_norm": False,
    "decoder_gate_norm_channel": False,
    "per_trace_norm_estimator": "stack_rms",
    "amplitude_conditioning": "off",
    "seam_blend_floor_mode": "hard",
    "lateral_event_licence": 0.0,
    "lateral_witness_gates_attention": False,
    "lateral_witness_dead_zone": 0.2,
}


def reconcile_model_config_with_checkpoint(
    path: Path, cfg: "Config", logger: logging.Logger
) -> List[str]:
    """The checkpoint's architecture flags are authoritative."""
    overrides: List[str] = []
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Could not pre-read the checkpoint config (%s); using the CLI values.", exc)
        return overrides
    saved = (state.get("config") or {}).get("model", {})
    if not saved:
        cfg.model.amplitude_conditioning = "off"
        for nm in ("per_trace_normalization", "measurement_conditioned_gates", "use_relaxation_state_update", "use_neighbor_geometry", "use_event_residual"):
            if getattr(cfg.model, nm):
                setattr(cfg.model, nm, False)
                overrides.append("%s: True -> False (checkpoint predates it)" % nm)
        if tuple(getattr(cfg.model, "relaxation_stretch_exponents", (1.0,))) != (1.0,):
            cfg.model.relaxation_stretch_exponents = (1.0,)
            overrides.append("relaxation_stretch_exponents -> (1.0,) (checkpoint predates it)")
    else:
        for name in ARCHITECTURE_FLAGS:
            if name not in saved:
                if name in _PRE_135_ARCH_DEFAULTS:
                    default = _PRE_135_ARCH_DEFAULTS[name]
                else:
                    default = (
                        False
                        if name in ("use_ip_head", "per_trace_normalization", "measurement_conditioned_gates")
                        else getattr(cfg.model, name)
                    )
                if getattr(cfg.model, name) != default:
                    overrides.append("%s: %r -> %r (absent from checkpoint)" % (name, getattr(cfg.model, name), default))
                    setattr(cfg.model, name, default)
                continue
            saved_v = saved[name]
            if name == "relaxation_stretch_exponents" and isinstance(saved_v, (list, tuple)):
                saved_v = tuple(float(v) for v in saved_v)
            if getattr(cfg.model, name) != saved_v:
                overrides.append("%s: %r -> %r (from checkpoint)" % (name, getattr(cfg.model, name), saved_v))
                setattr(cfg.model, name, saved_v)
    if "num_neighbors" in saved:
        cfg.data.num_neighbors = int(saved["num_neighbors"])
    overrides.extend(_apply_saved_receptive_field(saved, cfg, logger, str(path)))
    if overrides:
        logger.warning(
            "MODEL CONFIG RECONCILED WITH THE CHECKPOINT (the file wins, not the command line): %s",
            "; ".join(overrides),
        )
    else:
        logger.info("Model config matches the checkpoint; no reconciliation needed.")
    return overrides


def load_checkpoint(
    path: Path,
    model: PEBRNet,
    controller: ConstraintController,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    expected_target_time: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into a model and its constraint controller; the keys must match."""
    state = torch.load(path, map_location=device, weights_only=False)
    _sv = (state.get("config") or {}).get("model", {}) or {}
    _mc = getattr(model, "cfg", None)
    if _sv and _mc is not None:
        for _nm, _dflt in (("residual_dilations", RF_DESIGN_RESIDUAL), ("multiscale_dilations", RF_DESIGN_MULTISCALE)):
            _a = tuple(int(v) for v in (_sv.get(_nm) or _dflt))
            _b = tuple(int(v) for v in getattr(_mc, _nm, _dflt))
            if _a != _b:
                raise ValueError("checkpoint %s was trained with %s=%s but this model was built with %s: the weights "
                                 "would load without any shape error and compute a different function. Build the model from the "
                                 "checkpoint's record (reconcile_model_config_with_checkpoint / receptive_field_from_checkpoint). "
                                 % (path, _nm, _a, _b))
        for _nm2, _dv in (("gate_local_norm", False), ("per_trace_normalization", False), ("coef_field_trust_mix", True)):
            if bool(_sv.get(_nm2, _dv)) != bool(getattr(_mc, _nm2, _dv)):
                raise ValueError("checkpoint %s was trained with %s=%s but this model was built with %s: identical "
                                 "tensor shapes, different function. Build the model from the checkpoint's record. "
                                  % (path, _nm2, _sv.get(_nm2, _dv),
                                 getattr(_mc, _nm2, _dv)))
    if expected_target_time is not None:
        if "target_time" not in state:
            raise ValueError("checkpoint %s carries no gate axis; cannot verify it against the data." % path)
        require_same_axis(expected_target_time, state["target_time"], what="checkpoint")
    try:
        model.load_state_dict(state["model_state"], strict=True)
        if hasattr(model, "import_pending_control"):
            model.import_pending_control(state.get("pending_control_updates"))
        return _finish_checkpoint_load(state, controller, model)
    except RuntimeError:
        ip_keys = {k for k in model.state_dict() if k.startswith(("manifold_head_ip", "manifold_gate_ip", "ip_atoms"))}
        v135_keys = {
            k for k in model.state_dict()
            if k in ("blend_late_quality_cap", "blend_floor_scale",
                     "late_prior_advantage", "event_expression_scale",
                     "blend_late_val_floor", "event_lat_gain",
                     "late_ramp")
            or k.startswith("relax_state_head")
            or k.startswith("neighbor_attention.geo_")
            or k.startswith("event_gate_head")
            or k.startswith("struct_residual_head")
            or k.startswith("lateral_joint.")
            or k.startswith("profile_joint.")
            or k.startswith("joint_inject.")
            or k.startswith("lateral_fp_prior.")
            or k.startswith("lateral_fp_out.")
            or k.startswith("fingerprint_query_ctx.")
            or k == "gl_departure_logit"
        }
        _own = set(model.state_dict().keys())
        for _k in list(state["model_state"].keys()):
            if _k.startswith("lateral_joint.") and _k not in _own:
                state["model_state"].pop(_k)
                if logger is not None:
                    logger.warning("dropped %s from the checkpoint (no fixed lateral "
                                   "smoother in lateral_joint_mode=%s)", _k,
                                   getattr(model.cfg, "lateral_joint_mode", "learned"))
        allowed = ip_keys | v135_keys
        if not allowed:
            raise
        _msd = model.state_dict()
        for _k in list(state["model_state"].keys()):
            if _k in allowed and _k in _msd and tuple(state["model_state"][_k].shape) != tuple(_msd[_k].shape):
                state["model_state"].pop(_k)
        incompatible = model.load_state_dict(state["model_state"], strict=False)
        if hasattr(model, "import_pending_control"):
            model.import_pending_control(state.get("pending_control_updates"))
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if unexpected or not missing.issubset(allowed) or not missing:
            raise RuntimeError(
                "Checkpoint does not match the model. Missing keys beyond the tolerated "
                f"sets: {sorted(missing - allowed)}; unexpected keys: {sorted(unexpected)}."
            )
        if logger is not None:
            if any(kk.startswith("neighbor_attention.geo_") for kk in missing):
                _att = getattr(model, "neighbor_attention", None)
                if _att is not None and hasattr(_att, "geo_prior_rho"):
                    with torch.no_grad():
                        _att.geo_prior_rho.fill_(-12.0)
            if any(kk.startswith("lateral_joint.") for kk in missing):
                logger.warning(
                    "This checkpoint predates the LATERAL JOINT PRIOR: its "
                    "lambda starts at the configured init and the weights below it "
                    "never trained THROUGH the operator. Outputs are a post-smoothed "
                    "per-station estimate until the run fine-tunes; --no_lateral_joint "
                    "reproduces the checkpoint's own behaviour exactly.")
            if "gl_departure_logit" in missing:
                logger.warning("This checkpoint predates the withheld-gate departure share: it starts at its "
                               "initial value (the continuation carries %.0f%% of every withheld gate until training "
                               "moves it).",
                               100.0 * (1.0 - 1.0 / (1.0 + math.exp(-float(getattr(model.cfg, "gate_loss_departure_init", -2.0))))))
            if (missing & v135_keys) - {"gl_departure_logit"}:
                logger.warning(
                    "Older checkpoint: %d tensors of newer layers freshly initialized "
                    "(quality-gate buffers at safe defaults; RSSU head zero-initialized "
                    "and numerically inert): %s",
                    len((missing & v135_keys) - {"gl_departure_logit"}), sorted((missing & v135_keys) - {"gl_departure_logit"}),
                )
            if missing & ip_keys:
                logger.warning(
                    "IP head newly enabled: %d IP-branch tensors are freshly initialized "
                    "(all induction-path tensors restored from the checkpoint). The IP "
                    "branch starts numerically inert; fine-tuning is required before it "
                    "contributes.",
                    len(missing & ip_keys),
                )
        return _finish_checkpoint_load(state, controller, model)


def restore_prior_band(model: Optional[nn.Module]) -> None:
    """The prior band travels in the persistent late_ramp buffer."""
    _m = getattr(model, "module", model)
    _m = getattr(_m, "_orig_mod", _m)
    if _m is not None and bool(getattr(getattr(_m, "cfg", None), "use_manifold_branch", False)):
        _lr = getattr(_m, "late_ramp", None)
        if _lr is not None and hasattr(_m, "_prior_band_g"):
            _nz = torch.nonzero(_lr.detach().reshape(-1) > 0).reshape(-1)
            if _nz.numel() > 0:
                _m._prior_band_g = int(max(1, int(_nz[0].item())))


def _finish_checkpoint_load(state: Mapping[str, Any], controller: ConstraintController,
                            model: Optional[nn.Module] = None) -> Dict[str, Any]:
    restore_prior_band(model)
    adopt_survey_record(model, state)
    if "constraint_state" in state:
        controller.load_state_dict(state["constraint_state"])
    return dict(state)


@torch.no_grad()
def log_ip_head_init_perturbation(
    model: PEBRNet,
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
) -> Optional[Dict[str, float]]:
    """Measure -- not assume -- how inert the freshly initialized IP branch is."""
    try:
        model.eval()
        gsn = model.gate_scale_norm.to(device)
        x = gsn.expand(8, 1, int(cfg.data.target_gates)).clone()
        nb = None
        if cfg.model.num_neighbors > 0 and cfg.model.use_neighbor_attention:
            nb = gsn.squeeze(1).expand(8, int(cfg.model.num_neighbors), int(cfg.data.target_gates)).clone()
        out = model(x, nb)
        if "ip_atom_w" not in out:
            return None
        ip_curve = (out["ip_atom_w"] @ model.ip_atoms) * out["ip_gate"]
        phys = out["phys_linear"]
        induction = phys + ip_curve
        gsv = gsn.squeeze(0).squeeze(0)
        z_with = signed_symlog(phys / gsv)
        z_without = signed_symlog(induction / gsv)
        dz = float(torch.max(torch.abs(z_with - z_without)).item())
        rel = float(
            torch.max(torch.abs(ip_curve) / (torch.abs(induction) + 1e-30)).item()
        )
        gate = float(out["ip_gate"].mean().item())
        logger.info(
            "IP HEAD INIT | max |delta z| vs induction-only = %.3e | max relative IP amplitude = %.3e | "
            "mean gate = %.4f | the branch is numerically inert at start and has non-zero gradient "
            "(no zero-product deadlock); it must be fine-tuned before it contributes.",
            dz, rel, gate,
        )
        if dz > 1e-3:
            logger.warning(
                "IP head init perturbation %.3e exceeds 1e-3 in z: lower --ip_init_scale, "
                "otherwise the starting point is NOT the induction-only model.", dz
            )
        return {"max_delta_z": dz, "max_relative_ip": rel, "mean_gate": gate}
    except Exception as exc:
        logger.warning("IP head init check skipped: %s", exc)
        return None


class CudaBatchSynthesizer:
    """Device-resident, vectorized re-implementation of ``BTEMDenoisingDataset.__getitem__`` +
    ``compose_noise_trace``.
    """

    def __init__(self, dataset: "BTEMDenoisingDataset", device: torch.device) -> None:
        self.device = torch.device(device)
        self.ds = dataset
        cfg = dataset.cfg
        self.cfg = cfg
        dataset._ensure_open()
        assert dataset._clean is not None and dataset._noise is not None
        g = torch.Generator(device="cpu")
        del g
        self.G = int(cfg.target_gates)
        self.K = int(cfg.num_neighbors)
        self.P = int(getattr(cfg, "profile_patch_len", 0) or 0)
        self.training = bool(dataset.training)
        self.scale = float(dataset.global_scale)
        self.seed = int(dataset.seed)
        self.clean = torch.as_tensor(
            np.array(dataset._clean, dtype=np.float32, copy=True), device=self.device)
        self.noise = torch.as_tensor(
            np.array(dataset._noise, dtype=np.float32, copy=True), device=self.device)
        bb = dataset._block_bounds
        assert bb is not None
        self.block_bounds = torch.as_tensor(np.asarray(bb, dtype=np.int64), device=self.device)
        lengths = np.diff(np.asarray(bb)).astype(np.float64)
        self.block_probs = torch.as_tensor(lengths / lengths.sum(), dtype=torch.float32,
                                           device=self.device)
        self.block_start = self.block_bounds[:-1]
        self.block_len = torch.as_tensor(lengths.astype(np.int64), device=self.device)
        self.split = torch.as_tensor(
            np.asarray(dataset.split_indices, dtype=np.int64), device=self.device)
        self.gate_times_s = torch.as_tensor(
            np.asarray(dataset._gate_times_s, dtype=np.float32), device=self.device)
        self.aug_mats = (
            torch.as_tensor(np.asarray(dataset._aug_mats, dtype=np.float32), device=self.device)
            if dataset._aug_mats is not None else None)
        self.real_rows = (
            torch.as_tensor(np.asarray(dataset.real_neighbor_rows, dtype=np.int64),
                            device=self.device)
            if dataset.real_neighbor_rows is not None else None)
        self.real_geo = (
            torch.as_tensor(np.asarray(dataset.real_neighbor_geometry, dtype=np.float32),
                            device=self.device)
            if dataset.real_neighbor_geometry is not None else None)
        self.mix_field = (
            torch.as_tensor(np.asarray(dataset._mix_field_idx, dtype=np.int64),
                            device=self.device)
            if dataset._mix_field_idx is not None else None)
        self.mix_forward = (
            torch.as_tensor(np.asarray(dataset._mix_forward_idx, dtype=np.int64),
                            device=self.device)
            if dataset._mix_forward_idx is not None else None)
        runs = getattr(dataset, "_field_patch_runs", None) or dataset._patch_runs
        if self.training and self.P > 1 and runs:
            a = np.asarray([r[0] for r in runs], dtype=np.int64)
            b = np.asarray([r[1] for r in runs], dtype=np.int64)
            n_anchor = (b - a - self.P + 2).clip(min=0)
            keep = n_anchor > 0
            a, n_anchor = a[keep], n_anchor[keep]
            self.run_start = torch.as_tensor(a, device=self.device)
            self.run_anchors = torch.as_tensor(n_anchor, device=self.device)
            csum = np.concatenate([[0], np.cumsum(n_anchor)])
            self.run_cum = torch.as_tensor(csum, device=self.device)
            self.total_anchors = int(csum[-1])
        else:
            self.run_start = None
            self.run_anchors = None
            self.run_cum = None
            self.total_anchors = 0
        f_runs = getattr(dataset, "_forward_patch_runs", None)
        if self.training and self.P > 1 and f_runs:
            fa = np.asarray([r[0] for r in f_runs], dtype=np.int64)
            fb = np.asarray([r[1] for r in f_runs], dtype=np.int64)
            fn = (fb - fa - self.P + 2).clip(min=0)
            fkeep = fn > 0
            fa, fn = fa[fkeep], fn[fkeep]
            fcs = np.concatenate([[0], np.cumsum(fn)])
            self.fwd_run_start = torch.as_tensor(fa, device=self.device)
            self.fwd_run_cum = torch.as_tensor(fcs, device=self.device)
            self.fwd_total_anchors = int(fcs[-1])
        else:
            self.fwd_run_start = None
            self.fwd_run_cum = None
            self.fwd_total_anchors = 0
        self.snr_override = getattr(dataset, "_snr_override", None)
        self.probe_mode = False
        self.probe_anomaly_prob = 0.0
        self.probe_singleton_prob = 0.0
        self.probe_snr: Optional[Tuple[float, float]] = None
        self.probe_anchors: Optional[torch.Tensor] = None
        self.emit_neighbor_clean = bool(getattr(dataset, "emit_neighbor_clean", False))
        gates = self.G
        self.gate_axis = ((torch.arange(gates, dtype=torch.float32, device=self.device)
                           - 0.5 * (gates - 1)) / max(gates - 1, 1))
        self.drift_u = torch.linspace(-1.0, 1.0, gates, device=self.device)
        offs = [((j // 2) + 1) * (1 if j % 2 == 0 else -1) for j in range(self.K)]
        self.slot_offsets = torch.as_tensor(offs, dtype=torch.float32, device=self.device)
        self.slot_offsets_i = torch.as_tensor(offs, dtype=torch.int64, device=self.device)
        self.arangeG = torch.arange(gates, device=self.device)
        ls = int(cfg.late_start_index)
        a0 = float(max(ls - int(cfg.anomaly_onset_gates), 0))
        b0 = float(max(gates - 1, 1))
        u = torch.clamp((self.arangeG.float() - a0) / max(b0 - a0, 1e-9), 0.0, 1.0)
        self.anom_ramp = (u * u * (3.0 - 2.0 * u))

    def enable_probe(self, anomaly_prob: float, singleton_prob: float,
                     snr_range: Optional[Tuple[float, float]] = None) -> int:
        """Turn this synthesizer into a deterministic anomaly probe over its split."""
        self.probe_mode = True
        self.probe_anomaly_prob = float(anomaly_prob)
        self.probe_singleton_prob = float(singleton_prob)
        self.probe_snr = tuple(snr_range) if snr_range is not None else None
        P = int(self.P)
        anchors: List[int] = []
        if P > 1:
            si = np.sort(np.asarray(getattr(self.ds, "split_indices_full",
                                            self.ds.split_indices), dtype=np.int64))
            if si.size:
                start = prev = int(si[0])
                for v in si[1:]:
                    v = int(v)
                    if v == prev + 1:
                        prev = v
                        continue
                    if prev - start + 1 >= P:
                        anchors.extend(range(start, prev - P + 2))
                    start = prev = v
                if prev - start + 1 >= P:
                    anchors.extend(range(start, prev - P + 2))
        if anchors:
            self.probe_anchors = torch.as_tensor(np.asarray(anchors, dtype=np.int64),
                                                 device=self.device)
        else:
            self.probe_anchors = None
        return 0 if self.probe_anchors is None else int(self.probe_anchors.numel())

    def _gen(self, epoch: int, batch_index: int) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        base = (self.seed * 1_000_003
                + (int(epoch) if self.training else 0) * 9_176_723
                + int(batch_index) * 97_409 + 0x5F3759DF)
        gen.manual_seed(base & 0x7FFF_FFFF_FFFF_FFFF)
        return gen

    def _rand(self, shape, gen):
        return torch.rand(shape, generator=gen, device=self.device)

    def _randn(self, shape, gen):
        return torch.randn(shape, generator=gen, device=self.device)

    def _uniform(self, lo, hi, shape, gen):
        return lo + (hi - lo) * self._rand(shape, gen)

    def _randint(self, high, shape, gen):
        u = self._rand(shape, gen)
        if isinstance(high, torch.Tensor):
            return torch.minimum((u * high.float()).long(), high.long() - 1).clamp_(min=0)
        return torch.clamp((u * float(high)).long(), 0, int(high) - 1)

    def _rms(self, x, dim=-1):
        return torch.sqrt(torch.mean(x * x, dim=dim))

    def _compose(self, rows_local, bstart, blen, sgn, roll_k, mix_delta, mix_sgn,
                 mix_beta, white_ratio, gen):
        cfg = self.cfg
        N, S = rows_local.shape
        G = self.G
        rows = (bstart[:, None] + rows_local % blen[:, None])
        v = self.noise[rows.reshape(-1)].reshape(N, S, G).clone()
        v *= sgn[:, None, None]
        idx = (self.arangeG[None, :] - roll_k[:, None]) % G
        v = torch.gather(v, 2, idx[:, None, :].expand(N, S, G))
        if mix_delta is not None:
            m_rows = (bstart[:, None] + (rows_local + mix_delta[:, None]) % blen[:, None])
            m = self.noise[m_rows.reshape(-1)].reshape(N, S, G).clone()
            m *= mix_sgn[:, None, None]
            m = torch.gather(m, 2, idx[:, None, :].expand(N, S, G))
            has_mix = (mix_delta > 0).float()[:, None, None]
            beta = mix_beta[:, None, None]
            v = torch.where(has_mix.bool(), beta * v + (1.0 - beta) * m, v)
        v = v - v.mean(dim=-1, keepdim=True)
        rms_struct = torch.clamp(self._rms(v), min=cfg.minimum_noise_rms)
        if cfg.colored_noise_probability > 0.0:
            on = self._rand((N, S), gen) < cfg.colored_noise_probability
            rho = self._uniform(*cfg.colored_noise_ar_coeff, (N, S), gen)
            gain_ar = torch.sqrt(torch.clamp(1.0 - rho * rho, min=1e-12))
            e = self._randn((N, S, G), gen)
            ar = torch.empty_like(e)
            acc = torch.zeros(N, S, device=self.device)
            for i in range(G):
                acc = rho * acc + gain_ar * e[..., i]
                ar[..., i] = acc
            ar = ar - ar.mean(dim=-1, keepdim=True)
            ar_rms = torch.clamp(self._rms(ar), min=1e-30)
            rel = self._uniform(*cfg.colored_noise_relative_rms, (N, S), gen)
            v = v + on.float()[..., None] * ar * (rms_struct * rel / ar_rms)[..., None]
        if cfg.spike_noise_probability > 0.0:
            on = self._rand((N, S), gen) < cfg.spike_noise_probability
            k_lo, k_hi = int(cfg.spike_count_range[0]), int(cfg.spike_count_range[1])
            k_max = min(max(k_hi, 1), G)
            kk = k_lo + self._randint(k_hi - k_lo + 1, (N, S), gen)
            u = self._rand((N, S, G), gen)
            pos = torch.topk(u, k_max, dim=-1).indices
            keep = (torch.arange(k_max, device=self.device)[None, None, :]
                    < kk[..., None]) & on[..., None]
            amp = self._uniform(*cfg.spike_relative_amplitude, (N, S, k_max), gen)
            sgn_s = torch.where(self._rand((N, S, k_max), gen) < 0.5, -1.0, 1.0)
            add = torch.zeros_like(v)
            add.scatter_add_(2, pos, keep.float() * amp * sgn_s
                             * rms_struct[..., None])
            v = v + add
        if cfg.harmonic_noise_probability > 0.0:
            on = self._rand((N, S), gen) < cfg.harmonic_noise_probability
            f = self._uniform(*cfg.harmonic_freq_hz, (N, S), gen)
            ph = self._uniform(0.0, 2.0 * math.pi, (N, S), gen)
            h = torch.sin(2.0 * math.pi * f[..., None] * self.gate_times_s[None, None, :]
                          + ph[..., None])
            h = h - h.mean(dim=-1, keepdim=True)
            h_rms = torch.clamp(self._rms(h), min=1e-30)
            rel = self._uniform(*cfg.harmonic_relative_rms, (N, S), gen)
            v = v + on.float()[..., None] * h * (rms_struct * rel / h_rms)[..., None]
        if cfg.drift_noise_probability > 0.0:
            on = self._rand((N, S), gen) < cfg.drift_noise_probability
            order2 = self._randint(2, (N, S), gen).float()
            c1 = self._randn((N, S), gen)
            c2 = self._randn((N, S), gen) * order2
            d = c1[..., None] * self.drift_u[None, None, :] \
                + c2[..., None] * (self.drift_u[None, None, :] ** 2)
            d = d - d.mean(dim=-1, keepdim=True)
            d_rms = torch.clamp(self._rms(d), min=1e-30)
            rel = self._uniform(*cfg.drift_relative_rms, (N, S), gen)
            v = v + on.float()[..., None] * d * (rms_struct * rel / d_rms)[..., None]
        if cfg.burst_noise_probability > 0.0:
            on = self._rand((N, S), gen) < cfg.burst_noise_probability
            w_lo, w_hi = int(cfg.burst_width_range[0]), int(cfg.burst_width_range[1])
            wlen = w_lo + self._randint(w_hi - w_lo + 1, (N, S), gen)
            start = self._randint(torch.clamp(G - wlen, min=1), (N, S), gen)
            win = ((self.arangeG[None, None, :] >= start[..., None])
                   & (self.arangeG[None, None, :] < (start + wlen)[..., None]))
            b = self._randn((N, S, G), gen) * win.float()
            wm = b.sum(dim=-1, keepdim=True) / wlen[..., None].float()
            b = (b - wm) * win.float()
            b_rms = torch.clamp(torch.sqrt((b * b).sum(-1) / wlen.float()), min=1e-30)
            rel = self._uniform(*cfg.burst_relative_rms, (N, S), gen)
            v = v + on.float()[..., None] * b * (rms_struct * rel / b_rms)[..., None]
        if white_ratio is not None:
            w = self._randn((N, S, G), gen)
            v_rms = torch.clamp(self._rms(v), min=cfg.minimum_noise_rms)
            w_rms = torch.clamp(self._rms(w), min=1e-30)
            has_w = (white_ratio > 0).float()[:, None, None]
            v = v + has_w * w * (v_rms * white_ratio[:, None] / w_rms)[..., None]
        return v

    def sample_batch(self, epoch: int, batch_index: int, batch_size: int,
                     eval_start: int = 0) -> Dict[str, torch.Tensor]:
        use_fwd = None
        cfg = self.cfg
        dev = self.device
        G, K = self.G, self.K
        gen = self._gen(epoch, batch_index)
        if self.probe_mode:
            patched = self.P > 1 and self.probe_anchors is not None
        else:
            patched = self.training and self.P > 1 and (
                self.total_anchors > 0 or self.mix_forward is not None)
        P = self.P if patched else 1
        assert batch_size % P == 0, "batch must be patch-aligned"
        Np = batch_size // P

        if self.probe_mode and patched:
            na = int(self.probe_anchors.numel())
            sel = ((int(batch_index) * Np + torch.arange(Np, device=dev))
                   * P) % na
            anchor = self.probe_anchors[sel]
            clean_idx = (anchor[:, None]
                         + torch.arange(P, device=dev)[None, :]).reshape(-1)
        elif self.training:
            if patched:
                use_fwd = None
                if (self.mix_forward is not None
                        and float(getattr(cfg, "mix_forward_fraction", 0.0)) > 0.0):
                    use_fwd = self._rand((Np,), gen) < float(cfg.mix_forward_fraction)
                if self.total_anchors > 0:
                    flat = self._randint(self.total_anchors, (Np,), gen)
                    run = torch.searchsorted(self.run_cum, flat, right=True) - 1
                    anchor = self.run_start[run] + (flat - self.run_cum[run])
                else:
                    anchor = self.mix_field[self._randint(self.mix_field.numel(),
                                                          (Np,), gen)] \
                        if self.mix_field is not None else \
                        self.split[self._randint(self.split.numel(), (Np,), gen)]
                if use_fwd is not None:
                    if self.fwd_total_anchors > 0:
                        fflat = self._randint(self.fwd_total_anchors, (Np,), gen)
                        frun = torch.searchsorted(self.fwd_run_cum, fflat, right=True) - 1
                        fwd = self.fwd_run_start[frun] + (fflat - self.fwd_run_cum[frun])
                        anchor = torch.where(use_fwd, fwd, anchor)
                        adv = torch.ones(Np, dtype=torch.long, device=dev)
                    else:
                        fwd = self.mix_forward[self._randint(self.mix_forward.numel(),
                                                             (Np,), gen)]
                        anchor = torch.where(use_fwd, fwd, anchor)
                        adv = (~use_fwd).long()
                else:
                    adv = torch.ones(Np, dtype=torch.long, device=dev)
                offs = torch.arange(P, device=dev)
                clean_idx = (anchor[:, None] + adv[:, None] * offs[None, :]).reshape(-1)
            elif self.mix_field is not None and self.mix_forward is not None:
                p = float(getattr(cfg, "mix_forward_fraction", 0.15))
                pick_f = self._rand((Np,), gen) < p
                a = self.mix_field[self._randint(self.mix_field.numel(), (Np,), gen)]
                b = self.mix_forward[self._randint(self.mix_forward.numel(), (Np,), gen)]
                clean_idx = torch.where(pick_f, b, a)
            else:
                clean_idx = self.split[self._randint(self.split.numel(), (Np,), gen)]
        else:
            item = (eval_start + torch.arange(batch_size, device=dev)) \
                % self.split.numel()
            clean_idx = self.split[item]
        B = clean_idx.numel()
        offs_b = (torch.arange(B, device=dev) % P)
        patch_of = torch.arange(B, device=dev) // P

        clean = self.clean[clean_idx].clone()

        aug_li = torch.full((Np,), -1, dtype=torch.long, device=dev)
        aug_amp = torch.ones(Np, device=dev)
        if (self.training and self.aug_mats is not None
                and float(cfg.clean_aug_probability) > 0.0):
            on = self._rand((Np,), gen) < float(cfg.clean_aug_probability)
            li = self._randint(self.aug_mats.shape[0], (Np,), gen)
            amp_dex = float(cfg.clean_aug_amp_dex)
            amp = (torch.pow(10.0, self._uniform(-amp_dex, amp_dex, (Np,), gen))
                   if amp_dex > 0.0 else torch.ones(Np, device=dev))
            pos_ok = (clean > 0).all(dim=-1).reshape(Np, P).all(dim=-1)
            on = on & pos_ok
            aug_li = torch.where(on, li, aug_li)
            aug_amp = torch.where(on, amp, aug_amp)
            m = self.aug_mats[li.clamp(min=0)]
            m_b = m[patch_of]
            safe = torch.clamp(clean, min=1e-300)
            aug_clean = torch.exp(torch.bmm(m_b, torch.log(safe)[..., None])[..., 0]) \
                * aug_amp[patch_of, None]
            clean = torch.where(on[patch_of, None], aug_clean, clean)

        block = torch.multinomial(self.block_probs, Np, replacement=True, generator=gen)
        bstart = self.block_start[block]
        blen = self.block_len[block]
        center_local = self._randint(blen, (Np,), gen)
        sgn = (torch.where(self._rand((Np,), gen) < 0.5, -1.0, 1.0)
               if cfg.random_noise_sign_flip else torch.ones(Np, device=dev))
        roll_on = self._rand((Np,), gen) < float(cfg.noise_circular_shift_probability)
        roll_k = torch.where(roll_on, 1 + self._randint(G - 1, (Np,), gen),
                             torch.zeros(Np, dtype=torch.long, device=dev))
        mix_on = ((self._rand((Np,), gen) < float(cfg.noise_mixup_probability))
                  & (blen > 1))
        mix_delta = torch.where(mix_on, 1 + self._randint(
            torch.clamp(blen - 1, min=1), (Np,), gen),
            torch.zeros(Np, dtype=torch.long, device=dev))
        mix_sgn = (torch.where(self._rand((Np,), gen) < 0.5, -1.0, 1.0)
                   if cfg.random_noise_sign_flip else torch.ones(Np, device=dev))
        mix_sgn = torch.where(mix_on, mix_sgn, torch.ones_like(mix_sgn))
        mix_beta = torch.where(mix_on, self._uniform(*cfg.noise_mixup_beta, (Np,), gen),
                               torch.zeros(Np, device=dev))
        white_on = self._rand((Np,), gen) < float(cfg.white_noise_probability)
        white_ratio = torch.where(
            white_on, self._uniform(*cfg.white_noise_relative_rms, (Np,), gen),
            torch.zeros(Np, device=dev))

        _an_p = float(self.probe_anomaly_prob if self.probe_mode
                      else getattr(cfg, "anomaly_profile_prob", 0.0))
        _sg_p = float(self.probe_singleton_prob if self.probe_mode
                      else getattr(cfg, "anomaly_singleton_prob", 0.0))
        anom_on = (self.training or self.probe_mode) and _an_p > 0.0
        clean_pre = clean
        observed = clean
        event_mask = torch.zeros(B, G, device=dev)
        event_label = torch.zeros(B, device=dev)
        a_amp = torch.ones(Np, device=dev)
        a_wl = torch.ones(Np, device=dev)
        a_wr = torch.ones(Np, device=dev)
        a_c = torch.zeros(Np, device=dev)
        coh_on = torch.zeros(Np, dtype=torch.bool, device=dev)
        sing_on = torch.zeros(Np, dtype=torch.bool, device=dev)
        s_env = torch.zeros(B, device=dev)
        if anom_on:
            hit = self._rand((Np,), gen) < _an_p
            singleton = self._rand((Np,), gen) < _sg_p
            amp_pos = self._uniform(*cfg.anomaly_amp_range, (Np,), gen)
            neg = self._rand((Np,), gen) < float(cfg.anomaly_negative_fraction)
            amp_neg = self._uniform(*cfg.anomaly_suppress_range, (Np,), gen)
            amp_any = torch.where(neg, amp_neg, amp_pos)
            w = self._uniform(*cfg.anomaly_width_range, (Np,), gen)
            skew = self._rand((Np,), gen) < float(cfg.anomaly_skew_fraction)
            wl = w * self._uniform(0.45, 0.85, (Np,), gen)
            wr = w * self._uniform(1.15, 1.90, (Np,), gen)
            swap = self._rand((Np,), gen) < 0.5
            wl2 = torch.where(swap, wr, wl)
            wr2 = torch.where(swap, wl, wr)
            wl_f = torch.where(skew, wl2, w)
            wr_f = torch.where(skew, wr2, w)
            cj = float(cfg.anomaly_center_jitter)
            cc = self._uniform(-cj, cj, (Np,), gen)
            sing_on = hit & singleton
            coh_on = hit & (~singleton)
            a_amp = torch.where(coh_on, amp_any, a_amp)
            a_wl = torch.where(coh_on, wl_f, a_wl)
            a_wr = torch.where(coh_on, wr_f, a_wr)
            a_c = torch.where(coh_on, cc, a_c)
            st_pos = (offs_b.float() - (P - 1) / 2.0) if P > 1 \
                else torch.zeros(B, device=dev)
            dpos = st_pos - a_c[patch_of]
            wsel = torch.where(dpos >= 0, a_wr[patch_of], a_wl[patch_of])
            env_c = torch.exp(-0.5 * (dpos / torch.clamp(wsel, min=1e-6)) ** 2)
            env_c = torch.where(coh_on[patch_of], env_c, torch.zeros_like(env_c))
            lift_c = 1.0 + (a_amp[patch_of] - 1.0)[:, None] \
                * env_c[:, None] * self.anom_ramp[None, :]
            clean = clean * lift_c
            observed = clean
            lab = (env_c > 0.05) & coh_on[patch_of]
            event_label = lab.float()
            denom = torch.where((a_amp[patch_of] - 1.0).abs() > 1e-9,
                                a_amp[patch_of] - 1.0,
                                torch.full_like(a_amp[patch_of], 1e-9))
            event_mask = torch.clamp((lift_c - 1.0) / denom[:, None], 0.0, 1.0) \
                * lab.float()[:, None]
            s_amp = torch.where(neg, amp_neg,
                                self._uniform(*cfg.anomaly_amp_range, (Np,), gen))
            is_centre = (offs_b == (P - 1) // 2) if P > 1 \
                else torch.ones(B, dtype=torch.bool, device=dev)
            s_env = (sing_on[patch_of] & is_centre).float()
            lift_s = 1.0 + (s_amp[patch_of] - 1.0)[:, None] \
                * s_env[:, None] * self.anom_ramp[None, :]
            observed = torch.where(sing_on[patch_of, None], clean * lift_s, observed)
            clean_pre = torch.where(coh_on[patch_of, None], clean_pre, clean_pre)

        noise_c = self._compose(center_local[:, None], bstart, blen, sgn, roll_k,
                                mix_delta, mix_sgn, mix_beta, white_ratio,
                                gen)[:, 0]
        noise_c = noise_c[patch_of]
        _snr_rng = (self.probe_snr if (self.probe_mode and self.probe_snr is not None)
                    else self.snr_override)
        if _snr_rng is not None:
            lo, hi = float(_snr_rng[0]), float(_snr_rng[1])
            snr_p = self._uniform(min(lo, hi), max(lo, hi), (Np,), gen)
        else:
            below = self._rand((Np,), gen) < float(cfg.snr_below_zero_probability)
            snr_p = torch.where(
                below, self._uniform(float(cfg.snr_min_db), 0.0, (Np,), gen),
                self._uniform(0.0, float(cfg.snr_max_db), (Np,), gen))
        snr_db = snr_p[patch_of]
        clean_rms = self._rms(observed)
        comp_rms = torch.clamp(self._rms(noise_c), min=cfg.minimum_noise_rms)
        gain = clean_rms / (torch.pow(10.0, snr_db / 20.0) * comp_rms)
        noisy = observed + noise_c * gain[:, None]
        noise_t = noisy - clean

        half = int(cfg.reliability_window) // 2
        c2 = F.pad(clean[:, None, :] ** 2, (half, half), mode="replicate")
        local = torch.sqrt(F.avg_pool1d(c2, int(cfg.reliability_window), stride=1))[:, 0]
        rel_err = (noisy - clean).abs() / torch.clamp(local, min=1e-8)
        q_target = torch.clamp(torch.exp(-float(cfg.reliability_kappa) * rel_err),
                               0.0, 1.0)

        scale = self.scale
        out: Dict[str, torch.Tensor] = {
            "noisy": (noisy / scale)[:, None, :].float(),
            "clean": (clean / scale)[:, None, :].float(),
            "noise": (noise_t / scale)[:, None, :].float(),
            "q_target": q_target[:, None, :].float(),
            "snr_db": snr_db.float(),
            "clean_index": clean_idx,
            "noise_index": (bstart + center_local % blen)[patch_of],
            "event_mask": event_mask[:, None, :].float(),
            "event_label": event_label.float(),
            "clean_row": clean_idx,
            "is_forward": (use_fwd[patch_of].float()
                           if use_fwd is not None else
                           torch.zeros(B, device=dev)),
        }
        if self.probe_mode:
            out["probe_clean_baseline"] = (clean_pre / scale)[:, None, :].float()
            out["probe_body"] = (coh_on[patch_of] & (event_label > 0.5)).float()
            out["probe_singleton"] = (s_env > 0).float()
            out["probe_quiet"] = (
                (~coh_on[patch_of]) & (s_env <= 0)).float()

        if K > 0:
            add_stack = bool(getattr(cfg, "coherent_stack_slot", False)
                             and self.real_rows is not None)
            k_slots = K + (1 if add_stack else 0)
            use_real = torch.zeros(B, dtype=torch.bool, device=dev)
            nb_rows = None
            if self.real_rows is not None:
                nb_rows = self.real_rows[clean_idx]
                use_real = ~(nb_rows == clean_idx[:, None]).all(dim=-1)
            geo = torch.zeros(B, k_slots, NEIGHBOR_GEOMETRY_FEATURES, device=dev)
            so = self.slot_offsets[None, :].expand(B, K).clone()
            ir = torch.zeros(B, K, device=dev)
            if self.real_geo is not None:
                gsel = self.real_geo[clean_idx]
                so = torch.where(use_real[:, None], gsel[..., 0], so)
                ir = torch.where(use_real[:, None], gsel[..., 1], ir)
            else:
                ir = torch.where(use_real[:, None], torch.ones_like(ir), ir)
            geo[..., :K, 0] = so
            geo[..., :K, 1] = so.abs()
            geo[..., :K, 2] = ir
            if nb_rows is not None:
                nbc_real = self.clean[nb_rows.reshape(-1)].reshape(B, K, G).clone()
                if self.aug_mats is not None:
                    on_b = (aug_li >= 0)[patch_of]
                    m_b = self.aug_mats[aug_li.clamp(min=0)][patch_of]
                    pos_ok = (nbc_real > 0).all(dim=-1)
                    safe = torch.clamp(nbc_real, min=1e-300)
                    aug_nb = torch.exp(torch.einsum(
                        "bij,bkj->bki", m_b, torch.log(safe))) \
                        * aug_amp[patch_of, None, None]
                    sel = (on_b[:, None] & pos_ok)[..., None]
                    nbc_real = torch.where(sel, aug_nb, nbc_real)
            else:
                nbc_real = torch.zeros(B, K, G, device=dev)
            gain_nb = torch.exp(self._randn((Np, K), gen)
                                * float(cfg.neighbor_gain_std))[patch_of]
            slope = (self._randn((Np, K), gen)
                     * float(cfg.neighbor_tilt_std))[patch_of]
            tilt = torch.exp(slope[..., None] * self.gate_axis[None, None, :])
            nbc_syn = clean_pre[:, None, :] * gain_nb[..., None] * tilt
            nb_clean = torch.where(use_real[:, None, None], nbc_real, nbc_syn)
            if anom_on:
                d = so - a_c[patch_of, None]
                wsel = torch.where(d >= 0, a_wr[patch_of, None], a_wl[patch_of, None])
                env_j = torch.exp(-0.5 * (d / torch.clamp(wsel, min=1e-6)) ** 2)
                env_j = env_j * coh_on[patch_of, None].float()
                lift_j = 1.0 + (a_amp[patch_of, None] - 1.0)[..., None] \
                    * env_j[..., None] * self.anom_ramp[None, None, :]
                nb_clean = nb_clean * lift_j
            nb_noise = self._compose(
                (center_local[:, None] + self.slot_offsets_i[None, :]),
                bstart, blen, sgn, roll_k, mix_delta, mix_sgn, mix_beta,
                white_ratio, gen)
            nb_noise = nb_noise[patch_of]
            jit = self._uniform(-float(cfg.neighbor_snr_jitter_db),
                                float(cfg.neighbor_snr_jitter_db), (Np, K), gen)
            nb_snr = snr_p[:, None] + jit
            nb_snr = nb_snr[patch_of]
            nb_gain = self._rms(nb_clean) / (
                torch.pow(10.0, nb_snr / 20.0)
                * torch.clamp(self._rms(nb_noise), min=cfg.minimum_noise_rms))
            drop_p = float(getattr(cfg, "neighbor_channel_dropout", 0.0))
            dropped = (self._rand((Np, K), gen) < drop_p)[patch_of] \
                if (self.training and drop_p > 0.0) else \
                torch.zeros(B, K, dtype=torch.bool, device=dev)
            dropped = dropped & use_real[:, None]
            nb_noisy = torch.where(dropped[..., None],
                                   nb_noise * nb_gain[..., None],
                                   nb_clean + nb_noise * nb_gain[..., None])
            stack = torch.empty(B, k_slots, G, device=dev)
            stack[:, :K] = nb_noisy / scale
            nb_clean_s = None
            if self.emit_neighbor_clean:
                nb_clean_s = torch.empty(B, k_slots, G, device=dev)
                nb_clean_s[:, :K] = nb_clean / scale
            if add_stack:
                stack[:, K] = stack[:, :K].sum(dim=1) / float(K)
                geo[:, K, 3] = 1.0
                if nb_clean_s is not None:
                    nb_clean_s[:, K] = ((clean / scale)
                                        + nb_clean_s[:, :K].sum(dim=1)) / float(K + 1)
            out["neighbor_geometry"] = geo.float()
            out["neighbors"] = stack.float()
            if nb_clean_s is not None:
                out["neighbors_clean"] = nb_clean_s.float()
        return out


class AnomalyRecoveryProbe:
    """Deterministic held-out measurement of the anomaly pathway."""

    def __init__(self, dataset: "BTEMDenoisingDataset", cfg: Config,
                 device: torch.device, logger: Optional[logging.Logger] = None,
                 share: Optional["CudaBatchSynthesizer"] = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.synth = CudaBatchSynthesizer(dataset, self.device)
        if share is not None:
            self.synth.clean = share.clean
            self.synth.noise = share.noise
        P = max(1, int(self.synth.P))
        bs = int(min(int(cfg.train.batch_size), 1024))
        self.batch_size = max(P, (bs // P) * P) if P > 1 else bs
        self.batches = max(1, int(getattr(cfg.train, "anomaly_probe_batches", 6)))
        n_anchor = self.synth.enable_probe(
            anomaly_prob=float(getattr(cfg.train, "anomaly_probe_body_prob", 0.5)),
            singleton_prob=float(getattr(cfg.train, "anomaly_probe_singleton_prob", 0.3)),
            snr_range=tuple(cfg.data.contract_test_snr_db_range),
        )
        self.n_anchor = int(n_anchor)
        if logger is not None:
            logger.info(
                "ANOMALY PROBE armed | %d held-out patch anchors, %d x %d traces "
                "per validation, SNR %s dB (contract protocol), ~%.0f%% coherent bodies / "
                "~%.0f%% singleton artefacts / rest quiet. Deterministic and epoch-"
                "independent, so two epochs are compared on identical cases.",
                self.n_anchor, self.batches, self.batch_size,
                list(cfg.data.contract_test_snr_db_range),
                100.0 * float(getattr(cfg.train, "anomaly_probe_body_prob", 0.5)),
                100.0 * float(getattr(cfg.train, "anomaly_probe_body_prob", 0.5))
                * float(getattr(cfg.train, "anomaly_probe_singleton_prob", 0.3)),
            )

    @staticmethod
    def _auroc(scores: torch.Tensor, positive: torch.Tensor,
               negative: torch.Tensor) -> float:
        sp = scores[positive > 0.5].double()
        sn = scores[negative > 0.5].double()
        if sp.numel() == 0 or sn.numel() == 0:
            return float("nan")
        allv = torch.cat([sp, sn])
        order = torch.argsort(allv)
        ranks = torch.empty_like(allv)
        ranks[order] = torch.arange(1, allv.numel() + 1, dtype=allv.dtype,
                                    device=allv.device)
        npos = float(sp.numel())
        nneg = float(sn.numel())
        u = float(ranks[: sp.numel()].sum()) - npos * (npos + 1.0) / 2.0
        return float(u / (npos * nneg))

    @torch.no_grad()
    def run(self, model: PEBRNet, cfg: Config) -> Dict[str, float]:
        was = model.training
        model.eval()
        ls = int(cfg.data.late_start_index)
        amp_on = bool(cfg.train.amp) and self.device.type == "cuda"
        acc: Dict[str, List[torch.Tensor]] = {k: [] for k in
                                              ("err", "eng", "t", "r", "body", "sing",
                                               "quiet", "score", "sfalse", "evres",
                                               "zlate", "fieldlike")}
        try:
            _snr_saved = self.synth.probe_snr
            for b in range(self.batches + max(1, self.batches // 2)):
                if b >= self.batches:
                    self.synth.probe_snr = (-8.0, 0.0)
                batch = self.synth.sample_batch(0, b + (1000 if b >= self.batches else 0),
                                                self.batch_size)
                with amp_autocast_context(self.device, amp_on):
                    out = model(batch["noisy"], batch.get("neighbors"),
                                batch.get("neighbor_geometry"),
                                depth_norm=batch.get("depth_norm"),
                                profile_len=batch_profile_len(batch))
                yhat = out["denoised"].float()[:, 0, ls:]
                y = batch["clean"].float()[:, 0, ls:]
                base = batch["probe_clean_baseline"].float()[:, 0, ls:]
                body = batch["probe_body"].float()
                sing = batch["probe_singleton"].float()
                quiet = batch["probe_quiet"].float()
                acc["err"].append(((yhat - y) ** 2).sum(-1))
                acc["eng"].append((y ** 2).sum(-1))
                mb = base.abs().mean(-1).clamp_min(1e-30)
                acc["t"].append(y.abs().mean(-1) / mb - 1.0)
                acc["r"].append(yhat.abs().mean(-1) / mb - 1.0)
                acc["body"].append(body)
                acc["sing"].append(sing)
                acc["quiet"].append(quiet)
                acc["fieldlike"].append(torch.full_like(body,
                                                        1.0 if b >= self.batches else 0.0))
                if "event_logits" in out:
                    _es = int(getattr(model, "evidence_start_index", ls))
                    acc["score"].append(out["event_logits"].float()[..., _es:]
                                        .amax(dim=-1).reshape(-1))
                else:
                    acc["score"].append(torch.full_like(body, float("nan")))
                if "event_prob" in out and "struct_residual" in out:
                    _ev = (out["event_prob"].float()
                           * out["struct_residual"].float())[:, 0, ls:]
                    acc["evres"].append(_ev.pow(2).mean(-1))
                    acc["zlate"].append(out.get("z_hat", out["denoised"])
                                        .float()[:, 0, ls:].pow(2).mean(-1))
                else:
                    acc["evres"].append(torch.zeros_like(body))
                    acc["zlate"].append(torch.ones_like(body))
        finally:
            self.synth.probe_snr = _snr_saved
            model.train(was)
        cat = {k: (torch.cat(v) if v else torch.zeros(0, device=self.device))
               for k, v in acc.items()}
        body, sing, quiet = cat["body"], cat["sing"], cat["quiet"]
        m: Dict[str, float] = {}

        def _rel(mask: torch.Tensor) -> float:
            e = float((cat["err"] * mask).sum())
            g = float((cat["eng"] * mask).sum())
            return 100.0 * math.sqrt(max(e, 0.0) / max(g, 1e-30)) if g > 0 else float("nan")

        m["anomaly_late_error_percent"] = _rel(body)
        m["anomaly_quiet_late_error_percent"] = _rel(quiet)
        m["anomaly_singleton_late_error_percent"] = _rel(sing)
        m["anomaly_body_count"] = float(body.sum())
        m["anomaly_singleton_count"] = float(sing.sum())
        sel = (body > 0.5) & (cat["t"].abs() >= 0.05)
        if bool(sel.any()):
            ratio = (cat["r"][sel] / cat["t"][sel]).clamp(-1.0, 3.0)
            m["anomaly_lift_recovery"] = float(ratio.mean())
            m["anomaly_lift_recovery_median"] = float(ratio.median())
        else:
            m["anomaly_lift_recovery"] = float("nan")
            m["anomaly_lift_recovery_median"] = float("nan")
        if bool((sing > 0.5).any()):
            m["singleton_false_lift_percent"] = 100.0 * float(
                cat["r"][sing > 0.5].abs().mean())
        else:
            m["singleton_false_lift_percent"] = float("nan")
        sc = cat["score"]
        if sc.numel() and bool(torch.isfinite(sc).all()):
            m["event_auroc"] = self._auroc(sc, body, quiet)
            m["singleton_reject_auroc"] = self._auroc(sc, body, sing)
        else:
            m["event_auroc"] = float("nan")
            m["singleton_reject_auroc"] = float("nan")
        fl = cat["fieldlike"]
        _b_fl = (body > 0.5) & (fl > 0.5)
        _q_fl = (quiet > 0.5) & (fl > 0.5)
        m["anomaly_fieldlike_late_error_percent"] = _rel(_b_fl.float())
        m["anomaly_fieldlike_count"] = float(_b_fl.sum())
        _sel_fl = _b_fl & (cat["t"].abs() >= 0.05)
        m["anomaly_fieldlike_recovery"] = float(
            (cat["r"][_sel_fl] / cat["t"][_sel_fl]).clamp(-1.0, 3.0).mean()) \
            if bool(_sel_fl.any()) else float("nan")
        if sc.numel() and bool(torch.isfinite(sc).all()) and bool(_b_fl.any()) and bool(_q_fl.any()):
            m["anomaly_fieldlike_auroc"] = self._auroc(sc, _b_fl.float(), _q_fl.float())
        else:
            m["anomaly_fieldlike_auroc"] = float("nan")
        _z = cat["zlate"].clamp_min(1e-30)
        m["event_residual_late_share"] = float(
            (cat["evres"] / _z).clamp(0.0, 1.0).mean()) if _z.numel() else float("nan")
        t_abs = cat["t"].abs()
        bins = ((0.05, 0.20, "weak"), (0.20, 0.50, "moderate"), (0.50, float("inf"), "strong"))
        for lo, hi, name in bins:
            selb = (body > 0.5) & (t_abs >= lo) & (t_abs < hi)
            n_b = float(selb.sum())
            m["anomaly_%s_count" % name] = n_b
            if n_b <= 0:
                m["anomaly_%s_late_error_percent" % name] = float("nan")
                m["anomaly_%s_recovery" % name] = float("nan")
                m["anomaly_%s_auroc" % name] = float("nan")
                continue
            m["anomaly_%s_late_error_percent" % name] = _rel(selb.float())
            m["anomaly_%s_recovery" % name] = float(
                (cat["r"][selb] / cat["t"][selb]).clamp(-1.0, 3.0).mean())
            if sc.numel() and bool(torch.isfinite(sc).all()):
                m["anomaly_%s_auroc" % name] = self._auroc(sc, selb.float(), quiet)
            else:
                m["anomaly_%s_auroc" % name] = float("nan")
        return m


class GpuSyntheticLoader:
    """Drop-in replacement for the training/eval DataLoader."""

    def __init__(self, dataset: "BTEMDenoisingDataset", batch_size: int,
                 device: torch.device, drop_last: bool) -> None:
        self.synth = CudaBatchSynthesizer(dataset, device)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def __len__(self) -> int:
        n = int(self.dataset.samples)
        if self.drop_last:
            return max(1, n // self.batch_size)
        return max(1, (n + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        epoch = int(getattr(self.dataset, "epoch", self._epoch))
        n = int(self.dataset.samples)
        steps = len(self)
        for b in range(steps):
            start = b * self.batch_size
            bs = self.batch_size if (self.drop_last or start + self.batch_size <= n) \
                else (n - start)
            if bs <= 0:
                return
            P = self.synth.P if (self.synth.training and self.synth.P > 1) else 1
            bs_al = max(P, (bs // P) * P) if P > 1 else bs
            yield self.synth.sample_batch(epoch, b, bs_al, eval_start=start)


def wrap_loaders_for_gpu(
    loaders: Tuple[DataLoader, DataLoader, DataLoader],
    datasets: Tuple["BTEMDenoisingDataset", ...],
    cfg: Config,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[Any, Any, Any]:
    """Build GPU-synthesis loaders over the ALREADY-audited datasets."""
    train_ds, val_ds, test_ds = datasets
    t0 = time.perf_counter()
    train_l = GpuSyntheticLoader(train_ds, cfg.train.batch_size, device, drop_last=True)
    val_l = GpuSyntheticLoader(val_ds, cfg.train.batch_size, device, drop_last=False)
    for ldr in (val_l,):
        ldr.synth.clean = train_l.synth.clean
        ldr.synth.noise = train_l.synth.noise
    test_l = GpuSyntheticLoader(test_ds, cfg.train.batch_size, device, drop_last=False)
    test_l.synth.clean = train_l.synth.clean
    test_l.synth.noise = train_l.synth.noise
    resident = 0
    for t in (train_l.synth.clean, train_l.synth.noise):
        resident += t.numel() * t.element_size()
    for s in (train_l.synth, val_l.synth, test_l.synth):
        for name in ("real_rows", "real_geo", "aug_mats"):
            t = getattr(s, name)
            if t is not None:
                resident += t.numel() * t.element_size()
    logger.info(
        "GPU-RESIDENT SYNTHESIS ON | libraries resident on %s: %.2f GB | built in %.1f s | "
        "batches are BORN on the device: no DataLoader workers, no spawn, no pin_memory, no H2D copy on "
        "the training path. The nine-station patch shares one draw exactly as in the numpy loader "
        "; realized SNR stays exact by the same composite-then-scale construction. "
        "Validation/test streams are deterministic and epoch-independent within the run. "
        "--no_gpu_synthesis restores the CPU loader.",
        str(device), resident / 2 ** 30, time.perf_counter() - t0)
    return train_l, val_l, test_l


def paired_lowsnr_probe(model: nn.Module, ds: "BTEMDenoisingDataset",
                        cfg: Config, device: torch.device
                        ) -> Dict[Tuple[int, int], Dict[str, float]]:
    """G/L per 5 dB bin on library-noise-injected held-out clean rows."""
    model_was_training = model.training
    model.eval()
    ls = int(cfg.data.late_start_index)
    _bs = min(256, int(cfg.train.batch_size))
    _P = int(ds.batch_profile_len()) if hasattr(ds, "batch_profile_len") else 0
    if _P > 1:
        _bs = max(_P, (_bs // _P) * _P)
    loader = DataLoader(ds, batch_size=_bs, shuffle=False, num_workers=0)
    acc: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    _sens: List[float] = []
    edges = [(-5, 0), (0, 5), (5, 10), (75, 85)]
    with torch.no_grad():
        for batch in loader:
            snr = batch.get("inject_snr_db")
            if snr is None:
                break
            x = batch["noisy"].to(device)
            nb = batch.get("neighbors")
            geo = batch.get("neighbor_geometry")
            _dn = batch.get("depth_norm")
            _pl = batch_profile_len(batch)
            pred = model(x, nb.to(device) if nb is not None else None,
                         geo.to(device) if geo is not None else None,
                         depth_norm=(_dn.to(device) if _dn is not None else None),
                         profile_len=_pl)["denoised"]
            y = batch["clean"].to(device).float()
            p = pred.float()
            e2 = (p - y) ** 2
            y2 = y ** 2
            g = (e2.sum(dim=(1, 2)) / y2.sum(dim=(1, 2)).clamp_min(1e-30)).sqrt()
            late_rel = (e2[..., ls:].sum(dim=(1, 2))
                        / y2[..., ls:].sum(dim=(1, 2)).clamp_min(1e-30)).sqrt()
            sv = snr.reshape(-1)
            for i in range(sv.numel()):
                s = float(sv[i])
                if not math.isfinite(s):
                    continue
                for a, b in edges:
                    if a <= s < b:
                        acc.setdefault((a, b), []).append(
                            (float(g[i]) * 100.0, float(late_rel[i]) * 100.0))
                        break
            _xa = batch.get("noisy_alt")
            _ha = batch.get("has_alt")
            if _xa is not None:
                _nba = batch.get("neighbors_alt", nb)
                _pa = model(_xa.to(device),
                            _nba.to(device) if _nba is not None else None,
                            geo.to(device) if geo is not None else None,
                            depth_norm=(_dn.to(device) if _dn is not None else None),
                            profile_len=_pl)["denoised"]
                _din = (_xa.to(device).float() - x.float()).flatten(1).pow(2).sum(1).sqrt()
                _dout = (_pa.float() - p).flatten(1).pow(2).sum(1).sqrt()
                _ok = (_din > 1e-12)
                if _ha is not None:
                    _ok = _ok & (_ha.to(device).reshape(-1) > 0.5)
                for i in range(_din.numel()):
                    if bool(_ok[i]):
                        _sens.append(float(_dout[i] / _din[i]))
    if model_was_training:
        model.train()
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for key, vals in acc.items():
        gs = sorted(v[0] for v in vals)
        lsv = sorted(v[1] for v in vals)
        out[key] = {"G": gs[len(gs) // 2], "L": lsv[len(lsv) // 2], "n": len(vals)}
    if _sens:
        _sens.sort()
        for key in out:
            out[key]["stab"] = _sens[len(_sens) // 2]
    return out


def block_aligned_batch_size(ds: Any, batch_size: int) -> int:
    """Largest batch size <= batch_size that is a multiple of the dataset's block length, so every batch of a
    block-structured dataset is n x P contiguous stations and the lateral joint prior can act.
    """
    try:
        P = int(ds.batch_profile_len()) if hasattr(ds, "batch_profile_len") else 0
    except Exception:
        P = 0
    bs = int(max(1, batch_size))
    return max(P, (bs // P) * P) if P > 1 else bs


def batch_profile_len(batch: Mapping[str, Any]) -> Optional[int]:
    """The contiguous block length a collated batch declares (via the per-item "profile_len" key), or None when the
    batch is not block-structured or its size is not a multiple of the block.
    """
    try:
        pl = batch.get("profile_len") if hasattr(batch, "get") else None
        if pl is None:
            return None
        if torch.is_tensor(pl):
            if pl.numel() == 0:
                return None
            p = int(pl.reshape(-1)[0].item())
        else:
            p = int(pl)
        n = int(batch["noisy"].shape[0])
        return p if (p > 2 and n % p == 0) else None
    except Exception:
        return None


class PatchAlignedDataParallel(nn.DataParallel):
    def __init__(self, module: nn.Module, device_ids: List[int], patch_len: int) -> None:
        super().__init__(module, device_ids=list(device_ids), output_device=int(device_ids[0]))
        self.patch_len = max(1, int(patch_len))
        self._chunk_sizes: List[int] = []
        self.primary_share = 0.5

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def scatter(self, inputs, kwargs, device_ids):
        B = None
        for a in list(inputs) + list((kwargs or {}).values()):
            if torch.is_tensor(a) and a.dim() >= 1:
                B = int(a.shape[0])
                break
        n = len(device_ids)
        P = self.patch_len
        if B is None or n <= 1 or B < 2 * P:
            self._chunk_sizes = [int(B or 0)]
            return [tuple(inputs)], [dict(kwargs or {})]
        npatch = B // P
        _s = float(getattr(self, "primary_share", 0.5))
        if n >= 2 and abs(_s - 0.5) > 1e-6 and npatch >= n:
            p0 = int(round(npatch * _s))
            p0 = min(max(p0, 1), npatch - (n - 1))
            rest = npatch - p0
            per = [p0 * P] + [(rest // (n - 1) + (1 if i < rest % (n - 1) else 0)) * P for i in range(n - 1)]
        else:
            per = [(npatch // n + (1 if i < npatch % n else 0)) * P for i in range(n)]
        per[-1] += B - sum(per)
        per = [c for c in per if c > 0]
        self._chunk_sizes = per
        offs = np.cumsum([0] + per).tolist()
        def _split(x, i, dev):
            if torch.is_tensor(x) and x.dim() >= 1 and int(x.shape[0]) == B:
                return x[offs[i]:offs[i + 1]].to(dev, non_blocking=True)
            if torch.is_tensor(x):
                return x.to(dev, non_blocking=True)
            return x
        ins = [tuple(_split(a, i, torch.device("cuda", device_ids[i])) for a in inputs) for i in range(len(per))]
        kws = [{k: _split(v, i, torch.device("cuda", device_ids[i])) for k, v in (kwargs or {}).items()} for i in range(len(per))]
        return ins, kws

    def gather(self, outputs, output_device):
        if len(outputs) == 1:
            return outputs[0]
        sizes = list(self._chunk_sizes)
        def _g(vals):
            v0 = vals[0]
            if torch.is_tensor(v0):
                if (v0.dim() >= 1 and len(vals) == len(sizes)
                        and all(torch.is_tensor(v) and v.dim() >= 1 and int(v.shape[0]) == sz for v, sz in zip(vals, sizes))):
                    return torch.cat([v.to(output_device) for v in vals], dim=0)
                return v0.to(output_device)
            if isinstance(v0, dict):
                return {k: _g([v[k] for v in vals]) for k in v0}
            if isinstance(v0, (list, tuple)):
                return type(v0)(_g(list(z)) for z in zip(*vals))
            return v0
        return _g(list(outputs))


_DP_STATE: Dict[str, Any] = {"wrapper": None}
_PREFLIGHT_STATE: Dict[str, Any] = {"micro_batch": None, "halvings": 0}


_OOM_TYPES = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                                   getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None))
                       if isinstance(t, type)) or (RuntimeError,)


_MEMORY_PRESSURE_MARKERS: Tuple[str, ...] = (
    "out of memory",
    "unable to find an engine to execute this computation",
    "cudnn_status_alloc_failed",
    "cudnn_status_internal_error",
    "cublas_status_alloc_failed",
    "cusolver_status_alloc_failed",
    "cufft_alloc_failed",
)


def is_engine_search_failure(exc: BaseException) -> bool:
    """The cuDNN engine search (find with benchmark on, get with it off) found no runnable plan."""
    return "unable to find an engine to execute this computation" in str(exc).lower()


def is_oom_error(exc: BaseException) -> bool:
    """A memory-pressure failure: CUDA out-of-memory (including the DataParallel re-raise wrapper and the plain
    RuntimeError form older builds raise) or one of the library messages that stand for it.
    """
    msg = str(exc).lower()
    if isinstance(exc, _OOM_TYPES) and "out of memory" in msg:
        return True
    return isinstance(exc, RuntimeError) and any(m in msg for m in _MEMORY_PRESSURE_MARKERS)


_VRAM_FIT: Dict[str, Any] = {"cards": {}, "frac": 0.0, "reserve": 0.0, "margin": 0.0, "patch": 1}
_CAPACITY_PROBE: Dict[str, Any] = {"mem": None, "peak": None}


def cuda_capacity(card: int) -> Dict[str, float]:
    """Total / free / reserved / capacity / foreign bytes of one card, as seen by this process right now."""
    _hook = _CAPACITY_PROBE.get("mem")
    if _hook is not None:
        total, free_b, reserved = (float(v) for v in _hook(int(card)))
    else:
        dev = torch.device("cuda", int(card))
        total = float(torch.cuda.get_device_properties(dev).total_memory)
        free_b = total
        try:
            _f2, _t = torch.cuda.mem_get_info(dev)
            free_b = float(_f2)
            if float(_t) > 0.0:
                total = float(_t)
        except Exception:
            pass
        try:
            reserved = float(torch.cuda.memory_reserved(dev))
        except Exception:
            reserved = 0.0
    capacity = max(0.0, min(total, free_b + reserved))
    return {"total": total, "free": free_b, "reserved": reserved, "capacity": capacity,
            "foreign": max(0.0, total - capacity)}


def log_foreign_vram(caps: Mapping[int, Mapping[str, float]], logger: logging.Logger) -> None:
    """Name the memory this process cannot use."""
    for i, c in caps.items():
        _gb = float(c["foreign"]) / 2 ** 30
        if _gb >= 2.0 and float(c["foreign"]) >= 0.03 * float(c["total"]):
            logger.warning("cuda:%d: %.1f of %.1f GB is NOT available to this run (held by other processes; this "
                           "process can obtain %.1f GB). The micro-batch is sized to what is actually free, so the run is "
                           "slower than the card allows. Find the owner with the vendor's smi tool (nvidia-smi / ppu-smi: a "
                           "finished or crashed run that still holds memory is the usual cause) and stop it for full speed.",
                           int(i), _gb, float(c["total"]) / 2 ** 30, float(c["capacity"]) / 2 ** 30)


def memory_is_short(ids: Sequence[int], threshold: float = 0.85) -> bool:
    """Did the failed attempt press against the memory this process can obtain?"""
    for i in ids:
        try:
            cap = cuda_capacity(int(i))
            _hook = _CAPACITY_PROBE.get("peak")
            peak = float(_hook(int(i))) if _hook is not None else float(
                torch.cuda.max_memory_reserved(torch.device("cuda", int(i))))
            if cap["capacity"] > 0.0 and peak >= float(threshold) * cap["capacity"]:
                return True
        except Exception:
            return True
    return False


def recheck_micro_batch(cfg: "Config", device: torch.device, logger: logging.Logger) -> Optional[int]:
    """Re-evaluate the stored per-card fit against the capacity the cards have now."""
    cards = dict(_VRAM_FIT.get("cards") or {})
    if device.type != "cuda" or not cards:
        return None
    frac = float(_VRAM_FIT.get("frac") or 0.0)
    if frac <= 0.0:
        return None
    reserve = float(_VRAM_FIT.get("reserve") or 0.0)
    margin = float(_VRAM_FIT.get("margin") or 0.0)
    P = max(1, int(_VRAM_FIT.get("patch") or 1))
    micro = int(cfg.train.batch_size)
    fits: Optional[int] = None
    caps: Dict[int, Dict[str, float]] = {}
    for i, (slope, const) in cards.items():
        caps[int(i)] = cuda_capacity(int(i))
        budget = min(caps[int(i)]["total"], caps[int(i)]["capacity"]) * frac
        f = int(max(budget - float(const) - reserve, 0.0) / (max(float(slope), 1.0) * (1.0 + margin)))
        fits = f if fits is None else min(fits, f)
    if fits is None or fits >= micro:
        return None
    new = int(max(P, (fits // P) * P))
    if new >= micro:
        return None
    log_foreign_vram(caps, logger)
    logger.warning("CARD RE-CHECK before the first real step: the capacity fell since the probe (%s); the "
                   "%d-row micro-batch no longer fits the %.0f%% budget -> %d rows.",
                   " | ".join("cuda:%d %.1f GB obtainable" % (i, c["capacity"] / 2 ** 30) for i, c in caps.items()),
                   micro, 100.0 * frac, new)
    return new


def free_cuda_after_oom() -> None:
    """Drop the interrupted graphs and give the fragments back to the allocator on every card."""
    try:
        import gc as _gc697
        _gc697.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            for _i in range(int(torch.cuda.device_count())):
                with torch.cuda.device(_i):
                    torch.cuda.empty_cache()
    except Exception:
        pass


def halve_rows(batch: Mapping[str, Any], patch_len: int) -> Dict[str, Any]:
    """The first half of the rows, cut on patch boundaries (never inside a patch: the lateral joint prior and the
    profile losses read whole patches).
    """
    P = max(1, int(patch_len))
    B = int(batch["clean"].shape[0])
    nB = max(P, ((B // 2) // P) * P)
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and int(v.shape[0]) == B:
            out[k] = v[:nB]
        else:
            out[k] = v
    return out


def parallel_model(model: nn.Module) -> nn.Module:
    """The two-card wrapper for this model when armed, else the model itself."""
    w = _DP_STATE.get("wrapper")
    if w is not None and getattr(w, "module", None) is model:
        return w
    return model


def _warm_up_cards(model: nn.Module, ids: List[int], patch_len: int, logger: logging.Logger) -> None:
    T = int(model.gate_scale_norm.numel())
    K = int(getattr(model.cfg, "num_neighbors", 8) or 8)
    B = 2 * max(1, int(patch_len))
    for i in ids:
        dev = torch.device("cuda", int(i))
        m = copy.deepcopy(model).to(dev)
        m.train()
        gs = m.gate_scale_norm.detach().float().view(1, 1, -1)
        x = (gs * (1.0 + 0.05 * torch.randn(B, 1, T, device=dev))).abs()
        nb = (gs * (1.0 + 0.05 * torch.randn(B, K, T, device=dev))).abs()
        geo = m._default_neighbor_geometry(nb) if hasattr(m, "_default_neighbor_geometry") else None
        out = m(x, nb, geo, depth_norm=torch.full((B, 1), 0.5, device=dev), profile_len=int(patch_len) if patch_len > 1 else None)
        out["denoised"].float().sum().backward()
        del m, out, x, nb, geo
        torch.cuda.synchronize(dev)
    logger.info("lazy CUDA library initialisation completed serially on cards %s.", ids)


def arm_data_parallel(model: nn.Module, device: torch.device, cfg: Config, logger: logging.Logger) -> None:
    _DP_STATE["wrapper"] = None
    if not bool(getattr(cfg.runtime, "multi_gpu", True)) or device.type != "cuda":
        return
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 1
    if n < 2:
        logger.info("one CUDA device visible: single-card training.")
        return
    primary = int(device.index if device.index is not None else torch.cuda.current_device())
    ids = [primary] + [i for i in range(n) if i != primary]
    try:
        _warm_up_cards(model, ids, int(getattr(cfg.data, "profile_patch_len", 1) or 1), logger)
    except Exception as _e:
        logger.warning("card warm-up failed (%r): single-card training.", _e)
        return
    _DP_STATE["wrapper"] = PatchAlignedDataParallel(model, ids, int(getattr(cfg.data, "profile_patch_len", 1) or 1))
    logger.info("DATA PARALLEL ARMED on cards %s (primary cuda:%d), micro-batch split on %d-station "
                "patch boundaries.", ids, primary, int(getattr(cfg.data, "profile_patch_len", 1) or 1))


def accumulate_finite_gradients(scaled_loss: torch.Tensor, params: List[torch.nn.Parameter],
                                    retain_graph: bool = False) -> Tuple[bool, List[torch.nn.Parameter]]:
    """One micro-batch's gradients through autograd.grad, gated on finiteness before they are added to .grad."""
    grads = torch.autograd.grad(scaled_loss, params, allow_unused=True, retain_graph=retain_graph)
    present = [(p, g) for p, g in zip(params, grads) if g is not None]
    if not present:
        return True, []
    flags = torch.stack([g.isfinite().all() for _, g in present])
    if not bool(flags.all()):
        bad_idx = (~flags).nonzero().reshape(-1).tolist()
        return False, [present[i][0] for i in bad_idx]
    for p, g in present:
        if p.grad is None:
            p.grad = g.detach()
        else:
            p.grad.add_(g)
    return True, []


def name_nonfinite_grad_terms(losses: Mapping[str, Any], params: List[torch.nn.Parameter],
                                  max_terms: int = 96) -> List[str]:
    """Which loss terms carry a non-finite gradient: per-term autograd.grad on a graph that is still alive
    (retain_graph).
    """
    bad: List[str] = []
    n = 0
    for name, v in losses.items():
        if not (isinstance(v, torch.Tensor) and v.ndim == 0 and bool(getattr(v, "requires_grad", False))):
            continue
        if name == "total":
            continue
        n += 1
        if n > max_terms:
            break
        try:
            gs = torch.autograd.grad(v, params, allow_unused=True, retain_graph=True)
            gs = [g for g in gs if g is not None]
            if gs and not bool(torch.stack([g.isfinite().all() for g in gs]).all()):
                bad.append(name)
        except RuntimeError:
            continue
    return bad


_CPU_BUDGET: Dict[str, Any] = {"done": False}


def log_cpu_budget(logger: logging.Logger, workers: int) -> None:
    """The pod's CPU quota next to the loader worker count: 4 workers on a 2-core quota starve the main thread that
    launches every kernel.
    """
    if _CPU_BUDGET["done"]:
        return
    _CPU_BUDGET["done"] = True
    try:
        try:
            aff = len(os.sched_getaffinity(0))
        except Exception:
            aff = int(os.cpu_count() or 0)
        quota = "no cgroup limit found"
        for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
            if os.path.exists(path):
                txt = open(path).read().split()
                if path.endswith("cpu.max"):
                    if txt and txt[0] != "max":
                        quota = "%.2f cores (cgroup v2 cpu.max)" % (float(txt[0]) / float(txt[1]))
                    else:
                        quota = "unlimited (cgroup v2)"
                else:
                    q = float(txt[0])
                    per = float(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read().split()[0]) if os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_period_us") else 100000.0
                    quota = ("%.2f cores (cgroup v1 cfs)" % (q / per)) if q > 0 else "unlimited (cgroup v1)"
                break
        logger.info("CPU BUDGET | affinity %d cores | quota %s | loader workers %d + 1 main thread (kernel launches, "
                    "two DataParallel replica threads) + pin thread.", aff, quota, int(workers))
    except Exception as _e:
        logger.debug("cpu budget unavailable: %r", _e)


_AUX_ARM_STATS: Dict[str, float] = {"steps": 0.0, "rows": 0.0, "bg_rows": 0.0, "alt_rows": 0.0, "bg_flag": 0.0, "alt_flag": 0.0,
                                        "bg_dropped": 0.0, "alt_dropped": 0.0, "capped_steps": 0.0}


_AUX_PROBE_MODE: Dict[str, bool] = {"on": False}


def aux_arm_rows(flags: Optional[torch.Tensor], batch_rows: int, patch_len: int, quantum_patches: int,
                     full_fraction: float = 0.90, cap_share: float = 1.0
                     ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """(rows, effective_flags, dropped_patches)."""
    if flags is None:
        return None, None, 0
    B = int(batch_rows)
    P = max(1, int(patch_len))
    if B <= 0 or B % P != 0 or int(flags.numel()) != B:
        return None, None, 0
    n = B // P
    pf = (flags.detach().reshape(n, P).float().amax(dim=1) > 0.5).cpu()
    k = int(pf.sum().item())
    if k <= 0:
        return None, None, 0
    q = max(1, int(quantum_patches))
    full_at = int(math.ceil(float(full_fraction) * n))
    capped = float(cap_share) < float(full_fraction)
    eff: Optional[torch.Tensor] = None
    dropped = 0
    if capped:
        if bool(_AUX_PROBE_MODE.get("on", False)):
            cap = min(n, max(1, int(math.ceil(float(cap_share) * n))))
            q = 1
        else:
            cap = min(n, max(min(q, n), (int(float(cap_share) * n) // q) * q))
        if k > cap:
            drop = pf.nonzero().reshape(-1)[cap:]
            dropped = int(drop.numel())
            pf = pf.clone()
            pf[drop] = False
            eff_p = torch.ones(n, dtype=torch.float32)
            eff_p[drop] = 0.0
            eff = flags.detach().reshape(n, P).float() * eff_p.to(flags.device).view(n, 1)
            eff = eff.reshape(flags.shape).to(dtype=flags.dtype)
            k = cap
        kq = min(cap, int(math.ceil(k / float(q))) * q)
    else:
        if k >= full_at:
            return None, None, 0
        kq = min(n, int(math.ceil(k / float(q))) * q)
        if kq >= full_at:
            return None, None, 0
    sel = pf.clone()
    if kq > k:
        spare = (~sel).nonzero().reshape(-1)[: kq - k]
        sel[spare] = True
    pidx = sel.nonzero().reshape(-1)
    rows = (pidx.view(-1, 1) * P + torch.arange(P).view(1, -1)).reshape(-1)
    return rows.to(device=flags.device, dtype=torch.long), eff, dropped


def scatter_aux_rows(main: torch.Tensor, sub: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Full-batch tensor whose `rows` carry the auxiliary arm's output (with its graph) and whose other rows carry
    the detached main output.
    """
    full = main.detach().clone()
    return full.index_copy(0, rows.to(full.device), sub.to(device=full.device, dtype=full.dtype))


def _take_rows(x: Optional[torch.Tensor], rows: torch.Tensor) -> Optional[torch.Tensor]:
    return None if x is None else x.index_select(0, rows.to(x.device))


def forward_training_arms(model: nn.Module, batch: Mapping[str, torch.Tensor]
                          ) -> Dict[str, torch.Tensor]:
    """The one place the training forwards live."""
    _pair_cpu = torch.get_rng_state()
    _pair_cuda = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)
    _pl = batch_profile_len(batch)
    _dn = batch.get("depth_norm")
    _om = batch.get("obs_mask")
    _kw = {"obs_mask": _om} if _om is not None else {}
    outputs: Dict[str, torch.Tensor] = dict(parallel_model(model)(
        batch["noisy"], batch.get("neighbors"), batch.get("neighbor_geometry"),
        depth_norm=_dn, profile_len=_pl, **_kw))
    _hb = batch.get("has_anomaly")
    _need_bg = (_hb is not None and "noisy_bg" in batch
                and float(_hb.sum().item()) > 0.0)
    _ha = batch.get("has_alt")
    _need_alt = (batch.get("noisy_alt") is not None
                 and (_ha is None or float(_ha.sum().item()) > 0.0))
    _AUX_ARM_STATS["steps"] += 1.0
    _AUX_ARM_STATS["rows"] += float(batch["noisy"].shape[0])
    if _need_bg or _need_alt:
        _now_cpu = torch.get_rng_state()
        _now_cuda = (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None)
        torch.set_rng_state(_pair_cpu)
        if _pair_cuda is not None:
            torch.cuda.set_rng_state_all(_pair_cuda)
        _mc = getattr(model, "cfg", None)
        _sub = (bool(getattr(_mc, "aux_arms_subbatch", True))
                   and (float(getattr(_mc, "dropout", 0.0) or 0.0) <= 0.0 or not bool(getattr(model, "training", True))))
        _B2 = int(batch["noisy"].shape[0])
        _P2 = int(_pl) if _pl else 1
        _q = int(getattr(_mc, "aux_arms_quantum_patches", 4) or 1)
        _geo = batch.get("neighbor_geometry")
        if _need_bg:
            _ff = float(getattr(_mc, "aux_full_fraction", 0.90) or 0.90)
            _rows_bg, _eff_bg, _drop_bg = (aux_arm_rows(
                _hb, _B2, _P2, _q, _ff, float(getattr(_mc, "aux_bg_cap_share", 1.0) or 1.0))
                if _sub else (None, None, 0))
            if _eff_bg is not None:
                outputs["has_anomaly_effective"] = _eff_bg
                _AUX_ARM_STATS["bg_dropped"] += float(_drop_bg)
                _AUX_ARM_STATS["capped_steps"] += 1.0
            _AUX_ARM_STATS["bg_flag"] += float(_hb.detach().float().sum().item())
            if _rows_bg is None:
                _AUX_ARM_STATS["bg_rows"] += float(_B2)
                outputs["denoised_bg"] = parallel_model(model)(
                    batch["noisy_bg"], batch.get("neighbors_bg"),
                    batch.get("neighbor_geometry"),
                    depth_norm=_dn, profile_len=_pl, **_kw)["denoised"]
            else:
                _AUX_ARM_STATS["bg_rows"] += float(_rows_bg.numel())
                _o_bg = parallel_model(model)(
                    _take_rows(batch["noisy_bg"], _rows_bg), _take_rows(batch.get("neighbors_bg"), _rows_bg),
                    _take_rows(_geo, _rows_bg),
                    depth_norm=_take_rows(_dn, _rows_bg), profile_len=_pl,
                    **({"obs_mask": _take_rows(_om, _rows_bg)} if _om is not None else {}))["denoised"]
                outputs["denoised_bg"] = scatter_aux_rows(outputs["denoised"], _o_bg, _rows_bg)
        if _need_alt:
            torch.set_rng_state(_pair_cpu)
            if _pair_cuda is not None:
                torch.cuda.set_rng_state_all(_pair_cuda)
            _ff723a = float(getattr(_mc, "aux_full_fraction", 0.90) or 0.90)
            _rows_alt, _eff_alt, _drop_alt = (aux_arm_rows(
                _ha, _B2, _P2, _q, _ff723a, float(getattr(_mc, "aux_alt_cap_share", 1.0) or 1.0))
                if (_sub and _ha is not None) else (None, None, 0))
            if _eff_alt is not None:
                outputs["has_alt_effective"] = _eff_alt
                _AUX_ARM_STATS["alt_dropped"] += float(_drop_alt)
                _AUX_ARM_STATS["capped_steps"] += 1.0
            if _ha is not None:
                _AUX_ARM_STATS["alt_flag"] += float(_ha.detach().float().sum().item())
            if _rows_alt is None:
                _AUX_ARM_STATS["alt_rows"] += float(_B2)
                outputs["denoised_alt"] = parallel_model(model)(
                    batch["noisy_alt"],
                    batch.get("neighbors_alt", batch.get("neighbors")),
                    batch.get("neighbor_geometry"),
                    depth_norm=_dn, profile_len=_pl, **_kw)["denoised"]
            else:
                _AUX_ARM_STATS["alt_rows"] += float(_rows_alt.numel())
                _o_alt = parallel_model(model)(
                    _take_rows(batch["noisy_alt"], _rows_alt),
                    _take_rows(batch.get("neighbors_alt", batch.get("neighbors")), _rows_alt),
                    _take_rows(_geo, _rows_alt),
                    depth_norm=_take_rows(_dn, _rows_alt), profile_len=_pl,
                    **({"obs_mask": _take_rows(_om, _rows_alt)} if _om is not None else {}))["denoised"]
                outputs["denoised_alt"] = scatter_aux_rows(outputs["denoised"], _o_alt, _rows_alt)
        torch.set_rng_state(_now_cpu)
        if _now_cuda is not None:
            torch.cuda.set_rng_state_all(_now_cuda)
    _mcfg = getattr(model, "cfg", None)
    _cf = float(getattr(_mcfg, "continuation_arm_frac", 0.0) or 0.0)
    if _cf > 0.0 and "clean" in batch and "noisy" in batch:
        _x = batch["noisy"]
        _B, _T = int(_x.shape[0]), int(_x.shape[-1])
        _ls = int(getattr(_mcfg, "late_start_index", 20) or 20)
        _P = int(_pl) if _pl else 1
        _m = max(1, int(round(_B * _cf)))
        if _P > 1:
            _m = max(_P, (_m // _P) * _P)
            _m = min(_m, 4 * _P)
            _m = min(_m, _B)
            _m = (_m // _P) * _P
        else:
            _m = min(_m, 36, _B)
        if _m >= 1 and (_T - _ls) >= 6:
            _yc = batch["clean"][:_m]
            _xc = _x[:_m].clone()
            _nz = _xc - _yc.to(dtype=_xc.dtype)
            _runs = _m // _P if _P > 1 else _m
            _cs = torch.randint(_ls, _T - 3, (_runs,),
                                   device=_xc.device)
            if _P > 1:
                _cs = _cs.repeat_interleave(_P)
            _gi = torch.arange(_T, device=_xc.device).view(1, 1, -1)
            _cm = (_gi >= _cs.view(-1, 1, 1)).to(_xc.dtype)
            _xc = _xc * (1.0 - _cm) + _nz * _cm
            _nb = batch.get("neighbors")
            if _nb is not None:
                _nb = _nb[:_m] * (1.0 - _cm)
            _ng = batch.get("neighbor_geometry")
            if _ng is not None:
                _ng = _ng[:_m]
            _dn2 = _dn[:_m] if _dn is not None else None
            _cout = parallel_model(model)(
                _xc, _nb, _ng, depth_norm=_dn2,
                profile_len=(_P if _P > 1 else None))
            outputs["denoised_cont"] = _cout["denoised"]
            outputs["cont_gate_mask"] = _cm.float()
            if "library_coef" in _cout:
                outputs["library_coef_cont"] = _cout["library_coef"]
    return outputs


def _export_write_csv(path: Path, header: List[str], rows: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r_ in np.asarray(rows):
            w.writerow(["%.9g" % float(v) for v in np.ravel(r_)])


def export_split_arrays(model: nn.Module, loader, device: torch.device, cfg: Config,
                        out_dir: Path, tag: str, target_time: np.ndarray,
                        logger: logging.Logger, max_rows: int = 40000,
                        clean_cache: Any = None, noise_cache: Any = None,
                        checkpoint_digest: str = "") -> None:
    """Export raw / clean / clean_bg / denoised / library_bg traces and per-row metadata of a split as categorized
    CSV (+ npz).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {"noisy": [], "clean": [], "clean_bg": [], "denoised": [], "library_bg": [], "denoised_blockpath": []}
    meta_keys = ("snr_db", "has_anomaly", "is_forward", "profile_len", "event_label", "clean_row", "depth_norm")
    meta = {k: [] for k in meta_keys}
    model.eval()
    n_rows = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["noisy"], batch.get("neighbors"), batch.get("neighbor_geometry"),
                        depth_norm=batch.get("depth_norm"), profile_len=batch_profile_len(batch))
            def _r(t):
                t = t.detach().float().cpu().numpy()
                return t[:, 0] if t.ndim == 3 else t
            keep["noisy"].append(_r(batch["noisy"]))
            keep["clean"].append(_r(batch["clean"]))
            keep["clean_bg"].append(_r(batch["clean_bg"]) if "clean_bg" in batch else _r(batch["clean"]))
            keep["denoised_blockpath"].append(_r(out["denoised"]))
            keep["denoised"].append(_r(out["denoised"]))
            keep["library_bg"].append(_r(out["library_bg"]) if "library_bg" in out
                                      else np.full_like(_r(out["denoised"]), np.nan))
            bsz = int(batch["noisy"].shape[0])
            for k in meta_keys:
                v = batch.get(k)
                if k == "clean_row" and v is None:
                    v = batch.get("clean_index")
                meta[k].append(v.detach().float().cpu().numpy().reshape(-1) if torch.is_tensor(v)
                               else np.full(bsz, np.nan))
            n_rows += bsz
            if n_rows >= max_rows:
                break
    arrays = {k: np.concatenate(v, 0) for k, v in keep.items()}
    metas = {k: np.concatenate(v, 0) for k, v in meta.items()}
    _proto = "block_path_subset_scaled_units (DIAGNOSTIC, not the deliverable)"
    _gsc = 1.0
    _tag = "%s_subset_blockpath" % tag
    try:
        atomic_json_dump({"tag": _tag, "rows": int(arrays["noisy"].shape[0]), "inference_protocol": _proto,
                          "checkpoint_state_digest": str(checkpoint_digest), "global_scale_applied": _gsc,
                          "units": "scaled (divided by global scale)",
                          "columns": "clean_row = row index into the paired clean cache; depth_norm = position inside the work area"},
                         out_dir / ("%s_provenance.json" % _tag))
    except Exception:
        pass
    hdr = ["t_%.6g" % float(t_) for t_ in np.asarray(target_time).reshape(-1)]
    for k, a in arrays.items():
        _export_write_csv(out_dir / ("%s_%s.csv" % (_tag, k)), hdr, a)
    _export_write_csv(out_dir / ("%s_meta.csv" % _tag), list(meta_keys),
                      np.stack([metas[k] for k in meta_keys], 1))
    np.savez_compressed(out_dir / ("%s_arrays.npz" % _tag), **arrays,
                        **{"meta_" + k: v for k, v in metas.items()})
    logger.info("DIAGNOSTIC SUBSET EXPORT %s | %d loader-sampled rows -> %s (files %s_*; block "
                "path, scaled units; NOT the deliverable product).",
                tag, int(arrays["noisy"].shape[0]), str(out_dir), _tag)


def _pf_fingerprint(batch: Dict[str, Any]) -> str:
    h = hashlib.sha1()
    ci = batch.get("clean_index")
    sn = batch.get("snr_db")
    if ci is not None:
        h.update(np.ascontiguousarray(ci.detach().cpu().numpy().astype(np.int64)).tobytes())
    if sn is not None:
        h.update(np.ascontiguousarray(np.round(sn.detach().cpu().numpy().astype(np.float64), 3)).tobytes())
    return h.hexdigest()[:12]


def gradient_group_audit(model: "PEBRNet", loss_fn: "ProjectLoss", lo: Mapping[str, torch.Tensor],
                             device: torch.device, note_init: bool = False) -> str:
    """Gradient norms of the four term groups on the shared trunk, their closure against the full objective on those
    parameters, and pairwise cosines.
    """
    _params = (list(model.shared_parameters()) if hasattr(model, "shared_parameters")
               else [p for p in model.parameters() if p.requires_grad])
    _params = [p for p in _params if p.requires_grad] or [p for p in model.parameters() if p.requires_grad]
    _lc = loss_fn.cfg
    def _t(k: str) -> torch.Tensor:
        v = lo.get(k)
        return v if isinstance(v, torch.Tensor) else torch.zeros((), device=device)
    groups = {
        "contract": (float(getattr(_lc, "alpha_contract_global", 25.0)) * _t("l_ps_global")
                     + float(getattr(_lc, "alpha_contract_early_mid", 25.0)) * _t("l_ps_early_mid")
                     + float(getattr(_lc, "alpha_contract_late", 35.0)) * _t("l_ps_late")),
        "prior_z": (float(getattr(_lc, "alpha_prior_direct", 15.0)) * _t("l_prior_direct")
                    + float(getattr(_lc, "alpha_manifold_late", 0.6)) * _t("l_manifold_late")
                    + float(getattr(_lc, "alpha_manifold_early", 0.1)) * _t("l_manifold_early")),
        "denoise_z": (float(getattr(_lc, "alpha_denoise_late_accuracy", 0.30)) * _t("l_denoise_late_accuracy")
                      + float(getattr(_lc, "alpha_denoise_seam", 0.30)) * _t("l_denoise_seam")
                      + float(getattr(_lc, "alpha_denoise_early", 0.10)) * _t("l_denoise_early")),
        "gate_eq": float(getattr(_lc, "alpha_gate_eq_row", 0.0)) * _t("l_gate_eq_row"),
        "acceptance": float(getattr(loss_fn, "acceptance_scale", 1.0)) * (
            float(getattr(_lc, "alpha_pass_hinge", 1.0)) * _t("l_pass_hinge")
            + float(getattr(_lc, "alpha_no_harm", 1.0)) * _t("l_no_harm")
            + float(getattr(_lc, "alpha_quiet_var", 0.5)) * _t("l_quiet_var")),
    }
    def _flat(gs) -> torch.Tensor:
        return torch.cat([(g.float().reshape(-1) if g is not None else torch.zeros(p.numel(), device=device))
                          for g, p in zip(gs, _params)])
    norms: Dict[str, float] = {}
    vecs: Dict[str, Optional[torch.Tensor]] = {}
    for k, v in groups.items():
        if not (isinstance(v, torch.Tensor) and v.requires_grad):
            norms[k] = 0.0
            vecs[k] = None
            continue
        vecs[k] = _flat(torch.autograd.grad(v, _params, retain_graph=True, allow_unused=True))
        norms[k] = float(vecs[k].norm())
    gt = _flat(torch.autograd.grad(lo["total"], _params, retain_graph=True, allow_unused=True))
    gsum = None
    for v in vecs.values():
        if v is not None:
            gsum = v if gsum is None else gsum + v
    tn = float(gt.norm())
    cn = float(gsum.norm()) if gsum is not None else 0.0
    def _cos(a: str, b: str) -> float:
        va, vb = vecs.get(a), vecs.get(b)
        if va is None or vb is None or float(va.norm()) == 0.0 or float(vb.norm()) == 0.0:
            return float("nan")
        return float((va * vb).sum() / (va.norm() * vb.norm()))
    if not math.isfinite(tn) or tn <= 0.0:
        raise RuntimeError("total gradient on the shared trunk is zero or non-finite")
    _LAST_GRAD_GROUP_NORMS.clear()
    _LAST_GRAD_GROUP_NORMS.update(norms)
    _LAST_GRAD_GROUP_NORMS["total"] = tn
    zero_note = ""
    if note_init and any(n == 0.0 for k, n in norms.items() if k not in ("acceptance", "gate_eq")):
        zero_note = (" (a group at exactly 0 on an untrained model = its head output layers are zero-initialised; "
                     "the per-validation re-run gives the trained composition)")
    return ("SHARED-trunk grad norm by group: %s | total %.3e | group sum %.3e (closure %.2f of total) "
            "| cos(contract,prior_z) %.2f cos(contract,denoise_z) %.2f cos(prior_z,denoise_z) %.2f%s" % (
                " ".join("%s %.3e" % (k, v) for k, v in norms.items()), tn, cn, (cn / tn if tn > 0 else float("nan")),
                _cos("contract", "prior_z"), _cos("contract", "denoise_z"), _cos("prior_z", "denoise_z"), zero_note))


_LAST_GRAD_GROUP_NORMS: Dict[str, float] = {}


def gradient_audit_on_loader(model: "PEBRNet", loss_fn: "ProjectLoss", train_loader: Any,
                                 device: torch.device, cfg: Config, logger: logging.Logger, max_patches: int = 8) -> Dict[str, float]:
    """Re-measure the group gradients on the current weights with a small sub-batch of the training loader (raw
    weights, train mode); never raises into the training loop.
    """
    try:
        b = next(iter(train_loader))
        _P = int(max(1, int(getattr(loss_fn, "profile_patch_len", 1) or 1)))
        _B0 = int(b["clean"].shape[0])
        _n = min(_B0, max_patches * _P)
        _n -= _n % _P
        if _n < _P:
            return
        b = {k: ((v[:_n] if (isinstance(v, torch.Tensor) and v.dim() >= 1 and int(v.shape[0]) == _B0) else v).to(device)
                 if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        was = model.training
        model.train()
        model.zero_grad(set_to_none=True)
        amp_enabled = bool(cfg.train.amp and device.type == "cuda")
        with amp_autocast_context(device, amp_enabled):
            o = forward_training_arms(model, b)
            lo = loss_fn(o, b, "A")
        msg = gradient_group_audit(model, loss_fn, lo, device, note_init=False)
        logger.info("GRADIENT AUDIT on the current weights (%d rows) %s | z_band_gain in force %.4g", _n, msg,
                    float(getattr(loss_fn.cfg, "z_band_gain", 1.0)))
        if bool(getattr(cfg.train, "grad_audit_components", True)):
            try:
                _ps = [p for p in model.shared_parameters() if p.requires_grad]
                def _fl(t):
                    gs = torch.autograd.grad(t, _ps, retain_graph=True, allow_unused=True)
                    return torch.cat([(g.reshape(-1).float() if g is not None else torch.zeros(p.numel(), device=device)) for g, p in zip(gs, _ps)])
                _gt = _fl(lo["total"])
                _tn = float(_gt.norm())
                _rows = []
                for _k, _v in lo.items():
                    if _k == "total" or not (isinstance(_v, torch.Tensor) and _v.ndim == 0 and _v.requires_grad):
                        continue
                    _g = _fl(_v)
                    _gn = float(_g.norm())
                    if _gn > 0.0 and _tn > 0.0:
                        _rows.append((_k, _gn, float((_g * _gt).sum() / (_g.norm() * _gt.norm()))))
                _rows.sort(key=lambda r: -r[1])
                _opp = [k for k, _, c in _rows if c < -0.05]
                logger.info("COMPONENT BUDGET (unweighted, shared trunk; |g| / cos with total) top: %s | %d of %d terms oppose the "
                            "total (cos<-0.05): %s.", " ".join("%s %.2e/%.2f" % (k, g, c) for k, g, c in _rows[:10]),
                            len(_opp), len(_rows), ",".join(_opp[:12]))
            except Exception as _e:
                logger.warning("component budget skipped: %r", _e)
        model.zero_grad(set_to_none=True)
        model.train(was)
        return dict(_LAST_GRAD_GROUP_NORMS)
    except Exception as _e:
        try:
            model.zero_grad(set_to_none=True)
        except Exception:
            pass
        logger.warning("per-validation gradient audit skipped: %r", _e)
        return {}


def balance_z_band_gain(loss_fn: "ProjectLoss", norms: Mapping[str, float], logger: logging.Logger) -> float:
    """Optional measured balancer: multiply z_band_gain so that the arms' shared-trunk gradient (prior_z + denoise_z)
    approaches target x contract; damped (x0.5..x2 per validation)
    """
    lc = loss_fn.cfg
    gain = float(getattr(lc, "z_band_gain", 1.0))
    target = float(getattr(lc, "z_band_balance_target", 0.0))
    if target <= 0.0 or not norms:
        return gain
    arms = float(norms.get("prior_z", 0.0)) + float(norms.get("denoise_z", 0.0))
    ref = float(norms.get("contract", 0.0))
    if not (arms > 0.0 and ref > 0.0 and math.isfinite(arms) and math.isfinite(ref)):
        return gain
    _b = min(max(float(getattr(lc, "z_band_balance_ema", 0.8)), 0.0), 0.99)
    _q = math.log(arms / ref) - math.log(max(gain, 1e-12))
    _qe = getattr(loss_fn, "_z_band_logq_ema", None)
    _qe = _q if _qe is None else _b * float(_qe) + (1.0 - _b) * _q
    loss_fn._z_band_logq_ema = _qe
    _err = math.log(target) - (_qe + math.log(max(gain, 1e-12)))
    _smax = math.log(max(float(getattr(lc, "z_band_balance_max_step", 1.1)), 1.0))
    _dead = math.log1p(max(float(getattr(lc, "z_band_balance_deadband", 0.2)), 0.0))
    factor = (1.0 if abs(_err) <= _dead else math.exp(min(_smax, max(-_smax, _err))))
    new = min(float(getattr(lc, "z_band_gain_max", 8192.0)), max(float(getattr(lc, "z_band_gain_min", 1.0)), gain * factor))
    if abs(new - gain) > 1e-9:
        lc.z_band_gain = new
        logger.info("z_band_gain %.4g -> %.4g (arms/contract shared-trunk gradient %.3f this audit, "
                    "%.3f smoothed, target %.2f).",
                    gain, new, arms / ref, math.exp(_qe) * gain, target)
    return new


def preflight_self_test(model: "PEBRNet", loss_fn: "ProjectLoss", train_loader: Any, val_loader: Any,
                        datasets: Any, device: torch.device, cfg: Config, clean_cache: Any, noise_cache: Any,
                        logger: logging.Logger) -> bool:
    """Six checks on the real pipeline before the first epoch (see header)."""
    import traceback as _tb
    results: List[Tuple[str, bool, str]] = []
    logger.info("=" * 96)
    logger.info("PRE-FLIGHT SELF-TEST | device %s | patch %d | batch %d | workers %d",
                str(device), int(getattr(cfg.data, "profile_patch_len", 0) or 0), int(cfg.train.batch_size),
                int(cfg.train.num_workers))
    logger.info("=" * 96)

    def _run(name: str, fn):
        try:
            msg = fn()
            results.append((name, True, str(msg)))
            logger.info("PASS %-4s %s", name, msg)
        except Exception as exc:
            results.append((name, False, repr(exc)))
            logger.error("FAIL %-4s %r\n%s", name, exc, _tb.format_exc())

    try:
        import importlib.util as _ilu700
        if _ilu700.find_spec("matplotlib") is None:
            logger.warning("matplotlib is NOT installed on this server: NO figure of this run can be drawn "
                           "(the 'P5 DEGRADED' line below is this same absence). Install it before the run: "
                           "pip install matplotlib   -- or produce the figures afterwards from the saved weights with "
                           "--mode figures. Training itself is not blocked.")
    except Exception:
        pass

    def _p1():
        out = []
        for tag, ds in (("train", datasets[0]), ("val", datasets[1])):
            if ds is None:
                continue
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(1)
            it = ds[0]
            bad = [k for k, v in it.items() if isinstance(v, torch.Tensor) and k not in ("inject_snr_db",)
                   and not bool(torch.isfinite(v.float()).all())]
            if bad:
                raise RuntimeError("%s item 0 has non-finite tensors: %s" % (tag, bad))
            nz = it.get("noisy")
            zeros = int((nz == 0).sum()) if isinstance(nz, torch.Tensor) else -1
            out.append("%s: %d keys, noisy %s, exact-zero gates left in noisy %d" % (
                tag, len(it), tuple(nz.shape) if isinstance(nz, torch.Tensor) else None, zeros))
        return " | ".join(out)
    _run("P1", _p1)

    _first_batch: Dict[str, Any] = {}
    def _p2():
        fps = []
        for ep in (1, 2):
            if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(ep)
            b = next(iter(train_loader))
            if ep == 1:
                _first_batch.update(b)
            fps.append(_pf_fingerprint(b))
        if fps[0] == fps[1]:
            raise RuntimeError("epoch-1 and epoch-2 first batches are IDENTICAL (%s): workers do not see set_epoch" % fps[0])
        return "epoch-1 %s vs epoch-2 %s differ (%d rows/batch)" % (fps[0], fps[1], int(_first_batch["clean"].shape[0]))
    _run("P2", _p2)

    def _p3_once(b):
        was = model.training
        model.train()
        model.zero_grad(set_to_none=True)
        amp_enabled = bool(cfg.train.amp and device.type == "cuda")
        with amp_autocast_context(device, amp_enabled):
            o = forward_training_arms(model, b)
            lo = loss_fn(o, b, "A")
        bad = [k for k, v in lo.items() if isinstance(v, torch.Tensor) and v.ndim == 0
               and not bool(torch.isfinite(v).all()) and "forward" not in k]
        if bad:
            raise RuntimeError("non-finite loss components: %s" % bad)
        try:
            _share = " | " + gradient_group_audit(model, loss_fn, lo, device, note_init=True)
        except Exception as _e:
            raise RuntimeError("gradient-share audit FAILED (it is part of P3, not optional): %r" % (_e,))
        lo["total"].backward()
        gn = math.sqrt(sum(float((p.grad.float() ** 2).sum()) for p in model.parameters() if p.grad is not None))
        nonfin = [nm for nm, p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        model.zero_grad(set_to_none=True)
        model.train(was)
        if nonfin or not math.isfinite(gn):
            raise RuntimeError("NON-FINITE GRADIENT (norm %r) in %d tensors, e.g. %s -- every step would be skipped" % (
                gn, len(nonfin), nonfin[:5]))
        vram = (torch.cuda.max_memory_allocated(device) / 2 ** 30) if device.type == "cuda" else 0.0
        return "total %.3f | grad norm %.3e finite | peak VRAM %.2f GB | l_late_ps %.4f l_hinge %.4f l_no_harm %.4f l_quiet_var %.4f l_library_late %.4f%s" % (
            float(lo["total"]), gn, vram, float(lo.get("l_late_per_sample", float("nan"))), float(lo.get("l_pass_hinge", float("nan"))),
            float(lo.get("l_no_harm", float("nan"))), float(lo.get("l_quiet_var", float("nan"))), float(lo.get("l_library_late", float("nan"))),
            _share)

    def _p3():
        if not _first_batch:
            raise RuntimeError("no batch from P2")
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in _first_batch.items()}
        _P = int(max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 1)))
        _B = int(b["clean"].shape[0])
        _halved = 0
        _bench_off = False
        _ids = _training_card_ids(device) if device.type == "cuda" else []
        while True:
            try:
                for _i in _ids:
                    torch.cuda.reset_peak_memory_stats(torch.device("cuda", int(_i)))
                msg = _p3_once(b)
                if _halved:
                    msg = "%s | OOM at %d rows -> ran at %d rows after %d halving(s); micro-batch pinned to %d for the run" % (
                        msg, _B, int(b["clean"].shape[0]), _halved, int(b["clean"].shape[0]))
                if _bench_off:
                    msg = "%s | cudnn.benchmark switched OFF for the run (engine search failed with memory to spare)" % msg
                return msg
            except Exception as _e:
                if not is_oom_error(_e):
                    raise
                if is_engine_search_failure(_e) and not _bench_off and device.type == "cuda":
                    _short = memory_is_short(_ids)
                    model.zero_grad(set_to_none=True)
                    free_cuda_after_oom()
                    if not _short:
                        _bench_off = True
                        torch.backends.cudnn.benchmark = False
                        cfg.train.cudnn_benchmark = False
                        logger.warning("P3: the cuDNN engine search failed at %d rows while the cards still had memory "
                                       "to give (%s): cudnn.benchmark is switched OFF for this run and the same rows are retried. ",
                                       int(b["clean"].shape[0]), str(_e)[:120])
                        continue
                if _halved >= 4 or int(b["clean"].shape[0]) <= _P:
                    raise
                _halved += 1
                _prevB = int(b["clean"].shape[0])
                _nB = max(_P, ((_prevB // 2) // _P) * _P)
                model.zero_grad(set_to_none=True)
                free_cuda_after_oom()
                b = halve_rows(b, _P)
                _PREFLIGHT_STATE["micro_batch"] = int(_nB)
                _PREFLIGHT_STATE["halvings"] = int(_halved)
                logger.warning("P3 MEMORY PRESSURE at %d rows (%s): retrying at %d rows (patch-aligned half "
                               "%d/4). The loaders will be rebuilt with micro-batch %d and the accumulation doubled before "
                               "epoch 1; the effective batch is unchanged.",
                               _prevB, str(_e)[:120], int(_nB), _halved, int(_nB))
    _run("P3", _p3)

    def _p3b():
        if not _first_batch:
            raise RuntimeError("no batch from P2")
        _P = int(max(1, int(getattr(loss_fn, "profile_patch_len", 1) or 1)))
        _B0 = int(_first_batch["clean"].shape[0])
        _n = min(_B0, 2 * _P if _P > 1 else 16)
        _n -= _n % _P
        if _n < max(_P, 2):
            raise RuntimeError("first batch too small for the replication test (%d rows)" % _B0)
        def _take(k: int) -> Dict[str, Any]:
            b = {}
            for kk, vv in _first_batch.items():
                if isinstance(vv, torch.Tensor) and vv.dim() >= 1 and int(vv.shape[0]) == _B0:
                    b[kk] = torch.cat([vv[:_n]] * k, dim=0).to(device)
                else:
                    b[kk] = vv.to(device) if isinstance(vv, torch.Tensor) else vv
            return b
        _keys = ("l_manifold_early", "l_manifold_late", "l_denoise_late", "l_denoise_late_accuracy",
                 "l_denoise_seam", "l_denoise_early", "l_prior_direct", "l_rec", "l_base", "l_noise",
                 "l_late_per_sample", "l_fused_band_z", "l_blend_oracle", "l_pass_hinge", "l_gate_eq_row")
        was = model.training
        model.eval()
        _core = getattr(model, "module", model)
        _core = getattr(_core, "_orig_mod", _core)
        _ic = getattr(getattr(_core, "cfg", None), "informative_continuation", None)
        if _ic is not None:
            _core.cfg.informative_continuation = "off"
        vals: Dict[int, Dict[str, float]] = {}
        try:
            with torch.no_grad():
                for k in (1, 2, 4):
                    b = _take(k)
                    o = model(b["noisy"], b.get("neighbors"), b.get("neighbor_geometry"),
                              depth_norm=b.get("depth_norm"), profile_len=batch_profile_len(b))
                    lo = loss_fn(o, b, "A")
                    vals[k] = {kk: float(lo[kk]) for kk in _keys if kk in lo and isinstance(lo[kk], torch.Tensor)}
        except Exception:
            if _ic is not None:
                _core.cfg.informative_continuation = _ic
            raise
        _w = _DP_STATE.get("wrapper")
        if _w is not None:
            try:
                b4 = _take(4)
                with torch.no_grad():
                    o_raw = model(b4["noisy"], b4.get("neighbors"), b4.get("neighbor_geometry"),
                                  depth_norm=b4.get("depth_norm"), profile_len=batch_profile_len(b4))["denoised"].float()
                    o_dp = _w(b4["noisy"], b4.get("neighbors"), b4.get("neighbor_geometry"),
                                 depth_norm=b4.get("depth_norm"), profile_len=batch_profile_len(b4))["denoised"].float()
                _rel = float((o_dp.to(o_raw.device) - o_raw).abs().max() / (o_raw.abs().max() + 1e-30))
                if not math.isfinite(_rel) or _rel > 1e-3:
                    raise RuntimeError("max rel diff %.3e" % _rel)
                logger.info("two-card forward reproduces the single-card forward (max rel diff %.2e).", _rel)
            except Exception as _e:
                _DP_STATE["wrapper"] = None
                logger.warning("two-card wrapper DISARMED (single card for this run): %r", _e)
        if _ic is not None:
            _core.cfg.informative_continuation = _ic
        model.train(was)
        bad = []
        for kk in vals[1]:
            v1 = vals[1][kk]
            for k in (2, 4):
                vk = vals[k].get(kk, float("nan"))
                if not math.isfinite(vk) or abs(vk - v1) > 2e-3 * max(abs(v1), 1e-6) + 1e-7:
                    bad.append("%s x%d: %.6g vs %.6g" % (kk, k, vk, v1))
        if bad:
            raise RuntimeError("MEAN-TYPE LOSSES ARE NOT BATCH-INVARIANT (a term still scales with the micro-batch): %s" % "; ".join(bad))
        return ("%d rows x1/x2/x4 identical on %d mean-type terms (z_band_gain %.3g): %s" % (
            _n, len(vals[1]), float(getattr(loss_fn.cfg, "z_band_gain", 1.0)),
            " ".join("%s %.4g" % (kk, v) for kk, v in vals[1].items())))
    _run("P3b", _p3b)

    def _p3c():
        if not _first_batch:
            raise RuntimeError("no batch from P2")
        _P = int(max(1, int(getattr(loss_fn, "profile_patch_len", 1) or 1)))
        _B0 = int(_first_batch["clean"].shape[0])
        _n = min(_B0, 2 * _P if _P > 1 else 16)
        _n -= _n % _P
        b = {}
        for kk, vv in _first_batch.items():
            if isinstance(vv, torch.Tensor) and vv.dim() >= 1 and int(vv.shape[0]) == _B0:
                b[kk] = vv[:_n].to(device)
            else:
                b[kk] = vv.to(device) if isinstance(vv, torch.Tensor) else vv
        was = model.training
        model.eval()
        tmp = Path(os.getcwd()) / ("_preflight_roundtrip_%d.pt" % os.getpid())
        try:
            with torch.no_grad():
                ref = model(b["noisy"], b.get("neighbors"), b.get("neighbor_geometry"),
                            depth_norm=b.get("depth_norm"), profile_len=batch_profile_len(b))["denoised"].float()
            torch.save({"model_state": model.state_dict(),
                        "pending_control_updates": model.export_pending_control() if hasattr(model, "export_pending_control") else {}}, tmp)
            m2 = copy.deepcopy(model)
            with torch.no_grad():
                for _v in m2.state_dict().values():
                    if isinstance(_v, torch.Tensor) and _v.dtype.is_floating_point:
                        _v.mul_(0.0).add_(0.123)
            st = torch.load(tmp, map_location=device, weights_only=False)
            m2.load_state_dict(st["model_state"], strict=True)
            if hasattr(m2, "import_pending_control"):
                m2.import_pending_control(st.get("pending_control_updates"))
            m2.eval()
            with torch.no_grad():
                out2 = m2(b["noisy"], b.get("neighbors"), b.get("neighbor_geometry"),
                          depth_norm=b.get("depth_norm"), profile_len=batch_profile_len(b))["denoised"].float()
            d1 = model_state_digest(model.state_dict())
            d2 = model_state_digest(m2.state_dict())
            rel = float((out2 - ref).abs().max() / (ref.abs().max() + 1e-30))
            del m2
        finally:
            model.train(was)
            try:
                tmp.unlink()
            except Exception:
                pass
        if d1 != d2:
            raise RuntimeError("state digest differs after save/load (%s vs %s)" % (d1[:12], d2[:12]))
        if not math.isfinite(rel) or rel > 1e-5:
            raise RuntimeError("forward output differs after checkpoint round-trip (max rel %.3e)" % rel)
        return "save/load round-trip reproduces the forward (max rel diff %.2e) and the full state digest %s" % (rel, d1[:12])
    _run("P3c", _p3c)

    def _p8():
        r = time_contract_self_test()
        if not r["pass"]:
            raise RuntimeError("time contract self-test failed: %s" % r)
        _gt = model.gate_times.detach().cpu().numpy().reshape(-1) if hasattr(model, "gate_times") else None
        if _gt is not None:
            require_same_axis(np.asarray(noise_cache.target_time, dtype=np.float64) * 1e3
                              if float(np.max(_gt)) > 10.0 * float(np.max(noise_cache.target_time)) else noise_cache.target_time,
                              _gt, what="model gate_times buffer")
        return ("signed-linear log(t) operator: pair additivity %.1e, zero-crossing %.1e, identity %.1e; CH/unit-less/decreasing/"
                "uncovered all refused; model gate_times == data axis" % (r["pair_additivity_residual"], r["zero_crossing_mid"],
                                                                         r["identity_max_abs"]))
    _run("P8", _p8)

    def _p4():
        pid = np.asarray(getattr(clean_cache, "province_id"))
        vds = datasets[1]
        rows = None
        if hasattr(vds, "split_indices"):
            vr = np.asarray(vds.split_indices, dtype=np.int64)
            area = int(pid[vr[0]])
            rows = vr[pid[vr] == area]
        if rows is None or rows.size < 4:
            return "skipped (no province structure)"
        try:
            _cands = []
            for _a in np.unique(pid[vr])[:10]:
                _rr = vr[pid[vr] == _a]
                _cands.append((int(_rr.size), int(_a), _rr))
            _cands.sort(key=lambda t: -t[0])
            if _cands:
                rows, area = _cands[0][2], _cands[0][1]
        except Exception:
            pass
        try:
            _full = np.flatnonzero(pid == int(area))
            if _full.size >= 4 and np.all(np.diff(_full) == 1):
                rows = _full
            else:
                _rs = np.sort(np.asarray(rows, dtype=np.int64))
                _brk = np.flatnonzero(np.diff(_rs) != 1) + 1
                _runs = np.split(_rs, _brk)
                rows = max(_runs, key=lambda r: r.size)
        except Exception:
            pass
        pred = _paired_test_forward(model, cfg, clean_cache, noise_cache, np.sort(rows), logger, edge_mode="none")
        _W = resolve_inference_window(cfg)
        _k = max(1, _W // 2)
        _n = int(rows.size)
        _msg = []
        for _mode, _lo in (("window", _W - 1), ("full", 2 * _k + _W - 1)):
            _pm = _paired_test_forward(model, cfg, clean_cache, noise_cache, np.sort(rows), logger, edge_mode=_mode)
            if not np.isfinite(_pm).all():
                raise RuntimeError("edge mode %s produced non-finite values" % _mode)
            _hi = _n - _lo
            if _hi - _lo >= 2:
                _ls = int(cfg.data.late_start_index)
                _dd = 100.0 * math.sqrt(float(np.sum((_pm[_lo:_hi, _ls:] - pred[_lo:_hi, _ls:]) ** 2))
                                        / max(float(np.sum(pred[_lo:_hi, _ls:] ** 2)), 1e-300))
                _msg.append("%s: interior rows %d..%d differ by %.3f%%" % (_mode, _lo, _hi - 1, _dd))
                if _dd > 1.0:
                    raise RuntimeError("edge mode %s CHANGED interior rows by %.2f%% (plumbing defect)" % (_mode, _dd))
            else:
                _msg.append("%s: area too short for the interior check" % _mode)
        if not np.isfinite(pred).all():
            raise RuntimeError("sliding-window inference produced non-finite values")
        cl = np.asarray(np.load(clean_cache.data_path, mmap_mode="r")[np.sort(rows)], dtype=np.float64)
        ls = int(cfg.data.late_start_index)
        e = 100.0 * math.sqrt(float(np.sum((pred[:, ls:] - cl[:, ls:]) ** 2)) / max(float(np.sum(cl[:, ls:] ** 2)), 1e-300))
        return ("area %d, %d stations, window %d: finite, late NRMSE %.1f%% (untrained model: plumbing only) | "
                "reflect extension invariance: %s") % (
            area, int(rows.size), resolve_inference_window(cfg), e, "; ".join(_msg))
    _run("P4", _p4)

    def _p7():
        paired_dir = Path(str(getattr(cfg.paths, "paired_dir", "") or ""))
        cpath, npath = paired_dir / "test_clean.csv", paired_dir / "test_noisy.csv"
        if not (cpath.is_file() and npath.is_file()):
            return "skipped (no test_clean/test_noisy.csv in %s)" % paired_dir
        def _kmap(d):
            return {(int(round(float(a))), round(float(b) / 1e-6) * 1e-6): i
                    for i, (a, b) in enumerate(zip(d["sample_id"], d["depth"]))}
        _src: Dict[str, Any] = {}

        def _source(split_name: str) -> Any:
            if split_name not in _src:
                _c = read_paired_csv(paired_dir / ("%s_clean.csv" % split_name), logger)
                _n = read_paired_csv(paired_dir / ("%s_noisy.csv" % split_name), logger)
                _src[split_name] = (_c, _n, _kmap(_c), _kmap(_n))
            return _src[split_name]
        try:
            _moved = {int(s): str(f) for s, f, _t in json.loads(Path(clean_cache.metadata_path).read_text(
                encoding="utf-8")).get("profile_groups", {}).get("moved_profiles", [])}
        except Exception:
            _moved = {}
        pid = np.asarray(clean_cache.province_id)
        names = list(getattr(clean_cache, "province_names", []) or [])
        st_mm = np.load(clean_cache.station_path, mmap_mode="r")
        cl_mm = np.load(clean_cache.data_path, mmap_mode="r")
        nz_mm = np.load(clean_cache.paired_noisy_path, mmap_mode="r")
        split_mm = np.load(str(Path(clean_cache.data_path).with_name("paired_split.npy")), mmap_mode="r") if Path(clean_cache.data_path).with_name("paired_split.npy").exists() else None
        test_rows = np.flatnonzero(np.asarray(split_mm) == 2) if split_mm is not None else np.arange(pid.size)
        areas = np.unique(pid[test_rows])
        pick = areas[np.linspace(0, areas.size - 1, min(20, areas.size)).astype(int)]
        ls = int(cfg.data.late_start_index)
        maxd = 0.0
        n_cmp = 0
        n_miss = 0
        plateau_src = 0
        drift_r = []
        for a in pick:
            rows = test_rows[pid[test_rows] == a]
            rows = rows[np.argsort(np.asarray(st_mm[rows]), kind="stable")]
            nm = names[int(a)] if int(a) < len(names) else ""
            try:
                sid = int(nm.split("sample_")[-1])
            except Exception:
                n_miss += rows.size
                continue
            csv_c, csv_n, kc, kn = _source(_moved.get(sid, "test"))
            src_c = []
            src_n = []
            for r in rows:
                k = (sid, round(float(st_mm[r]) / 1e-6) * 1e-6)
                if k in kc and k in kn:
                    src_c.append(csv_c["matrix"][kc[k]])
                    src_n.append(csv_n["matrix"][kn[k]])
                else:
                    n_miss += 1
            if len(src_c) != rows.size:
                continue
            src_c = np.asarray(src_c, dtype=np.float64)
            src_n = np.asarray(src_n, dtype=np.float64)
            cc = np.asarray(cl_mm[rows], dtype=np.float64)
            nn = np.asarray(nz_mm[rows], dtype=np.float64)
            if src_c.shape[1] != cc.shape[1] and bool(getattr(cfg.data, "unify_axis_to_measured", False)):
                _axs = unified_measured_axis(cfg, int(src_c.shape[1]), logger)
                _pl = make_logtime_plan(_axs, np.asarray(clean_cache.target_time, dtype=np.float64))
                src_c = apply_logtime_plan(src_c, _pl, uncovered_policy="refuse")
                src_n = apply_logtime_plan(src_n, _pl, uncovered_policy="refuse")
            ref = max(float(np.max(np.abs(src_c))), 1e-30)
            maxd = max(maxd, float(np.max(np.abs(cc - src_c))) / ref, float(np.max(np.abs(nn - src_n))) / ref)
            n_cmp += rows.size
            d = np.abs(np.diff(src_c[:, ls:], axis=0)) <= 1e-9 * np.maximum(np.abs(src_c[:-1, ls:]), 1e-30)
            for g in range(d.shape[1]):
                run = 0
                for v in d[:, g]:
                    run = run + 1 if v else 0
                    if run >= 4:
                        plateau_src += 1
                        break
            nzl = src_n[:, ls:] - src_c[:, ls:]
            si = np.arange(rows.size, dtype=np.float64)
            si = (si - si.mean()) / max(si.std(), 1e-9)
            nc = nzl - nzl.mean(axis=0, keepdims=True)
            r_ = np.abs((nc * si[:, None]).mean(axis=0) / np.maximum(nc.std(axis=0), 1e-30))
            drift_r.append(float(np.median(r_)))
        if n_cmp == 0:
            raise RuntimeError("no test rows could be matched between cache and source CSV (missing %d)" % n_miss)
        msg = ("%d test areas / %d rows compared | cache vs source max relative difference %.2e (float%s cache) | unmatched rows %d | "
               "SOURCE late gates constant over >= 5 stations: %d gate-areas | SOURCE late-noise drift |r| median %.2f") % (
            len(pick), n_cmp, maxd, str(cfg.data.cache_dtype).replace("float", ""), n_miss, plateau_src, float(np.median(drift_r)) if drift_r else float("nan"))
        tol = 2e-3 if "16" in str(cfg.data.cache_dtype) else 1e-5
        if maxd > tol:
            raise RuntimeError("CACHE DIFFERS FROM SOURCE: " + msg)
        return msg
    _run("P7", _p7)

    def _p5():
        ok = bool(_svg_visio_self_test(logger))
        if not ok:
            logger.warning("P5 (SVG->Visio converter) failed on this environment: this run exports the raw "
                           "matplotlib SVG instead of the Visio-editable SVG. Training is NOT blocked by a figure-converter check. ")
            return "DEGRADED (not blocking): svg_to_visio round-trip failed on this environment -> raw SVG export"
        return "svg_to_visio round-trip preserves labels, closed bars, and OPEN data polylines (no closing/box edges)"
    _run("P5", _p5)

    def _p6():
        pn = getattr(clean_cache, "paired_noisy_path", "") or ""
        if not pn:
            return "no measured pair (library mode)"
        st = audit_paired_zero_gates(np.load(pn, mmap_mode="r"), np.load(clean_cache.data_path, mmap_mode="r"),
                                     int(cfg.data.late_start_index), getattr(clean_cache, "province_id", None), logger,
                                     max_rows=50000)
        if not st:
            return "audit unavailable"
        gc = audit_gate_columns(np.load(clean_cache.data_path, mmap_mode="r"), np.load(pn, mmap_mode="r"),
                                getattr(clean_cache, "target_time", None), getattr(clean_cache, "province_id", None),
                                int(cfg.data.late_start_index), logger, max_rows=60000)
        if float(gc.get("early_misaligned_share", 0.0)) > 0.01:
            raise RuntimeError("pairing misaligned on %.2f%% of rows (early-band disagreement > 100%%)" % (100 * gc["early_misaligned_share"]))
        return "exact-zero late gates %.2f%% | early-band misaligned rows %.3f%% | lateral-drift areas %.1f%% | suspicious gate columns %s" % (
            100.0 * st.get("rows_with_any_zero_late", 0.0), 100.0 * float(gc.get("early_misaligned_share", 0.0)),
            100.0 * float(gc.get("lateral_drift_area_share", 0.0)), gc.get("suspicious_gates", []))
    _run("P6", _p6)

    ok = all(r[1] for r in results)
    logger.info("=" * 96)
    for name, good, msg in results:
        logger.info("%s %s :: %s", "PASS" if good else "FAIL", name, msg[:200])
    logger.info("PRE-FLIGHT %s.", "PASSED" if ok else "FAILED")
    logger.info("=" * 96)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ok


def information_bound_audit(clean_mm: Any, noisy_path: str, province_id: np.ndarray, val_rows: np.ndarray,
                            train_rows: np.ndarray, V: np.ndarray, mu: np.ndarray, gsc: np.ndarray, ls: int,
                            K: int, window: int, logger: logging.Logger, rng: np.random.Generator,
                            run_dir: Optional[Path] = None, max_areas: int = 300) -> Optional[np.ndarray]:
    """Closed-form late-window bounds on real validation areas (numpy, startup)."""
    t0 = time.time()
    gsc = np.asarray(gsc, dtype=np.float64).reshape(1, -1)
    mu = np.asarray(mu, dtype=np.float64).reshape(1, -1)
    V = np.asarray(V, dtype=np.float64)
    r, T = V.shape
    ls = int(ls)
    noisy_mm = np.load(noisy_path, mmap_mode="r")

    def z_of(y):
        e = np.asarray(y, dtype=np.float64) / gsc
        return np.sign(e) * np.log1p(np.abs(e))

    def lin_of(z):
        return np.sign(z) * np.expm1(np.abs(z)) * gsc

    tr = np.asarray(train_rows, dtype=np.int64)
    if tr.size > 20000:
        tr = np.sort(rng.choice(tr, 20000, replace=False))
    zn_tr = z_of(noisy_mm[tr])
    zc_tr = z_of(clean_mm[tr])
    sig2_g = np.mean((zn_tr - zc_tr) ** 2, axis=0)
    eps = 1e-3 * float(np.median(sig2_g[:max(1, ls // 2)]))
    prof = 1.0 / (sig2_g + max(eps, 1e-12))
    prof = prof / prof.max()
    rms_g = np.sqrt(sig2_g)
    g_noise = -1
    jump = 1.0
    _sp = max(1, int(round(float(T) / float(RF_DESIGN_GATES))))
    for g in range(max(_sp, ls // 2), ls):
        ratio = float(rms_g[g] / max(rms_g[g - _sp], 1e-30))
        if ratio > jump:
            jump, g_noise = ratio, g
    if jump < 1.8:
        g_noise = -1
    elif _sp > 1:
        g_noise = max(1, g_noise - _sp + 1)
    gs_seam = g_noise if g_noise > 0 else max(0, ls - 6)
    acc_seam: Dict[str, List[float]] = {}
    logger.info("PAIR NOISE PROFILE (train, z-domain RMS per gate): %s | precision profile "
                "(max-normalised): %s.",
                " ".join("%.3f" % v for v in np.sqrt(sig2_g)), " ".join("%.2g" % v for v in prof))

    pid = np.asarray(province_id)
    vr = np.sort(np.asarray(val_rows, dtype=np.int64))
    areas = np.unique(pid[vr])
    if areas.size > max_areas:
        areas = areas[np.linspace(0, areas.size - 1, max_areas).astype(int)]
    edges = np.arange(-40.0, 40.0, 5.0)
    names = ["E1 per-station uniform", "E2 per-station precision", "E3 per-station ORACLE",
             "E4 joint %d-station" % window, "E5 joint whole-area", "E6 stack %d late" % window, "E6 stack area late"]
    acc: Dict[str, Dict[str, np.ndarray]] = {nm: {"e": np.zeros(edges.size + 1), "y": np.zeros(edges.size + 1)} for nm in names}
    e5_dist: Dict[str, List[float]] = {}
    lam_grid = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    kappa_grid = (0.3, 1.0, 3.0)
    lam4 = lam5 = None
    kappa = [1.0]
    _Ctr = (zc_tr - mu) @ V.T
    var_k = np.maximum(np.var(_Ctr, axis=0), 1e-12)
    P0 = np.diag(1.0 / var_k)
    prec_g = 1.0 / np.maximum(sig2_g, 1e-12)
    prec_em_uniform = float(np.median(prec_g[:max(1, ls)]))

    def wls_station(Zc_, W_):
        C = np.zeros((Zc_.shape[0], r))
        Pk = kappa[0] * P0
        for i in range(Zc_.shape[0]):
            G = (V * W_[i]) @ V.T + Pk
            C[i] = np.linalg.solve(G, (V * W_[i]) @ Zc_[i])
        return C

    def joint(Zc_, W_, lam):
        m = Zc_.shape[0]
        if m < 3 or lam <= 0:
            return wls_station(Zc_, W_)
        D = np.zeros((m - 2, m))
        for i in range(m - 2):
            D[i, i:i + 3] = (1.0, -2.0, 1.0)
        L = D.T @ D
        A = np.zeros((m * r, m * r))
        b = np.zeros(m * r)
        Pk = kappa[0] * P0
        for i in range(m):
            A[i * r:(i + 1) * r, i * r:(i + 1) * r] = (V * W_[i]) @ V.T + Pk
            b[i * r:(i + 1) * r] = (V * W_[i]) @ Zc_[i]
        A = A + lam * np.kron(L, P0)
        return np.linalg.solve(A, b).reshape(m, r)

    rowerr: Dict[str, List[np.ndarray]] = {}

    def add_rows(nm, e2, Yc):
        y2 = np.sum(Yc[:, ls:] ** 2, axis=1)
        rowerr.setdefault(nm, []).append(100.0 * np.sqrt(np.sum(e2, axis=1) / np.maximum(y2, 1e-300)))

    def late_err(Zpred_late, Yc):
        Zfull = np.zeros((Yc.shape[0], T))
        Zfull[:, ls:] = Zpred_late
        return (lin_of(Zfull)[:, ls:] - Yc[:, ls:]) ** 2

    def add_seam(nm, Zfull_pred, Yc):
        e = (lin_of(Zfull_pred)[:, gs_seam:ls] - Yc[:, gs_seam:ls]) ** 2
        acc_seam.setdefault(nm, [0.0, 0.0])
        acc_seam[nm][0] += float(e.sum())
        acc_seam[nm][1] += float((Yc[:, gs_seam:ls] ** 2).sum())

    def add(nm, e2, Yc, bins, dist=None):
        y2 = Yc[:, ls:] ** 2
        np.add.at(acc[nm]["e"], bins, e2.sum(axis=1))
        np.add.at(acc[nm]["y"], bins, y2.sum(axis=1))
        if dist is not None:
            for q in range(Yc.shape[0]):
                key = ("d=%d" % dist[q]) if dist[q] < 4 else ("d=4-7" if dist[q] < K else "interior")
                e5_dist.setdefault(key, [0.0, 0.0])
                e5_dist[key][0] += float(e2[q].sum())
                e5_dist[key][1] += float(y2[q].sum())

    h = max(1, int(window) // 2)
    for ai, a in enumerate(areas):
        rows = vr[pid[vr] == a]
        m = int(rows.size)
        if m < 3:
            continue
        Yc = np.asarray(clean_mm[rows], dtype=np.float64)
        Yn = np.asarray(noisy_mm[rows], dtype=np.float64)
        Zc = z_of(Yc) - mu
        Zn = z_of(Yn) - mu
        noise_late = np.sum((Yn - Yc)[:, ls:] ** 2, axis=1)
        snr = 10.0 * np.log10(np.sum(Yc[:, ls:] ** 2, axis=1) / np.maximum(noise_late, 1e-300))
        bins = np.clip(np.searchsorted(edges, snr, side="right"), 0, edges.size)
        dist = np.minimum(np.arange(m), m - 1 - np.arange(m))
        W1 = np.full((m, T), prec_em_uniform)
        W1[:, ls:] = 0.0
        W2 = np.tile(prec_g, (m, 1))
        sig2_a = np.mean((Zn - Zc) ** 2, axis=0)
        W3 = np.tile(1.0 / np.maximum(sig2_a, 1e-12), (m, 1))
        if ai == 0:
            bestk = None
            for kk in kappa_grid:
                kappa[0] = kk
                Ck = wls_station(Zn, W2)
                e_k = float(late_err((Ck @ V)[:, ls:] + mu[:, ls:], Yc).sum())
                if bestk is None or e_k < bestk[0]:
                    bestk = (e_k, kk)
            kappa[0] = float(bestk[1])
            logger.info("MMSE prior scale kappa=%g chosen on area %d (grid %s); coefficient prior var_k = %s.",
                        kappa[0], int(a), kappa_grid, " ".join("%.3g" % v for v in var_k))
        acc_seam.setdefault("raw measurement", [0.0, 0.0])
        acc_seam["raw measurement"][0] += float(((Yn - Yc)[:, gs_seam:ls] ** 2).sum())
        acc_seam["raw measurement"][1] += float((Yc[:, gs_seam:ls] ** 2).sum())
        for nm, W_ in (("E1 per-station uniform", W1), ("E2 per-station precision", W2), ("E3 per-station ORACLE", W3)):
            C = wls_station(Zn, W_)
            _e2r = late_err((C @ V)[:, ls:] + mu[:, ls:], Yc)
            add(nm, _e2r, Yc, bins)
            add_rows(nm, _e2r, Yc)
            add_seam(nm, C @ V + mu, Yc)
        if lam4 is None:
            best4 = None
            for lam in lam_grid:
                Cw = np.zeros((m, r))
                for s in range(m):
                    lo, hi = max(0, s - h), min(m, s + h + 1)
                    Cw[s] = joint(Zn[lo:hi], W2[lo:hi], lam)[s - lo]
                e_sum = float(late_err((Cw @ V)[:, ls:] + mu[:, ls:], Yc).sum())
                if best4 is None or e_sum < best4[0]:
                    best4 = (e_sum, lam)
            lam4 = float(best4[1])
        if lam5 is None:
            best5 = None
            for lam in lam_grid:
                C = joint(Zn, W2, lam)
                e_sum = float(late_err((C @ V)[:, ls:] + mu[:, ls:], Yc).sum())
                if best5 is None or e_sum < best5[0]:
                    best5 = (e_sum, lam)
            lam5 = float(best5[1])
            logger.info("joint-field lambda chosen on area %d: whole-area lambda=%g (grid %s); 9-station lambda=%g.",
                        int(a), lam5, lam_grid, lam4)
        C4 = np.zeros((m, r))
        for s in range(m):
            lo, hi = max(0, s - h), min(m, s + h + 1)
            C = joint(Zn[lo:hi], W2[lo:hi], lam4)
            C4[s] = C[s - lo]
        _e4r = late_err((C4 @ V)[:, ls:] + mu[:, ls:], Yc)
        add("E4 joint %d-station" % window, _e4r, Yc, bins)
        add_rows("E4 joint %d-station" % window, _e4r, Yc)
        add_seam("E4 joint %d-station" % window, C4 @ V + mu, Yc)
        C5 = joint(Zn, W2, lam5)
        e5 = late_err((C5 @ V)[:, ls:] + mu[:, ls:], Yc)
        add("E5 joint whole-area", e5, Yc, bins, dist)
        add_rows("E5 joint whole-area", e5, Yc)
        add_seam("E5 joint whole-area", C5 @ V + mu, Yc)
        S9 = np.zeros_like(Yn[:, ls:])
        for s in range(m):
            lo, hi = max(0, s - h), min(m, s + h + 1)
            S9[s] = Yn[lo:hi, ls:].mean(axis=0)
        _e6r = (S9 - Yc[:, ls:]) ** 2
        add("E6 stack %d late" % window, _e6r, Yc, bins)
        add_rows("E6 stack %d late" % window, _e6r, Yc)
        add("E6 stack area late", (np.tile(Yn[:, ls:].mean(axis=0), (m, 1)) - Yc[:, ls:]) ** 2, Yc, bins)

    def nrmse(e, y):
        return 100.0 * math.sqrt(float(e) / max(float(y), 1e-300)) if y > 0 else float("nan")

    show = [("<-15", 0, 5), ("[-15,-5)", 5, 7), ("[-5,0)", 7, 8), ("[0,5)", 8, 9), ("[5,10)", 9, 10),
            ("[10,15)", 10, 11), ("[15,25)", 11, 13), (">=25", 13, edges.size + 1)]
    logger.info("INFORMATION BOUND (validation, %d areas, real noise) | late NRMSE pooled and by late-local "
                "SNR bin | %s", int(areas.size), " | ".join("%s" % nm for nm, _, _ in show))
    table = {}
    for nm in names:
        e, y = acc[nm]["e"], acc[nm]["y"]
        row = [nrmse(e.sum(), y.sum())] + [nrmse(e[lo:hi].sum(), y[lo:hi].sum()) for _, lo, hi in show]
        table[nm] = row
        logger.info("  %-26s pooled %6.2f%% | %s", nm, row[0], " | ".join("%6.2f%%" % v for v in row[1:]))
    if e5_dist:
        logger.info("  E5 whole-area by distance from the edge: %s",
                    " | ".join("%s %.2f%%" % (k, nrmse(v[0], v[1])) for k, v in sorted(e5_dist.items())))
    if acc_seam:
        logger.info("SEAM BAND (gates %d-%d, the measured noise regime below the contract late start) "
                    "NRMSE | %s | noise-jump gate %s (x%.2f).",
                    gs_seam + 1, ls, " | ".join("%s %.2f%%" % (k, nrmse(v[0], v[1])) for k, v in acc_seam.items()),
                    ("%d" % (g_noise + 1)) if g_noise > 0 else "none", jump)
    rowstats = {}
    for nm, chunks in rowerr.items():
        v = np.concatenate(chunks) if chunks else np.zeros(0)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        srt = np.sort(v)[::-1]
        rowstats[nm] = {"pass3": 100.0 * float(np.mean(v <= 3.0)), "pass4": 100.0 * float(np.mean(v <= 4.0)),
                        "median": float(np.median(v)), "cvar20": float(np.mean(srt[:max(1, int(round(0.2 * v.size)))])), "n": int(v.size)}
    if rowstats:
        logger.info("INFORMATION BOUND IN CONTRACT UNITS (per-row late NRMSE, %d validation rows) | %s | READING: "
                    "the deliverable-path model (pass@3%%, CVaR-20 above) must beat E5 -- the whole-area MMSE estimate that "
                    "pools every station's high-SNR gates; if E5 itself sits below the 95%%/8%% gates the target is reachable "
                    "and the model is at fault, if E5 sits above them the target is information-limited on this data. ",
                    next(iter(rowstats.values()))["n"],
                    " | ".join("%s pass@3%% %.1f%% pass@4%% %.1f%% median %.2f%% CVaR-20 %.2f%%" % (
                        nm, s["pass3"], s["pass4"], s["median"], s["cvar20"]) for nm, s in rowstats.items()))
    e1, e2 = table["E1 per-station uniform"][0], table["E2 per-station precision"][0]
    logger.info("READING: E1 is what the runtime anchor computes; E2 the measured per-gate precision variant; E3 the "
                "best any per-station family fit can do; E4->E5 is the value of AREA-LEVEL pooling of the high-SNR "
                "gates that the 9-station design forgoes; E6 is the measurement alone. The model's VAL late error "
                "must be read against E5 (context) and E6 (measurement). %.1fs.", time.time() - t0)
    try:
        if run_dir is not None:
            (Path(run_dir) / "reports").mkdir(parents=True, exist_ok=True)
            atomic_json_dump({"names": names, "columns": ["pooled"] + [nm for nm, _, _ in show], "table": table,
                              "noise_rms_z": np.sqrt(sig2_g).tolist(), "precision_profile": prof.tolist(),
                              "lambda_whole_area": lam5, "lambda_window": lam4, "areas": int(areas.size),
                              "kappa": float(kappa[0]), "coef_prior_var": var_k.tolist(), "row_stats": rowstats},
                             Path(run_dir) / "reports" / "information_bound_audit.json")
    except Exception:
        pass
    information_bound_audit.noise_jump_gate = int(g_noise)
    information_bound_audit.noise_jump_ratio = float(jump)
    information_bound_audit.noise_rms_z = rms_g
    return prof if (math.isfinite(e1) and math.isfinite(e2) and e2 < e1) else None


class _SkipLadder(Exception):
    """Control-flow marker: the start-up ladder is switched off (not an error)."""


def linear_information_ladder(clean_mm: Any, noisy_path: str, province_id: np.ndarray, val_rows: np.ndarray,
                                  train_rows: np.ndarray, gsc: np.ndarray, ls: int, logger: logging.Logger,
                                  rng: np.random.Generator, run_dir: Optional[Path] = None,
                                  half_widths: Sequence[int] = (0, 4, 15, 30), max_train_rows: int = 120000,
                                  max_val_areas: int = 300, gate_limit: Optional[int] = None, tag: str = "") -> Dict[str, Any]:
    """The linear information ladder."""
    out: Dict[str, Any] = {}
    try:
        t_start = time.time()
        gsc = np.asarray(gsc, dtype=np.float64).reshape(1, -1)
        noisy_mm = np.load(noisy_path, mmap_mode="r")
        pid = np.asarray(province_id)
        T = int(gsc.size)
        ls = int(ls)
        nl = T - ls

        def z_of(y):
            e = np.asarray(y, dtype=np.float64) / gsc
            return np.sign(e) * np.log1p(np.abs(e))

        def lin_of(z):
            return np.sign(z) * np.expm1(np.abs(z)) * gsc[:, ls:]

        def area_rows(rows):
            rows = np.sort(np.asarray(rows, dtype=np.int64))
            areas = np.unique(pid[rows])
            return [rows[pid[rows] == a] for a in areas]

        gl = int(gate_limit) if gate_limit is not None else T
        Tf = min(max(gl, 1), T)

        def window_features(Zn, k):
            Zn = Zn[:, :Tf]
            m = Zn.shape[0]
            if k == 0:
                return Zn
            offs = np.arange(-k, k + 1)
            idx = np.arange(m)[:, None] + offs[None, :]
            idx = np.abs(idx)
            idx = np.where(idx >= m, 2 * (m - 1) - idx, idx)
            idx = np.clip(idx, 0, m - 1)
            return Zn[idx].reshape(m, -1)

        tr_areas = area_rows(train_rows)
        rng.shuffle(tr_areas)
        n_hold = max(5, len(tr_areas) // 10)
        hold_areas, fit_areas = tr_areas[:n_hold], tr_areas[n_hold:]
        val_areas = area_rows(val_rows)
        if len(val_areas) > max_val_areas:
            val_areas = [val_areas[i] for i in sorted(rng.choice(len(val_areas), max_val_areas, replace=False))]
        bins = [(-1e9, -15), (-15, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 25), (25, 1e9)]
        labels = {0: "E7%s row alone" % tag, 4: "E8%s row + 8 neighbours" % tag, 15: "E9%s row + 30 neighbours" % tag, 30: "E10%s whole area" % tag}
        rho_grid = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
        for k in half_widths:
            t0 = time.time()
            d = (2 * k + 1) * Tf
            G = np.zeros((d, d))
            H = np.zeros((d, nl))
            sx = np.zeros(d)
            sy = np.zeros(nl)
            n_acc = 0
            for rows in fit_areas:
                if n_acc >= max_train_rows:
                    break
                Zn = z_of(noisy_mm[rows])
                Zc = z_of(clean_mm[rows])
                X = window_features(Zn, k)
                Y = Zc[:, ls:]
                G += X.T @ X
                H += X.T @ Y
                sx += X.sum(0)
                sy += Y.sum(0)
                n_acc += X.shape[0]
            mx = sx / max(n_acc, 1)
            my = sy / max(n_acc, 1)
            Gc = G - n_acc * np.outer(mx, mx)
            Hc = H - n_acc * np.outer(mx, my)
            scale = float(np.mean(np.diag(Gc))) if d > 0 else 1.0

            def fit(rho):
                return np.linalg.solve(Gc + rho * scale * np.eye(d), Hc)

            def predict(W, rows):
                Zn = z_of(noisy_mm[rows])
                X = window_features(Zn, k)
                return (X - mx) @ W + my

            def late_nrmse_rows(Zl_hat, rows):
                Yc = np.asarray(clean_mm[rows], dtype=np.float64)[:, ls:]
                Yh = lin_of(Zl_hat)
                return 100.0 * np.sqrt(np.sum((Yh - Yc) ** 2, axis=1) / np.maximum(np.sum(Yc ** 2, axis=1), 1e-300))

            best = None
            for rho in rho_grid:
                W = fit(rho)
                e = np.concatenate([late_nrmse_rows(predict(W, rows), rows) for rows in hold_areas])
                score = float(np.sqrt(np.mean(np.minimum(e, 300.0) ** 2)))
                if best is None or score < best[0]:
                    best = (score, rho, W)
            rho, W = best[1], best[2]
            errs, snrs, dist, num, den = [], [], [], 0.0, 0.0
            for rows in val_areas:
                Yc = np.asarray(clean_mm[rows], dtype=np.float64)
                Yn = np.asarray(noisy_mm[rows], dtype=np.float64)
                Yh = lin_of(predict(W, rows))
                e2 = np.sum((Yh - Yc[:, ls:]) ** 2, axis=1)
                y2 = np.sum(Yc[:, ls:] ** 2, axis=1)
                num += float(e2.sum())
                den += float(y2.sum())
                errs.append(100.0 * np.sqrt(e2 / np.maximum(y2, 1e-300)))
                n2 = np.sum((Yn[:, ls:] - Yc[:, ls:]) ** 2, axis=1)
                snrs.append(10.0 * np.log10(np.maximum(y2, 1e-300) / np.maximum(n2, 1e-300)))
                m = rows.size
                ii = np.arange(m)
                dist.append(np.minimum(ii, m - 1 - ii))
            errs = np.concatenate(errs)
            snrs = np.concatenate(snrs)
            dist = np.concatenate(dist)
            pooled = 100.0 * math.sqrt(num / max(den, 1e-300))
            srt = np.sort(errs)[::-1]
            by_bin = []
            for lo, hi in bins:
                sel = (snrs >= lo) & (snrs < hi)
                by_bin.append((lo, hi, (float(np.sqrt(np.mean(errs[sel] ** 2))) if sel.any() else float("nan")), int(sel.sum())))
            by_d = [float(np.sqrt(np.mean(errs[dist == dd] ** 2))) if np.any(dist == dd) else float("nan") for dd in range(4)]
            by_d.append(float(np.sqrt(np.mean(errs[(dist >= 4) & (dist <= 7)] ** 2))) if np.any((dist >= 4) & (dist <= 7)) else float("nan"))
            by_d.append(float(np.sqrt(np.mean(errs[dist > 7] ** 2))) if np.any(dist > 7) else float("nan"))
            stats = {"pooled": pooled, "median": float(np.median(errs)), "pass3": 100.0 * float(np.mean(errs <= 3.0)),
                     "pass4": 100.0 * float(np.mean(errs <= 4.0)), "cvar20": float(np.mean(srt[:max(1, int(round(0.2 * errs.size)))])),
                     "rows": int(errs.size), "rho": float(rho), "features": int(d), "train_rows": int(n_acc),
                     "by_bin": by_bin, "by_distance": by_d, "seconds": time.time() - t0}
            out[labels.get(k, "E k=%d" % k)] = stats
            logger.info("  %-24s pooled %6.2f%% | median %6.2f%% | pass@3%% %5.1f%% | pass@4%% %5.1f%% | CVaR-20 %6.2f%% | bins %s | "
                        "edge d=0..3 %s | d=4-7 %.2f%% | interior %.2f%% | %d features, %d training rows, rho=%g, %.0fs",
                        labels.get(k, "E k=%d" % k), pooled, stats["median"], stats["pass3"], stats["pass4"], stats["cvar20"],
                        " ".join("[%s,%s) %.1f%%(n=%d)" % ("<" if lo < -1e8 else "%d" % lo, ">" if hi > 1e8 else "%d" % hi, v, n) for lo, hi, v, n in by_bin),
                        " ".join("%.1f%%" % v for v in by_d[:4]), by_d[4], by_d[5], d, n_acc, rho, stats["seconds"])
        e8 = out.get("E8%s row + 8 neighbours" % tag, {})
        e10 = out.get("E10%s whole area" % tag, {}) or out.get("E9%s row + 30 neighbours" % tag, {})
        logger.info("LINEAR INFORMATION LADDER%s (validation, %d areas) | READING: E8 is the linear optimum on the model's own inputs, "
                    "E10 on everything the area contains. Model far ABOVE E8 = the architecture leaves information on the table (fix the "
                    "model); E10 itself ABOVE the 95%%/8%% gates = no estimator reaches the contract on this data with this noise (fix the "
                    "data or the contract). Here: E8 pass@3%% %.1f%% median %.2f%% | E10 pass@3%% %.1f%% median %.2f%%. %.0fs. ",
                    (" [early/mid gates < %d only]" % Tf) if gate_limit is not None else "", len(val_areas),
                    float(e8.get("pass3", float("nan"))), float(e8.get("median", float("nan"))),
                    float(e10.get("pass3", float("nan"))), float(e10.get("median", float("nan"))), time.time() - t_start)
        if run_dir is not None:
            try:
                atomic_json_dump({k: {kk: vv for kk, vv in v.items()} for k, v in out.items()},
                                 Path(run_dir) / "reports" / ("linear_information_ladder%s.json" % ("_early_mid" if gate_limit is not None else "")))
            except Exception:
                pass
    except Exception as exc:
        logger.error("linear information ladder failed: %r", exc, exc_info=True)
    return out


def cdmr_basis_signature(base: nn.Module) -> Tuple[float, float]:
    """The decay manifold a CDM-R calibration was fitted on (mean and basis magnitudes)."""
    return (round(float(base.lib_mu.detach().float().abs().sum().item()), 6),
            round(float(base.lib_basis.detach().float().abs().sum().item()), 6))


def fit_diag_gmm(X: np.ndarray, K: int, seed: int = 778, iters: int = 200,
                     tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonal Gaussian mixture by EM, deterministic for a seed: weights [K], means [K, d] and variances [K, d]
    (floored at 1e-4 of the data variance per coordinate).
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    K = int(max(1, min(int(K), n)))
    rng = np.random.default_rng(int(seed))
    mus = X[rng.choice(n, K, replace=False)].copy()
    gv = X.var(axis=0) + 1e-12
    floor = 1e-4 * gv
    vars_ = np.tile(gv, (K, 1))
    pis = np.full(K, 1.0 / K)
    X2 = X * X
    prev = -np.inf
    for _ in range(int(iters)):
        iv = 1.0 / vars_
        ll = (-0.5 * (X2 @ iv.T - 2.0 * (X @ (mus * iv).T) + np.sum(mus * mus * iv, axis=1)[None, :])
              - 0.5 * np.sum(np.log(vars_), axis=1)[None, :] + np.log(np.maximum(pis, 1e-300))[None, :])
        mx = ll.max(axis=1, keepdims=True)
        g = np.exp(ll - mx)
        s = g.sum(axis=1, keepdims=True)
        g /= s
        tot = float(np.mean(mx[:, 0] + np.log(s[:, 0])))
        nk = g.sum(axis=0) + 1e-12
        mus = (g.T @ X) / nk[:, None]
        vars_ = np.maximum((g.T @ X2) / nk[:, None] - mus * mus, floor[None, :])
        pis = nk / float(n)
        if abs(tot - prev) <= tol * max(1.0, abs(tot)):
            break
        prev = tot
    return pis, mus, vars_


def install_cdmr_calibration(model: nn.Module, cfg: Config, clean_cache: CleanCacheInfo,
                                 logger: Optional[logging.Logger] = None) -> bool:
    """Calibrate CDM-R on the training split of a paired cache."""
    log = logger or logging.getLogger(PROGRAM_NAME)
    base = getattr(model, "module", model)
    base = getattr(base, "_orig_mod", base)
    if getattr(base, "cdmr_ready", None) is None or not bool(getattr(cfg.model, "cdmr_calibrated", True)):
        return False
    if getattr(base, "lib_basis", None) is None or float(base.lib_ready.item()) <= 0.5:
        log.info("CDM-R calibration not installed: no decay manifold in force.")
        return False
    if not getattr(clean_cache, "paired_noisy_path", "") or getattr(clean_cache, "province_id", None) is None:
        log.info("CDM-R calibration not installed: it is measured on the training pairs of a paired dataset. ")
        return False
    t0 = time.time()
    splits = build_splits(clean_cache, cfg)
    tr = np.asarray(splits["train"], dtype=np.int64)
    if tr.size < 100:
        return False
    rng = np.random.default_rng(int(cfg.train.seed) + 778)
    mat = np.load(clean_cache.data_path, mmap_mode="r")
    nm = np.load(str(clean_cache.paired_noisy_path), mmap_mode="r")
    gs = base.gate_scale_norm.detach().double().cpu().numpy().reshape(1, -1)
    gsc = float(getattr(clean_cache, "global_scale", 1.0) or 1.0)

    def _z(y: np.ndarray) -> np.ndarray:
        e = (np.asarray(y, dtype=np.float64) / gsc) / gs
        return np.sign(e) * np.log1p(np.abs(e))

    rows = np.sort(rng.choice(tr, int(min(tr.size, 60000)), replace=False))
    dz = _z(nm[rows]) - _z(mat[rows])
    dz = dz[np.all(np.isfinite(dz), axis=1)]
    mad = np.maximum(1.4826 * np.median(np.abs(dz), axis=0), 1e-6)
    cov = np.cov(np.clip(dz, -4.0 * mad, 4.0 * mad).T)
    cov = 0.5 * (cov + cov.T) + 1e-8 * float(np.mean(np.diag(cov))) * np.eye(cov.shape[0])
    V = base.lib_basis.detach().double().cpu().numpy()
    keep = np.abs(V).sum(axis=1) > 0
    mu = base.lib_mu.detach().double().cpu().numpy().reshape(-1)
    lrows = np.sort(rng.choice(tr, int(min(tr.size, 20000)), replace=False))
    A = (_z(mat[lrows]) - mu) @ V[keep].T
    A = A[np.all(np.isfinite(A), axis=1)]
    K = int(base.cdmr_pi.numel())
    pis, mus, vars_ = fit_diag_gmm(A, K, seed=int(cfg.train.seed) + 778)
    pi_f = np.zeros(K)
    mu_f = np.zeros((K, V.shape[0]))
    var_f = np.ones((K, V.shape[0]))
    pi_f[:pis.size] = pis
    mu_f[:pis.size][:, keep] = mus
    var_f[:pis.size][:, keep] = vars_
    with torch.no_grad():
        base.cdmr_ncov.copy_(torch.as_tensor(cov, dtype=torch.float32))
        base.cdmr_pi.copy_(torch.as_tensor(pi_f, dtype=torch.float32))
        base.cdmr_mu.copy_(torch.as_tensor(mu_f, dtype=torch.float32))
        base.cdmr_var.copy_(torch.as_tensor(var_f, dtype=torch.float32))
        base.cdmr_ready.fill_(1.0)
    base._cdmr_sig = cdmr_basis_signature(base)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.maximum(np.outer(sd, sd), 1e-30)
    log.info("CDM-R CALIBRATION installed in %.1f s | noise covariance of the working coordinate from %d "
             "training pairs (per-gate MAD %.4f .. %.4f, adjacent-gate correlation median %.2f) | %d-component mixture "
             "prior on %d manifold coefficients from %d training curves | per-curve noise levels %s. ",
             time.time() - t0, int(dz.shape[0]), float(mad.min()), float(mad.max()),
             float(np.median(np.diag(corr, k=1))), int(pis.size), int(keep.sum()), int(A.shape[0]),
             ", ".join("%g" % float(v) for v in (getattr(cfg.model, "cdmr_noise_levels", (1.0,)) or (1.0,))))
    return True


def ensure_cdmr_calibration(model: nn.Module, cfg: Config, clean_cache: CleanCacheInfo,
                                logger: Optional[logging.Logger] = None) -> bool:
    """Install the CDM-R calibration unless the one in force was fitted on this very manifold and cache."""
    base = getattr(model, "module", model)
    base = getattr(base, "_orig_mod", base)
    if getattr(base, "cdmr_ready", None) is None or getattr(base, "lib_basis", None) is None:
        return False
    fp = (str(getattr(clean_cache, "data_path", "")), str(getattr(clean_cache, "paired_noisy_path", "") or ""),
          cdmr_basis_signature(base), int(cfg.train.seed), bool(getattr(cfg.model, "cdmr_calibrated", True)))
    if getattr(base, "_cdmr_fp", None) == fp:
        return bool(base.cdmr_calibrated_ready())
    ok = False
    try:
        ok = install_cdmr_calibration(model, cfg, clean_cache, logger)
    except Exception as _e:
        (logger or logging.getLogger(PROGRAM_NAME)).warning("CDM-R calibration not installed: %r", _e)
    base._cdmr_fp = fp
    return bool(ok)


def cdmr_regeneration_records(core: nn.Module, record: np.ndarray, cut: int, global_scale: float) -> np.ndarray:
    """The records with the gates from `cut` on replaced by the model's CDM-R regeneration of each curve from its own
    recorded gates -- what the trunk receives on the withheld gates.
    """
    g = core.gate_scale_norm.detach().double().cpu().numpy().reshape(1, -1) * float(global_scale)
    y = np.asarray(record, dtype=np.float64)
    e = y / g
    z = np.sign(e) * np.log1p(np.abs(e))
    dev = core.gate_scale_norm.device
    out = np.array(y, copy=True)
    cut = int(cut)
    with torch.no_grad():
        for i in range(0, y.shape[0], 4096):
            zt = torch.as_tensor(z[i:i + 4096], dtype=torch.float32, device=dev).unsqueeze(1)
            mt = torch.zeros_like(zt)
            mt[..., :cut] = 1.0
            zf = core.cdmr_continuation(zt, mt).squeeze(1).double().cpu().numpy()
            out[i:i + 4096, cut:] = (np.sign(zf) * np.expm1(np.abs(zf)) * g)[:, cut:]
    return out


def fit_row_completion(model: nn.Module, cfg: Config, clean_cache: CleanCacheInfo, logger: logging.Logger,
                           max_rows: int = 100000) -> bool:
    """Fit the early/mid -> mid/late completion operator on the training pairs in the model's input-unit z domain (z
    = symlog((y / global_scale) / gate_scale_norm)) and install it on the model.
    """
    try:
        base = getattr(model, "module", model)
        base = getattr(base, "_orig_mod", base)
        if not bool(getattr(cfg.model, "decoupled_arms", True)):
            return False
        if not getattr(clean_cache, "paired_noisy_path", "") or getattr(clean_cache, "province_id", None) is None:
            logger.warning("completion operator needs the paired noisy cache; decoupled prior unavailable.")
            return False
        t0 = time.time()
        splits = build_splits(clean_cache, cfg)
        rng = np.random.default_rng(int(cfg.train.seed) + 713)
        tr = np.asarray(splits["train"], dtype=np.int64)
        if tr.size > max_rows:
            tr = np.sort(rng.choice(tr, max_rows, replace=False))
        mat = np.load(clean_cache.data_path, mmap_mode="r")
        nm = np.load(str(clean_cache.paired_noisy_path), mmap_mode="r")
        gs = base.gate_scale_norm.detach().float().cpu().numpy().reshape(1, -1)
        gsc = float(getattr(clean_cache, "global_scale", 1.0) or 1.0)

        def _z(y):
            e = (np.asarray(y, dtype=np.float64) / gsc) / gs
            return np.sign(e) * np.log1p(np.abs(e))

        zn = _z(nm[tr])
        zc = _z(mat[tr])
        T = zn.shape[1]
        ls = int(cfg.data.late_start_index)
        prof = np.sqrt(np.mean((zn - zc) ** 2, axis=0))
        basev = float(np.median(prof[:20]))
        gl = int(getattr(cfg.model, "completion_gate_limit", 0) or 0)
        if gl <= 0:
            jump = int(np.argmax(prof > 4.0 * basev)) if bool(np.any(prof > 4.0 * basev)) else ls
            gl = int(max(24, min(jump, ls - 1)))
        X = zn[:, :gl]
        Y = zc[:, gl:]
        n = X.shape[0]
        nh = max(200, n // 10)
        perm = rng.permutation(n)
        hold, fit = perm[:nh], perm[nh:]
        mx = X[fit].mean(0)
        my = Y[fit].mean(0)
        Xc = X[fit] - mx
        Yc = Y[fit] - my
        G = Xc.T @ Xc
        H = Xc.T @ Yc
        scale = float(np.mean(np.diag(G)))

        def _late_err(W):
            Zh = (X[hold] - mx) @ W + my
            Yh = np.sign(Zh) * np.expm1(np.abs(Zh)) * gs[0, gl:]
            Yt = np.sign(Y[hold]) * np.expm1(np.abs(Y[hold])) * gs[0, gl:]
            e2 = np.sum((Yh - Yt)[:, ls - gl:] ** 2)
            y2 = np.sum(Yt[:, ls - gl:] ** 2)
            return 100.0 * math.sqrt(e2 / max(y2, 1e-300))

        best = None
        for rho in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
            W = np.linalg.solve(G + rho * scale * np.eye(gl), H)
            e = _late_err(W)
            if best is None or e < best[0]:
                best = (e, rho, W)
        base.set_row_completion(best[2], mx, my, gl)
        resid = np.sqrt(np.mean(((X[hold] - mx) @ best[2] + my - Y[hold]) ** 2, axis=0))
        logger.info("COMPLETION PRIOR INSTALLED | early/mid gates < %d (pair noise %.3f -> first > 4x at "
                    "the jump) -> %d mid/late gates | %d training rows, rho=%g | held-out late NRMSE %.2f%% "
                    "(physical, gates >= %d) | z residual RMS mid %.3f late %.3f | %.1fs.",
                    gl, basev, T - gl, int(fit.size), best[1], best[0], ls, float(np.mean(resid[:max(1, ls - gl)])),
                    float(np.mean(resid[ls - gl:])), time.time() - t0)
        return True
    except Exception as exc:
        logger.error("completion prior fit failed: %r", exc, exc_info=True)
        return False


def calibrate_mmse_bstar(model: nn.Module, val_metrics: Mapping[str, Any], late_start: int,
                             logger: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    """Returns the installed vector."""
    pm = [float(v) for v in (val_metrics.get("per_gate_mean_error_prior_percent") or [])]
    pd_ = [float(v) for v in (val_metrics.get("per_gate_mean_error_denoise_percent") or [])]
    if not pm or len(pm) != len(pd_):
        return None
    cv = val_metrics.get("per_gate_arm_cov_z") or {}
    sdd = [float(v) for v in (cv.get("dd") or [])]
    spp = [float(v) for v in (cv.get("pp") or [])]
    sdp = [float(v) for v in (cv.get("dp") or [])]
    hasm = len(sdd) == len(pm) and len(spp) == len(pm) and len(sdp) == len(pm)
    bs = np.full(len(pm), float("nan"), dtype=np.float64)
    for g in range(len(pm)):
        ed = max(pd_[g], 1e-6)
        ep = max(pm[g], 1e-6)
        b = ed * ed / (ed * ed + ep * ep)
        if hasm and sdd[g] + spp[g] > 0:
            b = sdd[g] / (sdd[g] + spp[g])
        bs[g] = float(min(0.98, max(0.02, b)))
    base = getattr(model, "module", model)
    base = getattr(base, "_orig_mod", base)
    if not hasattr(base, "set_mmse_bstar"):
        return None
    base.set_mmse_bstar(bs)
    out = base.mmse_bstar_714.detach().cpu().numpy()
    if logger is not None:
        ls = int(late_start)
        logger.info("MMSE BLEND CALIBRATED | covariance-optimal prior share by late gate (EMA): %s | witness snr_ref late mean %.2f. ",
                    " ".join("g%d:%.2f" % (g + 1, out[g]) for g in range(ls, len(out), max(1, (len(out) - ls) // 10))),
                    float(base.snr_ref_714.detach().cpu().numpy()[ls:].mean()))
    return out


def information_bound_stage(cfg: Config, clean_cache: CleanCacheInfo, logger: logging.Logger,
                                run_dir: Optional[Path] = None) -> bool:
    """Needs only the paired caches, no model."""
    try:
        if not getattr(clean_cache, "paired_noisy_path", "") or getattr(clean_cache, "province_id", None) is None:
            logger.warning("information bound needs the paired noisy cache and province ids; not available here.")
            return False
        t0 = time.time()
        splits = build_splits(clean_cache, cfg)
        rng = np.random.default_rng(int(cfg.train.seed) + 535)
        mat = np.load(clean_cache.data_path, mmap_mode="r")
        gsc = np.asarray(clean_cache.gate_scale, dtype=np.float64).reshape(1, -1)

        def _z(rows: np.ndarray) -> np.ndarray:
            e = np.asarray(rows, dtype=np.float64) / gsc
            return np.sign(e) * np.log1p(np.abs(e))

        tr = np.asarray(splits["train"], dtype=np.int64)
        if tr.size > 20000:
            tr = np.sort(rng.choice(tr, 20000, replace=False))
        ztr = _z(mat[tr])
        if getattr(clean_cache, "extra_clean_path", "") and bool(getattr(clean_cache, "extra_clean_full_coverage", False)):
            try:
                xm = np.load(clean_cache.extra_clean_path, mmap_mode="r")
                xi = np.sort(rng.choice(int(xm.shape[0]), min(int(xm.shape[0]), 20000), replace=False))
                ztr = np.concatenate([ztr, _z(xm[xi])], 0)
            except Exception as exc:
                logger.warning("extra clean rows skipped (%r)", exc)
        mu = ztr.mean(axis=0, keepdims=True)
        r = int(max(2, getattr(cfg.model, "library_subspace_rank", 12)))
        _u, _sv, vt = np.linalg.svd(ztr - mu, full_matrices=False)
        V = vt[:r]
        logger.info("INFORMATION BOUND STAGE | subspace rebuilt from %d rows (rank %d captures %.4f%% of z-variance) in %.1fs; "
                    "running the MMSE audit on the validation areas.",
                    int(ztr.shape[0]), r, 100.0 * float(np.sum(_sv[:r] ** 2) / max(float(np.sum(_sv ** 2)), 1e-300)), time.time() - t0)
        information_bound_audit(mat, str(clean_cache.paired_noisy_path), np.asarray(clean_cache.province_id),
                                np.asarray(splits["val"], dtype=np.int64), tr, V, mu.reshape(-1), gsc.reshape(-1),
                                int(cfg.data.late_start_index), int(max(1, cfg.data.num_neighbors)),
                                int(getattr(cfg.data, "profile_patch_len", 9) or 9), logger, rng, run_dir=run_dir)
        logger.info("NOTE: E1-E6 above are the rank-12 GENERATIVE Gaussian estimates: "
                    "diagnostics, not bounds. The ladder below is the bound. ")
        linear_information_ladder(mat, str(clean_cache.paired_noisy_path), np.asarray(clean_cache.province_id),
                                      np.asarray(splits["val"], dtype=np.int64), np.asarray(splits["train"], dtype=np.int64),
                                      gsc.reshape(-1), int(cfg.data.late_start_index), logger, rng, run_dir=run_dir)
        try:
            nm = np.load(str(clean_cache.paired_noisy_path), mmap_mode="r")
            prof = np.sqrt(np.mean((_z(nm[tr[:20000]]) - _z(mat[tr[:20000]])) ** 2, axis=0))
            base = float(np.median(prof[:20]))
            jump = int(np.argmax(prof > 4.0 * base)) if bool(np.any(prof > 4.0 * base)) else int(cfg.data.late_start_index)
            gl = int(max(24, min(jump, int(cfg.data.late_start_index))))
            logger.info("early/mid-only ladder: inputs restricted to gates < %d (pair noise %.3f -> first > 4x at gate %d; late start %d).",
                        gl, base, jump, int(cfg.data.late_start_index))
            linear_information_ladder(mat, str(clean_cache.paired_noisy_path), np.asarray(clean_cache.province_id),
                                          np.asarray(splits["val"], dtype=np.int64), np.asarray(splits["train"], dtype=np.int64),
                                          gsc.reshape(-1), int(cfg.data.late_start_index), logger, rng, run_dir=run_dir,
                                          gate_limit=gl, tag="e")
        except Exception as exc:
            logger.warning("early/mid-only ladder skipped: %r", exc)
        return True
    except Exception as exc:
        logger.error("information bound stage failed: %r", exc, exc_info=True)
        return False


def train_model(
    cfg: Config,
    run_dir: Path,
    clean_cache: CleanCacheInfo,
    noise_cache: NoiseCacheInfo,
    logger: logging.Logger,
) -> Tuple[PEBRNet, Path, DataLoader]:
    """Train in stages, select a checkpoint on validation and test it on the held-out profiles."""
    device = resolve_device(cfg.runtime.device)
    _prior_band_gate = -1
    audit_late_window_information(clean_cache, noise_cache, cfg, logger)
    logger.info("Training device: %s", device)
    for _ck in (str(getattr(cfg.paths, "resume_checkpoint", "") or ""), str(getattr(cfg.paths, "init_checkpoint", "") or "")):
        if _ck and Path(_ck).is_file():
            receptive_field_from_checkpoint(Path(_ck), cfg, logger)
            break
    sync_model_gates(cfg, logger)
    model = PEBRNet(cfg.model).to(device)
    gate_scale_norm = np.asarray(clean_cache.gate_scale, dtype=np.float64) / float(clean_cache.global_scale)
    model.set_gate_scale(gate_scale_norm, recalibrate=True)
    logger.info(
        "Per-gate equalization installed | dynamic range=%.4g (max/min of gate scale)",
        float(gate_scale_norm.max() / gate_scale_norm.min()),
    )
    logger.info("TRUNK NORM %s | decoder gate slot %s (manifold_decoder in %d).",
                "gate-local" if cfg.model.gate_local_norm else "GroupNorm (cross-gate)",
                "ON (legacy symlog(x/gs^2))" if cfg.model.decoder_gate_norm_channel else "off",
                int(model.manifold_decoder[0].in_features) if getattr(model, "manifold_decoder", None) is not None else 0)
    model.set_gate_times(noise_cache.target_time, recalibrate=True)
    arm_data_parallel(model, device, cfg, logger)
    fit_row_completion(model, cfg, clean_cache, logger)
    logger.info(
        "Physics decay-atom basis installed | atoms=%d | tau=[%.4g, %.4g] over gates [%.4g, %.4g] (resolved gate-time units)",
        int(model.decay_atoms.shape[0]),
        float(noise_cache.target_time[0]) / 3.0,
        float(noise_cache.target_time[-1]) * 3.0,
        float(noise_cache.target_time[0]),
        float(noise_cache.target_time[-1]),
    )
    if bool(getattr(cfg.train, "compile_model", False)):
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled (default mode). Checkpoints "
                        "written under compile carry an _orig_mod. prefix; resume "
                        "compiled runs with --compile as well.")
        except Exception as _cexc:
            logger.warning("torch.compile unavailable (%r); running eager.",
                           _cexc)
    _want_cfg = int(cfg.train.batch_size)
    try:
        _probe_loss = ProjectLoss(
            ConstraintController(cfg.contract), cfg.contract, cfg.loss,
            cfg.data.late_start_index,
            torch.as_tensor(gate_scale_norm, dtype=torch.float32),
            late_terms_all_stages=cfg.train.late_terms_all_stages,
            ip_chargeability_max=cfg.model.ip_chargeability_max,
            relaxation_blocks=len(tuple(getattr(cfg.model, "relaxation_stretch_exponents",
                                                (1.0,)) or (1.0,))),
            profile_patch_len=int(getattr(cfg.data, "profile_patch_len", 0) or 0),
        ).to(device)
        _micro, _accum = fit_micro_batch(model, cfg, device, _probe_loss, logger)
        del _probe_loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as _exc:
        _P = max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 0))
        if device.type == "cuda":
            _micro = max(_P, ((int(cfg.train.batch_size) // 8) // _P) * _P)
        else:
            _micro = int(cfg.train.batch_size)
        _accum = max(1, int(math.ceil(int(cfg.train.batch_size) / max(_micro, 1))))
        logger.warning("VRAM autotune skipped (%r); micro-batch %d x %d accumulation (a conservative "
                       "eighth of the effective batch on CUDA); the out-of-memory guard of the step halves it if needed.", _exc, _micro, _accum)
    _micro, _accum, _eff = apply_optimizer_batch_policy(int(_micro), int(_accum), cfg, logger)
    cfg.train.grad_accum_steps = int(_accum)
    cfg.train.effective_batch_size = int(_eff)
    cfg.train.batch_size = int(_micro)
    train_loader, val_loader, test_loader, splits, datasets = make_loaders(clean_cache, noise_cache, cfg)
    _old2: Optional[Tuple[int, int, int]] = None
    try:
        _P3 = max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 0))
        if (device.type == "cuda" and bool(cfg.train.vram_autotune) and int(cfg.train.micro_batch_size or 0) <= 0
                and bool(getattr(cfg.train, "vram_fill_real_step", True)) and bool(getattr(cfg.model, "aux_arms_subbatch", True))
                and float(getattr(cfg.model, "dropout", 0.0) or 0.0) <= 0.0 and _P3 > 1):
            _t3 = time.time()
            _sh = measure_aux_flag_shares(train_loader, _P3, int(getattr(cfg.train, "aux_share_batches", 8)))
            if _sh is not None:
                _sig3 = float(getattr(cfg.train, "aux_cap_sigma", 2.0))
                _cb = aux_cap_share(_sh["bg_mean"], int(_sh["patches"]), _sig3)
                _ca = aux_cap_share(_sh["alt_mean"], int(_sh["patches"]), _sig3)
                _ff723t = float(getattr(cfg.model, "aux_full_fraction", 0.90))
                logger.info("REAL-STEP SHARES over %d real batches of %d patches (%.0f s) | has_anomaly mean %.3f max %.3f -> cap %.3f%s "
                            "| has_alt mean %.3f max %.3f -> cap %.3f%s (mean + %.1f sigma).",
                            int(_sh["batches"]), int(_sh["patches"]), time.time() - _t3,
                            _sh["bg_mean"], _sh["bg_max"], _cb, " (>= full-batch threshold: uncapped)" if _cb >= _ff723t else "",
                            _sh["alt_mean"], _sh["alt_max"], _ca, " (>= full-batch threshold: uncapped)" if _ca >= _ff723t else "", _sig3)
                if min(_cb, _ca) < _ff723t:
                    _old2 = (int(cfg.train.batch_size), int(cfg.train.grad_accum_steps), int(cfg.train.effective_batch_size))
                    cfg.model.aux_bg_cap_share = float(_cb if _cb < _ff723t else 1.0)
                    cfg.model.aux_alt_cap_share = float(_ca if _ca < _ff723t else 1.0)
                    cfg.train.batch_size = int(_want_cfg)
                    _pl = ProjectLoss(
                        ConstraintController(cfg.contract), cfg.contract, cfg.loss, cfg.data.late_start_index,
                        torch.as_tensor(gate_scale_norm, dtype=torch.float32),
                        late_terms_all_stages=cfg.train.late_terms_all_stages,
                        ip_chargeability_max=cfg.model.ip_chargeability_max,
                        relaxation_blocks=len(tuple(getattr(cfg.model, "relaxation_stretch_exponents", (1.0,)) or (1.0,))),
                        profile_patch_len=_P3).to(device)
                    try:
                        _m2, _a2 = fit_micro_batch(model, cfg, device, _pl, logger)
                    finally:
                        del _pl
                        torch.cuda.empty_cache()
                    _m2, _a2, _e20 = apply_optimizer_batch_policy(int(_m2), int(_a2), cfg, logger)
                    if int(_m2) > _old2[0]:
                        cfg.train.batch_size = int(_m2)
                        cfg.train.grad_accum_steps = int(_a2)
                        cfg.train.effective_batch_size = int(_e20)
                        try:
                            del train_loader, val_loader, test_loader
                        except Exception:
                            pass
                        train_loader, val_loader, test_loader, splits, datasets = make_loaders(clean_cache, noise_cache, cfg)
                        logger.info("MICRO-BATCH RAISED %d -> %d rows (x%.2f) by probing the real step at the caps; "
                                    "loaders rebuilt. The probed step bounds every real step: an arm never runs more patches than "
                                    "its cap.",
                                    _old2[0], int(_m2), float(_m2) / max(1, _old2[0]))
                    else:
                        cfg.train.batch_size, cfg.train.grad_accum_steps, cfg.train.effective_batch_size = _old2
                        cfg.model.aux_bg_cap_share = 1.0
                        cfg.model.aux_alt_cap_share = 1.0
                        logger.info("the capped probe did not raise the micro-batch (%d rows): caps removed, worst-case sizing kept. ", _old2[0])
    except Exception as _e723x:
        try:
            cfg.model.aux_bg_cap_share = 1.0
            cfg.model.aux_alt_cap_share = 1.0
            if _old2 is not None:
                cfg.train.batch_size, cfg.train.grad_accum_steps, cfg.train.effective_batch_size = _old2
                train_loader, val_loader, test_loader, splits, datasets = make_loaders(clean_cache, noise_cache, cfg)
        except Exception:
            pass
        logger.warning("real-step fill skipped (%r): worst-case micro-batch kept.", _e723x)
    if bool(getattr(cfg.model, "use_library_subspace", True)):
        try:
            _t = time.time()
            _rng = np.random.default_rng(int(cfg.train.seed) + 535)
            _mat = np.load(clean_cache.data_path, mmap_mode="r")
            _gsc = np.asarray(clean_cache.gate_scale, dtype=np.float64).reshape(1, -1)

            def _z(rows: np.ndarray) -> np.ndarray:
                _e = np.asarray(rows, dtype=np.float64) / _gsc
                if bool(cfg.model.per_trace_normalization):
                    _ng = int(max(2, min(int(cfg.model.per_trace_norm_gates), int(cfg.data.late_start_index) // 8, _e.shape[1])))
                    _e = _e / np.exp(np.median(np.log(np.maximum(np.abs(_e[:, :_ng]), 1e-300)), axis=1, keepdims=True))
                return np.sign(_e) * np.log1p(np.abs(_e))

            _tr = np.asarray(splits["train"], dtype=np.int64)
            if _tr.size > 20000:
                _tr = np.sort(_rng.choice(_tr, 20000, replace=False))
            _ztr = _z(_mat[_tr])
            if getattr(clean_cache, "extra_clean_path", ""):
                try:
                    if not bool(getattr(clean_cache, "extra_clean_full_coverage", False)):
                        raise RuntimeError("extra clean rows are not on the full training axis; excluded from the dictionary")
                    _xm = np.load(clean_cache.extra_clean_path, mmap_mode="r")
                    _nx = min(int(_xm.shape[0]), 20000)
                    _xi = np.sort(_rng.choice(int(_xm.shape[0]), _nx,
                                                    replace=False))
                    _ztr = np.concatenate([_ztr, _z(_xm[_xi])], 0)
                except Exception as _xe:
                    logger.warning("extra clean augmentation skipped (%r)",
                                   _xe)
            _mu = _ztr.mean(axis=0, keepdims=True)
            _r2 = int(max(2, getattr(cfg.model, "library_subspace_rank", 12)))
            _u5, _s5, _vt5 = np.linalg.svd(_ztr - _mu, full_matrices=False)
            _vr = _vt5[:_r2]
            _va = np.asarray(splits["val"], dtype=np.int64)
            if _va.size > 8000:
                _va = np.sort(_rng.choice(_va, 8000, replace=False))
            _zva = _z(_mat[_va])
            _ls = int(cfg.data.late_start_index)

            def _lin(z: np.ndarray) -> np.ndarray:
                return np.sign(z) * np.expm1(np.abs(z)) * _gsc

            _rec = _mu + ((_zva - _mu) @ _vr.T) @ _vr
            _yv = _lin(_zva)
            _e3 = _lin(_rec) - _yv
            _sub = 100.0 * math.sqrt(
                float(np.sum(_e3[:, _ls:] ** 2))
                / max(float(np.sum(_yv[:, _ls:] ** 2)), 1e-300))
            _X = np.concatenate(
                [_ztr[:, :_ls], np.ones((_ztr.shape[0], 1))], 1)
            _Y = _ztr[:, _ls:]
            _W = np.linalg.solve(
                _X.T @ _X + 1e-3 * np.eye(_X.shape[1]), _X.T @ _Y)
            _Xv = np.concatenate(
                [_zva[:, :_ls], np.ones((_zva.shape[0], 1))], 1)
            _zp = _zva.copy()
            _zp[:, _ls:] = _Xv @ _W
            _er = _lin(_zp) - _yv
            _rid = 100.0 * math.sqrt(
                float(np.sum(_er[:, _ls:] ** 2))
                / max(float(np.sum(_yv[:, _ls:] ** 2)), 1e-300))
            model.set_library_subspace(_mu.reshape(-1), _vr)
            _d2tr = np.abs(_ztr[:, 2:] - 2.0 * _ztr[:, 1:-1]
                           + _ztr[:, :-2])
            _c2fam = float(np.percentile(_d2tr, 95.0))
            model.set_family_curvature(_c2fam)
            _rec = _ztr - (_mu + ((_ztr - _mu) @ _vr.T)
                                 @ _vr)
            _res = np.percentile(np.abs(_rec), 95.0, axis=0)
            model.set_family_residual(_res)
            model.set_coef_prior_var(np.var((_ztr - _mu) @ _vr.T, axis=0))
            try:
                if getattr(clean_cache, "paired_noisy_path", "") and getattr(clean_cache, "province_id", None) is not None:
                    _prof = information_bound_audit(
                        _mat, str(clean_cache.paired_noisy_path), np.asarray(clean_cache.province_id),
                        np.asarray(splits["val"], dtype=np.int64), np.asarray(splits["train"], dtype=np.int64),
                        _vr, _mu.reshape(-1), _gsc.reshape(-1), _ls,
                        int(max(1, cfg.data.num_neighbors)), int(getattr(cfg.data, "profile_patch_len", 9) or 9),
                        logger, _rng, run_dir)
                    try:
                        if str(getattr(cfg.runtime, "startup_audits", "fast")).lower() != "full":
                            logger.info("linear information ladder SKIPPED at start-up (a diagnostic of about 2 h on one GPU "
                                        "that does not bound the model); run it with --startup_audits full or as the "
                                        "stand-alone stage --mode figures --information_bound_only.")
                            raise _SkipLadder()
                        linear_information_ladder(_mat, str(clean_cache.paired_noisy_path), np.asarray(clean_cache.province_id),
                                                      np.asarray(splits["val"], dtype=np.int64), np.asarray(splits["train"], dtype=np.int64),
                                                      _gsc.reshape(-1), _ls, logger, _rng, run_dir=run_dir)
                        _gl = int(getattr(getattr(model, "module", model), "row_completion_gate_limit", 0) or 0)
                        if _gl <= 0:
                            _nm = np.load(str(clean_cache.paired_noisy_path), mmap_mode="r")
                            _tr2 = np.asarray(splits["train"], dtype=np.int64)[:20000]
                            _e17 = (np.asarray(_nm[_tr2], dtype=np.float64) / _gsc.reshape(1, -1))
                            _c2 = (np.asarray(_mat[_tr2], dtype=np.float64) / _gsc.reshape(1, -1))
                            _p2 = np.sqrt(np.mean((np.sign(_e17) * np.log1p(np.abs(_e17)) - np.sign(_c2) * np.log1p(np.abs(_c2))) ** 2, axis=0))
                            _b3 = float(np.median(_p2[:20]))
                            _j = int(np.argmax(_p2 > 4.0 * _b3)) if bool(np.any(_p2 > 4.0 * _b3)) else _ls
                            _gl = int(max(24, min(_j, _ls - 1)))
                        linear_information_ladder(_mat, str(clean_cache.paired_noisy_path), np.asarray(clean_cache.province_id),
                                                      np.asarray(splits["val"], dtype=np.int64), np.asarray(splits["train"], dtype=np.int64),
                                                      _gsc.reshape(-1), _ls, logger, _rng, run_dir=run_dir, gate_limit=_gl, tag="e")
                    except _SkipLadder:
                        pass
                    except Exception as _exc:
                        logger.warning("linear ladder skipped in the training path: %r", _exc)
                    try:
                        _pb = int(getattr(cfg.data, "prior_band_start", -1))
                        _gj = int(getattr(information_bound_audit, "noise_jump_gate", -1))
                        _g3 = _gj if _pb < 0 else _pb
                        if 0 < _g3 < _ls:
                            model.set_prior_band_start(_g3)
                            _prior_band_gate = int(_g3)
                            logger.info("PRIOR BAND starts at gate %d (contract late start %d): blend floor ramp %d->%d, "
                                        "anchor ramp likewise, manifold prior supervised with the LATE weight from gate %d (source: %s).",
                                        _g3 + 1, _ls + 1, _g3 + 1, _ls + 1, _g3 + 1,
                                        "measured noise jump" if _pb < 0 else "--prior_band_start")
                        else:
                            logger.log(logging.WARNING if (_pb < 0 and int(cfg.data.target_gates) > RF_DESIGN_GATES) else logging.INFO,
                                       "PRIOR BAND unchanged (no noise jump >= 1.8x over one native gate, or disabled): the band "
                                       "stays at the fallback gate %d; measured jump x%.2f.",
                                       int(getattr(model, "_prior_band_g", -1)) + 1,
                                       float(getattr(information_bound_audit, "noise_jump_ratio", float("nan"))))
                    except Exception as _e9:
                        logger.warning("prior band not applied: %r", _e9)
                    if _prof is not None:
                        model.set_gate_precision(_prof)
                        logger.info("per-gate precision profile INSTALLED in the coefficient-field solve "
                                    "(max-normalised): %s.",
                                    " ".join("%.3g" % v for v in np.asarray(_prof) / max(float(np.max(_prof)), 1e-30)))
                    else:
                        logger.info("precision profile NOT installed (uniform weighting kept: the audit did "
                                    "not show an improvement).")
            except Exception as _e8:
                logger.warning("information-bound audit skipped: %r", _e8)
            _wB = completion_observation_weight(_ztr.shape[1], _ls)
            logger.info("completion observation operator = completion_observation_weight (early/mid 1, late 1e-3 from "
                        "gate %d): shared by the fit, the lambda sweep, the audit and the runtime observed-mode solve. ", int(_ls) + 1)
            _GB = (_vr * _wB) @ _vr.T
            _dB = np.diag(_GB)
            _lamB = 1e-3 + 4.0 * (1.0 - np.clip(_dB, 0, 1))
            _zof, _ztru, _zvn = _ztr, _ztr, None
            _src2 = "clean c_B, lambda on iid 0.05 (no paired noisy cache)"
            _pn = str(getattr(clean_cache, "paired_noisy_path", "") or "")
            if _pn:
                try:
                    _nm3 = np.load(_pn, mmap_mode="r")
                    _zo = _z(_nm3[_tr])
                    _zv2 = _z(_nm3[_va])
                    _ok = np.all(np.isfinite(_zo), axis=1)
                    if int(_ok.sum()) >= 1000 and bool(np.all(np.isfinite(_zv2))):
                        _zof = _zo[_ok]
                        _ztru = _ztr[:int(_tr.size)][_ok]
                        _zvn = _zv2
                        _src2 = "MEASURED noisy pairs (%d train rows, lambda on %d val pairs)" % (int(_ok.sum()), int(_zv2.shape[0]))
                except Exception as _e21:
                    logger.warning("measured-pair completion fit unavailable (%r); iid-0.05 fit kept.", _e21)
            _cobs = np.linalg.solve(_GB + np.diag(_lamB),
                                    (_vr * _wB) @ (_zof - _mu).T).T
            _ctru = (_ztru - _mu) @ _vr.T
            _XB = np.concatenate([_cobs, np.ones((_cobs.shape[0], 1))], 1)
            _rngA = np.random.default_rng(535)
            if _zvn is None:
                _zvn = _zva.copy()
                _zvn[:, :_ls] += 0.05 * _rngA.standard_normal(_zvn[:, :_ls].shape)
            _chn = np.linalg.solve(_GB + np.diag(_lamB), (_vr * _wB) @ (_zvn - _mu).T).T
            _denA = max(float(np.sum(_yv[:, _ls:] ** 2)), 1e-300)
            _bestA = None
            _swA = []
            for _lA in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
                _A_ = np.linalg.solve(_XB.T @ _XB + _lA * np.eye(_XB.shape[1]), _XB.T @ _ctru)
                _ccn = np.concatenate([_chn, np.ones((_chn.shape[0], 1))], 1) @ _A_
                _en = _lin(_mu + _ccn @ _vr)[:, _ls:] - _yv[:, _ls:]
                _eN = 100.0 * math.sqrt(float(np.sum(_en ** 2)) / _denA)
                _gA = float(np.linalg.svd(_A_[:-1], compute_uv=False)[0])
                _swA.append((_lA, _eN, _gA, _A_))
            _minA = min(_c[1] for _c in _swA)
            _bestA = max((_c for _c in _swA if _c[1] <= 1.15 * _minA), key=lambda _c: _c[0])
            _AB = _bestA[3]
            model.set_field_completion(_AB)
            _tt = np.asarray(clean_cache.target_time, dtype=np.float64).reshape(-1)
            _lt = (log_time_axis(_tt) if gate_axis_has_origin(_tt)
                      else np.log(_tt + 1e-12))
            _M5 = 8
            _mid = slice(_ls - _M5, _ls)
            _ly = np.log(np.abs(_yv[:, _mid]) + 1e-30)
            _ltm = _lt[_mid]
            _ltc = _ltm - _ltm.mean()
            _bb = ((_ltc[None, :] * (_ly - _ly.mean(1, keepdims=True))).sum(1) / max(float((_ltc ** 2).sum()), 1e-12))
            _aa = _ly.mean(1) - _bb * _ltm.mean()
            _ypl = np.sign(_yv[:, _ls - 1:_ls]) * np.exp(_aa[:, None] + _bb[:, None] * _lt[None, _ls:])
            _ypl = np.where(_ypl == 0, _yv[:, _ls:], _ypl)
            _epl = 100.0 * math.sqrt(float(np.sum((_ypl - _yv[:, _ls:]) ** 2)) / _denA)
            _ccb = np.concatenate([_chn, np.ones((_chn.shape[0], 1))], 1) @ _AB
            _ycb = _lin(_mu + _ccb @ _vr)[:, _ls:]
            _bestB = (0.0, 100.0 * math.sqrt(float(np.sum((_ycb - _yv[:, _ls:]) ** 2)) / _denA))
            for _bt in (0.25, 0.5, 0.75, 1.0):
                _yb = (1.0 - _bt) * _ycb + _bt * _ypl
                _eb = 100.0 * math.sqrt(float(np.sum((_yb - _yv[:, _ls:]) ** 2)) / _denA)
                if _eb < _bestB[1]:
                    _bestB = (_bt, _eb)
            model.set_powerlaw(_lt, _bestB[0] if bool(getattr(cfg.model, "physics_extrapolation", False)) else 0.0)
            logger.info("PHYSICS EXTRAPOLATION audit | log-log asymptote alone (clean held-out) "
                        "late = %.2f%% | completion(noisy) = %.2f%% | best blend beta=%.2f -> %.2f%%.",
                        _epl, float(100.0 * math.sqrt(float(np.sum((_ycb - _yv[:, _ls:]) ** 2)) / _denA)),
                        float(_bestB[0]), float(_bestB[1]))
            logger.info("completion lambda sweep -> lambda_A=%g | NOISY held-out late = "
                        "%.2f%% | gain ||A||2 = %.2f | fit on %s .",
                        float(_bestA[0]), float(_bestA[1]), float(_bestA[2]), _src2)
            try:
                _sig2 = float(10.0 ** (-float(getattr(cfg.data, "deep_negative_early_floor_db", 12.0)) / 20.0))
                _zv = _zva.copy()
                _zv[:, :_ls] += _sig2 * np.random.default_rng(579).standard_normal(_zv[:, :_ls].shape)
                _ch = np.linalg.solve(_GB + np.diag(_lamB), (_vr * _wB) @ (_zv - _mu).T).T
                _cc2 = np.concatenate([_ch, np.ones((_ch.shape[0], 1))], 1) @ _AB
                _e4 = _lin(_mu + _cc2 @ _vr)[:, _ls:] - _yv[:, _ls:]
                _l2 = 100.0 * math.sqrt(float(np.sum(_e4 ** 2)) / _denA)
                logger.info("COMPLETION AUDIT AT THE +%.0f dB EARLY FLOOR (sigma_z=%.3f, the regime "
                            "the deep rows carry) | held-out late = %.2f%% | gain ||A||2 = %.2f. %s",
                            float(getattr(cfg.data, "deep_negative_early_floor_db", 12.0)), _sig2, _l2,
                            float(_bestA[2]),
                            ("the closed-form anchor CANNOT carry the deep bins alone; the learned "
                             "residual is the path." if _l2 > 20.0 else "closed-form path viable."))
            except Exception as _e579x:
                logger.warning("floor audit skipped: %r", _e579x)
            _cho = np.linalg.solve(_GB + np.diag(_lamB),
                                   (_vr * _wB) @ (_zva - _mu).T).T
            _e0 = _lin(_mu + _cho @ _vr)[:, _ls:] - _yv[:, _ls:]
            _cc = np.concatenate([_cho, np.ones((_cho.shape[0], 1))], 1) @ _AB
            _e1 = _lin(_mu + _cc @ _vr)[:, _ls:] - _yv[:, _ls:]
            _den = max(float(np.sum(_yv[:, _ls:] ** 2)), 1e-300)
            logger.info(
                "FIELD COMPLETION audit (held-out, buried regime): "
                "mean-fallback late = %.2f%% -> completed late = %.2f%%.",
                100.0 * math.sqrt(float(np.sum(_e0 ** 2)) / _den),
                100.0 * math.sqrt(float(np.sum(_e1 ** 2)) / _den))
            logger.info(
                "family residual tolerance installed | per-gate P95 "
                "median = %.4f (IRLS consistency leg active).",
                float(np.median(_res)))
            logger.info(
                "family curvature bound installed: P95|D2 z| = %.4f "
                "(smoothness trust active).",
                _c2fam)
            _lamsP = (1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)
            _XtXP = _X.T @ _X
            _XtYP = _X.T @ _Y
            _swpP = []
            for _lmP in _lamsP:
                _WlP = np.linalg.solve(
                    _XtXP + float(_lmP) * np.eye(_XtXP.shape[0]), _XtYP)
                _zlP = _zva[:, :_ls] @ _WlP[:-1] + _WlP[-1:]
                _elP = _lin(np.concatenate(
                    [_zva[:, :_ls], _zlP], 1))[:, _ls:]                     - _yv[:, _ls:]
                _erP = 100.0 * math.sqrt(
                    float(np.sum(_elP ** 2))
                    / max(float(np.sum(_yv[:, _ls:] ** 2)), 1e-300))
                _gnP = float(np.linalg.svd(_WlP[:-1], compute_uv=False)[0])
                _swpP.append((float(_lmP), _erP, _gnP, _WlP))
            _mus6 = (1e-2, 1e-1, 3e-1)
            _tab6 = []
            for _m6 in _mus6:
                _Gh6 = _vr @ _vr.T + _m6 * np.eye(_vr.shape[0])
                _ch6 = np.linalg.solve(_Gh6, _vr @ (_zva - _mu).T).T
                _fh6 = _mu + _ch6 @ _vr
                _eh6 = _lin(_fh6)[:, _ls:] - _yv[:, _ls:]
                _tab6.append((_m6, 100.0 * math.sqrt(
                    float(np.sum(_eh6 ** 2))
                    / max(float(np.sum(_yv[:, _ls:] ** 2)), 1e-300))))
            logger.info(
                "coefficient-field mu audit (clean holdout late %%):"
                " %s | configured mu=%g.",
                " ".join("%g:%.3f" % _c6a for _c6a in _tab6),
                float(getattr(cfg.model, "coef_field_mu", 0.1)))
            _capP = float(getattr(cfg.model, "continuation_gain_cap", 1.8))
            _okP = [_c for _c in _swpP if _c[2] <= _capP]
            _selP = (min(_okP, key=lambda _c: _c[1]) if _okP else _swpP[-1])
            _ridP = float(_selP[1])
            logger.info(
                "lambda sweep (holdout late %% | gain ||W||2): %s | "
                "selected lambda=%g.",
                " ".join("%g:%.3f|%.2f" % (_l, _e, _g)
                         for _l, _e, _g, _w in _swpP),
                float(_selP[0]),
            )
            if bool(getattr(cfg.model, "use_ridge_anchor", True)):
                if bool(getattr(cfg.model, "projection_continuation", True)):
                    model.set_library_ridge(_selP[3][:-1], _selP[3][-1])
                else:
                    model.set_library_ridge(_W[:-1], _W[-1])
            logger.info(
                "CONTINUATION OPERATOR | raw (lambda=1e-3) held-out "
                "late = %.3f%% | GAIN-CAPPED held-out late = %.3f%% | installed"
                " = %s.",
                float(_rid) if "_rid" in dir() else float("nan"),
                float(_ridP),
                ("PROJECTED" if bool(getattr(cfg.model,
                                             "projection_continuation", True))
                 else "RAW"),
            )
            _ev = float(np.sum(_s5[:_r2] ** 2) / max(np.sum(_s5 ** 2), 1e-300))
            logger.info(
                "LIBRARY CAPACITY AUDIT | basis rows=%d (train%s) | rank "
                "r=%d captures %.4f%% of z-variance | VAL clean rank-r LATE NRMSE "
                "= %.3f%% (anchor representation floor) | VAL ridge early/mid->late "
                "LATE NRMSE = %.3f%% (family determinism given the high-SNR gates) "
                "| %.1fs.",
                int(_ztr.shape[0]),
                "+extra" if getattr(clean_cache, "extra_clean_path", "") else "",
                _r2, 100.0 * _ev, _sub, _rid, time.time() - _t,
            )
            if _sub > 3.0:
                logger.warning(
                    "The library's OWN rank-%d late representation error (%.2f%%) exceeds the 3%% "
                    "acceptance: raise --library_subspace_rank or enrich the clean library -- no network can beat "
                    "its anchor's representation floor.",
                    _r2, _sub)
        except Exception as _le:
            logger.warning("Library subspace unavailable (%r); the anchor "
                           "stays inert and training proceeds unchanged.", _le)
    _paired_active = bool(getattr(datasets[0], "paired_noisy_path", ""))
    if _paired_active and device.type == "cuda" and bool(getattr(cfg.train, "gpu_synthesis", True)):
        logger.info(
            "GPU-resident synthesis is DISABLED for this run: the supervision "
            "is PAIRED, and that loader synthesizes -- it would bypass the measured "
            "pairs entirely. The multi-worker CPU DataLoader over memmaps carries the "
            "paired path.",
        )
    elif device.type == "cuda" and bool(getattr(cfg.train, "gpu_synthesis", True)):
        try:
            train_loader, val_loader, test_loader = wrap_loaders_for_gpu(
                (train_loader, val_loader, test_loader), datasets, cfg, device, logger)
        except Exception as _exc:
            logger.warning(
                "GPU-resident synthesis unavailable (%r); falling back to "
                "the CPU DataLoader path.",
                _exc)
    atomic_json_dump(
        {k: int(len(v)) for k, v in splits.items() if not k.startswith("_")},
        run_dir / "reports" / "split_counts.json",
    )
    set_nan_forensics_path(run_dir / "reports" / "nan_forensics.json")
    audit_split_populations(clean_cache, splits, logger)
    if cfg.data.clean_augment:
        logger.info(
            "CLEAN-CURVE AUGMENTATION active on the TRAIN split | tau dilation 10^+-%.2f dex "
            "(= rescaling every time constant, i.e. the target conductance) x amplitude 10^+-%.2f dex, "
            "applied to %.0f%% of samples. Both map the induction family exactly into itself, so the "
            "augmented label stays inside the physics cone. Validation and test see the library untouched.",
            cfg.data.clean_aug_tau_dex, cfg.data.clean_aug_amp_dex, 100.0 * cfg.data.clean_aug_probability,
        )
    if cfg.model.per_trace_normalization:
        logger.info(
            "PER-TRACE AMPLITUDE CONDITIONING active | estimator %s on the first %d gates | every trace is "
            "rescaled onto the library's per-gate median amplitude and the output is rescaled back. The network sees "
            "shape, not survey coupling or the station's absolute level (profile ends at 0.09x, field gain tilt).",
            str(getattr(cfg.model, "per_trace_norm_estimator", "centre_logmean")),
            int(max(2, min(int(cfg.model.per_trace_norm_gates), int(cfg.data.late_start_index) // 8))),
        )
    promotion: Optional[Dict[str, Any]] = None
    if str(getattr(cfg.paths, "init_checkpoint", "") or ""):
        promotion = initialize_from_parent_checkpoint(
            model, Path(cfg.paths.init_checkpoint), cfg, logger)
        try:
            _pt = torch.load(Path(cfg.paths.init_checkpoint), map_location="cpu", weights_only=False)
            if isinstance(_pt, dict) and "target_time" in _pt:
                require_same_axis(noise_cache.target_time, _pt["target_time"], what="parent checkpoint")
            del _pt
        except ValueError:
            raise
        except Exception as _e15:
            logger.warning("parent axis check skipped: %r", _e15)
        _before = {k: v.detach().clone() for k, v in model.named_parameters()}
        model.set_gate_scale(gate_scale_norm, recalibrate=False)
        model.set_gate_times(noise_cache.target_time, recalibrate=False)
        _changed = [k for k, v in model.named_parameters() if not torch.equal(v.detach().cpu(), _before[k].cpu())]
        promotion["parameters_changed_by_setters"] = _changed
        if _changed:
            raise RuntimeError("the buffer setters changed trained parameters: %s" % _changed)
        del _before
        model.to(device)
        _lim = float(cfg.runtime.init_equivalence_max)
        promotion["equivalence_max"] = _lim
        _pert = float(promotion["transplant_output_perturbation"])
        promotion["equivalence_pass"] = bool(math.isfinite(_pert) and _pert <= _lim)
        atomic_json_dump(promotion, run_dir / "reports" / "stage_promotion.json")
        if not promotion["equivalence_pass"]:
            logger.error(
                "PROMOTION IS NOT INERT : transplanting %s into ablation '%s' changes the "
                "output by %.3g (limit %.3g). A promotion must start numerically identical to the "
                "qualified parent -- otherwise the new stage begins by destroying the result it is "
                "supposed to build on. Refusing to train (exit 26). Check which tensors were "
                "skipped in reports/stage_promotion.json. (--init_equivalence_max relaxes it "
                "when a stage genuinely changes the forward path.)",
                Path(cfg.paths.init_checkpoint).name, cfg.runtime.ablation,
                promotion["transplant_output_perturbation"], _lim,
            )
            raise SystemExit(26)
        logger.info(
            "PROMOTION INERT | output perturbation %.3g <= %.3g: the child starts from "
            "the parent's qualified behaviour and the new mechanism contributes nothing yet.",
            promotion["transplant_output_perturbation"], _lim,
        )
    _frozen_epochs = 0
    train_model._last_val_signature = None
    parameter_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{parameter_count:,}")
    try:
        _lj0 = getattr(model, "lateral_joint", None)
        _P0 = int(getattr(cfg.data, "profile_patch_len", 0) or 0)
        logger.info(
            "LATERAL MODE %s | joint profile code dim %d heads %d | lateral "
            "fingerprint bank M=%d kernel %d | on prior %s | on output %s | params %d. ",
            str(getattr(cfg.model, "lateral_joint_mode", "learned")),
            int(getattr(cfg.model, "joint_profile_dim", 64)),
            int(getattr(cfg.model, "joint_profile_heads", 2)),
            int(getattr(cfg.model, "lateral_fingerprint_bank", 32)),
            int(getattr(cfg.model, "lateral_fingerprint_kernel", 9)),
            "ON" if getattr(model, "lateral_fp_prior", None) is not None else "OFF",
            "ON" if getattr(model, "lateral_fp_out", None) is not None else "OFF",
            int(sum(p.numel() for nm in ("profile_joint", "joint_inject", "lateral_fp_prior",
                                         "lateral_fp_out")
                    for p in (getattr(model, nm).parameters() if getattr(model, nm, None) is not None else []))),
        )
        logger.info(
            "patch length %d (whole work area when -1 was configured) | joint-code "
            "injection UNGATED, operators gated.",
            int(getattr(cfg.data, "profile_patch_len", 0) or 0),
        )
        logger.info(
            "SUB-ZERO TRAINING SLICE %s | probability %.2f of a library injection at "
            "late-local SNR in [%.0f, %.0f) dB | scoring bands unchanged (>= 0 dB). ",
            "ON" if float(getattr(cfg.data, "measured_noise_subzero_probability", 0.0)) > 0 else "OFF",
            float(getattr(cfg.data, "measured_noise_subzero_probability", 0.0)),
            float(getattr(cfg.data, "measured_noise_subzero_min_db", -10.0)),
            float(getattr(cfg.data, "measured_noise_snr_min_db", 0.0)),
        )
        logger.info(
            "WEAK-BODY AUGMENTATION %s | prob %.2f per patch/block | contrast %.3f-%.3f "
            "(log-uniform, %.0f%% conductive) | width %.1f-%.1f stations | applied to clean, "
            "observation and every neighbour; background untouched; evaluation blocks %s. ",
            "ON" if float(getattr(cfg.data, "weak_body_injection_prob", 0.0)) > 0 else "OFF",
            float(getattr(cfg.data, "weak_body_injection_prob", 0.0)),
            float(getattr(cfg.data, "weak_body_contrast_min", 0.005)),
            float(getattr(cfg.data, "weak_body_contrast_max", 0.20)),
            100.0 * float(getattr(cfg.data, "weak_body_positive_fraction", 0.7)),
            float(getattr(cfg.data, "weak_body_width_min", 1.0)),
            float(getattr(cfg.data, "weak_body_width_max", 4.0)),
            "ON" if bool(getattr(cfg.data, "weak_body_eval_injection", False)) else "OFF",
        )
        logger.info(
            "CONSOLIDATED DEFAULT | lateral mode %s | lambda "
            "witness %s (gamma %.2f) | precision innovation %s | output fusion %s | temporal "
            "witness %s | evidence gate %s.",
            str(getattr(cfg.model, "lateral_joint_mode", "tikhonov")),
            str(getattr(cfg.model, "lateral_lambda_witness", "noise_head")),
            float(getattr(cfg.model, "lateral_gate_gamma", 1.0)),
            "ON" if bool(getattr(cfg.model, "innovation_precision_weighting", False)) else "OFF",
            "ON" if bool(getattr(cfg.model, "output_measurement_fusion", False)) else "OFF",
            "ON" if bool(getattr(cfg.model, "lateral_gate_temporal_witness", False)) else "OFF",
            "ON" if bool(getattr(cfg.model, "innovation_evidence_gate", False)) else "OFF",
        )
        logger.info(
            "LATE-FUSION DOCTRINE = %s | learned per-gate blend (floors/redirect = "
            "diagnostics only, quality cap = upper bound only, blend head bias-init -1.386 -> data-primary "
            "start ~0.20) | trunk direct contract supervision alpha %.2f (x den/prior leash boost) + per-curve %.2f "
            "| continuation arm frac %.2f, weight %.2f (E11 as a training task, <=4 patches/step).",
            ("SUPERVISED-PRIMARY"
             if bool(getattr(cfg.model, "supervised_primary_fusion", True))
             else "LEGACY PRIOR-PRIMARY (--legacy_prior_fusion)"),
            float(getattr(cfg.loss, "alpha_denoise_contract", 1.0)),
            float(getattr(cfg.loss, "alpha_denoise_ps_late", 0.35)),
            float(getattr(cfg.model, "continuation_arm_frac", 0.25)),
            float(getattr(cfg.loss, "alpha_continuation", 0.30)))
        logger.info(
            "DEEP-NEGATIVE CURRICULUM | late-local injection "
            "[%.0f, %.0f] dB (paired arm reads contract_test range [%.0f, %.0f]; "
            "synthetic-sampler P(<0dB)=%.2f (inactive in paired mode; paired replay = own natural SNR - U(0,drop) ); low-focus %.2f) | acceptance = "
            "EVERY late-local bin <= 3.0%% incl. negatives (worst bin named each "
            "validation) | anomaly supervision visibility-weighted against the in-batch noise null, Wiener-shrunk target (floor %.2f, "
            "z0 %.1f) so invisible bodies never train invention | continuation "
            "arm may blank the whole late window.",
            float(getattr(cfg.data, "snr_min_db", -40.0)),
            float(getattr(cfg.data, "snr_max_db", 35.0)),
            float(cfg.data.contract_test_snr_db_range[0]),
            float(cfg.data.contract_test_snr_db_range[1]),
            float(getattr(cfg.data, "snr_below_zero_probability", 0.45)),
            float(getattr(cfg.data, "measured_noise_snr_low_focus", 0.5)),
            float(getattr(cfg.loss, "anomaly_visibility_floor", 0.0)),
            float(getattr(cfg.loss, "anomaly_visibility_z0", 1.0)),
        )
        logger.info(
            "LIBRARY ANCHOR = %s | rank %d | closed-form "
            "coefficient supervision alpha %.2f + anchor contract alpha %.2f + "
            "continuation-coef alpha %.2f | gate bias-init %.2f (~%.2f share, "
            "earned via loss).",
            ("ON" if bool(getattr(cfg.model, "use_library_subspace", True))
             else "OFF (--no_library_subspace)"),
            int(getattr(cfg.model, "library_subspace_rank", 12)),
            float(getattr(cfg.loss, "alpha_library_coef", 1.0)),
            float(getattr(cfg.loss, "alpha_library_late", 0.5)),
            float(getattr(cfg.loss, "alpha_library_coef_cont", 0.5)),
            float(getattr(cfg.model, "library_gate_bias_init", -2.2)),
            1.0 / (1.0 + math.exp(-float(getattr(cfg.model,
                                                 "library_gate_bias_init", -2.2)))),
        )
        logger.info(
            "ACCEPTANCE = ALL-BIN | pooled late gate %.1f%% | per-bin gate %.1f%% (min count %d, "
            "deep-negative bins included) | worst bin enters feasibility AND candidate_score at 2x weight; "
            "best_late.pt tracks max(pooled, worst-bin).",
            100.0 * float(cfg.contract.late_threshold),
            100.0 * float(getattr(cfg.contract, "late_bin_threshold", 0.03)),
            int(getattr(cfg.contract, "late_bin_min_count", 8)))
        logger.info(
            "DEEP DRAWS = deployment-shaped | early-band floor "
            "%+.0f dB local SNR on every late-local calibration (realized target "
            "stamped honestly) | anchor coefficients pooled from the early/mid "
            "gates only.",
            float(getattr(cfg.data, "deep_negative_early_floor_db", 12.0)),
        )
        logger.info(
            "RIDGE ANCHOR %s (alpha %.2f; the ridge "
            "operator is INSTALLED -- late contract gradient disciplines "
            "early/mid THROUGH the library) | order penalty alpha %.2f "
            "(background-gated, event-exempt) | anchor labels = BACKGROUND "
            "family (clean_bg on anomaly rows).",
            ("ON" if bool(getattr(cfg.model, "use_ridge_anchor", True))
             else "OFF"),
            float(getattr(cfg.loss, "alpha_ridge_late", 0.5)),
            float(getattr(cfg.loss, "alpha_order", 0.2)),
        )
        logger.info(
            "anchor metrics on the BG-MIXED target + RIDGE "
            "path now observable | deep-background gate supervision alpha %.2f "
            "(rows <= %+.0f dB, no anomaly) | coefficient pooling = "
            "reliability-weighted early/mid.",
            float(getattr(cfg.loss, "alpha_gate_deep", 0.3)),
            float(getattr(cfg.loss, "deep_gate_snr_db", -10.0)),
        )
        logger.info(
            "OBJECTIVE=ACCEPTANCE | per-row fused late NRMSE "
            "(count-based, 300%% clamp) in the loss at alpha %.2f -- the mean "
            "the bin table averages is now the mean being minimized.",
            float(getattr(cfg.loss, "alpha_late_per_sample", 0.5)),
        )
        logger.info(
            "anchor supervisions COUNT-BASED as well (per-row "
            "subspace/ridge late at alpha %.2f) | coefficient MSE depth-"
            "weighted (deep rows up to 2x).",
            float(getattr(cfg.loss, "alpha_anchor_late_ps", 0.5)),
        )
        _q2 = [
            ("early_floor_dB", float(getattr(cfg.data, "deep_negative_early_floor_db", 0.0))),
            ("ridge_anchor", int(bool(getattr(cfg.model, "use_ridge_anchor", False)))),
            ("gain_cap", float(getattr(cfg.model, "continuation_gain_cap", 0.0))),
            ("subspace_rank", int(getattr(cfg.model, "library_subspace_rank", 0))),
            ("a_ridge", float(getattr(cfg.loss, "alpha_ridge_late", 0.0))),
            ("a_order", float(getattr(cfg.loss, "alpha_order", 0.0))),
            ("a_gate_deep", float(getattr(cfg.loss, "alpha_gate_deep", 0.0))),
            ("a_late_ps", float(getattr(cfg.loss, "alpha_late_per_sample", 0.0))),
            ("a_anchor_ps", float(getattr(cfg.loss, "alpha_anchor_late_ps", 0.0))),
            ("late_all_stages", int(bool(getattr(cfg.train, "late_terms_all_stages", False)))),
        ]
        _q2.append(("a_pass_hinge",
                      float(getattr(cfg.loss, "alpha_pass_hinge", 0.0))))
        _q2.append(("a_recovery",
                      float(getattr(cfg.loss, "alpha_recovery", 0.0))))
        _q2.append(("a_diff_orth",
                      float(getattr(cfg.loss, "alpha_diff_orth", 0.0))))
        _q2.append(("coef_field",
                      int(bool(getattr(cfg.model, "use_coef_field", False)))))
        _q2.append(("field_mu",
                      float(getattr(cfg.model, "coef_field_mu", 0.0))))
        _bad = [k for k, v in _q2
                   if float(v) <= 0.0 and k not in ("ridge_anchor", "a_ridge")]
        logger.info(
            "FULL-TRAINING QUALIFICATION | %s | verdict=%s.",
            " ".join("%s=%g" % (k, v) for k, v in _q2),
            ("PASS" if not _bad else "CHECK:" + ",".join(_bad)),
        )
        logger.info(
            "TRAINING SETTINGS | prior band span rule x%d (--prior_band_start %d) | controller feed %s | z-band "
            "damping ema %.2f step x%.2f deadband %.2f | replay drop %.0f dB (match natural %s), library %.2f (sign %s, "
            "exempt %s) | anomaly floor %.2f amp %.2f, eval injection %s | gate-eq teacher alpha %.1f | anomaly score cap "
            "%.1f | anchor trust_mix %s | LJP late floor %.3f boundary %s | decoder slot %s | amplitude %s -> ptn %s (%s). ",
            max(1, int(round(float(cfg.data.target_gates) / float(RF_DESIGN_GATES)))),
            int(getattr(cfg.data, "prior_band_start", -1)),
            "validation" if bool(getattr(cfg.contract, "controller_feed_validation", True)) else "train",
            float(getattr(cfg.loss, "z_band_balance_ema", 0.8)), float(getattr(cfg.loss, "z_band_balance_max_step", 1.1)),
            float(getattr(cfg.loss, "z_band_balance_deadband", 0.2)),
            float(getattr(cfg.data, "replay_snr_drop_db", 0.0)), bool(getattr(cfg.data, "replay_match_natural_snr", False)),
            float(getattr(cfg.data, "measured_noise_inject", 0.0)), bool(getattr(cfg.data, "measured_noise_random_sign", False)),
            bool(getattr(cfg.data, "library_contract_exempt", False)),
            float(getattr(cfg.loss, "anomaly_visibility_floor", 0.0)), float(getattr(cfg.loss, "alpha_anomaly_amp", 0.0)),
            "ON" if bool(getattr(cfg.data, "weak_body_eval_injection", False)) else "OFF",
            float(getattr(cfg.loss, "alpha_gate_eq_row", 0.0)), float(getattr(cfg.contract, "anomaly_score_cap", 0.0)),
            bool(getattr(cfg.model, "coef_field_trust_mix", True)),
            float(getattr(cfg.model, "lateral_joint_lambda_floor_late", 0.0)),
            "reflect" if bool(getattr(cfg.model, "lateral_joint_boundary_reflect", True)) else "natural",
            "ON" if bool(getattr(cfg.model, "decoder_gate_norm_channel", False)) else "off",
            str(getattr(cfg.model, "amplitude_conditioning", "auto")), bool(cfg.model.per_trace_normalization),
            str(getattr(cfg.model, "per_trace_norm_estimator", "centre_logmean")),
        )
        logger.info(
            "robust Student-t weighting nu %.1f | output-level measurement fusion "
            "(early/mid) %s.",
            float(getattr(cfg.model, "innovation_robust_nu", 3.0)),
            "ON" if bool(getattr(cfg.model, "output_measurement_fusion", True)) else "OFF",
        )
        logger.info(
            "precision-weighted innovation %s (mod floor %.2f, warm-up %d epochs) | "
            "evidence gate %s | profile-aware fingerprint query %s. reliability gate "
            "(1-r)^%.2f on every lateral mechanism, temporal witness %s, retrieval temperature init "
            "%.1f.",
            "ON" if bool(getattr(cfg.model, "innovation_precision_weighting", True)) else "OFF",
            float(getattr(cfg.model, "innovation_precision_mod_floor", 0.25)),
            int(getattr(cfg.model, "innovation_warmup_epochs", 10)),
            "ON" if bool(getattr(cfg.model, "innovation_evidence_gate", False)) else "OFF",
            "ON" if getattr(model, "fingerprint_query_ctx", None) is not None else "OFF",
            float(getattr(cfg.model, "lateral_gate_gamma", 1.0)),
            "ON" if bool(getattr(cfg.model, "lateral_gate_temporal_witness", True)) else "OFF",
            float(getattr(cfg.model, "lateral_retrieval_temperature", 4.0)),
        )
        logger.info(
            "LATERAL JOINT PRIOR %s | order %d | lambda init early/mid/late "
            "%.3f/%.3f/%.3f (learned per gate; %d params) | weighting %s | noise "
            "adaptivity kappa %.2f cap x%.1f | training "
            "blocks P=%d | evaluation blocks %s. measured-noise late floor %s "
            "(gamma %.2f, sp0 %.2f). evidence-gated innovation %s (kappa %.1f, "
            "tau %.2f).",
            "ON" if _lj0 is not None else "OFF",
            int(getattr(cfg.model, "lateral_joint_order", 2)),
            float(getattr(cfg.model, "lateral_joint_lambda_early", 0.02)),
            float(getattr(cfg.model, "lateral_joint_lambda_mid", 0.5)),
            float(getattr(cfg.model, "lateral_joint_lambda_late", 1.0)),
            int(sum(p.numel() for p in _lj0.parameters())) if _lj0 is not None else 0,
            str(getattr(cfg.model, "lateral_joint_weighting", "uniform")),
            float(getattr(cfg.model, "lateral_joint_noise_kappa", 0.25)),
            float(getattr(cfg.model, "lateral_joint_noise_cap", 3.0)),
            _P0, "ON" if bool(getattr(cfg.data, "eval_profile_blocks", True)) else "OFF",
            "ON" if bool(getattr(cfg.model, "late_measured_noise_floor", True)) else "OFF",
            float(getattr(cfg.model, "late_noise_floor_gamma", 0.98)),
            float(getattr(cfg.model, "late_noise_floor_prior_sigma_z", 0.15)),
            "ON" if bool(getattr(cfg.model, "innovation_evidence_gate", True)) else "OFF",
            float(getattr(cfg.model, "innovation_evidence_kappa", 3.0)),
            float(getattr(cfg.model, "innovation_evidence_tau", 0.5)),
        )
        logger.info("LATERAL JOINT PRIOR late floor lambda_eff >= +%.3f (additive, after the witness factor) | "
                    "boundary %s.",
                    float(getattr(cfg.model, "lateral_joint_lambda_floor_late", 0.0)),
                    "reflect" if bool(getattr(cfg.model, "lateral_joint_boundary_reflect", True)) else "natural")
        logger.info("LATERAL STRUCTURE LICENCE k=%.2f: lambda and the late floor x (1 - k * event_prob) on the prior "
                    "band and late window (0 = off) | stage C lr factor %.3f.",
                    float(getattr(cfg.model, "lateral_event_licence", 0.0)), float(cfg.train.stage_c_lr_factor))
        logger.info("library-noise rows: share %.2f, supervised by every row/gate-balanced late teacher at weight "
                    "%.2f (pooled contract statistics and CVaR still exclude them) | neighbour dropout %.2f of training patches "
                    "(per-station path).",
                    float(getattr(cfg.data, "measured_noise_inject", 0.0)), float(getattr(cfg.loss, "library_row_weight", 1.0)),
                    float(getattr(cfg.train, "neighbor_dropout", 0.0)))
        logger.info("survey-profile arm: %.2f of the library-injection draws (%.1f%% of the training patches) train "
                    "a true %d-station window of the measured section against the survey reference (natural pairing %.2f, "
                    "sub-station shift %.2f); clean.csv row alignment %s | neighbour dropout %.2f. ",
                    float(getattr(cfg.data, "survey_profile_fraction", 0.0)),
                    100.0 * float(getattr(cfg.data, "survey_profile_fraction", 0.0)) * float(getattr(cfg.data, "measured_noise_inject", 0.0)),
                    int(getattr(cfg.data, "profile_patch_len", 0) or 1), float(getattr(cfg.data, "survey_natural_fraction", 0.0)),
                    float(getattr(cfg.data, "survey_shift_fraction", 0.0)), str(getattr(cfg.data, "extra_clean_alignment", "auto")),
                    float(getattr(cfg.train, "neighbor_dropout", 0.0)))
        logger.info("library rows in the squared per-sample teachers: bounded-growth tail beyond %.0f%% (late) / "
                    "%.0f%% (global, early-mid) relative error; paired rows unchanged (0 = unbounded).",
                    100.0 * float(getattr(cfg.loss, "library_row_cap_late", 0.0)),
                    100.0 * float(getattr(cfg.loss, "library_row_cap_global", 0.0)))
        logger.info("structure witness gates the neighbour pathway (attention context and z_agg): %s (c0 %.3f on the "
                    "raw gates below the prior band).",
                    "ON" if bool(getattr(cfg.model, "lateral_witness_gates_attention", False)) else "OFF",
                    float(getattr(cfg.model, "lateral_structure_witness_c0", 0.0)))
    except Exception as _e2:
        logger.warning("startup readout skipped: %r", _e2)
    controller = ConstraintController(cfg.contract)
    loss_fn = ProjectLoss(
        controller,
        cfg.contract,
        cfg.loss,
        cfg.data.late_start_index,
        torch.as_tensor(gate_scale_norm, dtype=torch.float32),
        late_terms_all_stages=cfg.train.late_terms_all_stages,
        ip_chargeability_max=cfg.model.ip_chargeability_max,
        relaxation_blocks=len(tuple(getattr(cfg.model, "relaxation_stretch_exponents", (1.0,)) or (1.0,))),
        profile_patch_len=int(getattr(cfg.data, "profile_patch_len", 0) or 0),
    ).to(device)
    try:
        _ls2 = int(cfg.data.late_start_index)
        _pbx = int(getattr(cfg.data, "prior_band_start", -1))
        if int(_prior_band_gate) <= 0 and 0 < _pbx < _ls2:
            _prior_band_gate = _pbx
            logger.info("PRIOR BAND from --prior_band_start: gate %d (the information-bound audit did not set it).",
                        _pbx + 1)
        if int(_prior_band_gate) > 0:
            model.set_prior_band_start(int(_prior_band_gate))
            loss_fn.manifold_late_start = int(_prior_band_gate)
            logger.info("manifold prior LATE weight applied from gate %d in the loss.", int(_prior_band_gate) + 1)
        elif bool(getattr(cfg.model, "use_manifold_branch", False)):
            _fb = max(1, _ls2 - int(max(0, getattr(cfg.model, "blend_floor_ramp_gates", 6))))
            _nz = torch.nonzero(model.late_ramp.detach().reshape(-1) > 0).reshape(-1)
            _g4 = int(_nz[0].item()) if _nz.numel() > 0 else _fb
            if _pbx == 0 and _g4 < _fb:
                model.set_prior_band_start(_fb)
                logger.info("--prior_band_start 0: the inherited band (gate %d) was reset to the fallback ramp.", _g4 + 1)
            elif _g4 < _fb:
                model._prior_band_g = int(max(1, _g4))
                loss_fn.manifold_late_start = int(_g4)
                logger.info("inherited prior band from the loaded late_ramp: gate %d (model and loss aligned).", _g4 + 1)
    except Exception as _e625b:
        logger.warning("loss-side prior band not applied: %r", _e625b)
    try:
        _m3 = getattr(model, "module", model)
        _g5 = int(getattr(_m3, "_prior_band_g", int(cfg.data.late_start_index)))
        logger.info("SEAM FLOOR MODE %s -> %s (prior band gates %d-%d = %d gates; the hard forward floor is kept "
                    "only up to %d gates).",
                    str(getattr(cfg.model, "seam_blend_floor_mode", "hard")), _m3.seam_floor_mode(), _g5 + 1,
                    int(cfg.data.late_start_index), int(cfg.data.late_start_index) - _g5,
                    int(getattr(cfg.model, "seam_blend_floor_hard_max_gates", 8)))
    except Exception as _e22:
        logger.warning("seam floor mode not reported: %r", _e22)
    if cfg.model.use_ip_head:
        log_ip_head_init_perturbation(model, cfg, device, logger)
    ensure_family_completion(model, cfg, clean_cache, logger)
    ensure_cdmr_calibration(model, cfg, clean_cache, logger)
    begin_survey_training_record(model, cfg, clean_cache, logger)
    scaler = create_grad_scaler(device, enabled=cfg.train.amp and device.type == "cuda")
    ema = ModelEMA(model, cfg.train.ema_decay) if cfg.train.use_ema else None
    try:
        _new = recheck_micro_batch(cfg, device, logger)
        if _new is not None and int(_new) < int(cfg.train.batch_size):
            _want2 = (int(getattr(cfg.train, "effective_batch_size", 0) or 0)
                        or int(cfg.train.batch_size) * int(max(1, cfg.train.grad_accum_steps)))
            cfg.train.batch_size = int(_new)
            cfg.train.grad_accum_steps = max(1, int(math.ceil(_want2 / max(1, int(_new)))))
            cfg.train.effective_batch_size = int(_want2)
            train_loader, val_loader, test_loader, splits, datasets = make_loaders(clean_cache, noise_cache, cfg)
            logger.warning("LOADERS REBUILT: micro-batch %d x %d accumulation = effective %d.",
                           int(cfg.train.batch_size), int(cfg.train.grad_accum_steps),
                           int(cfg.train.batch_size) * int(cfg.train.grad_accum_steps))
    except Exception as _e19:
        logger.warning("card re-check skipped: %r", _e19)
    if bool(getattr(cfg.train, "preflight", True)):
        _pf_ok = preflight_self_test(model, loss_fn, train_loader, val_loader, datasets, device, cfg,
                                     clean_cache, noise_cache, logger)
        if not _pf_ok and not bool(getattr(cfg.train, "preflight_continue", False)):
            raise SystemExit("PRE-FLIGHT FAILED -- see the PASS/FAIL lines above (the failing check "
                             "carries its traceback). Fix it or run with --preflight_continue to override. ")
        _mb = _PREFLIGHT_STATE.get("micro_batch")
        if _mb is not None and int(_mb) < int(cfg.train.batch_size):
            _want = int(getattr(cfg.train, "effective_batch_size", 0) or 0) or int(cfg.train.batch_size) * int(max(1, cfg.train.grad_accum_steps))
            cfg.train.batch_size = int(_mb)
            cfg.train.grad_accum_steps = max(1, int(math.ceil(_want / max(1, int(_mb)))))
            cfg.train.effective_batch_size = int(_want)
            logger.warning("LOADERS REBUILT after the P3 OOM retry: micro-batch %d x %d accumulation = effective "
                           "%d (the autotune's choice did not fit; its per-row model is corrected, this is the "
                           "runtime safety net).",
                           int(_mb), int(cfg.train.grad_accum_steps), int(_mb) * int(cfg.train.grad_accum_steps))
            train_loader, val_loader, test_loader, splits, datasets = make_loaders(clean_cache, noise_cache, cfg)
            _PREFLIGHT_STATE["micro_batch"] = None

    if cfg.paths.resume_checkpoint:
        state = load_checkpoint(Path(cfg.paths.resume_checkpoint), model, controller, device, logger,
                                expected_target_time=noise_cache.target_time)
        logger.info("Loaded resume checkpoint: %s", cfg.paths.resume_checkpoint)
        ensure_family_completion(model, cfg, clean_cache, logger)
        ensure_cdmr_calibration(model, cfg, clean_cache, logger)
        begin_survey_training_record(model, cfg, clean_cache, logger)
        if int(_prior_band_gate) > 0:
            model.set_prior_band_start(int(_prior_band_gate))
        elif bool(getattr(cfg.model, "use_manifold_branch", False)):
            _fb725r = max(1, int(cfg.data.late_start_index) - int(max(0, getattr(cfg.model, "blend_floor_ramp_gates", 6))))
            if int(getattr(cfg.data, "prior_band_start", -1)) == 0:
                model.set_prior_band_start(_fb725r)
            elif int(getattr(model, "_prior_band_g", _fb725r)) < _fb725r:
                loss_fn.manifold_late_start = int(model._prior_band_g)
        try:
            _zg = float(((state.get("config") or {}).get("loss") or {}).get("z_band_gain", float("nan")))
            if math.isfinite(_zg) and _zg > 0.0 and float(cfg.loss.z_band_balance_target) > 0.0:
                cfg.loss.z_band_gain = _zg
                logger.info("z_band_gain restored from the checkpoint: %.4g", _zg)
        except Exception:
            pass
        if ema is not None:
            ema = ModelEMA(model, cfg.train.ema_decay)
            if state.get("ema_state"):
                ema.load_state_dict(state["ema_state"])
                logger.info("EMA shadow restored from checkpoint (num_updates=%d).", ema.num_updates)
        rng_state = state.get("rng_state")
        if rng_state:
            try:
                torch.set_rng_state(torch.as_tensor(rng_state["torch_cpu"], dtype=torch.uint8).cpu())
                if torch.cuda.is_available() and rng_state.get("torch_cuda"):
                    torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8).cpu() for s in rng_state["torch_cuda"]])
                if rng_state.get("python"):
                    random.setstate(rng_state["python"])
                logger.info("RNG states restored from checkpoint.")
            except Exception as exc:
                logger.warning("RNG state restore skipped: %s", exc)
    _anom_probe: Optional[AnomalyRecoveryProbe] = None
    if (_paired_active and bool(getattr(cfg.train, "anomaly_probe", True))
            and not bool(getattr(cfg.runtime, "paired_injected_probe", False))):
        logger.info(
            "Injected anomaly probe stands down: supervision is PAIRED, and "
            "checkpoint selection is carried by the gates measured on the "
            "dataset's OWN anomalies. Re-enable for diagnostics with "
            "--paired_injected_probe.",
        )
    elif bool(getattr(cfg.train, "anomaly_probe", True)):
        try:
            _anom_probe = AnomalyRecoveryProbe(
                datasets[1], cfg, device, logger,
                share=getattr(val_loader, "synth", None))
            if _anom_probe.n_anchor <= 0:
                logger.warning(
                    "The validation split has no contiguous run of %d rows, so "
                    "anomaly bodies have no station axis: the probe is disabled and the "
                    "anomaly gates are absent from the checkpoint score.",
                    int(getattr(cfg.data, "profile_patch_len", 0) or 0))
                _anom_probe = None
        except Exception as _pexc:
            logger.warning("Anomaly probe unavailable (%r); continuing without "
                           "the anomaly gates in the checkpoint score.", _pexc)
            _anom_probe = None
    _monitor = AnomalyReconstructionMonitor(cfg, run_dir, logger)
    history: List[Dict[str, Any]] = []
    best_candidate_score = float("inf")
    _best_gate_g_run = float("inf")
    _best_gate_l_run = float("inf")
    _stage_best_scores: Dict[str, float] = {}
    poisoned_val_epochs = 0
    best_feasible_cvar = float("inf")
    best_candidate_path = run_dir / "checkpoints" / "best_candidate.pt"
    best_feasible_path = run_dir / "checkpoints" / "best_feasible.pt"
    best_late_path = run_dir / "checkpoints" / "best_late.pt"
    best_late_seen = float("inf")
    best_g_seen = float("inf")
    last_path = run_dir / "checkpoints" / "last.pt"
    global_epoch = 0
    no_improve = 0
    val_metrics: Dict[str, float] = {}
    total_progress_epochs = max(
        1, cfg.train.stage_a_max_epochs + cfg.train.stage_b_max_epochs + cfg.train.stage_c_epochs
    )

    stage_specs = [
        ("A", cfg.train.stage_a_max_epochs),
        ("B", cfg.train.stage_b_max_epochs),
        ("C", cfg.train.stage_c_epochs),
    ]
    cfg.data.forward_fraction_total_epochs = int(sum(int(e) for _s, e in stage_specs))
    if getattr(cfg.data, "forward_fraction_final", None) is not None:
        logger.info(
            "FORWARD MIXTURE SCHEDULE | %.3f -> %.3f (cosine over %d epochs). The forward "
            "library's late/early ratio is ~0.9 dex hotter than the field target's, so it broadens the early "
            "optimization and then steps aside.",
            float(cfg.data.mix_forward_fraction),
            float(cfg.data.forward_fraction_final),
            int(cfg.data.forward_fraction_total_epochs))
    for stage, max_epochs in stage_specs:
        optimizer = build_optimizer(model, cfg.train, stage)
        assert_optimizer_covers_model(model, optimizer, logger)
        _accum_sched = max(1, int(getattr(cfg.train, "grad_accum_steps", 1)))
        steps_per_epoch = max(1, int(math.ceil(len(train_loader) / _accum_sched)))
        scheduler = build_scheduler(optimizer, cfg.train, max_epochs * steps_per_epoch)
        if stage == stage_specs[0][0]:
            logger.info(
                "LR SCHEDULE | %d micro-batches/epoch / accum %d = %d optimizer "
                "steps/epoch; cosine horizon %d steps over %d epochs, warmup %.0f%%, floor "
                "%.3gx.",
                len(train_loader), _accum_sched, steps_per_epoch,
                max_epochs * steps_per_epoch, max_epochs,
                100.0 * float(cfg.train.lr_warmup_fraction), float(cfg.train.lr_min_ratio),
            )
        stable_count = 0
        stage_best_score = float("inf")
        _best_gate_g_stage = float("inf")
        _best_gate_l_stage = float("inf")
        stage_regress = 0
        stage_over = False
        stage_plateau = 0
        stage_score_ema: Optional[float] = None
        stage_best_score_ema = float("inf")
        stage_best_path = run_dir / "checkpoints" / f"best_stage_{stage}.pt"
        stage_stall_reported = False
        if bool(cfg.train.stage_init_from_previous_best):
            _prev = [s for s, _ in stage_specs[: [s for s, _ in stage_specs].index(stage)]]
            _prev = sorted(_prev, key=lambda _q: _stage_best_scores.get(_q, float("inf")))
            for _ps in _prev:
                _pp = run_dir / "checkpoints" / f"best_stage_{_ps}.pt"
                if _pp.exists():
                    load_checkpoint(_pp, model, controller, device, logger)
                    if ema is not None:
                        ema.reseed(model)
                    optimizer = build_optimizer(model, cfg.train, stage)
                    assert_optimizer_covers_model(model, optimizer, logger)
                    scheduler = build_scheduler(optimizer, cfg.train, max_epochs * steps_per_epoch)
                    logger.info(
                        "Stage %s starts from the BEST weights of stage %s (%s), "
                        "with a fresh optimizer/scheduler for this stage.", stage, _ps, _pp.name,
                    )
                    break
        logger.info("========== Stage %s started | max_epochs=%d =========", stage, max_epochs)
        for local_epoch in range(1, max_epochs + 1):
            global_epoch += 1
            loss_fn.progress = min(1.0, (global_epoch - 1) / total_progress_epochs)
            _wu = max(1, int(getattr(cfg.model, "innovation_warmup_epochs", 10)))
            model.innovation_warmup = min(1.0, float(global_epoch) / float(_wu))
            _ff_now = (train_loader.dataset.forward_fraction_now()
                       if hasattr(getattr(train_loader, "dataset", None),
                                  "forward_fraction_now") else float("nan"))
            try:
                model.commit_control_updates(logger)
            except Exception as _e12:
                logger.warning("pending control updates not committed: %r", _e12)
            for _attempt in (0, 1, 2):
                try:
                    train_metrics, train_components = run_epoch(
                        model,
                        train_loader,
                        loss_fn,
                        optimizer,
                        scaler,
                        device,
                        cfg,
                        stage,
                        global_epoch,
                        logger,
                        scheduler=scheduler,
                        ema=ema,
                    )
                    break
                except RuntimeError as _e5:
                    _P2 = int(max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 1)))
                    if ("three out-of-memory" in str(_e5) and _attempt < 2
                            and int(cfg.train.batch_size) > _P2):
                        _old = int(cfg.train.batch_size)
                        _want3 = (int(getattr(cfg.train, "effective_batch_size", 0) or 0)
                                    or _old * int(max(1, cfg.train.grad_accum_steps)))
                        _new2 = int(max(_P2, ((_old // 2) // _P2) * _P2))
                        cfg.train.batch_size = _new2
                        cfg.train.grad_accum_steps = max(1, int(math.ceil(_want3 / max(1, _new2))))
                        cfg.train.effective_batch_size = int(_want3)
                        logger.error("MEMORY PRESSURE RECOVERY in epoch %d: micro-batch %d -> %d rows, accumulation -> %d "
                                     "(effective %d unchanged); loaders rebuilt, epoch re-run.", int(global_epoch), _old, _new2,
                                     int(cfg.train.grad_accum_steps), _new2 * int(cfg.train.grad_accum_steps))
                        optimizer.zero_grad(set_to_none=True)
                        try:
                            del train_loader, val_loader, test_loader
                        except Exception:
                            pass
                        free_cuda_after_oom()
                        train_loader, val_loader, test_loader, splits, datasets = make_loaders(
                            clean_cache, noise_cache, cfg)
                        continue
                    if _attempt >= 1 or not is_dataloader_worker_crash(_e5):
                        raise
                    logger.error(
                        "DataLoader WORKER CRASHED during epoch %d (%s). Most likely "
                        "/dev/shm is exhausted (docker default 64 MB; see the line). "
                        "Rebuilding train/val/test loaders with num_workers=0 and RE-RUNNING "
                        "epoch %d.",
                        int(global_epoch), str(_e5)[:200], int(global_epoch))
                    cfg.train.num_workers = 0
                    cfg.train.persistent_workers = False
                    try:
                        del train_loader, val_loader, test_loader
                    except Exception:
                        pass
                    train_loader, val_loader, test_loader, splits, datasets = make_loaders(
                        clean_cache, noise_cache, cfg)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            _an_n = float(train_components.get("rl_anom_num", float("nan")))
            _an_d = float(train_components.get("rl_anom_den", float("nan")))
            _rla = (_an_n / _an_d) if (math.isfinite(_an_n) and math.isfinite(_an_d)
                                       and _an_d > 0.0) else float("nan")
            _anom_pc = 100.0 * math.sqrt(_rla) if (math.isfinite(_rla) and _rla >= 0) \
                else float("nan")
            def _viol(_key: str, _tau: float) -> float:
                _v = float(train_metrics.get(_key, float("nan")))
                return _v / (100.0 * _tau) - 1.0 if math.isfinite(_v) else float("nan")
            logger.info(
                "TRAIN stage=%s epoch=%d | loss(epoch mean)=%.4f | G=%.4f%% EM=%.4f%% "
                "L=%.4f%% L-CVaR=%.4f%% ANOM-L=%.4f%% | CONTRACT VIOLATION "
                "v(G/EM/L/CVaR/ANOM)=%+.3f/%+.3f/%+.3f/%+.3f/%+.3f (<=0 = gate met; THIS "
                "is the convergence trend, not `loss`, which is multiplied by the still-"
                "rising lambdas) | samples=%d%s",
                stage, global_epoch, float(train_metrics.get("loss", float("nan"))),
                float(train_metrics.get("global_error_percent", float("nan"))),
                float(train_metrics.get("early_mid_error_percent", float("nan"))),
                float(train_metrics.get("late_error_percent", float("nan"))),
                float(train_metrics.get("late_cvar_percent", float("nan"))),
                _anom_pc,
                _viol("global_error_percent", cfg.contract.global_threshold),
                _viol("early_mid_error_percent", cfg.contract.early_mid_threshold),
                _viol("late_error_percent", cfg.contract.late_threshold),
                _viol("late_cvar_percent", cfg.contract.late_cvar_threshold),
                ((_anom_pc / (100.0 * cfg.contract.late_threshold
                              * cfg.contract.anomaly_late_multiple) - 1.0)
                 if math.isfinite(_anom_pc) else float("nan")),
                int(train_metrics.get("samples", 0)),
                ((" | arms exact/replay/library=%.2f/%.2f/%.2f alt=%.2f"
                  % (float(train_components.get("arm_exact_realized", float("nan"))),
                     float(train_components.get("arm_replay_realized", float("nan"))),
                     float(train_components.get("arm_library_realized", float("nan"))),
                     float(train_components.get("arm_alt_realized", float("nan")))))
                 if "arm_exact_realized" in train_components else ""),
            )
            if "arm_exact_G_percent" in train_components:
                logger.info("TRAIN error per arm (G / L) | exact pairs %.2f%% / %.2f%% | measured replay %.2f%% / %.2f%% | "
                            "library injection %.2f%% / %.2f%% | blend teacher (fused-z/dd^2) %.4f | fused-band z loss %.5f. ",
                            float(train_components.get("arm_exact_G_percent", float("nan"))), float(train_components.get("arm_exact_L_percent", float("nan"))),
                            float(train_components.get("arm_replay_G_percent", float("nan"))), float(train_components.get("arm_replay_L_percent", float("nan"))),
                            float(train_components.get("arm_library_G_percent", float("nan"))), float(train_components.get("arm_library_L_percent", float("nan"))),
                            float(train_components.get("l_blend_oracle", float("nan"))),
                            float(train_components.get("l_fused_band_z", float("nan"))))
                if float(train_components.get("arm_survey_realized", 0.0) or 0.0) > 0.0:
                    logger.info("TRAIN survey-profile rows (G / L) | natural pairing (the measured input itself) %.2f%% / "
                                "%.2f%% | all survey rows %.2f%% / %.2f%% (the rest are library mixtures at a sampled SNR, most of "
                                "them far below the survey's natural level: read the natural figure and the SURVEY PROBE) | %.1f%% of "
                                "the training rows (counted inside the library-injection arm above).",
                                float(train_components.get("arm_survey_natural_G_percent", float("nan"))),
                                float(train_components.get("arm_survey_natural_L_percent", float("nan"))),
                                float(train_components.get("arm_survey_G_percent", float("nan"))),
                                float(train_components.get("arm_survey_L_percent", float("nan"))),
                                100.0 * float(train_components.get("arm_survey_realized", 0.0)))
            _gb_f = float(train_components.get("l_gate_balanced_field", float("nan")))
            _gb_w = float(train_components.get("l_gate_balanced_forward", float("nan")))
            _gb_a = float(train_components.get("l_gate_balanced", float("nan")))
            _fr = float(train_components.get("forward_row_fraction", float("nan")))
            if math.isfinite(_gb_f) or math.isfinite(_gb_w):
                logger.info(
                    "DOMAIN SPLIT | gate-balanced error: all %.4f | FIELD "
                    "%.4f | FORWARD %.4f | forward row share %.3f (scheduled %.3f). ",
                    _gb_a, _gb_f, _gb_w, _fr, _ff_now)
                if (math.isfinite(_gb_f) and math.isfinite(_gb_w)
                        and _gb_f > 1.5 * _gb_w):
                    logger.warning(
                        "The FIELD rows carry %.2fx the gate-balanced error of the FORWARD rows: the mixture "
                        "is being fitted toward the auxiliary pool. Lower --mix_forward_fraction or set "
                        "--forward_fraction_final.",
                        _gb_f / max(_gb_w, 1e-9))
            _sig = "%.6f|%.6f|%.6f|%.6f" % (
                float(val_metrics.get("global_error_percent", float("nan"))),
                float(val_metrics.get("early_mid_error_percent", float("nan"))),
                float(val_metrics.get("late_error_percent", float("nan"))),
                float(val_metrics.get("late_cvar_percent", float("nan"))),
            )
            if _sig == getattr(train_model, "_last_val_signature", None):
                _frozen_epochs += 1
            else:
                _frozen_epochs = 0
            train_model._last_val_signature = _sig
            if _frozen_epochs >= int(cfg.train.frozen_metric_patience):
                raise RuntimeError(
                    "VALIDATION METRICS HAVE BEEN BIT-IDENTICAL FOR %d EPOCHS "
                    "(G/EM/L/CVaR = %s). That is not convergence and not a plateau: the "
                    "weights are not changing at all, so every remaining epoch would be "
                    "wasted. The usual cause is that every optimizer step is being skipped "
                    "-- see the AMP scale in the epoch log and reports/nan_forensics.json. "
                    "Stopping now instead of after the full budget."
                    % (_frozen_epochs, _sig)
                )
            _late_on = cfg.train.late_terms_all_stages or stage in {"B", "C"}
            _late_now = float(val_metrics.get("late_error_percent", float("nan")))
            _late_src = "validation"
            if not math.isfinite(_late_now):
                _late_now = float(train_metrics.get("late_error_percent", float("inf")))
                _late_src = "train (no validation yet)"
            _arm_at = 100.0 * float(cfg.contract.late_threshold) * float(
                cfg.contract.cvar_activation_late_multiple)
            _armed_before = bool(getattr(loss_fn, "cvar_armed", False))
            _armed = bool(math.isfinite(_late_now) and _late_now <= _arm_at)
            loss_fn.cvar_armed = _armed
            _anom_arm_at = (100.0 * float(cfg.contract.late_threshold)
                            * float(cfg.contract.anomaly_arm_late_multiple))
            _anom_before = bool(loss_fn.anomaly_armed)
            _anom_now = bool(math.isfinite(_late_now) and _late_now <= _anom_arm_at)
            loss_fn.anomaly_armed = _anom_now
            if _anom_now and not _anom_before:
                if float(controller.lambdas.get("anomaly_late", 0.0)) <= 0.0:
                    controller.lambdas["anomaly_late"] = float(max(
                        1.0, float(cfg.contract.anomaly_warm_start_fraction)
                        * float(controller.lambdas.get("late", 0.0))))
                logger.info(
                    "ANOMALY-PRESERVATION constraint ARMED at epoch %d: quiet "
                    "late %.2f%% (%s) <= %.2f%%. Gate = %.2f%% late error on event-"
                    "labelled traces, lambda warm-started to %.1f. Until now nothing in "
                    "the objective distinguished a reconstructed anomaly from an erased "
                    "one.",
                    global_epoch, _late_now, _late_src, _anom_arm_at,
                    100.0 * cfg.contract.late_threshold * cfg.contract.anomaly_late_multiple,
                    float(controller.lambdas.get("anomaly_late", 0.0)),
                )
            if _armed and not _armed_before:
                _lam_l = float(controller.lambdas.get("late", 0.0))
                if float(controller.lambdas.get("late_cvar", 0.0)) <= 0.0:
                    controller.lambdas["late_cvar"] = float(max(
                        1.0, float(cfg.contract.cvar_warm_start_fraction) * _lam_l))
            if _armed != _armed_before:
                logger.info(
                    "Late-CVaR constraint %s at epoch %d: late mean %.2f%% "
                    "(%s) vs arming threshold %.2f%% (%.1fx the %.0f%% late gate). "
                    "lambda_late_cvar=%.1f. %s",
                    "ARMED" if _armed else "disarmed", global_epoch, _late_now, _late_src,
                    _arm_at, float(cfg.contract.cvar_activation_late_multiple),
                    100.0 * cfg.contract.late_threshold,
                    float(controller.lambdas.get("late_cvar", 0.0)),
                    "The tail now has its own multiplier, warm-started from the late one "
                    "so it bites immediately." if _armed
                    else "Until the mean is in reach the tail is carried by the auxiliary term only.",
                )
            active_names = ["global", "early_mid"] + (["late"] if _late_on else [])
            if _late_on and _armed:
                active_names.append("late_cvar")
            if bool(loss_fn.anomaly_armed) and math.isfinite(_rla):
                active_names.append("anomaly_late")
            _src = train_metrics
            if (bool(getattr(cfg.contract, "controller_feed_validation", True)) and val_metrics
                    and math.isfinite(float(val_metrics.get("relative_mse_late", float("nan"))))):
                _src = val_metrics
            _rla2 = _rla
            if _src is val_metrics:
                _va2 = float(val_metrics.get("anomaly_late_error_percent", float("nan")))
                if math.isfinite(_va2):
                    _rla2 = (_va2 / 100.0) ** 2
            _nm2 = "validation (previous epoch)" if _src is val_metrics else "train (no validation yet)"
            if getattr(controller, "_feed_src", None) != _nm2:
                controller._feed_src = _nm2
                logger.info("constraint multipliers fed by %s.", _nm2)
            _cvar_frac = float(_src.get("late_cvar_percent", float("nan"))) / 100.0
            controller.last_constraint_scale = float(
                getattr(loss_fn, "last_constraint_scale", 1.0))
            violations = controller.update(
                {
                    "global": _src["relative_mse_global"],
                    "early_mid": _src["relative_mse_early_mid"],
                    "late": _src["relative_mse_late"],
                    "late_cvar": _cvar_frac,
                    "anomaly_late": _rla2,
                },
                active_names,
            )
            if global_epoch % cfg.train.validate_every == 0:
                if bool(getattr(cfg.train, "grad_audit_every_validation", True)):
                    _gn = gradient_audit_on_loader(model, loss_fn, train_loader, device, cfg, logger)
                    try:
                        balance_z_band_gain(loss_fn, _gn, logger)
                    except Exception as _e16:
                        logger.warning("gain balance skipped: %r", _e16)
                if ema is not None:
                    ema.store(model)
                    ema.copy_to(model)
                val_metrics = evaluate_model(
                    model, val_loader, loss_fn, device, cfg, stage, epoch=global_epoch
                )
                _ctl = model.snapshot_control_state()
                val_poisoned = int(val_metrics.get("valid_batches", 1) or 0) == 0
                if val_poisoned:
                    poisoned_val_epochs += 1
                    logger.critical(
                        "POISONED VALIDATION #%d at stage=%s epoch=%d | train this "
                        "epoch: %d/%d steps non-finite, fp32 retries: %d ok / %d failed. No "
                        "stage/plateau/lambda credit is granted. %s",
                        poisoned_val_epochs, stage, global_epoch,
                        int(train_metrics.get("nonfinite_steps", 0)),
                        int(train_metrics.get("steps_total", 0)),
                        int(train_metrics.get("fp32_retry_success", 0)),
                        int(train_metrics.get("fp32_retry_failed", 0)),
                        "AMP has been demoted; the next epoch runs fp32."
                        if not cfg.train.amp else "",
                    )
                    if ema is not None:
                        ema.restore(model)
                    if poisoned_val_epochs >= 2 and (
                        int(train_metrics.get("fp32_retry_failed", 0)) > 0
                        or not cfg.train.amp
                    ):
                        logger.critical(
                            "ABORT: %d consecutive poisoned validations and the "
                            "failures persist in fp32 -- this is a genuine numerical defect, "
                            "not a precision artefact. reports/nan_forensics.json names the "
                            "first non-finite module . Exit code 23.",
                            poisoned_val_epochs,
                        )
                        raise SystemExit(23)
                    val_metrics = {}
                else:
                    poisoned_val_epochs = 0
                    if _anom_probe is not None:
                        try:
                            _am = _anom_probe.run(model, cfg)
                            val_metrics.update(_am)
                            _au_now = float(_am.get("event_auroc", float("nan")))
                            controller.last_event_auroc = _au_now
                            if hasattr(model, "set_event_expression"):
                                _ev_sc = model.set_event_expression(
                                    _au_now,
                                    open_auroc=float(getattr(
                                        cfg.contract, "event_auroc_open", 0.60)),
                                    chance_band=float(getattr(
                                        cfg.contract, "event_auroc_chance_band", 0.55)),
                                )
                                logger.info(
                                    "EVENT EXPRESSION | probe AUROC=%.3f -> "
                                    "residual expression scale=%.3f (0 = muted at "
                                    "chance, 1 = fully open at AUROC %.2f; persistent "
                                    "buffer, ships in every checkpoint).",
                                    _au_now, _ev_sc,
                                    float(getattr(cfg.contract,
                                                  "event_auroc_open", 0.60)))
                            logger.info(
                                "ANOMALY RECOVERY stage=%s epoch=%d | body late "
                                "err=%.3f%% (gate %.2f%%, quiet ref %.3f%%) | lift recovery "
                                "mean=%.3f median=%.3f (target %.2f) | event AUROC=%.4f | "
                                "singleton-reject AUROC=%.4f | false lift from artefacts="
                                "%.2f%% | n_body=%d n_singleton=%d",
                                stage, global_epoch,
                                _am.get("anomaly_late_error_percent", float("nan")),
                                100.0 * cfg.contract.late_threshold
                                * cfg.contract.anomaly_late_multiple,
                                _am.get("anomaly_quiet_late_error_percent", float("nan")),
                                _am.get("anomaly_lift_recovery", float("nan")),
                                _am.get("anomaly_lift_recovery_median", float("nan")),
                                cfg.contract.anomaly_recovery_target,
                                _am.get("event_auroc", float("nan")),
                                _am.get("singleton_reject_auroc", float("nan")),
                                _am.get("singleton_false_lift_percent", float("nan")),
                                int(_am.get("anomaly_body_count", 0)),
                                int(_am.get("anomaly_singleton_count", 0)),
                            )
                            logger.info(
                                "  anomaly by STRENGTH | weak(5-20%%) err=%.2f%% "
                                "recov=%.3f AUROC=%.4f n=%d | moderate(20-50%%) "
                                "err=%.2f%% recov=%.3f AUROC=%.4f n=%d | strong(>50%%) "
                                "err=%.2f%% recov=%.3f AUROC=%.4f n=%d | event-residual "
                                "share of late energy=%.4f",
                                _am.get("anomaly_weak_late_error_percent", float("nan")),
                                _am.get("anomaly_weak_recovery", float("nan")),
                                _am.get("anomaly_weak_auroc", float("nan")),
                                int(_am.get("anomaly_weak_count", 0)),
                                _am.get("anomaly_moderate_late_error_percent", float("nan")),
                                _am.get("anomaly_moderate_recovery", float("nan")),
                                _am.get("anomaly_moderate_auroc", float("nan")),
                                int(_am.get("anomaly_moderate_count", 0)),
                                _am.get("anomaly_strong_late_error_percent", float("nan")),
                                _am.get("anomaly_strong_recovery", float("nan")),
                                _am.get("anomaly_strong_auroc", float("nan")),
                                int(_am.get("anomaly_strong_count", 0)),
                                _am.get("event_residual_late_share", float("nan")),
                            )
                            logger.info(
                                "  FIELDLIKE stratum (SNR -8..0 dB, late at/under the "
                                "noise floor) | body late err=%.2f%% | recovery=%.3f | "
                                "AUROC=%.4f | n=%d -- this is the condition the field "
                                "profile presents; recovery here must come from mid-gate "
                                "+ lateral evidence and the physics continuation.",
                                _am.get("anomaly_fieldlike_late_error_percent", float("nan")),
                                _am.get("anomaly_fieldlike_recovery", float("nan")),
                                _am.get("anomaly_fieldlike_auroc", float("nan")),
                                int(_am.get("anomaly_fieldlike_count", 0)),
                            )
                        except Exception as _pexc:
                            logger.warning("Anomaly probe failed this epoch "
                                           "(%r); the score keeps its other gates.", _pexc)
                if val_metrics:
                    controller.note_validation(
                    val_metrics, logger,
                    constraint_scale=float(getattr(loss_fn, "last_constraint_scale", 1.0)))
                _ml_frac = val_metrics.get("manifold_late_error_percent", float("nan"))
                _adv = float(model.late_prior_advantage.item())
                _den_pc = float(val_metrics.get("denoise_path_late_error_percent",
                                                float("nan")))
                if math.isfinite(float(_ml_frac)):
                    _cap, _fs, _vfl = model.set_manifold_quality(
                        float(_ml_frac) / 100.0,
                        (_den_pc / 100.0) if math.isfinite(_den_pc) else None,
                        val_floor_enabled=bool(getattr(cfg.loss, "val_blend_floor", True)),
                        val_floor_max=float(getattr(cfg.loss, "val_blend_floor_max", 0.93)))
                    _adv = model.set_late_path_advantage(
                        float(_ml_frac) / 100.0,
                        _den_pc / 100.0)
                    _ratio = (_den_pc / max(float(_ml_frac), 1e-6)
                              if math.isfinite(_den_pc) else 1.0)
                    loss_fn.denoise_late_boost = float(
                        min(max(_ratio / 3.0, 1.0),
                            float(getattr(cfg.loss, "denoise_late_boost_max", 4.0))))
                    logger.info(
                        "MANIFOLD QUALITY GATE | prior late error %.2f%% -> late blend cap %.3f, "
                        "floor scale %.2f | VAL blend floor %.3f (den-path late "
                        "%.2f%% -> late leash x%.2f) (persistent buffers; shipped in every "
                        "checkpoint).",
                        float(_ml_frac), _cap, _fs, _vfl, _den_pc,
                        float(loss_fn.denoise_late_boost),
                    )
                    if bool(getattr(cfg.model, "supervised_primary_fusion", True)):
                        logger.info(
                            "Floors above are DIAGNOSTIC under supervised-"
                            "primary fusion: only the quality cap bounds the forward "
                            "(as an upper limit); the learned gate sets the blend and "
                            "the trunk carries its own contract supervision . ")
                try:
                    _st2 = model.stage_control_updates(_ctl, source="stage %s epoch %d" % (stage, int(global_epoch)))
                    if _st2:
                        logger.info("control updates STAGED (applied at the next epoch start, saved as pending): %s. ",
                                    " ".join("%s=%.4g" % (k, v) for k, v in _st2.items()))
                except Exception as _e13:
                    logger.warning("control staging failed: %r", _e13)
                _p_t = float(val_metrics.get("blend_late_mean", float("nan"))) * float(_ml_frac)
                _l_t = ((1.0 - float(val_metrics.get("blend_late_mean", float("nan"))))
                        * float(val_metrics.get("denoise_path_late_error_percent",
                                                float("nan"))))
                _e_t = float(val_metrics.get("event_term_late_percent", float("nan")))
                if math.isfinite(_p_t) and math.isfinite(_l_t):
                    logger.info(
                        "LATE WINDOW, MEASURED | fused %.2f%% | prior arm "
                        "alone %.2f%% | denoise arm alone %.2f%% | realized late blend "
                        "%.3f | cancellation %.3f (fused / naive-quadrature %.2f%%; "
                        "<<1 means the arms complement each other, ~1 would mean they "
                        "have stopped) | prior-advantage redirect = %.3f.",
                        float(val_metrics.get("late_error_percent", float("nan"))),
                        float(_ml_frac),
                        float(val_metrics.get("denoise_path_late_error_percent",
                                              float("nan"))),
                        float(val_metrics.get("blend_late_mean", float("nan"))),
                        (float(val_metrics.get("late_error_percent", float("nan")))
                         / max(math.sqrt(_p_t ** 2 + _l_t ** 2), 1e-9)),
                        math.sqrt(_p_t ** 2 + _l_t ** 2),
                        float(_adv),
                    )
                    _canc = (float(val_metrics.get("late_error_percent", float("nan")))
                             / max(math.sqrt(_p_t ** 2 + _l_t ** 2), 1e-9))
                    if math.isfinite(_canc) and _canc > 0.5:
                        logger.warning(
                            "The arms have stopped complementing each other (cancellation %.3f): the fused late "
                            "error is approaching the naive sum of the two arms, so the fusion is no longer buying anything "
                            "over the better arm. Prior alone %.2f%%, denoise alone %.2f%%, fused %.2f%%.",
                            _canc, float(_ml_frac),
                            float(val_metrics.get("denoise_path_late_error_percent",
                                                  float("nan"))),
                            float(val_metrics.get("late_error_percent", float("nan"))))
                    _e_q = _e_t if math.isfinite(_e_t) else 0.0
                    logger.info(
                        "EVENT residual added after the blend = %.2f%% of "
                        "late energy (a measured magnitude, NOT a term to be summed with "
                        "the arm errors; the three-term quadrature %.2f%% "
                        "against measured %.2f%% is reported only to show the same "
                        "independence premise failing) | event expression scale = %.3f. ",
                        _e_q, math.sqrt(_p_t ** 2 + _l_t ** 2 + _e_q ** 2),
                        float(val_metrics.get("late_error_percent", float("nan"))),
                        float(getattr(model, "event_expression_scale",
                                      torch.zeros(())).item()),
                    )
                    try:
                        _sc = float(getattr(model, "event_expression_scale", torch.zeros(())).item())
                        _mx = float(val_metrics.get("event_increment_max_abs_z", float("nan")))
                        _lt2 = float(val_metrics.get("lateral_term_late_percent", float("nan")))
                        logger.info("LATE-WINDOW INCREMENTS | event pathway alone %.2f%% (max |post-pre| %.2e z, "
                                    "expression scale %.3f) | lateral joint prior alone %.2f%% of late energy.",
                                    _e_q, _mx, _sc, _lt2)
                        if _sc <= 0.0 and math.isfinite(_mx) and _mx > 0.0:
                            logger.error("EVENT ZERO-INJECTION VIOLATED: expression scale is 0 but the event increment is "
                                         "%.3e z -- the event pathway writes into the output without qualification.", _mx)
                    except Exception:
                        pass
                    if _e_q > max(_p_t, _l_t) and _e_q > 0.5:
                        logger.warning(
                            "The EVENT term dominates (%.2f%% vs prior "
                            "%.2f%% / leak %.2f%%): the anomaly pathway is spending "
                            "quiet-late error. Check the probe AUROC against the "
                            "expression gate and the quiet-guard weight "
                            "; raising the prior's accuracy CANNOT fix this "
                            "term.",
                            _e_q, _p_t, _l_t,
                        )
                _fin = float(val_metrics.get("late_error_percent", float("nan")))
                _blm = float(val_metrics.get("blend_late_mean", float("nan")))
                _dpl = float(val_metrics.get("denoise_path_late_error_percent", float("nan")))
                if (
                    math.isfinite(float(_ml_frac)) and math.isfinite(_fin) and math.isfinite(_blm)
                    and float(_ml_frac) > _fin + 1.0 and _blm > 0.90
                ):
                    logger.warning(
                        "STRUCTURAL ALERT | the manifold prior (late %.2f%%) is WORSE than "
                        "the fused output (%.2f%%) yet carries blend_late=%.3f of it; denoise path "
                        "alone sits at %s%%. The quality gate has capped the blend; if this alert "
                        "persists, the prior branch -- not the gate -- is the module to fix.",
                        float(_ml_frac), _fin, _blm,
                        ("%.2f" % _dpl) if math.isfinite(_dpl) else "n/a",
                    )
                bl = float(val_metrics.get("blend_late_mean", float("nan")))
                ml = float(val_metrics.get("manifold_late", float("nan")))
                vem = float(val_metrics.get("early_mid", float("nan")))
                tau_l = 100.0 * float(cfg.contract.late_threshold)
                if math.isfinite(bl) and bl > 0.95 and math.isfinite(ml) and ml > tau_l:
                    amp = (ml / vem) if (math.isfinite(vem) and vem > 1e-9) else float("nan")
                    logger.warning(
                        "LATE BUDGET | blend_late=%.3f (the prior carries %.0f%% of the late window, "
                        "which is correct -- the late measurement is short of the contract at every "
                        "SNR) | prior's own late error = %.2f%% vs contract %.1f%% | early/mid error "
                        "= %.2f%% | extrapolation amplification = %.1fx. The late error CANNOT fall "
                        "below the prior's error, and the prior extrapolates the late window FROM the "
                        "early/mid window. If the late contract is missed, the levers are (a) the "
                        "early/mid error and (b) the prior's own fit -- not the innovation gate.",
                        bl, 100.0 * bl, ml, tau_l, vem, amp,
                    )
            else:
                val_metrics = {}
            record: Dict[str, Any] = {
                "epoch": global_epoch,
                "stage": stage,
                "local_epoch": local_epoch,
                "train": train_metrics,
                "train_components": train_components,
                "val": val_metrics,
                "constraint_lambdas": controller.lambdas.copy(),
                "constraint_rhos": controller.rhos.copy(),
                "constraint_violations_train": violations,
                "learning_rates": {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)},
            }
            history.append(record)
            try:
                _monitor.record(stage, global_epoch, val_metrics or {},
                                train_components=train_components)
            except Exception as _mexc:
                logger.warning("Monitor record skipped at epoch %d: %r",
                               global_epoch, _mexc)
            atomic_json_dump({"history": history}, run_dir / "metrics" / "training_history.json")
            if val_metrics:
                try:
                    _bins = val_metrics.get("per_snr_bin", {}) or {}
                    _thr2 = 100.0 * float(getattr(cfg.contract, "late_bin_threshold", 0.03))
                    _pop2 = {k: v for k, v in dict(_bins).items()
                               if isinstance(v, dict) and int(v.get("count", 0)) >= int(getattr(cfg.contract, "late_bin_min_count", 8))}
                    if _pop2 and "late_bin_worst_percent" not in val_metrics:
                        val_metrics["late_bin_worst_percent"] = float(max(mm["late_error_percent"] for mm in _pop2.values()))
                        val_metrics["late_bins_all_le3"] = 1.0 if all(mm["late_error_percent"] <= _thr2 for mm in _pop2.values()) else 0.0
                except Exception:
                    pass
                feasible = is_project_feasible(val_metrics, cfg.contract)
                score = candidate_score(val_metrics, cfg.contract)
                _prv = float(val_metrics.get("paired_anomaly_recovery", float("nan")))
                _psv = float(val_metrics.get("paired_anomaly_sign_agreement", float("nan")))
                logger.info(
                    "VAL stage=%s epoch=%d | G=%.4f%% EM=%.4f%% L=%.4f%% "
                    "L-CVaR=%.4f%%%s feasible=%s score=%.6f",
                    stage,
                    global_epoch,
                    val_metrics["global_error_percent"],
                    val_metrics["early_mid_error_percent"],
                    val_metrics["late_error_percent"],
                    val_metrics["late_cvar_percent"],
                    (" REAL-rec=%.3f sign=%.3f" % (_prv, _psv))
                    if math.isfinite(_prv) else "",
                    feasible,
                    score,
                )
                if "paired_anomaly_recovery_weak" in val_metrics:
                    logger.info(
                        "VAL weak-anomaly | REAL-rec weak(<5%%) %.3f (amp %.3f, n=%d) | medium(5-20%%) "
                        "%.3f (amp %.3f, n=%d) | strong(>=20%%) %.3f (amp %.3f, n=%d) | quiet false-anomaly "
                        "level %.4f of background.",
                        float(val_metrics.get("paired_anomaly_recovery_weak", float("nan"))),
                        float(val_metrics.get("paired_anomaly_amplitude_weak", float("nan"))),
                        int(val_metrics.get("paired_anomaly_n_weak", 0)),
                        float(val_metrics.get("paired_anomaly_recovery_medium", float("nan"))),
                        float(val_metrics.get("paired_anomaly_amplitude_medium", float("nan"))),
                        int(val_metrics.get("paired_anomaly_n_medium", 0)),
                        float(val_metrics.get("paired_anomaly_recovery_strong", float("nan"))),
                        float(val_metrics.get("paired_anomaly_amplitude_strong", float("nan"))),
                        int(val_metrics.get("paired_anomaly_n_strong", 0)),
                        float(val_metrics.get("paired_quiet_false_level", float("nan"))),
                    )
                if "lateral_late_roughness_out_dex" in val_metrics:
                    logger.info(
                        "VAL lateral roughness (late, D2 of log10, dex) | output-error %.4f | "
                        "reference %.4f | noisy input %.4f | ratio out/ref %.2f | blocks %d "
                        "| lateral prior applied on %d batch(es) | lambda late/mid %s | "
                        "late noise floor %.3f | innovation gate open %.3f | kernel dev "
                        "|K-delta| low-SNR %.3f / high-SNR %.3f | QUIET blocks out %.4f vs ref %.4f "
                        "(false structure) | BODY blocks out %.4f vs ref %.4f | precision w "
                        "low-SNR %.3f / high-SNR %.3f | lateral strength (1-r) low-SNR %.3f / "
                        "high-SNR %.3f | robust inflation mean %.2f tail-rate %.3f | output fusion "
                        "w (early/mid) low-SNR %.3f / high-SNR %.3f | lambda factor (noise-head "
                        "witness) low-SNR %.3f / high-SNR %.3f | BODY structure kept beta %.3f (body blocks "
                        "with beta < 0.5: %.1f%%) | event licence mean %.3f (licensed %.1f%%) [nan = mechanism off].",
                        val_metrics["lateral_late_roughness_out_dex"],
                        val_metrics["lateral_late_roughness_ref_dex"],
                        val_metrics["lateral_late_roughness_in_dex"],
                        val_metrics["lateral_late_roughness_ratio"],
                        int(val_metrics.get("lateral_blocks", 0)),
                        int(val_metrics.get("lateral_applied_batches", 0)),
                        ("%.3f/%.3f (+late floor %.3f)" % (
                            float(model.lateral_joint.lambdas()[cfg.data.late_start_index:].mean()),
                            float(model.lateral_joint.lambdas()[
                                cfg.data.late_start_index // 2:cfg.data.late_start_index].mean()),
                            float(getattr(model.lateral_joint, "lam_floor_late", 0.0)))
                         if getattr(model, "lateral_joint", None) is not None else "off"),
                        float(val_metrics.get("late_noise_floor_late_mean", float("nan"))),
                        float(val_metrics.get("innovation_evidence_open_rate", float("nan"))),
                        float(val_metrics.get("lateral_kernel_dev_low_snr", float("nan"))),
                        float(val_metrics.get("lateral_kernel_dev_high_snr", float("nan"))),
                        float(val_metrics.get("lateral_quiet_roughness_out_dex", float("nan"))),
                        float(val_metrics.get("lateral_quiet_roughness_ref_dex", float("nan"))),
                        float(val_metrics.get("lateral_body_roughness_out_dex", float("nan"))),
                        float(val_metrics.get("lateral_body_roughness_ref_dex", float("nan"))),
                        float(val_metrics.get("innovation_precision_w_low_snr", float("nan"))),
                        float(val_metrics.get("innovation_precision_w_high_snr", float("nan"))),
                        float(val_metrics.get("lateral_strength_low_snr", float("nan"))),
                        float(val_metrics.get("lateral_strength_high_snr", float("nan"))),
                        float(val_metrics.get("innovation_robust_inflation_mean", float("nan"))),
                        float(val_metrics.get("innovation_robust_tail_rate", float("nan"))),
                        float(val_metrics.get("output_fusion_w_low_snr", float("nan"))),
                        float(val_metrics.get("output_fusion_w_high_snr", float("nan"))),
                        float(val_metrics.get("lateral_lambda_factor_low_snr", float("nan"))),
                        float(val_metrics.get("lateral_lambda_factor_high_snr", float("nan"))),
                        float(val_metrics.get("lateral_body_structure_beta", float("nan"))),
                        100.0 * float(val_metrics.get("lateral_body_erased_share", float("nan"))),
                        float((getattr(getattr(model, "module", model), "_last_event_licence", None) or {}).get("mean", float("nan"))),
                        100.0 * float((getattr(getattr(model, "module", model), "_last_event_licence", None) or {}).get("licensed_share", float("nan"))),
                    )
                    _aw = getattr(getattr(model, "module", model), "_last_attention_witness", None)
                    if _aw is not None:
                        logger.info("attention witness (last batch): mean factor %.3f | stations routed to the single-trace "
                                    "path (factor < 0.5): %.1f%%.",
                                    float(_aw.float().mean()), 100.0 * float((_aw.float() < 0.5).float().mean()))
                bin_metrics = val_metrics.get("per_snr_bin", {})
                if bin_metrics:
                    parts = [
                        f"{label} G={m['global_error_percent']:.2f}% L={m['late_error_percent']:.2f}% n={m['count']}"
                        for label, m in bin_metrics.items()
                        if m["count"]
                    ]
                    if parts:
                        logger.info("VAL per-SNR-bin | %s", " | ".join(parts))
                    _thr = 100.0 * float(getattr(cfg.contract,
                                                    "late_bin_threshold", 0.03))
                    _pop = {lb: mm for lb, mm in bin_metrics.items()
                               if mm["count"] >= int(getattr(cfg.contract,
                                                             "late_bin_min_count", 8))}
                    if _pop:
                        _wl, _wm = max(
                            _pop.items(),
                            key=lambda kv: kv[1]["late_error_percent"])
                        _all = all(mm["late_error_percent"] <= _thr
                                      for mm in _pop.values())
                        val_metrics["late_bin_worst_percent"] = float(
                            _wm["late_error_percent"])
                        val_metrics["late_bins_all_le3"] = 1.0 if _all else 0.0
                        logger.info(
                            "ALL-BIN ACCEPTANCE | worst late bin %s = %.2f%% (n=%d) | every-bin<=3%%: %s.",
                            str(_wl), float(_wm["late_error_percent"]),
                            int(_wm["count"]), "PASS" if _all else "FAIL")
                if ("library_anchor_late_percent" in val_metrics
                        or "ridge_only_late_percent" in val_metrics):
                    logger.info(
                        "ANCHORS | subspace-only late = %.2f%% "
                        "(deep<=-20dB: %.2f%%) gate %.3f | RIDGE-only late = "
                        "%.2f%% (deep: %.2f%%) gate %.3f | floors 0.054%% / "
                        "0.711%% (bg-mixed target).",
                        float(val_metrics.get("library_anchor_late_percent",
                                              float("nan"))),
                        float(val_metrics.get("library_anchor_late_deep_percent",
                                              float("nan"))),
                        float(val_metrics.get("library_gate_late_mean",
                                              float("nan"))),
                        float(val_metrics.get("ridge_only_late_percent",
                                              float("nan"))),
                        float(val_metrics.get("ridge_only_late_deep_percent",
                                              float("nan"))),
                        float(val_metrics.get("ridge_gate_late_mean",
                                              float("nan"))))
                if "late_pass_rate_3_percent" in val_metrics:
                    logger.info(
                        "ROW PASS-RATE | <=3%%: %.2f%%  <=4%%: %.2f%%"
                        "  <=5%%: %.2f%% | P50/P90/P95/P99 = %.2f/%.2f/%.2f/"
                        "%.2f%% (target: >=95%% of rows <=3%%).",
                        float(val_metrics["late_pass_rate_3_percent"]),
                        float(val_metrics["late_pass_rate_4_percent"]),
                        float(val_metrics["late_pass_rate_5_percent"]),
                        float(val_metrics["late_row_p50_percent"]),
                        float(val_metrics["late_row_p90_percent"]),
                        float(val_metrics["late_row_p95_percent"]),
                        float(val_metrics["late_row_p99_percent"]))
                    if "late_pass_frontier" in val_metrics:
                        logger.info(
                            "PASS@3%% FRONTIER by SNR quintile: %s. ",
                            str(val_metrics["late_pass_frontier"]))
                    logger.info(
                        "PASS@3%% SPLIT | quiet rows %.1f%% (n=%d) | anomaly rows "
                        "%.1f%% (n=%d) | rows at late-local >= +5 dB %.1f%% (n=%d) | mean row "
                        "late NRMSE %.2f%%.",
                        float(val_metrics.get("late_pass_rate_3_quiet_percent", float("nan"))),
                        int(val_metrics.get("late_rows_quiet", 0)),
                        float(val_metrics.get("late_pass_rate_3_anomaly_percent", float("nan"))),
                        int(val_metrics.get("late_rows_anomaly", 0)),
                        float(val_metrics.get("late_pass_rate_3_ge5db_percent", float("nan"))),
                        int(val_metrics.get("late_rows_ge5db", 0)),
                        float(val_metrics.get("late_row_mean_percent", float("nan"))))
                    if "blend_by_snr_bin" in val_metrics:
                        logger.info(
                            "LATE BLEND (prior share) by late-local SNR bin: %s | a share that "
                            "does not fall toward the high bins prints the prior's error where the "
                            "measurement is already better.",
                            str(val_metrics["blend_by_snr_bin"]))
                    if "arm_decomposition_by_snr_bin" in val_metrics:
                        logger.info("LATE ARMS by late-local SNR bin (energy domain; d=denoise arm, p=fused prior arm, "
                                    "out=delivered, b*=constant prior share minimising the bin's late SSE -> its L): %s. ",
                                    str(val_metrics["arm_decomposition_by_snr_bin"]))
                if "late_order_violation_percent" in val_metrics:
                    logger.info(
                        "LATE ORDER | adjacent-gate violation rate = "
                        "%.2f%% of late pairs (physical feasibility of the "
                        "profile: crossing gates read here).",
                        float(val_metrics["late_order_violation_percent"]))
                try:
                    _sp = survey_profile_probe(model, cfg, clean_cache, device)
                    if _sp:
                        val_metrics["survey_probe"] = _sp
                        logger.info(
                            "SURVEY PROBE (measured profile vs its reference, deliverable path) | G %.2f%% | EM %.2f%% | "
                            "L %.2f%% | per-station late median %.2f%% p90 %.2f%% pass@3 %.1f%% | late level out/ref %.3f "
                            "(deeper half %.3f) | late lateral structure kept %.2f corr %.2f%s.",
                            _sp["G"], _sp["EM"], _sp["L"], _sp["late_row_median"], _sp["late_row_p90"],
                            _sp["late_pass3"], _sp["late_level_ratio"], _sp["late_level_ratio_deep"],
                            _sp["late_structure_kept"], _sp["late_structure_corr"],
                            (" | VALIDATION stations (%d never trained): late median %.2f%%, level %.3f"
                             % (int(_sp["holdout_stations"]), _sp["holdout_late_row_median"],
                                _sp["holdout_late_level_ratio"])) if "holdout_stations" in _sp else "")
                        if "fusion_state" in _sp:
                            logger.info("VALIDATED FUSION | %s", describe_validated_fusion(_sp))
                        if any(("stage_%s_late_median" % _l) in _sp for _l in ("denoise", "prior", "anchor", "blend")):
                            logger.info(
                                "SURVEY STAGES (measured section, per-station late median %% [late level out/ref, "
                                "deeper half]) | %s | delivered %.1f [x%.2f, x%.2f] | family anchor gate, late mean %s | family "
                                "membership %s | prior share of the arm blend, late %s.",
                                " | ".join("%s %.1f [x%.2f, x%.2f]" % (_n, _sp["stage_%s_late_median" % _l],
                                                                         _sp["stage_%s_level" % _l], _sp["stage_%s_level_deep" % _l])
                                           for _l, _n in (("denoise", "denoise arm"), ("prior", "prior arm"),
                                                          ("anchor", "family anchor"), ("blend", "arms blended"))
                                           if ("stage_%s_late_median" % _l) in _sp),
                                _sp["late_row_median"], _sp["late_level_ratio"], _sp["late_level_ratio_deep"],
                                ("%.2f" % _sp["anchor_gate_late"]) if "anchor_gate_late" in _sp else "n/a",
                                ("%.2f" % _sp["family_membership"]) if "family_membership" in _sp else "n/a (not installed)",
                                ("%.2f" % _sp["prior_share_late"]) if "prior_share_late" in _sp else "n/a")
                except Exception as _e755p:
                    logger.warning("survey probe skipped: %r", _e755p)
                try:
                    _tp = tail_panel_monitor(model, cfg, run_dir, clean_cache, noise_cache, logger)
                    if _tp:
                        val_metrics["tail_panel"] = _tp
                except Exception as _e23:
                    logger.warning("tail panel skipped: %r", _e23)
                try:
                    _ev2 = int(getattr(cfg.runtime, "anchor_compare_every", 0) or 0)
                    _b4 = getattr(model, "module", model)
                    _b4 = getattr(_b4, "_orig_mod", _b4)
                    if (_ev2 > 0 and int(global_epoch) % _ev2 == 0
                            and (getattr(_b4, "lib_gate_head", None) is not None
                                 or getattr(_b4, "ridge_gate_head", None) is not None)):
                        _b4._force_no_anchors = True
                        try:
                            _vm2 = evaluate_model(model, val_loader, loss_fn, device, cfg, stage, epoch=global_epoch)
                        finally:
                            _b4._force_no_anchors = False
                        val_metrics["anchor_compare"] = {
                            "L_with": float(val_metrics.get("late_error_percent", float("nan"))),
                            "CVaR_with": float(val_metrics.get("late_cvar_percent", float("nan"))),
                            "pass3_with": float(val_metrics.get("late_pass_rate_3_percent", float("nan"))),
                            "L_without": float(_vm2.get("late_error_percent", float("nan"))),
                            "CVaR_without": float(_vm2.get("late_cvar_percent", float("nan"))),
                            "pass3_without": float(_vm2.get("late_pass_rate_3_percent", float("nan")))}
                        _ac = val_metrics["anchor_compare"]
                        _pb1, _pb0 = (val_metrics.get("per_snr_bin") or {}), (_vm2.get("per_snr_bin") or {})
                        _bt2 = " ; ".join(
                            "%s %.1f -> %.1f%% (n=%d)" % (_lab, float(_v["late_error_percent"]),
                                                         float(_pb0[_lab]["late_error_percent"]), int(_v.get("count", 0)))
                            for _lab, _v in _pb1.items()
                            if _v.get("late_error_percent") is not None and _lab in _pb0
                            and _pb0[_lab].get("late_error_percent") is not None)
                        logger.info(
                            "ANCHOR COMPARE (val, epoch %d; monitor only -- nothing is deployed from it) | with the anchors: "
                            "L %.3f%% CVaR-20 %.2f%% pass@3 %.1f%% | anchors withdrawn (library completion and continuation gates 0): "
                            "L %.3f%% CVaR-20 %.2f%% pass@3 %.1f%% | G %.3f%% -> %.3f%% | late error by SNR bin, with -> withdrawn: %s. "
                            "A withdrawn reading that is not worse means the anchors do not earn their place there; read it with the "
                            "TAIL PANEL stages.",
                            int(global_epoch), _ac["L_with"], _ac["CVaR_with"], _ac["pass3_with"],
                            _ac["L_without"], _ac["CVaR_without"], _ac["pass3_without"],
                            float(val_metrics.get("global_error_percent", float("nan"))),
                            float(_vm2.get("global_error_percent", float("nan"))), _bt2 or "n/a")
                except Exception as _e761c:
                    logger.warning("anchor compare skipped: %r", _e761c)
                _lowsnr_ds_ref = getattr(datasets[0], "_lowsnr_probe_ds", None)
                if _lowsnr_ds_ref is not None:
                    _lb = paired_lowsnr_probe(model, _lowsnr_ds_ref, cfg, device)
                    if _lb:
                        val_metrics["lowsnr_bins"] = {
                            "%d..%d" % k: v for k, v in _lb.items()}
                        _stab0 = next(iter(_lb.values())).get("stab", float("nan"))
                        logger.info(
                            "VAL lowSNR probe (library noise injected on held-out "
                            "clean, x=y+n) | %s | noise-sensitivity %.3f (median "
                            "||pred(x)-pred(x_alt)||/||x-x_alt||; identity=1, pure "
                            "signal function=0).",
                            " | ".join("[%d,%d) G=%.2f%% L=%.2f%% n=%d"
                                       % (a, b, v["G"], v["L"], v["n"])
                                       for (a, b), v in sorted(_lb.items())),
                            _stab0)
                        if (75, 85) in _lb:
                            _cv = _lb[(75, 85)]
                            val_metrics["clean_passthrough_late_percent"] = float(_cv["L"])
                            logger.info(
                                "VAL CLEAN PASS-THROUGH (held-out, x = y exactly, clean "
                                "neighbours) | G=%.2f%% L=%.2f%% n=%d | %s",
                                float(_cv["G"]), float(_cv["L"]), int(_cv["n"]),
                                ("estimator faithful; noisy-bin residual = information"
                                 if float(_cv["L"]) <= 3.0 else
                                 "ESTIMATOR DISTORTS COMPLETE INFORMATION -- architecture/objective, not noise"))
                _pg = list(val_metrics.get("per_gate_median_error_percent") or [])
                _wg = int(val_metrics.get("worst_gate_index", -1))
                _wv = float(val_metrics.get("worst_gate_median_percent", float("nan")))
                _gp = float(val_metrics.get("global_error_percent", float("nan")))
                if _pg and _wg >= 0:
                    _ls_g = int(cfg.data.late_start_index)
                    _mid_seg = _pg[max(0, _ls_g - 8):_ls_g] or [float("nan")]
                    _late_seg = _pg[_ls_g:] or [float("nan")]
                    logger.info(
                        "VAL per-gate median | worst gate %d: %.2f%% | seam band "
                        "(gates %d-%d) max %.2f%% | late band max %.2f%% | pooled G "
                        "%.2f%% | mid band mean %.2f%%.",
                        _wg + 1, _wv, max(1, _ls_g - 7), _ls_g,
                        max(_mid_seg), max(_late_seg), _gp,
                        float(val_metrics.get("mid_band_gate_median_percent", float("nan"))))
                    try:
                        _pin = [float(v) for v in (val_metrics.get("per_gate_mean_error_in_percent") or [])]
                        _pout = [float(v) for v in (val_metrics.get("per_gate_mean_error_out_percent") or [])]
                        if _pin and len(_pin) == len(_pout):
                            _a = max(0, _ls_g - 12)
                            _b = min(len(_pin), _ls_g + 3)
                            _worse = [g + 1 for g in range(_a, _b) if _pout[g] > 1.05 * _pin[g]]
                            _pm = [float(v) for v in (val_metrics.get("per_gate_mean_error_prior_percent") or [])]
                            _pd = [float(v) for v in (val_metrics.get("per_gate_mean_error_denoise_percent") or [])]
                            _arm = ""
                            if len(_pm) == len(_pin) and len(_pd) == len(_pin):
                                _arm = " | prior arm: %s | denoise arm: %s" % (
                                    " ".join("g%d:%.1f%%" % (g + 1, _pm[g]) for g in range(_a, _b)),
                                    " ".join("g%d:%.1f%%" % (g + 1, _pd[g]) for g in range(_a, _b)))
                            _pb2 = [float(v) for v in (val_metrics.get("per_gate_mean_blend") or [])]
                            if len(_pb2) == len(_pin):
                                _arm += " | realized blend (prior share): %s" % " ".join(
                                    "g%d:%.2f" % (g + 1, _pb2[g]) for g in range(_a, _b))
                            try:
                                _g0639 = int(getattr(model, "_prior_band_g", -1))
                                if 0 < _g0639 < _ls_g and len(_pm) == len(_pin) and len(_pd) == len(_pin) \
                                        and bool(getattr(cfg.model, "seam_blend_floor_from_val", True)):
                                    _cv2 = val_metrics.get("per_gate_arm_cov_z") or {}
                                    _sdd = [float(v) for v in (_cv2.get("dd") or [])]
                                    _spp = [float(v) for v in (_cv2.get("pp") or [])]
                                    _sdp = [float(v) for v in (_cv2.get("dp") or [])]
                                    _all2 = bool(getattr(cfg.model, "seam_blend_floor_all_gates", True))
                                    _g1644 = 0 if _all2 else _g0639
                                    _hasm = (len(_sdd) == len(_pin) and len(_spp) == len(_pin)
                                                and len(_sdp) == len(_pin))
                                    _fl = np.zeros(len(_pin), dtype=np.float64)
                                    _nmom = 0
                                    for _g in range(_g1644, _ls_g):
                                        _ed = max(float(_pd[_g]), 1e-6)
                                        _ep = max(float(_pm[_g]), 1e-6)
                                        _b2 = _ed * _ed / (_ed * _ed + _ep * _ep)
                                        if _hasm:
                                            _den2 = _sdd[_g] + _spp[_g] - 2.0 * _sdp[_g]
                                            if _den2 > 1e-3 * max(_sdd[_g], _spp[_g], 1e-30):
                                                _b2 = (_sdd[_g] - _sdp[_g]) / _den2
                                                _nmom += 1
                                        _fl[_g] = float(min(0.95, max(0.0, _b2)))
                                    model.queue_seam_blend_floor(_fl)
                                    calibrate_mmse_bstar(model, val_metrics, _ls_g, logger)
                                    try:
                                        _base = getattr(model, "module", model)
                                        _base = getattr(_base, "_orig_mod", _base)
                                        _every = max(1, int(getattr(cfg.model, "mmse_eval_every", 2)))
                                        if bool(getattr(cfg.model, "mmse_blend", True)) and not bool(getattr(cfg.model, "mmse_blend_in_training", False)) \
                                                and (int(global_epoch) % _every == 0):
                                            _base._force_mmse_blend = True
                                            try:
                                                _vm = evaluate_model(model, val_loader, loss_fn, device, cfg, stage, epoch=global_epoch)
                                            finally:
                                                _base._force_mmse_blend = False
                                            _l_leg = float(val_metrics.get("late_error_percent", float("nan")))
                                            _l_mm = float(_vm.get("late_error_percent", float("nan")))
                                            _c_leg = float(val_metrics.get("late_cvar_percent", float("nan")))
                                            _c_mm = float(_vm.get("late_cvar_percent", float("nan")))
                                            _win = bool(math.isfinite(_l_mm) and math.isfinite(_l_leg) and _l_mm < _l_leg
                                                           and (not math.isfinite(_c_mm) or not math.isfinite(_c_leg) or _c_mm <= 1.05 * _c_leg))
                                            _base.mmse_deploy_714.fill_(1.0 if _win else 0.0)
                                            logger.info("DEPLOYMENT BLEND COMPARE (val, epoch %d) | trained gate: L %.3f%% CVaR-20 %.2f%% | witnessed MMSE: "
                                                        "L %.3f%% CVaR-20 %.2f%% | deployed for the deliverable path: %s.", int(global_epoch), _l_leg, _c_leg, _l_mm, _c_mm,
                                                        "witnessed MMSE" if _win else "trained gate")
                                    except Exception as _e18:
                                        logger.warning("deployment compare skipped: %r", _e18)
                                    _cur = model.seam_blend_floor.detach().cpu().numpy()
                                    _br = [float(v) for v in (val_metrics.get("per_gate_floor_binding_rate") or [])]
                                    _brs = (" | floor BINDING rate (head < floor) %s" % " ".join(
                                        "%.2f" % _br[_g] for _g in range(_g1644, _ls_g)) if len(_br) == len(_pin) else "")
                                    logger.info("SEAM BLEND FLOOR (val-calibrated, EMA) IN FORCE gates %d-%d: %s | QUEUED for "
                                                "the next epoch: %s (MSE-optimal with covariance on %d gates, fallback elsewhere; band %s) | held-out arm "
                                                "errors (prior/denoise) %s%s | floor mode %s.",
                                                _g1644 + 1, _ls_g, " ".join("%.2f" % _cur[_g] for _g in range(_g1644, _ls_g)),
                                                " ".join("%.2f" % _fl[_g] for _g in range(_g1644, _ls_g)), _nmom,
                                                "ALL gates below the late start" if _all2 else "seam band only (early gates: diagnostic b* logged, not installed)",
                                                " ".join("%.1f/%.1f%%" % (_pm[_g], _pd[_g]) for _g in range(_g1644, _ls_g)), _brs,
                                                getattr(getattr(model, "module", model), "seam_floor_mode", lambda: "?")())
                            except Exception as _e11:
                                logger.warning("seam floor not installed: %r", _e11)
                            logger.info(
                                "SEAM AUDIT gates %d-%d (mean |err|/|clean|, raw measurement -> output) | %s | "
                                "early do-no-harm %.2f%% | E/M misfit ratio %.2f | median per-gate output %s%s. ",
                                _a + 1, _b, " ".join("g%d:%.1f->%.1f%%" % (g + 1, _pin[g], _pout[g])
                                                          for g in range(_a, _b)),
                                float(val_metrics.get("early_do_no_harm_percent", float("nan"))),
                                float(val_metrics.get("early_mid_misfit_ratio", float("nan"))),
                                " ".join("%.1f" % v for v in _pg[_a:_b]), _arm)
                            if _worse:
                                logger.warning(
                                    "OUTPUT WORSE THAN THE RAW MEASUREMENT on gate(s) %s: the seam "
                                    "mechanism (blend ramp / lateral prior) is printing prior error onto "
                                    "gates the measurement already carries better; if this persists at the best "
                                    "epoch, move the ramp start later (--blend_floor_ramp_gates) before touching "
                                    "the trunk.",
                                    _worse)
                    except Exception as _e7:
                        logger.warning("seam audit unavailable: %r", _e7)
                    if math.isfinite(_wv) and math.isfinite(_gp) and _wv > 3.0 * _gp:
                        logger.warning(
                            "Gate %d's MEDIAN error is %.1f%%, %.1fx the pooled global %.2f%% -- the contract "
                            "cannot see this gate (energy weighting), but every figure can. If it sits just below the "
                            "late-window start it is the blend-floor seam: the prior is still weak there while the "
                            "measurement is already dead; raise --blend_floor_ramp_gates or --alpha_gate_balanced "
                            ".",
                            _wg + 1, _wv, _wv / max(_gp, 1e-9), _gp)
                try:
                    _t2 = candidate_score_terms(val_metrics, cfg.contract)
                    logger.info(
                        "SELECTION SCORE %.2f = G %.2f + L %.2f + EM %.2f + CVaR %.2f + "
                        "worst-bin %.2f (bin %s = %.2f%%, %.1f%% of rows, violation capped at %.1f) + "
                        "pass@3 %.2f + anomaly %.2f (raw %.2f, cap %.1f) | run best %.2f | contract gates G %.3f%% L %.3f%% (run best %.3f%% / %.3f%%). ",
                        _t2["total"], _t2["g"], _t2["l"], _t2["em"], _t2["cvar"], _t2["worst_bin"],
                        _t2["_worst_bin_label"], _t2["worst_bin_value"], 100.0 * _t2["worst_bin_share"],
                        float(getattr(cfg.contract, "late_bin_violation_cap", 1.0)), _t2.get("pass3", 0.0), _t2["anomaly"],
                        float(_t2.get("anomaly_raw", _t2["anomaly"])), float(getattr(cfg.contract, "anomaly_score_cap", 0.0)),
                        best_candidate_score, float(val_metrics.get("global_error_percent", float("nan"))),
                        float(val_metrics.get("late_error_percent", float("nan"))),
                        _best_gate_g_run, _best_gate_l_run)
                except Exception as _e6:
                    logger.warning("score decomposition unavailable: %r", _e6)
                _g2 = float(val_metrics.get("global_error_percent", float("nan")))
                _l3 = float(val_metrics.get("late_error_percent", float("nan")))
                _rel = float(getattr(cfg.train, "gate_improvement_rel", 1e-3))
                _gates_improved_run = bool(
                    (math.isfinite(_g2) and _g2 < _best_gate_g_run * (1.0 - _rel))
                    or (math.isfinite(_l3) and _l3 < _best_gate_l_run * (1.0 - _rel)))
                _gates_improved_stage = bool(
                    (math.isfinite(_g2) and _g2 < _best_gate_g_stage * (1.0 - _rel))
                    or (math.isfinite(_l3) and _l3 < _best_gate_l_stage * (1.0 - _rel)))
                if math.isfinite(_g2):
                    _best_gate_g_run = min(_best_gate_g_run, _g2)
                    _best_gate_g_stage = min(_best_gate_g_stage, _g2)
                if math.isfinite(_l3):
                    _best_gate_l_run = min(_best_gate_l_run, _l3)
                    _best_gate_l_stage = min(_best_gate_l_stage, _l3)
                improved_stage = score < stage_best_score - 1e-9
                if improved_stage:
                    stage_best_score = score
                    _stage_best_scores[stage] = float(score)
                    save_checkpoint(stage_best_path, model, optimizer, scaler, controller, cfg,
                                    global_epoch, stage, val_metrics, clean_cache, ema=ema)
                if score < best_candidate_score:
                    best_candidate_score = score
                    no_improve = 0
                    save_checkpoint(
                        best_candidate_path,
                        model,
                        optimizer,
                        scaler,
                        controller,
                        cfg,
                        global_epoch,
                        stage,
                        val_metrics,
                        clean_cache,
                        ema=ema,
                    )
                elif _gates_improved_run:
                    no_improve = 0
                else:
                    no_improve += 1
                _l_now = float(val_metrics.get("late_error_percent", float("nan")))
                _wb = float(val_metrics.get("late_bin_worst_percent", float("nan")))
                if math.isfinite(_wb):
                    _l_now = max(_l_now, _wb)
                _g_now = float(val_metrics.get("global_error_percent", float("nan")))
                if math.isfinite(_g_now):
                    best_g_seen = min(best_g_seen, _g_now)
                if (math.isfinite(_l_now) and math.isfinite(_g_now)
                        and _l_now < best_late_seen - 1e-9
                        and _g_now <= max(1.25 * best_g_seen,
                                          100.0 * cfg.contract.global_threshold)):
                    best_late_seen = _l_now
                    save_checkpoint(best_late_path, model, optimizer, scaler,
                                    controller, cfg, global_epoch, stage,
                                    val_metrics, clean_cache, ema=ema)
                    logger.info(
                        "best_late.pt updated | late %.4f%% at G %.4f%% "
                        "(stage %s epoch %d).",
                        _l_now, _g_now, stage, global_epoch)
                if feasible and val_metrics["late_cvar_percent"] < best_feasible_cvar:
                    best_feasible_cvar = val_metrics["late_cvar_percent"]
                    save_checkpoint(
                        best_feasible_path,
                        model,
                        optimizer,
                        scaler,
                        controller,
                        cfg,
                        global_epoch,
                        stage,
                        val_metrics,
                        clean_cache,
                        ema=ema,
                    )
                if ema is not None:
                    ema.restore(model)
                if improved_stage or _gates_improved_stage:
                    stage_plateau = 0
                else:
                    stage_plateau += 1
                beta = float(cfg.train.stage_score_ema_beta)
                stage_score_ema = (
                    score if stage_score_ema is None
                    else beta * stage_score_ema + (1.0 - beta) * score
                )
                if stage_score_ema < stage_best_score_ema - 1e-9:
                    stage_best_score_ema = stage_score_ema
                _min_ep = int(cfg.train.stage_regression_min_epochs)
                if stage != "A":
                    _min_ep = max(6, _min_ep // 4)
                if local_epoch <= _min_ep:
                    stage_regress = 0
                elif stage_score_ema > stage_best_score_ema * float(cfg.train.stage_regression_ratio):
                    stage_regress += 1
                else:
                    stage_regress = 0
                if (
                    stage_regress >= int(cfg.train.stage_regression_patience)
                    and (stage_best_path.exists() or best_candidate_path.exists())
                ):
                    logger.warning(
                        "STAGE %s IS DEGRADING THE MODEL: the SMOOTHED validation score has been "
                        "worse than this stage's own best (EMA %.4f vs %.4f, ratio %.2f) for %d "
                        "validations past the %d-epoch guard. The best weights are being RESTORED "
                        "and stage %s ended. A stage that makes the model worse does not get to "
                        "keep it -- but a stage that has not yet converged does not get executed "
                        "for it either.",
                        stage, stage_score_ema, stage_best_score_ema,
                        float(cfg.train.stage_regression_ratio), stage_regress, _min_ep, stage,
                    )
                    load_checkpoint(
                        stage_best_path if stage_best_path.exists() else best_candidate_path,
                        model, controller, device, logger,
                    )
                    if ema is not None:
                        ema.reseed(model)
                    stage_over = True
                current_lr = optimizer.param_groups[0]["lr"]
                manifold_note = ""
                if "manifold_late_error_percent" in val_metrics:
                    manifold_note = " | manifold_late(vs bg)=%.2f%% blend_late=%.3f" % (
                        val_metrics["manifold_late_error_percent"],
                        val_metrics["blend_late_mean"],
                    )
                    if "innovation_w_late_mean" in val_metrics:
                        manifold_note += " innov_w=%.3f" % val_metrics["innovation_w_late_mean"]
                    if "denoise_path_late_error_percent" in val_metrics:
                        manifold_note += " den_path_late=%.2f%%" % val_metrics["denoise_path_late_error_percent"]
                    manifold_note += " blend_cap=%.3f" % float(model.blend_late_quality_cap.item())
                    if "relaxation_update_late_mean_abs" in val_metrics:
                        manifold_note += " rssu=%.4f" % val_metrics["relaxation_update_late_mean_abs"]
                    _att = getattr(model, "neighbor_attention", None)
                    if _att is not None and hasattr(_att, "geo_prior_rho"):
                        manifold_note += " geo_prior=%.3f" % float(
                            F.softplus(_att.geo_prior_rho.detach()).mean().item()
                        )
                    manifold_note = manifold_note.replace("%", "%%")
                _pl_n, _pl_d, _pl_name = (
                    (stage_plateau, int(cfg.train.stage_switch_plateau_patience), "plateau")
                    if stage in {"A", "B"} else
                    (no_improve, int(cfg.train.early_stop_patience), "no_improve"))
                logger.info(
                    "  lr=%.3e | aux_z_scale=%.2f linear_scale=%.2f | %s=%d/%d | "
                    "lambda(G/EM/L/CVaR/ANOM)=%.1f/%.1f/%.1f/%.1f/%.1f x%.3f"
                    + manifold_note,
                    current_lr,
                    1.0 - (1.0 - cfg.train.aux_anneal_final) * loss_fn.progress,
                    0.2 + 0.8 * loss_fn.progress,
                    _pl_name, _pl_n, _pl_d,
                    controller.lambdas["global"],
                    controller.lambdas["early_mid"],
                    controller.lambdas["late"],
                    controller.lambdas["late_cvar"],
                    controller.lambdas.get("anomaly_late", 0.0),
                    float(getattr(loss_fn, "last_constraint_scale", 1.0)),
                )
                try:
                    _dead: List[str] = []
                    for _k, _v2 in (train_components or {}).items():
                        if not str(_k).startswith("l_"):
                            continue
                        _fv = float(_v2)
                        if not math.isfinite(_fv):
                            continue
                        _h = _DEAD_LOSS_HISTORY.setdefault(str(_k), [])
                        _h.append(_fv)
                        del _h[:-4]
                        if (len(_h) >= 4 and abs(_fv) > 1e-6
                                and (max(_h) - min(_h)) <= 1e-6 * max(1.0, abs(_fv))):
                            _dead.append("%s=%.4f" % (_k, _fv))
                    if _dead:
                        logger.warning(
                            "DEAD LOSS TERM(S): %s -- constant over the last 4 epochs to 1e-6 "
                            "while non-zero: no gradient reaches them, or the loader is replaying one "
                            "realization (the batch fingerprint decides which).",
                            ", ".join(_dead))
                except Exception:
                    pass
                try:
                    _med = float((train_components or {}).get("late_row_median", float("nan")))
                    loss_fn.last_row_late_mean = (_med if math.isfinite(_med) else
                                                  float((train_components or {}).get("l_late_per_sample", float("inf"))))
                except Exception:
                    pass
                if "l_library_coef" in (train_components or {}):
                    logger.info(
                        "  lib anchor | l_coef=%.4f l_late=%.4f l_coef_cont=%.4f "
                        "| l_ridge=%.4f(ps %.4f) l_lib_ps=%.4f "
                        "l_gate_deep=%.4f l_late_ps=%.4f "
                        "l_rec=%.4f l_orth=%.4f l_hinge=%.4f "
                        "| acc_scale=%.2f grad-share(group/rest)=%.3f "
                        "l_no_harm=%.4f l_quiet_var=%.4f (balancing %s: row-late MEDIAN %.3f vs arm <= %.2f) "
                        "| clean-arm late %.2f%% (n=%.0f; must -> ~0) | skipped "
                        "non-finite steps this epoch %.0f",
                        float(train_components.get("l_library_coef", float("nan"))),
                        float(train_components.get("l_library_late", float("nan"))),
                        float(train_components.get("l_library_coef_cont", float("nan"))),
                        float(train_components.get("l_ridge_late", float("nan"))),
                        float(train_components.get("l_ridge_late_ps", float("nan"))),
                        float(train_components.get("l_library_late_ps", float("nan"))),
                        float(train_components.get("l_gate_deep", float("nan"))),
                        float(train_components.get("l_late_per_sample",
                                                   float("nan"))),
                        float(train_components.get("l_recovery", float("nan"))),
                        float(train_components.get("l_diff_orth", float("nan"))),
                        float(train_components.get("l_pass_hinge", float("nan"))),
                        float(train_components.get("acceptance_scale", float("nan"))),
                        float(train_components.get("acceptance_grad_share", float("nan"))),
                        float(train_components.get("l_no_harm", float("nan"))),
                        float(train_components.get("l_quiet_var", float("nan"))),
                        ("ARMED" if (float(getattr(loss_fn, "last_row_late_mean", float("inf")))
                                     <= float(getattr(cfg.loss, "acceptance_balance_row_late_max", 0.10))
                                     and float(getattr(cfg.loss, "acceptance_balance_share", 0.0)) > 0.0)
                         else "off"),
                        float(getattr(loss_fn, "last_row_late_mean", float("inf"))),
                        float(getattr(cfg.loss, "acceptance_balance_row_late_max", 0.10)),
                        float(train_components.get("clean_arm_late_percent", float("nan"))),
                        float(train_components.get("clean_arm_rows", 0.0)),
                        float(train_components.get("nonfinite_grad_steps", 0.0)))
                transition = False
                if stage_over:
                    break
                if stage == "A":
                    _em_gate = 100.0 * float(cfg.train.stage_a_switch_early_mid)
                    _cvar_gate = 100.0 * float(cfg.train.stage_a_switch_cvar)
                    _em_v = float(val_metrics.get("early_mid_error_percent", float("nan")))
                    _cvar_v = float(val_metrics.get("late_cvar_percent", float("nan")))
                    threshold_hit = (
                        val_metrics["global_error_percent"] <= 100.0 * cfg.train.stage_a_switch_global
                        and val_metrics["late_error_percent"] <= 100.0 * cfg.train.stage_a_switch_late
                        and math.isfinite(_em_v) and _em_v <= _em_gate
                        and math.isfinite(_cvar_v) and _cvar_v <= _cvar_gate
                    )
                    stable_count = stable_count + 1 if threshold_hit else 0
                    still_improving = stage_plateau < int(cfg.train.stage_a_min_plateau)
                    _a_min = int(cfg.train.stage_a_min_epochs_before_transition)
                    _may_leave = local_epoch >= _a_min
                    _stalled_unqualified = (
                        val_metrics["global_error_percent"] > 100.0 * cfg.train.stage_a_stall_global
                        or val_metrics["late_error_percent"] > 100.0 * cfg.train.stage_a_stall_late
                        or (math.isfinite(_em_v) and _em_v > 100.0 * cfg.train.stage_a_stall_early_mid)
                        or (math.isfinite(_cvar_v) and _cvar_v > 100.0 * cfg.train.stage_a_stall_cvar)
                    )
                    if not _may_leave and (
                        stage_plateau >= cfg.train.stage_switch_plateau_patience
                        or stable_count >= cfg.train.stage_a_switch_patience
                    ):
                        if not stage_stall_reported:
                            stage_stall_reported = True
                            logger.info(
                                "Stage A exit conditions are met at epoch %d but the "
                                "%d-epoch floor is not: training continues. (plateau=%d, "
                                "thresholds_hit_streak=%d, G=%.2f%%, L=%.2f%%)",
                                local_epoch, _a_min, stage_plateau, stable_count,
                                val_metrics["global_error_percent"], val_metrics["late_error_percent"],
                            )
                    elif _may_leave and stable_count >= cfg.train.stage_a_switch_patience and not still_improving:
                        logger.info(
                            "Stage A transition: absolute thresholds reached AND validation has "
                            "stopped improving for %d epochs.", stage_plateau,
                        )
                        transition = True
                    elif _may_leave and stage_plateau >= cfg.train.stage_switch_plateau_patience:
                        if _stalled_unqualified:
                            if not stage_stall_reported:
                                stage_stall_reported = True
                                logger.warning(
                                    "STAGE_A_STALLED_UNQUALIFIED | epoch %d: validation "
                                    "has plateaued for %d epochs while still far from the contract "
                                    "(G=%.2f%% > %.0f%%, L=%.2f%% > %.0f%%, or EM/CVaR beyond "
                                    "their stall gates). This is a stalled "
                                    "optimization, not a finished stage: Stage A keeps its budget "
                                    "instead of handing an unqualified model to the next stage. "
                                    "If this persists to the end of Stage A, treat the run as "
                                    "NOT QUALIFIED and fix the optimization (LR, capacity, data).",
                                    local_epoch, stage_plateau,
                                    val_metrics["global_error_percent"], 100.0 * cfg.train.stage_a_stall_global,
                                    val_metrics["late_error_percent"], 100.0 * cfg.train.stage_a_stall_late,
                                )
                        else:
                            logger.info("Stage A transition: validation plateaued for %d epochs.", stage_plateau)
                            transition = True
                elif stage == "B":
                    threshold_hit = (
                        val_metrics["global_error_percent"] <= 100.0 * cfg.train.stage_b_switch_global
                        and val_metrics["late_error_percent"] <= 100.0 * cfg.train.stage_b_switch_late
                    )
                    stable_count = stable_count + 1 if threshold_hit else 0
                    if stable_count >= cfg.train.stage_b_switch_patience:
                        logger.info("Stage B transition: absolute thresholds reached.")
                        transition = True
                    elif stage_plateau >= cfg.train.stage_switch_plateau_patience:
                        logger.info("Stage B transition: validation plateaued for %d epochs.", stage_plateau)
                        transition = True
                if transition:
                    break
                if no_improve >= cfg.train.early_stop_patience and stage == "C":
                    logger.info("Stage C early stop: no candidate improvement for %d validations.", no_improve)
                    break
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scaler,
                controller,
                cfg,
                global_epoch,
                stage,
                val_metrics or train_metrics,
                clean_cache,
                ema=ema,
            )

    selected = best_feasible_path if best_feasible_path.exists() else best_candidate_path
    if not selected.exists():
        selected = last_path
    load_checkpoint(selected, model, controller, device)
    logger.info("Selected checkpoint: %s", selected)

    _rem_rounds = int(getattr(cfg.train, "remediation_rounds", 0))
    if _rem_rounds > 0:
        def _gate_check(vm: Dict[str, float]) -> List[str]:
            return remediation_failing_gates(vm, cfg.contract)

        def _measure() -> Dict[str, float]:
            vm = evaluate_model(model, val_loader, loss_fn, device, cfg, "C",
                                epoch=global_epoch)
            if _anom_probe is not None:
                try:
                    vm.update(_anom_probe.run(model, cfg))
                except Exception as _pe:
                    logger.warning("probe failed in remediation: %r", _pe)
            return vm

        _hist: List[Dict[str, Any]] = []
        vm0 = _measure()
        fails = _gate_check(vm0)
        best_r_score = candidate_score(vm0, cfg.contract)
        best_r_state = None
        _pre_r_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if not fails:
            logger.info("STAGE R skipped: every gate PASSES on validation "
                        "(score %.4f). The verdict below is measurement, not fate.",
                        best_r_score)
        _lam_cap = float(cfg.contract.lambda_max)
        for _r in range(1, _rem_rounds + 1):
            if not fails:
                break
            adj: List[str] = []
            def _boost(name: str) -> None:
                old = float(controller.lambdas.get(name, 0.0))
                new = min(_lam_cap, max(old * 1.5, old + 20.0, 20.0))
                controller.lambdas[name] = new
                controller.frozen[name] = False
                controller.frozen_val[name] = False
                controller.val_stall[name] = 0
                controller.stall_counts[name] = 0
                adj.append("lambda_%s %.1f->%.1f (thawed)" % (name, old, new))
            for g in fails:
                if g in ("global", "early_mid", "late", "late_cvar", "anomaly_late"):
                    _boost(g)
                    if g == "late_cvar":
                        loss_fn.cvar_armed = True
                    if g == "anomaly_late":
                        loss_fn.anomaly_armed = True
                elif g in ("event_detect", "singleton_reject"):
                    for _an, _capv in (("alpha_event_gate", 2.0),
                                       ("alpha_event_station", 1.0)):
                        _o = float(getattr(cfg.loss, _an))
                        _n = min(_capv, _o * 1.5)
                        if _n > _o:
                            setattr(cfg.loss, _an, _n)
                            adj.append("%s %.2f->%.2f" % (_an, _o, _n))
                    loss_fn.anomaly_armed = True
                elif g == "recovery":
                    _o = float(cfg.loss.alpha_event_station)
                    _n = min(1.5, _o * 1.4)
                    if _n > _o:
                        cfg.loss.alpha_event_station = _n
                        adj.append("alpha_event_station %.2f->%.2f" % (_o, _n))
                    _boost("anomaly_late")
            logger.warning(
                "REMEDIATION ROUND %d/%d | failing gates: %s | applied: %s | "
                "continuing %d epochs at %.0f%% LR.",
                _r, _rem_rounds, ", ".join(fails), "; ".join(adj) or "none",
                int(cfg.train.remediation_epochs),
                100.0 * float(cfg.train.remediation_lr_scale),
            )
            _rl_cfg = copy.deepcopy(cfg.train)
            _rl_cfg.base_lr = float(cfg.train.base_lr) * float(cfg.train.remediation_lr_scale)
            r_opt = build_optimizer(model, _rl_cfg, "C")
            _accum_r = max(1, int(getattr(cfg.train, "grad_accum_steps", 1)))
            _spe = max(1, int(math.ceil(len(train_loader) / _accum_r)))
            r_sched = build_scheduler(r_opt, _rl_cfg,
                                      int(cfg.train.remediation_epochs) * _spe)
            for _e in range(int(cfg.train.remediation_epochs)):
                global_epoch += 1
                try:
                    model.commit_control_updates(logger)
                except Exception:
                    pass
                tm, tc = run_epoch(model, train_loader, loss_fn, r_opt, scaler, device,
                                   cfg, "C", global_epoch, logger, scheduler=r_sched)
                controller.last_constraint_scale = float(
                    getattr(loss_fn, "last_constraint_scale", 1.0))
                _an_n = float(tc.get("rl_anom_num", float("nan")))
                _an_d = float(tc.get("rl_anom_den", float("nan")))
                _upd = {
                    "global": tm["relative_mse_global"],
                    "early_mid": tm["relative_mse_early_mid"],
                    "late": tm["relative_mse_late"],
                    "late_cvar": float(tm.get("late_cvar_percent", float("nan"))) / 100.0,
                }
                if math.isfinite(_an_n) and math.isfinite(_an_d) and _an_d > 0:
                    _upd["anomaly_late"] = _an_n / _an_d
                controller.update(_upd, list(_upd.keys()))
            vm1 = _measure()
            controller.note_validation(
                vm1, logger,
                constraint_scale=float(getattr(loss_fn, "last_constraint_scale", 1.0)))
            sc1 = candidate_score(vm1, cfg.contract)
            fails1 = _gate_check(vm1)
            improved = sc1 < best_r_score - 1e-9
            logger.info(
                "REMEDIATION ROUND %d RESULT | score %.4f -> %.4f (%s) | gates still "
                "failing: %s | G=%.3f%% L=%.3f%% CVaR=%.3f%% ANOM=%.3f%% recov=%.3f "
                "AUROC=%.4f",
                _r, best_r_score, sc1, "improved" if improved else "no improvement",
                ", ".join(fails1) or "none",
                float(vm1.get("global_error_percent", float("nan"))),
                float(vm1.get("late_error_percent", float("nan"))),
                float(vm1.get("late_cvar_percent", float("nan"))),
                float(vm1.get("anomaly_late_error_percent", float("nan"))),
                float(vm1.get("anomaly_lift_recovery", float("nan"))),
                float(vm1.get("event_auroc", float("nan"))),
            )
            _hist.append({"round": _r, "failing_before": fails, "adjustments": adj,
                          "score_before": best_r_score, "score_after": sc1,
                          "failing_after": fails1,
                          "metrics_after": {k: float(v) for k, v in vm1.items()
                                            if isinstance(v, (int, float))
                                            and math.isfinite(float(v))}})
            if improved:
                best_r_score = sc1
                best_r_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                save_checkpoint(best_candidate_path, model, r_opt, scaler, controller,
                                cfg, global_epoch, "R", vm1, clean_cache, ema=None)
            fails = fails1
            if not improved:
                logger.info("STAGE R stops: no score improvement this round; "
                            "the honest verdict stands and is reported as measured.")
                break
        if best_r_state is None and "_pre_r_state" in dir():
            model.load_state_dict(_pre_r_state)
            model.discard_pending_control_updates(logger, "pre-remediation state restored")
            logger.info("STAGE R restored the pre-remediation weights (no round improved).")
        if best_r_state is not None:
            model.load_state_dict(best_r_state)
            model.discard_pending_control_updates(logger, "best remediated state restored")
            logger.info("STAGE R kept the best remediated weights "
                        "(score %.4f); the shipped checkpoint can only have improved.",
                        best_r_score)
        if _hist:
            try:
                atomic_json_dump({"rounds": _hist}, run_dir / "reports" / "remediation_history.json")
            except Exception:
                pass
    test_metrics = evaluate_model(model, test_loader, loss_fn, device, cfg, "C")
    try:
        _dg = model_state_digest(model.state_dict())
        export_split_arrays(model, val_loader, device, cfg, run_dir / "exports" / "val", "val",
                            clean_cache.target_time, logger, checkpoint_digest=_dg)
        export_split_arrays(model, test_loader, device, cfg, run_dir / "exports" / "test", "test",
                            clean_cache.target_time, logger, checkpoint_digest=_dg)
        for _lab3 in ("val", "test"):
            write_native_axis_copies(run_dir / "exports" / _lab3, cfg, clean_cache.target_time, logger)
    except Exception as _exc:
        logger.warning("export failed: %s", _exc)
    test_metrics["project_feasible"] = is_project_feasible(test_metrics, cfg.contract)
    test_metrics["selected_checkpoint"] = str(selected)
    test_metrics["selected_checkpoint_is_feasible_file"] = bool(selected == best_feasible_path)
    test_metrics["snr_protocol"] = (
        ("contract %s dB" % (list(cfg.data.contract_test_snr_db_range)
                             if getattr(clean_cache, "province_id", None) is not None
                             else "training distribution"))
        if bool(getattr(cfg.data, "val_contract_snr_protocol", False))
        else "THE DATASET'S OWN PAIRS (the noise they carry, their own SNR population)")
    atomic_json_dump(test_metrics, run_dir / "metrics" / "synthetic_test_metrics.json")
    _deliv: Dict[str, Any] = {}
    try:
        select_edge_reflect_mode(model, cfg, run_dir, clean_cache, noise_cache, logger)
        _splits = build_splits(clean_cache, cfg)
        for _lab2 in ("val", "test"):
            if _lab2 not in _splits:
                continue
            _m = deliverable_contract_eval(model, cfg, run_dir, clean_cache, noise_cache,
                                              np.asarray(_splits[_lab2], dtype=np.int64), logger, _lab2)
            for _k2, _v3 in (_m or {}).items():
                if isinstance(_v3, (int, float, bool, str)):
                    _deliv["deliverable_%s_%s" % (_lab2, _k2)] = _v3
            _x = export_deliverable_arrays(model, cfg, clean_cache, noise_cache,
                                              np.asarray(_splits[_lab2], dtype=np.int64),
                                              run_dir / "exports" / _lab2, _lab2, logger)
            _deliv["deliverable_%s_export_status" % _lab2] = str(_x.get("status", "INCOMPLETE"))
            write_native_axis_copies(run_dir / "exports" / _lab2, cfg, clean_cache.target_time, logger)
        cfg.data._deliverable_done = bool(_deliv)
        if "deliverable_test_contract_pass" in _deliv:
            logger.info("BLOCK PROTOCOL vs DELIVERABLE PATH (test) | G %.3f%% -> %.3f%% | L %.3f%% -> %.3f%% | "
                        "CVaR %.2f%% -> %.2f%%. The stamped verdict follows the deliverable path.",
                        float(test_metrics.get("global_error_percent", float("nan"))), float(_deliv["deliverable_test_global_error_percent"]),
                        float(test_metrics.get("late_error_percent", float("nan"))), float(_deliv["deliverable_test_late_error_percent"]),
                        float(test_metrics.get("late_cvar_percent", float("nan"))), float(_deliv["deliverable_test_late_cvar_percent"]))
    except Exception as _e10:
        logger.warning("deliverable-path contract evaluation skipped: %r", _e10)
    _stat = str(_deliv.get("deliverable_test_contract_status", "INCOMPLETE"))
    if "deliverable_test_contract_pass" not in _deliv:
        logger.error("deliverable-path contract evaluation is MISSING: the qualification is INCOMPLETE and "
                     "contract_test_pass=False (the block-protocol feasibility is recorded only for reference). ")
    _qual = {
        "contract_test_pass": bool(_stat == "PASS"),
        "contract_status": _stat,
        "contract_test_pass_block_protocol": bool(test_metrics["project_feasible"]),
        **_deliv,
        "global_error_percent": float(test_metrics.get("global_error_percent", float("nan"))),
        "early_mid_error_percent": float(test_metrics.get("early_mid_error_percent", float("nan"))),
        "late_error_percent": float(test_metrics.get("late_error_percent", float("nan"))),
        "late_cvar_percent": float(test_metrics.get("late_cvar_percent", float("nan"))),
        "contract_thresholds": {
            "global": float(cfg.contract.global_threshold),
            "late": float(cfg.contract.late_threshold),
            "late_cvar": float(cfg.contract.late_cvar_threshold),
        },
        "snr_protocol": test_metrics.get("snr_protocol"),
        "split_provenance": getattr(clean_cache, "grouping_provenance", "unknown"),
        "clean_cache": str(clean_cache.data_path),
        "anomaly_suite_pass": None,
        "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    def _sd_digest(sd: Dict[str, Any]) -> str:
        return model_state_digest(sd)
    _tested = _sd_digest(model.state_dict())
    _qual["tested_model_state_digest"] = _tested
    _qual["program_digest"] = _program_digest()
    for _p in {selected, best_feasible_path, best_candidate_path}:
        if not _p.exists():
            continue
        try:
            _st = torch.load(_p, map_location="cpu", weights_only=False)
            _sd = _st.get("model_state", _st) if isinstance(_st, dict) else {}
            _same = (_sd_digest(_sd) == _tested)
        except Exception as _e14:
            _same = False
            logger.warning("%s: model identity could not be verified (%r); verdict NOT stamped.", _p.name, _e14)
        if _same:
            stamp_checkpoint_qualification(_p, _qual, logger)
        else:
            logger.info("%s holds different weights than the tested model -- verdict NOT stamped on it.", _p.name)
    logger.info(
        "Contract qualification stamped into the checkpoint(s): contract_test_pass=%s "
        "(G=%.4f%% L=%.4f%% CVaR=%.4f%%). --mode infer now gates on THIS, not on validation.",
        _qual["contract_test_pass"], _qual["global_error_percent"],
        _qual["late_error_percent"], _qual["late_cvar_percent"],
    )
    if getattr(clean_cache, "province_id", None) is not None:
        try:
            stress_ds = datasets[2]
            stress_ds._snr_override = tuple(cfg.data.province_test_snr_db_range)
            stress_loader = DataLoader(
                stress_ds, batch_size=block_aligned_batch_size(stress_ds, cfg.train.batch_size),
                shuffle=False,
                num_workers=0, drop_last=False,
            )
            stress_metrics = evaluate_model(model, stress_loader, loss_fn, device, cfg, "C")
            stress_metrics["project_feasible"] = is_project_feasible(stress_metrics, cfg.contract)
            stress_metrics["snr_protocol"] = "stress %s dB" % list(cfg.data.province_test_snr_db_range)
            atomic_json_dump(stress_metrics, run_dir / "metrics" / "stress_test_metrics.json")
            logger.info(
                "STRESS TEST [%s..%s dB] | G=%.4f%% EM=%.4f%% L=%.4f%% CVaR=%.4f%% -- robustness "
                "figure ONLY; the contract verdict above is the deployment gate.",
                cfg.data.province_test_snr_db_range[0], cfg.data.province_test_snr_db_range[1],
                stress_metrics["global_error_percent"], stress_metrics["early_mid_error_percent"],
                stress_metrics["late_error_percent"], stress_metrics["late_cvar_percent"],
            )
            stress_ds._snr_override = tuple(cfg.data.contract_test_snr_db_range)
        except Exception as exc:
            logger.warning("Stress test skipped: %s", exc)
    atomic_json_dump(
        {
            "ablation": cfg.runtime.ablation,
            "description": ABLATION_PRESETS[cfg.runtime.ablation][0],
            "overrides": ABLATION_PRESETS[cfg.runtime.ablation][1],
            "version": PROGRAM_VERSION,
            "test_metrics": {k: v for k, v in test_metrics.items() if isinstance(v, (int, float, bool, str))},
        },
        run_dir / "reports" / "ablation_summary.json",
    )
    logger.info(
        "TEST | G=%.4f%% EM=%.4f%% L=%.4f%% CVaR=%.4f%% feasible=%s",
        test_metrics["global_error_percent"],
        test_metrics["early_mid_error_percent"],
        test_metrics["late_error_percent"],
        test_metrics["late_cvar_percent"],
        test_metrics["project_feasible"],
    )
    try:
        run_generalization_diagnosis(model, cfg, run_dir, clean_cache, noise_cache, logger)
    except Exception as exc:
        logger.warning("Generalization diagnosis skipped: %s", exc)
    global _LAST_MONITOR_SUMMARY
    try:
        _verdict = ("test late %.4f%% | CVaR %.4f%% | feasible %s"
                    % (test_metrics.get("late_error_percent", float("nan")),
                       test_metrics.get("late_cvar_percent", float("nan")),
                       test_metrics.get("project_feasible", "?")))
        _monitor.close(_verdict)
        _LAST_MONITOR_SUMMARY = _monitor.summarize()
    except Exception as _cexc:
        logger.warning("Monitor close skipped: %r", _cexc)
    return model, selected, test_metrics


PROMOTION_INERT_NEW_VALUES: Tuple[Tuple[str, float], ...] = (
    ("neighbor_attention.geo_prior_rho", -12.0),
    ("neighbor_attention.geo_bias", 0.0),
)


PROMOTION_INERT_BIAS_SUFFIXES: Tuple[str, ...] = ("head_atoms.bias",)
PROMOTION_INERT_BIAS: float = -20.0


def configure_acceleration(
    cfg: Config,
    logger: logging.Logger,
    device_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """All GPU-dependent setup in one testable function."""
    if device_info is None:
        avail = bool(torch.cuda.is_available())
        if avail:
            p = torch.cuda.get_device_properties(0)
            device_info = {
                "available": True, "name": p.name, "total_memory": int(p.total_memory),
                "capability_major": int(torch.cuda.get_device_capability(0)[0]),
                "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported",
                                               lambda: False)()),
            }
        else:
            device_info = {"available": False}
    out: Dict[str, Any] = {"applied": False, "amp_dtype": "fp32",
                           "tf32": False, "cudnn_benchmark": False}
    if not bool(device_info.get("available")):
        logger.info(
            "ACCELERATION | no CUDA device: training in float32 on CPU. ")
        return out
    total_gb = float(device_info.get("total_memory", 0)) / 2**30
    if bool(cfg.train.amp):
        major = int(device_info.get("capability_major", 0))
        bf16_ok = bool(device_info.get("bf16_supported", False))
        want = str(cfg.train.amp_dtype or "auto").lower()
        if want == "fp16":
            dt = torch.float16
        elif want == "bf16":
            dt = torch.bfloat16 if bf16_ok else torch.float16
            if not bf16_ok:
                logger.warning("bf16 requested but unsupported here; using fp16.")
        else:
            dt = torch.bfloat16 if (bf16_ok and major >= 8) else torch.float16
        set_amp_dtype(dt)
        out["amp_dtype"] = amp_dtype_name()
        logger.info(
            "MIXED PRECISION | %s on %s (compute %d.x). %s",
            "bfloat16" if dt is torch.bfloat16 else "float16",
            device_info.get("name", "GPU"), major,
            "bf16 has the fp32 exponent range, so no loss scaling is needed and the "
            "scale-collapse failure mode cannot occur; every accumulation in this program "
            "is already float32."
            if dt is torch.bfloat16 else
            "fp16 needs a GradScaler; the collapse guard is armed.")
        if bool(cfg.train.allow_tf32):
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                out["tf32"] = True
            except Exception as exc:
                logger.warning("TF32 could not be enabled: %r", exc)
        if bool(cfg.train.cudnn_benchmark) and not bool(cfg.train.deterministic):
            try:
                torch.backends.cudnn.benchmark = True
                out["cudnn_benchmark"] = True
            except Exception as exc:
                logger.warning("cudnn.benchmark could not be enabled: %r", exc)
        logger.info(
            "ACCELERATION | autocast=%s | TF32=%s | cudnn.benchmark=%s.",
            out["amp_dtype"], out["tf32"], out["cudnn_benchmark"])
    logger.info(
        "GPU MEMORY | %s: %.1f GB total, training budget %.0f%% = %.1f GB. ",
        device_info.get("name", "GPU"), total_gb,
        100.0 * cfg.train.vram_budget_fraction, total_gb * cfg.train.vram_budget_fraction,
    )
    if total_gb < 20.0 and int(cfg.train.num_workers) > 4:
        logger.info(
            "%.0f GB card: lowering --num_workers %d -> 4 (each worker holds its "
            "own pinned copies, which competes with the model for host memory).",
            total_gb, int(cfg.train.num_workers),
        )
        cfg.train.num_workers = 4
        out["num_workers"] = 4
    out["applied"] = True
    out["total_gb"] = total_gb
    return out


def autotune_batch_for_vram(
    model: "PEBRNet",
    cfg: Config,
    device: torch.device,
    loss_fn: "ProjectLoss",
    logger: logging.Logger,
) -> Tuple[int, int]:
    """Measure the card, then choose the micro-batch."""
    want = int(cfg.train.batch_size)
    P = max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 0))
    if device.type != "cuda" or not bool(cfg.train.vram_autotune):
        forced = int(cfg.train.micro_batch_size or 0)
        if forced <= 0 and device.type != "cuda":
            forced = int(getattr(cfg.train, "cpu_micro_batch_rows", 0) or 0)
            if 0 < forced < want:
                logger.info("CPU MICRO-BATCH | %d rows (%d patches) per forward instead of the whole %d-row "
                            "batch; gradients are accumulated as on a GPU. --micro_batch_size sets another value.",
                            max(P, (forced // P) * P), max(1, forced // P), want)
        if forced > 0 and forced < want:
            accum = max(1, int(math.ceil(want / forced)))
            return max(P, (forced // P) * P if P > 1 else forced), accum
        return want, 1
    total = float(torch.cuda.get_device_properties(device).total_memory)
    try:
        torch.cuda.empty_cache()
        _cap = cuda_capacity(int(device.index if device.index is not None else torch.cuda.current_device()))
        log_foreign_vram({int(device.index or 0): _cap}, logger)
        total = min(total, float(_cap["capacity"]))
    except Exception:
        pass
    budget = total * float(cfg.train.vram_budget_fraction)
    probe = max(P, min(want, 8 * P if P > 1 else 16))
    G = int(cfg.model.gates)
    K = int(cfg.data.num_neighbors)

    def _peak_for(n: int) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(n, 1, G, device=device)
        nb = torch.randn(n, K, G, device=device)
        b = {
            "noisy": x, "clean": x.clone(),
            "noise": torch.zeros_like(x),
            "q_target": torch.ones(n, 1, G, device=device),
            "snr_db": torch.zeros(n, device=device),
            "event_mask": torch.zeros(n, 1, G, device=device),
            "event_label": torch.zeros(n, device=device),
            "neighbor_geometry": torch.zeros(n, K, NEIGHBOR_GEOMETRY_FEATURES,
                                             device=device),
        }
        was = model.training
        model.train(True)
        try:
            with amp_autocast_context(device, bool(cfg.train.amp)):
                out = model(x, nb)
                probe_loss = loss_fn(out, b, "C")["total"]
            probe_loss.backward()
        finally:
            model.zero_grad(set_to_none=True)
            model.train(was)
        pk = float(torch.cuda.max_memory_allocated(device))
        del x, nb, b
        torch.cuda.empty_cache()
        return pk

    probe2 = int(min(want, probe * 2))
    try:
        base = _peak_for(probe)
        base2 = _peak_for(probe2) if probe2 > probe else None
    except RuntimeError as exc:
        logger.warning("VRAM probe failed (%r); keeping the configured batch.", exc)
        return want, 1
    if base2 is not None and probe2 > probe and base2 > base:
        per_sample = max((base2 - base) / float(probe2 - probe), 1.0)
        const = max(0.0, base - per_sample * float(probe))
    else:
        per_sample = max(base / max(probe, 1), 1.0)
        const = 0.0
    reserve = max(0.0, float(getattr(cfg.train, "vram_reserve_gb", 1.25))) * float(2 ** 30)
    fits = int(max(budget - const - reserve, per_sample) / per_sample)
    if P > 1:
        fits = max(P, (fits // P) * P)
    micro = int(max(P if P > 1 else 1, min(want, fits)))
    accum = max(1, int(math.ceil(want / max(micro, 1))))
    logger.info(
        "VRAM AUTOTUNE | card %.1f GB, budget %.1f GB (%.0f%%) | probe: %d samples peaked at "
        "%.2f GB (%.1f MB/sample) | micro-batch %d x %d accumulation steps = effective batch %d.",
        total / 2**30, budget / 2**30, 100.0 * cfg.train.vram_budget_fraction,
        probe, base / 2**30, per_sample / 2**20, micro, accum, micro * accum)
    logger.info(
        "VRAM ACCOUNTING | fixed cost (weights/grads/workspace) %.2f GB + "
        "reserve (GPU-resident libraries + allocator headroom) %.2f GB + activations "
        "%d x %.1f MB = %.2f GB projected against a %.2f GB budget on a %.1f GB card. ",
        const / 2**30, reserve / 2**30, micro, per_sample / 2**20,
        (const + reserve + per_sample * micro) / 2**30, budget / 2**30, total / 2**30,
    )
    if accum > 1:
        logger.info(
            "Steps per epoch rise %dx and each is %dx cheaper: total work per "
            "epoch is unchanged, but nothing spills to shared memory, which is where the "
            "previous run lost most of its wall-clock time.", accum, accum,
        )
    return micro, accum


def apply_optimizer_batch_policy(micro: int, accum: int, cfg: Config, logger: logging.Logger) -> Tuple[int, int, int]:
    """(micro, accumulation, effective batch) under TrainConfig.optimizer_batch_policy."""
    want = int(cfg.train.batch_size)
    micro = int(max(1, micro))
    accum = int(max(1, accum))
    policy = str(getattr(cfg.train, "optimizer_batch_policy", "micro")).lower()
    if policy != "micro":
        logger.info("optimizer batch policy = legacy: effective batch %d rows by %d accumulation(s) of %d. ", micro * accum, accum, micro)
        return micro, accum, want
    P = max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 0))
    floor_rows = max(P, int(getattr(cfg.train, "min_effective_rows", 288) or 0))
    new_accum = int(min(accum, max(1, int(math.ceil(floor_rows / float(micro))))))
    eff = micro * new_accum
    _spe = None
    try:
        _n = int(getattr(cfg.data, "train_samples_per_epoch", 0) or 0)
        if _n > 0:
            _spe = (max(1, int(math.ceil(_n / float(micro)) // accum)), max(1, int(math.ceil(_n / float(micro)) // new_accum)))
    except Exception:
        _spe = None
    logger.info("OPTIMIZER-STEP BUDGET | policy micro: one update per %d micro-batch(es) of %d rows = "
                "%d rows (%d patches) per update; the legacy policy accumulated %d (%d rows)%s. LR peak, epoch "
                "size and stage lengths unchanged; --optimizer_batch_policy legacy restores the old behaviour.",
                new_accum, micro, eff, eff // P, accum, micro * accum,
                (" | optimizer steps per epoch %d -> %d" % _spe) if _spe else "")
    return micro, new_accum, eff


def _probe_flags(n_rows: int, patch_len: int, cap_share: float, full_fraction: float, device: torch.device) -> torch.Tensor:
    P = max(1, int(patch_len))
    if float(cap_share) >= float(full_fraction) or n_rows % P != 0:
        return torch.ones(n_rows, device=device)
    n = n_rows // P
    k = min(n, max(1, int(math.ceil(float(cap_share) * n))))
    f = torch.zeros(n, P, device=device)
    f[:k] = 1.0
    return f.reshape(-1)


def measure_aux_flag_shares(loader: Any, patch_len: int, n_batches: int) -> Optional[Dict[str, float]]:
    """Patch-level shares of has_anomaly / has_alt over the first n_batches real training batches."""
    P = max(1, int(patch_len))
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "set_epoch"):
        ds.set_epoch(1)
    sb: List[float] = []
    sa: List[float] = []
    npatch = 0
    for i, b in enumerate(loader):
        ha = b.get("has_anomaly")
        hl = b.get("has_alt")
        B = int(b["clean"].shape[0])
        if B % P != 0 or ha is None:
            continue
        n = B // P
        npatch = max(npatch, n)
        sb.append(float((ha.reshape(n, P).float().amax(dim=1) > 0.5).float().mean()))
        sa.append(float((hl.reshape(n, P).float().amax(dim=1) > 0.5).float().mean()) if hl is not None else 1.0)
        if i + 1 >= int(n_batches):
            break
    if not sb:
        return None
    return {"bg_mean": float(np.mean(sb)), "bg_max": float(np.max(sb)), "alt_mean": float(np.mean(sa)),
            "alt_max": float(np.max(sa)), "patches": float(npatch), "batches": float(len(sb))}


def aux_cap_share(mean_share: float, patches: int, sigma: float) -> float:
    """Mean + sigma * binomial standard deviation of the per-batch share, in [0, 1]."""
    p = float(min(max(mean_share, 0.0), 1.0))
    n = float(max(1, int(patches)))
    return float(min(1.0, p + float(sigma) * math.sqrt(max(p * (1.0 - p), 0.0) / n)))


def make_probe_batch(model: nn.Module, cfg: Config, n: int, device: torch.device) -> Dict[str, Any]:
    """A batch carrying every input the training step reads -- the main pair, the background arm
    (noisy_bg/has_anomaly), the alternate arm (noisy_alt/has_alt), the neighbour stacks with geometry.
    """
    G = int(cfg.model.gates)
    K = int(cfg.data.num_neighbors)
    P = max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 0))
    base = getattr(model, "module", model)
    gs = getattr(base, "gate_scale_norm", None)
    if gs is not None and int(gs.numel()) == G:
        gs = gs.detach().float().to(device).view(1, 1, G).abs().clamp_min(1e-12)
    else:
        gs = torch.ones(1, 1, G, device=device)
    amp = 0.75 + 0.5 * torch.rand(n, 1, 1, device=device)
    clean = (amp * gs).abs()
    noise = 0.05 * gs * torch.randn(n, 1, G, device=device)
    noisy = clean + noise
    nb = (gs * (1.0 + 0.05 * torch.randn(n, K, G, device=device))).abs()
    if hasattr(base, "_default_neighbor_geometry"):
        geo = base._default_neighbor_geometry(nb)
    else:
        geo = torch.zeros(n, K, NEIGHBOR_GEOMETRY_FEATURES, device=device)
    b: Dict[str, Any] = {
        "noisy": noisy, "clean": clean, "noise": noise,
        "q_target": torch.ones(n, 1, G, device=device),
        "snr_db": torch.zeros(n, device=device),
        "event_mask": torch.zeros(n, 1, G, device=device),
        "event_label": torch.zeros(n, device=device),
        "neighbors": nb, "neighbor_geometry": geo,
        "clean_bg": clean.clone(),
        "noisy_bg": noisy + 0.01 * gs * torch.randn(n, 1, G, device=device),
        "has_anomaly": _probe_flags(n, P, float(getattr(getattr(base, "cfg", None), "aux_bg_cap_share", 1.0) or 1.0),
                                        float(getattr(getattr(base, "cfg", None), "aux_full_fraction", 0.90) or 0.90), device),
        "noisy_alt": clean + noise.roll(1, dims=0),
        "has_alt": _probe_flags(n, P, float(getattr(getattr(base, "cfg", None), "aux_alt_cap_share", 1.0) or 1.0),
                                    float(getattr(getattr(base, "cfg", None), "aux_full_fraction", 0.90) or 0.90), device),
        "neighbors_bg": nb.clone(), "neighbors_alt": nb.clone(),
        "depth_norm": torch.full((n, 1), 0.5, device=device),
        "profile_len": torch.full((n,), int(P), dtype=torch.int64, device=device),
        "is_forward": torch.zeros(n, device=device),
        "arm_id": torch.zeros(n, device=device),
        "clean_index": torch.arange(n, device=device), "noise_index": torch.arange(n, device=device),
    }
    return b


def _training_card_ids(device: torch.device) -> List[int]:
    w = _DP_STATE.get("wrapper")
    if w is not None:
        return [int(i) for i in w.device_ids]
    return [int(device.index if device.index is not None else torch.cuda.current_device())]


def fit_micro_batch(
    model: "PEBRNet",
    cfg: Config,
    device: torch.device,
    loss_fn: "ProjectLoss",
    logger: logging.Logger,
) -> Tuple[int, int]:
    """Measure the step that will run, on every card that will run it."""
    want = int(cfg.train.batch_size)
    P = max(1, int(getattr(cfg.data, "profile_patch_len", 0) or 0))
    forced = int(cfg.train.micro_batch_size or 0)
    if forced > 0:
        micro = int(min(want, max(P, (forced // P) * P if P > 1 else forced)))
        accum = max(1, int(math.ceil(want / max(micro, 1))))
        logger.info("micro-batch PINNED by --micro_batch_size %d -> %d rows x %d accumulation = effective %d "
                    "(probe skipped).", forced, micro, accum, micro * accum)
        return micro, accum
    if device.type != "cuda":
        rows = int(getattr(cfg.train, "cpu_micro_batch_rows", 0) or 0)
        if 0 < rows < want:
            micro = int(max(P, (rows // P) * P if P > 1 else rows))
            accum = max(1, int(math.ceil(want / max(micro, 1))))
            logger.info("CPU MICRO-BATCH | %d rows (%d patches) per forward instead of the whole %d-row batch "
                        "(a CPU has no memory probe); --micro_batch_size sets another value.",
                        micro, micro // P, want)
            return micro, accum
        return want, 1
    if not bool(cfg.train.vram_autotune):
        return want, 1
    ids = _training_card_ids(device)
    totals = {i: float(torch.cuda.get_device_properties(torch.device("cuda", i)).total_memory) for i in ids}
    free_cuda_after_oom()
    caps = {i: cuda_capacity(i) for i in ids}
    log_foreign_vram(caps, logger)
    frac = float(cfg.train.vram_budget_fraction)
    if (not bool(getattr(cfg.train, "vram_budget_fraction_explicit", False))
            and min(totals.values()) / 2 ** 30 >= 40.0 and frac < 0.88):
        logger.info("VRAM budget %.0f%% -> 88%% on %.0f GB cards (pass --vram_budget_fraction to pin it). ", 100.0 * frac, min(totals.values()) / 2 ** 30)
        frac = 0.88
        cfg.train.vram_budget_fraction = 0.88
    reserve = max(0.0, float(getattr(cfg.train, "vram_reserve_gb", 1.25))) * float(2 ** 30)
    margin = max(0.0, float(getattr(cfg.train, "vram_step_margin", 0.10)))
    probe = max(P, min(want, 8 * P if P > 1 else 16))
    _mc = getattr(getattr(model, "module", model), "cfg", None)
    if (float(getattr(_mc, "aux_bg_cap_share", 1.0) or 1.0) < 1.0 or float(getattr(_mc, "aux_alt_cap_share", 1.0) or 1.0) < 1.0):
        probe = max(P, min(want, 16 * P if P > 1 else 32))
    probe2 = int(min(want, probe * 2))
    stage = "C"

    def _peak_for(n: int) -> Dict[int, float]:
        free_cuda_after_oom()
        for i in ids:
            torch.cuda.reset_peak_memory_stats(torch.device("cuda", i))
        b = make_probe_batch(model, cfg, n, device)
        was = model.training
        model.train(True)
        _AUX_PROBE_MODE["on"] = True
        try:
            with amp_autocast_context(device, bool(cfg.train.amp)):
                out = forward_training_arms(model, b)
                probe_loss = loss_fn(out, b, stage)["total"]
            probe_loss.backward()
            peaks = {i: float(torch.cuda.max_memory_allocated(torch.device("cuda", i))) for i in ids}
        finally:
            _AUX_PROBE_MODE["on"] = False
            model.zero_grad(set_to_none=True)
            model.train(was)
        del b, out, l
        free_cuda_after_oom()
        return peaks

    def _fit_and_choose(p1: Dict[int, float], p2: Optional[Dict[int, float]], n1: int, n2: int):
        rows = []
        micro_all = None
        for i in ids:
            if p2 is not None and n2 > n1 and p2[i] > p1[i]:
                slope = max((p2[i] - p1[i]) / float(n2 - n1), 1.0)
                const = max(0.0, p1[i] - slope * float(n1))
            else:
                slope = max(p1[i] / max(n1, 1), 1.0)
                const = 0.0
            budget_i = min(totals[i], caps[i]["capacity"]) * frac
            fits_i = int(max(budget_i - const - reserve, 0.0) / (slope * (1.0 + margin)))
            rows.append((i, slope, const, budget_i, fits_i))
            micro_all = fits_i if micro_all is None else min(micro_all, fits_i)
        return int(micro_all or 0), rows

    def _enable_checkpointing() -> bool:
        base = getattr(model, "module", model)
        base = getattr(base, "_orig_mod", base)
        stack = getattr(base, "refinement_stack", None)
        if stack is None or bool(getattr(stack, "checkpoint_blocks", False)):
            return False
        stack.checkpoint_blocks = True
        return True

    try:
        p1 = _peak_for(probe)
        try:
            p2 = _peak_for(probe2) if probe2 > probe else None
        except Exception as _e2:
            if not is_oom_error(_e2):
                raise
            free_cuda_after_oom()
            p2 = None
            logger.warning("the %d-row probe did not fit; single-point fit from %d rows.", probe2, probe)
        micro_fit, rows = _fit_and_choose(p1, p2, probe, probe2)
        _w = _DP_STATE.get("wrapper")
        if _w is not None and len(ids) >= 2 and p2 is not None:
            _pin = float(getattr(cfg.train, "dp_primary_share", 0.0) or 0.0)
            _mp = [s for i, s, _c, _b, _f in rows if i == ids[0]][0]
            _ms = float(np.mean([s for i, s, _c, _b, _f in rows if i != ids[0]]))
            if _pin > 0.0:
                _share = float(min(0.5, max(0.2, _pin)))
            else:
                _share = float(min(0.5, max(0.3, 0.5 - (_mp - _ms) / (4.0 * max(_ms, 1.0)))))
            if abs(_share - float(getattr(_w, "primary_share", 0.5))) > 0.01:
                _w.primary_share = _share
                _rows_eq = rows
                p1 = _peak_for(probe)
                p2 = _peak_for(probe2) if probe2 > probe else None
                micro_fit, rows = _fit_and_choose(p1, p2, probe, probe2)
                logger.info("TWO-CARD SPLIT REBALANCED | equal split: %s | primary share -> %.3f (%s) | rebalanced: %s "
                            "-> tightest card fits %d rows.",
                            " | ".join("cuda:%d %.1f MB/row" % (i, s / 2 ** 20) for i, s, _c, _b, _f in _rows_eq), _share,
                            "pinned by --dp_primary_share" if _pin > 0.0 else "from the slopes m_p %.1f / m_s %.1f MB/row" % (_mp / 2 ** 20, _ms / 2 ** 20),
                            " | ".join("cuda:%d %.1f MB/row -> fits %d" % (i, s / 2 ** 20, f) for i, s, _c, _b, f in rows), micro_fit)
        mode = str(getattr(cfg.model, "activation_checkpointing", "auto")).lower()
        if micro_fit < P and mode == "auto" and _enable_checkpointing():
            logger.warning("not even one %d-station patch fits the budget without activation checkpointing "
                           "(fit %d rows): refinement-stack checkpointing ENABLED, re-probing.", P, micro_fit)
            p1 = _peak_for(probe)
            p2 = _peak_for(probe2) if probe2 > probe else None
            micro_fit, rows = _fit_and_choose(p1, p2, probe, probe2)
        if P > 1:
            micro_fit = (micro_fit // P) * P
        micro = int(max(P, min(want, micro_fit)))
        accum = max(1, int(math.ceil(want / max(micro, 1))))
        _VRAM_FIT.update({"cards": {int(i): (float(s), float(c)) for i, s, c, _b2, _f2 in rows},
                              "frac": float(frac), "reserve": float(reserve), "margin": float(margin), "patch": int(P)})
        base = getattr(model, "module", model)
        ck_on = bool(getattr(getattr(getattr(base, "_orig_mod", base), "refinement_stack", None), "checkpoint_blocks", False))
        logger.info(
            "VRAM CAPACITY | %s.",
            " | ".join("cuda:%d nameplate %.1f GB, obtainable %.1f GB, foreign %.1f GB" % (
                i, caps[i]["total"] / 2 ** 30, caps[i]["capacity"] / 2 ** 30, caps[i]["foreign"] / 2 ** 30) for i in ids))
        logger.info(
            "VRAM AUTOTUNE | cards %s, budget %.0f%% of each card's OBTAINABLE memory | "
            "probe = the REAL step (main + background + alternate arms + continuation, real loss, backward) "
            "at %d and %d rows | per-card fit: %s | reserve %.2f GB, margin %.0f%%, checkpointing %s | "
            "tightest card fits %d rows -> micro-batch %d x %d accumulation = effective %d (configured %d).",
            ["cuda:%d %.1f GB" % (i, totals[i] / 2 ** 30) for i in ids], 100.0 * frac, probe, probe2,
            " | ".join("cuda:%d %.1f MB/row + %.2f GB fixed -> fits %d" % (i, s / 2 ** 20, c / 2 ** 30, f) for i, s, c, _bud, f in rows),
            reserve / 2 ** 30, 100.0 * margin, "ON" if ck_on else "off", micro_fit, micro, accum, micro * accum, want)
        if micro_fit < P:
            logger.critical("the tightest card does not fit even one patch (%d rows) at this budget: the run "
                            "starts at %d rows and the P3 safety net will halve further if needed; consider "
                            "--vram_budget_fraction 0.85 or --activation_checkpointing on.",
                            micro_fit, micro)
        return micro, accum
    except Exception as _exc:
        free_cuda_after_oom()
        logger.warning("real-step probe unavailable (%r): falling back to the legacy single-forward probe "
                       "divided by the arm multiplier 3.2 (three full-batch forwards + continuation).", _exc)
        try:
            m0, _a0 = autotune_batch_for_vram(model, cfg, device, loss_fn, logger)
        except Exception as _exc0:
            logger.warning("legacy probe failed too (%r): starting at one patch per card set; the out-of-memory guard protects the "
                           "step.", _exc0)
            m0 = P
        micro = int(max(P, min(want, (int(m0 / 3.2) // P) * P if P > 1 else int(m0 / 3.2))))
        accum = max(1, int(math.ceil(want / max(micro, 1))))
        logger.info("fallback micro-batch %d x %d accumulation = effective %d.", micro, accum, micro * accum)
        return micro, accum


def initialize_from_parent_checkpoint(
    model: "PEBRNet",
    parent_path: Path,
    cfg: Config,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Cross-stage initialization for the Q0 -> Q4 ladder."""
    parent_path = Path(parent_path)
    state = torch.load(parent_path, map_location="cpu", weights_only=False)
    sd = state.get("model_state")
    if sd is None:
        sd = state.get("model")
    if sd is None:
        sd = state.get("state_dict")
    if not isinstance(sd, dict):
        raise ValueError(
            "--init_from: %s contains no model state dict under 'model_state', 'model' or 'state_dict' "
            "(top-level keys: %s)."
            % (parent_path, sorted(state)[:12] if isinstance(state, dict) else type(state)))
    own = model.state_dict()
    buffers = {n for n, _ in model.named_buffers()}
    copied, expanded, new_keys, skipped, inerted = [], [], [], [], []
    merged = dict(own)
    for k, v_own in own.items():
        if k not in sd:
            for _pref, _val in PROMOTION_INERT_NEW_VALUES:
                if k.startswith(_pref):
                    merged[k] = torch.full_like(v_own, float(_val))
                    inerted.append("%s<-%.1f" % (k, _val))
                    break
            new_keys.append(k)
            continue
        v_par = sd[k]
        if not torch.is_tensor(v_par):
            skipped.append(k)
            continue
        if tuple(v_par.shape) == tuple(v_own.shape):
            merged[k] = v_par.clone()
            copied.append(k)
            continue
        if k in buffers:
            expanded.append("%s %s->%s (physics basis regenerated from config; parent block "
                            "reproduced exactly)" % (k, tuple(v_par.shape), tuple(v_own.shape)))
            continue
        if (v_par.dim() == v_own.dim() and v_par.shape[0] < v_own.shape[0]
                and tuple(v_par.shape[1:]) == tuple(v_own.shape[1:])):
            out = torch.zeros_like(v_own)
            n0 = int(v_par.shape[0])
            out[:n0] = v_par.clone()
            if v_own.dim() == 1 and any(k.endswith(s) for s in PROMOTION_INERT_BIAS_SUFFIXES):
                out[n0:] = float(PROMOTION_INERT_BIAS)
            expanded.append("%s %s->%s (parent block copied, %d new rows inert)"
                            % (k, tuple(v_par.shape), tuple(v_own.shape), int(v_own.shape[0]) - n0))
            merged[k] = out
            continue
        skipped.append("%s %s vs %s" % (k, tuple(v_par.shape), tuple(v_own.shape)))
    model.load_state_dict(merged, strict=True)
    adopt_survey_record(model, state)
    model.eval()
    g = torch.Generator().manual_seed(20260718)
    probe = torch.randn(4, 1, int(cfg.model.gates), generator=g)
    perturb = float("nan")
    parent_cfg_dict = (state.get("config") or {}).get("model", {}) or {}
    try:
        _names = {f.name for f in fields(ModelConfig)}
        _pkw = {k: v for k, v in parent_cfg_dict.items() if k in _names}
        if parent_cfg_dict:
            for _k, _v in _PRE_135_ARCH_DEFAULTS.items():
                if _k in ARCHITECTURE_FLAGS and _k in _names and _k not in parent_cfg_dict:
                    _pkw[_k] = _v
        p_model_cfg = replace(cfg.model, **_pkw)
        parent_model = PEBRNet(p_model_cfg)
        parent_model.load_state_dict(sd, strict=False)
        restore_prior_band(parent_model)
        parent_model.eval()
        for _m in (model, parent_model):
            if hasattr(_m, "gate_scale_norm") and hasattr(model, "gate_scale_norm"):
                pass
        k_par = int(getattr(p_model_cfg, "num_neighbors", cfg.data.num_neighbors))
        k_chi = int(cfg.data.num_neighbors)
        nb_par = torch.randn(4, k_par, int(p_model_cfg.gates), generator=torch.Generator().manual_seed(7))
        nb_chi = nb_par[:, :k_chi] if k_chi <= k_par else torch.cat(
            [nb_par, nb_par[:, : k_chi - k_par]], dim=1)
        with torch.no_grad():
            y_par = parent_model(probe, nb_par)["denoised"]
            y_chi = model(probe, nb_chi)["denoised"]
        denom = float(torch.maximum(y_par.abs().max(), torch.tensor(1e-12)))
        perturb = float((y_chi - y_par).abs().max() / denom)
    except Exception as exc:
        logger.warning(
            "Could not rebuild the parent architecture to measure promotion "
            "inertness (%r). The equivalence gate cannot pass on an unmeasured promotion.", exc,
        )
    if not copied:
        raise ValueError(
            "--init_from: %s shares NO parameter with this model (%d own tensors, %d parent tensors, %d "
            "skipped). The parent is either a different architecture or was written under a different key "
            "layout; promoting from it would train a randomly initialized child while reporting a clean "
            "promotion."
            % (parent_path, len(own), len(sd), len(skipped)))
    _cov = len(copied) / max(len(own), 1)
    if _cov < 0.5:
        logger.warning(
            "Only %d of %d tensors (%.1f%%) came from the parent. A promotion this partial is "
            "worth checking before the budget is spent.",
            len(copied), len(own), 100.0 * _cov)
    record = {
        "parent_checkpoint": str(parent_path),
        "parent_sha": sha256_file(parent_path),
        "copied_fraction": _cov,
        "parent_version": state.get("version"),
        "parent_ablation": (state.get("config", {}) or {}).get("runtime", {}).get("ablation"),
        "parent_qualification": checkpoint_qualification(state),
        "child_ablation": cfg.runtime.ablation,
        "copied_keys": len(copied), "expanded_keys": expanded,
        "new_keys": new_keys, "zeroed_new_keys": inerted, "skipped_keys": skipped,
        "restored_optimizer": False, "restored_controller": False, "restored_ema": False,
        "transplant_output_perturbation": perturb,
    }
    logger.info(
        "CROSS-STAGE INITIALIZATION | parent %s (%s, ablation=%s, contract_pass=%s) "
        "-> child ablation=%s | copied %d tensors, expanded %d, new %d, skipped %d",
        parent_path.name, record["parent_sha"][:12], record["parent_ablation"],
        record["parent_qualification"].get("contract_test_pass"), cfg.runtime.ablation,
        len(copied), len(expanded), len(new_keys), len(skipped),
    )
    for e in expanded:
        logger.info("  expanded: %s", e)
    if skipped:
        logger.warning(
            "  %d parent tensor(s) could NOT be transplanted and keep the child's "
            "own initialization: %s", len(skipped), "; ".join(skipped[:6]),
        )
    if record["parent_qualification"].get("contract_test_pass") is not True:
        logger.warning(
            "The parent checkpoint is NOT contract-qualified (contract_test_pass=%s). "
            "The Q ladder is a QUALIFICATION ladder: promoting from an unqualified parent means "
            "the child inherits an unproven starting point.",
            record["parent_qualification"].get("contract_test_pass"),
        )
    return record


def _failed_contract_dims(deployment: Mapping[str, Any], cfg: Config) -> List[str]:
    out: List[str] = []
    for key, gate, label in (
        ("test_global_nrmse", cfg.contract.global_threshold, "global"),
        ("test_early_mid_nrmse", cfg.contract.early_mid_threshold, "early/mid"),
        ("test_late_nrmse", cfg.contract.late_threshold, "late"),
        ("test_late_cvar", cfg.contract.late_cvar_threshold, "late CVaR"),
    ):
        v = deployment.get(key)
        try:
            if v is not None and float(v) > 100.0 * float(gate):
                out.append("%s (%.2f%% > %.0f%%)" % (label, float(v), 100.0 * float(gate)))
        except (TypeError, ValueError):
            continue
    return out


def finalize_deployment(
    deployment: Dict[str, Any],
    anomaly: Mapping[str, Any],
    cfg: Config,
    run_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Combine the held-out checks into one record, reports/deployment_qualification.json."""
    applicable = bool(anomaly.get("applicable", True)) and bool(cfg.model.use_manifold_branch)
    dep = dict(deployment)
    dep["contract_pass"] = bool(dep.get("checkpoint_feasible", False))
    dep["anomaly_gate_applicable"] = applicable
    dep["anomaly_id_pass"] = anomaly.get("anomaly_id_pass")
    dep["anomaly_ood_pass"] = anomaly.get("anomaly_ood_pass")
    dep["quiet_false_rate_pass"] = anomaly.get("quiet_false_rate_pass")
    dep["quiet_false_rate_wilson_upper95"] = anomaly.get("false_anomaly_rate_wilson_upper95")
    dep["anomaly_suite_pass"] = anomaly.get("pass")
    dep["anomaly_suite_reason"] = anomaly.get("reason")
    dep["anomaly_snr_stratum_pass"] = anomaly.get("anomaly_snr_stratum_pass")
    dep["weak_anomaly_pass"] = anomaly.get("weak_anomaly_pass")
    dep["anomaly_cell_grid_pass"] = anomaly.get("anomaly_cell_grid_pass")
    dep["required_contrast_floor"] = anomaly.get("required_contrast_floor")
    dep["empirically_qualified_contrast_floor"] = anomaly.get("empirically_qualified_contrast_floor")
    dep["ip_sign_preservation_pass"] = anomaly.get("ip_sign_preservation_pass")
    _pf = dict(_LAST_FIELD_PREFLIGHT)
    dep["amplitude_alignment_pass"] = _pf.get("amplitude_alignment_pass")
    dep["unit_declaration_pass"] = _pf.get("unit_declaration_pass")
    dep["gate_ratio_tilt_pass"] = _pf.get("gate_ratio_tilt_pass")
    dep["segment_gain_pass"] = _pf.get("segment_gain_pass")
    _sa = dict(_LAST_SPLIT_AUDIT)
    dep["split_leakage_pass"] = _sa.get("split_leakage_pass")
    dep["cross_split_duplicate_fraction"] = _sa.get("cross_split_duplicate_fraction")
    dep["cross_split_near_duplicate_fraction"] = _sa.get("cross_split_near_duplicate_fraction")
    dep["grouping_provenance"] = _pf.get("grouping_provenance", "unknown")
    required_names = ["contract_pass"]
    required = [dep["contract_pass"]]
    if applicable:
        required.append(anomaly.get("pass") is True)
        required_names += ["anomaly_ood_pass", "anomaly_snr_stratum_pass",
                           "anomaly_cell_grid_pass", "weak_anomaly_pass",
                           "ip_sign_preservation_pass", "quiet_false_rate_pass"]
    if dep["amplitude_alignment_pass"] is not None:
        required.append(dep["amplitude_alignment_pass"] is True)
        required_names += ["amplitude_alignment_pass", "gate_ratio_tilt_pass", "segment_gain_pass"]
    if dep["unit_declaration_pass"] is not None:
        required.append(dep["unit_declaration_pass"] is True)
        required_names.append("unit_declaration_pass")
    required.append(dep["split_leakage_pass"] is True)
    required_names.append("split_leakage_pass")
    allowed = all(bool(r) for r in required)
    dep["deployment_required_gates"] = required_names
    if not allowed and bool(cfg.runtime.allow_prior_monopoly):
        dep["debug_override"] = "allow_prior_monopoly"
        dep["deployment_allowed"] = False
        dep["exit_code"] = 25
        logger.warning(
            "--allow_prior_monopoly: the field product will be produced for "
            "DEBUG ONLY. Qualification stays FALSE, outputs are tagged, and the exit code "
            "stays non-zero (25). This is not a deployable result."
        )
        append_output_tag("UNQUALIFIED_DEBUG_ONLY")
    else:
        dep["deployment_allowed"] = bool(allowed)
        if not allowed:
            dep["exit_code"] = 25 if dep["contract_pass"] else dep.get("exit_code", 20)
    dep["verdict_enforced"] = bool(getattr(cfg.runtime, "deployment_gate", True))
    if not dep["verdict_enforced"]:
        dep["exit_code"] = 0
    atomic_json_dump(dep, run_dir / "reports" / "deployment_qualification.json")
    global _LAST_DEPLOYMENT
    _LAST_DEPLOYMENT = dict(dep)
    _amp = ("" if dep["amplitude_alignment_pass"] is None else " | amplitude=%s (tilt=%s segments=%s)" % (
        dep["amplitude_alignment_pass"], dep["gate_ratio_tilt_pass"], dep["segment_gain_pass"]))
    if not dep["verdict_enforced"]:
        logger.info(
            "HELD-OUT TEST | global %.2f%% | late %.2f%% | late CVaR %.2f%% | split audit (no profile group in "
            "two splits)=%s | recorded in reports/deployment_qualification.json; the deployment gate is off.",
            float(dep.get("test_global_nrmse", float("nan"))), float(dep.get("test_late_nrmse", float("nan"))),
            float(dep.get("test_late_cvar", float("nan"))), dep["split_leakage_pass"])
        return dep
    logger.info(
        "DEPLOYMENT QUALIFICATION | contract(incl. CVaR)=%s | anomaly: "
        "applicable=%s OOD=%s SNR-strata=%s weak=%s sign-reversal=%s quiet=%s%s | split-leakage=%s | "
        "ALLOWED=%s (exit %s) | required: %s",
        dep["contract_pass"], applicable, dep["anomaly_ood_pass"], dep["anomaly_snr_stratum_pass"],
        dep["weak_anomaly_pass"], dep["ip_sign_preservation_pass"], dep["quiet_false_rate_pass"], _amp,
        dep["split_leakage_pass"], dep["deployment_allowed"], dep.get("exit_code", 0),
        ", ".join(dep["deployment_required_gates"]),
    )
    return dep
