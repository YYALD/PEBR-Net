# PEBR-Net, the prior- and evidence-bounded reconstruction network.
# MIT License, see LICENSE.
"""Command-line front end, the figure package and the dataset section figure.

This file is executed last by pebrnet/__init__.py into the package namespace; names defined in the other
files of the package are available here without imports.
"""
from __future__ import annotations

import shlex

# ----------------------------------------------------------------------------------------------- command line
PUBLIC_MODES = ("train", "all", "figures", "gateloss", "diagnose", "audit", "ablation_report")
_PRIVATE_FLAG = re.compile(
    r"field|survey|measured|noise_library|noise_dir|menu|evidence|(?:^--|_)ip(?:_|$)|init_from|promot"
    r"|section_autopsy|units_preset|amplitude_unit|time_unit|signal_quantity|normalization_history|unify_axis"
    r"|clean_dir|clean_csv|mix_clean|province|v1122|synth_|unqualified|prior_monopoly|legacy|deployment_gate"
    r"|profile_extras")


def _args_from_line(line: str) -> List[str]:
    return shlex.split(line, comments=True)


def build_parser() -> argparse.ArgumentParser:
    """The program's parser with the public modes; other options still parse but are hidden."""
    p = _build_parser_full()
    p.prog = "pebrnet"
    p.description = (
        "PEBR-Net, the prior- and evidence-bounded reconstruction network. "
        "Modes: train (train and test), all (train, test and the figure package), figures (figure package from a "
        "checkpoint), gateloss (gate-loss cohort from a checkpoint), diagnose, audit (read the data, build the split "
        "and run the data audits). Arguments can be read from files: pebrnet @configs/train.args")
    p.fromfile_prefix_chars = "@"
    p.convert_arg_line_to_args = _args_from_line
    for a in p._actions:
        if a.dest == "mode":
            a.choices = list(PUBLIC_MODES)
            a.help = " | ".join(PUBLIC_MODES)
        elif a.dest == "ablation" and a.choices:
            a.choices = [c for c in a.choices if not _PRIVATE_FLAG.search("_" + str(c))]
        elif a.option_strings and any(_PRIVATE_FLAG.search(o) for o in a.option_strings):
            a.help = argparse.SUPPRESS
    return p


# ----------------------------------------------------------------------------------------------- dataset figure
def _profile_sample_id(name: str) -> str:
    return str(name).rsplit("#sample_", 1)[-1]


def render_dataset_sections(clean_cache: "CleanCacheInfo", figure_dir: Any, logger: Optional[logging.Logger] = None,
                            stem: str = "fig_dataset_sections", unit: str = "nT/s") -> Dict[str, Any]:
    """Clean and noisy late gates of the median-SNR profile of each split, with three station decays."""
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    log = logger or logging.getLogger(PROGRAM_NAME)
    if getattr(clean_cache, "paired_split_id", None) is None or not getattr(clean_cache, "paired_noisy_path", ""):
        log.warning("dataset sections not rendered: the data are not a paired dataset")
        return {}
    clean = np.load(clean_cache.data_path, mmap_mode="r")
    noisy = np.load(clean_cache.paired_noisy_path, mmap_mode="r")
    depth = np.load(clean_cache.station_path, mmap_mode="r")
    pid = np.asarray(clean_cache.province_id, dtype=np.int64)
    split = np.asarray(clean_cache.paired_split_id, dtype=np.int64)
    names = list(clean_cache.province_names or [])
    t_ms = np.asarray(clean_cache.target_time, dtype=np.float64).reshape(-1) * 1000.0   # header times are seconds
    cols = snr_gate_columns(t_ms.size)
    picks = []
    for code, sp in enumerate(PAIRED_SPLIT_NAMES):
        rows = np.nonzero(split == code)[0]
        if rows.size == 0:
            continue
        c = np.asarray(clean[rows][:, cols], dtype=np.float64)
        e = np.asarray(noisy[rows][:, cols], dtype=np.float64) - c
        ids, inv = np.unique(pid[rows], return_inverse=True)
        ps = np.bincount(inv, weights=np.sum(c ** 2, axis=1))
        pn = np.bincount(inv, weights=np.sum(e ** 2, axis=1))
        snr = 10.0 * np.log10(np.maximum(ps, 1e-300) / np.maximum(pn, 1e-300))
        k = int(np.argsort(snr, kind="mergesort")[len(snr) // 2])
        r = rows[pid[rows] == ids[k]]
        r = r[np.argsort(np.asarray(depth[r]), kind="stable")]
        picks.append({"split": sp, "sample_id": _profile_sample_id(names[int(ids[k])]) if names else str(ids[k]),
                      "late_snr_db": float(snr[k]), "n_profiles": int(ids.size), "depth": np.asarray(depth[r]),
                      "clean": np.asarray(clean[r], dtype=np.float64), "noisy": np.asarray(noisy[r], dtype=np.float64)})
    if not picks:
        log.warning("dataset sections not rendered: no profile in any split")
        return {}
    titles = {"train": "Training", "val": "Validation", "test": "Test"}
    ylab = "Amplitude (log10(%s))" % unit
    with plt.rc_context(nature_rc()):
        fig, axes = plt.subplots(len(picks), 3, figsize=_mm(183.0, 52.0 * len(picks) + 10.0), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.0, 1.0, 0.85]})
        cmap = plt.get_cmap("viridis")
        letters = "abcdefghijkl"
        norm = Normalize(vmin=float(t_ms[cols[0]]), vmax=float(t_ms[cols[-1]]))
        for i, pk in enumerate(picks):
            c, n, z = pk["clean"], pk["noisy"], pk["depth"]
            lc, ln = log10_amplitude(c[:, cols]), log10_amplitude(n[:, cols])
            lo = min(float(np.nanmin(lc)), float(np.nanpercentile(ln, 1.0))) - 0.1
            hi = max(float(np.nanmax(lc)), float(np.nanpercentile(ln, 99.5))) + 0.2
            for j, ax in enumerate(axes[i, :2]):
                y = lc if j == 0 else ln
                for q, g in enumerate(cols):
                    ax.plot(z, y[:, q], color=cmap(norm(float(t_ms[g]))), lw=0.6)
                ax.set_ylim(lo, hi)
                ax.set_xlabel("Depth (m)")
                ax.set_ylabel(ylab)
                if j == 0:
                    ttl = "(%s) %s profile %s: clean reference" % (letters[3 * i], titles[pk["split"]], pk["sample_id"])
                else:
                    ttl = "(%s) noisy record: late SNR %.1f dB" % (letters[3 * i + 1], pk["late_snr_db"])
                ax.set_title(ttl, loc="left", fontsize=6.5)
            ax = axes[i, 2]
            lo_d = np.inf
            for q, colr in zip((0, len(z) // 2, len(z) - 1), (OKABE_ITO["blue"], OKABE_ITO["vermillion"],
                                                             OKABE_ITO["green"])):
                yc, yn = c[q], n[q]
                ax.plot(t_ms, log10_amplitude(yc), color=colr, lw=0.9, ls=(0, (3.0, 2.0)))
                pos = yn > 0
                ax.plot(t_ms[pos], np.log10(yn[pos]), ls="none", marker="o", mfc="white", mec=colr, ms=2.0, mew=0.5)
                neg = ~pos & (yn != 0)
                if np.any(neg):
                    ax.plot(t_ms[neg], np.log10(np.abs(yn[neg])), ls="none", marker="v", mfc="white", mec=colr,
                            ms=2.2, mew=0.5)
                lo_d = min(lo_d, float(np.nanmin(log10_amplitude(yc))))
                if np.any(yn != 0):
                    lo_d = min(lo_d, float(np.min(np.log10(np.abs(yn[yn != 0])))))
                ax.plot([], [], color=colr, lw=0.9, label="%.0f m" % float(z[q]))
            ax.plot([], [], ls="none", marker="v", mfc="white", mec="0.35", ms=2.6, mew=0.5,
                    label="negative (|amplitude|)")
            set_gate_time_xscale(ax, t_ms)
            ax.set_xlim(*gate_time_xlim(t_ms))
            if np.isfinite(lo_d):
                ax.set_ylim(bottom=lo_d - 0.3)
            ax.set_xlabel("Gate time (ms)")
            ax.set_ylabel(ylab)
            ax.legend(loc="lower left", fontsize=5.5)
            ax.set_title("(%s) decays: clean dashed, noisy markers" % letters[3 * i + 2], loc="left", fontsize=6.5)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes[:, :2].ravel().tolist(), location="bottom", shrink=0.5, aspect=40, pad=0.02)
        cb.set_label("Gate time of the late gates (ms)")
        _save_figure(fig, Path(figure_dir), stem)
        plt.close(fig)
    out = {pk["split"]: {"sample_id": pk["sample_id"], "late_snr_db": pk["late_snr_db"],
                         "n_profiles": pk["n_profiles"]} for pk in picks}
    log.info("%s rendered | median-SNR profile per split: %s", stem,
             "; ".join("%s %s (%.1f dB of %d profiles)" % (k, v["sample_id"], v["late_snr_db"], v["n_profiles"])
                       for k, v in out.items()))
    return out


# ----------------------------------------------------------------------------------------------- figure package
def produce_figure_package(model: "PEBRNet", cfg: "Config", run_dir: Path, clean_cache: "CleanCacheInfo",
                           noise_cache: "NoiseCacheInfo", checkpoint_path: Path, logger: logging.Logger,
                           run_field: bool = True, deployment: Optional[Mapping[str, Any]] = None,
                           simulation_only: bool = False) -> Dict[str, Any]:
    """The figure package of a trained network; every figure is verified in PDF, PNG and SVG."""
    cfg.runtime.plot = True
    expected: List[str] = []
    skipped: Dict[str, str] = {}
    results: Dict[str, Any] = {}

    def _stage(name: str, stems: Sequence[str], fn: Callable[[], Any]) -> Optional[Any]:
        expected.extend(stems)
        t0 = time.time()
        try:
            out = fn()
            logger.info("STAGE OK | %-38s %6.1f s", name, time.time() - t0)
            return out
        except Exception as exc:   # noqa: BLE001 - one stage must not cost the others
            logger.exception("FIGURE STAGE FAILED (%s): %s", name, exc)
            for st in stems:
                skipped[st] = "%s stage raised: %s" % (name, exc)
            return None

    if getattr(clean_cache, "paired_noisy_path", ""):
        _stage("edge-mode A/B on validation", [],
               lambda: select_edge_reflect_mode(model, cfg, run_dir, clean_cache, noise_cache, logger))
        _stage("deliverable-path contract", [],
               lambda: _deliverable_contract_stage(model, cfg, run_dir, clean_cache, noise_cache, logger))
        _stage("test profile comparison", ["fig_test_profile_comparison"],
               lambda: render_test_profile_comparison(model, cfg, run_dir, clean_cache, noise_cache, logger,
                                                      split="test"))
        _stage("validation profile comparison", ["fig_val_profile_comparison"],
               lambda: render_test_profile_comparison(model, cfg, run_dir, clean_cache, noise_cache, logger,
                                                      split="val"))
        results["gate_loss"] = _stage(
            "gate-loss cohort",
            ["fig_gate_loss_reconstruction", "fig_gate_loss_mechanisms", "fig_gate_loss_levels"],
            lambda: run_gate_loss_suite(model, cfg, run_dir, clean_cache, noise_cache, logger))
    results["manifest"] = verify_figures(run_dir, expected, skipped, logger, deployment=deployment)
    return results
