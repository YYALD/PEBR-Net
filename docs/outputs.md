# Outputs of a run

A run writes to four roots. The program defaults are below; each argument file in `configs/` sets its own under
`runs/<name>/` (for example `runs/train/output`, `runs/train/model`, ...).

| root | option | content |
|---|---|---|
| `runs/output` | `--output_root` | one folder per run (below) and the shared data cache `_cache/` |
| `runs/model` | `--model_root` | checkpoints of each run |
| `runs/figures` | `--plot_root` | figures of each run |
| `runs/logs` | `--log_root` | the run log, the reconstruction monitor and `runs_index.csv` |

The run folder `runs/output/<run>/` is named `PEBRNet_<date>_<time>`, or `ABL_<preset>_<date>_<time>` with
`--ablation`. It contains:

```
config.json            the complete configuration of the run
run.log                the run log (also under runs/logs)
checkpoints/   ->      link to runs/model/<run>/: best_feasible.pt or best_candidate.pt (selected), best_late.pt, best_stage_*.pt,
                       last.pt, and the TorchScript export (single-station, single-pass forward)
figures/       ->      link to runs/figures/<run>/ (PDF, 600-dpi PNG; Visio-editable SVG under svg_visio/)
metrics/               training_history.json, synthetic_test_metrics.json (held-out test metrics),
                       stress_test_metrics.json, ...
reports/               data_audit.json, split_counts.json, deployment_qualification.json (held-out test errors
                       and split audit), deliverable_contract_{val,test}.json, {val,test}_sections_metrics.csv,
                       {val,test}_profile_summary.json, gate_loss_suite.json, gate_loss_per_profile.csv,
                       FINAL_REPORT.txt
exports/               the deliverable-path reconstructions of the validation and test profiles
cache/                 run-local caches
```

With `--ablation`, figure and report names carry the preset name.

## Figures

| file | content |
|---|---|
| `fig_test_profile_comparison` | held-out test profiles: noisy record, clean reference and reconstruction, one curve per gate along the hole |
| `fig_val_profile_comparison` | the same for validation profiles |
| `fig_gate_loss_reconstruction` | sample decays with their late gates withheld, at every withheld level |
| `fig_gate_loss_mechanisms` | the gate-loss endpoints with CDM-R or E-GSR removed |
| `fig_gate_loss_levels` | every withheld level (9.7–71.0 %) and operator |

Amplitudes are plotted as log10 of the dataset's unit (`--paired_unit`, default nT/s).

## Reports to read first

- `reports/FINAL_REPORT.txt`: the held-out test errors (global, early/mid and late NRMSE, late CVaR, by SNR bin),
  the per-profile late NRMSE of Eq. 7 of the reconstruction and of the raw record (median and quartiles; also in
  `reports/test_profile_summary.json`), next to the constraint targets of training.
- `reports/gate_loss_suite.json`: for every level and operator, the median and quartiles of each endpoint.
  `full_record_late_nrmse_median_by_informative_continuation` gives the full record read with and without the
  informative-gate continuation.
- The run log lines `GATE-LOSS W03` ... `W22` and `FULL RECORD (0% withheld)`.
