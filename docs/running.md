# Running PEBR-Net

All commands run from the repository root. `python -m pebrnet` and the `pebrnet` command (after `pip install -e .`)
are the same program. The argument files in `configs/` hold the options of each run; options given after them on
the command line are added and, when repeated, take precedence.

## 0. The data

```bash
python scripts/download_data.py --from <downloaded archive or folder>
```

The dataset is distributed separately (https://doi.org/10.57760/sciencedb.014t4). The command copies the six data files to `data/paired/` and checks
their content ([data.md](data.md)).

## 1. Audit the data (no training)

```bash
python -m pebrnet @configs/audit.args
```

The audit reads the six files, builds the split, caches the data and reports the data checks. It takes a few
minutes on a CPU.

## 2. Train, test and render the figures

```bash
python -m pebrnet @configs/train.args
```

- Training runs in three stages with the defaults of [training.md](training.md). A CUDA GPU is strongly
  recommended; with two or more cards the run is data-parallel.
- The selected checkpoint is tested on the held-out test profiles, and the run ends with the figure package
  ([outputs.md](outputs.md)), including the gate-loss cohort on every test profile.
- The selected checkpoint is `runs/train/model/<run>/best_feasible.pt` when a checkpoint met the constraint targets
  of training on validation, otherwise `best_candidate.pt`; `reports/FINAL_REPORT.txt` names it.

The run exits with 0 when it completes. A non-zero exit code means that it stopped on an error; the log names it.

## 3. Evaluate a trained checkpoint

```bash
python -m pebrnet @configs/evaluate.args --checkpoint <path to the checkpoint>
```

This re-runs the held-out test and the figure package from the checkpoint, without training.
`python scripts/download_data.py --checkpoint` fetches a trained checkpoint once one is listed in
`data/sources.json`.

## 4. The gate-loss cohort alone

```bash
python -m pebrnet @configs/gate_loss.args --checkpoint <path to the checkpoint>
```

This withholds the last 3, 6, 12, 18, 19 and 22 of 31 gates on every held-out test profile and scores Full PEBR-Net,
CDM-R removed, E-GSR removed and the no-continuation baseline. The results are written to
`reports/gate_loss_suite.json` and `reports/gate_loss_per_profile.csv`, with the figures
`fig_gate_loss_reconstruction`, `fig_gate_loss_mechanisms` and `fig_gate_loss_levels`.

## Options worth knowing

- `--device auto|cpu|cuda|cuda:N`: `auto` takes the CUDA device with the most free memory, or the CPU.
- `--informative_continuation on|off` (default off): continuation beyond the last informative gate (manuscript
  Section 3.1).
- `--gate_loss_rate_bound input`: the rate bound of Eq. A5 (the envelope of Eq. 4) on the continued gates (off by
  default).
- `--seed` fixes data sampling and initialisation. GPU kernels are not bit-deterministic unless `--deterministic`
  is given.
- `python -m pebrnet --help` lists every option.

## Notes

- Windows limits paths to 260 characters. Keep the repository and `runs/` in a short folder; figure names carry
  run tags.
- A run on a CPU uses float32 and no mixed precision; it is practical for the audit, the tests and short checks.
