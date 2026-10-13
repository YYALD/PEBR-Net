# Training

This page lists what the program does when it trains with its defaults, which `configs/train.args` uses.
`tests/test_training_defaults.py` checks the values on this page against the code.

## Defaults

| parameter | default | option |
|---|---|---|
| optimiser | AdamW, β = (0.9, 0.999), weight decay 1 × 10⁻⁴ | |
| base learning rate | 1 × 10⁻³ | `--base_lr` |
| gradient-clipping norm | 1.0 | |
| weight moving average (EMA) | decay 0.999; validation and the selected checkpoints use the EMA weights | `--disable_ema` |
| random seed | 20260710 | `--seed` |
| batch | 512 patches of 9 stations, capped at 2048 rows: 227 patches = 2043 rows (see [Rows per optimiser update](#rows-per-optimiser-update)) | `--batch_size`, `--patch_batch_cap` |
| epoch | 200 000 training samples, then one validation | `--train_samples_per_epoch` |
| split | the dataset's `train` / `val` / `test` files; near-identical profiles are kept in one split | `--paired_split_policy` |
| constraint targets | global, early/mid and late NRMSE ≤ 3 %, late CVaR ≤ 8 %, enforced as augmented-Lagrangian constraints | |

## Stages

Training runs in three stages. Each stage builds a new optimiser and a new schedule, and starts from the best
weights of the previous stage (`--no_stage_warm_start` turns this off). The schedule rises linearly from 0.02 to 1
of the stage's rate over the first 3 % of the stage's optimiser steps, then falls along a cosine to 0.02.

| stage | epochs | learning rate (× base): trunk / noise head / router / refinement heads | ends |
|---|---|---|---|
| A, global foundation | 50 to 60 | 1.0 / 1.0 / 1.0 / 0.35 | after 60 epochs; or, from epoch 50 on, when the validation score has not improved for 8 epochs, or has met the stage-A thresholds for 5 consecutive epochs and not improved for 4. A plateau far from the contract does not end the stage. |
| B, late refinement | at most 80 | 0.10 / 0.20 / 0.50 / 1.00 | after 80 epochs; or when the validation global and late errors have been at most 3.5 % and 5 % for 5 consecutive epochs; or when the validation score has not improved for 8 epochs |
| C, joint calibration | at most 120 | 0.075 for all four | after 120 epochs, or after 60 epochs without a new best candidate |

Any stage also ends, with its best weights restored, when its smoothed validation score has been more than 1.5
times its best for 12 consecutive validations (counted after epoch 50 in stage A and after epoch 12 in stages B and C).

Three smaller groups follow their own rules. The per-gate prior standard deviation (`prior_log_sigma`) is trained
at 20 × base in every stage, without weight decay. The lateral-prior parameters follow the trunk's rate, without
weight decay. The uncertainty head follows the router.

If a constraint target is still not met on validation after stage C, up to two remediation rounds follow. Each raises
the multipliers of the failing constraints and trains 12 more epochs with the stage-C groups at 0.25 × their rate
(`--remediation_rounds`, `--no_remediation`).

The number of epochs therefore depends on the data, up to 60 + 80 + 120 + 2 × 12 = 284. Every run writes its
stage hand-overs and remediation rounds to its log.

Options: `--stage_epochs A,B,C` (maxima), `--stage_a_min_epochs`, `--stage_switch_plateau_patience`. The
thresholds are the `stage_a_switch_*` and `stage_b_switch_*` fields of `TrainConfig`.

## Rows per optimiser update

The 2043-row batch is not, in general, the number of rows per weight update.

- The micro-batch, the rows that go through the network at once, is measured on a CUDA accelerator at the start
  of a run and fitted to its free memory. It is a whole number of 9-station patches and at most the batch. On a
  CPU, which has no such probe, it is 144 rows (16 patches); `--micro_batch_size` sets another value on any device.
- Under the default policy, `--optimizer_batch_policy micro`, the weights are updated after every micro-batch.
  When a micro-batch has fewer than 288 rows, gradients are accumulated over as many micro-batches as needed to
  reach 288 rows. The number of rows per update therefore depends on the accelerator's memory.
- `--optimizer_batch_policy legacy` accumulates whole micro-batches up to the full batch, so each update uses at
  least 2043 rows on any card.
- `--vram_gb` and `--gpu_large` switch on a preset that sets the batch, the epoch size, the validation size and
  the number of loader workers from the card's memory, and multiplies the base learning rate by the square root
  of the ratio of that batch to 768. Values given explicitly on the command line are kept. The defaults do not use
  this preset.

Every run writes the micro-batch, the accumulation and the rows per update to its log (`OPTIMIZER-STEP BUDGET`).
Report them with any result.

## Devices

- With two or more visible CUDA devices, training is data-parallel across them, split on patch boundaries.
  `--single_gpu` turns this off; an explicit `--device cuda:N` makes only card N visible.
- Without CUDA, the program runs on the CPU with 144-row micro-batches (about 10 GB of memory and 8 s per 144 rows
  in our test). This is practical for the tests, the data audit and short checks, not for training the network on
  the full dataset.
- GPU kernels are not bit-deterministic unless `--deterministic` is given.

## Software

- The test suite (`python -m pytest`) passes on a CPU under Windows 11 in two environments. `requirements.txt`
  pins NumPy, pandas, Matplotlib and pytest of both and asks for PyTorch 2.9 or newer, so that a GPU build installed
  beforehand is kept:
  - Python 3.14.0 with PyTorch 2.12.0, NumPy 2.3.4, pandas 3.0.3, Matplotlib 3.10.7 and pytest 9.1.1;
  - Python 3.10.11 with PyTorch 2.9.0, NumPy 2.2.6, pandas 2.3.3, Matplotlib 3.10.7 and pytest 8.4.2.
