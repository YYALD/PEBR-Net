# Changelog

## Unreleased

- CDM-R during training: the calibrated posterior is computed on the host in float64, whatever the device; a
  continuation that is not finite, or leaves the network's z-domain on a withheld gate, takes the uncalibrated
  continuation; the withheld-gate output stays in that domain (`tests/test_cdmr_guard.py`).
- Start-up check: a failure on the gradients is diagnosed (loss terms, parameters, rows) and the step is retried in
  float32 and on one card; the first configuration that passes is used for the run.
- Two-card training: when a batch's withheld gates fell on one card's share only, the cards returned different
  outputs and the gather stopped with `KeyError: 'gate_loss_fill_z'`. The forward now returns the same outputs on
  every card, the gather names any difference, and the start-up check compares a two-card gate-loss forward with one
  card (`tests/test_two_card.py` runs the split and the gather on the CPU).
- GPU memory probe: the probe of the real training step raised an error after measuring and fell back to a coarser
  estimate (smaller micro-batches); it now completes.
- The CDM-R calibration is matched to the decay manifold in force by an exact digest of its values, the same on
  every device, instead of rounded sums.
- The curvature of the gate-loss endpoints is κ(t) = dp/d ln t as defined by Eq. A13 (the sign was reversed); the
  endpoint values do not change.
- Citation metadata (`CITATION.cff`, `.zenodo.json`, README): title, affiliations and references as in the
  manuscript.
- CDM-R noise model: the per-curve noise levels now reach 256 times the calibrated level and the noise covariance is
  shrunk 1 % toward its diagonal, so that curves of the simulated-noise augmentation are continued inside the
  network's z-domain (before, part of them took the uncalibrated continuation).
- E-GSR during training on paired data: its expression scale follows the detector's AUROC on the validation
  profiles' own anomalies; before, it stayed at zero.
- A micro-batch whose gradient is not finite is recomputed in float32 and kept when its gradient is finite; one that
  stays non-finite is dropped alone and no longer cancels the update of its accumulation window.
- Figures: the profile comparison keeps a reconstruction below the clean curves inside the axes; markers outside the
  axes are clipped in the SVG files as in the PDF and PNG files; the depth legend of the dataset figure shows the line
  and marker styles of its curves.
- E-GSR composes with the complete baseline as in Eq. 6: its evidence-gated residual is added after the lateral joint
  prior and, on withheld gates, after the learned departure from the CDM-R continuation, at unit weight (before, both
  scaled it). The gate-loss report records the learned departure share by position in the withheld window.
- Option `--event_evidence_normalization local_noise`: the E-GSR evidence in units of the local noise of the
  neighbourhood; the default keeps the fixed scale.
- Lateral joint prior: its strength is learned for each station and gate from the network's features by a small head
  trained through the solve, and a class-balanced loss weighs the lateral second differences of the rows that touch a
  body and of the rows that do not equally; the strength has no fixed lower bound. `--lateral_joint_form learned` keeps
  one learned strength per gate for every station; `--alpha_lateral_balanced` sets the weight of the loss.

## V1.0.0 (2026-10-07)

First release.

- PEBR-Net (`pebrnet.PEBRNet`) with CDM-R and E-GSR. CDM-R continues each decay from its own recorded gates; on
  withheld gates the output departs from the continuation by a learned share.
- Training, validation and testing on the paired dataset (https://doi.org/10.57760/sciencedb.014t4); the files are
  checked by content (`scripts/download_data.py`).
- Training augmentation with simulated noise (`--sim_noise_fraction`) and withheld late gates (`--tail_loss_prob`).
- Optional continuation beyond the last informative gate (`--informative_continuation on`) and rate bound of Eq. A5
  (`--gate_loss_rate_bound input`); both off by default.
- Gate-loss cohort (`--mode gateloss`, and a stage of the figure package).
- Figure package and held-out test report (`reports/FINAL_REPORT.txt`); amplitude axes in log10 of the data unit
  (`--paired_unit`, default nT/s).
- Argument files for the audit, training, evaluation and the gate-loss cohort (`configs/`).
- Style checks (ruff) and tests in the continuous-integration workflow.
