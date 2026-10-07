# Changelog

## Unreleased

- The curvature of the gate-loss endpoints is κ(t) = dp/d ln t as defined by Eq. A13 (the sign was reversed); the
  endpoint values do not change.
- Citation metadata (`CITATION.cff`, `.zenodo.json`, README): title, affiliations and references as in the
  manuscript.

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
