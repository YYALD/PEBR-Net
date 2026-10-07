# Changelog

## 1.234.0

- CDM-R continues each decay from its own recorded gates with a revised noise model and coefficient prior.
- On withheld gates the output departs from the CDM-R continuation by a learned share (`gl_departure_logit`).
- Amplitude axes of the figures show log10 of the data unit (`--paired_unit`, default nT/s).
- The dataset is checked by content (`data/paired_content.json`).

## 1.233.0: first public release

- PEBR-Net (`pebrnet.PEBRNet`) with CDM-R and E-GSR.
- Training, validation and testing on the paired dataset (https://doi.org/10.57760/sciencedb.014t4), checked by
  `scripts/download_data.py`.
- Training augmentation with simulated noise (`--sim_noise_fraction`) and withheld late gates (`--tail_loss_prob`).
- Optional continuation beyond the last informative gate (`--informative_continuation on`) and rate bound of Eq. 4
  (`--gate_loss_rate_bound input`); both off by default.
- Gate-loss cohort (`--mode gateloss`, and a stage of the figure package).
- Figure package and held-out test report (`reports/FINAL_REPORT.txt`).
- Argument files for the audit, training, evaluation and the gate-loss cohort (`configs/`).
- Style checks (ruff) and tests in the continuous-integration workflow.
