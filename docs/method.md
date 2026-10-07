# From the manuscript to the code

This page shows where the method of *Depth of evidential support for concealed conductors in noise-buried late-time
borehole transient electromagnetic data* (Ye, Zhang and Yu) is implemented. Section and equation numbers refer to
the manuscript.

## The network (Section 3.4)

`pebrnet.PEBRNet` is the network (file `pebrnet/model.py`).

| Manuscript | Code |
|---|---|
| Bounded working domain | `gate_scale_norm`, `signed_symlog` / `signed_symexp`; reported errors are computed in physical units |
| Convolutional trunk | `DualDomainStem`, `MultiScaleTemporalBlock`, `ResidualRefinementStack` |
| Cross-trace attention | `NeighborCrossAttention`, `ProfileJointEncoder`, `LateralJointPrior` |
| CDM-R (Section 3.1) | the relaxation dictionary `decay_atoms` with non-negative weights; `cdmr_continuation` continues each decay from its own recorded gates and `_gate_loss_fill` passes the continuation to the network (`forward(..., obs_mask=...)`); on withheld gates `_withheld_departure` lets the output depart from the continuation by a learned share |
| Continuation beyond the last informative gate (Section 3.1) | `--informative_continuation on` (off by default); `informative_mask` |
| Rate bound, Eq. 4 | `_rate_bound`, optional (`--gate_loss_rate_bound input`), off by default |
| E-GSR (Section 3.2, Eq. 6) | `event_gate_head`, `struct_residual_head`, `event_expression_scale` |
| Augmented-Lagrangian training (Eq. 7 as constraints) | `ConstraintController` and `ProjectLoss` (`pebrnet/losses.py`) |
| Training parameters | `TrainConfig`, `LossConfig`, `ContractConfig` (`pebrnet/config.py`), listed in [training.md](training.md) |

## Training data

| Item | Code |
|---|---|
| The paired dataset | `build_paired_cache`; gate times from the CSV headers. See [data.md](data.md). |
| Split | `group_paired_profiles` keeps near-identical profiles in one split |
| Late window of the signal-to-noise ratio (gates 24–31) | `snr_gate_columns`, `realised_late_snr_db` |
| Training augmentation | `simulate_late_noise` (`--sim_noise_fraction`), gate-loss augmentation (`--tail_loss_prob`) |

Validation and testing use only the dataset's own noisy records.

## Endpoints (Section 3.3, Appendix A6)

| Manuscript | Code |
|---|---|
| Global and late-window NRMSE, Eq. 7 | `gate_loss_endpoints` (`global_nrmse`, `late_nrmse`); per profile in `reports/test_profile_summary.json` |
| Local decay exponent p(t) and curvature κ(t), Eq. A13 | `decay_law_descriptors` (undefined at a gate at t = 0) |
| Amplitude NRMSE, decay-slope RMSE and temporal-structure NRMSE over the withheld gates | `gate_loss_endpoints` (`amplitude_nrmse`, `slope_rmse`, `structure_nrmse`) |

## The gate-loss cohort (Section 5.2)

`run_gate_loss_suite` (`--mode gateloss`, and a stage of the figure package) withholds the last 3, 6, 12 and 18 of
31 gates (9.7–58.1 %, the manuscript) and 19 and 22 of 31 (61.3 % and 71.0 %) on every station of every held-out
test profile. It compares Full PEBR-Net, CDM-R removed, E-GSR removed and a no-continuation baseline, and writes
`reports/gate_loss_suite.json`, `reports/gate_loss_per_profile.csv` and the figures `fig_gate_loss_reconstruction`,
`fig_gate_loss_mechanisms` and `fig_gate_loss_levels`.
