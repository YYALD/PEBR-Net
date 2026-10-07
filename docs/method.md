# From the manuscript to the code

This page shows where the method of *Depth of evidential support for concealed conductors in noise-buried late-time
borehole transient electromagnetic data* (Ye, Zhang and Yu) is implemented. Section, equation and appendix numbers
refer to the manuscript.

## PEBR-Net (Sections 3.2 and 3.4)

`pebrnet.PEBRNet` (file `pebrnet/model.py`) is the prior- and evidence-bounded reconstruction network.

| Manuscript | Code |
|---|---|
| Signed logarithmic coordinate normalized by a robust per-gate scale; reported quantities in the linear response domain (Section 3.1) | `gate_scale_norm`, `signed_symlog` / `signed_symexp`; reported errors are computed in physical units |
| Normalized log-time and square-root-time channels, convolutional trunk and multiscale dilated residual trunk (Section 3.4) | `_build_position_channels`, `DualDomainStem`, `MultiScaleTemporalBlock`, `ResidualRefinementStack` |
| Depth pooling as cross-trace attention (Section 3.4) | `NeighborCrossAttention`, `ProfileJointEncoder`, `LateralJointPrior` |
| CDM-R: relaxation state on the grid of Eq. 3 (Sections 3.2 and 3.4) | the fixed dictionary `decay_atoms` (24 time constants spaced logarithmically from a third of the first positive gate time to three times the last gate time; stretching exponents 1, 0.8 and 0.6) with non-negative weights; `cdmr_continuation` continues each decay from its own recorded gates and `_gate_loss_fill` passes the continuation to the network (`forward(..., obs_mask=...)`); on withheld gates `_withheld_departure` lets the output depart from the continuation by a learned share |
| Continuation beyond the last informative gate (Section 3.1) | `--informative_continuation on` (off by default); `informative_mask` |
| Rate bound of Eq. A5, which confines the continuation to the envelope of Eq. 4 | `_rate_bound`, optional (`--gate_loss_rate_bound input`), off by default |
| E-GSR and the composition of Eq. 6 (Sections 3.2 and 3.4) | `event_gate_head`, `struct_residual_head`, `event_expression_scale` |
| Error measures of Eq. 7 enforced as inequality constraints through an augmented-Lagrangian scheme (Section 3.4) | `ConstraintController` and `ProjectLoss` (`pebrnet/losses.py`) |
| Training parameters | `TrainConfig`, `LossConfig`, `ContractConfig` (`pebrnet/config.py`), listed in [training.md](training.md) |

## Training data

| Item | Code |
|---|---|
| The paired dataset | `build_paired_cache`; gate times from the CSV headers. See [data.md](data.md). |
| Split | `group_paired_profiles` keeps near-identical profiles in one split |
| Late gates 24–31 of the signal-to-noise ratio (Section 2.1) | `snr_gate_columns`, `realised_late_snr_db` |
| Training augmentation | `simulate_late_noise` (`--sim_noise_fraction`), gate-loss augmentation (`--tail_loss_prob`) |

Validation and testing use only the dataset's own noisy records.

## Endpoints (Section 3.3 and Appendix A6)

| Manuscript | Code |
|---|---|
| Global and late-window relative errors, Eq. 7 | `gate_loss_endpoints` (`global_nrmse`, `late_nrmse`); per profile in `reports/test_profile_summary.json` |
| Local decay exponent p(t) and curvature κ(t) = dp/d ln t, Eq. A13 | `decay_law_descriptors` (undefined at a gate at t = 0) |

## The gate-loss cohort (Sections 2.1 and 5.2)

`run_gate_loss_suite` (`--mode gateloss`, and a stage of the figure package) withholds the last 3, 6, 12 and 18 of
31 gates (9.7–58.1 %, the manuscript) and 19 and 22 of 31 (61.3 % and 71.0 %) on every station of every held-out
test profile. It compares Full PEBR-Net, CDM-R removed, E-GSR removed and a no-continuation baseline on the amplitude
NRMSE, the decay-slope RMSE (of p) and the temporal-structure NRMSE (of κ) over the withheld gates and on the
late-window error of Eq. 7 (`gate_loss_endpoints`), and writes `reports/gate_loss_suite.json`,
`reports/gate_loss_per_profile.csv` and the figures `fig_gate_loss_reconstruction`, `fig_gate_loss_mechanisms` and
`fig_gate_loss_levels`.
