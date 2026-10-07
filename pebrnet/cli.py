# PEBR-Net -- prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.
# MIT License, see LICENSE.
"""Command line: the argument parser, the configuration builder and main().

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

def _build_parser_full() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PEBR-Net: reconstruction of noise-buried late-time borehole TEM responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.fromfile_prefix_chars = "@"
    p.convert_arg_line_to_args = arg_file_line
    p.add_argument("--mode", choices=["all", "audit", "train", "diagnose", "figures", "ablation_report",
                                      "gateloss"], default=None)
    p.add_argument("--ablation", choices=list(ABLATION_PRESETS), default=None,
                   help="Train/evaluate a named ablation variant; output files are tagged accordingly.")
    p.add_argument("--v1122_parity", action="store_true",
                   help="Legacy late fusion: blend = floor + (1-floor)*sigmoid(head) with a step floor at "
                        "--manifold_blend_floor, and none of the quality cap, validation floor.")
    p.add_argument("--stage_budget_legacy", action="store_true",
                   help="Restore the A=120 / B=0 / C=60 budget with stage_c_lr_factor 0.04.")
    p.add_argument("--no_menu", action="store_true",
                   help="Never show the interactive menu; use CLI values/defaults directly.")
    p.add_argument("--clean_csv", default=PathConfig.clean_csv)
    p.add_argument("--clean_dir", default=PathConfig.clean_dir,
                   help="Directory of per-province clean CSVs (rows=stations, cols=depth+gates, one file per "
                        "province).")
    p.add_argument("--no_clean_dir", action="store_true",
                   help="Ignore the province directory and use the single-file "
                        "--clean_csv library with the old within-file split instead.")
    p.add_argument("--no_mix_clean_csv", action="store_true",
                   help="In province mode, do not mix the single-file --clean_csv library into training.")
    p.add_argument("--no_stack_slot", action="store_true",
                   help="Do not append the virtual coherent-stack slot to the neighbour stack.")
    p.add_argument("--neighbor_channel_dropout", type=float, default=None,
                   help="Training-time probability that a real neighbour arrives as a dead channel (pure noise at"
                        " realistic amplitude).")
    p.add_argument("--mix_forward_fraction", "--mix_clean_csv_fraction", type=float,
                   default=None, dest="mix_forward_fraction",
                   help="Fraction of training draws taken from the FORWARD-simulation auxiliary domain (default "
                        "0.15).")
    p.add_argument("--field_group_size", type=int, default=None,
                   help="Consecutive FIELD-library rows per spatial block for the "
                        "leakage-free train/val/test split (default 64).")
    p.add_argument("--mix_clean_csv_max_rows", type=int, default=None,
                   help="Cap the number of single-file library rows mixed into "
                        "training (0 = all, the default).")
    p.add_argument("--province_val_fraction", type=float, default=None,
                   help="Fraction of provinces held out for validation (default 0.15).")
    p.add_argument("--province_test_fraction", type=float, default=None,
                   help="Fraction of provinces held out for the strong-noise test "
                        "(default 0.15).")
    p.add_argument("--province_test_snr_db", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="SNR range (dB) for the held-out province test set.")
    p.add_argument("--paired_dir", default=PathConfig.paired_dir,
                   help="Directory holding the six paired files {train,val,test}_{clean,noisy}.csv.")
    p.add_argument("--no_paired", action="store_true",
                   help="Ignore the paired files and use the synthetic "
                        "noise-injection path.")
    p.add_argument("--paired_background_window", type=int, default=None,
                   help="Stations in the running-median lateral background "
                        "(default 9).")
    p.add_argument("--paired_event_threshold", type=float, default=None,
                   help="Late-window relative departure from the lateral background that counts as an anomaly "
                        "(default 0.15).")
    p.add_argument("--paired_split_policy", type=str, default=None, choices=["grouped", "files"],
                   help="Grouped (default): near-identical clean profiles are kept in one split; files: the split"
                        " of the files as given.")
    p.add_argument("--paired_copy_rel_tol", type=float, default=None,
                   help="Largest relative difference, entry by entry, between two clean profiles that are "
                        "counted as near-identical (default 2e-3).")
    p.add_argument("--alpha_gate_monotonic", type=float, default=None,
                   help="Weight of the gate-time monotonicity hinge "
                        "(default 0.2; 0 disables).")
    p.add_argument("--no_gate_monotonic", action="store_true",
                   help="Do not project field/test outputs onto the "
                        "monotone-decay cone.")
    p.add_argument("--paired_anomaly_oversample", type=float, default=None,
                   help="Fraction of training draws aimed at anomalous stations (patch mode: at anchors whose "
                        "window contains one).")
    p.add_argument("--paired_hard_snr_oversample", type=float, default=None,
                   help="Fraction of training draws aimed at rows whose late input SNR is below the train split's"
                        " 10th percentile.")
    p.add_argument("--noise_library_dir", default=PathConfig.noise_library_dir,
                   help="Directory of the measured borehole-noise library "
                        "(per-component CSVs).")
    p.add_argument("--measured_noise_inject", type=float, default=None,
                   help="Fraction of training draws that inject the measured borehole noise at a sampled late-"
                        "window SNR.")
    p.add_argument("--paired_residual_replay", type=float, default=None,
                   help="Fraction of training draws whose observation is rebuilt as this clean curve under "
                        "another train station's measured residual.")
    p.add_argument("--neighbor_noise_equicorrelation", type=float, default=None,
                   help="Assumed equicorrelation of neighbour-trace noise for the stacking gain (0 = iid).")
    p.add_argument("--gate_monotonic_projection", action="store_true",
                   help="Opt-in: hard pav monotone projection of the delivered output.")
    p.add_argument("--paired_alt_corruption", type=float, default=None,
                   help="Fraction of training draws that also emit a second real-noise corruption of the same "
                        "clean curve, enabling the invariance term.")
    p.add_argument("--alpha_real_noise_invariance", type=float, default=None,
                   help="Weight of the real-noise invariance term (f(y+n1)~f(y+n2)).")
    p.add_argument("--alpha_noise_orthogonal", type=float, default=None,
                   help="Weight of the early/mid Wiener-orthogonality term.")
    p.add_argument("--alpha_roughness_cap", type=float, default=None,
                   help="Weight of the roughness cap.")
    p.add_argument("--paired_injected_probe", action="store_true",
                   help="Re-arm the injected anomaly probe on paired data "
                        "(diagnostic; selection is carried by the measured checkpoint "
                        "gates either way).")
    p.add_argument("--no_paired_neighbors", action="store_true",
                   help="Do not feed the measured adjacent stations to the "
                        "neighbour attention.")
    p.add_argument("--noise_dir", default=PathConfig.noise_dir)
    p.add_argument("--field_csv", default=PathConfig.field_csv)
    p.add_argument("--no_field", action="store_true",
                   help="Run without a measured profile (known-truth paired study): no field ingestion, survey "
                        "arm, amplitude audit or field stage.")
    p.add_argument("--single_gpu", action="store_true", help="Disable the two-card data parallelism.")
    p.add_argument("--unified_gates", type=int, default=None,
                   help="Unified training length (gates) on the measured span; 0 = the largest native "
                        "count among the signal sources.")
    p.add_argument("--field_native_gates", type=int, default=None,
                   help="Column count of the measured profile file (default 31).")
    p.add_argument("--no_unify_axis", action="store_true",
                   help="Keep the gate times of the paired CSV headers.")
    p.add_argument("--field_gate_table", default=None,
                   help="File with the field instrument's gate-centre times (one per gate; unit via "
                        "--field_gate_time_unit).")
    p.add_argument("--field_gate_time_unit", default=None, choices=["s", "ms", "us"],
                   help="Unit of --field_gate_table (default ms).")
    p.add_argument("--output_root", default=PathConfig.output_root)
    p.add_argument("--model_root", default=PathConfig.model_root,
                   help="Root for checkpoints.")
    p.add_argument("--log_root", default=PathConfig.log_root,
                   help="Root for run logs, the anomaly/reconstruction monitor "
                        "and the per-epoch metric tables.")
    p.add_argument("--log_retain", type=int, default=60,
                   help="How many run logs to keep under --log_root "
                        "(0 = keep everything).")
    p.add_argument("--gpu_large", action="store_true",
                   help="Size batch, epoch and learning rate from the card's own memory.")
    p.add_argument("--aux_arms_memory_factor", type=float, default=None,
                   help="Memory factor the large-VRAM preset divides by to budget the three training forwards "
                        "(main+background+alternate).")
    p.add_argument("--vram_gb", type=float, default=None,
                   help="Declare the usable VRAM in GB instead of querying it "
                        "(implies --gpu_large).")
    p.add_argument("--plot_root", default=PathConfig.plot_root,
                   help="Root for figures.")
    p.add_argument("--checkpoint", default="", help="Checkpoint for infer mode or resume training.")
    p.add_argument("--run_name", default="")
    p.add_argument("--device", default="cuda:1", help="Default cuda:1 (falls back to auto when that index is "
                                                      "absent); auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--num_workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--no_remediation", action="store_true",
                   help="Disable the qualification-driven remediation rounds "
                        "after Stage C.")
    p.add_argument("--remediation_rounds", type=int, default=TrainConfig.remediation_rounds)
    p.add_argument("--no_anomaly_probe", action="store_true",
                   help="Disable the held-out anomaly-recovery probe and drop "
                        "its gates from the checkpoint score.")
    p.add_argument("--anomaly_probe_batches", type=int,
                   default=TrainConfig.anomaly_probe_batches,
                   help="Batches of held-out anomaly cases per validation.")
    p.add_argument("--no_gpu_synthesis", action="store_true",
                   help="Disable GPU-resident batch synthesis and use the "
                        "original CPU DataLoader path.")
    p.add_argument("--compile", action="store_true",
                   help="Torch.compile the model (Linux/Triton recommended); falls back to eager automatically if"
                        " compilation fails.")
    p.add_argument("--train_samples_per_epoch", type=int, default=DataConfig.train_samples_per_epoch)
    p.add_argument("--val_samples", type=int, default=DataConfig.val_samples)
    p.add_argument("--test_samples", type=int, default=DataConfig.test_samples)
    p.add_argument("--stage_epochs", default=None, help="A,B,C epoch maxima, comma list (default: the stage maxima of TrainConfig; nothing auto-escalates).")
    p.add_argument("--base_lr", type=float, default=TrainConfig.base_lr)
    p.add_argument("--lr_min_ratio", type=float, default=TrainConfig.lr_min_ratio,
                   help="Cosine-decay floor as a fraction of base LR.")
    p.add_argument("--disable_ema", action="store_true", help="Turn off weight EMA.")
    p.add_argument("--val_snr_protocol", choices=["measured", "contract"], default=None,
                   help="'measured' (default): validation and test score the dataset's own pairs; "
                        "'contract': inject library noise swept over contract_test_snr_db_range.")
    p.add_argument("--ema_decay", type=float, default=TrainConfig.ema_decay)
    p.add_argument("--stage_switch_plateau_patience", type=int,
                   default=TrainConfig.stage_switch_plateau_patience,
                   help="Advance the curriculum stage after this many plateaued validations.")
    p.add_argument("--num_neighbors", type=int, default=None,
                   help="Neighbor traces for multi-trace cross-attention (0 disables; default 8, "
                        "province mode raises to 12 unless set).")
    p.add_argument("--disable_neighbor_attention", action="store_true",
                   help="Force the single-trace model (no neighbor cross-attention).")
    p.add_argument("--disable_manifold_branch", action="store_true",
                   help="Ablation: turn off the manifold-regeneration branch.")
    p.add_argument("--manifold_blend_floor", type=float, default=None,
                   help="Late-gate blend floor toward the manifold branch (default 0.6).")
    p.add_argument("--lateral_joint_mode", type=str, default=None,
                   choices=["learned", "tikhonov", "off"],
                   help="Learned (default): joint time-profile manifold code + lateral fingerprint operators, no "
                        "fixed smoother; tikhonov: fixed penalty (ablation); off.")
    p.add_argument("--lateral_fingerprint_bank", type=int, default=None,
                   help="Number of lateral prototype signatures (default 32).")
    p.add_argument("--no_lateral_fp_prior", action="store_true",
                   help="Do not apply the lateral fingerprint operator to the prior.")
    p.add_argument("--no_lateral_fp_output", action="store_true",
                   help="Do not apply the lateral fingerprint operator to the output.")
    p.add_argument("--no_lateral_joint", action="store_true",
                   help="Alias of --lateral_joint_mode off.")
    p.add_argument("--lateral_joint_lambda_late", type=float, default=None,
                   help="Initial lateral smoothing weight on the late gates "
                        "(learned per gate afterwards; default 1.0).")
    p.add_argument("--lateral_joint_lambda_mid", type=float, default=None,
                   help="Initial lateral smoothing weight on the mid gates "
                        "(default 0.5).")
    p.add_argument("--lateral_joint_order", type=int, default=None,
                   help="Difference order of the lateral penalty, 1..3 "
                        "(default 3).")
    p.add_argument("--lateral_joint_lambda_floor_late", type=float, default=None,
                   help="Additive floor on the effective late lateral lambda (default 0.10; 0 = off).")
    p.add_argument("--lateral_joint_boundary", choices=["natural", "reflect"], default=None,
                   help="Boundary of the lateral solve: natural (default) or reflect.")
    p.add_argument("--lateral_event_licence", type=float, default=None,
                   help="Structure licence of the lateral prior: lambda x (1 - k * event_prob) on the prior band and "
                        "late window (default 1.0; 0 = off).")
    p.add_argument("--stage_c_lr_factor", type=float, default=None,
                   help="Stage C learning-rate factor (default 0.075).")
    p.add_argument("--no_witness_attention", action="store_true",
                   help="Do not gate the neighbour attention context / z_agg by the structure witness "
                        "(A/B only).")
    p.add_argument("--neighbor_dropout", type=float, default=None,
                   help="Share of training patches whose neighbour stack is replaced by the centre trace "
                        "(default 0).")
    p.add_argument("--survey_profile_fraction", type=float, default=None,
                   help="Share of the library-injection draws that train a true 9-station window of the measured "
                        "survey profile against its reference (default 0.20; 0 = off).")
    p.add_argument("--no_survey_profile", action="store_true",
                   help="Disable the survey-profile arm and the clean.csv row alignment.")
    p.add_argument("--survey_reference_csv", type=str, default=None,
                   help="Survey reference (the teacher's clean result; same layout as --field_csv).")
    p.add_argument("--survey_holdout_m", type=str, default=None,
                   help="'lo,hi' (station units): stations in this band are never a survey training target or "
                        "neighbour.")
    p.add_argument("--extra_clean_alignment", choices=["auto", "on", "off"], default=None,
                   help="Align each clean.csv row to the library early/mid level, as a field station is aligned "
                        "(auto: only when clean.csv covers the survey reference).")
    p.add_argument("--library_row_weight", type=float, default=None,
                   help="Weight of library-noise rows in the row/gate-balanced late teachers (default 0.5; "
                        "0 = exempt).")
    p.add_argument("--library_row_cap_late", type=float, default=None,
                   help="Late-window error cap of library rows in the squared per-sample teacher (default 0.10; "
                        "0 = unbounded).")
    p.add_argument("--library_row_cap_global", type=float, default=None,
                   help="Global / early-mid error cap of library rows in the squared per-sample teachers "
                        "(default 0.05; 0 = unbounded).")
    p.add_argument("--lateral_joint_noise_kappa", type=float, default=None,
                   help="Measured-noise adaptivity of lambda (0 = fixed; default 0.25, capped at "
                        "--lateral_joint_noise_cap 3.0).")
    p.add_argument("--lateral_joint_noise_cap", type=float, default=None,
                   help="Cap of the noise-adaptive lambda factor (default 3.0).")
    p.add_argument("--lateral_joint_weighting", type=str, default=None,
                   choices=["uniform", "sigma"],
                   help="Per-station data weights: uniform (default) or the "
                        "calibrated uncertainty head.")
    p.add_argument("--no_late_noise_floor", action="store_true",
                   help="Disable the measured-noise late floor.")
    p.add_argument("--late_noise_floor_gamma", type=float, default=None,
                   help="Floor scale gamma (default 0.98).")
    p.add_argument("--no_innovation_evidence_gate", action="store_true",
                   help="Disable the evidence-gated innovation.")
    p.add_argument("--innovation_evidence_gate", action="store_true",
                   help="Enable the evidence gate (off by default).")
    p.add_argument("--no_subzero_training_slice", action="store_true",
                   help="Library injections never below 0 dB.")
    p.add_argument("--subzero_training_probability", type=float, default=None,
                   help="Probability of a sub-zero library injection (default 0.25).")
    p.add_argument("--field_alignment_mode", type=str, default=None, choices=["station", "profile"],
                   help="Field amplitude alignment: per-station on the clean window "
                        "(default) or the single profile gain.")
    p.add_argument("--no_weak_body_injection", action="store_true",
                   help="Disable the weak-body augmentation.")
    p.add_argument("--weak_body_injection_prob", type=float, default=None,
                   help="Per-patch probability of a synthetic weak body (default 0.35).")
    p.add_argument("--lateral_lambda_witness", type=str, default=None,
                   choices=["noise_head", "scatter", "none"],
                   help="Reliability witness scaling the Tikhonov lambda per station "
                        "(default noise_head).")
    p.add_argument("--lateral_gate_gamma", type=float, default=None,
                   help="Exponent of the reliability gate (1-r)^gamma on every "
                        "lateral mechanism (default 1.0).")
    p.add_argument("--no_lateral_temporal_witness", action="store_true",
                   help="Use only the stacked-scatter reliability (no temporal "
                        "witness).")
    p.add_argument("--innovation_warmup_epochs", type=int, default=None,
                   help="Epochs over which the innovation ramps in (default 10).")
    p.add_argument("--innovation_robust_nu", type=float, default=None,
                   help="Student-t degrees of freedom of the prior error law "
                        "(default 3; large = Gaussian).")
    p.add_argument("--no_output_measurement_fusion", action="store_true",
                   help="Disable the output-level measurement fusion on the "
                        "early/mid gates.")
    p.add_argument("--no_innovation_precision", action="store_true",
                   help="Disable the per-sample precision-weighted innovation.")
    p.add_argument("--innovation_precision_mod_floor", type=float, default=None,
                   help="Floor of the learned innovation modulator (default 0.25).")
    p.add_argument("--innovation_evidence_kappa", type=float, default=None,
                   help="Sigma threshold of the coherent late-window disagreement "
                        "(default 3.0).")
    p.add_argument("--no_eval_profile_blocks", action="store_true",
                   help="Evaluate on scattered rows (pre-1.134) instead of "
                        "P-station blocks.")
    p.add_argument("--disable_manifold_innovation", action="store_true",
                   help="Ablation: no Kalman-style measurement update on the late gates.")
    p.add_argument("--disable_physics_atoms", action="store_true",
                   help="Ablation: free-form manifold head instead of the decay-atom spectrum.")
    p.add_argument("--disable_noise_enrichment", action="store_true",
                   help="Ablation: real-composite noise only (no colored Gaussian / spikes).")
    p.add_argument("--disable_position_encoding", action="store_true",
                   help="Ablation: remove the diffusion-coordinate position channels.")
    p.add_argument(
        "--spectrum_backend",
        choices=["real_dft", "rfft_manual"],
        default=LossConfig.spectrum_backend,
        help="CUDA-safe spectral-loss backend; real_dft avoids all complex CUDA pointwise kernels.",
    )
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--late_start", type=int, default=21, help="1-based late start channel; 21 means CH21.")
    p.add_argument("--snr_min_db", type=float, default=DataConfig.snr_min_db)
    p.add_argument("--prior_width_mult", type=float,
                   default=ModelConfig.prior_width_mult,
                   help="Manifold-prior decoder width multiplier "
                        "(1.0 = baseline, 2.0 doubles).")
    p.add_argument("--signal_fingerprint_bank", type=int,
                   default=ModelConfig.signal_fingerprint_bank,
                   help="Prototype count of the signal-fingerprint "
                        "memory bank (0 disables).")
    p.add_argument("--snr_max_db", type=float, default=DataConfig.snr_max_db)
    p.add_argument(
        "--snr_below_zero_probability",
        type=float,
        default=DataConfig.snr_below_zero_probability,
        help="Fraction of injected samples with SNR < 0 dB; 0.25 yields 75%% at >= 0 dB.",
    )
    p.add_argument("--split_min_groups", type=int, default=DataConfig.split_min_groups,
                   help="Minimum number of contiguity-preserving groups; the group size is reduced to reach it.")
    p.add_argument("--split_min_val_groups", type=int, default=DataConfig.split_min_val_groups,
                   help="Minimum validation groups (the effective validation sample size).")
    p.add_argument("--no_split_stratify", action="store_true",
                   help="Do not balance the curve population across splits (diagnostic only).")
    p.add_argument("--enable_clean_augment", action="store_true",
                   help="Turn on the physics-exact clean-curve augmentation (off by default: the proven "
                        "configuration did not have it).")
    p.add_argument("--disable_clean_augment", action="store_true",
                   help="Turn off the physics-exact tau-dilation / amplitude augmentation.")
    p.add_argument("--clean_aug_tau_dex", type=float, default=DataConfig.clean_aug_tau_dex,
                   help="Half-width in decades of the tau-dilation range (0.15 -> 0.71x..1.41x).")
    p.add_argument("--v1122_exact", action="store_true",
                   help="Legacy training path, the control arm for attributing a change: turns off the "
                        "measurement gates, per-trace normalization, clean-curve augmentation.")
    p.add_argument("--alpha_manifold_nrmse_late", type=float, default=None,
                   help="Weight of the contract-aligned late term on the manifold branch (default 2.5).")
    p.add_argument("--enable_measurement_gates", action="store_true",
                   help="Turn on the measurement-conditioned innovation/blend gates (calibrated Kalman gain).")
    p.add_argument("--disable_measurement_gates", action="store_true",
                   help="Revert to the free sigmoid blend/innovation gates.")
    p.add_argument("--enable_per_trace_norm", action="store_true",
                   help="Force per-trace amplitude conditioning on (centre-trace log-median estimator on the "
                        "earliest gates).")
    p.add_argument("--disable_per_trace_norm", action="store_true",
                   help="Force the population-z working domain (no per-trace conditioning).")
    p.add_argument("--overwrite_cache", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--gpu16", action="store_true",
                   help="One-shot preset for a 16 GB GPU: batch_size 768, mixed precision on, 4 workers "
                        "persistent, a right-sized 60k-sample epoch with 8k-sample validation.")
    p.add_argument("--no_gradient_projection", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--figure_set", choices=["full", "minimal"], default="full",
                   help="Field figure detail level when plotting is on.")
    p.add_argument("--evidence_missing_fractions", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7",
                   help="Missing late-signal fractions for --mode evidence (comma list, each in [0.03, 0.7]; "
                        "default 0.1,...,0.7).")
    p.add_argument("--evidence_missing_scope", choices=["global", "late", "both"], default="both",
                   help="'global': the last p%% of all gates lose their signal; 'late': the last p%% of the late "
                        "window does; 'both' reports each.")
    p.add_argument("--evidence_samples", type=int, default=RuntimeConfig.evidence_samples,
                   help="Held-out test traces evaluated by the evidence suite.")
    p.add_argument("--evidence_tail_gates", type=int, default=RuntimeConfig.evidence_tail_gates,
                   help="Kept gates used by the exponential/power-law tail-fit baselines.")
    p.add_argument("--evidence_center_only", action="store_true",
                   help="Loose ablation: truncate only the centre trace (default is strict: signal removed from "
                        "centre and all neighbours).")
    p.add_argument("--evidence_bootstrap", type=int, default=RuntimeConfig.evidence_bootstrap,
                   help="Bootstrap resamples behind every reported confidence interval.")
    p.add_argument("--force_infer_unqualified", action="store_true",
                   help="Debug only: run field inference from an unqualified checkpoint.")
    p.add_argument("--no_field_alignment", action="store_true",
                   help="Disable the unit-tested profile-level field amplitude alignment.")
    p.add_argument("--contract_test_snr_db", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="SNR range (dB) of the contract held-out test that decides feasibility (default -5 20).")
    p.add_argument("--stage_regression_min_epochs", type=int, default=None,
                   help="Stage-A epochs before the regression protection may trigger (default 50).")
    p.add_argument("--physics_residual_scale", type=float, default=None,
                   help="Bound of the free z-domain residual on the physics prior.")
    p.add_argument("--no_relaxation_states", action="store_true",
                   help="Restore the single-Debye dictionary (stretch exponents (1.0,)).")
    p.add_argument("--relaxation_update_scale", type=float, default=None,
                   help="Log-amplitude bound of the measurement-gated relaxation-state "
                        "update (default 0.20; 0 disables the update's effect).")
    p.add_argument("--no_relaxation_state_update", action="store_true",
                   help="Do not instantiate the RSSU head at all.")
    p.add_argument("--no_manifold_quality_gate", action="store_true",
                   help="Disable the quality-gated late blend cap (not recommended).")
    p.add_argument("--no_neighbor_geometry", action="store_true",
                   help="Disable the lateral neighbor-geometry pathway (content-only "
                        "attention).")
    p.add_argument("--no_event_residual", action="store_true",
                   help="Disable the event-gated structured geological residual (ablation wo_event).")
    p.add_argument("--struct_residual_scale", type=float, default=None,
                   help="Bound (z units) of the structured residual (default 1.5).")
    p.add_argument("--anomaly_profile_prob", type=float, default=None,
                   help="Probability that a training sample carries a coherent finite-width profile anomaly "
                        "across its neighbours.")
    p.add_argument("--allow_prior_monopoly", action="store_true",
                   help="Debug only.")
    p.add_argument("--anomaly_suite_per_family", type=int, default=None,
                   help="Qualification profiles per OOD family (default 24).")
    p.add_argument("--anomaly_suite_per_weak", type=int, default=None,
                   help="Profiles per weak-contrast stratum (default 24).")
    p.add_argument("--anomaly_detection_contrast", type=float, default=None,
                   help="Smallest late-window relative contrast worth interpreting (default 0.05 = 5%%).")
    p.add_argument("--weak_anomaly_required_contrast", type=float, default=None,
                   help="Weak-contrast strata at or above this value must reach the recall gate for deployment "
                        "(default 0.10 = 10%%).")
    p.add_argument("--weak_anomaly_recall_min", type=float, default=None,
                   help="Required detection recall inside the gated weak strata "
                        "(default 0.90).")
    p.add_argument("--anomaly_suite_profiles", type=int, default=None,
                   help="Ground-truth profiles per anomaly family in the E11 suite "
                        "(default 48; each is evaluated ID and OOD).")
    p.add_argument("--anomaly_suite_quiet", type=int, default=None,
                   help="Anomaly-free profiles used for the false-anomaly rate.")
    p.add_argument("--anomaly_negative_fraction", type=float, default=None,
                   help="Fraction of training anomalies that suppress the late window (resistive bodies) instead "
                        "of lifting it (default 0.25).")
    p.add_argument("--anomaly_skew_fraction", type=float, default=None,
                   help="Fraction of training anomalies with an asymmetric lateral "
                        "envelope (default 0.35).")
    p.add_argument("--stage_a_min_epochs", type=int, default=None,
                   help="Minimum Stage-A epochs before any transition to the next stage, including the plateau "
                        "exit (default 50).")
    p.add_argument("--no_stage_warm_start", action="store_true",
                   help="Do not start each stage from the previous stage's best "
                        "weights (default is to warm-start).")
    p.add_argument("--field_units_preset", type=str, default=None,
                   choices=["nt_s_ms", "nt_s_s"],
                   help="Declare all four field-unit answers at once, for --no_menu runs: 'nt_s_ms' = dB/dt in "
                        "nT/s, gate times in ms, raw (no area/moment normalization)")
    p.add_argument("--field_signal_quantity", type=str, default=None,
                   help="What the field CSV contains: 'dB/dt' or 'B'.")
    p.add_argument("--field_amplitude_unit", type=str, default=None,
                   help="Amplitude unit of the field CSV (V, mV, nV, V/(A.m2), ...).")
    p.add_argument("--field_time_unit", type=str, default=None,
                   help="Time unit of the gate centres in the field CSV ('ms' or 's').")
    p.add_argument("--field_receiver_area_m2", type=float, default=None,
                   help="Effective receiver area, 0 if the data is already normalized.")
    p.add_argument("--field_transmitter_moment_am2", type=float, default=None,
                   help="Transmitter moment, 0 if the data is already normalized.")
    p.add_argument("--field_normalization_history", type=str, default=None,
                   help="What has already been applied to this profile, e.g.")
    p.add_argument("--no_split_dedup_by_shape", action="store_true",
                   help="Do not move duplicated curve shapes to one side of the split.")
    p.add_argument("--no_split_shape_first", action="store_true",
                   help="Use the drop-based de-duplication instead of the shape-first stratified split.")
    p.add_argument("--late_priority_ratio", type=float, default=None,
                   help="While the late gate is violated, lambda(late) is floored at this ratio of max(lambda_G, "
                        "lambda_EM).")
    p.add_argument("--anomaly_lambda_cap_ratio", type=float, default=None,
                   help="Lambda(anomaly_late) <= this ratio x lambda(late).")
    p.add_argument("--event_auroc_open", type=float, default=None,
                   help="Probe AUROC at which the event residual's expression "
                        "scale reaches 1.0 (default 0.60; muted at/below the chance band).")
    p.add_argument("--forward_late_recon_weight", type=float, default=None,
                   help="Weight of forward-library rows' late gates in the z-domain reconstruction terms (default"
                        " 0.5; 1.0 = unweighted).")
    p.add_argument("--forward_fraction_final", type=float, default=None,
                   help="Decay the forward-library mixture fraction to this value (cosine, over the whole "
                        "training budget).")
    p.add_argument("--prior_band_start", type=int, default=None,
                   help="First gate (0-based) of the prior-assisted band: -1 = measured noise jump (default), "
                        "0 = off (fallback ramp), else the gate.")
    p.add_argument("--blend_floor_ramp_gates", type=int, default=None,
                   help="Gates over which the late blend floor ramps in, ending at the late-window start (default"
                        " 6).")
    p.add_argument("--alpha_gate_eq_row", type=float, default=None,
                   help="Row-balanced gate-equalised full-band teacher of the fused output (default 48; 0 "
                        "disables).")
    p.add_argument("--alpha_gate_balanced", type=float, default=None,
                   help="Weight of the gate-balanced error term (each gate normalized by its own clean RMS, all "
                        "gates equal).")
    p.add_argument("--field_blend_late_max", type=float, default=None,
                   help="Cap the late blend at field inference (e.g. 0.7 = the measured/denoised path carries "
                        ">=30%% of the late window).")
    p.add_argument("--no_field_route_autopsy", action="store_true",
                   help="Skip the single-trace counterfactual pass and the "
                        "route autopsy during field inference.")
    p.add_argument("--field_export_single_trace", action="store_true",
                   help="Export the field product from the single-trace counterfactual (neighbour slots = the "
                        "station's own trace): no lateral mixing, no lateral evidence.")
    p.add_argument("--val_blend_floor_max", type=float, default=None,
                   help="Hard ceiling on the validation-driven late blend floor (default 0.93)")
    p.add_argument("--no_val_blend_floor", action="store_true",
                   help="Disable the validation-driven late blend floor.")
    p.add_argument("--legacy_prior_fusion", action="store_true",
                   help="Restore the legacy prior-primary late fusion bit-for-bit (witnessed floors + validation "
                        "floor + redirect clamp the forward again).")
    p.add_argument("--continuation_arm_frac", type=float, default=None,
                   help="Fraction of each batch re-forwarded with the late window replaced by its measured noise "
                        "(E11 as a training task; whole patches, <=4 patches/step).")
    p.add_argument("--alpha_denoise_seam", type=float, default=None,
                   help="Weight of the direct arm's gate-balanced z-L1 on the seam band "
                        "[prior band start, late start).")
    p.add_argument("--alpha_denoise_early", type=float, default=None,
                   help="Weight of the direct arm's gate-balanced z-L1 on the early band.")
    p.add_argument("--z_band_gain", type=float, default=None,
                   help="Explicit, batch-invariant gain of the z-band arm terms (prior early/late, denoise "
                        "leash/accuracy/seam/early).")
    p.add_argument("--z_band_balance_target", type=float, default=None,
                   help=">0 adapts z_band_gain at every validation so that the arms' shared-trunk gradient "
                        "(prior_z+denoise_z) / contract group -> target; 0 = fixed gain.")
    p.add_argument("--z_band_balance_ema", type=float, default=None,
                   help="EMA weight of the gain-free audit log-ratio across validations (default 0.8).")
    p.add_argument("--z_band_balance_max_step", type=float, default=None,
                   help="Largest multiplicative z_band_gain step per validation (default 1.1).")
    p.add_argument("--z_band_balance_deadband", type=float, default=None,
                   help="Relative deadband around the target inside which the gain is held (default 0.2).")
    p.add_argument("--no_seam_floor_all_gates", action="store_true",
                   help="(kept for compatibility; the seam-band-only floor is now the default).")
    p.add_argument("--seam_floor_all_gates", action="store_true",
                   help="Install the validation-calibrated floor on every gate below the late start (population-"
                        "optimal constant; not a per-row safe floor -- ablation only).")
    p.add_argument("--seam_blend_floor_mode", type=str, default=None, choices=["auto", "hard", "soft", "off"],
                   help="Auto (default) = hard on a seam of <= 8 gates, soft on a wider prior band; hard = "
                        "forward max(head, floor)")
    p.add_argument("--uncovered_gate_policy", type=str, default=None, choices=["refuse", "hold_edge", "nan"],
                   help="What to do with target gates outside a source's time support (extra clean, noise "
                        "library): refuse = disable that source (default)")
    p.add_argument("--no_extra_clean", action="store_true", help="Disable the extra clean-only dataset.")
    p.add_argument("--no_measured_noise_library", action="store_true", help="Disable the measured-noise library.")
    p.add_argument("--no_seam_blend_floor", action="store_true",
                   help="Disable the seam blend floor in the forward (a loaded floor buffer is ignored, no floor "
                        "is queued).")
    p.add_argument("--no_lateral_witness_measurement_floor", action="store_true",
                   help="Let the noise-head witness alone set the lateral lambda factor "
                        "(default: floored by the measurement-only witness).")
    p.add_argument("--alpha_denoise_contract", type=float, default=None,
                   help="Weight of the trunk's direct pooled late-NRMSE contract supervision (rides the boost).")
    p.add_argument("--alpha_denoise_ps_late", type=float, default=None,
                   help="Weight of the trunk's equal-per-curve late NRMSE "
                        "(clamped at 300%%).")
    p.add_argument("--alpha_continuation", type=float, default=None,
                   help="Weight of the continuation-arm loss.")
    p.add_argument("--anomaly_visibility_floor", type=float, default=None,
                   help="Minimum weight of the late-window anomaly recovery supervision on bodies invisible at "
                        "the sample's realized late-local SNR.")
    p.add_argument("--no_library_subspace", action="store_true",
                   help="Disable the survey-library subspace anchor "
                        "(basis build, closed-form coefficient supervision, and "
                        "the output-side gate).")
    p.add_argument("--library_subspace_rank", type=int, default=None,
                   help="Rank of the clean-library subspace (default 12).")
    p.add_argument("--alpha_library_coef", type=float, default=None,
                   help="Weight of the closed-form coefficient "
                        "supervision.")
    p.add_argument("--acceptance_balance_share", type=float, default=None,
                   help="Target ||grad(acceptance group)|| / ||grad(rest)|| on the "
                        "shared trunk (default 0.5; 0 disables balancing).")
    p.add_argument("--acceptance_balance_row_late_max", type=float, default=None,
                   help="Arm balancing only once the previous epoch's mean per-row "
                        "late NRMSE is <= this (default 0.10).")
    p.add_argument("--no_preflight", action="store_true",
                   help="Skip the pre-flight self-test before epoch 1.")
    p.add_argument("--preflight_continue", action="store_true",
                   help="Continue training even if a pre-flight check fails.")
    p.add_argument("--acceptance_scale_init", type=float, default=None,
                   help="Static scale of the acceptance group before/without "
                        "balancing (default 10).")
    p.add_argument("--raw_ridge_continuation", action="store_true",
                   help="Install the raw ridge operator instead of the projected (family-constrained) "
                        "continuation.")
    p.add_argument("--no_ridge_anchor", action="store_true",
                   help="Disable the installed library ridge "
                        "continuation operator.")
    p.add_argument("--alpha_order", type=float, default=None,
                   help="Weight of the late cross-gate order penalty "
                        "(0 disables).")
    p.add_argument("--alpha_gate_deep", type=float, default=None,
                   help="Weight of the deep-background anchor-gate "
                        "supervision (0 disables).")
    p.add_argument("--alpha_recovery", type=float, default=None,
                   help="Weight of the per-row recovery calibration "
                        "(rec->1, weak strata 3x; 0 disables).")
    p.add_argument("--alpha_diff_orth", type=float, default=None,
                   help="Weight of the orthogonal shape-error term "
                        "(0 disables).")
    p.add_argument("--alpha_anchor_late_ps", type=float, default=None,
                   help="Weight of the per-row anchor late NRMSE "
                        "(subspace + ridge; 0 disables).")
    p.add_argument("--alpha_late_per_sample", type=float, default=None,
                   help="Weight of the per-row fused late NRMSE (count-based, the acceptance's own mean; 0 "
                        "disables).")
    p.add_argument("--no_late_cvar_rebase", action="store_true",
                   help="Keep the contract late-CVaR gate for control and "
                        "selection even when the startup ceiling proves it "
                        "population-limited on this split.")
    p.add_argument("--late_cvar_rebase_margin", type=float, default=None,
                   help="Margin over the measured ceiling used when the "
                        "tail gate is rebased (default 0.10 = ceiling x 1.10).")
    p.add_argument("--amp_dtype", type=str, default=None, choices=["auto", "bf16", "fp16"],
                   help="Mixed-precision format.")
    p.add_argument("--no_tf32", action="store_true",
                   help="Disable TF32 for the float32 ops (slower, marginally more precise).")
    p.add_argument("--micro_batch_size", type=int, default=None,
                   help="Force the per-forward micro-batch (0 = measure the card and choose automatically).")
    p.add_argument("--vram_budget_fraction", type=float, default=None,
                   help="Fraction of total VRAM the training step may use (default 0.72).")
    p.add_argument("--no_vram_autotune", action="store_true",
                   help="Disable the memory probe and use --batch_size as given.")
    p.add_argument("--decoupled_arms", action="store_true",
                   help="Opt-in: closed-form completion prior + NLL-calibrated fusion (A/B only).")
    p.add_argument("--gate_local_norm", action="store_true",
                   help="Opt-in: gate-local trunk normalisation instead of GroupNorm (A/B only).")
    p.add_argument("--no_gate_local_norm", action="store_true",
                   help="Force the cross-gate GroupNorm trunk (the default).")
    p.add_argument("--decoder_gate_norm_channel", action="store_true",
                   help="Legacy: re-enable the symlog(x/gs^2) decoder slot.")
    p.add_argument("--anchor_completion_only", action="store_true",
                   help="Opt-in: library anchor coefficients = completion A[c_B,1] only, no per-mode mix "
                        "(worse on noisy input in the proxy A/B; A/B only).")
    p.add_argument("--mmse_blend_in_training", action="store_true",
                   help="Opt-in: apply the witnessed MMSE blend during training too (regressed in the sandbox; A/B only).")
    p.add_argument("--mmse_eval_every", type=int, default=None,
                   help="Run the deployment compare (extra validation pass with the MMSE blend) every N epochs (default 2).")
    p.add_argument("--no_mmse_blend", action="store_true",
                   help="Restore the learned late gate with its floors and caps (A/B only).")
    p.add_argument("--no_bounded_input", action="store_true",
                   help="Do not clip the working-domain input z to input_z_bounds.")
    p.add_argument("--completion_gate_limit", type=int, default=None,
                   help="Highest gate (exclusive) the completion prior may read; 0 = auto from the pair noise profile.")
    p.add_argument("--no_native_presentation", action="store_true",
                   help="Draw and export on the 256-gate model axis instead of the measured 31-gate axis.")
    p.add_argument("--information_bound_only", action="store_true",
                   help="In --mode figures: run only the information-bound audit (E1-E6, contract units) and "
                        "exit.")
    p.add_argument("--sim_noise_fraction", type=float, default=None,
                   help="Share of training patches whose noise is the online simulated late-time noise (default "
                        "0.20; 0 = off).")
    p.add_argument("--sim_noise_snr_db", type=str, default=None,
                   help="Designed late-SNR range 'lo,hi' (dB) of the simulated noise (default -30,10).")
    p.add_argument("--tail_loss_prob", type=float, default=None,
                   help="Share of training patches that withhold their late tail (gate-loss augmentation; "
                        "default 0.15, 0 = off).")
    p.add_argument("--tail_loss_fraction", type=str, default=None,
                   help="Withheld fraction range 'lo,hi' of the augmentation (default 0.05,0.75).")
    p.add_argument("--no_gate_loss_continuation", action="store_true",
                   help="Do not continue withheld gates by CDM-R in the forward (the trunk then reads whatever the "
                        "withheld gates hold).")
    p.add_argument("--gate_loss_rate_bound", choices=["off", "input"], default=None,
                   help="Hold the local relaxation rate of the continued tail in [0, nu*] (Eq. 4); default off.")
    p.add_argument("--informative_continuation", choices=["off", "on"], default=None,
                   help="Inference: remove the late window beyond each trace's last informative gate (decided by "
                        "the network's own reliability) and continue it by CDM-R.")
    p.add_argument("--no_deployment_gate", action="store_true",
                   help="Record the deployment verdict without enforcing it (exit code 0, no unqualified tags; "
                        "the setting of the public release).")
    p.add_argument("--no_profile_extras", action="store_true",
                   help="Only the profile comparisons next to the profile figures (no best 5 / worst 5 / bodies "
                        "/ weak-anomaly figures, no worst-section autopsies).")
    p.add_argument("--paired_unit", type=str, default=None,
                   help="Amplitude unit of the paired dataset, shown on the figure axes as log10(unit) "
                        "(default nT/s).")
    p.add_argument("--no_cdmr_calibration", action="store_true",
                   help="CDM-R with the stack-scatter likelihood instead of the measured noise covariance and "
                        "the mixture prior.")
    p.add_argument("--gate_loss_withheld", type=str, default=None,
                   help="Withheld gates per 31 of the gate-loss suite, comma list (default 3,6,12,18,19,22).")
    p.add_argument("--gate_loss_profiles", type=int, default=None,
                   help="Test profiles scored by the gate-loss suite (default 240, evenly spaced; 0 = all).")
    p.add_argument("--gate_loss_fan_withheld", type=int, default=None,
                   help="Withheld gates (per 31) of the common error fan and the input-quality clusters "
                        "(default 18 = 58.1%%, the manuscript's Fig. 10).")
    p.add_argument("--no_gate_loss_noise_tail", action="store_true",
                   help="Skip the no-continuation baseline of the gate-loss suite.")
    p.add_argument("--evidence_uninformed_cut", action="store_true",
                   help="E1/E11: do not tell the network where the evidence ends (the earlier protocol).")
    p.add_argument("--no_family_completion", action="store_true",
                   help="Do not fit the survey family's own completion map: the library anchor completes every "
                        "row with the paired family's map.")
    p.add_argument("--no_family_gate_floor", action="store_true",
                   help="Keep the family completion but leave the anchor's gate to the learned head alone.")
    p.add_argument("--family_common_mode_stations", type=int, default=None,
                   help="Half-width (stations) of the running median that defines the laterally common part of "
                        "(completion - network) the family's share acts on.")
    p.add_argument("--no_family_validated_fusion", action="store_true",
                   help="Do not measure the fusion share on the validation stations: the family completion keeps "
                        "its full share of a buried gate (kappa = 1).")
    p.add_argument("--survey_val_blocks", type=int, default=None,
                   help="Number of validation blocks of the measured section, placed by rule at equal spacing "
                        "(default 2; 0 = train every station, no held-out evidence).")
    p.add_argument("--survey_val_block_len", type=int, default=None,
                   help="Stations per validation block (default 5).")
    p.add_argument("--survey_val_fold", type=int, default=None,
                   help="Cyclic shift of the validation blocks in block lengths (default 0): the rotation that "
                        "makes every station a validation station in some fold.")
    p.add_argument("--survey_background_window", type=int, default=None,
                   help="Stations of the running median behind the survey rows' event labels (default 0 = chosen "
                        "from the section's own data by the rule).")
    p.add_argument("--anchor_compare_every", type=int, default=None,
                   help="Every N epochs one extra validation pass with both anchors withdrawn, printed beside the "
                        "regular one (monitor only; default 4, 0 = off).")
    p.add_argument("--tail_panel_areas", type=int, default=None,
                   help="Number of hardest validation sections scored stage by stage at every validation "
                        "(default 8; 0 = off).")
    p.add_argument("--section_autopsy_only", action="store_true",
                   help="In --mode figures: run only the test/validation profile sections with the stage autopsy "
                        "of their 3 worst areas and the measured section against its.")
    p.add_argument("--edge_reflect_mode", type=str, default=None, choices=["auto", "none", "window", "full"],
                   help="Edge-extension mode of the deliverable path.")
    p.add_argument("--startup_audits", type=str, default=None, choices=["fast", "full"],
                   help="Fast (default): only the start-up audits that install something in the model; full: also"
                        " the linear information ladders (about 2 h on one GPU).")
    p.add_argument("--receptive_field_scaling", type=str, default=None, choices=["auto", "off"],
                   help="Auto (default): rescale the trunk dilations to the gate axis (256 gates -> residual "
                        "1,2,4,8,16,32 / multiscale 1,2,4,8); off: keep the 31-gate design.")
    p.add_argument("--dropout", type=float, default=None,
                   help="Feature dropout (default 0.0; any value > 0 also switches the auxiliary arms "
                        "back to full-batch forwards).")
    p.add_argument("--no_aux_arms_subbatch", action="store_true",
                   help="Run the background and alternate arms on the full batch (legacy).")
    p.add_argument("--no_vram_fill_real_step", action="store_true",
                   help="Keep the worst-case VRAM sizing (every row flagged for both auxiliary arms).")
    p.add_argument("--aux_cap_sigma", type=float, default=None,
                   help="Auxiliary-arm cap = mean flag share + this many binomial sigmas (default 2.0).")
    p.add_argument("--optimizer_batch_policy", type=str, default=None, choices=["micro", "legacy"],
                   help="Micro (default): one optimizer step per micro-batch (>= --min_effective_rows rows); "
                        "legacy: the effective batch by accumulation.")
    p.add_argument("--min_effective_rows", type=int, default=None,
                   help="Smallest number of rows one optimizer step may average over (default 288 = 32 patches).")
    p.add_argument("--dp_primary_share", type=float, default=None,
                   help="Share of each micro-batch's patches on the primary card (0 = balance it from the "
                        "measured per-card memory slopes; 0.5 = equal split).")
    p.add_argument("--vram_step_margin", type=float, default=None,
                   help="Headroom above the fitted per-row cost of the real training step (default 0.10).")
    p.add_argument("--activation_checkpointing", type=str, default=None, choices=["on", "off", "auto"],
                   help="Recompute the refinement stack in backward (~3x less activation memory, ~20%% more "
                        "compute).")
    p.add_argument("--no_shm_free_transport", action="store_true",
                   help="Send loader batches through shared memory (old path) instead of numpy bytes; "
                        "re-enables the /dev/shm audit.")
    p.add_argument("--patch_batch_cap", type=int, default=None,
                   help="Ceiling for the automatic batch enlargement in profile-patch mode (default 2048).")
    p.add_argument("--profile_patch_len", type=int, default=None,
                   help="Train on contiguous station patches of this length so the objective can constrain the "
                        "profile direction: lateral first and second differences.")
    p.add_argument("--anomaly_suite_per_cell", type=int, default=None,
                   help="Cases per required qualification cell (family x realized contrast bin x SNR stratum).")
    p.add_argument("--anomaly_suite_stations", type=int, default=None,
                   help="Length of the contiguous held-out station segment each test "
                        "profile is built from (default 25).")
    p.add_argument("--init_from", type=str, default="",
                   help="Parent checkpoint to initialize from when promoting up the q0->q4 ladder.")
    p.add_argument("--init_equivalence_max", type=float, default=None,
                   help="Maximum relative output perturbation a promotion may cause (default 1e-5).")
    p.add_argument("--confirm_full_from_scratch", action="store_true",
                   help="Required to start --ablation full from random init.")
    p.add_argument("--late_cvar_threshold", type=float, default=None,
                   help="Late-window CVaR contract gate as a fraction (default 0.08 = 8%%).")
    p.add_argument("--allow_amplitude_drift", action="store_true",
                   help="Debug only.")
    p.add_argument("--neighbor_distance_prior", type=float, default=None,
                   help="Initial per-head distance-prior strength on |offset| in median-spacing units.")
    p.add_argument("--no_forward_admission_filter", action="store_true",
                   help="Admit every forward-simulation curve regardless of the field "
                        "descriptor envelope.")
    p.add_argument("--halt_on_nan", action="store_true",
                   help="Stop at the first non-finite loss with a full forensic report "
                        "instead of skipping the step (debugging).")
    p.add_argument("--skip_figure_package", action="store_true",
                   help="In --mode all, stop after training + field inference instead of "
                        "running the full test and figure package (24 figures).")
    p.add_argument("--evidence_skip_ip", action="store_true",
                   help="Run only E1-E7 in evidence mode (skip the IP identification suite).")
    p.add_argument("--doi_sigma", type=float, default=RuntimeConfig.doi_sigma_s_per_m,
                   help="Host conductivity (S/m) used to convert the last reliably "
                        "reconstructed gate into a diffusion depth of investigation.")
    p.add_argument("--doi_tolerance", type=float, default=RuntimeConfig.doi_gate_tolerance_percent,
                   help="Cumulative-NRMSE tolerance (%%) defining the last reliable gate.")
    p.add_argument("--ip_profiles", type=int, default=RuntimeConfig.ip_profiles,
                   help="Synthetic borehole profiles in the IP suite.")
    p.add_argument("--ip_stations", type=int, default=RuntimeConfig.ip_stations,
                   help="Stations per synthetic profile.")
    p.add_argument("--ip_score_threshold", type=float, default=RuntimeConfig.ip_score_threshold,
                   help="Fused decay+profile matched-filter score (in sigma) above which polarization is "
                        "declared.")
    p.add_argument("--ip_min_amplitude_sigma", type=float, default=RuntimeConfig.ip_min_amplitude_sigma,
                   help="Absolute floor (in noise sigma) on the fitted polarization amplitude.")
    p.add_argument("--ip_guard_hard", action="store_true",
                   help="Hard 0/1 preservation gate instead of the soft ramp.")
    p.add_argument("--enable_ip_head", action="store_true",
                   help="Add the signed dual-spectrum IP head to the model (off by default).")
    p.add_argument("--no_torchscript", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.paths.clean_csv = args.clean_csv
    cfg.paths.clean_dir = "" if getattr(args, "no_clean_dir", False) else (getattr(args, "clean_dir", "") or "")
    if getattr(args, "no_mix_clean_csv", False):
        cfg.data.mix_clean_csv = False
    if getattr(args, "mix_clean_csv_max_rows", None) is not None:
        cfg.data.mix_clean_csv_max_rows = max(0, int(args.mix_clean_csv_max_rows))
    if getattr(args, "no_stack_slot", False):
        cfg.data.coherent_stack_slot = False
    if getattr(args, "neighbor_channel_dropout", None) is not None:
        cfg.data.neighbor_channel_dropout = min(0.5, max(0.0, float(args.neighbor_channel_dropout)))
    if getattr(args, "mix_forward_fraction", None) is not None:
        cfg.data.mix_forward_fraction = min(0.9, max(0.0, float(args.mix_forward_fraction)))
    if getattr(args, "field_group_size", None) is not None:
        cfg.data.field_group_size = max(1, int(args.field_group_size))
    if getattr(args, "province_val_fraction", None) is not None:
        cfg.data.province_val_fraction = float(args.province_val_fraction)
    if getattr(args, "province_test_fraction", None) is not None:
        cfg.data.province_test_fraction = float(args.province_test_fraction)
    if getattr(args, "province_test_snr_db", None) is not None:
        cfg.data.province_test_snr_db_range = (
            float(args.province_test_snr_db[0]), float(args.province_test_snr_db[1])
        )
    cfg.paths.paired_dir = getattr(args, "paired_dir", "") or ""
    if getattr(args, "no_paired", False):
        cfg.data.paired_supervision = False
    if getattr(args, "paired_background_window", None) is not None:
        cfg.data.paired_background_window = max(3, int(args.paired_background_window))
    if getattr(args, "paired_event_threshold", None) is not None:
        cfg.data.paired_event_threshold = max(1e-4, float(args.paired_event_threshold))
    if getattr(args, "paired_split_policy", None) is not None:
        cfg.data.paired_split_policy = str(args.paired_split_policy)
    if getattr(args, "paired_copy_rel_tol", None) is not None:
        cfg.data.paired_copy_rel_tol = max(0.0, float(args.paired_copy_rel_tol))
    if getattr(args, "no_paired_neighbors", False):
        cfg.data.paired_real_neighbors = False
    if getattr(args, "alpha_gate_monotonic", None) is not None:
        cfg.loss.alpha_gate_monotonic = max(0.0, float(args.alpha_gate_monotonic))
    if getattr(args, "no_gate_monotonic", False):
        cfg.data.gate_monotonic_projection = False
    if getattr(args, "paired_anomaly_oversample", None) is not None:
        cfg.data.paired_anomaly_oversample = float(
            np.clip(args.paired_anomaly_oversample, 0.0, 0.9))
    if getattr(args, "paired_injected_probe", False):
        cfg.runtime.paired_injected_probe = True
    cfg.paths.noise_library_dir = getattr(args, "noise_library_dir", "") or ""
    if getattr(args, "measured_noise_inject", None) is not None:
        cfg.data.measured_noise_inject = float(
            np.clip(args.measured_noise_inject, 0.0, 0.9))
    if getattr(args, "paired_residual_replay", None) is not None:
        cfg.data.paired_residual_replay = float(
            np.clip(args.paired_residual_replay, 0.0, 0.9))
    if getattr(args, "neighbor_noise_equicorrelation", None) is not None:
        cfg.model.neighbor_noise_equicorrelation = float(
            np.clip(args.neighbor_noise_equicorrelation, 0.0, 0.999))
    if getattr(args, "gate_monotonic_projection", False):
        cfg.data.gate_monotonic_projection = True
    if getattr(args, "paired_alt_corruption", None) is not None:
        cfg.data.paired_alt_corruption = float(np.clip(args.paired_alt_corruption, 0.0, 1.0))
    if getattr(args, "alpha_real_noise_invariance", None) is not None:
        cfg.loss.alpha_real_noise_invariance = max(
            0.0, float(args.alpha_real_noise_invariance))
    if getattr(args, "alpha_noise_orthogonal", None) is not None:
        cfg.loss.alpha_noise_orthogonal = max(0.0, float(args.alpha_noise_orthogonal))
    if getattr(args, "alpha_roughness_cap", None) is not None:
        cfg.loss.alpha_roughness_cap = max(0.0, float(args.alpha_roughness_cap))
    if getattr(args, "paired_hard_snr_oversample", None) is not None:
        cfg.data.paired_hard_snr_oversample = float(
            np.clip(args.paired_hard_snr_oversample, 0.0, 0.9))
    cfg.paths.noise_dir = args.noise_dir
    cfg.paths.field_csv = "" if bool(getattr(args, "no_field", False)) else args.field_csv
    if bool(getattr(args, "no_unify_axis", False)):
        cfg.data.unify_axis_to_measured = False
    if bool(getattr(args, "single_gpu", False)):
        cfg.runtime.multi_gpu = False
    if getattr(args, "unified_gates", None) is not None:
        cfg.data.unified_gates = max(0, int(args.unified_gates))
    if getattr(args, "field_native_gates", None) is not None:
        cfg.data.field_native_gates = max(2, int(args.field_native_gates))
    if getattr(args, "field_gate_table", None):
        cfg.paths.field_gate_table = str(args.field_gate_table)
    if getattr(args, "field_gate_time_unit", None):
        cfg.data.field_gate_time_unit = str(args.field_gate_time_unit)
    cfg.paths.output_root = args.output_root
    cfg.paths.model_root = getattr(args, "model_root", "") or ""
    cfg.paths.plot_root = getattr(args, "plot_root", "") or ""
    cfg.paths.log_root = getattr(args, "log_root", "") or ""
    if getattr(args, "val_snr_protocol", None) is not None:
        cfg.data.val_contract_snr_protocol = (args.val_snr_protocol == "contract")
    cfg.paths.resume_checkpoint = args.checkpoint if args.mode == "train" else ""
    cfg.runtime.mode = args.mode
    cfg.runtime.run_name = args.run_name
    if getattr(args, "vram_gb", None) is not None or getattr(args, "gpu_large", False):
        _gb = float(getattr(args, "vram_gb", None) or 0.0)
        if _gb <= 0.0:
            _gb = 16.0
            try:
                if torch.cuda.is_available():
                    _d = resolve_device(args.device)
                    if _d.type == "cuda":
                        _fi, _ti = torch.cuda.mem_get_info(
                            _d.index if _d.index is not None else 0)
                        _gb = float(_fi) / float(2 ** 30)
            except Exception:
                _gb = 16.0
        _base_bs, _base_lr = 768.0, float(TrainConfig.base_lr)
        _arms = float(getattr(args, "aux_arms_memory_factor", None) or 2.2)
        _scale = max(1.0, _gb / 16.0) / max(1.0, _arms)
        _bs = int(max(256, min(16384, round(_base_bs * _scale / 64.0) * 64)))
        _lr_mult = math.sqrt(_bs / _base_bs)
        cfg.train.large_vram_preset = {
            "card_gb": _gb,
            "batch_size": _bs,
            "num_workers": int(max(4, min(16, round(4 * math.sqrt(_scale))))),
            "train_samples_per_epoch": int(max(60_000, min(600_000,
                                                           round(60_000 * _scale / 1000) * 1000))),
            "val_samples": int(max(8_000, min(40_000,
                                              round(8_000 * math.sqrt(_scale) / 500) * 500))),
            "lr_multiplier": _lr_mult,
            "base_lr": float(_base_lr * _lr_mult),
        }
    if getattr(args, "gpu16", False):
        cfg.train.batch_size = 768
        cfg.train.num_workers = 4
        cfg.train.persistent_workers = True
        cfg.train.pin_memory = True
        cfg.data.train_samples_per_epoch = 60_000
        cfg.data.val_samples = 8_000
        cfg.data.test_samples = 8_000
    cfg.runtime.device = args.device
    cfg.runtime.overwrite_cache = bool(args.overwrite_cache)
    cfg.runtime.plot = not args.no_plot
    cfg.runtime.figure_set = getattr(args, "figure_set", "full")
    cfg.runtime.export_torchscript = not args.no_torchscript
    try:
        fr = tuple(sorted({round(float(x), 4) for x in str(args.evidence_missing_fractions).split(",") if x.strip()}))
    except ValueError as exc:
        raise ValueError(
            "--evidence_missing_fractions must be a comma list of numbers, e.g. 0.1,0.2,0.3,0.4,0.5"
        ) from exc
    if not fr or any(not (0.03 <= f <= 0.70) for f in fr):
        raise ValueError("--evidence_missing_fractions entries must lie in [0.03, 0.70].")
    cfg.runtime.evidence_missing_fractions = fr
    cfg.runtime.evidence_missing_scope = str(args.evidence_missing_scope)
    cfg.runtime.evidence_samples = max(32, int(args.evidence_samples))
    cfg.runtime.evidence_tail_gates = max(3, int(args.evidence_tail_gates))
    cfg.runtime.evidence_center_only = bool(args.evidence_center_only)
    cfg.runtime.evidence_bootstrap = max(0, int(args.evidence_bootstrap))
    cfg.runtime.evidence_skip_ip = bool(args.evidence_skip_ip)
    cfg.runtime.skip_figure_package = bool(args.skip_figure_package)
    cfg.runtime.halt_on_nan = bool(args.halt_on_nan)
    cfg.runtime.force_infer_unqualified = bool(getattr(args, "force_infer_unqualified", False))
    if getattr(args, "no_field_alignment", False):
        cfg.runtime.field_amplitude_alignment = False
    if getattr(args, "contract_test_snr_db", None) is not None:
        cfg.data.contract_test_snr_db_range = (
            float(args.contract_test_snr_db[0]), float(args.contract_test_snr_db[1])
        )
    if getattr(args, "stage_regression_min_epochs", None) is not None:
        cfg.train.stage_regression_min_epochs = max(0, int(args.stage_regression_min_epochs))
    if getattr(args, "physics_residual_scale", None) is not None:
        cfg.model.physics_residual_scale = max(0.0, float(args.physics_residual_scale))
    if getattr(args, "no_relaxation_states", False):
        cfg.model.relaxation_stretch_exponents = (1.0,)
    if getattr(args, "relaxation_update_scale", None) is not None:
        cfg.model.relaxation_update_scale = max(0.0, float(args.relaxation_update_scale))
    if getattr(args, "no_relaxation_state_update", False):
        cfg.model.use_relaxation_state_update = False
    if getattr(args, "no_manifold_quality_gate", False):
        cfg.model.manifold_quality_gate = False
    if getattr(args, "no_forward_admission_filter", False):
        cfg.data.forward_admission_filter = False
    if getattr(args, "no_neighbor_geometry", False):
        cfg.model.use_neighbor_geometry = False
    if getattr(args, "no_event_residual", False):
        cfg.model.use_event_residual = False
    if getattr(args, "struct_residual_scale", None) is not None:
        cfg.model.struct_residual_scale = max(0.0, float(args.struct_residual_scale))
    if getattr(args, "anomaly_profile_prob", None) is not None:
        cfg.data.anomaly_profile_prob = min(max(float(args.anomaly_profile_prob), 0.0), 1.0)
    if getattr(args, "allow_prior_monopoly", False):
        cfg.runtime.allow_prior_monopoly = True
    _preset = getattr(args, "field_units_preset", None)
    if _preset:
        _pv = {"nt_s_ms": ("dB/dt", "nT/s", "ms",
                           "raw, no area or moment normalization"),
               "nt_s_s": ("dB/dt", "nT/s", "s",
                          "raw, no area or moment normalization")}[str(_preset)]
        for _a, _v in zip(("field_signal_quantity", "field_amplitude_unit",
                           "field_time_unit", "field_normalization_history"), _pv):
            if getattr(args, _a, None) is None:
                setattr(args, _a, _v)
    for _a, _c in (("field_signal_quantity", "field_signal_quantity"),
                   ("field_amplitude_unit", "field_amplitude_unit"),
                   ("field_time_unit", "field_time_unit"),
                   ("field_normalization_history", "field_normalization_history")):
        if getattr(args, _a, None) is not None:
            setattr(cfg.data, _c, str(getattr(args, _a)))
    if getattr(args, "field_receiver_area_m2", None) is not None:
        cfg.data.field_receiver_area_m2 = float(args.field_receiver_area_m2)
    if getattr(args, "field_transmitter_moment_am2", None) is not None:
        cfg.data.field_transmitter_moment_am2 = float(args.field_transmitter_moment_am2)
    if getattr(args, "no_split_dedup_by_shape", False):
        cfg.data.split_dedup_by_shape = False
    if getattr(args, "v1122_parity", False):
        cfg.model.v1122_parity = True
    if getattr(args, "stage_budget_legacy", False):
        cfg.train.stage_a_max_epochs = 120
        cfg.train.stage_b_max_epochs = 0
        cfg.train.stage_c_epochs = 60
        cfg.train.stage_c_lr_factor = 0.04
    if getattr(args, "no_split_shape_first", False):
        cfg.data.split_shape_first = False
    if getattr(args, "late_priority_ratio", None) is not None:
        cfg.contract.late_priority_ratio = float(args.late_priority_ratio)
    if getattr(args, "anomaly_lambda_cap_ratio", None) is not None:
        cfg.contract.anomaly_lambda_cap_ratio = float(args.anomaly_lambda_cap_ratio)
    if getattr(args, "event_auroc_open", None) is not None:
        cfg.contract.event_auroc_open = float(args.event_auroc_open)
    if getattr(args, "forward_late_recon_weight", None) is not None:
        cfg.loss.forward_late_recon_weight = float(args.forward_late_recon_weight)
    if getattr(args, "no_val_blend_floor", False):
        cfg.loss.val_blend_floor = False
    if getattr(args, "legacy_prior_fusion", False):
        cfg.model.supervised_primary_fusion = False
    if getattr(args, "continuation_arm_frac", None) is not None:
        cfg.model.continuation_arm_frac = max(0.0, min(1.0, float(args.continuation_arm_frac)))
    if getattr(args, "alpha_denoise_contract", None) is not None:
        cfg.loss.alpha_denoise_contract = max(0.0, float(args.alpha_denoise_contract))
    if getattr(args, "alpha_denoise_seam", None) is not None:
        cfg.loss.alpha_denoise_seam = max(0.0, float(args.alpha_denoise_seam))
    if getattr(args, "alpha_denoise_early", None) is not None:
        cfg.loss.alpha_denoise_early = max(0.0, float(args.alpha_denoise_early))
    if getattr(args, "z_band_gain", None) is not None:
        cfg.loss.z_band_gain = max(0.0, float(args.z_band_gain))
    if getattr(args, "z_band_balance_target", None) is not None:
        cfg.loss.z_band_balance_target = max(0.0, float(args.z_band_balance_target))
    if getattr(args, "z_band_balance_ema", None) is not None:
        cfg.loss.z_band_balance_ema = min(0.99, max(0.0, float(args.z_band_balance_ema)))
    if getattr(args, "z_band_balance_max_step", None) is not None:
        cfg.loss.z_band_balance_max_step = max(1.0, float(args.z_band_balance_max_step))
    if getattr(args, "z_band_balance_deadband", None) is not None:
        cfg.loss.z_band_balance_deadband = max(0.0, float(args.z_band_balance_deadband))
    if bool(getattr(args, "no_seam_floor_all_gates", False)):
        cfg.model.seam_blend_floor_all_gates = False
    if bool(getattr(args, "seam_floor_all_gates", False)):
        cfg.model.seam_blend_floor_all_gates = True
    if getattr(args, "seam_blend_floor_mode", None) is not None:
        cfg.model.seam_blend_floor_mode = str(args.seam_blend_floor_mode)
    if getattr(args, "uncovered_gate_policy", None) is not None:
        cfg.data.uncovered_gate_policy = str(args.uncovered_gate_policy)
    if bool(getattr(args, "no_extra_clean", False)):
        cfg.data.extra_clean_enabled = False
    if bool(getattr(args, "no_measured_noise_library", False)):
        cfg.data.measured_noise_enabled = False
    if bool(getattr(args, "no_seam_blend_floor", False)):
        cfg.model.seam_blend_floor_enabled = False
        cfg.model.seam_blend_floor_from_val = False
    if bool(getattr(args, "no_lateral_witness_measurement_floor", False)):
        cfg.model.lateral_witness_measurement_floor = False
    if getattr(args, "alpha_denoise_ps_late", None) is not None:
        cfg.loss.alpha_denoise_ps_late = max(0.0, float(args.alpha_denoise_ps_late))
    if getattr(args, "alpha_continuation", None) is not None:
        cfg.loss.alpha_continuation = max(0.0, float(args.alpha_continuation))
    if getattr(args, "anomaly_visibility_floor", None) is not None:
        cfg.loss.anomaly_visibility_floor = min(1.0, max(0.0, float(args.anomaly_visibility_floor)))
    if getattr(args, "no_library_subspace", False):
        cfg.model.use_library_subspace = False
    if getattr(args, "library_subspace_rank", None) is not None:
        cfg.model.library_subspace_rank = max(2, int(args.library_subspace_rank))
    if getattr(args, "alpha_library_coef", None) is not None:
        cfg.loss.alpha_library_coef = max(0.0, float(args.alpha_library_coef))
    if getattr(args, "acceptance_balance_share", None) is not None:
        cfg.loss.acceptance_balance_share = max(0.0, float(args.acceptance_balance_share))
    if getattr(args, "acceptance_balance_row_late_max", None) is not None:
        cfg.loss.acceptance_balance_row_late_max = max(0.0, float(args.acceptance_balance_row_late_max))
    if getattr(args, "acceptance_scale_init", None) is not None:
        cfg.loss.acceptance_scale_init = max(0.0, float(args.acceptance_scale_init))
    if bool(getattr(args, "no_preflight", False)):
        cfg.train.preflight = False
    if bool(getattr(args, "preflight_continue", False)):
        cfg.train.preflight_continue = True
    if getattr(args, "no_ridge_anchor", False):
        cfg.model.use_ridge_anchor = False
    if getattr(args, "raw_ridge_continuation", False):
        cfg.model.projection_continuation = False
    if getattr(args, "alpha_order", None) is not None:
        cfg.loss.alpha_order = max(0.0, float(args.alpha_order))
    if getattr(args, "alpha_gate_deep", None) is not None:
        cfg.loss.alpha_gate_deep = max(0.0, float(args.alpha_gate_deep))
    if getattr(args, "alpha_recovery", None) is not None:
        cfg.loss.alpha_recovery = max(0.0, float(args.alpha_recovery))
    if getattr(args, "alpha_diff_orth", None) is not None:
        cfg.loss.alpha_diff_orth = max(0.0, float(args.alpha_diff_orth))
    if getattr(args, "alpha_anchor_late_ps", None) is not None:
        cfg.loss.alpha_anchor_late_ps = max(
            0.0, float(args.alpha_anchor_late_ps))
    if getattr(args, "alpha_late_per_sample", None) is not None:
        cfg.loss.alpha_late_per_sample = max(
            0.0, float(args.alpha_late_per_sample))
    if getattr(args, "val_blend_floor_max", None) is not None:
        cfg.loss.val_blend_floor_max = float(args.val_blend_floor_max)
    if getattr(args, "no_field_route_autopsy", False):
        cfg.runtime.field_route_autopsy = False
    if getattr(args, "field_export_single_trace", False):
        cfg.runtime.field_export_single_trace = True
    if getattr(args, "field_blend_late_max", None) is not None:
        cfg.runtime.field_blend_late_max = float(args.field_blend_late_max)
    if getattr(args, "forward_fraction_final", None) is not None:
        cfg.data.forward_fraction_final = float(args.forward_fraction_final)
    if getattr(args, "prior_band_start", None) is not None:
        cfg.data.prior_band_start = int(args.prior_band_start)
    if getattr(args, "blend_floor_ramp_gates", None) is not None:
        cfg.model.blend_floor_ramp_gates = max(0, int(args.blend_floor_ramp_gates))
    if getattr(args, "alpha_gate_balanced", None) is not None:
        cfg.loss.alpha_gate_balanced = float(args.alpha_gate_balanced)
    if getattr(args, "alpha_gate_eq_row", None) is not None:
        cfg.loss.alpha_gate_eq_row = max(0.0, float(args.alpha_gate_eq_row))
    if getattr(args, "no_late_cvar_rebase", False):
        cfg.contract.late_cvar_rebase = False
    if getattr(args, "late_cvar_rebase_margin", None) is not None:
        cfg.contract.late_cvar_rebase_margin = float(args.late_cvar_rebase_margin)
    if getattr(args, "amp_dtype", None) is not None:
        cfg.train.amp_dtype = str(args.amp_dtype)
    if getattr(args, "no_tf32", False):
        cfg.train.allow_tf32 = False
    if getattr(args, "micro_batch_size", None) is not None:
        cfg.train.micro_batch_size = max(0, int(args.micro_batch_size))
    if getattr(args, "vram_budget_fraction", None) is not None:
        cfg.train.vram_budget_fraction = float(np.clip(args.vram_budget_fraction, 0.2, 0.95))
        cfg.train.vram_budget_fraction_explicit = True
    if getattr(args, "dp_primary_share", None) is not None:
        cfg.train.dp_primary_share = float(np.clip(args.dp_primary_share, 0.0, 0.5))
    if getattr(args, "startup_audits", None) is not None:
        cfg.runtime.startup_audits = str(args.startup_audits)
    if getattr(args, "edge_reflect_mode", None) is not None:
        cfg.data.edge_reflect_mode = str(args.edge_reflect_mode)
    if getattr(args, "tail_panel_areas", None) is not None:
        cfg.runtime.tail_panel_areas = max(0, int(args.tail_panel_areas))
    if getattr(args, "anchor_compare_every", None) is not None:
        cfg.runtime.anchor_compare_every = max(0, int(args.anchor_compare_every))
    if getattr(args, "no_family_completion", False):
        cfg.model.family_completion = False
    if getattr(args, "no_family_gate_floor", False):
        cfg.model.family_gate_floor = False
    if getattr(args, "tail_loss_prob", None) is not None:
        cfg.data.tail_loss_prob = max(0.0, min(1.0, float(args.tail_loss_prob)))
    if getattr(args, "tail_loss_fraction", None):
        try:
            _t = [float(v) for v in str(args.tail_loss_fraction).split(",") if str(v).strip()]
            if len(_t) != 2 or not (0.0 <= _t[0] < _t[1] <= 0.95):
                raise ValueError(args.tail_loss_fraction)
            cfg.data.tail_loss_fraction = (float(_t[0]), float(_t[1]))
        except Exception:
            raise SystemExit("--tail_loss_fraction expects 'lo,hi' with 0 <= lo < hi <= 0.95.")
    if getattr(args, "no_gate_loss_continuation", False):
        cfg.model.gate_loss_continuation = False
    if getattr(args, "gate_loss_rate_bound", None):
        cfg.model.gate_loss_rate_bound = str(args.gate_loss_rate_bound)
    if getattr(args, "gate_loss_withheld", None):
        try:
            _w = tuple(sorted({int(v) for v in str(args.gate_loss_withheld).split(",") if str(v).strip()}))
            if not _w or min(_w) < 1 or max(_w) > 28:
                raise ValueError(args.gate_loss_withheld)
            cfg.runtime.gate_loss_withheld = _w
        except Exception:
            raise SystemExit("--gate_loss_withheld expects a comma list of integers in 1..28.")
    if getattr(args, "gate_loss_profiles", None) is not None:
        cfg.runtime.gate_loss_profiles = max(0, int(args.gate_loss_profiles))
    if getattr(args, "informative_continuation", None):
        cfg.model.informative_continuation = str(args.informative_continuation)
    if getattr(args, "no_deployment_gate", False):
        cfg.runtime.deployment_gate = False
    if getattr(args, "no_profile_extras", False):
        cfg.runtime.profile_extras = False
    if getattr(args, "paired_unit", None):
        cfg.data.paired_unit = str(args.paired_unit).strip()
    if getattr(args, "no_cdmr_calibration", False):
        cfg.model.cdmr_calibrated = False
    if getattr(args, "gate_loss_fan_withheld", None) is not None:
        cfg.runtime.gate_loss_fan_withheld = int(np.clip(int(args.gate_loss_fan_withheld), 1, 28))
    if getattr(args, "no_gate_loss_noise_tail", False):
        cfg.runtime.gate_loss_noise_tail_baseline = False
    if getattr(args, "evidence_uninformed_cut", False):
        cfg.runtime.evidence_informed_cut = False
    if getattr(args, "sim_noise_fraction", None) is not None:
        cfg.data.sim_noise_fraction = max(0.0, min(1.0, float(args.sim_noise_fraction)))
    if getattr(args, "sim_noise_snr_db", None):
        try:
            _s = [float(v) for v in str(args.sim_noise_snr_db).replace(";", ",").split(",") if str(v).strip()]
            if len(_s) != 2 or not _s[1] > _s[0]:
                raise ValueError(args.sim_noise_snr_db)
            cfg.data.sim_noise_snr_db = (float(_s[0]), float(_s[1]))
        except Exception:
            raise SystemExit("--sim_noise_snr_db expects 'lo,hi' in dB with hi > lo.")
    if getattr(args, "no_family_validated_fusion", False):
        cfg.model.family_validated_fusion = False
    if getattr(args, "family_common_mode_stations", None) is not None:
        cfg.model.family_common_mode_stations = max(0, int(args.family_common_mode_stations))
    if getattr(args, "survey_val_blocks", None) is not None:
        cfg.data.survey_val_blocks = max(0, int(args.survey_val_blocks))
    if getattr(args, "survey_val_block_len", None) is not None:
        cfg.data.survey_val_block_len = max(1, int(args.survey_val_block_len))
    if getattr(args, "survey_val_fold", None) is not None:
        cfg.data.survey_val_fold = int(args.survey_val_fold)
    if getattr(args, "survey_background_window", None) is not None:
        cfg.data.survey_background_window = max(0, int(args.survey_background_window))
    if getattr(args, "receptive_field_scaling", None) is not None:
        cfg.model.receptive_field_scaling = str(args.receptive_field_scaling)
    if getattr(args, "dropout", None) is not None:
        cfg.model.dropout = float(np.clip(args.dropout, 0.0, 0.9))
    if getattr(args, "no_aux_arms_subbatch", False):
        cfg.model.aux_arms_subbatch = False
    if getattr(args, "no_vram_fill_real_step", False):
        cfg.train.vram_fill_real_step = False
    if getattr(args, "aux_cap_sigma", None) is not None:
        cfg.train.aux_cap_sigma = float(max(0.0, args.aux_cap_sigma))
    if getattr(args, "optimizer_batch_policy", None) is not None:
        cfg.train.optimizer_batch_policy = str(args.optimizer_batch_policy)
    if getattr(args, "min_effective_rows", None) is not None:
        cfg.train.min_effective_rows = int(max(1, args.min_effective_rows))
    if getattr(args, "no_vram_autotune", False):
        cfg.train.vram_autotune = False
    if getattr(args, "vram_step_margin", None) is not None:
        cfg.train.vram_step_margin = float(max(0.0, args.vram_step_margin))
    if getattr(args, "activation_checkpointing", None) is not None:
        cfg.model.activation_checkpointing = str(args.activation_checkpointing)
    if getattr(args, "no_shm_free_transport", False):
        cfg.train.shm_free_transport = False
    if getattr(args, "no_native_presentation", False):
        cfg.runtime.native_presentation = False
    if getattr(args, "decoupled_arms", False):
        cfg.model.decoupled_arms = True
    if getattr(args, "gate_local_norm", False):
        cfg.model.gate_local_norm = True
    if getattr(args, "no_gate_local_norm", False):
        cfg.model.gate_local_norm = False
    if getattr(args, "decoder_gate_norm_channel", False):
        cfg.model.decoder_gate_norm_channel = True
    if getattr(args, "anchor_completion_only", False):
        cfg.model.coef_field_trust_mix = False
    if getattr(args, "no_mmse_blend", False):
        cfg.model.mmse_blend = False
    if getattr(args, "mmse_blend_in_training", False):
        cfg.model.mmse_blend_in_training = True
    if getattr(args, "mmse_eval_every", None) is not None:
        cfg.model.mmse_eval_every = max(1, int(args.mmse_eval_every))
    if getattr(args, "no_bounded_input", False):
        cfg.model.bounded_input = False
    if getattr(args, "completion_gate_limit", None) is not None:
        cfg.model.completion_gate_limit = max(0, int(args.completion_gate_limit))
    if getattr(args, "patch_batch_cap", None) is not None:
        cfg.train.patch_batch_cap = max(64, int(args.patch_batch_cap))
    if getattr(args, "profile_patch_len", None) is not None:
        cfg.data.profile_patch_len = max(-1, int(args.profile_patch_len))
    if getattr(args, "anomaly_suite_per_cell", None) is not None:
        cfg.runtime.anomaly_suite_per_cell = max(1, int(args.anomaly_suite_per_cell))
    if getattr(args, "anomaly_suite_stations", None) is not None:
        cfg.runtime.anomaly_suite_stations = max(5, int(args.anomaly_suite_stations))
    cfg.paths.init_checkpoint = str(getattr(args, "init_from", "") or "")
    if getattr(args, "init_equivalence_max", None) is not None:
        cfg.runtime.init_equivalence_max = float(args.init_equivalence_max)
    cfg.runtime.confirm_full_from_scratch = bool(getattr(args, "confirm_full_from_scratch", False))
    if (getattr(args, "ablation", "full") == "full" and not cfg.paths.init_checkpoint
            and getattr(args, "mode", "all") in {"all", "train"}):
        cfg.runtime.first_run_full_notice = True
    if getattr(args, "late_cvar_threshold", None) is not None:
        cfg.contract.late_cvar_threshold = float(args.late_cvar_threshold)
    if getattr(args, "allow_amplitude_drift", False):
        cfg.runtime.allow_amplitude_drift = True
    if getattr(args, "anomaly_suite_per_family", None) is not None:
        cfg.runtime.anomaly_suite_per_family = max(2, int(args.anomaly_suite_per_family))
    if getattr(args, "anomaly_suite_per_weak", None) is not None:
        cfg.runtime.anomaly_suite_per_weak = max(2, int(args.anomaly_suite_per_weak))
    if getattr(args, "anomaly_detection_contrast", None) is not None:
        cfg.runtime.anomaly_detection_contrast = max(1e-4, float(args.anomaly_detection_contrast))
    if getattr(args, "weak_anomaly_required_contrast", None) is not None:
        cfg.runtime.weak_anomaly_required_contrast = max(0.0, float(args.weak_anomaly_required_contrast))
    if getattr(args, "weak_anomaly_recall_min", None) is not None:
        cfg.runtime.weak_anomaly_recall_min = min(1.0, max(0.0, float(args.weak_anomaly_recall_min)))
    if getattr(args, "anomaly_suite_profiles", None) is not None:
        cfg.runtime.anomaly_suite_profiles = max(4, int(args.anomaly_suite_profiles))
    if getattr(args, "anomaly_suite_quiet", None) is not None:
        cfg.runtime.anomaly_suite_quiet = max(4, int(args.anomaly_suite_quiet))
    if getattr(args, "anomaly_negative_fraction", None) is not None:
        cfg.data.anomaly_negative_fraction = min(max(float(args.anomaly_negative_fraction), 0.0), 1.0)
    if getattr(args, "anomaly_skew_fraction", None) is not None:
        cfg.data.anomaly_skew_fraction = min(max(float(args.anomaly_skew_fraction), 0.0), 1.0)
    if getattr(args, "stage_a_min_epochs", None) is not None:
        cfg.train.stage_a_min_epochs_before_transition = max(0, int(args.stage_a_min_epochs))
    if getattr(args, "no_stage_warm_start", False):
        cfg.train.stage_init_from_previous_best = False
    if getattr(args, "neighbor_distance_prior", None) is not None:
        cfg.model.neighbor_distance_prior_init = max(0.0, float(args.neighbor_distance_prior))
    if args.alpha_manifold_nrmse_late is not None:
        cfg.loss.alpha_manifold_nrmse_late = float(args.alpha_manifold_nrmse_late)
    if bool(args.v1122_exact):
        cfg.model.measurement_conditioned_gates = False
        cfg.model.per_trace_normalization = False
        cfg.model.amplitude_conditioning = "off"
        cfg.data.clean_augment = False
        cfg.loss.manifold_lambda_follow_max = 1.0
        cfg.loss.alpha_manifold_nrmse_late = 0.5
        logging.getLogger(PROGRAM_NAME).info(
            "--v1122_exact: measurement gates OFF, per-trace normalization OFF, clean "
            "augmentation OFF, manifold lambda-following OFF, alpha_manifold_nrmse_late=0.5. This is "
            "the legacy training path; the unclipped per-sample late term (which consumed most of "
            "the physics head's gradient on curves whose relative late error is unfixable) stays clipped."
        )
    if float(args.doi_sigma) <= 0.0:
        raise ValueError("--doi_sigma must be a positive conductivity in S/m.")
    cfg.runtime.doi_sigma_s_per_m = float(args.doi_sigma)
    cfg.runtime.doi_gate_tolerance_percent = float(args.doi_tolerance)
    cfg.runtime.ip_profiles = max(4, int(args.ip_profiles))
    cfg.runtime.ip_stations = max(9, int(args.ip_stations))
    cfg.runtime.ip_score_threshold = float(args.ip_score_threshold)
    cfg.runtime.ip_min_amplitude_sigma = float(args.ip_min_amplitude_sigma)
    cfg.runtime.ip_guard_hard = bool(args.ip_guard_hard)
    cfg.model.use_ip_head = bool(args.enable_ip_head)
    cfg.data.split_min_groups = max(3, int(args.split_min_groups))
    cfg.data.split_min_val_groups = max(1, int(args.split_min_val_groups))
    cfg.data.split_stratify = not bool(args.no_split_stratify)
    if bool(args.enable_clean_augment):
        cfg.data.clean_augment = True
    if bool(args.disable_clean_augment):
        cfg.data.clean_augment = False
    cfg.data.clean_aug_tau_dex = max(0.0, float(args.clean_aug_tau_dex))
    if bool(args.enable_per_trace_norm):
        cfg.model.per_trace_normalization = True
        cfg.model.amplitude_conditioning = "on"
    if bool(args.disable_per_trace_norm):
        cfg.model.per_trace_normalization = False
        cfg.model.amplitude_conditioning = "off"
    if bool(args.enable_measurement_gates):
        cfg.model.measurement_conditioned_gates = True
    if bool(args.disable_measurement_gates):
        cfg.model.measurement_conditioned_gates = False
    cfg.train.batch_size = args.batch_size
    cfg.train.num_workers = args.num_workers
    if bool(getattr(args, "no_remediation", False)):
        cfg.train.remediation_rounds = 0
    elif getattr(args, "remediation_rounds", None) is not None:
        cfg.train.remediation_rounds = max(0, int(args.remediation_rounds))
    if bool(getattr(args, "no_anomaly_probe", False)):
        cfg.train.anomaly_probe = False
    if getattr(args, "anomaly_probe_batches", None) is not None:
        cfg.train.anomaly_probe_batches = max(1, int(args.anomaly_probe_batches))
    if bool(getattr(args, "no_gpu_synthesis", False)):
        cfg.train.gpu_synthesis = False
    if bool(getattr(args, "compile", False)):
        cfg.train.compile_model = True
    cfg.train.seed = args.seed
    cfg.train.base_lr = args.base_lr
    cfg.train.lr_min_ratio = float(args.lr_min_ratio)
    cfg.train.use_ema = not args.disable_ema
    cfg.train.ema_decay = float(args.ema_decay)
    cfg.train.stage_switch_plateau_patience = int(args.stage_switch_plateau_patience)
    cfg.model.use_neighbor_attention = not args.disable_neighbor_attention
    cfg.model.num_neighbors = 0 if args.disable_neighbor_attention else int(args.num_neighbors if args.num_neighbors is not None else ModelConfig.num_neighbors)
    cfg.data.num_neighbors = cfg.model.num_neighbors
    cfg.train._stage_epochs_default = (args.stage_epochs is None)
    cfg.model._neighbors_default = (
        (not args.disable_neighbor_attention) and args.num_neighbors is None
    )
    cfg.model.use_manifold_branch = not args.disable_manifold_branch
    cfg.model.late_start_index = cfg.data.late_start_index
    if args.manifold_blend_floor is not None:
        cfg.model.manifold_blend_floor_late = float(args.manifold_blend_floor)
    cfg.model.use_manifold_innovation = not args.disable_manifold_innovation
    if getattr(args, "lateral_joint_mode", None):
        cfg.model.lateral_joint_mode = str(args.lateral_joint_mode)
    if getattr(args, "no_lateral_joint", False):
        cfg.model.lateral_joint_prior = False
        cfg.model.lateral_joint_mode = "off"
    if getattr(args, "lateral_fingerprint_bank", None) is not None:
        cfg.model.lateral_fingerprint_bank = max(1, int(args.lateral_fingerprint_bank))
    if getattr(args, "no_lateral_fp_prior", False):
        cfg.model.lateral_fingerprint_on_prior = False
    if getattr(args, "no_lateral_fp_output", False):
        cfg.model.lateral_fingerprint_on_output = False
    if getattr(args, "lateral_joint_lambda_late", None) is not None:
        cfg.model.lateral_joint_lambda_late = max(0.0, float(args.lateral_joint_lambda_late))
    if getattr(args, "lateral_joint_lambda_mid", None) is not None:
        cfg.model.lateral_joint_lambda_mid = max(0.0, float(args.lateral_joint_lambda_mid))
    if getattr(args, "lateral_joint_order", None) is not None:
        cfg.model.lateral_joint_order = int(max(1, min(3, int(args.lateral_joint_order))))
    if getattr(args, "lateral_joint_lambda_floor_late", None) is not None:
        cfg.model.lateral_joint_lambda_floor_late = max(0.0, float(args.lateral_joint_lambda_floor_late))
    if getattr(args, "lateral_joint_boundary", None) is not None:
        cfg.model.lateral_joint_boundary_reflect = (str(args.lateral_joint_boundary) == "reflect")
    if getattr(args, "lateral_event_licence", None) is not None:
        cfg.model.lateral_event_licence = max(0.0, min(1.0, float(args.lateral_event_licence)))
    if getattr(args, "stage_c_lr_factor", None) is not None:
        cfg.train.stage_c_lr_factor = max(1e-4, float(args.stage_c_lr_factor))
    if getattr(args, "no_witness_attention", False):
        cfg.model.lateral_witness_gates_attention = False
    if getattr(args, "neighbor_dropout", None) is not None:
        cfg.train.neighbor_dropout = max(0.0, min(1.0, float(args.neighbor_dropout)))
    if getattr(args, "survey_profile_fraction", None) is not None:
        cfg.data.survey_profile_fraction = max(0.0, min(1.0, float(args.survey_profile_fraction)))
    if getattr(args, "extra_clean_alignment", None) is not None:
        cfg.data.extra_clean_alignment = str(args.extra_clean_alignment)
    if getattr(args, "no_survey_profile", False):
        cfg.data.survey_profile_fraction = 0.0
        cfg.data.extra_clean_alignment = "off"
    if getattr(args, "survey_reference_csv", None):
        cfg.paths.survey_reference_csv = str(args.survey_reference_csv)
    if getattr(args, "survey_holdout_m", None):
        try:
            _h = [float(_v) for _v in str(args.survey_holdout_m).replace(";", ",").split(",") if str(_v).strip()]
            if len(_h) == 2 and _h[1] > _h[0]:
                cfg.data.survey_holdout_lo_m, cfg.data.survey_holdout_hi_m = float(_h[0]), float(_h[1])
            else:
                raise ValueError(args.survey_holdout_m)
        except Exception:
            raise SystemExit("--survey_holdout_m expects 'lo,hi' with hi > lo (station units).")
    if getattr(args, "library_row_weight", None) is not None:
        cfg.loss.library_row_weight = max(0.0, min(1.0, float(args.library_row_weight)))
    if getattr(args, "library_row_cap_late", None) is not None:
        cfg.loss.library_row_cap_late = max(0.0, float(args.library_row_cap_late))
    if getattr(args, "library_row_cap_global", None) is not None:
        cfg.loss.library_row_cap_global = max(0.0, float(args.library_row_cap_global))
    if getattr(args, "lateral_joint_weighting", None):
        cfg.model.lateral_joint_weighting = str(args.lateral_joint_weighting)
    if getattr(args, "lateral_joint_noise_kappa", None) is not None:
        cfg.model.lateral_joint_noise_kappa = max(0.0, float(args.lateral_joint_noise_kappa))
    if getattr(args, "lateral_joint_noise_cap", None) is not None:
        cfg.model.lateral_joint_noise_cap = max(1.0, float(args.lateral_joint_noise_cap))
    if getattr(args, "no_late_noise_floor", False):
        cfg.model.late_measured_noise_floor = False
    if getattr(args, "late_noise_floor_gamma", None) is not None:
        cfg.model.late_noise_floor_gamma = float(min(1.0, max(0.0, args.late_noise_floor_gamma)))
    if getattr(args, "no_innovation_evidence_gate", False):
        cfg.model.innovation_evidence_gate = False
    if getattr(args, "innovation_evidence_gate", False):
        cfg.model.innovation_evidence_gate = True
    if getattr(args, "no_innovation_precision", False):
        cfg.model.innovation_precision_weighting = False
    if getattr(args, "lateral_gate_gamma", None) is not None:
        cfg.model.lateral_gate_gamma = max(0.0, float(args.lateral_gate_gamma))
    if getattr(args, "lateral_lambda_witness", None):
        cfg.model.lateral_lambda_witness = str(args.lateral_lambda_witness)
    if getattr(args, "field_alignment_mode", None):
        cfg.runtime.field_alignment_mode = str(args.field_alignment_mode)
    if getattr(args, "no_subzero_training_slice", False):
        cfg.data.measured_noise_subzero_probability = 0.0
    if getattr(args, "subzero_training_probability", None) is not None:
        cfg.data.measured_noise_subzero_probability = float(min(1.0, max(0.0, args.subzero_training_probability)))
    if getattr(args, "no_weak_body_injection", False):
        cfg.data.weak_body_injection_prob = 0.0
    if getattr(args, "weak_body_injection_prob", None) is not None:
        cfg.data.weak_body_injection_prob = float(min(1.0, max(0.0, args.weak_body_injection_prob)))
    if getattr(args, "innovation_robust_nu", None) is not None:
        cfg.model.innovation_robust_nu = max(1.0, float(args.innovation_robust_nu))
    if getattr(args, "no_output_measurement_fusion", False):
        cfg.model.output_measurement_fusion = False
    if getattr(args, "no_lateral_temporal_witness", False):
        cfg.model.lateral_gate_temporal_witness = False
    if getattr(args, "innovation_warmup_epochs", None) is not None:
        cfg.model.innovation_warmup_epochs = max(1, int(args.innovation_warmup_epochs))
    if getattr(args, "innovation_precision_mod_floor", None) is not None:
        cfg.model.innovation_precision_mod_floor = float(min(1.0, max(0.0, args.innovation_precision_mod_floor)))
    if getattr(args, "innovation_evidence_kappa", None) is not None:
        cfg.model.innovation_evidence_kappa = max(0.0, float(args.innovation_evidence_kappa))
    if getattr(args, "no_eval_profile_blocks", False):
        cfg.data.eval_profile_blocks = False
    cfg.model.use_physics_atoms = not args.disable_physics_atoms
    cfg.model.use_position_encoding = not args.disable_position_encoding
    if args.disable_noise_enrichment:
        cfg.data.colored_noise_probability = 0.0
        cfg.data.spike_noise_probability = 0.0
        cfg.data.harmonic_noise_probability = 0.0
        cfg.data.drift_noise_probability = 0.0
        cfg.data.burst_noise_probability = 0.0
    cfg.loss.spectrum_backend = args.spectrum_backend
    cfg.train.amp = not args.no_amp
    cfg.train.use_priority_gradient_projection = not args.no_gradient_projection
    cfg.train.deterministic = bool(args.deterministic)
    cfg.data.train_samples_per_epoch = args.train_samples_per_epoch
    cfg.data.val_samples = args.val_samples
    cfg.data.test_samples = args.test_samples
    _lp = getattr(cfg.train, "large_vram_preset", None)
    if _lp:
        _explicit: List[str] = []
        if int(args.batch_size) == int(TrainConfig.batch_size):
            cfg.train.batch_size = int(_lp["batch_size"])
        else:
            _explicit.append("batch_size")
        if int(args.num_workers) == int(TrainConfig.num_workers):
            cfg.train.num_workers = int(_lp["num_workers"])
        else:
            _explicit.append("num_workers")
        if int(args.train_samples_per_epoch) == int(DataConfig.train_samples_per_epoch):
            cfg.data.train_samples_per_epoch = int(_lp["train_samples_per_epoch"])
        else:
            _explicit.append("train_samples_per_epoch")
        if int(args.val_samples) == int(DataConfig.val_samples):
            cfg.data.val_samples = int(_lp["val_samples"])
        else:
            _explicit.append("val_samples")
        if int(args.test_samples) == int(DataConfig.test_samples):
            cfg.data.test_samples = int(_lp["val_samples"])
        else:
            _explicit.append("test_samples")
        if float(args.base_lr) == float(TrainConfig.base_lr):
            cfg.train.base_lr = float(_lp["base_lr"])
        else:
            _explicit.append("base_lr")
        cfg.train.persistent_workers = True
        cfg.train.pin_memory = True
        _lp.update({
            "batch_size": int(cfg.train.batch_size),
            "num_workers": int(cfg.train.num_workers),
            "train_samples_per_epoch": int(cfg.data.train_samples_per_epoch),
            "val_samples": int(cfg.data.val_samples),
            "base_lr": float(cfg.train.base_lr),
            "explicit_overrides": _explicit,
        })
    cfg.data.late_start_index = args.late_start - 1
    if not (0 <= cfg.data.late_start_index < cfg.data.target_gates):
        raise ValueError("late_start must be between 1 and target_gates.")
    cfg.data.snr_min_db = float(args.snr_min_db)
    cfg.model.prior_width_mult = float(args.prior_width_mult)
    cfg.model.signal_fingerprint_bank = int(args.signal_fingerprint_bank)
    cfg.data.snr_max_db = float(args.snr_max_db)
    cfg.data.snr_below_zero_probability = float(args.snr_below_zero_probability)
    if not (cfg.data.snr_min_db < cfg.data.snr_max_db):
        raise ValueError("SNR range must be non-empty: snr_min_db < snr_max_db.")
    if not (0.0 <= cfg.data.snr_below_zero_probability < 1.0):
        raise ValueError("snr_below_zero_probability must lie in [0, 1).")
    cfg.model.gates = cfg.data.target_gates
    parts = [int(x.strip()) for x in (
        args.stage_epochs
        or "%d,%d,%d" % (
            TrainConfig.stage_a_max_epochs, TrainConfig.stage_b_max_epochs, TrainConfig.stage_c_epochs
        )
    ).split(",")]
    if len(parts) != 3 or any(x < 0 for x in parts):
        raise ValueError("--stage_epochs must be three non-negative integers, e.g. 60,80,40")
    cfg.train.stage_a_max_epochs, cfg.train.stage_b_max_epochs, cfg.train.stage_c_epochs = parts
    apply_ablation_preset(cfg, args.ablation)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the program with the given command-line arguments and return the process exit code."""
    args = build_parser().parse_args(argv)
    args = maybe_run_interactive_menu(args)
    cfg = config_from_args(args)
    if cfg.runtime.mode == "ablation_report":
        root = Path(cfg.paths.output_root)
        rows = []
        for summary in sorted(root.glob("*/reports/ablation_summary.json")):
            try:
                with summary.open("r", encoding="utf-8") as f:
                    rec = json.load(f)
                tm = rec.get("test_metrics", {})
                rows.append(
                    {
                        "run": summary.parent.parent.name,
                        "ablation": rec.get("ablation", "?"),
                        "G_percent": tm.get("global_error_percent"),
                        "EM_percent": tm.get("early_mid_error_percent"),
                        "L_percent": tm.get("late_error_percent"),
                        "L_CVaR_percent": tm.get("late_cvar_percent"),
                        "feasible": tm.get("project_feasible"),
                        "version": rec.get("version"),
                    }
                )
            except Exception:
                continue
        if not rows:
            print(f"No ablation_summary.json found under {root}")
            return 1
        df = pd.DataFrame(rows).sort_values(["ablation", "run"])
        out = root / "ablation_report.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(df.to_string(index=False))
        print(f"\nAblation report written: {out}")
        return 0
    set_global_seed(cfg.train.seed, cfg.train.deterministic)
    run_dir = make_run_dir(cfg)
    logger = setup_logging(run_dir, getattr(cfg.paths, "log_root", ""), run_dir.name,
                           int(getattr(args, "log_retain", 60)))
    try:
        _self = Path(__file__).resolve()
        _sha = sha256_file(_self)[:12]
        _mt = _dt.datetime.fromtimestamp(
            _self.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        _self, _sha, _mt = Path(sys.argv[0]), "unknown", "unknown"
    logger.info("%s v%s | file %s | sha256 %s | mtime %s.",
                PROGRAM_NAME, PROGRAM_VERSION, _self, _sha, _mt)
    for _pn in globals().get("_PIN_NOTES", []) or []:
        logger.info("%s", _pn)
    _lc = cfg.loss
    logger.info(
        "SUPERVISION DOCTRINE | P1 signal-is-learned: manifold(bg) "
        "early/late %.2f/%.2f, lateral d1/d2/newext/mgate %s, anomaly "
        "diff/amp/sign %.2f/%.2f/%.2f, monotone %.2f | P2 noise-is-told-apart: "
        "realization-consistency %.2f, carryover centre/neighbour %.2f/%.2f "
        "(measured per-gate weights, Wiener-direction allowance) | P3 "
        "noise-tests-stability: per-SNR bins + [-5,10]dB probe (accuracy + "
        "realization-stability) + injected triptych.",
        float(_lc.alpha_manifold_early), float(_lc.alpha_manifold_late),
        ("ON(P=%d)" % int(cfg.data.profile_patch_len)
         if int(cfg.data.profile_patch_len) > 2 else "OFF"),
        float(_lc.alpha_anomaly_diff), float(_lc.alpha_anomaly_amp),
        float(_lc.alpha_anomaly_sign), float(_lc.alpha_gate_monotonic),
        float(_lc.alpha_real_noise_invariance),
        float(getattr(_lc, "alpha_noise_carryover", 0.10)),
        float(getattr(_lc, "alpha_neighbor_carryover", 0.30)))
    logger.info(
        "DOCTRINE ADDENDUM | P1b trunk-is-supervised: "
        "denoise contract %.2f (x den/prior leash boost) + per-curve %.2f; continuation "
        "arm frac %.2f weight %.2f | fusion: %s.",
        float(getattr(_lc, "alpha_denoise_contract", 1.0)),
        float(getattr(_lc, "alpha_denoise_ps_late", 0.35)),
        float(getattr(cfg.model, "continuation_arm_frac", 0.25)),
        float(getattr(_lc, "alpha_continuation", 0.30)),
        ("SUPERVISED-PRIMARY"
         if bool(getattr(cfg.model, "supervised_primary_fusion", True))
         else "LEGACY PRIOR-PRIMARY"))
    logger.info("Run directory: %s", run_dir)
    _dev0 = resolve_device(cfg.runtime.device)
    if _dev0.type == "cuda":
        try:
            _probe = torch.empty(1, device=_dev0)
            _cur = int(torch.cuda.current_device())
            _got = int(_probe.device.index)
            _want = int(_dev0.index if _dev0.index is not None else _cur)
            _fr, _tt = torch.cuda.mem_get_info(_got)
            logger.info(
                "DEVICE IN USE | requested %s -> current cuda:%d, probe tensor on cuda:%d | %.1f/%.1f "
                "GB free on that card.",
                cfg.runtime.device, _cur, _got, _fr / 2**30, _tt / 2**30)
            if _got != _want or _cur != _want:
                logger.error(
                    "DEVICE MISMATCH -- REFUSING TO TRAIN: asked for index %d but the current device "
                    "is %d and a new tensor landed on %d. Allocations will not go where you asked. Remedy: export "
                    "CUDA_VISIBLE_DEVICES=%d and run with --device cuda:0; or try the other physical index.", _want, _cur, _got, _want)
                raise SystemExit(34)
            del _probe
        except Exception as _pexc0:
            logger.warning("Device probe failed: %s", _pexc0)
    import atexit as _atexit
    _atexit.register(mirror_run_outputs, cfg, run_dir, logger)
    _run_started = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_run_index() -> None:
        try:
            s = dict(_LAST_MONITOR_SUMMARY or {})
            _dep = dict(_LAST_DEPLOYMENT or {})
            append_run_index(cfg, run_dir, {
                "run": run_dir.name,
                "started": _run_started,
                "finished": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": cfg.runtime.mode,
                "supervision": ("paired" if (
                    bool(getattr(cfg.data, "paired_supervision", True))
                    and str(getattr(cfg.paths, "paired_dir", "") or "").strip()
                    and Path(str(cfg.paths.paired_dir)).is_dir()
                    and (Path(str(cfg.paths.paired_dir)) / "train_clean.csv").is_file()
                ) else "synthetic"),
                "ablation": cfg.runtime.ablation,
                "device": cfg.runtime.device,
                "epochs": s.get("epochs", ""),
                "global_percent": s.get("at_best_global_percent", ""),
                "early_mid_percent": s.get("at_best_early_mid_percent", ""),
                "late_percent": s.get("best_late_percent", ""),
                "late_cvar_percent": s.get("at_best_late_cvar_percent", ""),
                "anomaly_recovery": s.get("final_anomaly_body_recovery", ""),
                "anomaly_auroc": s.get("at_best_anomaly_body_auroc", ""),
                "false_anomaly_upper95": _dep.get("quiet_false_rate_wilson_upper95", ""),
                "feasible": _dep.get("checkpoint_feasible", ""),
                "exit_code": _dep.get("exit_code", ""),
            }, logger)
        except Exception as _iexc:
            logger.warning("Run index skipped: %r", _iexc)

    _atexit.register(_write_run_index)
    logger.info(
        "OUTPUT ROUTING | data & reports -> %s | checkpoints -> %s (%s) | figures -> %s (%s) | "
        "logs & monitor -> %s.",
        run_dir, cfg.runtime.resolved_checkpoint_dir, cfg.runtime.checkpoint_dir_note,
        cfg.runtime.resolved_figure_dir, cfg.runtime.figure_dir_note,
        getattr(cfg.paths, "log_root", "") or "(run directory only)")
    try:
        _dev = resolve_device(cfg.runtime.device)
        if _dev.type == "cuda":
            _p = torch.cuda.get_device_properties(_dev)
            _free = float("nan")
            try:
                _free = float(torch.cuda.mem_get_info(_dev)[0]) / 2 ** 30
            except Exception:
                pass
            logger.info(
                "DEVICE | %s -> index %d of %d visible | %s | %.1f GB total%s.",
                cfg.runtime.device, 0 if _dev.index is None else _dev.index,
                torch.cuda.device_count(), _p.name, _p.total_memory / 2 ** 30,
                (", %.1f GB free" % _free) if math.isfinite(_free) else "")
        else:
            logger.info("DEVICE | %s.", _dev)
    except Exception as _dexc:
        logger.warning("Device probe: %s", _dexc)
    _lp = getattr(cfg.train, "large_vram_preset", None)
    if _lp:
        logger.info(
            "LARGE-VRAM PRESET | card %.1f GB -> batch %d, %d train samples/epoch, %d val samples, "
            "%d workers, base LR %.3g (x%.2f = sqrt of the batch ratio). The contract, the model and every "
            "gate are UNCHANGED, so this run stays comparable with every earlier one.",
            _lp["card_gb"], _lp["batch_size"], _lp["train_samples_per_epoch"],
            _lp["val_samples"], _lp["num_workers"], _lp["base_lr"], _lp["lr_multiplier"])
    try:
        _paired_startup = paired_split_files(
            Path(str(getattr(cfg.paths, "paired_dir", "") or ".")))
    except Exception as _pexc:
        logger.error("%s", _pexc)
        return 31
    if _paired_startup and bool(getattr(cfg.data, "paired_supervision", True)):
        _f_rep = float(getattr(cfg.data, "paired_residual_replay", 0.0))
        _f_inj = float(getattr(cfg.data, "measured_noise_inject", 0.0))
        _f_alt = float(getattr(cfg.data, "paired_alt_corruption", 0.0))
        _f_sim = float(getattr(cfg.data, "sim_noise_fraction", 0.0) or 0.0)
        _f_tail = float(getattr(cfg.data, "tail_loss_prob", 0.0) or 0.0)
        logger.info(
            "SUPERVISION MODE: PAIRED. Target = the clean member of each pair; observation "
            "= its noisy partner; noise target = their difference, i.e. the noise the dataset carries "
            "(measured or simulated, as the dataset was made). TRAINING ARMS (every one satisfies x = y + "
            "n), drawn per patch in this order: online simulated noise %.2f | residual replay %.2f | "
            "noise-library injection %.2f | otherwise the dataset's own pair. An alternate corruption (%.2f) "
            "feeds the invariance term; %.2f of the training patches withhold their last gates (gate-loss "
            "augmentation). Donors never cross a work area and neighbours share the centre's "
            "route by construction.",
            _f_sim, _f_rep, _f_inj, _f_alt, _f_tail)
    elif _paired_startup:
        logger.warning(
            "Paired files exist at %s but --no_paired was given: this run uses "
            "SYNTHETIC noise injection.", cfg.paths.paired_dir,
        )
    else:
        logger.info(
            "SUPERVISION MODE: SYNTHETIC (no paired files at %s).",
            cfg.paths.paired_dir)
    if cfg.runtime.ablation != "full":
        desc, overrides = ABLATION_PRESETS[cfg.runtime.ablation]
        logger.info("Ablation preset: %s | %s | overrides=%s", cfg.runtime.ablation, desc, overrides)
    elif getattr(cfg.runtime, "first_run_full_notice", False) and not bool(
            cfg.runtime.confirm_full_from_scratch):
        logger.error(
            "REFUSING TO START --ablation full FROM RANDOM INITIALIZATION . full "
            "enables neighbour geometry, the stretched relaxation dictionary, RSSU, the event "
            "pathway and physics_residual_scale=%.2f at once: whatever such a run measures "
            "cannot be attributed to any single mechanism, and a failure cannot be localized. "
            "The qualification route is q0_minimal -> q1_geometry -> q2_stretch -> q3_rssu -> "
            "q4_event, each promoted with --init_from <parent.pt> (the promotion is checked "
            "for numerical inertness before training starts). Re-run without --no_menu to pick "
            "that route interactively, or pass --confirm_full_from_scratch to run the final "
            "configuration anyway. Exit code 27.",
            float(cfg.model.physics_residual_scale),
        )
        return 27
    elif getattr(cfg.runtime, "first_run_full_notice", False):
        logger.info(
            "NOTE : --ablation full enables neighbour geometry, the stretched "
            "relaxation dictionary, RSSU, the event pathway and physics_residual_scale=%.2f "
            "simultaneously, and is being started from random initialization. That is a FINAL "
            "configuration, not a first qualification run: nothing measured here can be "
            "attributed to any single mechanism. The reviewed ladder is "
            "q0_minimal -> q1_geometry -> q2_stretch -> q3_rssu -> q4_event, each stage "
            "warm-started from the previous stage's qualified weights via --init_from.",
            float(cfg.model.physics_residual_scale),
        )
    atomic_json_dump(dataclass_to_dict(cfg), run_dir / "config.json")
    logger.info("PyTorch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(resolve_device(cfg.runtime.device)))
        logger.info("PyTorch compiled CUDA: %s", getattr(torch.version, "cuda", None))

    if cfg.runtime.mode in {"all", "infer", "figures"} and field_profile_present(cfg):
        _d0 = _unit_declaration(cfg, logger)
        if not _d0["pass"]:
            logger.warning(
                "FIELD OUTPUT WILL BE BLOCKED : the survey units are not declared "
                "(%s missing). Training, validation and every simulated suite still run and are "
                "still meaningful -- but the amplitude gate cannot pass, so no field CSV or "
                "field figure will be written at the end of this run. This is said NOW rather "
                "than after the training budget has been spent. Declare with "
                "--field_signal_quantity / --field_amplitude_unit / --field_time_unit / "
                "--field_normalization_history, or answer the menu prompt.",
                ", ".join(_d0["missing_fields"]),
            )
    _P = int(getattr(cfg.data, "profile_patch_len", 0) or 0)
    if _P > 1:
        _want = min(int(cfg.train.batch_size) * _P, int(cfg.train.patch_batch_cap))
        _new_bs = max(_P, (_want // _P) * _P)
        if _new_bs != cfg.train.batch_size:
            logger.info(
                "Patch training keeps the INDEPENDENT-anchor count: effective batch %d -> %d (%d "
                "patches x %d stations). Nine traces of a patch share one background, noise draw and anomaly, so "
                "they are one draw seen nine ways. The memory cost is handled by gradient accumulation, not by a "
                "bigger forward.",
                cfg.train.batch_size, _new_bs, _new_bs // _P, _P)
            cfg.train.batch_size = _new_bs
        if cfg.train.batch_size % _P != 0:
            _bs = max(_P, (cfg.train.batch_size // _P) * _P)
            logger.warning(
                "Profile-patch training needs a batch size that is a multiple of the "
                "patch length (%d); %d -> %d.", _P, cfg.train.batch_size, _bs,
            )
            cfg.train.batch_size = _bs
        logger.info(
            "STATION-DIRECTION TRAINING ON | patch length %d: each batch is %d "
            "contiguous-station patches, and the objective carries lateral d1/d2, an excess "
            "second-difference (new extremum) penalty and multi-gate consistency. This relies "
            "on the contiguous-block ordering assumption -- the same one the real-neighbour "
            "channel relies on -- which is audited by the duplicate probe and reported in the "
            "qualification.", _P, cfg.train.batch_size // _P,
        )
    if torch.cuda.is_available() and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    configure_acceleration(cfg, logger)
    run_numeric_preflight(cfg, logger)
    clean_cache, noise_cache, audit = audit_and_prepare_data(cfg, run_dir, logger)
    if int(getattr(cfg.data, "profile_patch_len", 0) or 0) == -1:
        _pid = getattr(clean_cache, "province_id", None)
        _Pres = 0
        if _pid is not None:
            _sizes = np.bincount(np.asarray(_pid, dtype=np.int64))
            _sizes = _sizes[_sizes > 0]
            if _sizes.size:
                _vals, _cnts = np.unique(_sizes, return_counts=True)
                _Pres = int(_vals[np.argmax(_cnts)])
        if _Pres <= 2:
            logger.warning("Work-area size could not be resolved (no province "
                           "structure); falling back to 9-station patches.")
            _Pres = 9
        cfg.data.profile_patch_len = int(_Pres)
        _P = int(_Pres)
        _want = min(int(cfg.train.batch_size) * _P, int(cfg.train.patch_batch_cap))
        _new_bs = max(_P, (_want // _P) * _P)
        logger.info(
            "WHOLE-AREA PATCHES | work-area size %d stations (modal province length) -> "
            "profile_patch_len %d | effective batch %d -> %d (%d areas x %d stations; the autotune "
            "keeps the micro-batch a multiple). Every lateral mechanism now trains on the SECTION it will "
            "infer on.",
            _P, _P, cfg.train.batch_size, _new_bs, _new_bs // _P, _P)
        cfg.train.batch_size = _new_bs
    _ceiling = log_identification_ceiling(clean_cache, cfg, logger, noise_cache=noise_cache)
    if _ceiling:
        try:
            cfg.contract.ceiling_late_pooled_percent = float(_ceiling.get("pooled", float("nan")))
            cfg.contract.ceiling_late_median_percent = float(_ceiling.get("median", float("nan")))
            cfg.contract.ceiling_late_cvar_percent = float(_ceiling.get("cvar", float("nan")))
            _cv = float(_ceiling.get("cvar", float("nan")))
            _gate = 100.0 * float(cfg.contract.late_cvar_threshold)
            if (bool(getattr(cfg.contract, "late_cvar_rebase", True))
                    and math.isfinite(_cv) and _cv > _gate):
                _m = float(getattr(cfg.contract, "late_cvar_rebase_margin", 0.10))
                cfg.contract.late_cvar_threshold_effective = (_cv / 100.0) * (1.0 + _m)
                logger.warning(
                    "TAIL GATE REBASED FOR CONTROL/SELECTION | the identification ceiling's own "
                    "worst-decile mean on this split is %.2f%% > the %.1f%% contract gate, so late-CVaR pressure, "
                    "the checkpoint score and the feasibility flag now use %.2f%% (= ceiling x %.2f). The ORIGINAL "
                    "%.1f%% is still reported in every metric line; it remains unreachable on this split by "
                    "construction, not by any property of the model. Disable with --no_late_cvar_rebase.",
                    _cv, _gate, _cv * (1.0 + _m), 1.0 + _m, _gate)
        except Exception as _cexc:
            logger.warning("Ceiling stash skipped: %r", _cexc)
    log_physics_basis_fit(clean_cache, noise_cache, cfg, logger)
    if cfg.runtime.mode == "audit":
        logger.info("Audit-only mode complete.")
        write_final_report(cfg, run_dir, None, logger)
        return 0

    if cfg.runtime.mode == "diagnose":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in diagnose mode.")
        device = resolve_device(cfg.runtime.device)
        reconcile_model_config_with_checkpoint(Path(args.checkpoint), cfg, logger)
        sync_model_gates(cfg, logger)
        model = PEBRNet(cfg.model).to(device)
        controller = ConstraintController(cfg.contract)
        state = load_checkpoint(Path(args.checkpoint), model, controller, device, logger)
        if "global_scale" in state:
            clean_cache.global_scale = float(state["global_scale"])
        if "gate_scale" in state:
            clean_cache.gate_scale = np.asarray(state["gate_scale"], dtype=np.float64)
        if "target_time" in state:
            require_same_axis(clean_cache.target_time, state["target_time"], what="checkpoint")
        run_generalization_diagnosis(model, cfg, run_dir, clean_cache, noise_cache, logger)
        write_final_report(cfg, run_dir, Path(args.checkpoint), logger)
        return 0

    if cfg.runtime.mode == "gateloss":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in gateloss mode.")
        device = resolve_device(cfg.runtime.device)
        reconcile_model_config_with_checkpoint(Path(args.checkpoint), cfg, logger)
        sync_model_gates(cfg, logger)
        model = PEBRNet(cfg.model).to(device)
        controller = ConstraintController(cfg.contract)
        state = load_checkpoint(Path(args.checkpoint), model, controller, device, logger)
        if "global_scale" in state:
            clean_cache.global_scale = float(state["global_scale"])
        if "gate_scale" in state:
            clean_cache.gate_scale = np.asarray(state["gate_scale"], dtype=np.float64)
        if "target_time" in state:
            require_same_axis(clean_cache.target_time, state["target_time"], what="checkpoint")
        cfg.runtime.plot = not bool(getattr(args, "no_plot", False))
        run_gate_loss_suite(model, cfg, run_dir, clean_cache, noise_cache, logger)
        write_final_report(cfg, run_dir, Path(args.checkpoint), logger)
        logger.info("GATELOSS suite complete. Run directory: %s", run_dir)
        return 0

    if cfg.runtime.mode == "figures":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in figures mode.")
        device = resolve_device(cfg.runtime.device)
        reconcile_model_config_with_checkpoint(Path(args.checkpoint), cfg, logger)
        sync_model_gates(cfg, logger)
        model = PEBRNet(cfg.model).to(device)
        controller = ConstraintController(cfg.contract)
        state = load_checkpoint(Path(args.checkpoint), model, controller, device, logger)
        if "global_scale" in state:
            clean_cache.global_scale = float(state["global_scale"])
        if "gate_scale" in state:
            clean_cache.gate_scale = np.asarray(state["gate_scale"], dtype=np.float64)
        if "target_time" in state:
            require_same_axis(clean_cache.target_time, state["target_time"], what="checkpoint")
        _auto = bool(getattr(args, "section_autopsy_only", False))
        if not _auto:
            information_bound_stage(cfg, clean_cache, logger, run_dir)
        fit_row_completion(model, cfg, clean_cache, logger)
        if bool(getattr(args, "information_bound_only", False)):
            logger.info("--information_bound_only: done.")
            return 0
        if _auto:
            return section_autopsy_pass(model, cfg, run_dir, clean_cache, noise_cache, Path(args.checkpoint), logger)
        _contract = verify_frozen_contract(model, cfg, clean_cache, noise_cache, state, logger)
        _contract_pass = bool(_contract.get("project_feasible", False))
        _pf = field_preflight(cfg, run_dir, clean_cache, logger)
        _anom = certify_anomaly_preservation(model, cfg, run_dir, clean_cache, noise_cache, logger)
        deployment = finalize_deployment(
            {"checkpoint_feasible": _contract_pass,
             "test_global_nrmse": float(_contract.get("global_error_percent", float("nan"))),
             "test_late_nrmse": float(_contract.get("late_error_percent", float("nan"))),
             "test_late_cvar": float(_contract.get("late_cvar_percent", float("nan"))),
             "qualification_source": "figures mode: re-run held-out contract test",
             "exit_code": 0 if (_contract_pass or not cfg.runtime.deployment_gate) else 20},
            _anom, cfg, run_dir, logger,
        )
        _run_field = (deployment.get("deployment_allowed") is True) or bool(
            cfg.runtime.force_infer_unqualified)
        if not _run_field and field_profile_present(cfg):
            logger.error(
                "FIGURE PACKAGE NOT DEPLOYMENT-QUALIFIED : contract=%s anomaly=%s "
                "amplitude=%s. The field section and every figure derived from it are NOT "
                "produced; the simulation-side figures still are. Exit code %d.",
                deployment.get("contract_pass"), deployment.get("anomaly_suite_pass"),
                deployment.get("amplitude_alignment_pass"), int(deployment.get("exit_code", 20)),
            )
        elif not _run_field and not cfg.runtime.deployment_gate:
            logger.info("Held-out test: global %.2f%% | late %.2f%% | late CVaR %.2f%%; the figure package "
                        "follows.", float(deployment.get("test_global_nrmse", float("nan"))),
                        float(deployment.get("test_late_nrmse", float("nan"))),
                        float(deployment.get("test_late_cvar", float("nan"))))
        elif not _run_field:
            logger.info("Held-out test: accuracy contract met %s, anomaly preservation %s; the figure package "
                        "follows (exit code %d).", deployment.get("contract_pass"),
                        deployment.get("anomaly_suite_pass"), int(deployment.get("exit_code", 20)))
        produce_figure_package(
            model, cfg, run_dir, clean_cache, noise_cache, Path(args.checkpoint), logger,
            run_field=_run_field, deployment=deployment,
        )
        stamp_checkpoint_qualification(
            Path(args.checkpoint),
            {"contract_test_pass": _contract_pass,
             "global_error_percent": float(_contract.get("global_error_percent", float("nan"))),
             "late_error_percent": float(_contract.get("late_error_percent", float("nan"))),
             "late_cvar_percent": float(_contract.get("late_cvar_percent", float("nan"))),
             "anomaly_suite_pass": _anom.get("pass"),
             "deployment_allowed": deployment.get("deployment_allowed")},
            logger,
        )
        write_final_report(cfg, run_dir, Path(args.checkpoint), logger)
        logger.info("Figure package complete. Run directory: %s", run_dir)
        return int(deployment.get("exit_code", 0))

    selected_checkpoint: Optional[Path] = None
    model: Optional[PEBRNet] = None
    deployment: Dict[str, Any] = {"deployment_allowed": True, "checkpoint_feasible": True, "exit_code": 0}
    if cfg.runtime.mode in {"all", "train"}:
        model, selected_checkpoint, run_test_metrics = train_model(cfg, run_dir, clean_cache, noise_cache, logger)
        export_torchscript(model, run_dir, cfg, logger)
        _feasible = bool(run_test_metrics.get("project_feasible", False))
        _had_feasible_ckpt = bool(run_test_metrics.get("selected_checkpoint_is_feasible_file", False))
        deployment = {
            "checkpoint_feasible": _feasible,
            "had_feasible_checkpoint": _had_feasible_ckpt,
            "test_global_nrmse": float(run_test_metrics.get("global_error_percent", float("nan"))),
            "test_late_nrmse": float(run_test_metrics.get("late_error_percent", float("nan"))),
            "test_late_cvar": float(run_test_metrics.get("late_cvar_percent", float("nan"))),
            "deployment_allowed": _feasible,
            "exit_code": 0 if (_feasible or not cfg.runtime.deployment_gate) else (21 if _had_feasible_ckpt else 20),
            "verdict_enforced": bool(cfg.runtime.deployment_gate),
        }
        atomic_json_dump(deployment, run_dir / "reports" / "deployment_qualification.json")
        if not _feasible:
            if cfg.runtime.deployment_gate and cfg.runtime.require_feasible_checkpoint \
                    and not cfg.runtime.force_infer_unqualified:
                logger.error(
                    "NO FIELD-QUALIFIED CHECKPOINT : G=%.2f%% (gate %.0f%%), "
                    "L=%.2f%% (gate %.0f%%), CVaR=%.2f%% (gate %.0f%%) -- FAILING: %s. FIELD "
                    "output is blocked: no field CSV, no field figures, no master figure, "
                    "nothing from the measured profile may be interpreted, published or "
                    "inverted. The simulated evidence suite IS still rendered and is "
                    "tagged as unqualified. (Debug-only field override: "
                    "--force_infer_unqualified.) Exit code %d.",
                    deployment["test_global_nrmse"], 100.0 * cfg.contract.global_threshold,
                    deployment["test_late_nrmse"], 100.0 * cfg.contract.late_threshold,
                    deployment["test_late_cvar"], 100.0 * cfg.contract.late_cvar_threshold,
                    ", ".join(_failed_contract_dims(deployment, cfg)) or "the held-out test",
                    deployment["exit_code"],
                )
                if cfg.runtime.plot and cfg.runtime.mode in {"all", "figures"}:
                    _diag_field = bool(cfg.runtime.force_infer_unqualified)
                    _fails = ", ".join(_failed_contract_dims(deployment, cfg)) or "the held-out test"
                    set_unqualified_banner(
                        "NOT QUALIFIED FOR INTERPRETATION -- diagnostic output of an "
                        "unqualified checkpoint (%s; run %s, exit %d). Not evidence; not for "
                        "inversion or publication."
                        % (_fails, run_dir.name, int(deployment["exit_code"]))
                    )
                    append_output_tag("MODEL_UNQUALIFIED_SIMULATION_ONLY"
                                      if not _diag_field else "UNQUALIFIED_DEBUG_ONLY")
                    logger.warning(
                        ("DIAGNOSTIC PACKAGE RENDERED | the contract failed on "
                         "%s. The field figures ARE produced -- watermarked, footnoted with the "
                         "failing gate and tagged %s -- because you cannot fix what you cannot "
                         "see; they are diagnostics and nothing in them may be interpreted, "
                         "inverted or published. The run still exits %d."
                         if _diag_field else
                         "SIMULATED EVIDENCE STILL RENDERED | the contract failed on "
                        "%s, so no field artefact is produced -- but the evidence "
                        "suite (E1-E7), the polarization suite (E8-E10), the diagnosis figures "
                        "and the simulated denoising figure are computed on HELD-OUT SIMULATED "
                        "curves and are exactly what is needed to diagnose the failure. Every "
                        "file carries the %s tag and the run still exits %d: these are "
                        "diagnostics, NOT publication evidence of a qualified model."),
                        _fails, output_tag(), int(deployment["exit_code"]),
                    )
                    try:
                        produce_figure_package(
                            model, cfg, run_dir, clean_cache, noise_cache, selected_checkpoint,
                            logger, run_field=_diag_field, deployment=deployment,
                            simulation_only=not _diag_field,
                        )
                    except Exception as _exc:
                        logger.exception(
                            "SIMULATION-ONLY FIGURE PACKAGE FAILED: %s. The contract verdict "
                            "above is unchanged.", _exc,
                        )
                elif cfg.runtime.plot:
                    render_simulation_denoising_figure(model, cfg, run_dir, clean_cache, noise_cache, logger)
                _lk = (_LAST_SPLIT_AUDIT or {}).get("cross_split_duplicate_fraction")
                _near = (_LAST_SPLIT_AUDIT or {}).get("cross_split_near_duplicate_fraction")
                _dims = _failed_contract_dims(deployment, cfg)
                _advice: List[str] = []
                if len(_dims) == 1 and _dims[0].startswith("late CVaR"):
                    _advice.append(
                        "ONLY the late CVaR failed: the median late error is inside the gate and "
                        "the TAIL is not. CVaR is the mean of the worst 20% of held-out curves, "
                        "so this is a hard-subset problem, not a general accuracy problem -- look "
                        "at reports/late_tail_composition.json, which names the worst-20%% membership, "
                        "before changing anything global")
                if _lk is not None and float(_lk) > 0.0:
                    _advice.append(
                        "the split audit found %.2f%% exactly-duplicated and %.2f%% "
                        "near-duplicated shapes across train/test, so part of the held-out score "
                        "is memorization AND deployment is blocked by split_leakage_pass "
                        "regardless of the contract: raise --split_group_size (or supply profile "
                        "metadata) and re-split before reading any of these numbers"
                        % (100.0 * float(_lk), 100.0 * float(_near or 0.0)))
                if _advice:
                    logger.error("WHAT TO CHANGE NEXT : %s.", "; ".join(_advice))
                write_final_report(cfg, run_dir, selected_checkpoint, logger)
                return int(deployment["exit_code"])
            if cfg.runtime.force_infer_unqualified:
                append_output_tag("UNQUALIFIED_DEBUG_ONLY")
                logger.warning(
                    "FORCED UNQUALIFIED RUN : --force_infer_unqualified is set. Every "
                    "output file from here on carries the __UNQUALIFIED_DEBUG_ONLY tag, every "
                    "figure row is stamped RENDERED_BUT_MODEL_UNQUALIFIED, and the exit code "
                    "stays %d. These files are debugging aids, not results.",
                    deployment["exit_code"],
                )
        if cfg.runtime.mode == "train":
            if cfg.runtime.plot:
                render_simulation_denoising_figure(model, cfg, run_dir, clean_cache, noise_cache, logger)
            write_final_report(cfg, run_dir, selected_checkpoint, logger)
            return int(deployment["exit_code"])

    assert model is not None and selected_checkpoint is not None
    if cfg.runtime.mode == "all" and not cfg.runtime.skip_figure_package and cfg.runtime.plot:
        field_preflight(cfg, run_dir, clean_cache, logger)
        _anom = certify_anomaly_preservation(model, cfg, run_dir, clean_cache, noise_cache, logger)
        deployment = finalize_deployment(deployment, _anom, cfg, run_dir, logger)
        stamp_checkpoint_qualification(
            selected_checkpoint,
            {"anomaly_suite_pass": _anom.get("pass"),
             "anomaly_ood_pass": _anom.get("anomaly_ood_pass"),
             "quiet_false_rate_pass": _anom.get("quiet_false_rate_pass"),
             "deployment_allowed": deployment.get("deployment_allowed")},
            logger,
        )
        produce_figure_package(
            model, cfg, run_dir, clean_cache, noise_cache, selected_checkpoint, logger,
            run_field=(deployment.get("deployment_allowed") is True)
            or bool(cfg.runtime.force_infer_unqualified),
            deployment=deployment,
        )
    else:
        if cfg.runtime.plot:
            render_simulation_denoising_figure(model, cfg, run_dir, clean_cache, noise_cache, logger)
        field_preflight(cfg, run_dir, clean_cache, logger)
        _anom = certify_anomaly_preservation(model, cfg, run_dir, clean_cache, noise_cache, logger)
        deployment = finalize_deployment(deployment, _anom, cfg, run_dir, logger)
        run_field_if_qualified(model, selected_checkpoint, cfg, run_dir, clean_cache,
                               logger, deployment)
    write_final_report(cfg, run_dir, selected_checkpoint, logger)
    return int(deployment.get("exit_code", 0))


def _query_free_memory_mib() -> List[int]:
    import subprocess
    try:
        _out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
            env=dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID"))
    except Exception:
        return []
    if _out.returncode != 0:
        return []
    vals: List[int] = []
    for line in _out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(int(float(line.split()[0])))
        except (ValueError, IndexError):
            return []
    return vals


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
