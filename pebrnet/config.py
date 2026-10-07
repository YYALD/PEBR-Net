# PEBR-Net -- prior- and evidence-bounded reconstruction of noise-buried late-time borehole TEM responses.
# MIT License, see LICENSE.
"""Configuration: the dataclass configuration tree and the ablation presets.

This file is executed by pebrnet/__init__.py into the package namespace, in the order listed there;
names defined in the other files of the package are available here without imports.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import io
import re
import logging
import math
import os
import random
from collections import OrderedDict
import shutil
import sys
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torch.utils.checkpoint as _torch_checkpoint_698
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install a CUDA-compatible PyTorch build first."
    ) from exc


PROGRAM_NAME = "PEBR_NET"
PROJECT_ROOT = "."
PROGRAM_VERSION = "1.234.0"
ROW_DENOM_EPS = 1e-20


@dataclass
class PathConfig:
    init_checkpoint: str = ""
    paired_dir: str = "data/paired"
    clean_csv: str = ""
    clean_dir: str = ""
    noise_dir: str = ""
    noise_library_dir: str = ""
    survey_reference_csv: str = ""
    field_csv: str = ""
    field_gate_table: str = ""
    output_root: str = "runs/output"
    model_root: str = "runs/model"
    plot_root: str = "runs/figures"
    log_root: str = "runs/logs"
    resume_checkpoint: str = ""


@dataclass
class DataConfig:
    target_gates: int = 31
    time_extrapolation_tolerance: float = 0.25
    colored_noise_probability: float = 0.35
    colored_noise_relative_rms: Tuple[float, float] = (0.10, 0.50)
    colored_noise_ar_coeff: Tuple[float, float] = (0.0, 0.95)
    spike_noise_probability: float = 0.15
    spike_count_range: Tuple[int, int] = (1, 3)
    spike_relative_amplitude: Tuple[float, float] = (1.0, 4.0)
    harmonic_noise_probability: float = 0.20
    harmonic_freq_hz: Tuple[float, float] = (30.0, 600.0)
    harmonic_relative_rms: Tuple[float, float] = (0.10, 0.40)
    drift_noise_probability: float = 0.20
    drift_relative_rms: Tuple[float, float] = (0.10, 0.40)
    burst_noise_probability: float = 0.10
    burst_width_range: Tuple[int, int] = (2, 6)
    burst_relative_rms: Tuple[float, float] = (0.5, 2.0)
    late_start_index: int = 20
    field_gate_time_unit: str = "ms"
    unify_axis_to_measured: bool = False
    unified_axis_source: str = "noise_library"
    unified_gates: int = 0
    field_native_gates: int = 31
    extra_clean_enabled: bool = True
    measured_noise_enabled: bool = True
    uncovered_gate_policy: str = "refuse"
    paired_supervision: bool = True
    paired_background_window: int = 0
    paired_event_threshold: float = 0.15
    paired_split_policy: str = "grouped"
    paired_copy_rel_tol: float = 2.0e-3
    paired_real_neighbors: bool = True
    paired_anomaly_oversample: float = 0.35
    paired_hard_snr_oversample: float = 0.15
    weak_body_injection_prob: float = 0.35
    weak_body_contrast_min: float = 0.005
    weak_body_contrast_max: float = 0.20
    weak_body_width_min: float = 1.0
    weak_body_width_max: float = 4.0
    weak_body_positive_fraction: float = 0.7
    weak_body_eval_injection: bool = False
    paired_residual_replay: float = 0.20
    measured_noise_inject: float = 0.0
    val_contract_snr_protocol: bool = False
    injection_snr_calibrated_on_late_window: bool = True
    measured_noise_snr_low_focus: float = 0.5
    measured_noise_snr_min_db: float = -50.0
    measured_noise_subzero_probability: float = 0.0
    edge_reflect_mode: str = "auto"
    edge_reflect_k: int = 0
    prior_band_start: int = -1
    edge_patch_fraction: float = 0.15
    deep_negative_early_floor_db: float = 12.0
    impute_zero_gates: bool = True
    paired_clean_slice: float = 0.05
    replay_snr_curriculum: float = 0.50
    measured_noise_subzero_min_db: float = -10.0
    replay_snr_drop_db: float = 15.0
    replay_match_natural_snr: bool = True
    measured_noise_random_sign: bool = True
    library_contract_exempt: bool = True
    extra_clean_fraction: float = 0.25
    survey_profile_fraction: float = 0.0
    sim_noise_fraction: float = 0.20
    sim_noise_snr_db: Tuple[float, float] = (-30.0, 10.0)
    sim_noise_regimes: Tuple[str, ...] = ("smooth_bias", "spiky", "mixed")
    tail_loss_prob: float = 0.15
    tail_loss_fraction: Tuple[float, float] = (0.05, 0.75)
    paired_unit: str = "nT/s"
    survey_natural_fraction: float = 0.5
    survey_shift_fraction: float = 0.5
    survey_scale_jitter_db: float = 3.0
    survey_holdout_lo_m: float = 0.0
    survey_holdout_hi_m: float = 0.0
    survey_val_blocks: int = 0
    survey_val_block_len: int = 5
    survey_val_fold: int = 0
    survey_background_window: int = 0
    extra_clean_alignment: str = "auto"
    measured_noise_snr_max_db: float = 30.0
    measured_noise_component_mix: bool = True
    paired_alt_corruption: float = 0.25
    gate_monotonic_projection: bool = False
    clean_chunksize: int = 4096
    cache_dtype: str = "float32"
    province_val_fraction: float = 0.15
    province_test_fraction: float = 0.15
    province_split_seed: int = 20260715
    province_test_snr_db_range: Tuple[float, float] = (-8.0, 6.0)
    contract_test_snr_db_range: Tuple[float, float] = (-50.0, 30.0)
    forward_admission_filter: bool = True
    forward_admission_quantile: float = 0.005
    mix_clean_csv: bool = True
    mix_clean_csv_max_rows: int = 0
    field_group_size: int = 64
    mix_forward_fraction: float = 0.15
    forward_fraction_final: Optional[float] = None
    forward_fraction_total_epochs: int = 0
    split_shape_first: bool = True
    neighbor_channel_dropout: float = 0.0
    coherent_stack_slot: bool = False
    profile_patch_len: int = 9
    inference_window: int = 0
    eval_profile_blocks: bool = True
    field_signal_quantity: str = ""
    field_amplitude_unit: str = ""
    field_time_unit: str = ""
    field_receiver_area_m2: float = 0.0
    field_transmitter_moment_am2: float = 0.0
    field_normalization_history: str = ""

    split_group_size: int = 256
    split_dedup_by_shape: bool = True
    split_min_groups: int = 120
    split_min_val_groups: int = 6
    split_stratify: bool = True
    clean_augment: bool = False
    clean_aug_tau_dex: float = 0.15
    clean_aug_amp_dex: float = 0.30
    clean_aug_probability: float = 0.80
    train_fraction: float = 0.90
    val_fraction: float = 0.05
    train_samples_per_epoch: int = 200_000
    val_samples: int = 20_000
    test_samples: int = 20_000
    scale_sample_rows: int = 100_000
    global_scale_percentile: float = 95.0
    max_noise_rows: int = 500_000
    minimum_noise_rms: float = 1.0e-8
    snr_min_db: float = -40.0
    snr_max_db: float = 35.0
    snr_below_zero_probability: float = 0.45
    random_noise_sign_flip: bool = True
    noise_circular_shift_probability: float = 0.50
    noise_mixup_probability: float = 0.25
    noise_mixup_beta: Tuple[float, float] = (0.30, 0.70)
    white_noise_probability: float = 0.30
    white_noise_relative_rms: Tuple[float, float] = (0.02, 0.10)
    reliability_kappa: float = 1.5
    anomaly_profile_prob: float = 0.25
    anomaly_amp_range: Tuple[float, float] = (1.3, 3.0)
    anomaly_width_range: Tuple[float, float] = (1.0, 5.0)
    anomaly_center_jitter: float = 2.0
    anomaly_singleton_prob: float = 0.15
    anomaly_onset_gates: int = 3
    anomaly_negative_fraction: float = 0.25
    anomaly_suppress_range: Tuple[float, float] = (0.35, 0.75)
    anomaly_skew_fraction: float = 0.35
    reliability_window: int = 5
    num_neighbors: int = 8
    neighbor_gain_std: float = 0.06
    neighbor_tilt_std: float = 0.05
    neighbor_snr_jitter_db: float = 3.0


@dataclass
class ModelConfig:
    prior_width_mult: float = 1.0
    signal_fingerprint_bank: int = 96
    fingerprint_dim: int = 48
    lateral_joint_mode: str = "tikhonov"
    joint_profile_dim: int = 64
    joint_profile_heads: int = 2
    joint_profile_max_rel: int = 8
    lateral_fingerprint_bank: int = 32
    lateral_fingerprint_kernel: int = 9
    lateral_fingerprint_on_prior: bool = True
    lateral_fingerprint_on_output: bool = True
    lateral_joint_prior: bool = True
    lateral_joint_order: int = 3
    lateral_joint_lambda_late: float = 1.0
    lateral_joint_lambda_mid: float = 0.5
    lateral_joint_lambda_early: float = 0.02
    lateral_joint_weighting: str = "uniform"
    lateral_joint_noise_kappa: float = 0.0
    lateral_joint_noise_cap: float = 3.0
    late_measured_noise_floor: bool = True
    late_noise_floor_gamma: float = 0.98
    late_noise_floor_prior_sigma_z: float = 0.15
    late_noise_floor_prior_sigma_rel: float = 0.15
    innovation_evidence_gate: bool = False
    innovation_evidence_kappa: float = 3.0
    innovation_evidence_tau: float = 0.5
    innovation_precision_weighting: bool = False
    innovation_precision_mod_floor: float = 0.25
    innovation_warmup_epochs: int = 10
    innovation_robust_nu: float = 3.0
    output_measurement_fusion: bool = False
    lateral_gate_gamma: float = 1.0
    lateral_lambda_witness: str = "noise_head"
    lateral_witness_measurement_floor: bool = True
    lateral_joint_symmetric: bool = True
    lateral_joint_lambda_floor_late: float = 0.10
    lateral_joint_boundary_reflect: bool = False
    lateral_event_licence: float = 1.0
    lateral_event_licence_p0: float = 0.1
    lateral_two_scale: bool = False
    lateral_wide_multiplier: float = 12.0
    lateral_coherence_kappa: float = 3.0
    lateral_coherence_tau: float = 0.5
    lateral_gate_temporal_witness: bool = False
    lateral_retrieval_temperature: float = 4.0
    gates: int = 31
    base_channels: int = 64
    feature_channels: int = 128
    branch_channels: int = 32
    multiscale_kernels: Tuple[int, ...] = (3, 5, 7, 11)
    multiscale_dilations: Tuple[int, ...] = (1, 1, 2, 2)
    residual_blocks: int = 12
    residual_dilations: Tuple[int, ...] = (1, 2, 4)
    receptive_field_scaling: str = "auto"
    group_norm_groups: int = 8
    dropout: float = 0.0
    aux_arms_subbatch: bool = True
    aux_arms_quantum_patches: int = 4
    aux_bg_cap_share: float = 1.0
    aux_alt_cap_share: float = 1.0
    aux_full_fraction: float = 0.90
    activation_checkpointing: str = "auto"
    decoupled_arms: bool = False
    gate_local_norm: bool = False
    bounded_input: bool = True
    input_z_bounds: Tuple[float, float] = (-1.0, 6.0)
    mmse_blend: bool = True
    mmse_blend_in_training: bool = False
    mmse_eval_every: int = 2
    mmse_bstar_init: float = 0.65
    completion_gate_limit: int = 0
    attention_heads: int = 4
    refinement_limit: float = 0.75
    use_uncertainty_head: bool = True
    use_neighbor_attention: bool = True
    num_neighbors: int = 8
    use_manifold_branch: bool = True
    use_position_encoding: bool = True
    decoder_gate_norm_channel: bool = False
    use_neighbor_geometry: bool = True
    neighbor_distance_prior_init: float = 0.10
    use_event_residual: bool = True
    event_misfit_scale: float = 2.0
    struct_residual_scale: float = 1.2
    neighbor_noise_equicorrelation: float = 0.5
    manifold_blend_floor_late: float = 0.6
    v1122_parity: bool = False
    blend_floor_ramp_gates: int = 6
    manifold_pool_heads: int = 4
    use_manifold_innovation: bool = True
    manifold_innovation_clip: float = 1.5
    use_physics_atoms: bool = True
    physics_atom_count: int = 24
    physics_residual_scale: float = 0.15
    relaxation_stretch_exponents: Tuple[float, ...] = (1.0, 0.8, 0.6)
    use_relaxation_state_update: bool = True
    relaxation_update_scale: float = 0.20
    manifold_quality_gate: bool = True
    manifold_cap_min: float = 0.75
    manifold_cap_max: float = 0.999
    manifold_cap_error_log_scale: float = 0.55
    manifold_cap_error_midpoint: float = 0.06
    manifold_cap_error_scale: float = 0.015
    manifold_cap_max_step: float = 0.02
    per_trace_normalization: bool = False
    per_trace_norm_gates: int = 20
    amplitude_conditioning: str = "off"
    per_trace_norm_estimator: str = "centre_logmean"
    measurement_conditioned_gates: bool = False
    prior_sigma_floor: float = 0.05
    innovation_mod_floor: float = 0.5
    blend_ceiling_k: float = 0.0
    supervised_primary_fusion: bool = True
    continuation_arm_frac: float = 0.25
    use_library_subspace: bool = True
    library_subspace_rank: int = 12
    library_gate_bias_init: float = -2.2
    use_ridge_anchor: bool = True
    learned_continuation: bool = True
    physics_extrapolation: bool = False
    projection_continuation: bool = True
    continuation_gain_cap: float = 1.8
    use_coef_field: bool = True
    coef_field_learned_delta: bool = True
    use_profile_attention: bool = True
    coef_field_mu: float = 1e-3
    coef_field_gate_precision: bool = True
    coef_field_trust_mix: bool = True
    family_completion: bool = True
    family_gate_floor: bool = True
    family_validated_fusion: bool = True
    family_common_mode_stations: int = 2
    gate_loss_continuation: bool = True
    gate_loss_rate_bound: str = "off"
    gate_loss_departure: bool = True
    gate_loss_departure_init: float = -2.0
    informative_continuation: str = "off"
    informative_kappa: float = 1.5
    informative_min_gates: int = 8
    cdmr_calibrated: bool = True
    cdmr_prior_components: int = 64
    cdmr_noise_levels: Tuple[float, ...] = (0.5, 1.0, 2.0)
    seam_blend_floor_from_val: bool = True
    seam_blend_floor_all_gates: bool = False
    seam_blend_floor_enabled: bool = True
    seam_blend_floor_mode: str = "auto"
    seam_blend_floor_hard_max_gates: int = 8
    lateral_structure_witness_c0: float = 0.12
    lateral_witness_gates_attention: bool = True
    lateral_witness_dead_zone: float = 0.2
    ridge_gate_bias_init: float = -1.8
    use_ip_head: bool = False
    ip_atom_count: int = 16
    ip_tau_min_factor: float = 1.0
    ip_tau_max_factor: float = 20.0
    ip_chargeability_max: float = 0.9
    ip_init_scale: float = 1.0e-4
    late_start_index: int = 20


@dataclass
class ContractConfig:
    global_threshold: float = 0.03
    early_mid_threshold: float = 0.03
    row_pass_rate_target: float = 0.95
    late_threshold: float = 0.03
    late_bin_threshold: float = 0.03
    late_bin_min_count: int = 8
    late_cvar_threshold: float = 0.08
    initial_lambda_global: float = 0.0
    initial_lambda_early_mid: float = 0.0
    initial_lambda_late: float = 0.0
    initial_rho_global: float = 1.0
    initial_rho_early_mid: float = 1.0
    initial_rho_late: float = 1.5
    initial_lambda_late_cvar: float = 0.0
    initial_rho_late_cvar: float = 1.5
    cvar_violation_cap: float = 6.0
    cvar_activation_late_multiple: float = 3.0
    lambda_total_cap: float = 400.0
    initial_lambda_anomaly_late: float = 0.0
    initial_rho_anomaly_late: float = 1.5
    anomaly_arm_late_multiple: float = 5.0
    anomaly_warm_start_fraction: float = 0.5
    cvar_warm_start_fraction: float = 0.25
    anomaly_late_multiple: float = 2.0
    anomaly_recovery_target: float = 0.80
    anomaly_score_cap: float = 2.0
    anomaly_score_scale: float = 40.0
    event_auroc_target: float = 0.95
    late_priority_ratio: float = 1.0
    anomaly_lambda_cap_ratio: float = 1.0
    event_auroc_open: float = 0.60
    event_auroc_chance_band: float = 0.55
    late_cvar_rebase: bool = True
    late_cvar_rebase_margin: float = 0.10
    controller_feed_validation: bool = True
    rho_growth: float = 1.1
    rho_max: float = 4.0
    lambda_max: float = 8.0
    violation_soft_cap: float = 2.0
    stagnation_ratio: float = 0.95
    stagnation_patience: int = 3
    violation_ema: float = 0.6
    lambda_update_scale: float = 0.25
    lambda_stall_patience: int = 6
    lambda_stall_tolerance: float = 0.01
    lambda_stall_decay: float = 0.98
    lambda_stall_min: float = 10.0


@dataclass
class LossConfig:
    spectrum_backend: str = "real_dft"
    alpha_reconstruction: float = 0.20
    alpha_gate_balanced: float = 0.35
    alpha_gate_eq_row: float = 48.0
    gate_eq_row_tau: float = 0.03
    gate_eq_row_kappa: float = 0.03
    gate_eq_row_cap: float = 0.12
    alpha_do_no_harm: float = 0.6
    alpha_noise_orthogonal: float = 0.0
    alpha_real_noise_invariance: float = 0.15
    alpha_roughness_cap: float = 0.4
    alpha_gate_worst: float = 0.30
    alpha_base_reconstruction: float = 0.15
    alpha_noise: float = 0.50
    alpha_late_percurve: float = 0.5
    alpha_em_percurve: float = 0.5
    alpha_d1: float = 0.25
    alpha_d2: float = 0.10
    alpha_spectrum: float = 0.10
    alpha_event_gate: float = 0.5
    alpha_event_struct: float = 1.0
    alpha_event_quiet: float = 0.2
    quiet_guard_follow: bool = True
    forward_late_recon_weight: float = 0.5
    val_blend_floor: bool = True
    val_blend_floor_max: float = 0.93
    denoise_late_boost_max: float = 4.0
    alpha_event_station: float = 0.75
    event_pos_weight_max: float = 50.0
    alpha_residual_d2: float = 0.0
    alpha_profile_d1: float = 0.5
    alpha_profile_d2: float = 0.25
    alpha_profile_newext: float = 0.25
    alpha_profile_multigate: float = 0.15
    alpha_reliability: float = 0.20
    alpha_correction: float = 0.10
    alpha_correction_global: float = 0.02
    alpha_cvar: float = 0.30
    alpha_anomaly_diff: float = 1.0
    alpha_anomaly_amp: float = 0.0
    alpha_anomaly_sign: float = 0.5
    alpha_anomaly_patch: float = 1.0
    carryover_seam_gates: int = 7
    alpha_noise_carryover: float = 0.10
    alpha_neighbor_carryover: float = 0.30
    neighbor_carryover_rho: float = 0.5
    alpha_gate_monotonic: float = 0.2
    alpha_uncertainty: float = 0.05
    cvar_fraction: float = 0.20
    cvar_clip: float = 3.0
    alpha_metric_early: float = 0.25
    manifold_prior_target_background: bool = False
    alpha_prior_direct: float = 15.0
    alpha_contract_global: float = 40.0
    alpha_contract_early_mid: float = 40.0
    alpha_contract_late: float = 50.0
    alpha_metric_late: float = 0.45
    alpha_manifold_early: float = 0.10
    alpha_manifold_late: float = 0.60
    alpha_blend_oracle: float = 2.0
    alpha_fused_band_z: float = 15.0
    blend_oracle_all_arms: bool = True
    alpha_denoise_late: float = 0.25
    alpha_denoise_late_accuracy: float = 0.30
    alpha_denoise_seam: float = 0.30
    alpha_denoise_early: float = 0.10
    z_band_gain: float = 256.0
    z_band_balance_target: float = 1.0
    z_band_huber_delta: float = 0.02
    z_band_gain_min: float = 1.0
    z_band_gain_max: float = 8192.0
    z_band_balance_ema: float = 0.8
    z_band_balance_max_step: float = 1.1
    z_band_balance_deadband: float = 0.2
    strict_loss_terms: bool = True
    alpha_seam_floor_soft: float = 1.0
    denoise_late_leash_z: float = 0.5
    alpha_denoise_contract: float = 1.0
    alpha_denoise_ps_late: float = 0.35
    alpha_continuation: float = 0.30
    anomaly_visibility_floor: float = 0.0
    anomaly_visibility_z0: float = 1.0
    anomaly_visibility_width: float = 0.75
    alpha_library_coef: float = 1.0
    alpha_library_late: float = 0.5
    alpha_library_coef_cont: float = 0.5
    alpha_no_harm: float = 1.0
    no_harm_snr_db: float = 25.0
    anomaly_union_window: bool = True
    alpha_quiet_var: float = 0.5
    quiet_var_target: float = 0.01
    library_row_weight: float = 0.5
    library_row_cap_late: float = 0.10
    library_row_cap_global: float = 0.05
    acceptance_scale_init: float = 10.0
    acceptance_balance_share: float = 0.3
    acceptance_balance_every: int = 25
    acceptance_balance_max: float = 250.0
    acceptance_balance_start_epoch: int = 5
    acceptance_balance_row_late_max: float = 0.10
    acceptance_hinge_cap: float = 0.07
    alpha_ridge_late: float = 0.5
    alpha_order: float = 0.2
    alpha_gate_deep: float = 0.3
    alpha_late_per_sample: float = 0.5
    alpha_anchor_late_ps: float = 0.5
    alpha_recovery: float = 0.3
    alpha_pass_hinge: float = 1.0
    alpha_diff_orth: float = 0.15
    deep_gate_snr_db: float = -10.0
    alpha_manifold_late_rel: float = 0.8
    alpha_manifold_nrmse_late: float = 0.5
    alpha_manifold_ps_late: float = 0.25
    alpha_fusion_advantage: float = 0.10
    fusion_advantage_margin: float = 0.002
    alpha_noise_leak: float = 0.05
    ps_late_energy_floor: float = 0.0
    alpha_prior_sigma_calibration: float = 2.0
    manifold_lambda_follow_ref: float = 50.0
    manifold_lambda_follow_max: float = 20.0
    alpha_manifold_residual: float = 0.05
    alpha_spectrum_smooth: float = 0.02
    alpha_ip_chargeability: float = 0.50
    alpha_ip_early_positivity: float = 1.00
    alpha_ip_spectrum_smooth: float = 0.02
    alpha_ip_sparsity: float = 0.02
    eps: float = 1.0e-8


@dataclass
class TrainConfig:
    seed: int = 20260710
    batch_size: int = 512
    num_workers: int = 4
    gpu_synthesis: bool = True
    anomaly_probe: bool = True
    anomaly_probe_batches: int = 6
    anomaly_probe_body_prob: float = 0.5
    anomaly_probe_singleton_prob: float = 0.3
    remediation_rounds: int = 2
    remediation_epochs: int = 12
    remediation_lr_scale: float = 0.25
    compile_model: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    preflight: bool = True
    preflight_continue: bool = False
    prefetch_factor: int = 2
    shm_safety_fraction: float = 0.5
    amp: bool = True
    amp_dtype: str = "auto"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    deterministic: bool = False
    stage_a_max_epochs: int = 60
    stage_b_max_epochs: int = 80
    stage_c_epochs: int = 120
    stage_a_switch_global: float = 0.05
    stage_a_switch_late: float = 0.08
    stage_a_switch_patience: int = 5
    stage_a_min_plateau: int = 4
    stage_regression_ratio: float = 1.50
    stage_regression_patience: int = 12
    stage_regression_min_epochs: int = 50
    stage_a_switch_early_mid: float = 0.05
    stage_a_switch_cvar: float = 0.16
    stage_a_stall_early_mid: float = 0.09
    stage_a_stall_cvar: float = 0.30
    amp_scale_floor: float = 1.0e-4
    amp_scale_reset_value: float = 1024.0
    amp_scale_max_resets: int = 3
    frozen_metric_patience: int = 3
    patch_batch_cap: int = 2048
    grad_accum_steps: int = 1
    micro_batch_size: int = 0
    cpu_micro_batch_rows: int = 144
    vram_step_margin: float = 0.05
    vram_budget_fraction_explicit: bool = False
    dp_primary_share: float = 0.0
    shm_free_transport: bool = True
    effective_batch_size: int = 0
    vram_budget_fraction: float = 0.72
    vram_reserve_gb: float = 1.25
    heartbeat_steps: int = 8
    optimizer_batch_policy: str = "micro"
    min_effective_rows: int = 288
    vram_fill_real_step: bool = True
    aux_cap_sigma: float = 2.0
    aux_share_batches: int = 8
    vram_autotune: bool = True
    large_vram_preset: Optional[Dict[str, Any]] = None
    stage_a_min_epochs_before_transition: int = 50
    stage_a_stall_global: float = 0.08
    stage_a_stall_late: float = 0.15
    stage_init_from_previous_best: bool = True
    stage_score_ema_beta: float = 0.8
    late_terms_all_stages: bool = True
    stage_b_switch_global: float = 0.035
    stage_b_switch_late: float = 0.050
    stage_b_switch_patience: int = 5
    base_lr: float = 1.0e-3
    stage_b_shared_lr_factor: float = 0.10
    stage_b_noise_head_lr_factor: float = 0.20
    stage_b_router_lr_factor: float = 0.50
    stage_b_refine_lr_factor: float = 1.00
    prior_sigma_lr_factor: float = 20.0
    stage_c_lr_factor: float = 0.075
    neighbor_dropout: float = 0.0
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    lr_warmup_fraction: float = 0.03
    lr_min_ratio: float = 0.02
    use_ema: bool = True
    ema_decay: float = 0.999
    stage_switch_plateau_patience: int = 8
    aux_anneal_final: float = 0.4
    use_priority_gradient_projection: bool = True
    gradient_projection_every: int = 2
    validate_every: int = 1
    log_interval: int = 100
    early_stop_patience: int = 60


@dataclass
class RuntimeConfig:
    native_presentation: bool = True
    startup_audits: str = "fast"
    device: str = "cuda:1"
    multi_gpu: bool = True
    mode: str = "all"
    run_name: str = ""
    overwrite_cache: bool = False
    resolved_checkpoint_dir: str = ""
    resolved_figure_dir: str = ""
    checkpoint_dir_note: str = ""
    figure_dir_note: str = ""
    plot: bool = True
    allow_prior_monopoly: bool = False
    anomaly_suite_profiles: int = 48
    anomaly_suite_per_family: int = 12
    anomaly_suite_per_weak: int = 24
    anomaly_suite_per_cell: int = 8
    anomaly_suite_per_diag_cell: int = 3
    anomaly_suite_quiet: int = 96
    anomaly_suite_min_backgrounds: int = 64
    anomaly_suite_min_per_stratum: int = 8
    anomaly_suite_stations: int = 25
    paired_injected_probe: bool = False
    anomaly_suite_spacing_jitter: float = 0.15
    anomaly_suite_dead_trace_prob: float = 0.10
    amplitude_segment_spread_max: float = 1.20
    amplitude_gate_tilt_max: float = 1.50
    anomaly_detection_contrast: float = 0.05
    anomaly_sign_agreement_min: float = 0.95
    ip_sign_preservation_min: float = 0.90
    weak_anomaly_required_contrast: float = 0.05
    anomaly_new_extrema_max: float = 0.05
    weak_anomaly_recall_min: float = 0.90
    quiet_false_rate_max: float = 0.05
    first_run_full_notice: bool = False
    confirm_full_from_scratch: bool = False
    init_equivalence_max: float = 1e-5
    allow_amplitude_drift: bool = False
    export_torchscript: bool = True
    ablation: str = "full"
    figure_set: str = "full"
    evidence_missing_fractions: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    gate_loss_withheld: Tuple[int, ...] = (3, 6, 12, 18, 19, 22)
    gate_loss_reference_gates: int = 31
    gate_loss_profiles: int = 240
    gate_loss_fan_withheld: int = 18
    gate_loss_noise_tail_baseline: bool = True
    profile_extras: bool = False
    evidence_informed_cut: bool = True
    evidence_missing_scope: str = "both"
    evidence_samples: int = 512
    evidence_tail_gates: int = 6
    evidence_center_only: bool = False
    evidence_bootstrap: int = 2000
    evidence_skip_ip: bool = False
    skip_figure_package: bool = False
    section_autopsy_count: int = 3
    tail_panel_areas: int = 8
    anchor_compare_every: int = 4
    halt_on_nan: bool = False
    require_feasible_checkpoint: bool = True
    force_infer_unqualified: bool = False
    deployment_gate: bool = False
    field_amplitude_alignment: bool = True
    field_alignment_mode: str = "station"
    doi_sigma_s_per_m: float = 0.01
    doi_gate_tolerance_percent: float = 10.0
    ip_profiles: int = 48
    ip_stations: int = 41
    ip_snr_db: Tuple[float, ...] = (-3.0, 5.0, 15.0, 25.0)
    ip_score_threshold: float = 4.0
    ip_min_amplitude_sigma: float = 3.0
    ip_guard_hard: bool = False


@dataclass
class Config:
    """The configuration of a run: paths, data, model, losses, training, constraint targets and runtime."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def dataclass_to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: dataclass_to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


ABLATION_PRESETS: "OrderedDict[str, Tuple[str, Dict[str, Any]]]" = OrderedDict(
    [
        ("full", ("Complete model (no ablation)", {})),
        ("wo_manifold", ("No manifold-regeneration branch", {"model.use_manifold_branch": False})),
        ("wo_atoms", ("Free-form manifold head, no decay-atom physics cone", {"model.use_physics_atoms": False})),
        ("wo_innovation", ("No Kalman-style measurement innovation", {"model.use_manifold_innovation": False})),
        ("wo_neighbor", ("Single-trace: no neighbor cross-attention", {"model.use_neighbor_attention": False})),
        ("wo_posenc", ("No diffusion-coordinate position channels", {"model.use_position_encoding": False})),
        ("wo_nbgeo", ("No lateral neighbor geometry: content-only attention", {"model.use_neighbor_geometry": False})),
        ("wo_event", ("No event-gated structured residual", {"model.use_event_residual": False})),
        ("q0_minimal", ("Q0 minimal field-only baseline: no geometry, no RSSU, single relaxation "
                        "family, small physics residual, no event pathway, no anomaly injection ",
                        {"model.use_neighbor_geometry": False,
                         "model.use_relaxation_state_update": False,
                         "model.relaxation_stretch_exponents": (1.0,),
                         "model.use_event_residual": False,
                         "model.physics_residual_scale": 0.03,
                         "data.anomaly_profile_prob": 0.0,
                         "data.clean_augment": False,
                       "data.profile_patch_len": 0})),
        ("q1_geometry", ("Q0 + lateral neighbour geometry",
                         {"model.use_relaxation_state_update": False,
                          "model.relaxation_stretch_exponents": (1.0,),
                          "model.use_event_residual": False,
                          "model.physics_residual_scale": 0.03,
                          "data.anomaly_profile_prob": 0.0,
                          "data.clean_augment": False,
                       "data.profile_patch_len": 0})),
        ("q2_stretch", ("Q1 + stretched relaxation dictionary",
                        {"model.use_relaxation_state_update": False,
                         "model.use_event_residual": False,
                         "model.physics_residual_scale": 0.03,
                         "data.anomaly_profile_prob": 0.0,
                         "data.clean_augment": False,
                       "data.profile_patch_len": 0})),
        ("q3_rssu", ("Q2 + relaxation-state update unit",
                     {"model.use_event_residual": False,
                      "model.physics_residual_scale": 0.03,
                      "data.anomaly_profile_prob": 0.0,
                      "data.clean_augment": False,
                       "data.profile_patch_len": 0})),
        ("q4_event", ("Q3 + event pathway, anomaly training family and STATION-DIRECTION "
                      "profile losses (the full anomaly-preserving model at the reduced "
                      "physics residual)",
                      {"model.physics_residual_scale": 0.03,
                       "data.clean_augment": False,
                       "data.profile_patch_len": 9})),
        ("wo_blendfloor", ("No late blend floor (learned gate only)", {"model.manifold_blend_floor_late": 0.0})),
        ("wo_measgates", ("Free late gates: no measurement-conditioned blend / Kalman gain",
                          {"model.measurement_conditioned_gates": False})),
        ("wo_ip", ("Induction-only head: no signed dual-spectrum IP branch", {"model.use_ip_head": False})),
        ("wo_ptn", ("Per-trace amplitude conditioning OFF -- the population-z working domain",
                    {"model.amplitude_conditioning": "off"})),
        ("w_ptn", ("Per-trace amplitude conditioning forced ON, including the 31-gate design",
                   {"model.amplitude_conditioning": "on"})),
        ("wo_cleanaug", ("No physics-exact clean-curve augmentation", {"data.clean_augment": False})),
        ("wo_stratify", ("Unstratified group split", {"data.split_stratify": False})),
        (
            "wo_enrichment",
            (
                "Real composite noise only, no synthetic enrichment",
                {
                    "data.colored_noise_probability": 0.0,
                    "data.spike_noise_probability": 0.0,
                    "data.harmonic_noise_probability": 0.0,
                    "data.drift_noise_probability": 0.0,
                    "data.burst_noise_probability": 0.0,
                },
            ),
        ),
    ]
)
