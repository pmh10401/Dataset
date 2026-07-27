import argparse
import csv
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/minho/Documents/Dataset")
PYTHON_BIN = "/opt/homebrew/bin/python3"
STATE_PATH = DATA_DIR / "rank_experiments_sequential_state.json"
LOG_PATH = DATA_DIR / "rank_experiments_sequential.log"
QUEUE_LOCK_PATH = DATA_DIR / "rank_experiments_sequential.lock"
HEARTBEAT_PATH = DATA_DIR / "rank_experiments_dashboard_heartbeat.json"


def build_child_environment(base_env=None, *, worker_count="7"):
    """Build a stable import environment for every repository experiment entrypoint."""
    child_env = dict(os.environ if base_env is None else base_env)
    existing = [item for item in child_env.get("PYTHONPATH", "").split(os.pathsep) if item]
    root = str(ROOT)
    child_env["PYTHONPATH"] = os.pathsep.join([root, *[item for item in existing if item != root]])
    child_env["RANK_EXPERIMENT_WORKERS"] = str(worker_count)
    return child_env

EXPERIMENTS = [
    {
        "id": "rank_v1_train_evt",
        "script": ROOT / "run_rank_ensemble_calibration.py",
        "detail_csv": DATA_DIR / "vae_results_rank_ensemble_calibration_train_evt.csv",
        "summary_csv": DATA_DIR / "vae_results_rank_ensemble_calibration_train_evt_summary.csv",
        "stdout_log": DATA_DIR / "rank_ensemble_train_evt_stdout.log",
    },
    {
        "id": "rank_threshold_calibration",
        "script": ROOT / "run_rank_ensemble_threshold_calibration.py",
        "detail_csv": DATA_DIR / "rank_ensemble_threshold_calibration.csv",
        "summary_csv": DATA_DIR / "rank_ensemble_threshold_calibration_summary.csv",
        "stdout_log": DATA_DIR / "rank_ensemble_threshold_calibration_stdout.log",
    },
    {
        "id": "experiment_26_rocket",
        "script": ROOT / "run_experiment_26_rocket_parallel.py",
        "detail_csv": DATA_DIR / "experiment_26_rocket_results.csv",
        "summary_csv": DATA_DIR / "experiment_26_rocket_summary.csv",
        "stdout_log": DATA_DIR / "experiment_26_rocket_stdout.log",
    },
    {
        "id": "experiment_27_rocket_score_variants",
        "script": ROOT / "run_experiment_27_rocket_score_variants.py",
        "detail_csv": DATA_DIR / "experiment_27_rocket_score_variants_results.csv",
        "summary_csv": DATA_DIR / "experiment_27_rocket_score_variants_summary.csv",
        "stdout_log": DATA_DIR / "experiment_27_rocket_score_variants_stdout.log",
    },
    {
        "id": "experiment_27e_rocket_1024_top32",
        "script": ROOT / "run_experiment_27e_rocket_1024_top32.py",
        "detail_csv": DATA_DIR / "experiment_27e_rocket_1024_top32_results.csv",
        "summary_csv": DATA_DIR / "experiment_27e_rocket_1024_top32_summary.csv",
        "stdout_log": DATA_DIR / "experiment_27e_rocket_1024_top32_stdout.log",
    },
    {
        "id": "experiment_27f_rstsf_interval",
        "script": ROOT / "run_experiment_27f_rstsf_interval.py",
        "detail_csv": DATA_DIR / "experiment_27f_rstsf_interval_results.csv",
        "summary_csv": DATA_DIR / "experiment_27f_rstsf_interval_summary.csv",
        "stdout_log": DATA_DIR / "experiment_27f_rstsf_interval_stdout.log",
    },
    {
        "id": "experiment_28_minirocket_multirocket_features",
        "script": ROOT / "run_experiment_28_minirocket_multirocket_features.py",
        "detail_csv": DATA_DIR / "experiment_28_minirocket_multirocket_features_results.csv",
        "summary_csv": DATA_DIR / "experiment_28_minirocket_multirocket_features_summary.csv",
        "stdout_log": DATA_DIR / "experiment_28_minirocket_multirocket_features_stdout.log",
    },
    {
        "id": "experiment_29_train_normal_threshold_calibration",
        "script": ROOT / "run_experiment_29_train_normal_threshold_calibration.py",
        "detail_csv": DATA_DIR / "experiment_29_train_normal_threshold_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_29_train_normal_threshold_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_29_train_normal_threshold_calibration_stdout.log",
    },
    {
        "id": "experiment_30_knn_threshold_sweep",
        "script": ROOT / "run_experiment_30_knn_threshold_sweep.py",
        "detail_csv": DATA_DIR / "experiment_30_knn_threshold_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_30_knn_threshold_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_30_knn_threshold_sweep_stdout.log",
    },
    {
        "id": "experiment_31_knn_operational_budget_sweep",
        "script": ROOT / "run_experiment_31_knn_operational_budget_sweep.py",
        "detail_csv": DATA_DIR / "experiment_31_knn_operational_budget_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_31_knn_operational_budget_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_31_knn_operational_budget_sweep_stdout.log",
    },
    {
        "id": "experiment_32_knn_score_capacity_sweep",
        "script": ROOT / "run_experiment_32_knn_score_capacity_sweep.py",
        "detail_csv": DATA_DIR / "experiment_32_knn_score_capacity_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_32_knn_score_capacity_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_32_knn_score_capacity_sweep_stdout.log",
    },
    {
        "id": "experiment_34_balanced_feature_capacity_sweep",
        "script": ROOT / "run_experiment_34_balanced_feature_capacity_sweep.py",
        "detail_csv": DATA_DIR / "experiment_34_balanced_feature_capacity_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_34_balanced_feature_capacity_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_34_balanced_feature_capacity_sweep_stdout.log",
    },
    {
        "id": "experiment_35_balanced_threshold_policy_sweep",
        "script": ROOT / "run_experiment_35_balanced_threshold_policy_sweep.py",
        "detail_csv": DATA_DIR / "experiment_35_balanced_threshold_policy_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_35_balanced_threshold_policy_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_35_balanced_threshold_policy_sweep_stdout.log",
    },
    {
        "id": "experiment_36_balanced_score_normalization_sweep",
        "script": ROOT / "run_experiment_36_balanced_score_normalization_sweep.py",
        "detail_csv": DATA_DIR / "experiment_36_balanced_score_normalization_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_36_balanced_score_normalization_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_36_balanced_score_normalization_sweep_stdout.log",
    },
    {
        "id": "experiment_37_balanced_bagged_rocket_ensemble",
        "script": ROOT / "run_experiment_37_balanced_bagged_rocket_ensemble.py",
        "detail_csv": DATA_DIR / "experiment_37_balanced_bagged_rocket_ensemble_results.csv",
        "summary_csv": DATA_DIR / "experiment_37_balanced_bagged_rocket_ensemble_summary.csv",
        "stdout_log": DATA_DIR / "experiment_37_balanced_bagged_rocket_ensemble_stdout.log",
    },
    {
        "id": "experiment_38_balanced_actual_length_handling",
        "script": ROOT / "run_experiment_38_balanced_actual_length_handling.py",
        "detail_csv": DATA_DIR / "experiment_38_balanced_actual_length_handling_results.csv",
        "summary_csv": DATA_DIR / "experiment_38_balanced_actual_length_handling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_38_balanced_actual_length_handling_stdout.log",
    },
    {
        "id": "experiment_39_balanced_candidate_retest",
        "script": ROOT / "run_experiment_39_balanced_candidate_retest.py",
        "detail_csv": DATA_DIR / "experiment_39_balanced_candidate_retest_results.csv",
        "summary_csv": DATA_DIR / "experiment_39_balanced_candidate_retest_summary.csv",
        "stdout_log": DATA_DIR / "experiment_39_balanced_candidate_retest_stdout.log",
    },
    {
        "id": "experiment_35_original_threshold_policy_sweep",
        "script": ROOT / "run_experiment_35_original_threshold_policy_sweep.py",
        "detail_csv": DATA_DIR / "experiment_35_original_threshold_policy_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_35_original_threshold_policy_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_35_original_threshold_policy_sweep_stdout.log",
    },
    {
        "id": "experiment_36_original_score_normalization_sweep",
        "script": ROOT / "run_experiment_36_original_score_normalization_sweep.py",
        "detail_csv": DATA_DIR / "experiment_36_original_score_normalization_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_36_original_score_normalization_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_36_original_score_normalization_sweep_stdout.log",
    },
    {
        "id": "experiment_37_original_bagged_rocket_ensemble",
        "script": ROOT / "run_experiment_37_original_bagged_rocket_ensemble.py",
        "detail_csv": DATA_DIR / "experiment_37_original_bagged_rocket_ensemble_results.csv",
        "summary_csv": DATA_DIR / "experiment_37_original_bagged_rocket_ensemble_summary.csv",
        "stdout_log": DATA_DIR / "experiment_37_original_bagged_rocket_ensemble_stdout.log",
    },
    {
        "id": "experiment_38_original_actual_length_handling",
        "script": ROOT / "run_experiment_38_original_actual_length_handling.py",
        "detail_csv": DATA_DIR / "experiment_38_original_actual_length_handling_results.csv",
        "summary_csv": DATA_DIR / "experiment_38_original_actual_length_handling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_38_original_actual_length_handling_stdout.log",
    },
    {
        "id": "experiment_39_original_candidate_retest",
        "script": ROOT / "run_experiment_39_original_candidate_retest.py",
        "detail_csv": DATA_DIR / "experiment_39_original_candidate_retest_results.csv",
        "summary_csv": DATA_DIR / "experiment_39_original_candidate_retest_summary.csv",
        "stdout_log": DATA_DIR / "experiment_39_original_candidate_retest_stdout.log",
    },
    {
        "id": "experiment_40_original_score_normalization_sweep",
        "script": ROOT / "run_experiment_40_original_score_normalization_sweep.py",
        "detail_csv": DATA_DIR / "experiment_40_original_score_normalization_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_40_original_score_normalization_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_40_original_score_normalization_sweep_stdout.log",
    },
    {
        "id": "experiment_41_multi_aug_robust_baseline",
        "script": ROOT / "run_experiment_41_multi_aug_robust_baseline.py",
        "detail_csv": DATA_DIR / "experiment_41_multi_aug_robust_baseline_results.csv",
        "summary_csv": DATA_DIR / "experiment_41_multi_aug_robust_baseline_summary.csv",
        "stdout_log": DATA_DIR / "experiment_41_multi_aug_robust_baseline_stdout.log",
    },
    {
        "id": "experiment_42_multi_aug_robust_operational",
        "script": ROOT / "run_experiment_42_multi_aug_robust_operational.py",
        "detail_csv": DATA_DIR / "experiment_42_multi_aug_robust_operational_results.csv",
        "summary_csv": DATA_DIR / "experiment_42_multi_aug_robust_operational_summary.csv",
        "stdout_log": DATA_DIR / "experiment_42_multi_aug_robust_operational_stdout.log",
    },
    {
        "id": "experiment_43_explanation_space_transforms",
        "script": ROOT / "run_experiment_43_explanation_space_transforms.py",
        "detail_csv": DATA_DIR / "experiment_43_explanation_space_transforms_results.csv",
        "summary_csv": DATA_DIR / "experiment_43_explanation_space_transforms_summary.csv",
        "stdout_log": DATA_DIR / "experiment_43_explanation_space_transforms_stdout.log",
    },
    {
        "id": "experiment_44_classical_embedding_baselines",
        "script": ROOT / "run_experiment_44_classical_embedding_baselines.py",
        "detail_csv": DATA_DIR / "experiment_44_classical_embedding_baselines_results.csv",
        "summary_csv": DATA_DIR / "experiment_44_classical_embedding_baselines_summary.csv",
        "stdout_log": DATA_DIR / "experiment_44_classical_embedding_baselines_stdout.log",
    },
    {
        "id": "experiment_45_model_hard_diagnostic_harness",
        "script": ROOT / "run_experiment_45_model_hard_diagnostic_harness.py",
        "detail_csv": DATA_DIR / "experiment_45_model_hard_diagnostic_harness_results.csv",
        "summary_csv": DATA_DIR / "experiment_45_model_hard_diagnostic_harness_summary.csv",
        "stdout_log": DATA_DIR / "experiment_45_model_hard_diagnostic_harness_stdout.log",
    },
    {
        "id": "experiment_46_model_hard_interval_drcif_lite",
        "script": ROOT / "run_experiment_46_model_hard_interval_drcif_lite.py",
        "detail_csv": DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_results.csv",
        "summary_csv": DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_summary.csv",
        "stdout_log": DATA_DIR / "experiment_46_model_hard_interval_drcif_lite_stdout.log",
    },
    {
        "id": "experiment_47_model_hard_frequency_rocket",
        "script": ROOT / "run_experiment_47_model_hard_frequency_rocket.py",
        "detail_csv": DATA_DIR / "experiment_47_model_hard_frequency_rocket_results.csv",
        "summary_csv": DATA_DIR / "experiment_47_model_hard_frequency_rocket_summary.csv",
        "stdout_log": DATA_DIR / "experiment_47_model_hard_frequency_rocket_stdout.log",
    },
    {
        "id": "experiment_48_model_hard_shapelet_prototype",
        "script": ROOT / "run_experiment_48_model_hard_shapelet_prototype.py",
        "detail_csv": DATA_DIR / "experiment_48_model_hard_shapelet_prototype_results.csv",
        "summary_csv": DATA_DIR / "experiment_48_model_hard_shapelet_prototype_summary.csv",
        "stdout_log": DATA_DIR / "experiment_48_model_hard_shapelet_prototype_stdout.log",
    },
    {
        "id": "experiment_49_model_hard_anomaly_injection",
        "script": ROOT / "run_experiment_49_model_hard_anomaly_injection.py",
        "detail_csv": DATA_DIR / "experiment_49_model_hard_anomaly_injection_results.csv",
        "summary_csv": DATA_DIR / "experiment_49_model_hard_anomaly_injection_summary.csv",
        "stdout_log": DATA_DIR / "experiment_49_model_hard_anomaly_injection_stdout.log",
    },
    {
        "id": "experiment_50_model_hard_timeseries_imaging",
        "script": ROOT / "run_experiment_50_model_hard_timeseries_imaging.py",
        "detail_csv": DATA_DIR / "experiment_50_model_hard_timeseries_imaging_results.csv",
        "summary_csv": DATA_DIR / "experiment_50_model_hard_timeseries_imaging_summary.csv",
        "stdout_log": DATA_DIR / "experiment_50_model_hard_timeseries_imaging_stdout.log",
    },
    {
        "id": "experiment_51_full_timeseries_imaging_selector_probe",
        "script": ROOT / "run_experiment_51_full_timeseries_imaging_selector_probe.py",
        "detail_csv": DATA_DIR / "experiment_51_full_timeseries_imaging_selector_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_51_full_timeseries_imaging_selector_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_51_full_timeseries_imaging_selector_probe_stdout.log",
    },
    {
        "id": "experiment_52_imaging_multiscale_fusion_probe",
        "script": ROOT / "run_experiment_52_imaging_multiscale_fusion_probe.py",
        "detail_csv": DATA_DIR / "experiment_52_imaging_multiscale_fusion_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_52_imaging_multiscale_fusion_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_52_imaging_multiscale_fusion_probe_stdout.log",
    },
    {
        "id": "experiment_53a_imaging_texture_features_probe",
        "script": ROOT / "run_experiment_53a_imaging_texture_features_probe.py",
        "detail_csv": DATA_DIR / "experiment_53a_imaging_texture_features_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_53a_imaging_texture_features_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_53a_imaging_texture_features_probe_stdout.log",
    },
    {
        "id": "experiment_53b_imaging_train_robust_scaling_probe",
        "script": ROOT / "run_experiment_53b_imaging_train_robust_scaling_probe.py",
        "detail_csv": DATA_DIR / "experiment_53b_imaging_train_robust_scaling_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_53b_imaging_train_robust_scaling_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_53b_imaging_train_robust_scaling_probe_stdout.log",
    },
    {
        "id": "experiment_54_imaging_resolution_pca_sweep",
        "script": ROOT / "run_experiment_54_imaging_resolution_pca_sweep.py",
        "detail_csv": DATA_DIR / "experiment_54_imaging_resolution_pca_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_54_imaging_resolution_pca_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_54_imaging_resolution_pca_sweep_stdout.log",
    },
    {
        "id": "experiment_55_imaging_scaling_sweep",
        "script": ROOT / "run_experiment_55_imaging_scaling_sweep.py",
        "detail_csv": DATA_DIR / "experiment_55_imaging_scaling_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_55_imaging_scaling_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_55_imaging_scaling_sweep_stdout.log",
    },
    {
        "id": "experiment_56_imaging_glcm_texture_probe",
        "script": ROOT / "run_experiment_56_imaging_glcm_texture_probe.py",
        "detail_csv": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_56_imaging_glcm_texture_probe_stdout.log",
    },
    {
        "id": "experiment_57_imaging_small_cnn_mps_probe",
        "script": ROOT / "run_experiment_57_imaging_small_cnn_mps_probe.py",
        "detail_csv": DATA_DIR / "experiment_57_imaging_small_cnn_mps_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_57_imaging_small_cnn_mps_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_57_imaging_small_cnn_mps_probe_stdout.log",
    },
    {
        "id": "experiment_58_imaging_pretrained_cnn_feature_probe",
        "script": ROOT / "run_experiment_58_imaging_pretrained_cnn_feature_probe.py",
        "detail_csv": DATA_DIR / "experiment_58_imaging_pretrained_cnn_feature_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_58_imaging_pretrained_cnn_feature_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_58_imaging_pretrained_cnn_feature_probe_stdout.log",
    },
    {
        "id": "experiment_59_rocket_imaging_selector",
        "script": ROOT / "run_experiment_59_rocket_imaging_selector.py",
        "detail_csv": DATA_DIR / "experiment_59_rocket_imaging_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_59_rocket_imaging_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_59_rocket_imaging_selector_stdout.log",
    },
    {
        "id": "experiment_60_selector_fp_guard_variants",
        "script": ROOT / "run_experiment_60_selector_fp_guard_variants.py",
        "detail_csv": DATA_DIR / "experiment_60_selector_fp_guard_variants_results.csv",
        "summary_csv": DATA_DIR / "experiment_60_selector_fp_guard_variants_summary.csv",
        "stdout_log": DATA_DIR / "experiment_60_selector_fp_guard_variants_stdout.log",
    },
    {
        "id": "experiment_61_selector_index_agreement",
        "script": ROOT / "run_experiment_61_selector_index_agreement.py",
        "detail_csv": DATA_DIR / "experiment_61_selector_index_agreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_61_selector_index_agreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_61_selector_index_agreement_stdout.log",
    },
    {
        "id": "experiment_62_selector_guarded_index_agreement",
        "script": ROOT / "run_experiment_62_selector_guarded_index_agreement.py",
        "detail_csv": DATA_DIR / "experiment_62_selector_guarded_index_agreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_62_selector_guarded_index_agreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_62_selector_guarded_index_agreement_stdout.log",
    },
    {
        "id": "experiment_63_guarded_cap_sweep",
        "script": ROOT / "run_experiment_63_guarded_cap_sweep.py",
        "detail_csv": DATA_DIR / "experiment_63_guarded_cap_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_63_guarded_cap_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_63_guarded_cap_sweep_stdout.log",
    },
    {
        "id": "experiment_64_guarded_with_fallback",
        "script": ROOT / "run_experiment_64_guarded_with_fallback.py",
        "detail_csv": DATA_DIR / "experiment_64_guarded_with_fallback_results.csv",
        "summary_csv": DATA_DIR / "experiment_64_guarded_with_fallback_summary.csv",
        "stdout_log": DATA_DIR / "experiment_64_guarded_with_fallback_stdout.log",
    },
    {
        "id": "experiment_65_confidence_tier_selector",
        "script": ROOT / "run_experiment_65_confidence_tier_selector.py",
        "detail_csv": DATA_DIR / "experiment_65_confidence_tier_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_65_confidence_tier_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_65_confidence_tier_selector_stdout.log",
    },
    {
        "id": "experiment_66_train_normal_alert_budget",
        "script": ROOT / "run_experiment_66_train_normal_alert_budget.py",
        "detail_csv": DATA_DIR / "experiment_66_train_normal_alert_budget_results.csv",
        "summary_csv": DATA_DIR / "experiment_66_train_normal_alert_budget_summary.csv",
        "stdout_log": DATA_DIR / "experiment_66_train_normal_alert_budget_stdout.log",
    },
    {
        "id": "experiment_67_hard_family_fallback_selector",
        "script": ROOT / "run_experiment_67_hard_family_fallback_selector.py",
        "detail_csv": DATA_DIR / "experiment_67_hard_family_fallback_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_67_hard_family_fallback_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_67_hard_family_fallback_selector_stdout.log",
    },
    {
        "id": "experiment_68_final_operational_selector",
        "script": ROOT / "run_experiment_68_final_operational_selector.py",
        "detail_csv": DATA_DIR / "experiment_68_final_operational_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_68_final_operational_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_68_final_operational_selector_stdout.log",
    },
    {
        "id": "experiment_68b_final_operational_fallback_sweep",
        "script": ROOT / "run_experiment_68b_final_operational_fallback_sweep.py",
        "detail_csv": DATA_DIR / "experiment_68b_final_operational_fallback_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_68b_final_operational_fallback_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_68b_final_operational_fallback_sweep_stdout.log",
    },
    {
        "id": "experiment_69_operational_train_exceed_calibration",
        "script": ROOT / "run_experiment_69_operational_train_exceed_calibration.py",
        "detail_csv": DATA_DIR / "experiment_69_operational_train_exceed_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_69_operational_train_exceed_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_69_operational_train_exceed_calibration_stdout.log",
    },
    {
        "id": "experiment_69b_no_prediction_fallback_calibration",
        "script": ROOT / "run_experiment_69b_no_prediction_fallback_calibration.py",
        "detail_csv": DATA_DIR / "experiment_69b_no_prediction_fallback_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_69b_no_prediction_fallback_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_69b_no_prediction_fallback_calibration_stdout.log",
    },
    {
        "id": "experiment_70_zero_mode_family_repair_selector",
        "script": ROOT / "run_experiment_70_zero_mode_family_repair_selector.py",
        "detail_csv": DATA_DIR / "experiment_70_zero_mode_family_repair_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_70_zero_mode_family_repair_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_70_zero_mode_family_repair_selector_stdout.log",
    },
    {
        "id": "experiment_71a_large_data_rocket_fallback",
        "script": ROOT / "run_experiment_71a_large_data_rocket_fallback.py",
        "detail_csv": DATA_DIR / "experiment_71a_large_data_rocket_fallback_results.csv",
        "summary_csv": DATA_DIR / "experiment_71a_large_data_rocket_fallback_summary.csv",
        "stdout_log": DATA_DIR / "experiment_71a_large_data_rocket_fallback_stdout.log",
    },
    {
        "id": "experiment_71b_large_data_rocket_review_tier",
        "script": ROOT / "run_experiment_71b_large_data_rocket_review_tier.py",
        "detail_csv": DATA_DIR / "experiment_71b_large_data_rocket_review_tier_results.csv",
        "summary_csv": DATA_DIR / "experiment_71b_large_data_rocket_review_tier_summary.csv",
        "stdout_log": DATA_DIR / "experiment_71b_large_data_rocket_review_tier_stdout.log",
    },
    {
        "id": "experiment_72a_large_data_rank_ensemble",
        "script": ROOT / "run_experiment_72a_large_data_rank_ensemble.py",
        "detail_csv": DATA_DIR / "experiment_72a_large_data_rank_ensemble_results.csv",
        "summary_csv": DATA_DIR / "experiment_72a_large_data_rank_ensemble_summary.csv",
        "stdout_log": DATA_DIR / "experiment_72a_large_data_rank_ensemble_stdout.log",
    },
    {
        "id": "experiment_72b_large_data_source_disagreement",
        "script": ROOT / "run_experiment_72b_large_data_source_disagreement.py",
        "detail_csv": DATA_DIR / "experiment_72b_large_data_source_disagreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_72b_large_data_source_disagreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_72b_large_data_source_disagreement_stdout.log",
    },
    {
        "id": "experiment_73a_large_rank_rocket_guard",
        "script": ROOT / "run_experiment_73a_large_rank_rocket_guard.py",
        "detail_csv": DATA_DIR / "experiment_73a_large_rank_rocket_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73a_large_rank_rocket_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73a_large_rank_rocket_guard_stdout.log",
    },
    {
        "id": "experiment_73b_large_rank_two_model_guard",
        "script": ROOT / "run_experiment_73b_large_rank_two_model_guard.py",
        "detail_csv": DATA_DIR / "experiment_73b_large_rank_two_model_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73b_large_rank_two_model_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73b_large_rank_two_model_guard_stdout.log",
    },
    {
        "id": "experiment_73c_large_rank_budget_guard",
        "script": ROOT / "run_experiment_73c_large_rank_budget_guard.py",
        "detail_csv": DATA_DIR / "experiment_73c_large_rank_budget_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73c_large_rank_budget_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73c_large_rank_budget_guard_stdout.log",
    },
    {
        "id": "experiment_73d_large_rank_combined_guard",
        "script": ROOT / "run_experiment_73d_large_rank_combined_guard.py",
        "detail_csv": DATA_DIR / "experiment_73d_large_rank_combined_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_73d_large_rank_combined_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_73d_large_rank_combined_guard_stdout.log",
    },
    {
        "id": "experiment_74a_large_rank_margin_guard",
        "script": ROOT / "run_experiment_74a_large_rank_margin_guard.py",
        "detail_csv": DATA_DIR / "experiment_74a_large_rank_margin_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_74a_large_rank_margin_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74a_large_rank_margin_guard_stdout.log",
    },
    {
        "id": "experiment_74b_large_rank_family_budget",
        "script": ROOT / "run_experiment_74b_large_rank_family_budget.py",
        "detail_csv": DATA_DIR / "experiment_74b_large_rank_family_budget_results.csv",
        "summary_csv": DATA_DIR / "experiment_74b_large_rank_family_budget_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74b_large_rank_family_budget_stdout.log",
    },
    {
        "id": "experiment_74c_large_rank_margin_family_guard",
        "script": ROOT / "run_experiment_74c_large_rank_margin_family_guard.py",
        "detail_csv": DATA_DIR / "experiment_74c_large_rank_margin_family_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_74c_large_rank_margin_family_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74c_large_rank_margin_family_guard_stdout.log",
    },
    {
        "id": "experiment_74d_large_rank_review_tier_split",
        "script": ROOT / "run_experiment_74d_large_rank_review_tier_split.py",
        "detail_csv": DATA_DIR / "experiment_74d_large_rank_review_tier_split_results.csv",
        "summary_csv": DATA_DIR / "experiment_74d_large_rank_review_tier_split_summary.csv",
        "stdout_log": DATA_DIR / "experiment_74d_large_rank_review_tier_split_stdout.log",
    },
    {
        "id": "experiment_75_score_source_diagnostic_harness",
        "script": ROOT / "run_experiment_75_score_source_diagnostic_harness.py",
        "detail_csv": DATA_DIR / "experiment_75_score_source_diagnostic_harness_results.csv",
        "summary_csv": DATA_DIR / "experiment_75_score_source_diagnostic_harness_summary.csv",
        "stdout_log": DATA_DIR / "experiment_75_score_source_diagnostic_harness_stdout.log",
    },
    {
        "id": "experiment_76_spectral_derivative_rocket_hard_family",
        "script": ROOT / "run_experiment_76_spectral_derivative_rocket_hard_family.py",
        "detail_csv": DATA_DIR / "experiment_76_spectral_derivative_rocket_hard_family_results.csv",
        "summary_csv": DATA_DIR / "experiment_76_spectral_derivative_rocket_hard_family_summary.csv",
        "stdout_log": DATA_DIR / "experiment_76_spectral_derivative_rocket_hard_family_stdout.log",
    },
    {
        "id": "experiment_77_shapelet_prototype_low_train_guard",
        "script": ROOT / "run_experiment_77_shapelet_prototype_low_train_guard.py",
        "detail_csv": DATA_DIR / "experiment_77_shapelet_prototype_low_train_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_77_shapelet_prototype_low_train_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_77_shapelet_prototype_low_train_guard_stdout.log",
    },
    {
        "id": "experiment_78_robust_density_corrected_score",
        "script": ROOT / "run_experiment_78_robust_density_corrected_score.py",
        "detail_csv": DATA_DIR / "experiment_78_robust_density_corrected_score_results.csv",
        "summary_csv": DATA_DIR / "experiment_78_robust_density_corrected_score_summary.csv",
        "stdout_log": DATA_DIR / "experiment_78_robust_density_corrected_score_stdout.log",
    },
    {
        "id": "experiment_79_vae_epoch_sweep_guarded",
        "script": ROOT / "run_experiment_79_vae_epoch_sweep_guarded.py",
        "detail_csv": DATA_DIR / "experiment_79_vae_epoch_sweep_guarded_results.csv",
        "summary_csv": DATA_DIR / "experiment_79_vae_epoch_sweep_guarded_summary.csv",
        "stdout_log": DATA_DIR / "experiment_79_vae_epoch_sweep_guarded_stdout.log",
    },
    {
        "id": "experiment_80_train_only_score_selector",
        "script": ROOT / "run_experiment_80_train_only_score_selector.py",
        "detail_csv": DATA_DIR / "experiment_80_train_only_score_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_80_train_only_score_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_80_train_only_score_selector_stdout.log",
    },
    {
        "id": "experiment_81_aeon_multirocket_official_full",
        "script": ROOT / "run_experiment_81_aeon_multirocket_official_full.py",
        "detail_csv": DATA_DIR / "experiment_81_aeon_multirocket_official_full_results.csv",
        "summary_csv": DATA_DIR / "experiment_81_aeon_multirocket_official_full_summary.csv",
        "stdout_log": DATA_DIR / "experiment_81_aeon_multirocket_official_full_stdout.log",
    },
    {
        "id": "experiment_82_hydra_hard_family_subset",
        "script": ROOT / "run_experiment_82_hydra_hard_family_subset.py",
        "detail_csv": DATA_DIR / "experiment_82_hydra_hard_family_subset_results.csv",
        "summary_csv": DATA_DIR / "experiment_82_hydra_hard_family_subset_summary.csv",
        "stdout_log": DATA_DIR / "experiment_82_hydra_hard_family_subset_stdout.log",
    },
    {
        "id": "experiment_83_multirocket_hydra_hard339",
        "script": ROOT / "run_experiment_83_multirocket_hydra_hard339.py",
        "detail_csv": DATA_DIR / "experiment_83_multirocket_hydra_hard339_results.csv",
        "summary_csv": DATA_DIR / "experiment_83_multirocket_hydra_hard339_summary.csv",
        "stdout_log": DATA_DIR / "experiment_83_multirocket_hydra_hard339_stdout.log",
    },
    {
        "id": "experiment_84_feature_pruning_operational_stability",
        "script": ROOT / "run_experiment_84_feature_pruning_operational_stability.py",
        "detail_csv": DATA_DIR / "experiment_84_feature_pruning_operational_stability_results.csv",
        "summary_csv": DATA_DIR / "experiment_84_feature_pruning_operational_stability_summary.csv",
        "stdout_log": DATA_DIR / "experiment_84_feature_pruning_operational_stability_stdout.log",
    },
    {
        "id": "experiment_85_exp84_hard_specialist_selector",
        "script": ROOT / "run_experiment_85_exp84_hard_specialist_selector.py",
        "detail_csv": DATA_DIR / "experiment_85_exp84_hard_specialist_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_85_exp84_hard_specialist_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_85_exp84_hard_specialist_selector_stdout.log",
    },
    {
        "id": "experiment_86_train_only_agreement_exp84_selector",
        "script": ROOT / "run_experiment_86_train_only_agreement_exp84_selector.py",
        "detail_csv": DATA_DIR / "experiment_86_train_only_agreement_exp84_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_86_train_only_agreement_exp84_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_86_train_only_agreement_exp84_selector_stdout.log",
    },
    {
        "id": "experiment_87_exp84_index_diagnostics",
        "script": ROOT / "run_experiment_87_exp84_index_diagnostics.py",
        "detail_csv": DATA_DIR / "experiment_87_exp84_index_diagnostics_results.csv",
        "summary_csv": DATA_DIR / "experiment_87_exp84_index_diagnostics_summary.csv",
        "stdout_log": DATA_DIR / "experiment_87_exp84_index_diagnostics_stdout.log",
    },
    {
        "id": "experiment_88_true_agreement_exp84_selector",
        "script": ROOT / "run_experiment_88_true_agreement_exp84_selector.py",
        "detail_csv": DATA_DIR / "experiment_88_true_agreement_exp84_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_88_true_agreement_exp84_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_88_true_agreement_exp84_selector_stdout.log",
    },
    {
        "id": "experiment_89_74d_with_exp84_candidate",
        "script": ROOT / "run_experiment_89_74d_with_exp84_candidate.py",
        "detail_csv": DATA_DIR / "experiment_89_74d_with_exp84_candidate_results.csv",
        "summary_csv": DATA_DIR / "experiment_89_74d_with_exp84_candidate_summary.csv",
        "stdout_log": DATA_DIR / "experiment_89_74d_with_exp84_candidate_stdout.log",
    },
    {
        "id": "experiment_90_zero_f1_repair_selector",
        "script": ROOT / "run_experiment_90_zero_f1_repair_selector.py",
        "detail_csv": DATA_DIR / "experiment_90_zero_f1_repair_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_90_zero_f1_repair_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_90_zero_f1_repair_selector_stdout.log",
    },
    {
        "id": "experiment_91_guarded_candidate_union_repair",
        "script": ROOT / "run_experiment_91_guarded_candidate_union_repair.py",
        "detail_csv": DATA_DIR / "experiment_91_guarded_candidate_union_repair_results.csv",
        "summary_csv": DATA_DIR / "experiment_91_guarded_candidate_union_repair_summary.csv",
        "stdout_log": DATA_DIR / "experiment_91_guarded_candidate_union_repair_stdout.log",
    },
    {
        "id": "experiment_92_operational_hybrid_selector",
        "script": ROOT / "run_experiment_92_operational_hybrid_selector.py",
        "detail_csv": DATA_DIR / "experiment_92_operational_hybrid_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_92_operational_hybrid_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_92_operational_hybrid_selector_stdout.log",
    },
    {
        "id": "experiment_93_nonpos_candidate_reranker",
        "script": ROOT / "run_experiment_93_nonpos_candidate_reranker.py",
        "detail_csv": DATA_DIR / "experiment_93_nonpos_candidate_reranker_results.csv",
        "summary_csv": DATA_DIR / "experiment_93_nonpos_candidate_reranker_summary.csv",
        "stdout_log": DATA_DIR / "experiment_93_nonpos_candidate_reranker_stdout.log",
    },
    {
        "id": "experiment_94_nonpos_rank_consensus_v2",
        "script": ROOT / "run_experiment_94_nonpos_rank_consensus_v2.py",
        "detail_csv": DATA_DIR / "experiment_94_nonpos_rank_consensus_v2_results.csv",
        "summary_csv": DATA_DIR / "experiment_94_nonpos_rank_consensus_v2_summary.csv",
        "stdout_log": DATA_DIR / "experiment_94_nonpos_rank_consensus_v2_stdout.log",
    },
    {
        "id": "experiment_95_topk_review_tier",
        "script": ROOT / "run_experiment_95_topk_review_tier.py",
        "detail_csv": DATA_DIR / "experiment_95_topk_review_tier_results.csv",
        "summary_csv": DATA_DIR / "experiment_95_topk_review_tier_summary.csv",
        "stdout_log": DATA_DIR / "experiment_95_topk_review_tier_stdout.log",
    },
    {
        "id": "experiment_96_review_tier_operational_workflow",
        "script": ROOT / "run_experiment_96_review_tier_operational_workflow.py",
        "detail_csv": DATA_DIR / "experiment_96_review_tier_operational_workflow_results.csv",
        "summary_csv": DATA_DIR / "experiment_96_review_tier_operational_workflow_summary.csv",
        "stdout_log": DATA_DIR / "experiment_96_review_tier_operational_workflow_stdout.log",
    },
    {
        "id": "experiment_94b_corrected_nonpos_rank_consensus",
        "script": ROOT / "run_experiment_94b_corrected_nonpos_rank_consensus.py",
        "detail_csv": DATA_DIR / "experiment_94b_corrected_nonpos_rank_consensus_results.csv",
        "summary_csv": DATA_DIR / "experiment_94b_corrected_nonpos_rank_consensus_summary.csv",
        "stdout_log": DATA_DIR / "experiment_94b_corrected_nonpos_rank_consensus_stdout.log",
    },
    {
        "id": "experiment_97_zero_f1_feature_need_diagnosis",
        "script": ROOT / "run_experiment_97_zero_f1_feature_need_diagnosis.py",
        "detail_csv": DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_results.csv",
        "summary_csv": DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_summary.csv",
        "stdout_log": DATA_DIR / "experiment_97_zero_f1_feature_need_diagnosis_stdout.log",
    },
    {
        "id": "experiment_98_tiny_train_normal_pooling",
        "script": ROOT / "run_experiment_98_tiny_train_normal_pooling.py",
        "detail_csv": DATA_DIR / "experiment_98_tiny_train_normal_pooling_results.csv",
        "summary_csv": DATA_DIR / "experiment_98_tiny_train_normal_pooling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_98_tiny_train_normal_pooling_stdout.log",
    },
    {
        "id": "experiment_98b_train_only_tiny_pooling",
        "script": ROOT / "run_experiment_98b_train_only_tiny_pooling.py",
        "detail_csv": DATA_DIR / "experiment_98b_train_only_tiny_pooling_results.csv",
        "summary_csv": DATA_DIR / "experiment_98b_train_only_tiny_pooling_summary.csv",
        "stdout_log": DATA_DIR / "experiment_98b_train_only_tiny_pooling_stdout.log",
    },
    {
        "id": "experiment_99_spectral_derivative_feature_score",
        "script": ROOT / "run_experiment_99_spectral_derivative_feature_score.py",
        "detail_csv": DATA_DIR / "experiment_99_spectral_derivative_feature_score_results.csv",
        "summary_csv": DATA_DIR / "experiment_99_spectral_derivative_feature_score_summary.csv",
        "stdout_log": DATA_DIR / "experiment_99_spectral_derivative_feature_score_stdout.log",
    },
    {
        "id": "experiment_100_spectral_review_and_guard",
        "script": ROOT / "run_experiment_100_spectral_review_and_guard.py",
        "detail_csv": DATA_DIR / "experiment_100_spectral_review_and_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_100_spectral_review_and_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_100_spectral_review_and_guard_stdout.log",
    },
    {
        "id": "experiment_101_shapelet_normal_prototype",
        "script": ROOT / "run_experiment_101_shapelet_normal_prototype.py",
        "detail_csv": DATA_DIR / "experiment_101_shapelet_normal_prototype_results.csv",
        "summary_csv": DATA_DIR / "experiment_101_shapelet_normal_prototype_summary.csv",
        "stdout_log": DATA_DIR / "experiment_101_shapelet_normal_prototype_stdout.log",
    },
    {
        "id": "experiment_102_feature_source_selector",
        "script": ROOT / "run_experiment_102_feature_source_selector.py",
        "detail_csv": DATA_DIR / "experiment_102_feature_source_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_102_feature_source_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_102_feature_source_selector_stdout.log",
    },
    {
        "id": "experiment_103_higher_dim_review_sources",
        "script": ROOT / "run_experiment_103_higher_dim_review_sources.py",
        "detail_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_results.csv",
        "summary_csv": DATA_DIR / "experiment_103_higher_dim_review_sources_summary.csv",
        "stdout_log": DATA_DIR / "experiment_103_higher_dim_review_sources_stdout.log",
    },
    {
        "id": "experiment_104_score_dimensionality_sweep",
        "script": ROOT / "run_experiment_104_score_dimensionality_sweep.py",
        "detail_csv": DATA_DIR / "experiment_104_score_dimensionality_sweep_results.csv",
        "summary_csv": DATA_DIR / "experiment_104_score_dimensionality_sweep_summary.csv",
        "stdout_log": DATA_DIR / "experiment_104_score_dimensionality_sweep_stdout.log",
    },
    {
        "id": "experiment_105_score_combination_methods",
        "script": ROOT / "run_experiment_105_score_combination_methods.py",
        "detail_csv": DATA_DIR / "experiment_105_score_combination_methods_results.csv",
        "summary_csv": DATA_DIR / "experiment_105_score_combination_methods_summary.csv",
        "stdout_log": DATA_DIR / "experiment_105_score_combination_methods_stdout.log",
    },
    {
        "id": "experiment_106_gated_score_combo_selector",
        "script": ROOT / "run_experiment_106_gated_score_combo_selector.py",
        "detail_csv": DATA_DIR / "experiment_106_gated_score_combo_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_106_gated_score_combo_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_106_gated_score_combo_selector_stdout.log",
    },
    {
        "id": "experiment_107_exp103_combo_disagreement",
        "script": ROOT / "run_experiment_107_exp103_combo_disagreement.py",
        "detail_csv": DATA_DIR / "experiment_107_exp103_combo_disagreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_107_exp103_combo_disagreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_107_exp103_combo_disagreement_stdout.log",
    },
    {
        "id": "experiment_108_imaging_pretrained_vit_feature_probe",
        "script": ROOT / "run_experiment_108_imaging_pretrained_vit_feature_probe.py",
        "detail_csv": DATA_DIR / "experiment_108_imaging_pretrained_vit_feature_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_108_imaging_pretrained_vit_feature_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_108_imaging_pretrained_vit_feature_probe_stdout.log",
    },
    {
        "id": "experiment_109_vit_compression_alternatives",
        "script": ROOT / "run_experiment_109_vit_compression_alternatives.py",
        "detail_csv": DATA_DIR / "experiment_109_vit_compression_alternatives_results.csv",
        "summary_csv": DATA_DIR / "experiment_109_vit_compression_alternatives_summary.csv",
        "stdout_log": DATA_DIR / "experiment_109_vit_compression_alternatives_stdout.log",
    },
    {
        "id": "experiment_110_exp84_score_backend_probe",
        "script": ROOT / "run_experiment_110_exp84_score_backend_probe.py",
        "detail_csv": DATA_DIR / "experiment_110_exp84_score_backend_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_110_exp84_score_backend_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_110_exp84_score_backend_probe_stdout.log",
    },
    {
        "id": "experiment_111_vit_fast_compression_probe",
        "script": ROOT / "run_experiment_111_vit_fast_compression_probe.py",
        "detail_csv": DATA_DIR / "experiment_111_vit_fast_compression_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_111_vit_fast_compression_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_111_vit_fast_compression_probe_stdout.log",
    },
    {
        "id": "experiment_111b_vit_manifold_compression_probe",
        "script": ROOT / "run_experiment_111b_vit_manifold_compression_probe.py",
        "detail_csv": DATA_DIR / "experiment_111b_vit_manifold_compression_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_111b_vit_manifold_compression_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_111b_vit_manifold_compression_probe_stdout.log",
    },
    {
        "id": "experiment_112_parametric_umap_oof_probe",
        "script": ROOT / "run_experiment_112_parametric_umap_oof_probe.py",
        "detail_csv": DATA_DIR / "experiment_112_parametric_umap_oof_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_112_parametric_umap_oof_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_112_parametric_umap_oof_probe_stdout.log",
        "python_bin": ROOT / ".venv-parametric-umap" / "bin" / "python",
    },
    {
        "id": "experiment_113_train_normal_conformal_fusion",
        "script": ROOT / "run_experiment_113_train_normal_conformal_fusion.py",
        "detail_csv": DATA_DIR / "experiment_113_train_normal_conformal_fusion_results.csv",
        "summary_csv": DATA_DIR / "experiment_113_train_normal_conformal_fusion_summary.csv",
        "stdout_log": DATA_DIR / "experiment_113_train_normal_conformal_fusion_stdout.log",
    },
    {
        "id": "experiment_114_pseudo_anomaly_score_probe",
        "script": ROOT / "run_experiment_114_pseudo_anomaly_score_probe.py",
        "detail_csv": DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_114_pseudo_anomaly_score_probe_stdout.log",
    },
    {
        "id": "experiment_115_local_normal_state_score",
        "script": ROOT / "run_experiment_115_local_normal_state_score.py",
        "detail_csv": DATA_DIR / "experiment_115_local_normal_state_score_results.csv",
        "summary_csv": DATA_DIR / "experiment_115_local_normal_state_score_summary.csv",
        "stdout_log": DATA_DIR / "experiment_115_local_normal_state_score_stdout.log",
    },
    {
        "id": "experiment_116_train_only_reliability_reranker",
        "script": ROOT / "run_experiment_116_train_only_reliability_reranker.py",
        "detail_csv": DATA_DIR / "experiment_116_train_only_reliability_reranker_results.csv",
        "summary_csv": DATA_DIR / "experiment_116_train_only_reliability_reranker_summary.csv",
        "stdout_log": DATA_DIR / "experiment_116_train_only_reliability_reranker_stdout.log",
    },
    {
        "id": "experiment_117_train_only_candidate_budget",
        "script": ROOT / "run_experiment_117_train_only_candidate_budget.py",
        "detail_csv": DATA_DIR / "experiment_117_train_only_candidate_budget_results.csv",
        "summary_csv": DATA_DIR / "experiment_117_train_only_candidate_budget_summary.csv",
        "stdout_log": DATA_DIR / "experiment_117_train_only_candidate_budget_stdout.log",
    },
    {
        "id": "experiment_118_rocket512_knn3_exp93_source_probe",
        "script": ROOT / "run_experiment_118_rocket512_knn3_exp93_source_probe.py",
        "detail_csv": DATA_DIR / "experiment_118_rocket512_knn3_exp93_source_probe_results.csv",
        "summary_csv": DATA_DIR / "experiment_118_rocket512_knn3_exp93_source_probe_summary.csv",
        "stdout_log": DATA_DIR / "experiment_118_rocket512_knn3_exp93_source_probe_stdout.log",
    },
    {
        "id": "experiment_119a_exp93_rank_order_validation",
        "script": ROOT / "run_experiment_119a_exp93_rank_order_validation.py",
        "detail_csv": DATA_DIR / "experiment_119a_exp93_rank_order_validation_results.csv",
        "summary_csv": DATA_DIR / "experiment_119a_exp93_rank_order_validation_summary.csv",
        "stdout_log": DATA_DIR / "experiment_119a_exp93_rank_order_validation_stdout.log",
    },
    {
        "id": "experiment_119b_rocket256_512_validated_rank_compare",
        "script": ROOT / "run_experiment_119b_rocket256_512_validated_rank_compare.py",
        "detail_csv": DATA_DIR / "experiment_119b_rocket256_512_validated_rank_compare_results.csv",
        "summary_csv": DATA_DIR / "experiment_119b_rocket256_512_validated_rank_compare_summary.csv",
        "stdout_log": DATA_DIR / "experiment_119b_rocket256_512_validated_rank_compare_stdout.log",
    },
    {
        "id": "experiment_120_exp93_reference_rebaseline",
        "script": ROOT / "run_experiment_120_exp93_reference_rebaseline.py",
        "detail_csv": DATA_DIR / "experiment_120_exp93_reference_rebaseline_results.csv",
        "summary_csv": DATA_DIR / "experiment_120_exp93_reference_rebaseline_summary.csv",
        "stdout_log": DATA_DIR / "experiment_120_exp93_reference_rebaseline_stdout.log",
    },
    {
        "id": "experiment_121_exp95_validated_rank_review",
        "script": ROOT / "run_experiment_121_exp95_validated_rank_review.py",
        "detail_csv": DATA_DIR / "experiment_121_exp95_validated_rank_review_results.csv",
        "summary_csv": DATA_DIR / "experiment_121_exp95_validated_rank_review_summary.csv",
        "stdout_log": DATA_DIR / "experiment_121_exp95_validated_rank_review_stdout.log",
    },
    {
        "id": "experiment_122_exp96_validated_rank_workflow",
        "script": ROOT / "run_experiment_122_exp96_validated_rank_workflow.py",
        "detail_csv": DATA_DIR / "experiment_122_exp96_validated_rank_workflow_results.csv",
        "summary_csv": DATA_DIR / "experiment_122_exp96_validated_rank_workflow_summary.csv",
        "stdout_log": DATA_DIR / "experiment_122_exp96_validated_rank_workflow_stdout.log",
    },
    {
        "id": "experiment_123_exp103_validated_rank_review_sources",
        "script": ROOT / "run_experiment_123_exp103_validated_rank_review_sources.py",
        "detail_csv": DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_results.csv",
        "summary_csv": DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_summary.csv",
        "stdout_log": DATA_DIR / "experiment_123_exp103_validated_rank_review_sources_stdout.log",
    },
    {
        "id": "experiment_124_exp104_validated_rank_dimensions",
        "script": ROOT / "run_experiment_124_exp104_validated_rank_dimensions.py",
        "detail_csv": DATA_DIR / "experiment_124_exp104_validated_rank_dimensions_results.csv",
        "summary_csv": DATA_DIR / "experiment_124_exp104_validated_rank_dimensions_summary.csv",
        "stdout_log": DATA_DIR / "experiment_124_exp104_validated_rank_dimensions_stdout.log",
    },
    {
        "id": "experiment_125_exp105_validated_rank_combinations",
        "script": ROOT / "run_experiment_125_exp105_validated_rank_combinations.py",
        "detail_csv": DATA_DIR / "experiment_125_exp105_validated_rank_combinations_results.csv",
        "summary_csv": DATA_DIR / "experiment_125_exp105_validated_rank_combinations_summary.csv",
        "stdout_log": DATA_DIR / "experiment_125_exp105_validated_rank_combinations_stdout.log",
    },
    {
        "id": "experiment_126_exp106_validated_rank_gated_combo",
        "script": ROOT / "run_experiment_126_exp106_validated_rank_gated_combo.py",
        "detail_csv": DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_results.csv",
        "summary_csv": DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_summary.csv",
        "stdout_log": DATA_DIR / "experiment_126_exp106_validated_rank_gated_combo_stdout.log",
    },
    {
        "id": "experiment_127_exp107_validated_rank_disagreement",
        "script": ROOT / "run_experiment_127_exp107_validated_rank_disagreement.py",
        "detail_csv": DATA_DIR / "experiment_127_exp107_validated_rank_disagreement_results.csv",
        "summary_csv": DATA_DIR / "experiment_127_exp107_validated_rank_disagreement_summary.csv",
        "stdout_log": DATA_DIR / "experiment_127_exp107_validated_rank_disagreement_stdout.log",
    },
    {
        "id": "experiment_128_rocket512_only_review_selector",
        "script": ROOT / "run_experiment_128_rocket512_only_review_selector.py",
        "detail_csv": DATA_DIR / "experiment_128_rocket512_only_review_selector_results.csv",
        "summary_csv": DATA_DIR / "experiment_128_rocket512_only_review_selector_summary.csv",
        "stdout_log": DATA_DIR / "experiment_128_rocket512_only_review_selector_stdout.log",
    },
    {
        "id": "experiment_129_rocket512_base_selector_rebuild",
        "script": ROOT / "run_experiment_129_rocket512_base_selector_rebuild.py",
        "detail_csv": DATA_DIR / "experiment_129_rocket512_base_selector_rebuild_results.csv",
        "summary_csv": DATA_DIR / "experiment_129_rocket512_base_selector_rebuild_summary.csv",
        "stdout_log": DATA_DIR / "experiment_129_rocket512_base_selector_rebuild_stdout.log",
    },
    {
        "id": "experiment_130_rocket512_two_block_knn",
        "script": ROOT / "run_experiment_130_rocket512_two_block_knn.py",
        "detail_csv": DATA_DIR / "experiment_130_rocket512_two_block_knn_results.csv",
        "summary_csv": DATA_DIR / "experiment_130_rocket512_two_block_knn_summary.csv",
        "stdout_log": DATA_DIR / "experiment_130_rocket512_two_block_knn_stdout.log",
    },
    {
        "id": "experiment_131_rocket_block_b_calibration",
        "script": ROOT / "run_experiment_131_rocket_block_b_calibration.py",
        "detail_csv": DATA_DIR / "experiment_131_rocket_block_b_calibration_results.csv",
        "summary_csv": DATA_DIR / "experiment_131_rocket_block_b_calibration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_131_rocket_block_b_calibration_stdout.log",
    },
    {
        "id": "experiment_132_block_b_review_integration",
        "script": ROOT / "run_experiment_132_block_b_review_integration.py",
        "detail_csv": DATA_DIR / "experiment_132_block_b_review_integration_results.csv",
        "summary_csv": DATA_DIR / "experiment_132_block_b_review_integration_summary.csv",
        "stdout_log": DATA_DIR / "experiment_132_block_b_review_integration_stdout.log",
    },
    {
        "id": "experiment_133_block_b_confidence_tiers",
        "script": ROOT / "run_experiment_133_block_b_confidence_tiers.py",
        "detail_csv": DATA_DIR / "experiment_133_block_b_confidence_tiers_results.csv",
        "summary_csv": DATA_DIR / "experiment_133_block_b_confidence_tiers_summary.csv",
        "stdout_log": DATA_DIR / "experiment_133_block_b_confidence_tiers_stdout.log",
    },
    {
        "id": "experiment_134_block_b_review_tail_guard",
        "script": ROOT / "run_experiment_134_block_b_review_tail_guard.py",
        "detail_csv": DATA_DIR / "experiment_134_block_b_review_tail_guard_results.csv",
        "summary_csv": DATA_DIR / "experiment_134_block_b_review_tail_guard_summary.csv",
        "stdout_log": DATA_DIR / "experiment_134_block_b_review_tail_guard_stdout.log",
    },
    {
        "id": "experiment_135_block_c_review_confirmation",
        "script": ROOT / "run_experiment_135_block_c_review_confirmation.py",
        "detail_csv": DATA_DIR / "experiment_135_block_c_review_confirmation_results.csv",
        "summary_csv": DATA_DIR / "experiment_135_block_c_review_confirmation_summary.csv",
        "stdout_log": DATA_DIR / "experiment_135_block_c_review_confirmation_stdout.log",
    },
    {
        "id": "experiment_136_family_holdout_review_audit",
        "script": ROOT / "run_experiment_136_family_holdout_review_audit.py",
        "detail_csv": DATA_DIR / "experiment_136_family_holdout_review_audit_results.csv",
        "summary_csv": DATA_DIR / "experiment_136_family_holdout_review_audit_summary.csv",
        "stdout_log": DATA_DIR / "experiment_136_family_holdout_review_audit_stdout.log",
    },
    {
        "id": "experiment_137_operational_triage",
        "script": ROOT / "run_experiment_137_operational_triage.py",
        "detail_csv": DATA_DIR / "experiment_137_operational_triage_results.csv",
        "summary_csv": DATA_DIR / "experiment_137_operational_triage_summary.csv",
        "stdout_log": DATA_DIR / "experiment_137_operational_triage_stdout.log",
    },
    {
        "id": "experiment_153_157_train_only_improvements",
        "label": "Exp153-157 · TRAIN-only calibration and coverage sensitivities",
        "script": ROOT / "experiments/exp153_157/run_experiment_153_157_pipeline.py",
        "detail_csv": ROOT / "outputs/exp153_157_train_only_improvements/23_exp157_results.csv",
        "summary_csv": ROOT / "outputs/exp153_157_train_only_improvements/24_exp157_summary.csv",
        "stdout_log": ROOT / "outputs/exp153_157_train_only_improvements/full_pipeline_queue.log",
        "expected_datasets": 1117,
    },
    {
        "id": "experiment_165_small_sample_calibration",
        "label": "Exp165 · small-sample TRAIN-only similarity calibration audit",
        "script": ROOT / "experiments/exp165/run_experiment_165_small_sample_calibration.py",
        "detail_csv": ROOT / "outputs/exp165_small_sample_calibration/03_dataset_alpha_summary.csv",
        "summary_csv": ROOT / "outputs/exp165_small_sample_calibration/04_global_summary.csv",
        "stdout_log": ROOT / "outputs/exp165_small_sample_calibration/full_run.log",
        "expected_datasets": 316,
    },
    {
        "id": "experiment_166_small_sample_anomaly_coverage",
        "label": "Exp166 · small-sample similarity calibration anomaly coverage",
        "script": ROOT / "experiments/exp166/run_experiment_166_small_sample_anomaly_coverage.py",
        "detail_csv": ROOT / "outputs/exp166_small_sample_anomaly_coverage/04_dataset_metrics.csv",
        "summary_csv": ROOT / "outputs/exp166_small_sample_anomaly_coverage/05_global_summary.csv",
        "stdout_log": ROOT / "outputs/exp166_small_sample_anomaly_coverage/full_run.log",
        "expected_datasets": 316,
    },
    {
        "id": "experiment_167_score_calibration_parity_audit",
        "label": "Exp167 · fold-matched score calibration parity audit",
        "script": ROOT / "experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py",
        "detail_csv": ROOT / "outputs/exp167_168_score_calibration_parity/02_parity_audit.csv",
        "summary_csv": ROOT / "outputs/exp167_168_score_calibration_parity/03_fold_score_summary.csv",
        "stdout_log": ROOT / "outputs/exp167_168_score_calibration_parity/full_run.log",
        "expected_datasets": 316,
    },
    {
        "id": "experiment_168_fold_matched_parity_evaluation",
        "label": "Exp168 · fold-matched parity review sensitivity",
        "script": ROOT / "experiments/exp167_168/run_experiment_167_168_score_calibration_parity.py",
        "detail_csv": ROOT / "outputs/exp167_168_score_calibration_parity/06_exp168_dataset_metrics.csv",
        "summary_csv": ROOT / "outputs/exp167_168_score_calibration_parity/07_exp168_global_summary.csv",
        "stdout_log": ROOT / "outputs/exp167_168_score_calibration_parity/full_run.log",
        "expected_datasets": 316,
    },
    {
        "id": "experiment_169_oob_pooled_calibration_audit",
        "label": "Exp169 · OOB pooled calibration audit",
        "script": ROOT / "experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py",
        "detail_csv": ROOT / "outputs/exp169_170_oob_pooled_calibration/04_exp169_dataset_audit.csv",
        "summary_csv": ROOT / "outputs/exp169_170_oob_pooled_calibration/05_exp169_global_summary.csv",
        "stdout_log": ROOT / "outputs/exp169_170_oob_pooled_calibration/full_run.log",
        "expected_datasets": 316,
    },
    {
        "id": "experiment_170_oob_pooled_review_evaluation",
        "label": "Exp170 · OOB pooled review evaluation",
        "script": ROOT / "experiments/exp169_170/run_experiment_169_170_oob_pooled_calibration.py",
        "detail_csv": ROOT / "outputs/exp169_170_oob_pooled_calibration/08_exp170_dataset_metrics.csv",
        "summary_csv": ROOT / "outputs/exp169_170_oob_pooled_calibration/09_exp170_global_summary.csv",
        "stdout_log": ROOT / "outputs/exp169_170_oob_pooled_calibration/full_run.log",
        "expected_datasets": 316,
    },
    {
        "id": "experiment_171_train_normal_risk_policy_freeze",
        "label": "Exp171 · TRAIN-normal risk policy freeze",
        "script": ROOT / "experiments/exp171_172/run_experiment_171_172_train_risk_policy.py",
        "args": ["--phase", "audit"],
        "detail_csv": ROOT / "outputs/exp171_172_train_risk_policy/02_exp171_policy_candidates.csv",
        "summary_csv": ROOT / "outputs/exp171_172_train_risk_policy/02_exp171_policy_candidates.csv",
        "stdout_log": ROOT / "outputs/exp171_172_train_risk_policy/full_run.log",
        "expected_datasets": 316,
    },
    {
        "id": "experiment_172_frozen_train_risk_policy_review",
        "label": "Exp172 · frozen TRAIN-risk review policy",
        "script": ROOT / "experiments/exp171_172/run_experiment_171_172_train_risk_policy.py",
        "args": ["--phase", "predict-evaluate"],
        "detail_csv": ROOT / "outputs/exp171_172_train_risk_policy/06_exp172_dataset_metrics.csv",
        "summary_csv": ROOT / "outputs/exp171_172_train_risk_policy/07_exp172_global_summary.csv",
        "stdout_log": ROOT / "outputs/exp171_172_train_risk_policy/full_run.log",
        "expected_datasets": 316,
    },
    {
        "id": "experiment_173_balanced_train_selector_freeze",
        "label": "Exp173 · balanced TRAIN-only selector freeze",
        "script": ROOT / "experiments/exp173_174/run_experiment_173_174_balanced_train_selector.py",
        "args": ["--phase", "audit"],
        "detail_csv": ROOT / "outputs/exp173_174_balanced_train_selector/02_exp173_policy_candidates.csv",
        "summary_csv": ROOT / "outputs/exp173_174_balanced_train_selector/02_exp173_policy_candidates.csv",
        "stdout_log": ROOT / "outputs/exp173_174_balanced_train_selector/full_run.log",
        "expected_datasets": 316,
    },
    {
        "id": "experiment_174_balanced_frozen_review",
        "label": "Exp174 · balanced frozen review policy",
        "script": ROOT / "experiments/exp173_174/run_experiment_173_174_balanced_train_selector.py",
        "args": ["--phase", "predict-evaluate"],
        "detail_csv": ROOT / "outputs/exp173_174_balanced_train_selector/06_exp174_dataset_metrics.csv",
        "summary_csv": ROOT / "outputs/exp173_174_balanced_train_selector/07_exp174_global_summary.csv",
        "stdout_log": ROOT / "outputs/exp173_174_balanced_train_selector/full_run.log",
        "expected_datasets": 316,
    },
    *[
        {
            "id": exp_id,
            "label": label,
            "script": ROOT / "experiments/exp177_185/run_experiment_177_185_unattended.py",
            "args": ["--stage", stage, "--workers", "7"] if stage in {"215", "216", "218"} else ["--stage", stage],
            "detail_csv": DATA_DIR / f"{exp_id}_results.csv",
            "summary_csv": DATA_DIR / f"{exp_id}_summary.csv",
            "stdout_log": DATA_DIR / "exp177_185_unattended" / f"exp{stage}_stdout.log",
            "expected_datasets": expected,
        }
        for stage, exp_id, label, expected in (
            ("177", "experiment_177_train_oof_review_policy_freeze", "Exp177 · TRAIN OOF review policy freeze", 639),
            ("178", "experiment_178_frozen_policy_review_prediction", "Exp178 · frozen policy review prediction", 639),
            ("179", "experiment_179_canonical_multiscale_feature_cache", "Exp179 · canonical multiscale feature cache", 639),
            ("180", "experiment_180_canonical_score_backend_prediction", "Exp180 · canonical score backend prediction", 639),
            ("181", "experiment_181_existing_new_source_fusion_prediction", "Exp181 · existing/new source fusion prediction", 639),
            ("182", "experiment_182_five_hour_checkpoint_evaluation", "Exp182 · five-hour checkpoint evaluation", 639),
            ("183", "experiment_183_wavelet_multiscale_feature_prediction", "Exp183 · wavelet multiscale feature prediction", 639),
            ("184", "experiment_184_extension_source_fusion_prediction", "Exp184 · extension source fusion prediction", 639),
            ("185", "experiment_185_final_independent_reaggregation", "Exp185 · final independent reaggregation", 639),
        )
    ],
    *[
        {
            "id": exp_id,
            "label": label,
            "script": ROOT / "experiments/exp186_200/run_experiment_186_200.py",
            "args": ["--stage", stage],
            "detail_csv": DATA_DIR / f"{exp_id}_results.csv",
            "summary_csv": DATA_DIR / f"{exp_id}_summary.csv",
            "stdout_log": DATA_DIR / "exp186_200_unattended" / f"exp{stage}_stdout.log",
            "expected_datasets": expected,
        }
        for stage, exp_id, label, expected in (
            ("186", "experiment_186_conservative_canonical_knn_agreement", "Exp186 · Conservative plus Canonical KNN agreement", 639),
            ("187", "experiment_187_conservative_dual_canonical_agreement", "Exp187 · Conservative plus dual canonical agreement", 639),
            ("188", "experiment_188_train_oof_stability_guarded_review", "Exp188 · TRAIN-OOF stability guarded review", 639),
            ("189", "experiment_189_train_normal_workload_guarded_review", "Exp189 · TRAIN-normal workload guarded review", 639),
            ("190", "experiment_190_three_tier_review_routing", "Exp190 · Three-tier review routing", 639),
            ("191", "experiment_191_conservative_canonical_shadow_agreement", "Exp191 · Conservative-canonical shadow agreement", 639),
            ("192", "experiment_192_conservative_dual_backend_shadow_agreement", "Exp192 · Conservative-dual-backend shadow agreement", 639),
            ("193", "experiment_193_three_existing_source_canonical_confirmation", "Exp193 · Three-existing-source plus canonical confirmation", 639),
            ("194", "experiment_194_fold_stable_shadow_agreement", "Exp194 · Fold-stable shadow agreement", 639),
            ("195", "experiment_195_zero_normal_candidate_shadow_guard", "Exp195 · Zero-normal-candidate shadow guard", 639),
            ("196", "experiment_196_expanded_train_risk_review", "Exp196 · Expanded TRAIN-risk review", 639),
            ("197", "experiment_197_conservative_canonical_unique_union", "Exp197 · Conservative plus canonical-unique union", 639),
            ("198", "experiment_198_existing_one_canonical_alpha_001", "Exp198 · Existing-one plus canonical alpha 0.01", 639),
            ("199", "experiment_199_expanded_two_of_six_train_workload_guard", "Exp199 · Expanded two-of-six with TRAIN workload guard", 639),
            ("200", "experiment_200_frozen_frontier_reaggregation", "Exp200 · Frozen frontier reaggregation", 639),
        )
    ],
    *[
        {
            "id": exp_id,
            "label": label,
            "script": ROOT / "experiments/exp201_205/run_experiment_201_205.py",
            "args": ["--stage", stage],
            "detail_csv": DATA_DIR / f"{exp_id}_results.csv",
            "summary_csv": DATA_DIR / f"{exp_id}_summary.csv",
            "stdout_log": DATA_DIR / "exp201_205_large_zero_f1_recovery" / f"exp{stage}_stdout.log",
            "expected_datasets": expected,
        }
        for stage, exp_id, label, expected in (
            ("201", "experiment_201_frozen_large_size_baseline", "Exp201 · Frozen large-size baseline", 251),
            ("202", "experiment_202_alternative_3of3_train_risk_rescue", "Exp202 · Alternative 3-of-3 TRAIN-risk rescue", 251),
            ("203", "experiment_203_current_alternative_cross_confirmation", "Exp203 · Current-alternative cross confirmation", 251),
            ("204", "experiment_204_size_stratified_shadow_router", "Exp204 · Size-stratified shadow router", 251),
            ("205", "experiment_205_independent_large_zero_f1_frontier", "Exp205 · Independent large Zero-F1 frontier", 251),
        )
    ],
    *[
        {
            "id": exp_id,
            "label": label,
            "script": ROOT / "experiments/exp206_210/run_experiment_206_210.py",
            "args": ["--stage", stage, "--workers", "7"] if stage == "206" else ["--stage", stage],
            "detail_csv": DATA_DIR / "exp206_210_residual_coverage" / (f"{exp_id}_results.csv" if stage != "206" else "01_score_cache_manifest.csv"),
            "summary_csv": DATA_DIR / "exp206_210_residual_coverage" / (f"{exp_id}_summary.csv" if stage not in ["206", "207"] else "exp207_summary.csv" if stage == "207" else "01_score_cache_manifest.csv"),
            "stdout_log": DATA_DIR / "exp206_210_residual_coverage" / f"exp{stage}_stdout.log",
            "expected_datasets": expected,
        }
        for stage, exp_id, label, expected in (
            ("206", "experiment_206_alternative_source_oof_reconstruction", "Exp206 · Alternative-Source OOF Reconstruction", 251),
            ("207", "experiment_207_train_oof_conditional_novelty_audit", "Exp207 · TRAIN-OOF Conditional Novelty Audit", 251),
            ("208", "experiment_208_continuous_alternative_source_fusion", "Exp208 · Continuous Alternative-Source Fusion", 251),
            ("209", "experiment_209_seven_source_joint_normal_novelty", "Exp209 · Seven-Source Joint Normal Novelty", 251),
            ("210", "experiment_210_independent_frontier_reaggregation", "Exp210 · Independent Frontier Reaggregation", 251),
        )
    ],
    *[
        {
            "id": exp_id,
            "label": label,
            "script": ROOT / "experiments/exp211_214/run_experiment_211_214.py",
            "args": ["--stage", stage, "--workers", "7"] if stage == "212" else ["--stage", stage],
            "detail_csv": DATA_DIR / "exp211_214_calibration_transport" / f"{exp_id}_results.csv",
            "summary_csv": DATA_DIR / "exp211_214_calibration_transport" / f"{exp_id}_summary.csv",
            "stdout_log": DATA_DIR / "exp211_214_calibration_transport" / f"exp{stage}_stdout.log",
            "expected_datasets": expected,
        }
        for stage, exp_id, label, expected in (
            ("211", "experiment_211_calibration_transport_audit", "Exp211 · Calibration Transport Audit", 251),
            ("212", "experiment_212_fold_ensemble_query_scoring", "Exp212 · Fold-Ensemble Query Scoring", 251),
            ("213", "experiment_213_train_only_source_stability_guard", "Exp213 · TRAIN-Only Source Stability Guard", 251),
            ("214", "experiment_214_calibration_transport_reaggregation", "Exp214 · Calibration Transport Reaggregation", 251),
        )
    ],
    *[
        {
            "id": exp_id,
            "label": label,
            "script": ROOT / "experiments/exp215_217/run_experiment_215_217.py",
            "args": ["--stage", stage],
            "detail_csv": DATA_DIR / "exp215_217_shapelet_stability" / f"{exp_id}_results.csv",
            "summary_csv": DATA_DIR / "exp215_217_shapelet_stability" / f"{exp_id}_summary.csv",
            "stdout_log": DATA_DIR / "exp215_217_shapelet_stability" / f"exp{stage}_stdout.log",
            "expected_datasets": expected,
        }
        for stage, exp_id, label, expected in (
            ("215", "experiment_215_fixed_dictionary_transport_control", "Exp215 · Fixed Dictionary Transport Control", 251),
            ("216", "experiment_216_normal_consensus_medoid_transport", "Exp216 · Normal Consensus Medoid Transport", 251),
            ("217", "experiment_217_train_normal_shapelet_risk_audit", "Exp217 · TRAIN-Normal Shapelet Risk Audit", 251),
            ("218", "experiment_218_true_bootstrap_consensus_normal_prototypes", "Exp218 · True Bootstrap-Consensus Normal Prototypes", 251),
        )
    ],
    {
        "id": "experiment_391_fn_zero_reachability_audit",
        "script": (
            ROOT
            / "experiments/exp391_fn_reachability"
            / "run_experiment_391_fn_reachability_audit.py"
        ),
        "detail_csv": (
            DATA_DIR
            / "exp391_fn_zero_reachability_audit"
            / "exp391_dataset_audit.csv"
        ),
        "summary_csv": (
            DATA_DIR
            / "exp391_fn_zero_reachability_audit"
            / "exp391_rule_summary.csv"
        ),
        "stdout_log": (
            DATA_DIR
            / "exp391_fn_zero_reachability_audit"
            / "exp391_stdout.log"
        ),
    },
    {
        "id": "experiment_392_label_guided_score_discovery_audit",
        "script": (
            ROOT
            / "experiments/exp392_label_guided_discovery"
            / "run_experiment_392_label_guided_score_discovery_audit.py"
        ),
        "detail_csv": (
            DATA_DIR
            / "exp392_label_guided_score_discovery_audit"
            / "exp392_individual_source_summary.csv"
        ),
        "summary_csv": (
            DATA_DIR
            / "exp392_label_guided_score_discovery_audit"
            / "exp392_pair_triple_summary.csv"
        ),
        "stdout_log": (
            DATA_DIR
            / "exp392_label_guided_score_discovery_audit"
            / "exp392_stdout.log"
        ),
    },
    {
        "id": "experiment_393_396_normal_process_grammar",
        "label": "Exp393-396 · Normal Process Grammar review evidence",
        "script": (
            ROOT
            / "experiments/exp393_396"
            / "run_experiment_393_396_pipeline.py"
        ),
        "args": [
            "--stage",
            "all",
            "--workers",
            "7",
            "--output-dir",
            str(DATA_DIR / "exp393_396_normal_process_grammar"),
        ],
        "detail_csv": (
            DATA_DIR
            / "exp393_396_normal_process_grammar"
            / "exp396_review_integration.csv"
        ),
        "summary_csv": (
            DATA_DIR
            / "exp393_396_normal_process_grammar"
            / "exp396_summary.csv"
        ),
        "stdout_log": (
            DATA_DIR
            / "exp393_396_normal_process_grammar"
            / "exp393_396_stdout.log"
        ),
        "expected_datasets": 1117,
        "archive_outputs": False,
    },
]

EXP267_273_ROOT = DATA_DIR / "exp267_273_zero_support_fn_recovery"
_EXP267_273_STAGE_ROWS = (
    ("267", "experiment_267_zero_support_fn_diagnostic", "Exp267 · Zero-support FN diagnostic", "exp267/02_zero_support_fn_dataset.csv", "exp267/03_zero_support_fn_summary.csv"),
    ("268", "experiment_268_multiscale_scattering_deviation", "Exp268 · Multiscale scattering deviation", "source_cache/scattering/cache_manifest.csv", "source_cache/scattering/cache_manifest.csv"),
    ("269", "experiment_269_ordered_path_signature_deviation", "Exp269 · Ordered path-signature deviation", "source_cache/signature/cache_manifest.csv", "source_cache/signature/cache_manifest.csv"),
    ("270a", "experiment_270a_rff_normal_forecast", "Exp270a · RFF normal forecasting", "source_cache/rff_forecast/cache_manifest.csv", "source_cache/rff_forecast/cache_manifest.csv"),
    ("270b", "experiment_270b_causal_tcn_normal_forecast", "Exp270b · Causal TCN normal forecasting", "source_cache/tcn_forecast/cache_manifest.csv", "source_cache/tcn_forecast/cache_manifest.csv"),
    ("270c", "experiment_270c_forecast_risk_audit", "Exp270c · Forecast TRAIN-risk audit", "exp270_train_risk_dataset.csv", "exp270_train_risk_summary.csv"),
    ("271", "experiment_271_new_source_qualification", "Exp271 · TRAIN-only source qualification", "exp271_qualification_summary.csv", "exp271_qualification_summary.csv"),
    ("272", "experiment_272_recovery_review_selector", "Exp272 · Recovery review selector", "exp272_recovery_summary.csv", "exp272_recovery_summary.csv"),
    ("273", "experiment_273_autonomous_promotion_audit", "Exp273 · Hypothetical autonomous promotion audit", "exp273_dataset_results.csv", "exp273_dataset_results.csv"),
)
EXPERIMENTS.extend(
    {
        "id": exp_id,
        "label": label,
        "script": ROOT / "scratch/run_exp267_273_zero_support_fn_recovery.py",
        "args": ["--stage", stage, "--workers", "7", "--output-root", str(EXP267_273_ROOT)],
        "detail_csv": EXP267_273_ROOT / detail,
        "summary_csv": EXP267_273_ROOT / summary,
        "stdout_log": EXP267_273_ROOT / "logs" / f"{exp_id}.log",
        "expected_datasets": 1117,
        "archive_outputs": False,
    }
    for stage, exp_id, label, detail, summary in _EXP267_273_STAGE_ROWS
)

DEFAULT_QUEUE = ["experiment_26_rocket"]
EXP267_273_BATCH = tuple(row[1] for row in _EXP267_273_STAGE_ROWS)
EXP274_275_ROOT = DATA_DIR / "exp274_275_large_fn_recovery"
_EXP274_275_STAGE_ROWS = (
    ("274", "experiment_274_large_train_multisource_fn_recovery", "Exp274 · Large-TRAIN multisource FN recovery", "exp274_summary.csv", "exp274_summary.csv"),
    ("275", "experiment_275_recovery_shadow_precision_guard", "Exp275 · Recovery Shadow precision guard", "exp275_dataset_results.csv", "exp275_summary.csv"),
)
EXPERIMENTS.extend(
    {
        "id": exp_id,
        "label": label,
        "script": ROOT / "scratch/run_exp274_275_large_fn_recovery.py",
        "args": ["--stage", stage, "--output-root", str(EXP274_275_ROOT)],
        "detail_csv": EXP274_275_ROOT / detail,
        "summary_csv": EXP274_275_ROOT / summary,
        "stdout_log": EXP274_275_ROOT / "logs" / f"{exp_id}.log",
        "expected_datasets": 1117,
        "archive_outputs": False,
    }
    for stage, exp_id, label, detail, summary in _EXP274_275_STAGE_ROWS
)
EXP274_275_BATCH = tuple(row[1] for row in _EXP274_275_STAGE_ROWS)
EXP276_277_ROOT = DATA_DIR / "exp276_277_shadow_confidence"
_EXP276_277_STAGE_ROWS = (
    ("276", "experiment_276_recovery_shadow_confidence_freeze", "Exp276 · Recovery Shadow confidence freeze", "exp276_summary.csv", "exp276_summary.csv"),
    ("277", "experiment_277_recovery_shadow_confidence_evaluation", "Exp277 · Recovery Shadow confidence evaluation", "exp277_dataset_results.csv", "exp277_summary.csv"),
)
EXPERIMENTS.extend(
    {
        "id": exp_id,
        "label": label,
        "script": ROOT / "scratch/run_exp276_277_shadow_confidence.py",
        "args": ["--stage", stage, "--output-root", str(EXP276_277_ROOT)],
        "detail_csv": EXP276_277_ROOT / detail,
        "summary_csv": EXP276_277_ROOT / summary,
        "stdout_log": EXP276_277_ROOT / "logs" / f"{exp_id}.log",
        "expected_datasets": 1117,
        "archive_outputs": False,
    }
    for stage, exp_id, label, detail, summary in _EXP276_277_STAGE_ROWS
)
EXP276_277_BATCH = tuple(row[1] for row in _EXP276_277_STAGE_ROWS)
EXP278_281_ROOT = DATA_DIR / "exp278_281_train_only_recovery"
_EXP278_281_STAGE_ROWS = (
    ("278", "experiment_278_train_risk_resolution_audit", "Exp278 · TRAIN risk-resolution audit", "exp278_risk_resolution_dataset.csv", "exp278_summary.csv"),
    ("279", "experiment_279_train_calibration_stability", "Exp279 · TRAIN calibration stability", "exp279_candidate_stability.csv", "exp279_summary.csv"),
    ("280", "experiment_280_event_merge_applicability_audit", "Exp280 · Event-merge applicability audit", "exp280_event_applicability.csv", "exp280_summary.csv"),
    ("281", "experiment_281_train_only_two_stage_verifier", "Exp281 · TRAIN-only two-stage verifier", "exp281_dataset_results.csv", "exp281_summary.csv"),
)
EXPERIMENTS.extend(
    {
        "id": exp_id, "label": label,
        "script": ROOT / "scratch/run_exp278_281_train_only_recovery.py",
        "args": ["--stage", stage, "--output-root", str(EXP278_281_ROOT)],
        "detail_csv": EXP278_281_ROOT / detail, "summary_csv": EXP278_281_ROOT / summary,
        "stdout_log": EXP278_281_ROOT / "logs" / f"{exp_id}.log",
        "expected_datasets": 1117, "archive_outputs": False,
    }
    for stage, exp_id, label, detail, summary in _EXP278_281_STAGE_ROWS
)
EXP278_281_BATCH = tuple(row[1] for row in _EXP278_281_STAGE_ROWS)
EXP282_284_ROOT = DATA_DIR / "exp282_284_prototype_distance_verifier"
_EXP282_284_STAGE_ROWS = (
    ("282", "experiment_282_prototype_distance_cache", "Exp282 · Prototype distance cache", "exp282_summary.csv", "exp282_summary.csv"),
    ("283", "experiment_283_prototype_distance_prediction", "Exp283 · Prototype distance prediction", "exp283_summary.csv", "exp283_summary.csv"),
    ("284", "experiment_284_prototype_distance_evaluation", "Exp284 · Prototype distance evaluation", "exp284_dataset_results.csv", "exp284_summary.csv"),
)
EXPERIMENTS.extend({
    "id": exp_id, "label": label, "script": ROOT / "scratch/run_exp282_284_prototype_distance_verifier.py",
    "args": ["--stage", stage, "--output-root", str(EXP282_284_ROOT)],
    "detail_csv": EXP282_284_ROOT / detail, "summary_csv": EXP282_284_ROOT / summary,
    "stdout_log": EXP282_284_ROOT / "logs" / f"{exp_id}.log", "expected_datasets": 1117, "archive_outputs": False,
} for stage, exp_id, label, detail, summary in _EXP282_284_STAGE_ROWS)
EXP282_284_BATCH = tuple(row[1] for row in _EXP282_284_STAGE_ROWS)
EXP285_287_ROOT = DATA_DIR / "exp285_287_synthetic_shape_verifier"
_EXP285_287_STAGE_ROWS = (
    ("285", "experiment_285_synthetic_shape_generation_audit", "Exp285 · Synthetic shape generation audit", "exp285_generation_audit.csv", "exp285_summary.csv"),
    ("286", "experiment_286_synthetic_shape_prediction", "Exp286 · Synthetic shape prediction", "exp286_summary.csv", "exp286_summary.csv"),
    ("287", "experiment_287_synthetic_shape_evaluation", "Exp287 · Synthetic shape evaluation", "exp287_dataset_results.csv", "exp287_summary.csv"),
)
EXPERIMENTS.extend({"id": exp_id, "label": label, "script": ROOT / "scratch/run_exp285_287_synthetic_shape_verifier.py",
    "args": ["--stage", stage, "--output-root", str(EXP285_287_ROOT)], "detail_csv": EXP285_287_ROOT / detail,
    "summary_csv": EXP285_287_ROOT / summary, "stdout_log": EXP285_287_ROOT / "logs" / f"{exp_id}.log",
    "expected_datasets": 1117, "archive_outputs": False} for stage, exp_id, label, detail, summary in _EXP285_287_STAGE_ROWS)
EXP285_287_BATCH = tuple(row[1] for row in _EXP285_287_STAGE_ROWS)
EXP288_ROOT = DATA_DIR / "exp288_recovery_priority_counterexample"
EXPERIMENTS.append({"id": "experiment_288_recovery_priority_counterexample_audit", "label": "Exp288 · Recovery Priority FP counterexample audit",
    "script": ROOT / "scratch/run_exp288_recovery_priority_counterexample.py", "args": ["--output-root", str(EXP288_ROOT)],
    "detail_csv": EXP288_ROOT / "exp288_counterexample_results.csv", "summary_csv": EXP288_ROOT / "exp288_summary.csv",
    "stdout_log": EXP288_ROOT / "logs" / "experiment_288_recovery_priority_counterexample_audit.log", "expected_datasets": 1117, "archive_outputs": False})
EXP289_ROOT = DATA_DIR / "exp289_recovery_priority_label_consistency_audit"
EXPERIMENTS.append({"id": "experiment_289_recovery_priority_label_consistency_audit", "label": "Exp289 · Recovery Priority label consistency audit",
    "script": ROOT / "scratch/run_exp289_label_consistency_audit.py", "args": ["--output-root", str(EXP289_ROOT)],
    "detail_csv": EXP289_ROOT / "exp289_candidate_label_consistency.csv", "summary_csv": EXP289_ROOT / "exp289_tier_summary.csv",
    "stdout_log": EXP289_ROOT / "logs" / "experiment_289_recovery_priority_label_consistency_audit.log", "expected_datasets": 1117, "archive_outputs": False})
EXP290_ROOT = DATA_DIR / "exp290_train_raw_exact_normal_guard"
EXPERIMENTS.append({"id": "experiment_290_train_raw_exact_normal_guard", "label": "Exp290 · TRAIN raw exact normal guard",
    "script": ROOT / "scratch/run_exp290_train_raw_exact_normal_guard.py", "args": ["--phase", "all", "--output-root", str(EXP290_ROOT)],
    "detail_csv": EXP290_ROOT / "exp290_results.csv", "summary_csv": EXP290_ROOT / "exp290_summary.csv",
    "stdout_log": EXP290_ROOT / "logs" / "experiment_290_train_raw_exact_normal_guard.log", "expected_datasets": 1117, "archive_outputs": False})
EXP291_ROOT = DATA_DIR / "exp291_final_operational_routing_freeze"
EXPERIMENTS.append({"id": "experiment_291_final_operational_routing_freeze", "label": "Exp291 · Final operational routing freeze",
    "script": ROOT / "scratch/run_exp291_final_operational_routing_freeze.py", "args": ["--phase", "all", "--output-root", str(EXP291_ROOT)],
    "detail_csv": EXP291_ROOT / "exp291_results.csv", "summary_csv": EXP291_ROOT / "exp291_summary.csv",
    "stdout_log": EXP291_ROOT / "logs" / "experiment_291_final_operational_routing_freeze.log", "expected_datasets": 1117, "archive_outputs": False})
EXP292_ROOT = DATA_DIR / "exp292_standard_review_evidence_tiering"
EXPERIMENTS.append({"id": "experiment_292_standard_review_evidence_tiering", "label": "Exp292 · Standard review evidence tiering",
    "script": ROOT / "scratch/run_exp292_standard_review_evidence_tiering.py", "args": ["--phase", "all", "--output-root", str(EXP292_ROOT)],
    "detail_csv": EXP292_ROOT / "exp292_results.csv", "summary_csv": EXP292_ROOT / "exp292_summary.csv",
    "stdout_log": EXP292_ROOT / "logs" / "experiment_292_standard_review_evidence_tiering.log", "expected_datasets": 1117, "archive_outputs": False})
EXP293_ROOT = DATA_DIR / "exp293_train_only_moe_expert_audit"
EXPERIMENTS.append({"id": "experiment_293_train_only_moe_expert_audit", "label": "Exp293 · TRAIN-only MoE expert audit",
    "script": ROOT / "scratch/run_exp293_294_train_only_moe_lite.py", "args": ["--stage", "293", "--output-root", str(EXP293_ROOT)],
    "detail_csv": EXP293_ROOT / "exp293_expert_audit.csv", "summary_csv": EXP293_ROOT / "exp293_summary.csv",
    "stdout_log": EXP293_ROOT / "logs" / "experiment_293_train_only_moe_expert_audit.log", "expected_datasets": 1117, "archive_outputs": False})
EXP294_ROOT = DATA_DIR / "exp294_frozen_sparse_moe_review_router"
EXPERIMENTS.append({"id": "experiment_294_frozen_sparse_moe_review_router", "label": "Exp294 · Frozen sparse MoE review router",
    "script": ROOT / "scratch/run_exp293_294_train_only_moe_lite.py", "args": ["--stage", "294-all", "--output-root", str(EXP294_ROOT)],
    "detail_csv": EXP294_ROOT / "exp294_results.csv", "summary_csv": EXP294_ROOT / "exp294_summary.csv",
    "stdout_log": EXP294_ROOT / "logs" / "experiment_294_frozen_sparse_moe_review_router.log", "expected_datasets": 1117, "archive_outputs": False})
EXP295_ROOT = DATA_DIR / "exp295_moe_hard_alert_confidence_tiering"
EXPERIMENTS.append({"id": "experiment_295_moe_hard_alert_confidence_tiering", "label": "Exp295 · MoE Hard alert confidence tiering",
    "script": ROOT / "scratch/run_exp295_moe_hard_confidence_tiering.py", "args": ["--phase", "all", "--output-root", str(EXP295_ROOT)],
    "detail_csv": EXP295_ROOT / "exp295_results.csv", "summary_csv": EXP295_ROOT / "exp295_summary.csv",
    "stdout_log": EXP295_ROOT / "logs" / "experiment_295_moe_hard_alert_confidence_tiering.log", "expected_datasets": 1117, "archive_outputs": False})
EXP296_ROOT = DATA_DIR / "exp296_cross_gate_hard_confidence_consensus"
EXPERIMENTS.append({"id": "experiment_296_cross_gate_hard_confidence_consensus", "label": "Exp296 · Cross-Gate Hard confidence consensus",
    "script": ROOT / "scratch/run_exp296_cross_gate_hard_confidence_consensus.py", "args": ["--phase", "all", "--output-root", str(EXP296_ROOT)],
    "detail_csv": EXP296_ROOT / "exp296_results.csv", "summary_csv": EXP296_ROOT / "exp296_summary.csv",
    "stdout_log": EXP296_ROOT / "logs" / "experiment_296_cross_gate_hard_confidence_consensus.log", "expected_datasets": 1117, "archive_outputs": False})
EXP297_ROOT = DATA_DIR / "exp297_bootstrap_confidence_stability"
EXPERIMENTS.append({"id": "experiment_297_bootstrap_confidence_stability", "label": "Exp297 · TRAIN-bootstrap confidence stability",
    "script": ROOT / "scratch/run_exp297_bootstrap_confidence_stability.py", "args": ["--phase", "all", "--output-root", str(EXP297_ROOT)],
    "detail_csv": EXP297_ROOT / "exp297_results.csv", "summary_csv": EXP297_ROOT / "exp297_summary.csv",
    "stdout_log": EXP297_ROOT / "logs" / "experiment_297_bootstrap_confidence_stability.log", "expected_datasets": 1117, "archive_outputs": False})
EXP298_ROOT = DATA_DIR / "exp298_exp201_current_hard_rebase"
EXPERIMENTS.append({"id": "experiment_298_exp201_current_hard_rebase", "label": "Exp298 · Exp201 candidates rebased on current Hard",
    "script": ROOT / "scratch/run_exp298_exp201_current_hard_rebase.py", "args": ["--phase", "all", "--output-root", str(EXP298_ROOT)],
    "detail_csv": EXP298_ROOT / "exp298_results.csv", "summary_csv": EXP298_ROOT / "exp298_summary.csv",
    "stdout_log": EXP298_ROOT / "logs" / "experiment_298_exp201_current_hard_rebase.log", "expected_datasets": 251, "archive_outputs": False})
EXP299_ROOT = DATA_DIR / "exp299_multirocket_capacity_consensus"
EXPERIMENTS.append({"id": "experiment_299_multirocket_capacity_consensus", "label": "Exp299 · MultiROCKET capacity consensus",
    "script": ROOT / "scratch/run_exp299_multirocket_capacity_consensus.py", "args": ["--phase", "all", "--output-root", str(EXP299_ROOT)],
    "detail_csv": EXP299_ROOT / "exp299_results.csv", "summary_csv": EXP299_ROOT / "exp299_summary.csv",
    "stdout_log": EXP299_ROOT / "logs" / "experiment_299_multirocket_capacity_consensus.log", "expected_datasets": 251, "archive_outputs": False})
EXP300_ROOT = DATA_DIR / "exp300_independent_evidence_consensus"
EXPERIMENTS.append({"id": "experiment_300_independent_evidence_consensus", "label": "Exp300 · Independent evidence consensus",
    "script": ROOT / "scratch/run_exp300_independent_evidence_consensus.py", "args": ["--phase", "all", "--output-root", str(EXP300_ROOT)],
    "detail_csv": EXP300_ROOT / "exp300_results.csv", "summary_csv": EXP300_ROOT / "exp300_summary.csv",
    "stdout_log": EXP300_ROOT / "logs" / "experiment_300_independent_evidence_consensus.log", "expected_datasets": 251, "archive_outputs": False})
EXP301_ROOT = DATA_DIR / "exp301_train_risk_stability_routing"
EXPERIMENTS.append({"id": "experiment_301_train_risk_stability_routing", "label": "Exp301 · TRAIN risk and stability routing",
    "script": ROOT / "scratch/run_exp301_train_risk_stability_routing.py", "args": ["--phase", "all", "--output-root", str(EXP301_ROOT)],
    "detail_csv": EXP301_ROOT / "exp301_results.csv", "summary_csv": EXP301_ROOT / "exp301_summary.csv",
    "stdout_log": EXP301_ROOT / "logs" / "experiment_301_train_risk_stability_routing.log", "expected_datasets": 251, "archive_outputs": False})
EXP302_ROOT = DATA_DIR / "exp302_moe_automatic_alert_expansion_frontier"
EXPERIMENTS.append({"id": "experiment_302_moe_automatic_alert_expansion_frontier", "label": "Exp302 · MoE automatic alert expansion frontier",
    "script": ROOT / "scratch/run_exp302_moe_automatic_alert_expansion_frontier.py", "args": ["--phase", "all", "--output-root", str(EXP302_ROOT)],
    "detail_csv": EXP302_ROOT / "exp302_results.csv", "summary_csv": EXP302_ROOT / "exp302_summary.csv",
    "stdout_log": EXP302_ROOT / "logs" / "experiment_302_moe_automatic_alert_expansion_frontier.log", "expected_datasets": 1117, "archive_outputs": False})
EXP303_ROOT = DATA_DIR / "exp303_expanded_train_only_gates"
EXPERIMENTS.append({"id": "experiment_303_expanded_train_only_gates", "label": "Exp303 · Expanded TRAIN-only Gates",
    "script": ROOT / "scratch/run_exp303_expanded_train_only_gates.py", "args": ["--phase", "all", "--output-root", str(EXP303_ROOT)],
    "detail_csv": EXP303_ROOT / "exp303_results.csv", "summary_csv": EXP303_ROOT / "exp303_summary.csv",
    "stdout_log": EXP303_ROOT / "logs" / "experiment_303_expanded_train_only_gates.log", "expected_datasets": 1117, "archive_outputs": False})
EXP304_ROOT = DATA_DIR / "exp304_reconstruction_fifth_source_common_support"
EXPERIMENTS.append({"id": "experiment_304_reconstruction_fifth_source_common_support", "label": "Exp304 · Reconstruction fifth-source common support",
    "script": ROOT / "scratch/run_exp304_reconstruction_fifth_source_common_support.py", "args": ["--phase", "all", "--output-root", str(EXP304_ROOT)],
    "detail_csv": EXP304_ROOT / "exp304_results.csv", "summary_csv": EXP304_ROOT / "exp304_summary.csv",
    "stdout_log": EXP304_ROOT / "logs" / "experiment_304_reconstruction_fifth_source_common_support.log", "expected_datasets": 251, "archive_outputs": False})
EXP305_ROOT = DATA_DIR / "exp305_wavelet_reconstruction_cross_confirmation"
EXPERIMENTS.append({"id": "experiment_305_wavelet_reconstruction_cross_confirmation", "label": "Exp305 · Wavelet-reconstruction cross-confirmation",
    "script": ROOT / "scratch/run_exp305_wavelet_reconstruction_cross_confirmation.py", "args": ["--phase", "all", "--output-root", str(EXP305_ROOT)],
    "detail_csv": EXP305_ROOT / "exp305_results.csv", "summary_csv": EXP305_ROOT / "exp305_summary.csv",
    "stdout_log": EXP305_ROOT / "logs" / "experiment_305_wavelet_reconstruction_cross_confirmation.log", "expected_datasets": 251, "archive_outputs": False})
EXP306_ROOT = DATA_DIR / "exp306_componentized_operational_routing"
EXPERIMENTS.append({"id": "experiment_306_componentized_operational_routing", "label": "Exp306 · Componentized operational routing",
    "script": ROOT / "scratch/run_exp306_componentized_operational_routing.py", "args": ["--phase", "all", "--output-root", str(EXP306_ROOT)],
    "detail_csv": EXP306_ROOT / "exp306_results.csv", "summary_csv": EXP306_ROOT / "exp306_summary.csv",
    "stdout_log": EXP306_ROOT / "logs" / "experiment_306_componentized_operational_routing.log", "expected_datasets": 1117, "archive_outputs": False})
EXP307_ROOT = DATA_DIR / "exp307_expanded_six_expert_moe"
EXPERIMENTS.append({"id": "experiment_307_expanded_six_expert_moe", "label": "Exp307 · Expanded six-expert MoE",
    "script": ROOT / "scratch/run_exp307_expanded_six_expert_moe.py", "args": ["--phase", "all", "--output-root", str(EXP307_ROOT)],
    "detail_csv": EXP307_ROOT / "exp307_results.csv", "summary_csv": EXP307_ROOT / "exp307_summary.csv",
    "stdout_log": EXP307_ROOT / "logs" / "experiment_307_expanded_six_expert_moe.log", "expected_datasets": 1117, "archive_outputs": False})
EXP309_316_META = {
    309: ("experiment_309_multiscale_patch_discord", "Multi-scale patch discord"),
    310: ("experiment_310_multiwindow_change_point", "Multi-window change point"),
    311: ("experiment_311_wavelet_phase_coherence", "Wavelet phase coherence"),
    312: ("experiment_312_nonlearning_evidence_fusion", "Non-learning evidence fusion"),
    313: ("experiment_313_predictive_residual", "Predictive residual"),
    314: ("experiment_314_contrastive_normal_memory", "Contrastive normal memory"),
    315: ("experiment_315_deep_svdd_normal_boundary", "Deep SVDD normal boundary"),
    316: ("experiment_316_full_fn_recovery_fusion", "Full FN recovery fusion"),
}
for _number, (_experiment_id, _label) in EXP309_316_META.items():
    _output_root = DATA_DIR / _experiment_id.replace("experiment_", "exp", 1)
    _args = ["--experiment", str(_number), "--phase", "all", "--output-root", str(_output_root)]
    if _number in (309, 310, 311):
        _args.extend(["--workers", "7", "--device", "cpu"])
    elif _number in (313, 314, 315):
        _args.extend(["--workers", "1", "--device", "auto"])
    else:
        _args.extend(["--workers", "1", "--device", "cpu"])
    EXPERIMENTS.append({
        "id": _experiment_id, "label": f"Exp{_number} · {_label}",
        "script": ROOT / "scratch/run_exp309_316_pipeline.py", "args": _args,
        "detail_csv": _output_root / f"exp{_number}_results.csv",
        "summary_csv": _output_root / f"exp{_number}_summary.csv",
        "stdout_log": _output_root / "logs" / f"{_experiment_id}.log",
        "expected_datasets": 1117, "archive_outputs": False,
    })
EXP309_316_BATCH = [experiment_id for experiment_id, _label in EXP309_316_META.values()]
EXP317_318_META = {
    317: ("experiment_317_train_normal_source_reliability_audit", "TRAIN-normal source reliability audit"),
    318: ("experiment_318_train_reliability_weighted_fusion", "Reliability-weighted FN recovery fusion"),
}
for _number, (_experiment_id, _label) in EXP317_318_META.items():
    _output_root = DATA_DIR / _experiment_id.replace("experiment_", "exp", 1)
    EXPERIMENTS.append({
        "id": _experiment_id, "label": f"Exp{_number} · {_label}",
        "script": ROOT / "scratch/run_exp317_318_train_reliability_fusion.py",
        "args": ["--experiment", str(_number), "--output-root", str(_output_root)],
        "detail_csv": _output_root / ("exp317_reliability_audit.csv" if _number == 317 else "exp318_results.csv"),
        "summary_csv": _output_root / f"exp{_number}_summary.csv",
        "stdout_log": _output_root / "logs" / f"{_experiment_id}.log",
        "expected_datasets": 1117, "archive_outputs": False,
    })
EXP317_318_BATCH = [experiment_id for experiment_id, _label in EXP317_318_META.values()]
EXP319_321_META = {
    319: ("experiment_319_candidate_level_train_reliability", "Candidate-level TRAIN reliability"),
    320: ("experiment_320_existing_review_reliability_ranking", "Existing review reliability ranking"),
    321: ("experiment_321_frozen_dual_lane_architecture", "Frozen dual-lane architecture"),
}
for _number, (_experiment_id, _label) in EXP319_321_META.items():
    _output_root = DATA_DIR / _experiment_id.replace("experiment_", "exp", 1)
    EXPERIMENTS.append({
        "id": _experiment_id, "label": f"Exp{_number} · {_label}",
        "script": ROOT / "scratch/run_exp319_321_candidate_reliability_architecture.py",
        "args": ["--experiment", str(_number), "--phase", "all", "--output-root", str(_output_root)],
        "detail_csv": _output_root / f"exp{_number}_results.csv",
        "summary_csv": _output_root / f"exp{_number}_summary.csv",
        "stdout_log": _output_root / "logs" / f"{_experiment_id}.log",
        "expected_datasets": 1117, "archive_outputs": False,
    })
EXP319_321_BATCH = [experiment_id for experiment_id, _label in EXP319_321_META.values()]
EXP322_324_META = {
    322: ("experiment_322_exp291_weight_lineage_audit", "Exp291 weight lineage audit"),
    323: ("experiment_323_exp116_reliability_selection_replay", "Exp116 reliability selection replay"),
    324: ("experiment_324_exp291_reliability_route_parity", "Exp291 reliability route parity"),
}
for _number, (_experiment_id, _label) in EXP322_324_META.items():
    _output_root = DATA_DIR / _experiment_id.replace("experiment_", "exp", 1)
    EXPERIMENTS.append({
        "id": _experiment_id, "label": f"Exp{_number} · {_label}",
        "script": ROOT / "scratch/run_exp322_324_exp291_reliability_weighting.py",
        "args": ["--experiment", str(_number), "--output-root", str(_output_root)],
        "detail_csv": _output_root / f"exp{_number}_results.csv",
        "summary_csv": _output_root / f"exp{_number}_summary.csv",
        "stdout_log": _output_root / "logs" / f"{_experiment_id}.log",
        "expected_datasets": 1117 if _number != 322 else 6, "archive_outputs": False,
    })
EXP322_324_BATCH = [experiment_id for experiment_id, _label in EXP322_324_META.values()]
EXP325_327_META = {
    325: ("experiment_325_full_source_train_reliability_cache", "Five-source TRAIN reliability cache"),
    326: ("experiment_326_full_source_reliability_exp93_replay", "Five-source reliability Exp93 replay"),
    327: ("experiment_327_exp291_upstream_reliability_impact_audit", "Exp291 upstream reliability impact audit"),
}
for _number, (_experiment_id, _label) in EXP325_327_META.items():
    _output_root = DATA_DIR / _experiment_id.replace("experiment_", "exp", 1)
    EXPERIMENTS.append({
        "id": _experiment_id, "label": f"Exp{_number} · {_label}",
        "script": ROOT / "scratch/run_exp325_327_full_source_reliability.py",
        "args": ["--experiment", str(_number), "--workers", "7", "--output-root", str(_output_root)],
        "detail_csv": _output_root / f"exp{_number}_results.csv",
        "summary_csv": _output_root / f"exp{_number}_summary.csv",
        "stdout_log": _output_root / "logs" / f"{_experiment_id}.log",
        "expected_datasets": 1117, "archive_outputs": False,
    })
EXP325_327_BATCH = [experiment_id for experiment_id, _label in EXP325_327_META.values()]
EXP328_ID = "experiment_328_large_fn_reachability_audit"
EXP328_ROOT = DATA_DIR / "exp328_large_fn_reachability_audit"
EXPERIMENTS.append({
    "id": EXP328_ID, "label": "Exp328 · Large-dataset FN reachability audit",
    "script": ROOT / "scratch/run_exp328_large_fn_reachability.py",
    "args": ["--phase", "all", "--workers", "7", "--output-root", str(EXP328_ROOT)],
    "detail_csv": EXP328_ROOT / "exp328_results.csv",
    "summary_csv": EXP328_ROOT / "exp328_summary.csv",
    "stdout_log": EXP328_ROOT / "logs" / f"{EXP328_ID}.log",
    "expected_datasets": 251, "archive_outputs": False,
})
EXP329_330_META = {
    329: ("experiment_329_large_candidate_train_verifier", "Large candidate TRAIN-only verifier"),
    330: ("experiment_330_large_learned_normal_representation", "Large learned normal representation"),
}
for _number, (_experiment_id, _label) in EXP329_330_META.items():
    _output_root = DATA_DIR / ("exp329_large_candidate_train_verifier" if _number == 329 else "exp330_large_learned_normal_representation")
    EXPERIMENTS.append({
        "id": _experiment_id, "label": f"Exp{_number} · {_label}",
        "script": ROOT / "scratch/run_exp329_330_large_dataset_improvements.py",
        "args": ["--experiment", str(_number), "--phase", "all", "--workers", "7", "--output-root", str(_output_root)],
        "detail_csv": _output_root / f"exp{_number}_results.csv",
        "summary_csv": _output_root / f"exp{_number}_summary.csv",
        "stdout_log": _output_root / "logs" / f"{_experiment_id}.log",
        "expected_datasets": 251, "archive_outputs": False,
    })
EXP329_330_BATCH = [experiment_id for experiment_id, _label in EXP329_330_META.values()]
EXP177_185_BATCH = [
    "experiment_177_train_oof_review_policy_freeze",
    "experiment_178_frozen_policy_review_prediction",
    "experiment_179_canonical_multiscale_feature_cache",
    "experiment_180_canonical_score_backend_prediction",
    "experiment_181_existing_new_source_fusion_prediction",
    "experiment_182_five_hour_checkpoint_evaluation",
    "experiment_183_wavelet_multiscale_feature_prediction",
    "experiment_184_extension_source_fusion_prediction",
    "experiment_185_final_independent_reaggregation",
]

EXP186_200_BATCH = [
    "experiment_186_conservative_canonical_knn_agreement",
    "experiment_187_conservative_dual_canonical_agreement",
    "experiment_188_train_oof_stability_guarded_review",
    "experiment_189_train_normal_workload_guarded_review",
    "experiment_190_three_tier_review_routing",
    "experiment_191_conservative_canonical_shadow_agreement",
    "experiment_192_conservative_dual_backend_shadow_agreement",
    "experiment_193_three_existing_source_canonical_confirmation",
    "experiment_194_fold_stable_shadow_agreement",
    "experiment_195_zero_normal_candidate_shadow_guard",
    "experiment_196_expanded_train_risk_review",
    "experiment_197_conservative_canonical_unique_union",
    "experiment_198_existing_one_canonical_alpha_001",
    "experiment_199_expanded_two_of_six_train_workload_guard",
    "experiment_200_frozen_frontier_reaggregation",
]

EXP201_205_BATCH = [
    "experiment_201_frozen_large_size_baseline",
    "experiment_202_alternative_3of3_train_risk_rescue",
    "experiment_203_current_alternative_cross_confirmation",
    "experiment_204_size_stratified_shadow_router",
    "experiment_205_independent_large_zero_f1_frontier",
]

EXP206_210_BATCH = [
    "experiment_206_alternative_source_oof_reconstruction",
    "experiment_207_train_oof_conditional_novelty_audit",
    "experiment_208_continuous_alternative_source_fusion",
    "experiment_209_seven_source_joint_normal_novelty",
    "experiment_210_independent_frontier_reaggregation",
]

EXP211_214_BATCH = [
    "experiment_211_calibration_transport_audit",
    "experiment_212_fold_ensemble_query_scoring",
    "experiment_213_train_only_source_stability_guard",
    "experiment_214_calibration_transport_reaggregation",
]

EXP215_217_BATCH = [
    "experiment_215_fixed_dictionary_transport_control",
    "experiment_216_normal_consensus_medoid_transport",
    "experiment_217_train_normal_shapelet_risk_audit",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message):
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def load_state():
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    else:
        state = {"completed": [], "running": None, "history": []}
    state.setdefault("completed", [])
    state.setdefault("running", None)
    state.setdefault("history", [])
    state.setdefault("queue", [])
    return state


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def python_processes():
    out = subprocess.run(
        ["ps", "-axo", "pid,ppid,stat,etime,command"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    matches = []
    for line in out[1:]:
        if (
            "run_rank_ensemble_calibration" in line
            or "run_rank_ensemble_threshold_calibration" in line
            or "run_experiment_" in line
            or "run_rank_experiments_sequential.py" in line
        ):
            if str(os.getpid()) not in line:
                matches.append(line.strip())
    return matches


def active_rank_processes():
    experiment_scripts = {exp["script"].name for exp in EXPERIMENTS}
    matches = []
    for line in python_processes():
        if "python" not in line:
            continue
        command = line.split(None, 4)[-1] if len(line.split(None, 4)) >= 5 else line
        parts = command.split()
        script_names = {Path(part).name for part in parts if part.endswith(".py")}
        if script_names & experiment_scripts:
            matches.append(line)
    return matches


def active_rank_processes_or_none():
    """Return active experiments, or defer queue work when process inspection fails."""
    try:
        return active_rank_processes()
    except (BlockingIOError, OSError, subprocess.CalledProcessError) as exc:
        log(f"PROCESS inspection unavailable ({exc}); queue will retry")
        return None


def pid_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def reconcile_running_state(state):
    running = state.get("running")
    if not running:
        return state
    pids = [running.get("child_pid"), running.get("pid")]
    if any(pid_alive(pid) for pid in pids):
        return state
    stale = dict(running)
    state["running"] = None
    recovered_summary = None
    recovered = False
    try:
        exp = get_experiment(stale.get("id"))
        log_text = exp["stdout_log"].read_text(errors="replace") if exp["stdout_log"].exists() else ""
        recovered_summary = summarize_csv(exp["detail_csv"])
        expected = int(exp.get("expected_datasets") or 0)
        summary_complete = bool(
            recovered_summary
            and expected > 0
            and int(recovered_summary.get("datasets") or 0) >= expected
            and exp.get("summary_csv")
            and Path(exp["summary_csv"]).exists()
            and Path(exp["summary_csv"]).stat().st_size > 0
        )
        recovered = bool(recovered_summary and (f"{exp['id']} finished" in log_text or summary_complete))
    except (KeyError, OSError, ValueError):
        exp = None
    if recovered:
        state.setdefault("history", []).append(
            {
                "id": stale.get("id"),
                "finished_at": now(),
                "return_code": 0,
                "recovered_state": True,
                "summary": recovered_summary,
            }
        )
        if stale.get("id") not in state.get("completed", []):
            state.setdefault("completed", []).append(stale.get("id"))
        state["queue"] = [item for item in state.get("queue", []) if item != stale.get("id")]
        save_state(state)
        log(f"RECOVER completed {stale.get('id')} from finished output")
        return state
    state.setdefault("history", []).append(
        {
            "id": stale.get("id"),
            "finished_at": now(),
            "return_code": "stale_running_state_cleared",
            "summary": {
                "reason": "No live runner or child PID remained for recorded running state.",
                "stale_running": stale,
            },
        }
    )
    save_state(state)
    log(f"STALE running state cleared id={stale.get('id')} pid={stale.get('pid')} child_pid={stale.get('child_pid')}")
    return state


def get_experiment(exp_id):
    for exp in EXPERIMENTS:
        if exp["id"] == exp_id:
            return exp
    raise SystemExit(f"Unknown experiment id: {exp_id}")


def next_experiment(state):
    queue = [exp_id for exp_id in state.get("queue", []) if exp_id not in state.get("completed", [])]
    if queue:
        return get_experiment(queue[0])
    completed = set(state.get("completed", []))
    for exp in EXPERIMENTS:
        if exp["id"] not in completed:
            return exp
    return None


def queue_experiment(exp_id):
    get_experiment(exp_id)
    state = reconcile_running_state(load_state())
    if exp_id in state.get("completed", []):
        log(f"QUEUE skip already completed {exp_id}")
        return
    if exp_id not in state.get("queue", []):
        state.setdefault("queue", []).append(exp_id)
        save_state(state)
        log(f"QUEUE add {exp_id}")
    else:
        log(f"QUEUE already pending {exp_id}")


def remove_from_queue(exp_id):
    state = load_state()
    queue = state.get("queue", [])
    if exp_id in queue:
        state["queue"] = [item for item in queue if item != exp_id]
        save_state(state)


def summarize_csv(path):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"rows": 0, "datasets": 0, "configs": {}}
    dataset_names = {
        row.get("dataset_name") or row.get("dataset")
        for row in rows
        if row.get("dataset_name") or row.get("dataset")
    }
    dataset_count = len(dataset_names)
    if not dataset_count:
        declared_counts = []
        for row in rows:
            value = row.get("datasets_total") or row.get("datasets")
            try:
                declared_counts.append(int(float(value)))
            except (TypeError, ValueError):
                continue
        if declared_counts:
            dataset_count = max(declared_counts)
    supported_f1_keys = ("f1_evt", "hard_f1", "f1", "validated_f1")
    f1_key = next(
        (key for key in supported_f1_keys if key in rows[0]),
        None,
    )
    if f1_key is None:
        return {
            "rows": len(rows),
            "datasets": dataset_count,
            "configs": {},
        }
    configs = {}
    for row in rows:
        group_name = row.get("config_name") or row.get("source_experiment") or "default"
        if "strategy" in row:
            group_name = f"{row['config_name']}::{row['strategy']}"
        elif row.get("threshold_method"):
            group_name = f"{row['config_name']}::{row['threshold_method']}"
        try:
            configs.setdefault(group_name, []).append(float(row.get(f1_key, 0.0)))
        except (TypeError, ValueError):
            configs.setdefault(group_name, []).append(0.0)
    return {
        "rows": len(rows),
        "datasets": dataset_count,
        "configs": {
            name: {
                "mean_f1_evt": sum(values) / len(values),
                "median_f1_evt": sorted(values)[len(values) // 2],
            }
            for name, values in sorted(configs.items())
        },
    }


def write_heartbeat(update, replace=False):
    current = {} if replace else load_heartbeat()
    current.update(update)
    current["updated_unix"] = time.time()
    temporary = HEARTBEAT_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2))
    temporary.replace(HEARTBEAT_PATH)


def load_heartbeat():
    try:
        return json.loads(HEARTBEAT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def update_heartbeat_from_output(experiment_id, line):
    match = re.search(
        r"Progress:\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*rows=(\d+)\s+(?:last|dataset)=([^\s]+)(?:\s+errors=(\d+))?",
        line,
    )
    if not match:
        return
    write_heartbeat(
        {
            "experiment_id": experiment_id,
            "status": "running",
            "done": int(match.group(1)),
            "expected": int(match.group(2)),
            "rows": int(match.group(3)),
            "current_dataset": match.group(4),
            "errors": int(match.group(5) or 0),
            "updated_at": now(),
        }
    )


def run_experiment(exp, overlap_checked=False):
    if not overlap_checked:
        active = active_rank_processes()
        if active:
            raise SystemExit("Another rank experiment is already running:\n" + "\n".join(active))
    if not exp["script"].exists():
        raise SystemExit(f"Script does not exist: {exp['script']}")

    ensure_output_directories(exp)
    archive_existing_outputs(exp)

    state = load_state()
    state["running"] = {"id": exp["id"], "started_at": now(), "pid": os.getpid()}
    save_state(state)
    write_heartbeat(
        {
            "experiment_id": exp["id"],
            "status": "starting",
            "done": 0,
            "expected": None,
            "rows": 0,
            "current_dataset": "",
            "errors": 0,
            "updated_at": now(),
        }
    )

    worker_count = os.environ.get("RANK_EXPERIMENT_WORKERS", "7")
    log(f"START {exp['id']} script={exp['script']} rank_workers={worker_count}")
    start = time.time()
    with exp["stdout_log"].open("w") as stdout_log:
        batch_meta = state.get("exp186_200_batch") or state.get("exp177_185_batch") or {}
        child_env = build_child_environment(worker_count=worker_count)
        if (exp["id"] in EXP177_185_BATCH or exp["id"] in EXP186_200_BATCH) and batch_meta.get("hard_deadline_unix"):
            child_env["EXP177_185_BATCH_DEADLINE_UNIX"] = str(batch_meta["hard_deadline_unix"])
        process = subprocess.Popen(
            [
                str(exp.get("python_bin", PYTHON_BIN)),
                str(exp["script"]),
                *[str(argument) for argument in exp.get("args", [])],
            ],
            cwd=str(ROOT),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        state["running"]["child_pid"] = process.pid
        save_state(state)
        write_heartbeat({"child_pid": process.pid, "status": "running", "updated_at": now()})
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                stdout_log.write(line)
                stdout_log.flush()
                update_heartbeat_from_output(exp["id"], line)
            return_code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            return_code = process.wait()
            raise

    elapsed = time.time() - start
    final_summary = summarize_csv(exp["detail_csv"])
    state = load_state()
    state["running"] = None
    state.setdefault("history", []).append(
        {
            "id": exp["id"],
            "finished_at": now(),
            "return_code": return_code,
            "elapsed_minutes": round(elapsed / 60, 2),
            "summary": final_summary,
        }
    )
    if return_code == 0 and exp["id"] not in state.get("completed", []):
        state.setdefault("completed", []).append(exp["id"])
    if return_code == 0:
        state["queue"] = [item for item in state.get("queue", []) if item != exp["id"]]
    save_state(state)
    write_heartbeat(
        {
            "experiment_id": exp["id"],
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "done": int((final_summary or {}).get("datasets") or 0),
            "expected": int(exp.get("expected_datasets") or 0),
            "rows": int((final_summary or {}).get("rows") or 0),
            "current_dataset": "",
            "errors": 0 if return_code == 0 else 1,
            "updated_at": now(),
        }
    )
    log(f"END {exp['id']} return_code={return_code} elapsed_min={elapsed / 60:.2f}")
    if return_code != 0:
        raise SystemExit(return_code)


def ensure_output_directories(exp):
    for key in ("detail_csv", "summary_csv", "stdout_log"):
        exp[key].parent.mkdir(parents=True, exist_ok=True)


def archive_existing_outputs(exp):
    if not exp.get("archive_outputs", True):
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for key in ["detail_csv", "summary_csv", "stdout_log"]:
        path = exp[key]
        if path.exists():
            archive_path = path.with_name(f"{path.stem}.before_{exp['id']}_{stamp}{path.suffix}")
            path.rename(archive_path)
            log(f"ARCHIVE {path} -> {archive_path}")


def ensure_output_directories(exp):
    for key in ("detail_csv", "summary_csv", "stdout_log"):
        Path(exp[key]).parent.mkdir(parents=True, exist_ok=True)


def cmd_status():
    state = reconcile_running_state(load_state())
    print(json.dumps(state, indent=2, sort_keys=True))
    active = active_rank_processes()
    if active:
        print("\nActive rank processes:")
        print("\n".join(active))
    for exp in EXPERIMENTS:
        summary = summarize_csv(exp["detail_csv"])
        if summary:
            print(f"\n{exp['id']} partial summary:")
            print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_list():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queued = set(state.get("queue", []))
    running_id = (state.get("running") or {}).get("id")
    for exp in EXPERIMENTS:
        if exp["id"] == running_id:
            marker = "running"
        elif exp["id"] in completed:
            marker = "done"
        elif exp["id"] in queued:
            marker = "queued"
        else:
            marker = "pending"
        print(f"{marker:7s} {exp['id']} -> {exp['script'].name}")


def cmd_run_queue(poll_seconds=30):
    with QUEUE_LOCK_PATH.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("QUEUE runner already active; exiting duplicate runner")
            return
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()
        log("QUEUE runner started")
        while True:
            state = reconcile_running_state(load_state())
            skipped_any = False
            for batch_name, batch_list, label in [
                ("exp177_185_batch", EXP177_185_BATCH, "Exp177-185"),
                ("exp186_200_batch", EXP186_200_BATCH, "Exp186-200"),
                ("exp206_210_batch", EXP206_210_BATCH, "Exp206-210"),
                ("exp211_214_batch", EXP211_214_BATCH, "Exp211-214"),
            ]:
                batch_meta = state.get(batch_name) or {}
                hard_deadline = float(batch_meta.get("hard_deadline_unix") or 0)
                if hard_deadline and time.time() >= hard_deadline:
                    remaining = [item for item in state.get("queue", []) if item in batch_list]
                    if remaining:
                        state["queue"] = [item for item in state.get("queue", []) if item not in batch_list]
                        state.setdefault("skipped", []).extend(
                            {"id": item, "skipped_at": now(), "reason": "12-hour batch hard deadline"}
                            for item in remaining
                        )
                        save_state(state)
                        log(f"QUEUE {label} hard deadline reached; skipped {len(remaining)} remaining stages")
                        skipped_any = True
            if skipped_any:
                continue
            queue = [exp_id for exp_id in state.get("queue", []) if exp_id not in state.get("completed", [])]
            if not queue:
                log("QUEUE empty; runner exiting")
                return
            active = active_rank_processes_or_none()
            if active is None:
                time.sleep(min(poll_seconds, 5))
                continue
            if state.get("running") or active:
                time.sleep(poll_seconds)
                continue
            exp_id = queue[0]
            log(f"QUEUE starting {exp_id}")
            try:
                run_experiment(get_experiment(exp_id), overlap_checked=True)
            except SystemExit as exc:
                state = load_state()
                state["queue"] = [item for item in state.get("queue", []) if item != exp_id]
                state.setdefault("failed", []).append(
                    {"id": exp_id, "failed_at": now(), "return_code": exc.code}
                )
                save_state(state)
                log(f"QUEUE recorded failure {exp_id} return_code={exc.code}; continuing")


def queue_exp177_185_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP177_185_BATCH]
    queue.extend(item for item in EXP177_185_BATCH if item not in completed)
    state["queue"] = queue
    registered_unix = time.time()
    state["exp177_185_batch"] = {
        "registered_at": now(),
        "registered_unix": registered_unix,
        "hard_deadline_unix": registered_unix + 12 * 60 * 60,
        "worker_count": int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")),
        "soft_checkpoint_hours": 5,
        "hard_limit_hours": 12,
        "agent_monitoring": False,
        "experiments": EXP177_185_BATCH,
    }
    save_state(state)
    log(f"QUEUE registered Exp177-185 batch ({len(EXP177_185_BATCH)} experiments)")


def queue_exp186_200_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP186_200_BATCH]
    queue.extend(item for item in EXP186_200_BATCH if item not in completed)
    state["queue"] = queue
    registered_unix = time.time()
    state["exp186_200_batch"] = {
        "registered_at": now(),
        "registered_unix": registered_unix,
        "hard_deadline_unix": registered_unix + 12 * 60 * 60,
        "worker_count": int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")),
        "soft_checkpoint_hours": 5,
        "hard_limit_hours": 12,
        "agent_monitoring": False,
        "experiments": EXP186_200_BATCH,
    }
    save_state(state)
    log(f"QUEUE registered Exp186-200 batch ({len(EXP186_200_BATCH)} experiments)")


def queue_exp201_205_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP201_205_BATCH]
    queue.extend(item for item in EXP201_205_BATCH if item not in completed)
    state["queue"] = queue
    state["exp201_205_batch"] = {
        "registered_at": now(),
        "registered_unix": time.time(),
        "worker_count": int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")),
        "agent_monitoring": False,
        "execution_mode": "sequential_one_stage_at_a_time",
        "experiments": EXP201_205_BATCH,
    }
    save_state(state)
    log(f"QUEUE registered Exp201-205 batch ({len(EXP201_205_BATCH)} experiments)")


def queue_exp206_210_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP206_210_BATCH]
    queue.extend(item for item in EXP206_210_BATCH if item not in completed)
    state["queue"] = queue
    state["exp206_210_batch"] = {
        "registered_at": now(),
        "registered_unix": time.time(),
        "worker_count": int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")),
        "agent_monitoring": False,
        "execution_mode": "sequential_one_stage_at_a_time",
        "experiments": EXP206_210_BATCH,
    }
    save_state(state)
    log(f"QUEUE registered Exp206-210 batch ({len(EXP206_210_BATCH)} experiments)")


def queue_exp211_214_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP211_214_BATCH]
    queue.extend(item for item in EXP211_214_BATCH if item not in completed)
    state["queue"] = queue
    state["exp211_214_batch"] = {
        "registered_at": now(),
        "registered_unix": time.time(),
        "worker_count": int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")),
        "agent_monitoring": False,
        "execution_mode": "sequential_one_stage_at_a_time",
        "experiments": EXP211_214_BATCH,
    }
    save_state(state)
    log(f"QUEUE registered Exp211-214 batch ({len(EXP211_214_BATCH)} experiments)")


def queue_exp215_217_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP215_217_BATCH]
    queue.extend(item for item in EXP215_217_BATCH if item not in completed)
    state["queue"] = queue
    state["exp215_217_batch"] = {
        "registered_at": now(),
        "registered_unix": time.time(),
        "worker_count": 1,
        "agent_monitoring": False,
        "execution_mode": "sequential_one_stage_at_a_time",
        "experiments": EXP215_217_BATCH,
    }
    save_state(state)
    log(f"QUEUE registered Exp215-217 batch ({len(EXP215_217_BATCH)} experiments)")


def queue_exp267_273_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP267_273_BATCH]
    queue.extend(item for item in EXP267_273_BATCH if item not in completed)
    state["queue"] = queue
    state["exp267_273_batch"] = {
        "registered_at": now(),
        "registered_unix": time.time(),
        "worker_count": 7,
        "agent_monitoring": False,
        "execution_mode": "sequential_one_stage_at_a_time",
        "experiments": list(EXP267_273_BATCH),
    }
    save_state(state)
    log(f"QUEUE registered Exp267-273 batch ({len(EXP267_273_BATCH)} experiments)")


def queue_exp274_275_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP274_275_BATCH]
    queue.extend(item for item in EXP274_275_BATCH if item not in completed)
    state["queue"] = queue
    state["exp274_275_batch"] = {
        "registered_at": now(),
        "registered_unix": time.time(),
        "worker_count": 1,
        "agent_monitoring": False,
        "execution_mode": "sequential_one_stage_at_a_time",
        "experiments": list(EXP274_275_BATCH),
        "autonomous_policy_changed": False,
        "review_only": True,
    }
    save_state(state)
    log(f"QUEUE registered Exp274-275 batch ({len(EXP274_275_BATCH)} experiments)")


def queue_exp276_277_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP276_277_BATCH]
    queue.extend(item for item in EXP276_277_BATCH if item not in completed)
    state["queue"] = queue
    state["exp276_277_batch"] = {
        "registered_at": now(),
        "registered_unix": time.time(),
        "worker_count": 1,
        "agent_monitoring": False,
        "execution_mode": "sequential_one_stage_at_a_time",
        "experiments": list(EXP276_277_BATCH),
        "autonomous_policy_changed": False,
        "review_only": True,
    }
    save_state(state)
    log(f"QUEUE registered Exp276-277 batch ({len(EXP276_277_BATCH)} experiments)")


def queue_exp278_281_batch():
    state = reconcile_running_state(load_state())
    completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP278_281_BATCH]
    queue.extend(item for item in EXP278_281_BATCH if item not in completed)
    state["queue"] = queue
    state["exp278_281_batch"] = {
        "registered_at": now(), "registered_unix": time.time(), "worker_count": 1,
        "agent_monitoring": False, "execution_mode": "sequential_one_stage_at_a_time",
        "experiments": list(EXP278_281_BATCH), "autonomous_policy_changed": False,
        "review_only": True,
    }
    save_state(state)
    log(f"QUEUE registered Exp278-281 batch ({len(EXP278_281_BATCH)} experiments)")


def queue_exp282_284_batch():
    state = reconcile_running_state(load_state()); completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP282_284_BATCH]
    queue.extend(item for item in EXP282_284_BATCH if item not in completed); state["queue"] = queue
    state["exp282_284_batch"] = {
        "registered_at": now(), "registered_unix": time.time(), "worker_count": 1,
        "agent_monitoring": False, "execution_mode": "sequential_one_stage_at_a_time",
        "experiments": list(EXP282_284_BATCH), "autonomous_policy_changed": False, "review_only": True,
    }
    save_state(state); log(f"QUEUE registered Exp282-284 batch ({len(EXP282_284_BATCH)} experiments)")


def queue_exp285_287_batch():
    state = reconcile_running_state(load_state()); completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP285_287_BATCH]
    queue.extend(item for item in EXP285_287_BATCH if item not in completed); state["queue"] = queue
    state["exp285_287_batch"] = {"registered_at": now(), "registered_unix": time.time(), "worker_count": 1,
        "agent_monitoring": False, "execution_mode": "sequential_one_stage_at_a_time", "experiments": list(EXP285_287_BATCH),
        "autonomous_policy_changed": False, "review_only": True}
    save_state(state); log(f"QUEUE registered Exp285-287 batch ({len(EXP285_287_BATCH)} experiments)")


def queue_exp309_316_batch():
    state = reconcile_running_state(load_state()); completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP309_316_BATCH]
    queue.extend(item for item in EXP309_316_BATCH if item not in completed); state["queue"] = queue
    state["exp309_316_batch"] = {
        "registered_at": now(), "registered_unix": time.time(), "worker_count": 7,
        "agent_monitoring": False, "execution_mode": "sequential_one_experiment_at_a_time",
        "experiments": list(EXP309_316_BATCH), "automatic_policy_changed": False,
        "research_shadow_only": True,
    }
    save_state(state); log(f"QUEUE registered Exp309-316 batch ({len(EXP309_316_BATCH)} experiments)")


def queue_exp317_318_batch():
    state = reconcile_running_state(load_state()); completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP317_318_BATCH]
    queue.extend(item for item in EXP317_318_BATCH if item not in completed); state["queue"] = queue
    state["exp317_318_batch"] = {
        "registered_at": now(), "registered_unix": time.time(), "worker_count": 1,
        "agent_monitoring": False, "execution_mode": "sequential_one_experiment_at_a_time",
        "experiments": list(EXP317_318_BATCH), "automatic_policy_changed": False,
        "research_shadow_only": True,
    }
    save_state(state); log("QUEUE registered Exp317-318 batch (2 experiments)")


def queue_exp319_321_batch():
    state = reconcile_running_state(load_state()); completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP319_321_BATCH]
    queue.extend(item for item in EXP319_321_BATCH if item not in completed); state["queue"] = queue
    state["exp319_321_batch"] = {
        "registered_at": now(), "registered_unix": time.time(), "worker_count": 1,
        "agent_monitoring": False, "execution_mode": "sequential_one_experiment_at_a_time",
        "experiments": list(EXP319_321_BATCH), "automatic_policy_changed": False,
        "research_shadow_only": True,
    }
    save_state(state); log("QUEUE registered Exp319-321 batch (3 experiments)")


def queue_exp322_324_batch():
    state = reconcile_running_state(load_state()); completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP322_324_BATCH]
    queue.extend(item for item in EXP322_324_BATCH if item not in completed); state["queue"] = queue
    state["exp322_324_batch"] = {
        "registered_at": now(), "registered_unix": time.time(), "worker_count": 1,
        "agent_monitoring": False, "execution_mode": "sequential_one_experiment_at_a_time",
        "experiments": list(EXP322_324_BATCH), "automatic_policy_changed": False,
        "partial_three_source_scope": True,
    }
    save_state(state); log("QUEUE registered Exp322-324 batch (3 experiments)")


def queue_exp325_327_batch():
    state = reconcile_running_state(load_state()); completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP325_327_BATCH]
    queue.extend(item for item in EXP325_327_BATCH if item not in completed); state["queue"] = queue
    state["exp325_327_batch"] = {
        "registered_at": now(), "registered_unix": time.time(), "worker_count": 7,
        "agent_monitoring": False, "execution_mode": "sequential_one_experiment_at_a_time",
        "experiments": list(EXP325_327_BATCH), "automatic_policy_changed": False,
        "research_shadow_only": True, "all_five_source_weighting": True,
    }
    save_state(state); log("QUEUE registered Exp325-327 batch (3 experiments)")


def queue_exp329_330_batch():
    state = reconcile_running_state(load_state()); completed = set(state.get("completed", []))
    queue = [item for item in state.get("queue", []) if item not in EXP329_330_BATCH]
    queue.extend(item for item in EXP329_330_BATCH if item not in completed); state["queue"] = queue
    state["exp329_330_batch"] = {
        "registered_at": now(), "registered_unix": time.time(), "worker_count": 7,
        "agent_monitoring": False, "execution_mode": "sequential_one_experiment_at_a_time",
        "experiments": list(EXP329_330_BATCH), "automatic_policy_changed": False,
        "research_shadow_only": True,
    }
    save_state(state); log("QUEUE registered Exp329-330 batch (2 experiments)")


def main():
    parser = argparse.ArgumentParser(description="Run rank experiments one at a time.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("status")
    sub.add_parser("run-next")
    sub.add_parser("run-queue")
    sub.add_parser("queue-exp177-185")
    sub.add_parser("queue-exp186-200")
    sub.add_parser("queue-exp201-205")
    sub.add_parser("queue-exp206-210")
    sub.add_parser("queue-exp211-214")
    sub.add_parser("queue-exp215-217")
    sub.add_parser("queue-exp267-273")
    sub.add_parser("queue-exp274-275")
    sub.add_parser("queue-exp276-277")
    sub.add_parser("queue-exp278-281")
    sub.add_parser("queue-exp282-284")
    sub.add_parser("queue-exp285-287")
    sub.add_parser("queue-exp288")
    sub.add_parser("queue-exp289")
    sub.add_parser("queue-exp290")
    sub.add_parser("queue-exp291")
    sub.add_parser("queue-exp292")
    sub.add_parser("queue-exp293-294")
    sub.add_parser("queue-exp295")
    sub.add_parser("queue-exp296")
    sub.add_parser("queue-exp297")
    sub.add_parser("queue-exp298")
    sub.add_parser("queue-exp299")
    sub.add_parser("queue-exp300")
    sub.add_parser("queue-exp301")
    sub.add_parser("queue-exp302")
    sub.add_parser("queue-exp303")
    sub.add_parser("queue-exp304")
    sub.add_parser("queue-exp305")
    sub.add_parser("queue-exp306")
    sub.add_parser("queue-exp307")
    sub.add_parser("queue-exp309-316")
    sub.add_parser("queue-exp317-318")
    sub.add_parser("queue-exp319-321")
    sub.add_parser("queue-exp322-324")
    sub.add_parser("queue-exp325-327")
    sub.add_parser("queue-exp328")
    sub.add_parser("queue-exp329-330")
    queue_one = sub.add_parser("queue")
    queue_one.add_argument("experiment_id")
    run_one = sub.add_parser("run")
    run_one.add_argument("experiment_id")
    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "status":
        cmd_status()
    elif args.command == "run-next":
        exp = next_experiment(load_state())
        if exp is None:
            log("No pending experiments.")
            return
        run_experiment(exp)
    elif args.command == "queue":
        queue_experiment(args.experiment_id)
    elif args.command == "run-queue":
        cmd_run_queue()
    elif args.command == "queue-exp177-185":
        queue_exp177_185_batch()
    elif args.command == "queue-exp186-200":
        queue_exp186_200_batch()
    elif args.command == "queue-exp201-205":
        queue_exp201_205_batch()
    elif args.command == "queue-exp206-210":
        queue_exp206_210_batch()
    elif args.command == "queue-exp211-214":
        queue_exp211_214_batch()
    elif args.command == "queue-exp215-217":
        queue_exp215_217_batch()
    elif args.command == "queue-exp267-273":
        queue_exp267_273_batch()
    elif args.command == "queue-exp274-275":
        queue_exp274_275_batch()
    elif args.command == "queue-exp276-277":
        queue_exp276_277_batch()
    elif args.command == "queue-exp278-281":
        queue_exp278_281_batch()
    elif args.command == "queue-exp282-284":
        queue_exp282_284_batch()
    elif args.command == "queue-exp285-287":
        queue_exp285_287_batch()
    elif args.command == "queue-exp288":
        queue_experiment("experiment_288_recovery_priority_counterexample_audit")
    elif args.command == "queue-exp289":
        queue_experiment("experiment_289_recovery_priority_label_consistency_audit")
    elif args.command == "queue-exp290":
        queue_experiment("experiment_290_train_raw_exact_normal_guard")
    elif args.command == "queue-exp291":
        queue_experiment("experiment_291_final_operational_routing_freeze")
    elif args.command == "queue-exp292":
        queue_experiment("experiment_292_standard_review_evidence_tiering")
    elif args.command == "queue-exp293-294":
        queue_experiment("experiment_293_train_only_moe_expert_audit")
        queue_experiment("experiment_294_frozen_sparse_moe_review_router")
    elif args.command == "queue-exp295":
        queue_experiment("experiment_295_moe_hard_alert_confidence_tiering")
    elif args.command == "queue-exp296":
        queue_experiment("experiment_296_cross_gate_hard_confidence_consensus")
    elif args.command == "queue-exp297":
        queue_experiment("experiment_297_bootstrap_confidence_stability")
    elif args.command == "queue-exp298":
        queue_experiment("experiment_298_exp201_current_hard_rebase")
    elif args.command == "queue-exp299":
        queue_experiment("experiment_299_multirocket_capacity_consensus")
    elif args.command == "queue-exp300":
        queue_experiment("experiment_300_independent_evidence_consensus")
    elif args.command == "queue-exp301":
        queue_experiment("experiment_301_train_risk_stability_routing")
    elif args.command == "queue-exp302":
        queue_experiment("experiment_302_moe_automatic_alert_expansion_frontier")
    elif args.command == "queue-exp303":
        queue_experiment("experiment_303_expanded_train_only_gates")
    elif args.command == "queue-exp304":
        queue_experiment("experiment_304_reconstruction_fifth_source_common_support")
    elif args.command == "queue-exp305":
        queue_experiment("experiment_305_wavelet_reconstruction_cross_confirmation")
    elif args.command == "queue-exp306":
        queue_experiment("experiment_306_componentized_operational_routing")
    elif args.command == "queue-exp307":
        queue_experiment("experiment_307_expanded_six_expert_moe")
    elif args.command == "queue-exp309-316":
        queue_exp309_316_batch()
    elif args.command == "queue-exp317-318":
        queue_exp317_318_batch()
    elif args.command == "queue-exp319-321":
        queue_exp319_321_batch()
    elif args.command == "queue-exp322-324":
        queue_exp322_324_batch()
    elif args.command == "queue-exp325-327":
        queue_exp325_327_batch()
    elif args.command == "queue-exp328":
        queue_experiment(EXP328_ID)
    elif args.command == "queue-exp329-330":
        queue_exp329_330_batch()
    elif args.command == "run":
        run_experiment(get_experiment(args.experiment_id))


if __name__ == "__main__":
    main()
