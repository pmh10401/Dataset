# Evaluation summary

Hard alert metrics are autonomous. Standard/Priority review and combined F1 are human-assisted diagnostics. No TEST metric selects a variant.

```json
[
  {
    "abstained_datasets": 1095,
    "datasets": 1117,
    "experiment_id": "experiment_157_train_only_risk_controlled_alpha_policy",
    "hard_alerts": 368,
    "hard_fp": 20,
    "hard_precision": 0.9456521739130435,
    "hard_tp": 348,
    "mean_combined_f1": 0.01064786993711852,
    "mean_combined_f1_scope": "human_assisted_diagnostic_only",
    "mean_hard_f1": 0.009958584730738991,
    "mean_hard_recall": 0.008280908712316827,
    "priority_review_candidates": 8,
    "priority_review_fp": 2,
    "priority_review_precision": 0.75,
    "priority_review_tp": 6,
    "risk_target_per_run": 0.005,
    "standard_review_candidates": 46,
    "standard_review_fp": 16,
    "standard_review_precision": 0.6521739130434783,
    "standard_review_tp": 30,
    "variant": "risk_target_0.005_per_run"
  },
  {
    "abstained_datasets": 1010,
    "datasets": 1117,
    "experiment_id": "experiment_157_train_only_risk_controlled_alpha_policy",
    "hard_alerts": 905,
    "hard_fp": 102,
    "hard_precision": 0.887292817679558,
    "hard_tp": 803,
    "mean_combined_f1": 0.04272667355766758,
    "mean_combined_f1_scope": "human_assisted_diagnostic_only",
    "mean_hard_f1": 0.038831543602676086,
    "mean_hard_recall": 0.032007067087174386,
    "priority_review_candidates": 32,
    "priority_review_fp": 12,
    "priority_review_precision": 0.625,
    "priority_review_tp": 20,
    "risk_target_per_run": 0.01,
    "standard_review_candidates": 236,
    "standard_review_fp": 109,
    "standard_review_precision": 0.538135593220339,
    "standard_review_tp": 127,
    "variant": "risk_target_0.010_per_run"
  },
  {
    "abstained_datasets": 907,
    "datasets": 1117,
    "experiment_id": "experiment_157_train_only_risk_controlled_alpha_policy",
    "hard_alerts": 1549,
    "hard_fp": 251,
    "hard_precision": 0.8379599741768883,
    "hard_tp": 1298,
    "mean_combined_f1": 0.08948809353760398,
    "mean_combined_f1_scope": "human_assisted_diagnostic_only",
    "mean_hard_f1": 0.08314430202285385,
    "mean_hard_recall": 0.07542760971055969,
    "priority_review_candidates": 68,
    "priority_review_fp": 31,
    "priority_review_precision": 0.5441176470588235,
    "priority_review_tp": 37,
    "risk_target_per_run": 0.02,
    "standard_review_candidates": 701,
    "standard_review_fp": 424,
    "standard_review_precision": 0.39514978601997147,
    "standard_review_tp": 277,
    "variant": "risk_target_0.020_per_run"
  }
]
```
