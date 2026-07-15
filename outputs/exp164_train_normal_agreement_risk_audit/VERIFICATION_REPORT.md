# Exp164 Verification Report

This report validates the outputs of the Exp164 risk audit run.

## Summary of Checks

- **Total manifest entries checked:** 1117
- **Eligible (computed) caches verified:** 955
- **Abstained datasets count:** 162
- **Total normal runs analyzed:** 146796
- **Cache File SHA256 integrity check:** PASS (All hashes match the manifest)
- **Evaluation field safety check:** PASS (All caches verified contains_test_values=false, contains_test_labels=false, contains_evaluation_metrics=false)
- **Dataset summary values check:** PASS (04_dataset_alpha_summary.csv matches independent calculation)
- **Global summary values check:** PASS (05_global_summary.csv matches independent calculation)
- **Risk target sensitivity check:** PASS (06_risk_target_sensitivity.csv matches independent calculation)
- **Source dependence metrics check:** PASS (07_source_dependence.csv matches independent calculation)

## Conclusion

The Exp164 result caches and aggregate metrics match this independent reaggregation. No TEST access was detected in the audited Exp164 data path; this does not establish end-to-end provenance or prospective validity.
