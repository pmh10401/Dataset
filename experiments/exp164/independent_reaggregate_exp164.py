#!/usr/bin/env python3
"""
Independent Reaggregate Validation for Exp164.
Verifies all caches and runner summary csvs.
"""
from __future__ import annotations

import csv
import json
import hashlib
import sys
from statistics import NormalDist
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

def independent_wilson_upper_bound(false_reviews: int, runs: int, confidence: float = 0.95) -> float:
    if runs == 0:
        return 1.0
    p = false_reviews / runs
    z = NormalDist().inv_cdf(confidence)
    denom = 1.0 + z * z / runs
    center = p + z * z / (2.0 * runs)
    spread = z * np.sqrt(p * (1.0 - p) / runs + z * z / (4.0 * runs * runs))
    return float((center + spread) / denom)

def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_and_reaggregate(output_dir_path: str):
    output_dir = Path(output_dir_path)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir

    print(f"Starting Independent Verification on {output_dir}")

    required_outputs = (
        "01_audit_contract.json",
        "02_cache_manifest.csv",
        "03_errors.csv",
        "04_dataset_alpha_summary.csv",
        "05_global_summary.csv",
        "06_risk_target_sensitivity.csv",
        "07_source_dependence.csv",
        "08_threshold_resolution.csv",
        "09_n_train_bin_summary.csv",
    )
    missing_outputs = [name for name in required_outputs if not (output_dir / name).exists()]
    if missing_outputs:
        raise FileNotFoundError(f"Missing required Exp164 outputs: {missing_outputs}")

    contract = json.loads((output_dir / "01_audit_contract.json").read_text())

    manifest_file = output_dir / "02_cache_manifest.csv"
    if not manifest_file.exists():
        print(f"Error: Manifest {manifest_file} not found")
        sys.exit(1)

    # Read manifest
    manifest_entries = []
    with manifest_file.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_entries.append(row)

    print(f"Read {len(manifest_entries)} entries from manifest.")
    manifest_names = [row["dataset"] for row in manifest_entries]
    if len(manifest_names) != int(contract["population_size"]):
        raise ValueError("Manifest population size does not match the frozen run contract")
    if len(set(manifest_names)) != len(manifest_names):
        raise ValueError("Manifest contains duplicate dataset names")

    with (output_dir / "03_errors.csv").open() as f:
        if sum(1 for _ in csv.DictReader(f)):
            raise ValueError("03_errors.csv contains dataset errors")

    # 1. Validate files and hashes
    abstained_count = 0
    computed_count = 0
    total_runs = 0

    # keys: (alpha, src) -> [false_reviews, total_runs]
    recalc_global = {}
    alphas = (0.005, 0.01, 0.02, 0.05)
    sources = ("spectral", "shapelet", "interval")
    variants = ("train_audit_2of3", "train_audit_3of3")

    # dataset -> alpha -> src -> (false_reviews, runs, rate, w_upper)
    recalc_dataset_alpha = {}

    # For dependence recalculation
    all_flags = {a: {src: [] for src in sources} for a in alphas}

    for row in manifest_entries:
        ds = row["dataset"]
        status = row["status"]
        if status == "abstained":
            abstained_count += 1
            continue

        computed_count += 1
        rel_path = row["file_path"]
        expected_hash = row["file_hash"]

        cache_path = PROJECT_DIR / rel_path
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file {cache_path} not found")

        # Verify SHA256
        actual_hash = get_sha256(cache_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Cache hash mismatch for {ds}! Expected {expected_hash}, got {actual_hash}")

        # Load and verify arrays
        with np.load(cache_path, allow_pickle=False) as data:
            n_train = int(row["n_train"])
            total_runs += n_train

            # check labels/values false flags
            meta_json = data["metadata_json"]
            meta_str = str(meta_json) if meta_json.ndim == 0 else str(meta_json[0])
            meta = json.loads(meta_str)
            if meta.get("dataset") != ds or int(meta.get("n_train", -1)) != n_train:
                raise ValueError(f"Cache metadata identity mismatch for {ds}")
            if meta.get("source_config_hash") != contract["global_config_hash"]:
                raise ValueError(f"Cache config hash mismatch for {ds}")
            for k in ("contains_test_values", "contains_test_labels", "contains_evaluation_metrics"):
                if meta.get(k) is not False:
                    raise ValueError(f"Forbidden true flag for {k} in {ds}")

            flags_src = {
                "spectral": data["flags_spectral"],
                "shapelet": data["flags_shapelet"],
                "interval": data["flags_interval"],
                "train_audit_2of3": data["agreement_2of3"],
                "train_audit_3of3": data["agreement_3of3"],
            }

            if not np.array_equal(data["pseudo_run_indices"], np.arange(n_train)):
                raise ValueError(f"Pseudo-run coverage mismatch for {ds}")
            if np.any(data["fit_counts"] <= 0) or np.any(data["fit_counts"] >= n_train):
                raise ValueError(f"Invalid fit counts for {ds}")
            if np.any(data["target_lengths"] < 8) or np.any(data["target_lengths"] > 2048):
                raise ValueError(f"Invalid target lengths for {ds}")

            independent_source_flags = []
            for source in sources:
                scores = data[f"query_score_{source}"][:, None]
                thresholds = data[f"thresholds_{source}"]
                expected_flags = np.isfinite(scores) & (scores > thresholds)
                if not np.array_equal(flags_src[source], expected_flags):
                    raise ValueError(f"Score/threshold/flag mismatch for {ds} {source}")
                independent_source_flags.append(expected_flags)
            source_count = np.stack(independent_source_flags, axis=0).sum(axis=0)
            if not np.array_equal(flags_src["train_audit_2of3"], source_count >= 2):
                raise ValueError(f"2-of-3 agreement mismatch for {ds}")
            if not np.array_equal(flags_src["train_audit_3of3"], source_count >= 3):
                raise ValueError(f"3-of-3 agreement mismatch for {ds}")

            # append flags for dependence
            for idx, a in enumerate(alphas):
                for src in sources:
                    all_flags[a][src].append(flags_src[src][:, idx])

            # compute alpha/src summaries
            recalc_dataset_alpha[ds] = {}
            for idx, a in enumerate(alphas):
                recalc_dataset_alpha[ds][a] = {}
                for src in list(sources) + list(variants):
                    flags = flags_src[src][:, idx]
                    false_revs = int(np.sum(flags))
                    w_upper = independent_wilson_upper_bound(false_revs, n_train)
                    recalc_dataset_alpha[ds][a][src] = (false_revs, n_train, w_upper)

                    key = (a, src)
                    if key not in recalc_global:
                        recalc_global[key] = [0, 0]
                    recalc_global[key][0] += false_revs
                    recalc_global[key][1] += n_train

    # 2. Compare dataset alpha summary
    summary_file = output_dir / "04_dataset_alpha_summary.csv"
    with summary_file.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row["dataset"]
                a = float(row["alpha"])
                src = row["source_or_variant"]

                exp_false_revs = int(row["false_reviews"])
                exp_runs = int(row["runs"])
                exp_rate = float(row["rate"])
                exp_w_upper = float(row["wilson_upper_bound"])

                ref_false_revs, ref_runs, ref_w_upper = recalc_dataset_alpha[ds][a][src]

                assert exp_false_revs == ref_false_revs, f"Mismatch false_reviews for {ds} {a} {src}"
                assert exp_runs == ref_runs, f"Mismatch runs for {ds} {a} {src}"
                assert abs(exp_w_upper - ref_w_upper) < 1e-5, f"Mismatch Wilson upper for {ds} {a} {src}: {exp_w_upper} vs {ref_w_upper}"
    print("04_dataset_alpha_summary.csv: VERIFIED (100% Match)")

    # 3. Compare global summary
    global_file = output_dir / "05_global_summary.csv"
    with global_file.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = float(row["alpha"])
                src = row["source_or_variant"]
                exp_false_revs = int(row["false_reviews"])
                exp_runs = int(row["runs"])
                exp_w_upper = float(row["wilson_upper_bound"])

                ref_false_revs, ref_runs = recalc_global[(a, src)]
                ref_w_upper = independent_wilson_upper_bound(ref_false_revs, ref_runs)

                assert exp_false_revs == ref_false_revs, f"Mismatch global false_reviews for {a} {src}"
                assert exp_runs == ref_runs, f"Mismatch global runs for {a} {src}"
                assert abs(exp_w_upper - ref_w_upper) < 1e-5, f"Mismatch global Wilson upper for {a} {src}"
    print("05_global_summary.csv: VERIFIED (100% Match)")

    # 4. Compare sensitivity
    sensitivity_file = output_dir / "06_risk_target_sensitivity.csv"
    with sensitivity_file.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = float(row["alpha"])
                var = row["variant"]
                target = float(row["risk_target"])
                exp_passed = int(row["passed_datasets"])

                # recount passed
                ref_passed = 0
                for ds, alpha_map in recalc_dataset_alpha.items():
                    _, _, w_upper = alpha_map[a][var]
                    if w_upper <= target:
                        ref_passed += 1
                assert exp_passed == ref_passed, f"Mismatch sensitivity count for {a} {var} {target}: expected {exp_passed}, got {ref_passed}"
    print("06_risk_target_sensitivity.csv: VERIFIED (100% Match)")

    # 5. Compare source dependence
    dep_file = output_dir / "07_source_dependence.csv"
    with dep_file.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = float(row["alpha"])
                s1 = row["source_1"]
                s2 = row["source_2"]
                exp_phi = float(row["phi_correlation"])
                exp_jaccard = float(row["jaccard_similarity"])

                f1 = np.concatenate(all_flags[a][s1]) if all_flags[a][s1] else np.array([], dtype=bool)
                f2 = np.concatenate(all_flags[a][s2]) if all_flags[a][s2] else np.array([], dtype=bool)

                tp = np.sum(f1 & f2)
                fp = np.sum(f1 & ~f2)
                fn = np.sum(~f1 & f2)
                tn = np.sum(~f1 & ~f2)

                union = tp + fp + fn
                ref_jaccard = tp / union if union > 0 else 1.0

                denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                ref_phi = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

                assert abs(exp_phi - ref_phi) < 1e-5, f"Mismatch dependence phi for {a} {s1} {s2}"
                assert abs(exp_jaccard - ref_jaccard) < 1e-5, f"Mismatch dependence jaccard for {a} {s1} {s2}"
    print("07_source_dependence.csv: VERIFIED (100% Match)")

    # Generate VERIFICATION_REPORT.md
    report_file = output_dir / "VERIFICATION_REPORT.md"
    with report_file.open("w") as f:
        f.write("# Exp164 Verification Report\n\n")
        f.write("This report validates the outputs of the Exp164 risk audit run.\n\n")
        f.write("## Summary of Checks\n\n")
        f.write(f"- **Total manifest entries checked:** {len(manifest_entries)}\n")
        f.write(f"- **Eligible (computed) caches verified:** {computed_count}\n")
        f.write(f"- **Abstained datasets count:** {abstained_count}\n")
        f.write(f"- **Total normal runs analyzed:** {total_runs}\n")
        f.write("- **Cache File SHA256 integrity check:** PASS (All hashes match the manifest)\n")
        f.write("- **Evaluation field safety check:** PASS (All caches verified contains_test_values=false, contains_test_labels=false, contains_evaluation_metrics=false)\n")
        f.write("- **Dataset summary values check:** PASS (04_dataset_alpha_summary.csv matches independent calculation)\n")
        f.write("- **Global summary values check:** PASS (05_global_summary.csv matches independent calculation)\n")
        f.write("- **Risk target sensitivity check:** PASS (06_risk_target_sensitivity.csv matches independent calculation)\n")
        f.write("- **Source dependence metrics check:** PASS (07_source_dependence.csv matches independent calculation)\n\n")
        f.write("## Conclusion\n\n")
        f.write("The Exp164 result caches and aggregate metrics match this independent reaggregation. No TEST access was detected in the audited Exp164 data path; this does not establish end-to-end provenance or prospective validity.\n")

    print(f"Verification report created successfully at {report_file}")
    sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 independent_reaggregate_exp164.py <output_dir>")
        sys.exit(1)
    verify_and_reaggregate(sys.argv[1])
