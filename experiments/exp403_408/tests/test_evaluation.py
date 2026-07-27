from __future__ import annotations

import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from experiments.exp403_408.artifact_io import write_jsonl_atomic
from experiments.exp403_408.contracts import ALPHAS, FrozenConfig, PredictionRow
from experiments.exp403_408.foundation import FrozenLaneUniverse, load_frozen_lanes
from experiments.exp403_408.prediction import (
    HARD_CONFIDENCE_NAME,
    PREDICTION_MANIFEST_NAME,
    freeze_provenance_contract,
    load_expected_provenance,
    write_prediction,
)

from experiments.exp403_408.evaluation import (
    EvaluationLabels,
    demotion_gate,
    evaluate_exp405,
    evaluate_hard_confidence,
)
from experiments.exp403_408.independent_reaggregate import independent_reaggregate


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def reporting_fixture(output: Path) -> Path:
    """Create only frozen publications consumed by the independent auditor."""
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows = [
        {
            "dataset": "alpha_normal_0",
            "candidate_index": 0,
            "confidence_by_alpha": {str(alpha): "recheck" for alpha in ALPHAS},
        },
        {
            "dataset": "beta_normal_0",
            "candidate_index": 1,
            "confidence_by_alpha": {str(alpha): "high" for alpha in ALPHAS},
        },
    ]
    prediction_path = output / HARD_CONFIDENCE_NAME
    _write_jsonl(prediction_path, prediction_rows)
    prediction_sha256 = sha256(prediction_path.read_bytes()).hexdigest()
    (output / PREDICTION_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "hard_confidence_file": HARD_CONFIDENCE_NAME,
                "artifact_hashes": {"prediction": prediction_sha256},
                "frozen_hard_lanes": {
                    "alpha_normal_0": [0],
                    "beta_normal_0": [1],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output / "exp405_demotion_evaluation.json").write_text(
        json.dumps(
            {
                "recheck_count": 1,
                "false_positive_count": 1,
                "true_positive_count": 0,
                "false_positive_fraction": 1.0,
                "baseline_true_positive_retention": 1.0,
                "train_stability": 0.99,
                "false_positive_fraction_wilson_interval": [0.20654931437723745, 1.0],
                "false_positive_fraction_bootstrap_interval": [1.0, 1.0],
                "baseline_true_positive_retention_bootstrap_interval": [1.0, 1.0],
                "dataset_concentration": 1.0,
                "lineage_concentration": 1.0,
                "lane_parity": True,
                "passes_research_gate": True,
                "operational_policy_changed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output / "exp405_demotion_dataset_metrics.csv").write_text(
        "dataset,lineage,train_stability,baseline_hard_tp,recheck_count,recheck_false_positive_count,recheck_true_positive_count\n"
        "alpha_normal_0,alpha,0.99,0,1,1,0\n"
        "beta_normal_0,beta,0.95,1,0,0,0\n",
        encoding="utf-8",
    )
    (output / "exp405_demotion_candidate_evaluation.csv").write_text(
        "dataset,candidate_index,is_recheck,is_true_anomaly\n"
        "alpha_normal_0,0,1,0\n"
        "beta_normal_0,1,0,1\n",
        encoding="utf-8",
    )
    audit_rows = [
        {"dataset": "alpha_normal_0", "train_size": 20},
        {"dataset": "beta_normal_0", "train_size": 50},
    ]
    _write_jsonl(output / "exp403_feature_audit.jsonl", audit_rows)
    candidates = {
        "tri_view_only": {"0.005": [{"candidate_index": 2}]},
        "synthetic_only": {"0.005": [{"candidate_index": 2}]},
        "tri_view_and_synthetic": {"0.005": [{"candidate_index": 2}]},
        "one_new_plus_existing_independent": {"0.005": [{"candidate_index": 2}]},
        "two_new_plus_existing_independent": {"0.005": [{"candidate_index": 2}]},
    }
    consensus_rows = [
        {
            "dataset": "alpha_normal_0",
            "consensus": {
                "hard_indices": [0],
                "standard_indices": [2],
                "priority_indices": [3],
                "candidates_by_variant": candidates,
            },
        },
        {
            "dataset": "beta_normal_0",
            "consensus": {
                "hard_indices": [1],
                "standard_indices": [],
                "priority_indices": [3],
                "candidates_by_variant": {
                    name: {"0.005": []} for name in candidates
                },
            },
        },
    ]
    _write_jsonl(output / "exp408_consensus_results.jsonl", consensus_rows)
    header = (
        "dataset,lineage,train_size,variant,alpha,candidate_count,tp,fp,precision,"
        "fn_recovered,zero_f1_before,zero_f1_after,standard_overlap,priority_overlap,"
        "review_outside_existing,baseline_hard_tp,baseline_hard_fp,baseline_hard_fn\n"
    )
    rows = []
    for variant in candidates:
        rows.extend(
            (
                f"alpha_normal_0,alpha,20,{variant},0.005,1,1,0,1.0,1,1,0,1,0,0,0,1,1\n",
                f"beta_normal_0,beta,50,{variant},0.005,0,0,0,0.0,0,0,0,0,0,0,1,0,0\n",
            )
        )
    (output / "exp408_variant_metrics.csv").write_text(header + "".join(rows), encoding="utf-8")
    (output / "exp408_variant_candidate_evaluation.csv").write_text(
        "dataset,variant,alpha,candidate_index,is_true_anomaly\n"
        + "".join(
            f"alpha_normal_0,{variant},0.005,2,1\n" for variant in candidates
        ),
        encoding="utf-8",
    )
    (output / "exp403_408_frozen_test_labels.csv").write_text(
        "dataset,candidate_index,is_true_anomaly\n"
        "alpha_normal_0,0,0\nalpha_normal_0,1,0\nalpha_normal_0,2,1\nalpha_normal_0,3,0\n"
        "beta_normal_0,0,0\nbeta_normal_0,1,1\nbeta_normal_0,2,0\nbeta_normal_0,3,0\n",
        encoding="utf-8",
    )
    frozen = (
        "exp404_prediction_manifest.json", "exp404_hard_confidence.jsonl",
        "exp405_demotion_evaluation.json", "exp405_demotion_dataset_metrics.csv",
        "exp405_demotion_candidate_evaluation.csv", "exp403_feature_audit.jsonl",
        "exp408_consensus_results.jsonl", "exp408_variant_metrics.csv",
        "exp408_variant_candidate_evaluation.csv", "exp403_408_frozen_test_labels.csv",
    )
    hashes = {name: sha256((output / name).read_bytes()).hexdigest() for name in frozen}
    (output / "exp403_408_evaluation_input_manifest.json").write_text(
        json.dumps({"artifact_sha256": hashes, "provenance_contract_sha256": "fixture-pinned"}, sort_keys=True),
        encoding="utf-8",
    )
    hashes["exp403_408_evaluation_input_manifest.json"] = sha256(
        (output / "exp403_408_evaluation_input_manifest.json").read_bytes()
    ).hexdigest()
    (output / "exp403_408_evaluation_publication_manifest.json").write_text(
        json.dumps({"artifact_sha256": hashes}, sort_keys=True), encoding="utf-8"
    )
    return output


def refresh_evaluation_hashes(output: Path, *names: str) -> None:
    """Simulate a forged official publication whose manifests were rehashed."""
    input_path = output / "exp403_408_evaluation_input_manifest.json"
    input_manifest = json.loads(input_path.read_text(encoding="utf-8"))
    for name in names:
        input_manifest["artifact_sha256"][name] = sha256((output / name).read_bytes()).hexdigest()
    input_path.write_text(json.dumps(input_manifest, sort_keys=True), encoding="utf-8")
    publication_path = output / "exp403_408_evaluation_publication_manifest.json"
    publication_manifest = json.loads(publication_path.read_text(encoding="utf-8"))
    for name in names:
        publication_manifest["artifact_sha256"][name] = sha256((output / name).read_bytes()).hexdigest()
    publication_manifest["artifact_sha256"][input_path.name] = sha256(input_path.read_bytes()).hexdigest()
    publication_path.write_text(json.dumps(publication_manifest, sort_keys=True), encoding="utf-8")


class LabelLoaderSpy:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, datasets: set[str]) -> dict[str, EvaluationLabels]:
        self.calls += 1
        return {
            dataset: EvaluationLabels(
                test_labels=("normal",) * 8,
                train_oof_normal_stability=0.991,
                lineage=dataset.split("_normal_", 1)[0],
            )
            for dataset in datasets
        }


def valid_rows(universe: FrozenLaneUniverse) -> list[PredictionRow]:
    return [
        PredictionRow(
            dataset=dataset,
            candidate_index=index,
            scores_by_alpha={str(alpha): 0.5 for alpha in ALPHAS},
            confidence_by_alpha={str(alpha): "recheck" for alpha in ALPHAS},
            eligibility="classical_committee",
        )
        for dataset, indices in universe.hard_lanes().items()
        for index in indices
    ]


def provenance_contract(root: Path) -> tuple[Path, str]:
    sources = root / "sources"
    sources.mkdir(exist_ok=True)
    artifacts = {
        "feature_artifacts": {"exp403_features": sources / "exp403_features.npz"},
        "input_artifacts": {"exp291_lanes": sources / "exp291_lanes.jsonl"},
        "baseline_artifacts": {"exp292_baseline": sources / "exp292_baseline.jsonl"},
        "config_artifacts": {"frozen_config": sources / "config.json"},
        "code_artifacts": {"exp404_committee": sources / "classical_committee.py"},
    }
    for category in artifacts.values():
        for artifact in category.values():
            artifact.write_text(artifact.name, encoding="utf-8")
    path = root / "trusted" / "exp404_provenance_contract.json"
    return path, freeze_provenance_contract(path, **artifacts)


def rewrite_prediction_rows(output: Path, rows: list[dict[str, object]]) -> None:
    prediction_path = output / HARD_CONFIDENCE_NAME
    manifest_path = output / PREDICTION_MANIFEST_NAME
    prediction_sha256 = write_jsonl_atomic(prediction_path, rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["prediction"] = prediction_sha256
    manifest["rows"] = len(rows)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def expected_identities(
    prediction: list[dict[str, object]],
) -> frozenset[tuple[str, int]]:
    return frozenset(
        (str(row["dataset"]), int(row["candidate_index"])) for row in prediction
    )


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.universe = load_frozen_lanes()

    def test_labels_are_not_loaded_before_prediction_hash_verification(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path, contract_sha256 = provenance_contract(root)
            provenance = load_expected_provenance(
                contract_path, expected_sha256=contract_sha256
            )
            output = root / "prediction"
            write_prediction(
                valid_rows(self.universe),
                output,
                config=FrozenConfig(),
                frozen_lane_universe=self.universe,
                expected_provenance=provenance,
            )
            output.joinpath("exp404_hard_confidence.jsonl").write_text(
                "tampered\n", encoding="utf-8"
            )
            loader = LabelLoaderSpy()

            with self.assertRaisesRegex(ValueError, "SHA256"):
                evaluate_exp405(
                    output,
                    loader,
                    frozen_lane_universe=self.universe,
                    provenance_contract_path=contract_path,
                    expected_provenance_sha256=contract_sha256,
                )

            self.assertEqual(loader.calls, 0)

    def test_labels_are_not_loaded_before_external_anchor_verification(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path, contract_sha256 = provenance_contract(root)
            provenance = load_expected_provenance(
                contract_path, expected_sha256=contract_sha256
            )
            output = root / "prediction"
            write_prediction(
                valid_rows(self.universe),
                output,
                config=FrozenConfig(),
                frozen_lane_universe=self.universe,
                expected_provenance=provenance,
            )
            loader = LabelLoaderSpy()

            with self.assertRaisesRegex(ValueError, "SHA256"):
                evaluate_exp405(
                    output,
                    loader,
                    frozen_lane_universe=self.universe,
                    provenance_contract_path=contract_path,
                    expected_provenance_sha256="0" * 64,
                )

            self.assertEqual(loader.calls, 0)

    def test_success_gate_requires_all_three_conditions(self) -> None:
        decision = demotion_gate(fp_fraction=.81, tp_retention=.995, train_stability=.991)

        self.assertTrue(decision.passes_research_gate)
        self.assertFalse(demotion_gate(.79, .995, .991).passes_research_gate)
        self.assertFalse(demotion_gate(.81, .989, .991).passes_research_gate)
        self.assertFalse(demotion_gate(.81, .995, .989).passes_research_gate)

    def test_omitted_or_substituted_frozen_lane_stops_before_labels(self) -> None:
        for mutation in ("omit", "substitute"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                contract_path, contract_sha256 = provenance_contract(root)
                provenance = load_expected_provenance(
                    contract_path, expected_sha256=contract_sha256
                )
                output = root / "prediction"
                write_prediction(
                    valid_rows(self.universe),
                    output,
                    config=FrozenConfig(),
                    frozen_lane_universe=self.universe,
                    expected_provenance=provenance,
                )
                rows = [
                    json.loads(line)
                    for line in output.joinpath(HARD_CONFIDENCE_NAME)
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                if mutation == "omit":
                    rows.pop()
                else:
                    rows[-1]["candidate_index"] = int(rows[-1]["candidate_index"]) + 100_000
                rewrite_prediction_rows(output, rows)
                loader = LabelLoaderSpy()

                with self.assertRaisesRegex(ValueError, "frozen Hard.*coverage"):
                    evaluate_exp405(
                        output,
                        loader,
                        frozen_lane_universe=self.universe,
                        provenance_contract_path=contract_path,
                        expected_provenance_sha256=contract_sha256,
                    )

                self.assertEqual(loader.calls, 0)

    def test_counterfactual_reports_intervals_concentration_and_lane_parity(
        self,
    ) -> None:
        prediction = [
            {
                "dataset": "alpha_normal_0",
                "candidate_index": 0,
                "confidence_by_alpha": {str(alpha): "recheck" for alpha in ALPHAS},
            },
            {
                "dataset": "beta_normal_0",
                "candidate_index": 1,
                "confidence_by_alpha": {str(alpha): "recheck" for alpha in ALPHAS},
            },
            {
                "dataset": "gamma_normal_0",
                "candidate_index": 2,
                "confidence_by_alpha": {str(alpha): "high" for alpha in ALPHAS},
            },
        ]
        labels = {
            "alpha_normal_0": EvaluationLabels(("normal",), 0.99, "alpha"),
            "beta_normal_0": EvaluationLabels(("normal", "anomaly"), 0.98, "beta"),
            "gamma_normal_0": EvaluationLabels(
                ("normal", "normal", "anomaly"), 0.97, "gamma"
            ),
        }

        result = evaluate_hard_confidence(
            prediction,
            labels,
            expected_hard_identities=expected_identities(prediction),
        )

        self.assertEqual(result.recheck_count, 2)
        self.assertEqual((result.false_positive_count, result.true_positive_count), (1, 1))
        self.assertAlmostEqual(result.false_positive_fraction, 0.5)
        self.assertAlmostEqual(result.baseline_true_positive_retention, 0.5)
        self.assertAlmostEqual(result.train_stability, 0.985)
        self.assertEqual(
            result.false_positive_fraction_wilson_interval,
            (0.09453120573423074, 0.9054687942657693),
        )
        self.assertLessEqual(
            result.false_positive_fraction_bootstrap_interval[0],
            result.false_positive_fraction,
        )
        self.assertGreaterEqual(
            result.false_positive_fraction_bootstrap_interval[1],
            result.false_positive_fraction,
        )
        self.assertAlmostEqual(result.dataset_concentration, 0.5)
        self.assertAlmostEqual(result.lineage_concentration, 0.5)
        self.assertTrue(result.lane_parity)
        self.assertFalse(result.passes_research_gate)
        self.assertFalse(result.operational_policy_changed)

    def test_counterfactual_requires_complete_label_metadata(self) -> None:
        prediction = [
            {
                "dataset": "alpha_normal_0",
                "candidate_index": 1,
                "confidence_by_alpha": {str(alpha): "recheck" for alpha in ALPHAS},
            }
        ]
        labels = {
            "alpha_normal_0": EvaluationLabels(("normal",), 0.99, "alpha"),
        }

        with self.assertRaisesRegex(ValueError, "label index"):
            evaluate_hard_confidence(
                prediction,
                labels,
                expected_hard_identities=expected_identities(prediction),
            )

    def test_counterfactual_rejects_missing_expected_hard_identity(self) -> None:
        prediction = [
            {
                "dataset": "alpha_normal_0",
                "candidate_index": 0,
                "confidence_by_alpha": {str(alpha): "recheck" for alpha in ALPHAS},
            }
        ]
        labels = {
            "alpha_normal_0": EvaluationLabels(("normal",), 0.99, "alpha"),
        }

        with self.assertRaisesRegex(ValueError, "frozen Hard.*coverage"):
            evaluate_hard_confidence(
                prediction,
                labels,
                expected_hard_identities=frozenset(
                    {("alpha_normal_0", 0), ("alpha_normal_0", 1)}
                ),
            )

    def test_counterfactual_rejects_invalid_binary_labels(self) -> None:
        prediction = [
            {
                "dataset": "alpha_normal_0",
                "candidate_index": 0,
                "confidence_by_alpha": {str(alpha): "recheck" for alpha in ALPHAS},
            }
        ]
        expected = expected_identities(prediction)
        for invalid in (None, "", float("nan"), float("inf"), "unknown", 2, -1):
            with self.subTest(invalid=repr(invalid)), self.assertRaisesRegex(
                ValueError, "TEST label"
            ):
                evaluate_hard_confidence(
                    prediction,
                    {"alpha_normal_0": EvaluationLabels((invalid,), 0.99, "alpha")},
                    expected_hard_identities=expected,
                )

    def test_counterfactual_rejects_missing_label_dataset(self) -> None:
        prediction = [
            {
                "dataset": "alpha_normal_0",
                "candidate_index": 0,
                "confidence_by_alpha": {str(alpha): "recheck" for alpha in ALPHAS},
            },
            {
                "dataset": "beta_normal_0",
                "candidate_index": 0,
                "confidence_by_alpha": {str(alpha): "recheck" for alpha in ALPHAS},
            },
        ]

        with self.assertRaisesRegex(ValueError, "dataset coverage"):
            evaluate_hard_confidence(
                prediction,
                {"alpha_normal_0": EvaluationLabels(("normal",), 0.99, "alpha")},
                expected_hard_identities=expected_identities(prediction),
            )

    def test_independent_reaggregation_matches_frozen_official_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = independent_reaggregate(reporting_fixture(Path(directory)))

        self.assertEqual(report.mismatch_count, 0)
        self.assertEqual(report.metrics["exp404"]["recheck_count"], 1)
        self.assertEqual(report.metrics["exp405"]["false_positive_count"], 1)
        self.assertEqual(
            report.metrics["exp408"]["tri_view_and_synthetic"]["0.005"]["fn_recovered"],
            1,
        )

    def test_independent_reaggregation_rejects_float_difference_above_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = reporting_fixture(Path(directory))
            path = output / "exp405_demotion_evaluation.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["train_stability"] = 0.99 + 1.1e-12
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA256|independent reaggregation mismatch"):
                independent_reaggregate(output)

    def test_independent_reaggregation_requires_candidate_level_evaluation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = reporting_fixture(Path(directory))
            (output / "exp408_variant_candidate_evaluation.csv").unlink()

            with self.assertRaisesRegex(ValueError, "candidate evaluation|frozen artifact"):
                independent_reaggregate(output)

    def test_independent_reaggregation_rejects_altered_baseline_fn_and_consensus_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = reporting_fixture(Path(directory))
            path = output / "exp408_variant_metrics.csv"
            text = path.read_text(encoding="utf-8").replace(",0,1,1\n", ",0,1,0\n", 1)
            path.write_text(text, encoding="utf-8")
            refresh_evaluation_hashes(output, path.name)
            with self.assertRaisesRegex(ValueError, "baseline_hard_fn"):
                independent_reaggregate(output)

        with tempfile.TemporaryDirectory() as directory:
            output = reporting_fixture(Path(directory))
            (output / "exp408_consensus_results.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                independent_reaggregate(output)

    def test_independent_reaggregation_rejects_rehashed_exp408_tp_integer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = reporting_fixture(Path(directory))
            path = output / "exp408_variant_metrics.csv"
            path.write_text(
                path.read_text(encoding="utf-8").replace(",1,1,0,1.0,1,1,0", ",1,0,0,1.0,1,1,0", 1),
                encoding="utf-8",
            )
            refresh_evaluation_hashes(output, path.name)

            with self.assertRaisesRegex(ValueError, r"\.tp"):
                independent_reaggregate(output)


if __name__ == "__main__":
    unittest.main()
