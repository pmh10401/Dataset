from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
import unittest

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    canonical_result_sha256,
)
from experiments.exp417_causal_multihorizon_shadow.reporting import (
    TrustedCheckpointProvenance,
)
from experiments.exp417_causal_multihorizon_shadow.tests.test_reporting import (
    fixture_summary,
)
from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    BenchmarkSelectionProof,
    ParallelRun,
    RuntimeBenchmark,
    RuntimeTrial,
    ShardAssignment,
    ShardPlan,
    WorkerCompletion,
    benchmark_selection_proof_payload,
)


EXACT_FILES = (
    "exp419_shadow_predictions.jsonl",
    "exp419_dataset_qualification.csv",
    "exp419_oof_normal_metrics.csv",
    "exp419_synthetic_safety.csv",
    "exp419_backend_benchmark.json",
    "exp419_parallel_runtime_benchmark.json",
    "exp419_frozen_config.json",
    "exp419_shard_plan.json",
    "EXP419_RESULT_REPORT_KO.md",
    "MANIFEST.json",
)


def fixture_inputs():
    from experiments.exp419_deterministic_cpu_parallel.reporting import (
        _derive_exp417_provenance,
    )

    summary = fixture_summary()
    population = summary.context.datasets
    assignments = (
        ShardAssignment(0, population[::2], len(population[::2])),
        ShardAssignment(1, population[1::2], len(population[1::2])),
    )
    plan = ShardPlan(2, assignments, population)
    result_digests = {
        result.dataset: canonical_result_sha256(result)
        for result in summary.results
    }
    record_digests = {
        dataset: sha256(
            f"canonical-record:{digest}".encode("ascii")
        ).hexdigest()
        for dataset, digest in result_digests.items()
    }
    provenance = _derive_exp417_provenance(summary)
    trusted = TrustedCheckpointProvenance(
        checkpoint_provenance_sha256=sha256(
            canonical_json_bytes(provenance.to_payload())
        ).hexdigest(),
        result_evidence_sha256=result_digests,
        checkpoint_payload_sha256=record_digests,
    )
    completions = tuple(
        WorkerCompletion(
            assignment.shard_index,
            1000 + assignment.shard_index,
            assignment.datasets,
            {name: record_digests[name] for name in assignment.datasets},
            20.0 + assignment.shard_index,
            1_000_000 + assignment.shard_index,
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
            "completed",
            "",
        )
        for assignment in assignments
    )
    run = ParallelRun(
        plan,
        completions,
        record_digests,
        {
            "run_summary": summary,
            "plan_sha256": plan.sha256,
            "aggregate_peak_rss_bytes": 2_000_001,
        },
        True,
        "completed",
    )
    benchmark_datasets = tuple(sorted(population[:3]))
    selection_proof = BenchmarkSelectionProof(
        anchor_datasets=benchmark_datasets[:2],
        added_datasets=benchmark_datasets[2:],
        selected_datasets=benchmark_datasets,
        proxy_headroom_gate=2.5,
        total_cost=100,
        shard_loads={
            2: (50, 50),
            4: (40, 20, 20, 20),
            6: (40, 12, 12, 12, 12, 12),
        },
        proxy_headrooms={2: 2.0, 4: 2.5, 6: 2.5},
    )
    trials = tuple(
        RuntimeTrial(
            worker_count,
            {1: 100.0, 2: 40.0, 4: 50.0, 6: 60.0}[worker_count],
            2_000_000 * worker_count,
            True,
            True,
            "",
            benchmark_datasets,
            sha256(f"trial:{worker_count}".encode()).hexdigest(),
        )
        for worker_count in FROZEN_CONFIG.worker_candidates
    )
    benchmark = RuntimeBenchmark(
        trials,
        2,
        "qualified_parallel_runtime",
        benchmark_datasets,
        selection_proof,
    )
    return run, benchmark, plan, trusted


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            render_exp419_payloads,
        )

        cls.parallel_run, cls.benchmark, cls.plan, cls.trusted = fixture_inputs()
        cls.payloads = dict(
            render_exp419_payloads(
                cls.parallel_run, cls.benchmark, cls.plan, cls.trusted
            )
        )

    def test_renders_exact_ten_bound_label_free_files(self) -> None:
        self.assertEqual(tuple(self.payloads), EXACT_FILES)
        predictions = [
            json.loads(line)
            for line in self.payloads[EXACT_FILES[0]].splitlines()
        ]
        forbidden = {
            "ground_truth", "label", "labels", "tp", "fp", "fn", "tn",
            "precision", "recall", "f1", "mean_combined_f1",
            "actual_anomaly_location", "human_review",
        }
        self.assertTrue(predictions)
        self.assertFalse(
            forbidden
            & {
                key.casefold()
                for row in predictions
                for key in row
            }
        )

        config = json.loads(self.payloads["exp419_frozen_config.json"])
        self.assertEqual(config["experiment_id"], "Exp419")
        self.assertEqual(
            config["runtime_config"]["worker_candidates"], [1, 2, 4, 6]
        )
        self.assertEqual(
            config["exp417_scientific_evidence"]["frozen_config"],
            json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG.exp417_config))),
        )
        shard = json.loads(self.payloads["exp419_shard_plan.json"])
        self.assertEqual(shard["sha256"], self.plan.sha256)
        runtime = json.loads(
            self.payloads["exp419_parallel_runtime_benchmark.json"]
        )
        self.assertEqual(runtime["selected_worker_count"], 2)
        self.assertEqual(
            runtime["selection_proof"],
            benchmark_selection_proof_payload(
                self.benchmark.selection_proof
            ),
        )
        self.assertEqual(
            runtime["selection_proof"]["sha256"],
            self.benchmark.selection_proof.sha256,
        )
        self.assertEqual(runtime["sha256"], self.benchmark.sha256)
        self.assertEqual(
            [trial["worker_count"] for trial in runtime["trials"]],
            [1, 2, 4, 6],
        )
        report = self.payloads["EXP419_RESULT_REPORT_KO.md"].decode()
        self.assertIn("Shadow-only", report)
        self.assertIn("label-free", report)
        self.assertIn("lane 승격 없음", report)
        manifest = json.loads(self.payloads["MANIFEST.json"])
        self.assertEqual(
            set(manifest["artifact_sha256"]),
            set(EXACT_FILES) - {"MANIFEST.json"},
        )
        self.assertEqual(manifest["git_commit"], "a" * 40)
        self.assertEqual(
            manifest["trusted_checkpoint_provenance_sha256"],
            self.trusted.checkpoint_provenance_sha256,
        )
        self.assertEqual(manifest["benchmark_sha256"], self.benchmark.sha256)
        self.assertEqual(manifest["shard_plan_sha256"], self.plan.sha256)
        self.assertEqual(manifest["selected_worker_count"], 2)
        self.assertEqual(manifest["aggregate_peak_rss_bytes"], 2_000_001)
        self.assertEqual(
            manifest["canonical_checkpoint_record_sha256"],
            dict(self.trusted.checkpoint_payload_sha256),
        )
        self.assertEqual(
            set(manifest["exp419_source_sha256"]),
            {
                "experiments/exp419_deterministic_cpu_parallel/planner.py",
                "experiments/exp419_deterministic_cpu_parallel/workers.py",
                "experiments/exp419_deterministic_cpu_parallel/coordinator.py",
                "experiments/exp419_deterministic_cpu_parallel/reporting.py",
                "experiments/exp419_deterministic_cpu_parallel/publication.py",
            },
        )
        self.assertEqual(
            manifest["exp419_source_roles"],
            {
                "scheduler":
                    "experiments/exp419_deterministic_cpu_parallel/planner.py",
                "worker":
                    "experiments/exp419_deterministic_cpu_parallel/workers.py",
                "merge":
                    "experiments/exp419_deterministic_cpu_parallel/coordinator.py",
                "reporting":
                    "experiments/exp419_deterministic_cpu_parallel/reporting.py",
                "publication":
                    "experiments/exp419_deterministic_cpu_parallel/publication.py",
            },
        )

    def test_verify_rejects_every_rendered_selection_proof_mutation(
        self,
    ) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            verify_exp419_payloads,
        )

        mutations = (
            lambda proof: proof["anchor_datasets"].__setitem__(
                0, "mutated-anchor"
            ),
            lambda proof: proof["added_datasets"].__setitem__(
                0, "mutated-addition"
            ),
            lambda proof: proof["selected_datasets"].reverse(),
            lambda proof: proof.update({"proxy_headroom_gate": 2.0}),
            lambda proof: proof.update({"total_cost": 101}),
            lambda proof: proof["shard_loads"].__setitem__(
                "4", [40, 20, 19, 21]
            ),
            lambda proof: proof["proxy_headrooms"].__setitem__("4", 2.6),
            lambda proof: proof.update({"sha256": "0" * 64}),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                runtime = json.loads(
                    self.payloads[
                        "exp419_parallel_runtime_benchmark.json"
                    ]
                )
                mutation(runtime["selection_proof"])
                mutated = dict(self.payloads)
                mutated["exp419_parallel_runtime_benchmark.json"] = (
                    canonical_json_bytes(runtime) + b"\n"
                )
                with self.assertRaises(ValueError):
                    verify_exp419_payloads(
                        mutated,
                        trusted_checkpoint=self.trusted,
                        expected_selection_proof=
                            self.benchmark.selection_proof,
                    )

    def test_verify_rejects_coherent_alternate_selection_proof_bundle(
        self,
    ) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            render_exp419_payloads,
            verify_exp419_payloads,
        )

        datasets = self.benchmark.benchmark_datasets
        alternate_proof = BenchmarkSelectionProof(
            anchor_datasets=(datasets[0], datasets[2]),
            added_datasets=(datasets[1],),
            selected_datasets=datasets,
            proxy_headroom_gate=self.benchmark.selection_proof.proxy_headroom_gate,
            total_cost=self.benchmark.selection_proof.total_cost,
            shard_loads=self.benchmark.selection_proof.shard_loads,
            proxy_headrooms=self.benchmark.selection_proof.proxy_headrooms,
        )
        forged_benchmark = RuntimeBenchmark(
            self.benchmark.trials,
            self.benchmark.selected_worker_count,
            self.benchmark.status,
            datasets,
            alternate_proof,
        )
        forged_payloads = dict(
            render_exp419_payloads(
                self.parallel_run,
                forged_benchmark,
                self.plan,
                self.trusted,
            )
        )

        self.assertIsNone(
            verify_exp419_payloads(
                forged_payloads,
                trusted_checkpoint=self.trusted,
                expected_selection_proof=alternate_proof,
            )
        )
        with self.assertRaisesRegex(ValueError, "selection proof"):
            verify_exp419_payloads(
                forged_payloads,
                trusted_checkpoint=self.trusted,
                expected_selection_proof=self.benchmark.selection_proof,
            )

    def test_render_recomputes_runtime_qualification_and_rejects_forged_selection(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            render_exp419_payloads,
        )

        selected = self.benchmark.trials[1]
        attacks = (
            replace(selected, parity=False),
            replace(selected, finite=False),
            replace(selected, fallback="worker fallback"),
            replace(selected, peak_rss_bytes=FROZEN_CONFIG.memory_limit_bytes + 1),
            replace(selected, elapsed_seconds=60.0),
        )
        for trial in attacks:
            trials = (
                self.benchmark.trials[0],
                trial,
                *self.benchmark.trials[2:],
            )
            forged = RuntimeBenchmark(
                trials,
                2,
                "qualified_parallel_runtime",
                self.benchmark.benchmark_datasets,
                self.benchmark.selection_proof,
            )
            with self.subTest(trial=trial), self.assertRaisesRegex(
                ValueError, "recomputed|qualification"
            ):
                render_exp419_payloads(
                    self.parallel_run, forged, self.plan, self.trusted
                )

        faster_four = replace(
            self.benchmark.trials[2], elapsed_seconds=20.0
        )
        forged = RuntimeBenchmark(
            (
                self.benchmark.trials[0],
                self.benchmark.trials[1],
                faster_four,
                self.benchmark.trials[3],
            ),
            2,
            "qualified_parallel_runtime",
            self.benchmark.benchmark_datasets,
            self.benchmark.selection_proof,
        )
        with self.assertRaisesRegex(ValueError, "recomputed|qualification"):
            render_exp419_payloads(
                self.parallel_run, forged, self.plan, self.trusted
            )

    def test_render_rejects_thread_and_aggregate_memory_fabrication(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            render_exp419_payloads,
        )

        bad_completion = replace(
            self.parallel_run.completions[0],
            thread_environment={
                "OMP_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
        )
        completions = (
            bad_completion,
            *self.parallel_run.completions[1:],
        )
        bad_run = ParallelRun(
            self.plan,
            completions,
            self.parallel_run.record_digests,
            self.parallel_run.summary,
            True,
            "completed",
        )
        with self.assertRaisesRegex(ValueError, "thread"):
            render_exp419_payloads(
                bad_run, self.benchmark, self.plan, self.trusted
            )

        zero_run = ParallelRun(
            self.plan,
            self.parallel_run.completions,
            self.parallel_run.record_digests,
            {
                **dict(self.parallel_run.summary),
                "aggregate_peak_rss_bytes": 0,
            },
            True,
            "completed",
        )
        with self.assertRaisesRegex(ValueError, "memory|RSS"):
            render_exp419_payloads(
                zero_run, self.benchmark, self.plan, self.trusted
            )

    def test_verify_recomputes_runtime_and_rejects_thread_memory_fabrication(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            _benchmark_payload,
            verify_exp419_payloads,
        )

        bad_trial = replace(self.benchmark.trials[1], parity=False)
        forged_benchmark = RuntimeBenchmark(
            (
                self.benchmark.trials[0],
                bad_trial,
                *self.benchmark.trials[2:],
            ),
            2,
            "qualified_parallel_runtime",
            self.benchmark.benchmark_datasets,
            self.benchmark.selection_proof,
        )
        runtime_mutation = dict(self.payloads)
        runtime_mutation["exp419_parallel_runtime_benchmark.json"] = (
            canonical_json_bytes(_benchmark_payload(forged_benchmark)) + b"\n"
        )
        with self.assertRaisesRegex(ValueError, "recomputed|qualification"):
            verify_exp419_payloads(
                runtime_mutation,
                trusted_checkpoint=self.trusted,
                expected_selection_proof=self.benchmark.selection_proof,
            )

        for mutation in ("thread", "aggregate"):
            config = json.loads(self.payloads["exp419_frozen_config.json"])
            if mutation == "thread":
                config["parallel_run"]["completions"][0][
                    "thread_environment"
                ]["OMP_NUM_THREADS"] = "8"
            else:
                config["parallel_run"]["aggregate_peak_rss_bytes"] = 0
            mutated = dict(self.payloads)
            mutated["exp419_frozen_config.json"] = (
                canonical_json_bytes(config) + b"\n"
            )
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ValueError, "thread|RSS"
            ):
                verify_exp419_payloads(
                    mutated,
                    trusted_checkpoint=self.trusted,
                    expected_selection_proof=
                        self.benchmark.selection_proof,
                )

    def test_multi_attempt_resume_aggregate_uses_individual_peak_lower_bound(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            render_exp419_payloads,
            verify_exp419_payloads,
        )

        # Each shard maximum can come from a different resume attempt.  The
        # coordinator persists max(parent + workers) per attempt, not the sum
        # of maxima observed across unrelated attempts.
        aggregate = max(
            completion.peak_rss_bytes
            for completion in self.parallel_run.completions
        )
        self.assertLess(
            aggregate,
            sum(
                completion.peak_rss_bytes
                for completion in self.parallel_run.completions
            ),
        )
        resumed = ParallelRun(
            self.plan,
            self.parallel_run.completions,
            self.parallel_run.record_digests,
            {
                **dict(self.parallel_run.summary),
                "aggregate_peak_rss_bytes": aggregate,
            },
            True,
            "completed",
        )

        payloads = dict(
            render_exp419_payloads(
                resumed, self.benchmark, self.plan, self.trusted
            )
        )
        verify_exp419_payloads(
            payloads,
            trusted_checkpoint=self.trusted,
            expected_selection_proof=self.benchmark.selection_proof,
        )
        self.assertEqual(
            json.loads(payloads["MANIFEST.json"])[
                "aggregate_peak_rss_bytes"
            ],
            aggregate,
        )

    def test_verifier_rejects_each_payload_mutation(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            verify_exp419_payloads,
        )

        verify_exp419_payloads(
            self.payloads,
            trusted_checkpoint=self.trusted,
            expected_selection_proof=self.benchmark.selection_proof,
        )
        for name in EXACT_FILES:
            mutated = dict(self.payloads)
            mutated[name] = mutated[name] + b" "
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_exp419_payloads(
                    mutated,
                    trusted_checkpoint=self.trusted,
                    expected_selection_proof=
                        self.benchmark.selection_proof,
                )

    def test_semantic_candidate_worker_shard_model_and_evaluation_attacks_fail(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            verify_exp419_payloads,
        )

        attacks: list[tuple[str, bytes]] = []
        attacks.append(
            (
                "exp419_shadow_predictions.jsonl",
                self.payloads["exp419_shadow_predictions.jsonl"]
                + self.payloads["exp419_shadow_predictions.jsonl"].splitlines(
                    keepends=True
                )[0],
            )
        )
        dataset_lines = self.payloads[
            "exp419_dataset_qualification.csv"
        ].splitlines(keepends=True)
        attacks.append(
            (
                "exp419_dataset_qualification.csv",
                b"".join(dataset_lines[:-1]),
            )
        )
        prediction = json.loads(
            self.payloads["exp419_shadow_predictions.jsonl"].splitlines()[0]
        )
        prediction["precision"] = 1.0
        attacks.append(
            (
                "exp419_shadow_predictions.jsonl",
                canonical_json_bytes(prediction) + b"\n",
            )
        )
        runtime = json.loads(
            self.payloads["exp419_parallel_runtime_benchmark.json"]
        )
        runtime["selected_worker_count"] = 4
        attacks.append(
            (
                "exp419_parallel_runtime_benchmark.json",
                canonical_json_bytes(runtime) + b"\n",
            )
        )
        shard = json.loads(self.payloads["exp419_shard_plan.json"])
        shard["assignments"][0]["datasets"][0], shard["assignments"][1]["datasets"][0] = (
            shard["assignments"][1]["datasets"][0],
            shard["assignments"][0]["datasets"][0],
        )
        attacks.append(
            ("exp419_shard_plan.json", canonical_json_bytes(shard) + b"\n")
        )
        config = json.loads(self.payloads["exp419_frozen_config.json"])
        config["exp417_scientific_evidence"]["provenance"][
            "model_contract_sha256"
        ] = "f" * 64
        attacks.append(
            ("exp419_frozen_config.json", canonical_json_bytes(config) + b"\n")
        )
        for name, replacement in attacks:
            mutated = dict(self.payloads)
            mutated[name] = replacement
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_exp419_payloads(
                    mutated,
                    trusted_checkpoint=self.trusted,
                    expected_selection_proof=
                        self.benchmark.selection_proof,
                )

    def test_source_digest_manifest_missing_tampered_and_wrong_values_fail(self) -> None:
        from experiments.exp419_deterministic_cpu_parallel.reporting import (
            verify_exp419_payloads,
        )

        for mutation in ("missing", "wrong", "extra"):
            manifest = json.loads(self.payloads["MANIFEST.json"])
            sources = manifest["exp419_source_sha256"]
            first = next(iter(sources))
            if mutation == "missing":
                sources.pop(first)
            elif mutation == "wrong":
                sources[first] = "f" * 64
            else:
                sources["experiments/exp419_deterministic_cpu_parallel/alias.py"] = (
                    "e" * 64
                )
            mutated = dict(self.payloads)
            mutated["MANIFEST.json"] = canonical_json_bytes(manifest) + b"\n"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_exp419_payloads(
                    mutated,
                    trusted_checkpoint=self.trusted,
                    expected_selection_proof=
                        self.benchmark.selection_proof,
                )
