"""Independent semantic tests for Exp417's seven rendered payloads."""

from __future__ import annotations

import csv
from dataclasses import asdict
from hashlib import sha256
import io
import json
from types import MappingProxyType
import unittest

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointProvenance,
)
from experiments.exp417_causal_multihorizon_shadow.backend import (
    FROZEN_TASK7_BACKEND_AUDIT_SHA256,
)
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
    DatasetResult,
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    HorizonScore,
    RunSummary,
    SeedDecision,
    TestInstanceDecision,
)


def fixture_lanes() -> FrozenLaneSnapshot:
    return FrozenLaneSnapshot(
        hard={"dataset-0000": tuple(range(10_000, 12_218))},
        standard={"dataset-0000": tuple(range(20_000, 20_806))},
        priority={"dataset-0000": tuple(range(30_000, 30_031))},
        automatic_normal={"dataset-0000": tuple(range(40_000, 40_035))},
    )


def fixture_context() -> FrozenRunContext:
    datasets = tuple(f"dataset-{index:04d}" for index in range(1117))
    return FrozenRunContext(
        database_sha256=FROZEN_CONFIG.database_sha256,
        datasets=datasets,
        lanes=fixture_lanes(),
        test_identity_manifest={
            dataset: tuple(
                (index, 100)
                for index in range(100 if dataset == "dataset-0000" else 1)
            )
            for dataset in datasets
        },
        code_commit="a" * 40,
        runtime_versions={
            "numba": "0.66.0",
            "numpy": "2.3.5",
            "python": "3.14.0",
            "torch": "2.13.0",
        },
    )


def fixture_backend() -> BackendAudit:
    return BackendAudit(
        backend="cpu",
        candidate_identity=True,
        max_normalized_error=4.0948700075205124e-07,
        spearman=0.9999999999845686,
        speedup=0.6709759390840965,
        fallback=False,
        fallback_reason=None,
    )


def fixture_group() -> GroupQualification:
    return GroupQualification(
        length=100,
        series_count=100,
        status="qualified",
        oof_candidate_rate=0.01,
        wilson_upper=0.05,
        pooled_synthetic_sensitivity=0.70,
        per_type_synthetic_sensitivity={
            "amplitude_scaling": 0.50,
            "gradual_drift": 0.50,
            "level_shift": 0.50,
            "period_change": 0.50,
        },
        seed_agreement=True,
    )


def fixture_seed_decisions(*, candidate: bool) -> tuple[SeedDecision, ...]:
    scores = {
        1: HorizonScore(
            1, 2.5 if candidate else 0.25, 0.01 if candidate else 0.5,
            0.05 if candidate else 0.0,
        ),
        2: HorizonScore(
            2, 2.0 if candidate else 0.20, 0.01 if candidate else 0.5,
            0.05 if candidate else 0.0,
        ),
        10: HorizonScore(10, 0.5, 0.5, 0.0),
    }
    return tuple(
        SeedDecision(seed=seed, scores=scores, candidate=candidate)
        for seed in FROZEN_CONFIG.model_seeds
    )


def fixture_summary() -> RunSummary:
    context = fixture_context()
    group = fixture_group()
    backend = fixture_backend()
    results: list[DatasetResult] = []
    for dataset in context.datasets:
        contract = FrozenTrainContract(
            dataset=dataset,
            groups=(group,),
            backend=backend,
            train_identity_sha256=sha256(dataset.encode("utf-8")).hexdigest(),
        )
        decisions = tuple(
            TestInstanceDecision(
                dataset=dataset,
                instance_index=index,
                length=100,
                seed_decisions=fixture_seed_decisions(
                    candidate=dataset == "dataset-0000" and index == 0
                ),
            )
            for index, _ in context.test_identity_manifest[dataset]
        )
        evidence = DatasetDecisionEvidence(
            dataset=dataset,
            context=context,
            frozen_train_contract=contract,
            database_sha256=context.database_sha256,
            train_identity_sha256=contract.train_identity_sha256,
            decisions=decisions,
        )
        results.append(
            DatasetResult(
                dataset=dataset,
                test_size=context.test_sizes[dataset],
                group_count=1,
                groups=(group,),
                candidates=evidence.derived_candidates,
                decision_evidence=evidence,
                lanes=context.lanes,
                context=context,
                status="qualified",
            )
        )
    return RunSummary(
        results=tuple(results),
        total_test=sum(context.test_sizes.values()),
        publication_allowed=True,
        status="complete",
        context=context,
    )


def fixture_provenance() -> Exp417CheckpointProvenance:
    return Exp417CheckpointProvenance(
        git_commit="a" * 40,
        database_sha256=FROZEN_CONFIG.database_sha256,
        exp291_prediction_sha256=FROZEN_CONFIG.exp291_prediction_sha256,
        exp291_manifest_sha256=FROZEN_CONFIG.exp291_manifest_sha256,
        exp292_prediction_sha256=FROZEN_CONFIG.exp292_prediction_sha256,
        exp292_manifest_sha256=FROZEN_CONFIG.exp292_manifest_sha256,
        frozen_config_sha256=sha256(
            canonical_json_bytes(asdict(FROZEN_CONFIG))
        ).hexdigest(),
        preprocessing_contract_sha256="1" * 64,
        fold_contract_sha256="2" * 64,
        model_contract_sha256="3" * 64,
        model_seeds=FROZEN_CONFIG.model_seeds,
        perturbation_seeds=FROZEN_CONFIG.perturbation_seeds,
        runtime_versions={
            "numba": "0.66.0",
            "numpy": "2.3.5",
            "python": "3.14.0",
            "torch": "2.13.0",
        },
        backend_contract_sha256=FROZEN_TASK7_BACKEND_AUDIT_SHA256,
    )


def rendered_payloads() -> dict[str, bytes]:
    from experiments.exp417_causal_multihorizon_shadow.reporting import (
        render_exp417_payloads,
    )

    summary = fixture_summary()
    return dict(
        render_exp417_payloads(
            summary, fixture_provenance(), fixture_trusted_checkpoint(summary)
        )
    )


def fixture_trusted_checkpoint(summary: RunSummary | None = None):
    from experiments.exp417_causal_multihorizon_shadow.reporting import (
        TrustedCheckpointProvenance,
    )
    from experiments.exp417_causal_multihorizon_shadow.runner import (
        _result_evidence_digest,
    )

    selected = fixture_summary() if summary is None else summary
    return TrustedCheckpointProvenance(
        checkpoint_provenance_sha256=sha256(
            canonical_json_bytes(fixture_provenance().to_payload())
        ).hexdigest(),
        result_evidence_sha256={
            result.dataset: _result_evidence_digest(result)
            for result in selected.results
        },
    )


def _json_lines(payload: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


def _rewrite_json_lines(rows: list[dict[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _rewrite_csv(rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode("utf-8")


class ReportingTests(unittest.TestCase):
    def test_renders_exact_canonical_seven_payloads(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow.reporting import (
            NON_MANIFEST_FILES,
            verify_exp417_payloads,
        )

        payloads = rendered_payloads()

        self.assertEqual(set(payloads), set(NON_MANIFEST_FILES))
        self.assertEqual(len(payloads), 7)
        verify_exp417_payloads(
            payloads, trusted_checkpoint=fixture_trusted_checkpoint()
        )
        prediction_rows = _json_lines(
            payloads["exp417_shadow_predictions.jsonl"]
        )
        self.assertEqual(
            [(row["dataset"], row["instance_index"]) for row in prediction_rows],
            [("dataset-0000", 0)],
        )
        self.assertEqual(
            {
                key
                for row in prediction_rows
                for key in row
                if key in {"is_hard", "is_standard", "is_priority", "is_automatic_normal"}
            },
            {"is_hard", "is_standard", "is_priority", "is_automatic_normal"},
        )
        self.assertTrue(
            all(
                row[key] is False
                for row in prediction_rows
                for key in ("is_hard", "is_standard", "is_priority", "is_automatic_normal")
            )
        )
        for name in (
            "exp417_backend_benchmark.json",
            "exp417_frozen_config.json",
        ):
            parsed = json.loads(payloads[name])
            self.assertEqual(payloads[name], canonical_json_bytes(parsed) + b"\n")

        qualification = list(
            csv.reader(
                io.StringIO(
                    payloads["exp417_dataset_qualification.csv"].decode("utf-8")
                )
            )
        )
        self.assertEqual(
            qualification[0],
            [
                "dataset", "status", "test_size", "group_count",
                "observed_candidate_count", "published_candidate_count",
                "candidate_rate", "dataset_workload_passed",
                "terminal_stage", "terminal_reason",
            ],
        )
        self.assertEqual(qualification[1][-2:], ["", ""])
        self.assertEqual(
            [row[0] for row in qualification[1:]],
            sorted(row[0] for row in qualification[1:]),
        )
        report = payloads["EXP417_RESULT_REPORT_KO.md"].decode("utf-8")
        self.assertIn("정상 TRAIN만", report)
        self.assertIn("TEST 라벨을 사용하지 않음", report)
        self.assertIn("Shadow", report)
        self.assertIn("자동 경보로 승격하지 않음", report)

    def test_global_workload_failure_cannot_render_official_payloads(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow.reporting import (
            render_exp417_payloads,
        )

        summary = fixture_summary()
        first = summary.results[0]
        contract = first.decision_evidence.frozen_train_contract
        decisions = tuple(
            TestInstanceDecision(
                dataset=first.dataset,
                instance_index=index,
                length=100,
                seed_decisions=fixture_seed_decisions(candidate=index < 13),
            )
            for index in range(first.test_size)
        )
        evidence = DatasetDecisionEvidence(
            dataset=first.dataset,
            context=summary.context,
            frozen_train_contract=contract,
            database_sha256=summary.context.database_sha256,
            train_identity_sha256=contract.train_identity_sha256,
            decisions=decisions,
        )
        blocked_first = DatasetResult(
            first.dataset,
            first.test_size,
            first.group_count,
            first.groups,
            (),
            evidence,
            first.lanes,
            first.context,
            "abstained",
        )
        blocked_results = (blocked_first, *summary.results[1:])
        blocked = RunSummary(
            blocked_results,
            summary.total_test,
            False,
            "global_workload_gate_failed",
            summary.context,
        )
        with self.assertRaisesRegex(ValueError, "global|publication"):
            render_exp417_payloads(
                blocked,
                fixture_provenance(),
                fixture_trusted_checkpoint(summary),
            )

    def test_independent_semantic_verifier_rejects_every_tamper_class(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow.reporting import (
            verify_exp417_payloads,
        )

        original = rendered_payloads()
        cases: dict[str, dict[str, bytes]] = {}

        for name, field, value in (
            ("candidate_score", "score", 9.0),
            ("candidate_p_value", "p_value", 0.5),
            ("candidate_persistence", "persistence", 0.0),
        ):
            changed = dict(original)
            rows = _json_lines(changed["exp417_shadow_predictions.jsonl"])
            rows[0]["seed_decisions"][0]["scores"][0][field] = value
            changed["exp417_shadow_predictions.jsonl"] = _rewrite_json_lines(rows)
            cases[name] = changed

        changed = dict(original)
        rows = _json_lines(changed["exp417_shadow_predictions.jsonl"])
        rows[0]["seed_decisions"][0]["candidate"] = False
        changed["exp417_shadow_predictions.jsonl"] = _rewrite_json_lines(rows)
        cases["seed_consensus"] = changed

        changed = dict(original)
        rows = list(csv.reader(io.StringIO(
            changed["exp417_dataset_qualification.csv"].decode("utf-8")
        )))
        rows[1][1] = "failed"
        changed["exp417_dataset_qualification.csv"] = _rewrite_csv(rows)
        cases["dataset_status"] = changed

        changed = dict(original)
        rows = list(csv.reader(io.StringIO(
            changed["exp417_oof_normal_metrics.csv"].decode("utf-8")
        )))
        rows[1][4] = "0.02"
        changed["exp417_oof_normal_metrics.csv"] = _rewrite_csv(rows)
        cases["qualification_rate"] = changed

        changed = dict(original)
        config = json.loads(changed["exp417_frozen_config.json"])
        config["run"]["total_test"] += 1
        changed["exp417_frozen_config.json"] = canonical_json_bytes(config) + b"\n"
        cases["workload_total"] = changed

        changed = dict(original)
        backend = json.loads(changed["exp417_backend_benchmark.json"])
        backend["audits"][0]["speedup"] = 1.2
        changed["exp417_backend_benchmark.json"] = canonical_json_bytes(backend) + b"\n"
        cases["backend_audit"] = changed

        changed = dict(original)
        config = json.loads(changed["exp417_frozen_config.json"])
        config["frozen_config_sha256"] = "f" * 64
        changed["exp417_frozen_config.json"] = canonical_json_bytes(config) + b"\n"
        cases["config_hash"] = changed

        changed = dict(original)
        config = json.loads(changed["exp417_frozen_config.json"])
        config["context"]["database_sha256"] = "f" * 64
        changed["exp417_frozen_config.json"] = canonical_json_bytes(config) + b"\n"
        cases["source_hash"] = changed

        changed = dict(original)
        config = json.loads(changed["exp417_frozen_config.json"])
        config["governance"]["official_prediction"] = False
        changed["exp417_frozen_config.json"] = canonical_json_bytes(config) + b"\n"
        cases["governance"] = changed

        changed = dict(original)
        rows = list(csv.reader(io.StringIO(
            changed["exp417_dataset_qualification.csv"].decode("utf-8")
        )))
        rows[1], rows[2] = rows[2], rows[1]
        changed["exp417_dataset_qualification.csv"] = _rewrite_csv(rows)
        cases["csv_order"] = changed

        changed = dict(original)
        report = changed["EXP417_RESULT_REPORT_KO.md"].decode("utf-8")
        changed["EXP417_RESULT_REPORT_KO.md"] = report.replace(
            "전체 TEST 수: 1216", "전체 TEST 수: 1217"
        ).encode("utf-8")
        cases["report_count"] = changed

        for name, payloads in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_exp417_payloads(
                    payloads, trusted_checkpoint=fixture_trusted_checkpoint()
                )

    def test_coherent_candidate_rewrite_cannot_recompute_its_own_trust_root(self) -> None:
        """Changing a valid score plus every mutable local hash must still fail."""
        from experiments.exp417_causal_multihorizon_shadow.reporting import (
            verify_exp417_payloads,
        )

        payloads = rendered_payloads()
        rows = _json_lines(payloads["exp417_shadow_predictions.jsonl"])
        rows[0]["seed_decisions"][0]["scores"][0]["score"] = 3.0
        body = dict(rows[0])
        del body["evidence_sha256"]
        rows[0]["evidence_sha256"] = sha256(canonical_json_bytes(body)).hexdigest()
        payloads["exp417_shadow_predictions.jsonl"] = _rewrite_json_lines(rows)
        config = json.loads(payloads["exp417_frozen_config.json"])
        if "artifact_bindings" in config:
            config["artifact_bindings"]["exp417_shadow_predictions.jsonl"] = sha256(
                payloads["exp417_shadow_predictions.jsonl"]
            ).hexdigest()
        payloads["exp417_frozen_config.json"] = canonical_json_bytes(config) + b"\n"

        with self.assertRaises(ValueError):
            verify_exp417_payloads(
                payloads, trusted_checkpoint=fixture_trusted_checkpoint()
            )

    def test_coherent_backend_rewrite_cannot_recompute_its_own_trust_root(self) -> None:
        """A structurally valid CPU audit must still equal pinned Task 7 evidence."""
        from experiments.exp417_causal_multihorizon_shadow.reporting import (
            verify_exp417_payloads,
        )

        payloads = rendered_payloads()
        backend = json.loads(payloads["exp417_backend_benchmark.json"])
        for audit in backend["audits"]:
            audit["speedup"] = 0.5
        payloads["exp417_backend_benchmark.json"] = canonical_json_bytes(backend) + b"\n"
        config = json.loads(payloads["exp417_frozen_config.json"])
        if "artifact_bindings" in config:
            config["artifact_bindings"]["exp417_backend_benchmark.json"] = sha256(
                payloads["exp417_backend_benchmark.json"]
            ).hexdigest()
        payloads["exp417_frozen_config.json"] = canonical_json_bytes(config) + b"\n"

        with self.assertRaises(ValueError):
            verify_exp417_payloads(
                payloads, trusted_checkpoint=fixture_trusted_checkpoint()
            )

    def test_reordered_seed_evidence_is_noncanonical_even_with_recomputed_hashes(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow.reporting import (
            verify_exp417_payloads,
        )

        payloads = rendered_payloads()
        rows = _json_lines(payloads["exp417_shadow_predictions.jsonl"])
        rows[0]["seed_decisions"].reverse()
        body = dict(rows[0])
        del body["evidence_sha256"]
        rows[0]["evidence_sha256"] = sha256(canonical_json_bytes(body)).hexdigest()
        payloads["exp417_shadow_predictions.jsonl"] = _rewrite_json_lines(rows)
        config = json.loads(payloads["exp417_frozen_config.json"])
        if "artifact_bindings" in config:
            config["artifact_bindings"]["exp417_shadow_predictions.jsonl"] = sha256(
                payloads["exp417_shadow_predictions.jsonl"]
            ).hexdigest()
        payloads["exp417_frozen_config.json"] = canonical_json_bytes(config) + b"\n"

        with self.assertRaises(ValueError):
            verify_exp417_payloads(
                payloads, trusted_checkpoint=fixture_trusted_checkpoint()
            )

    def test_external_checkpoint_map_rejects_missing_extra_and_mutated_entries(self) -> None:
        from experiments.exp417_causal_multihorizon_shadow.reporting import (
            TrustedCheckpointProvenance,
            verify_exp417_payloads,
        )

        payloads = rendered_payloads()
        trusted = fixture_trusted_checkpoint()
        base = dict(trusted.result_evidence_sha256)
        variants = []
        missing = dict(base)
        missing.pop(next(iter(missing)))
        variants.append(missing)
        extra = dict(base)
        extra["foreign"] = "f" * 64
        variants.append(extra)
        mutated = dict(base)
        mutated[next(iter(mutated))] = "f" * 64
        variants.append(mutated)
        for index, mapping in enumerate(variants):
            anchor = TrustedCheckpointProvenance(
                trusted.checkpoint_provenance_sha256, mapping
            )
            with self.subTest(index=index), self.assertRaises(ValueError):
                verify_exp417_payloads(payloads, trusted_checkpoint=anchor)

    def test_fully_coherent_result_rewrite_fails_external_checkpoint_root(self) -> None:
        """Even all regenerated payload-local evidence cannot replace checkpoint truth."""
        from experiments.exp417_causal_multihorizon_shadow.reporting import (
            render_exp417_payloads,
            verify_exp417_payloads,
        )

        original = fixture_summary()
        first = original.results[0]
        evidence = first.decision_evidence
        decision = evidence.decisions[0]
        seed = decision.seed_decisions[0]
        scores = dict(seed.scores)
        score = scores[1]
        scores[1] = HorizonScore(
            score.horizon, 3.0, score.p_value, score.persistence
        )
        changed_seed = SeedDecision(seed.seed, scores, seed.candidate)
        changed_decision = TestInstanceDecision(
            decision.dataset,
            decision.instance_index,
            decision.length,
            (changed_seed, *decision.seed_decisions[1:]),
        )
        changed_evidence = DatasetDecisionEvidence(
            evidence.dataset,
            evidence.context,
            evidence.frozen_train_contract,
            evidence.database_sha256,
            evidence.train_identity_sha256,
            (changed_decision, *evidence.decisions[1:]),
        )
        changed_first = DatasetResult(
            first.dataset,
            first.test_size,
            first.group_count,
            first.groups,
            changed_evidence.derived_candidates,
            changed_evidence,
            first.lanes,
            first.context,
            first.status,
        )
        forged = RunSummary(
            (changed_first, *original.results[1:]),
            original.total_test,
            True,
            "complete",
            original.context,
        )
        forged_payloads = dict(
            render_exp417_payloads(
                forged,
                fixture_provenance(),
                fixture_trusted_checkpoint(forged),
            )
        )

        with self.assertRaises(ValueError):
            verify_exp417_payloads(
                forged_payloads,
                trusted_checkpoint=fixture_trusted_checkpoint(original),
            )


if __name__ == "__main__":
    unittest.main()
