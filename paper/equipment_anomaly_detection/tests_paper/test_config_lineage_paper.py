"""Behavioral contracts for the immutable paper rerun identity and provenance."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest

from paper.equipment_anomaly_detection.config_paper import FROZEN_CONFIG, LINEAGE_LOCK_PATH, PAPER_ROOT
from paper.equipment_anomaly_detection.lineage_paper import (
    LINEAGE_NODES,
    _ast_import_callers,
    validate_artifact_parity_claim,
    validate_codemap_triplet,
    validate_lineage_lock,
)


class ConfigLineageTests(unittest.TestCase):
    def test_ast_caller_scanner_resolves_module_form_direct_and_aliased_imports(self) -> None:
        """Module aliases are caller edges, not attribute imports to discard."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sources_paper").mkdir()
            for name in ("data_paper.py", "contracts_paper.py", "sources_paper/hard_sources_paper.py"):
                (root / name).write_text("", encoding="utf-8")
            (root / "caller_paper.py").write_text(
                "from paper.equipment_anomaly_detection import data_paper\n"
                "from paper.equipment_anomaly_detection.sources_paper import hard_sources_paper as hard\n"
                "import paper.equipment_anomaly_detection.contracts_paper as contracts\n"
                "import numpy as np\n",
                encoding="utf-8",
            )
            (root / "relative_paper.py").write_text(
                "from . import data_paper as data\n", encoding="utf-8"
            )
            callers = _ast_import_callers(root)
        self.assertEqual(callers["data_paper.py"], frozenset({"caller_paper.py", "relative_paper.py"}))
        self.assertEqual(callers["contracts_paper.py"], frozenset({"caller_paper.py"}))
        self.assertEqual(callers["sources_paper/hard_sources_paper.py"], frozenset({"caller_paper.py"}))

    def test_artifact_parity_claim_is_hash_bound_and_precisely_scoped(self) -> None:
        """Artifact parity may claim only the frozen smoke's available historical stages."""
        claim = validate_artifact_parity_claim(PAPER_ROOT)
        self.assertEqual(claim["available_stages"], ("exp137", "exp236"))
        self.assertEqual(claim["unavailable_stages"], ("exp74d", "exp89", "exp90", "exp93", "exp253"))
        self.assertTrue(claim["freeze_before_artifact_read"])
        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary) / "paper"
            for source in PAPER_ROOT.rglob("*"):
                if source.is_file() and "__pycache__" not in source.parts:
                    target = copied_root / source.relative_to(PAPER_ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            claim_path = copied_root / "artifact_parity_claim_paper.json"
            claim_path.write_text(claim_path.read_text(encoding="utf-8").replace('"exp253"', '"exp999"'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                validate_codemap_triplet(copied_root)

    def test_artifact_parity_claim_rejects_semantic_identity_mutations_without_lock_help(self) -> None:
        """Claim identity is semantic: hashes cannot authorize broader datasets or paths."""
        original = json.loads((PAPER_ROOT / "artifact_parity_claim_paper.json").read_text(encoding="utf-8"))
        mutations = (
            ("dataset", lambda claim: claim["frozen_datasets"].__setitem__(0, "Other_normal_0")),
            ("order", lambda claim: claim["frozen_datasets"].reverse()),
            ("path", lambda claim: claim["artifacts"]["exp137"].__setitem__(0, "results/other.csv")),
            ("stages", lambda claim: claim.__setitem__("available_stages", ["exp137"])),
            ("freeze", lambda claim: claim.__setitem__("freeze_before_artifact_read", False)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label, mutate in mutations:
                claim = json.loads(json.dumps(original))
                mutate(claim)
                (root / "artifact_parity_claim_paper.json").write_text(json.dumps(claim), encoding="utf-8")
                with self.assertRaises(ValueError, msg=label):
                    validate_artifact_parity_claim(root)

    def test_codemap_callers_are_complete_static_ast_import_edges(self) -> None:
        """Every direct paper import is represented by its callee's caller list."""
        codemap = validate_codemap_triplet(PAPER_ROOT)
        actual = _ast_import_callers(PAPER_ROOT)
        self.assertEqual(
            {module: frozenset(detail["callers"]) for module, detail in codemap.modules.items()},
            actual,
        )

    def test_frozen_identity_and_counts(self) -> None:
        """A changed public rerun identity or aggregate changes the frozen experiment."""
        self.assertEqual(FROZEN_CONFIG.run_identity, "paper-proposal-clean-rerun-20260812")
        self.assertEqual(FROZEN_CONFIG.database_path, "/Users/mac/Documents/Dataset/univariate_ts.db")
        self.assertEqual(
            FROZEN_CONFIG.database_sha256,
            "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
        )
        self.assertEqual(FROZEN_CONFIG.excluded_datasets, ("CornellWhaleChallenge", "Wafer_normal_1"))
        self.assertEqual(FROZEN_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_CONFIG.test_row_count, 195900)
        self.assertEqual(FROZEN_CONFIG.ablation_alert_counts, (2218, 2248, 2244, 2222))

    def test_lineage_has_no_prediction_coordinate_or_label_input(self) -> None:
        """Old alert coordinates and TEST labels must not enter the rerun path."""
        nodes = validate_lineage_lock(LINEAGE_LOCK_PATH)
        self.assertEqual(
            {node.node_id for node in nodes},
            {
                "rocket_exp40", "exp55_best", "exp56_best", "exp84_family_guard",
                "exp84_count_cap", "rocket_block_b", "canonical_normal_state",
                "legacy_fullfit_route_evidence", "legacy_hard_route_chain",
                "exp234_three_source", "minirocket_1024_seed_intersection",
                "minirocket_4096_seed_intersection", "raw_exact_normal_guard",
                "low_frequency_two_depth", "hard_empty_budget_30", "train_omission_stability",
            },
        )
        self.assertFalse(any(node.input_kind == "prediction_coordinates" for node in nodes))
        self.assertTrue(all(node.label_access is False for node in nodes))

    def test_public_lineage_nodes_is_the_validated_immutable_registry(self) -> None:
        """Consumers receive the one validated tuple, rather than a mutable reparse."""
        self.assertIsInstance(LINEAGE_NODES, tuple)
        self.assertEqual(LINEAGE_NODES, validate_lineage_lock(LINEAGE_LOCK_PATH))

    def test_origin_validation_is_static_and_never_executes_a_legacy_module(self) -> None:
        """Provenance checking must not trigger legacy import-time effects or dependencies."""
        legacy_origin = "run_experiment_149_150_virtual_run_conformal_policy"
        sys.modules.pop(legacy_origin, None)
        validate_lineage_lock(LINEAGE_LOCK_PATH)
        self.assertNotIn(legacy_origin, sys.modules)

    def test_lineage_rejects_nonfinite_json_and_invalid_field_types(self) -> None:
        """Malformed lineage cannot silently become a provenance dependency."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lineage_paper.json"
            path.write_text("[NaN]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_lineage_lock(path)

            path.write_text('[{"node_id":"first","node_id":"second"}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                validate_lineage_lock(path)

            payload = json.loads(LINEAGE_LOCK_PATH.read_text(encoding="utf-8"))
            payload[0]["label_access"] = "false"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "types"):
                validate_lineage_lock(path)

            payload = json.loads(LINEAGE_LOCK_PATH.read_text(encoding="utf-8"))
            payload[0]["source_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA differs"):
                validate_lineage_lock(path)

    def test_paper_codemap_triplet_is_complete_and_hash_bound(self) -> None:
        """A missing or modified codemap artifact must invalidate the complete triplet."""
        codemap = validate_codemap_triplet(PAPER_ROOT)
        self.assertEqual(
            set(codemap.required_fields),
            {"callers", "effects", "tests", "source_origins"},
        )
        self.assertEqual(
            set(codemap.modules),
            {
                str(path.relative_to(PAPER_ROOT))
                for path in PAPER_ROOT.rglob("*_paper.py")
                if "__pycache__" not in path.parts
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary) / "paper"
            copied_root.mkdir()
            for source in PAPER_ROOT.rglob("*"):
                if not source.is_file() or "__pycache__" in source.parts:
                    continue
                target = copied_root / source.relative_to(PAPER_ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            (copied_root / "codemap_paper.html").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                validate_codemap_triplet(copied_root)

            (copied_root / "codemap_paper.html").write_bytes((PAPER_ROOT / "codemap_paper.html").read_bytes())
            (copied_root / "codemap_paper.lock").write_text(
                '{"schema":"paper-codemap-lock.v1","schema":"paper-codemap-lock.v1","hashes":{}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                validate_codemap_triplet(copied_root)

    def test_all_new_static_files_use_approved_paper_naming(self) -> None:
        """The approved standalone namespace never introduces an antigravity filename."""
        names = tuple(path.name for path in PAPER_ROOT.rglob("*") if path.is_file())
        self.assertTrue(names)
        self.assertTrue(all("paper" in name for name in names))
        self.assertTrue(all("antigravity" not in name for name in names))

    def test_codemap_rejects_a_visible_html_graph_that_diverges_from_json(self) -> None:
        """Human-readable codemap edges cannot drift while embedded JSON remains valid."""
        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary) / "paper"
            for source in PAPER_ROOT.rglob("*"):
                if not source.is_file() or "__pycache__" in source.parts:
                    continue
                target = copied_root / source.relative_to(PAPER_ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            html_path = copied_root / "codemap_paper.html"
            html_path.write_text(
                html_path.read_text(encoding="utf-8").replace("standalone namespace", "wrong namespace", 1),
                encoding="utf-8",
            )
            lock_path = copied_root / "codemap_paper.lock"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["hashes"]["codemap_paper.html"] = sha256(html_path.read_bytes()).hexdigest()
            lock_path.write_text(json.dumps(lock, separators=(",", ":")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "HTML graph differs"):
                validate_codemap_triplet(copied_root)

    def test_codemap_lock_binds_the_contracts_that_define_frozen_defaults(self) -> None:
        """Changing a frozen default is rejected even when config_paper itself is unchanged."""
        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary) / "paper"
            for source in PAPER_ROOT.rglob("*"):
                if not source.is_file() or "__pycache__" in source.parts:
                    continue
                target = copied_root / source.relative_to(PAPER_ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            contracts = copied_root / "contracts_paper.py"
            contracts.write_text(
                contracts.read_text(encoding="utf-8").replace("dataset_count: int = 1117", "dataset_count: int = 1118"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                validate_codemap_triplet(copied_root)

    def test_codemap_rejects_missing_required_scientific_caller_edge(self) -> None:
        """The checked graph is regenerated from the hard-route AST dependency edges."""
        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary) / "paper"
            for source in PAPER_ROOT.rglob("*"):
                if source.is_file() and "__pycache__" not in source.parts:
                    target = copied_root / source.relative_to(PAPER_ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            codemap_path = copied_root / "codemap_paper.json"
            codemap = json.loads(codemap_path.read_text(encoding="utf-8"))
            codemap["modules"]["sources_paper/rocket_paper.py"]["callers"].remove("sources_paper/legacy_route_sources_paper.py")
            codemap_path.write_text(json.dumps(codemap, separators=(",", ":")), encoding="utf-8")
            lock_path = copied_root / "codemap_paper.lock"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["hashes"]["codemap_paper.json"] = sha256(codemap_path.read_bytes()).hexdigest()
            lock_path.write_text(json.dumps(lock, separators=(",", ":")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "caller graph"):
                validate_codemap_triplet(copied_root)


if __name__ == "__main__":
    unittest.main()
