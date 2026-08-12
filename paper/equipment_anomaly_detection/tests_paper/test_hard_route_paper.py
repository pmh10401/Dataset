"""Synthetic, label-free regression contracts for the guarded Hard route."""
from __future__ import annotations

import ast
from collections import Counter
import inspect
import math
import tempfile
import unittest
import subprocess
from unittest.mock import patch
from pathlib import Path

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    EXPECTED_HARD_SOURCES,
    HardSourceBundle,
    HardSourceResult,
)
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import (
    ConfirmationBlocks,
    FullFitScoreView,
    LegacyRouteEvidence,
    WEAK_FAMILIES_PAPER,
    _fullfit_primary,
    build_legacy_route_evidence,
)
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import ConfigurationScoresPaper, MiniRocketKernelResult, MiniRocketResult, SEEDS_PAPER
from paper.equipment_anomaly_detection.hard_route_paper import (
    _exp93_order,
    RouteLanes,
    compare_parity,
    exact_normal_guard,
    freeze_canonical_smoke,
    minimum_source_expansion,
    replay_exp133_134_135_137,
    replay_exp74d_89_90_93,
    replay_hard_route,
)


def _origin_functions(path: str, names: tuple[str, ...]) -> dict[str, object]:
    """Execute only requested pure origin AST definitions; never import legacy I/O."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("requested origin helper is unavailable")
    namespace = {"np": np, "math": math, "Counter": Counter}
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), namespace)
    return namespace


def _values() -> RawDataset:
    train = tuple(np.asarray([0.0, 1.0, 2.0], dtype=np.float32) + index for index in range(12))
    test = tuple(np.asarray([0.0, 1.0, 2.0], dtype=np.float32) + index for index in range(6))
    return RawDataset(
        dataset="WeakFamily_normal_1",
        metadata_length=3,
        train_indices=tuple(range(len(train))), train_blobs=tuple(b"train" + bytes([index]) for index in range(len(train))), train_rows=train,
        test_indices=tuple(range(len(test))), test_blobs=(b"different", b"train\x02", b"different-2", b"different-3", b"different-4", b"different-5"), test_rows=test,
    )


def _view(name: str, test: tuple[float, ...], *, rate: float = .015) -> FullFitScoreView:
    train = np.linspace(0.0, 1.1, 12)
    return FullFitScoreView(name=name, train_score=train, test_score=np.asarray(test), rate=rate)


def _evidence() -> LegacyRouteEvidence:
    # Full-fit candidates deliberately disagree with the OOF evidence below.
    primary = {
        "rocket_exp40": _view("rocket_exp40", (0.1, 2.0, 1.8, 0.2, 0.3, 0.4)),
        "exp55_best": _view("exp55_best", (0.1, 1.9, 1.7, 0.2, 0.3, 0.4)),
        "exp56_best": _view("exp56_best", (0.1, 1.8, 1.6, 0.2, 0.3, 0.4)),
    }
    exp84 = _view("exp84", (0.1, 2.1, 1.5, 0.2, 0.3, 0.4), rate=.03)
    blocks = ConfirmationBlocks(
        block_a=_view("block_a", (0.0, 1.0, 2.0, 0.0, 0.0, 0.0), rate=.015),
        block_b_015=_view("block_b", (0.0, 2.0, 0.1, 0.0, 0.0, 0.0), rate=.015),
        block_b_01=_view("block_b", (0.0, 2.0, 0.1, 0.0, 0.0, 0.0), rate=.01),
        block_c=_view("block_c", (0.0, 0.0, 2.0, 0.0, 0.0, 0.0), rate=.01),
    )
    return LegacyRouteEvidence(
        dataset="WeakFamily_normal_1", family="WeakFamily", n_train=12, n_test=6,
        test_median_length=3, primary=primary, exp84=exp84, confirmation_blocks=blocks,
    )


def _sources() -> HardSourceBundle:
    by_name = {}
    for name in EXPECTED_HARD_SOURCES:
        # p=1/201 for every OOF source except a two-source agreement at index 4.
        train_oof = np.arange(200, dtype=float)
        test = np.zeros(6, dtype=float)
        if name in {"rocket_exp40", "exp55_best"}:
            test[4] = 1000.0
        by_name[name] = HardSourceResult(
            name=name, origin=object(), train_oof_score=train_oof, test_score=test,
            test_pvalue=np.ones(6), threshold=.01, test_native_pass=np.zeros(6, dtype=bool),
            eligible=True, eligibility_reason="eligible",
        )
    return HardSourceBundle(EXPECTED_HARD_SOURCES, by_name)


def _mini() -> MiniRocketResult:
    def kernel(requested: int, dimension: int) -> MiniRocketKernelResult:
        by_seed = {seed: ConfigurationScoresPaper(np.arange(200, dtype=float), np.array([0., 0., 0., 1000., 0., 0.]), dimension) for seed in SEEDS_PAPER}
        return MiniRocketKernelResult(requested, dimension, by_seed, frozenset({3}))
    k1024, k4096 = kernel(1024, 1008), kernel(4096, 4032)
    return MiniRocketResult(k1024.candidate, k4096.candidate, {1024: k1024, 4096: k4096})


class HardRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = _values()
        self.evidence = _evidence()
        self.sources = _sources()
        self.mini = _mini()

    def test_fullfit_evidence_drives_exp74d_not_task4_oof(self) -> None:
        """Changing Task4 OOF values cannot change the frozen Exp74d selection."""
        exp74d, exp89, exp90, exp93 = replay_exp74d_89_90_93(self.evidence)
        self.assertIn(1, exp74d.hard | exp74d.standard | exp74d.priority)
        self.assertEqual(exp89, exp90)
        self.assertIsInstance(exp93, frozenset)

    def test_exp84_family_guard_uses_the_frozen_weak_family_set(self) -> None:
        self.assertIn("ChlorineConcentration", WEAK_FAMILIES_PAPER)
        self.assertNotIn("InlineSkate", WEAK_FAMILIES_PAPER)
        guarded = self.evidence.exp84_family_guard
        self.assertEqual(guarded.rate, .02)

    def test_exp234_uses_only_five_task4_oof_sources(self) -> None:
        lanes = RouteLanes(frozenset({1}), frozenset({2, 3}), frozenset())
        result = minimum_source_expansion(lanes, self.sources, alpha=.01)
        self.assertEqual(result.minimum_two, frozenset({4}))
        self.assertEqual(result.minimum_three, frozenset())
        self.sources.by_name["rocket_block_b"].test_score[4] = 1000.0
        self.assertEqual(minimum_source_expansion(lanes, self.sources, alpha=.01).minimum_two, frozenset({4}))

    def test_confirmation_route_keeps_lanes_disjoint(self) -> None:
        exp93 = frozenset({1, 2})
        lanes = replay_exp133_134_135_137(exp93, self.evidence.confirmation_blocks)
        self.assertFalse(lanes.hard & lanes.standard)
        self.assertFalse(lanes.hard & lanes.priority)
        self.assertFalse(lanes.standard & lanes.priority)

    def test_exp134_uses_block_b_one_percent_not_exp133_one_point_five(self) -> None:
        """A candidate between the tails is Standard in Exp133 but eligible for Exp134."""
        train = np.arange(100, dtype=float)
        blocks = ConfirmationBlocks(
            block_a=FullFitScoreView("block_a", train, np.array([150.0, 0.0]), .015),
            block_b_015=FullFitScoreView("block_b", train, np.array([99.5, 0.0]), .015),
            block_b_01=FullFitScoreView("block_b", train, np.array([99.5, 0.0]), .01),
            block_c=FullFitScoreView("block_c", train, np.array([150.0, 0.0]), .01),
        )
        lanes = replay_exp133_134_135_137(frozenset({1}), blocks)
        self.assertEqual(lanes.hard, frozenset())
        self.assertEqual(lanes.standard, frozenset({1}))
        self.assertEqual(lanes.priority, frozenset({0}))

    def test_exp93_primary_orders_are_stable_descending(self) -> None:
        # Equal scores distinguish Exp93's stable primitive from Exp74d's origin quicksort.
        primary = {name: _view(name, (1.0, 1.0, 1.0, 0.0), rate=.015) for name in ("rocket_exp40", "exp55_best", "exp56_best")}
        exp84 = _view("exp84", (1.0, 1.0, 1.0, 0.0), rate=.03)
        blocks = ConfirmationBlocks(_view("a", (0.,) * 4), _view("b", (0.,) * 4), _view("b", (0.,) * 4, rate=.01), _view("c", (0.,) * 4, rate=.01))
        evidence = LegacyRouteEvidence("tie_normal_1", "tie", 12, 4, 3, primary, exp84, blocks)
        _, _, exp90, exp93 = replay_exp74d_89_90_93(evidence)
        self.assertEqual(exp90, exp93)
        # This asserts the copied Exp93 source primitive itself, not a set iteration accident.
        from paper.equipment_anomaly_detection.hard_route_paper import _exp93_order
        self.assertEqual(_exp93_order(primary["rocket_exp40"], 3), [0, 1, 2])

    def test_exp74d_and_exp90_tied_score_primitives_match_unimported_origin_ast(self) -> None:
        names = ("selection", "indices_at_least", "cap_indices", "cap_indices_count", "exp69_base_selection", "rank_ensemble_indices", "top_score_indices", "rank_ensemble_guarded_indices", "large_data_budget", "top_score_order")
        origin = _origin_functions("run_experiment_60_62_rocket_imaging_selector_variants.py", names)
        bundles = {name: {"indices": set(view.candidate), "test_scores": view.test_score} for name, view in self.evidence.primary.items()}
        base = origin["exp69_base_selection"](bundles)["indices"]
        budget1, budget2 = origin["large_data_budget"](np.zeros(6), .01, 2, 5), origin["large_data_budget"](np.zeros(6), .02, 3, 8)
        primary = origin["rank_ensemble_guarded_indices"](bundles, budget2, "rocket_top", max(budget2, budget1 * 2))
        review = origin["cap_indices_count"](origin["rank_ensemble_indices"](bundles, budget2) - primary, bundles["rocket_exp40"], budget1)
        ours, _, _, _ = replay_exp74d_89_90_93(self.evidence)
        self.assertEqual(ours.hard, frozenset(base))
        self.assertEqual(ours.standard, frozenset())
        self.assertEqual(_exp93_order(self.evidence.primary["rocket_exp40"], 6), origin["top_score_order"](bundles["rocket_exp40"], 6))
        exp90_origin = _origin_functions("run_experiment_90_zero_f1_repair_selector.py", ("first_top_index_from_bundle",))["first_top_index_from_bundle"]
        exp90_origin.__globals__["top_score_indices"] = origin["top_score_indices"]
        self.assertEqual(exp90_origin(bundles["rocket_exp40"]), next(iter(origin["top_score_indices"](bundles["rocket_exp40"], 1))))

    def test_abc_count_cap_scoring_matches_the_origin_threshold_primitive(self) -> None:
        origin = _origin_functions("run_experiment_40_original_score_normalization_sweep.py", ("clean_scores", "count_cap_threshold"))["count_cap_threshold"]
        train = np.asarray([0., 1., 2., 3., 4., 5., 6., 7.])
        from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import count_cap_threshold_paper
        for rate in (.015, .01):
            self.assertEqual(count_cap_threshold_paper(train, rate), origin(train, rate))

    def test_fullfit_exp74d_primary_rates_are_all_one_point_five_percent(self) -> None:
        values = _values()
        with patch("paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper._fullfit_primary", side_effect=lambda name, train, test, target: FullFitScoreView(name, np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), .015)), patch("paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper._confirmation_blocks", return_value=self.evidence.confirmation_blocks):
            evidence = build_legacy_route_evidence(values)
        self.assertEqual({view.rate for view in evidence.primary.values()}, {.015})

    def test_transductive_target_is_bounded_historical_train_test_median(self) -> None:
        long = RawDataset("long", 4, tuple(range(2)), (b"a", b"b"), (np.zeros(3000, dtype=np.float32), np.zeros(3001, dtype=np.float32)), (0,), (b"c",), (np.zeros(4000, dtype=np.float32),))
        blocks = ConfirmationBlocks(*(FullFitScoreView(name, np.arange(2, dtype=float), np.zeros(1), rate) for name, rate in (("a", .015), ("b", .015), ("b", .01), ("c", .01))))
        with patch("paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper._fullfit_primary", side_effect=lambda name, train, test, target: FullFitScoreView(name, np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), .015)), patch("paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper._confirmation_blocks", return_value=blocks):
            evidence = build_legacy_route_evidence(long)
        self.assertEqual(evidence.transductive_target_length, 2048)

    def test_exp40_raw_rocket_features_go_directly_to_density_local_gap(self) -> None:
        train, test = np.ones((3, 8), dtype=float), np.ones((2, 8), dtype=float)
        from paper.equipment_anomaly_detection.sources_paper import legacy_route_sources_paper
        self.assertNotIn("scale_feature_pair_paper", vars(legacy_route_sources_paper))
        with patch("paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper.rocket_feature_pair_paper", return_value=(np.ones((3, 5)), np.ones((2, 5)))), patch("paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper.density_knn_score_pair_paper", return_value=(np.ones(3), np.ones(2))) as density:
            _fullfit_primary("rocket_exp40", train, test, 8)
        self.assertEqual(density.call_args.args[0].shape, (3, 5))

    def test_scientific_views_are_defensive_read_only_and_validate_shapes(self) -> None:
        scores = np.array([0.0, 1.0])
        view = FullFitScoreView("x", scores, scores, .015)
        scores[0] = 99.0
        self.assertEqual(view.train_score[0], 0.0)
        self.assertFalse(view.train_score.flags.writeable)
        with self.assertRaises(ValueError):
            FullFitScoreView("bad", np.ones(2), np.ones((2, 1)), .015)

    def test_ten_dataset_new_smoke_freezes_before_a_separate_oracle_compare(self) -> None:
        new = {f"raw-smoke-{index:02d}": frozenset({index % 3}) for index in range(10)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = freeze_canonical_smoke(new, root / "new-route.json")
            self.assertEqual(len(frozen.datasets), 10)
            self.assertTrue(frozen.path.is_file())
            oracle = root / "oracle.jsonl"
            oracle.write_text("\n".join(__import__("json").dumps({"dataset": name, "indices": sorted(indices)}) for name, indices in new.items()) + "\n", encoding="utf-8")
            self.assertTrue(compare_parity(new, oracle).exact)

    def test_capable_ten_dataset_actual_raw_smoke_freezes_then_compares_origin_route_and_abc(self) -> None:
        """Bounded smoke uses actual raw scorers; no coordinate artifact enters scoring."""
        oracle = Path("/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python")
        if not oracle.is_file():
            self.skipTest("capable frozen oracle interpreter is unavailable")
        code = r'''
import logging, tempfile
from pathlib import Path
logging.FileHandler=lambda *a, **k: logging.StreamHandler()
import numpy as np
import run_experiment_132_block_b_review_integration as exp132
import run_experiment_135_block_c_review_confirmation as exp135
import run_experiment_60_62_rocket_imaging_selector_variants as exp60
from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import freeze_canonical_smoke, replay_exp74d_89_90_93, replay_exp133_134_135_137, compare_parity
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import build_legacy_route_evidence
from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import z_normalize_paper
import json
rows={}
for dataset in range(10):
  rng=np.random.default_rng(20260812+dataset)
  train=rng.normal(size=(5,16+dataset%3)).astype('float32'); test=rng.normal(size=(3,16+dataset%3)).astype('float32')
  values=RawDataset(f'raw-smoke-{dataset:02d}',16+dataset%3,tuple(range(5)),tuple(bytes([dataset,i])*4 for i in range(5)),tuple(train),tuple(range(3)),tuple(bytes([dataset,20+i])*4 for i in range(3)),tuple(test))
  evidence=build_legacy_route_evidence(values)
  ztrain,ztest=z_normalize_paper(train),z_normalize_paper(test)
  a_train,a_test,b_train,b_test,*_=exp132.block_scores(ztrain,ztest)
  np.testing.assert_allclose(a_train,evidence.confirmation_blocks.block_a.train_score,rtol=1e-6,atol=1e-6)
  np.testing.assert_allclose(a_test,evidence.confirmation_blocks.block_a.test_score,rtol=1e-6,atol=1e-6)
  np.testing.assert_allclose(b_train,evidence.confirmation_blocks.block_b_015.train_score,rtol=1e-6,atol=1e-6)
  np.testing.assert_allclose(b_test,evidence.confirmation_blocks.block_b_015.test_score,rtol=1e-6,atol=1e-6)
  c_indices,c_score,*_=exp135.block_c_candidates(ztrain,ztest)
  np.testing.assert_allclose(c_score,evidence.confirmation_blocks.block_c.test_score,rtol=1e-6,atol=1e-6)
  assert c_indices == set(evidence.confirmation_blocks.block_c.candidate)
  bundles={name:{'indices':set(view.candidate),'test_scores':view.test_score} for name,view in evidence.primary.items()}
  origin_base=exp60.exp69_base_selection(bundles)['indices']
  paper74, paper89, paper90, paper93=replay_exp74d_89_90_93(evidence)
  assert paper74.hard == frozenset(origin_base) and not paper74.standard
  lanes137=replay_exp133_134_135_137(paper93,evidence.confirmation_blocks)
  rows[values.dataset]=frozenset(lanes137.hard|lanes137.standard|lanes137.priority)
with tempfile.TemporaryDirectory() as directory:
  root=Path(directory); frozen=freeze_canonical_smoke(rows,root/'new.json')
  oracle_path=root/'origin-route.jsonl'
  oracle_path.write_text(''.join(json.dumps({'dataset':name,'indices':sorted(indices)})+'\n' for name,indices in rows.items()),encoding='utf-8')
  assert compare_parity(rows,oracle_path).exact and frozen.path.is_file()
print('actual-raw-10-smoke-parity')
'''
        completed = subprocess.run([str(oracle), "-c", code], cwd=Path.cwd(), check=True, capture_output=True, text=True, timeout=180)
        self.assertIn("actual-raw-10-smoke-parity", completed.stdout)

    def test_route_preserves_stages_and_exact_blob_guard(self) -> None:
        with patch("paper.equipment_anomaly_detection.hard_route_paper.build_legacy_route_evidence", return_value=self.evidence):
            result = replay_hard_route(self.values, self.sources, self.mini)
        self.assertEqual(result.exact_normal_removed, frozenset({1}))
        self.assertNotIn(1, result.automatic_hard)
        self.assertTrue(result.exp253.hard.issuperset(result.automatic_hard))

    def test_exact_normal_guard_is_byte_exact_not_value_equivalent(self) -> None:
        automatic, removed = exact_normal_guard(frozenset({0, 1}), (b"x",), (b"x", b"x\x00"))
        self.assertEqual(automatic, frozenset({1}))
        self.assertEqual(removed, frozenset({0}))

    def test_route_function_cannot_accept_oracle_coordinates(self) -> None:
        self.assertNotIn("oracle_path", inspect.signature(replay_hard_route).parameters)
        self.assertNotIn("frozen_coordinates", inspect.signature(replay_hard_route).parameters)


if __name__ == "__main__":
    unittest.main()
