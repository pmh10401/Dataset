"""Synthetic, label-free regression contracts for the guarded Hard route."""
from __future__ import annotations

import ast
from collections import Counter
import inspect
import math
import tempfile
import unittest
import subprocess
from types import SimpleNamespace
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
from paper.equipment_anomaly_detection.sources_paper.rocket_paper import MAX_TRAIN_REFERENCE_PAPER
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import ConfigurationScoresPaper, MiniRocketKernelResult, MiniRocketResult, SEEDS_PAPER
from paper.equipment_anomaly_detection.hard_route_paper import (
    _exp93_order,
    RouteLanes,
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
    namespace = {"np": np, "math": math, "Counter": Counter, "MAX_TOP": 12}
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

    def test_legacy_evidence_distinguishes_capped_rocket_from_full_train_sources(self) -> None:
        """Treating every full-fit source as raw-full TRAIN rejects valid large-TRAIN evidence."""
        n_train, n_test, capped_train = MAX_TRAIN_REFERENCE_PAPER + 1, 3, MAX_TRAIN_REFERENCE_PAPER

        def view(name: str, train_size: int, *, rate: float = .015) -> FullFitScoreView:
            return FullFitScoreView(name, np.zeros(train_size), np.zeros(n_test), rate)

        def evidence(*, rocket_size: int = capped_train, exp55_size: int = n_train) -> LegacyRouteEvidence:
            primary = {
                "rocket_exp40": view("rocket_exp40", rocket_size),
                "exp55_best": view("exp55_best", exp55_size),
                "exp56_best": view("exp56_best", n_train),
            }
            blocks = ConfirmationBlocks(
                view("block_a", capped_train),
                view("block_b", capped_train),
                view("block_b", capped_train, rate=.01),
                view("block_c", capped_train, rate=.01),
            )
            return LegacyRouteEvidence("synthetic_normal_1", "synthetic", n_train, n_test, 8, primary, view("exp84", n_train), blocks)

        accepted = evidence()
        self.assertEqual(accepted.primary["rocket_exp40"].train_score.size, capped_train)
        self.assertEqual(accepted.primary["exp55_best"].train_score.size, n_train)
        self.assertEqual(accepted.primary["exp56_best"].train_score.size, n_train)
        self.assertEqual(accepted.exp84.train_score.size, n_train)
        self.assertEqual(
            tuple(view.train_score.size for view in (
                accepted.confirmation_blocks.block_a,
                accepted.confirmation_blocks.block_b_015,
                accepted.confirmation_blocks.block_b_01,
                accepted.confirmation_blocks.block_c,
            )),
            (capped_train, capped_train, capped_train, capped_train),
        )
        with self.assertRaisesRegex(ValueError, "rocket_exp40 score shape"):
            evidence(rocket_size=capped_train - 1)
        with self.assertRaisesRegex(ValueError, "exp55_best score shape"):
            evidence(exp55_size=capped_train)

    def test_exp74d_hard_precedes_overlapping_standard_review_from_label_free_evidence(self) -> None:
        """Leaving a hard member in Exp74d review makes RouteLanes reject the raw-evidence replay."""
        n_train, n_test = 207, 250
        train = lambda threshold: np.r_[np.zeros(n_train - 4), np.full(4, threshold)]
        def score(**values):
            result = np.zeros(n_test)
            for index, value in values.items():
                result[int(index)] = value
            return result
        rocket = FullFitScoreView("rocket_exp40", train(98.5), score(**{"42": 100., "248": 99., "10": 98., "11": 97., "12": 96., "13": 95., "35": 10., "247": 9., "249": 8.}), .015)
        exp55 = FullFitScoreView("exp55_best", train(98.), score(**{"248": 100., "249": 99., "35": 97.9, "247": 97.8, "42": 97.7}), .015)
        exp56 = FullFitScoreView("exp56_best", train(98.), score(**{"249": 100., "248": 99., "35": 97.9, "247": 97.8, "42": 97.7}), .015)
        blocks = ConfirmationBlocks(*(FullFitScoreView(name, train(1.), np.zeros(n_test), rate) for name, rate in (("a", .015), ("b", .015), ("b", .01), ("c", .01))))
        evidence = LegacyRouteEvidence("synthetic_normal_1", "synthetic", n_train, n_test, 16, {"rocket_exp40": rocket, "exp55_best": exp55, "exp56_best": exp56}, FullFitScoreView("exp84", train(1.), np.zeros(n_test), .015), blocks)
        lanes, _, _, _ = replay_exp74d_89_90_93(evidence)
        self.assertEqual(lanes.hard, frozenset({42, 248, 249}))
        self.assertEqual(lanes.standard, frozenset({35, 247}))
        self.assertFalse(lanes.hard & lanes.standard)

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

    def test_exp90_tied_rocket_fallback_uses_origin_top_score_index_zero(self) -> None:
        """Exp90 must preserve ``np.argsort(-scores)[:1]``, not reversed argsort."""
        train = np.linspace(0.0, 1.1, 12)
        primary = {
            "rocket_exp40": FullFitScoreView("rocket_exp40", train, np.zeros(20), .015),
            "exp55_best": FullFitScoreView("exp55_best", train, np.full(20, -1.0), .015),
            "exp56_best": FullFitScoreView("exp56_best", train, np.full(20, -1.0), .015),
        }
        exp84 = FullFitScoreView("exp84", np.zeros(12), np.full(20, -1.0), .03)
        blocks = ConfirmationBlocks(*(FullFitScoreView(name, train, np.zeros(20), rate) for name, rate in (("a", .015), ("b", .015), ("b", .01), ("c", .01))))
        evidence = LegacyRouteEvidence("tie-normal", "tie", 12, 20, 16, primary, exp84, blocks)
        _, exp89, exp90, _ = replay_exp74d_89_90_93(evidence)
        self.assertEqual(exp89, frozenset())
        self.assertEqual(exp90, frozenset({0}))

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

    def test_executable_origin_exp133_to_137_selectors_match_paper_lanes(self) -> None:
        """Execute selector ASTs with label-free stubs and compare their lane decisions."""
        parse = lambda value: {int(item) for item in str(value).split()} if value else set()
        blocks = self.evidence.confirmation_blocks
        paper = replay_exp133_134_135_137(frozenset({1, 2}), blocks)
        exp133 = _origin_functions("run_experiment_133_block_b_confidence_tiers.py", ("run_one",))["run_one"]
        exp133.__globals__.update({"load_dataset_data": lambda _: (None, None, np.zeros(6)), "parse_indices": parse, "make_row": lambda _d, _y, _h, _b, config, selected, high, standard, _reason: {"config": config, "selected": selected, "high": high, "standard": standard}})
        rows133 = exp133(("fixture", {"selected_indices": "1 2"}, {"selected_indices": "1"}))
        tiered = next(row for row in rows133 if row["config"] == "tiered_all_validated_exp93_hard_alerts")
        self.assertEqual(frozenset(tiered["high"]), paper.hard)
        self.assertEqual(frozenset(tiered["standard"]), paper.standard)

        exp134 = _origin_functions("run_experiment_134_block_b_review_tail_guard.py", ("run_one",))["run_one"]
        exp134.__globals__.update({
            "load_dataset_data": lambda _: (np.zeros((12, 3)), np.zeros((6, 3)), np.zeros(6)),
            "z_normalize": lambda values: values,
            "np": np,
            "parse_indices": parse,
            "count_cap_threshold": lambda scores, rate: (blocks.block_a.threshold if scores is blocks.block_a.train_score else (blocks.block_b_01.threshold if rate == .01 else blocks.block_b_015.threshold), rate, 0),
            "RATES": (.005, .01, .015),
            "base": SimpleNamespace(block_scores=lambda _t, _q: (blocks.block_a.train_score, blocks.block_a.test_score, blocks.block_b_015.train_score, blocks.block_b_015.test_score, 12, 0), make_row=lambda _d, _y, _h, config, review, _reason, _diag: {"config": config, "review": review}),
        })
        rows134 = exp134(("fixture", {"selected_indices": "1 2"}, {}))
        review134 = next(row["review"] for row in rows134 if row["config"] == "review_block_ba_tail_pct_1p0_top1")
        self.assertEqual(frozenset(review134) & frozenset(blocks.block_c.candidate), paper.priority)

        exp135 = _origin_functions("run_experiment_135_block_c_review_confirmation.py", ("run_one",))["run_one"]
        exp135.__globals__.update({
            "load_dataset_data": lambda _: (np.zeros((12, 3)), np.zeros((6, 3)), np.zeros(6)),
            "z_normalize": lambda values: values,
            "parse_indices": parse, "as_float": float,
            "block_c_candidates": lambda _t, _q: (set(blocks.block_c.candidate), blocks.block_c.test_score, 12, 0, .01, 0),
            "base": SimpleNamespace(make_row=lambda _d, _y, _h, config, review, _reason, _diag: {"config": config, "review": review}),
        })
        rows135 = exp135(("fixture", {"selected_indices": "1 2", "review_candidate_indices": "2", "train_normal_count": "12", "tiny_train": "0"}, {"high_confidence_count": "1", "standard_confidence_count": "1"}))
        origin_priority = next(row["review"] for row in rows135 if row["config"] == "review_tail1pct_all_standard_and_block_c")
        self.assertEqual(origin_priority, set())

        origin137 = _origin_functions("run_experiment_137_operational_triage.py", ("route_tiers",))["route_tiers"]
        tiers137 = origin137(set(paper.hard), set(paper.standard), set(paper.priority), 6)
        self.assertEqual(frozenset(tiers137["hard"]), paper.hard)
        self.assertEqual(frozenset(tiers137["standard_review"]), paper.standard)
        self.assertEqual(frozenset(tiers137["priority_review"]), paper.priority)

    def test_executable_origin_exp89_90_93_selector_primitives_match_paper_outputs(self) -> None:
        """Direct AST execution covers Exp89/90 selection primitives and Exp93 replacement."""
        names = ("first_top_index", "exp84_bundle_from_row")
        exp89 = _origin_functions("run_experiment_89_74d_with_exp84_candidate.py", names)
        exp89["exp84_bundle_from_row"].__globals__.update({"parse_indices": lambda value: {int(item) for item in str(value).split()} if value else set(), "as_float": lambda value, default=0.0: float(value) if value is not None else default})
        row = {"selected_indices": "", "top_score_indices": "0 1 2", "train_exceed_rate": ".01", "top1_threshold_margin": "1", "top1_top2_margin": "0"}
        parsed = exp89["exp84_bundle_from_row"](row, 20)
        self.assertEqual(exp89["first_top_index"]({"top_score_indices": "0 1 2"}), 0)
        self.assertEqual(parsed["top1"], 0)

        exp90 = _origin_functions("run_experiment_90_zero_f1_repair_selector.py", ("first_top_index_from_bundle",))["first_top_index_from_bundle"]
        origin60 = _origin_functions("run_experiment_60_62_rocket_imaging_selector_variants.py", ("top_score_indices",))
        exp90.__globals__["top_score_indices"] = origin60["top_score_indices"]
        tie_bundle = {"test_scores": np.zeros(20)}
        self.assertEqual(exp90(tie_bundle), 0)

        primary = {
            "rocket_exp40": FullFitScoreView("rocket_exp40", np.linspace(0, 1, 12), np.arange(20, dtype=float), .015),
            "exp55_best": FullFitScoreView("exp55_best", np.linspace(0, 1, 12), np.r_[np.arange(1, 20), 0.0], .015),
            "exp56_best": FullFitScoreView("exp56_best", np.linspace(0, 1, 12), np.r_[np.arange(2, 20), 0.0, 1.0], .015),
        }
        exp84 = FullFitScoreView("exp84", np.linspace(0, 1, 12), np.arange(20, dtype=float), .03)
        blocks = ConfirmationBlocks(*(FullFitScoreView(name, np.linspace(0, 1, 12), np.zeros(20), rate) for name, rate in (("a", .015), ("b", .015), ("b", .01), ("c", .01))))
        evidence = LegacyRouteEvidence("selector-fixture", "fixture", 12, 20, 16, primary, exp84, blocks)
        _, _, paper90, paper93 = replay_exp74d_89_90_93(evidence)
        origin93 = _origin_functions("run_experiment_93_nonpos_candidate_reranker.py", ("score_candidates", "sorted_candidates", "weak_base_indices"))
        maps = {name: {index: rank for rank, index in enumerate(_exp93_order(view), 1)} for name, view in (("rocket", primary["rocket_exp40"]), ("exp55", primary["exp55_best"]), ("exp56", primary["exp56_best"]))}
        scored = origin93["score_candidates"](set().union(*(set(order) for order in maps.values())) | set(paper90), maps, {"rocket": .36, "exp55": .22, "exp56": .22})
        candidate = next(index for index in origin93["sorted_candidates"](scored) if index not in paper90)
        weak = origin93["weak_base_indices"](set(paper90), scored)
        if weak and len(paper90) <= 1 and scored[candidate]["support"] >= 2 and scored[candidate]["best_rank"] <= 3:
            self.assertIn(candidate, paper93)

    def test_executable_origin_exp89_and_exp93_nonvacuously_match_paper_selectors(self) -> None:
        """Origin Exp89/93 selector outputs must equal paper selectors on active branches."""
        from paper.equipment_anomaly_detection.hard_route_paper import _replay_exp93_from_exp90

        parse = lambda value: {int(item) for item in str(value).split()} if value else set()
        train = np.zeros(12)
        primary = {
            name: FullFitScoreView(name, train, np.full(6, -1.0), .015)
            for name in ("rocket_exp40", "exp55_best", "exp56_best")
        }
        exp84 = FullFitScoreView("exp84", train, np.array([0., 6., 5., 4., 3., 2.]), .03)
        blocks = ConfirmationBlocks(*(FullFitScoreView(name, train, np.zeros(6), rate) for name, rate in (("a", .015), ("b", .015), ("b", .01), ("c", .01))))
        evidence = LegacyRouteEvidence("selector-normal", "fixture", 12, 6, 3, primary, exp84, blocks)
        _, paper89, _, _ = replay_exp74d_89_90_93(evidence)
        exp89 = _origin_functions("run_experiment_89_74d_with_exp84_candidate.py", ("choose_rows_for_dataset", "parse_indices", "exp84_bundle_from_row"))["choose_rows_for_dataset"]
        bundles = {name: {"indices": set(), "test_scores": view.test_score} for name, view in primary.items()}
        exp89.__globals__.update({
            "CALIBRATION_PROFILES": {"relaxed_15pct": {}},
            "as_float": lambda value, default=0.0: float(value) if value not in (None, "") else default,
            "load_candidate_predictions": lambda _name, threshold_rates: ({"train_series": [0] * 12}, np.zeros(6), bundles),
            "is_large_data_case": lambda _record, _test: False,
            "large_data_budget": lambda _test, **_kwargs: 2,
            "top_score_indices": lambda bundle, count: set(np.argsort(-bundle["test_scores"])[:count].tolist()),
            "indices_at_least": lambda minimum, sets: {index for index in set().union(*sets) if sum(index in values for values in sets) >= minimum},
            "cap_indices_count": lambda indices, _bundle, count: set(sorted(indices)[:count]),
            "passthrough_row": lambda *_args, **_kwargs: {},
            "computed_row": lambda _dataset, _record, _test, _bundles, _base, selector, indices, *_args: {"selector_name": selector, "selected_indices": set(indices)},
        })
        rows89 = exp89(("fixture", {"selected_indices": ""}, {"selected_indices": ""}, {("fixture", "family_guard_v1"): {"selected_indices": "1", "top_score_indices": "1 2 3 4 5 0", "train_exceed_rate": ".015", "top1_threshold_margin": "0", "top1_top2_margin": "0"}}))
        origin89 = next(row["selected_indices"] for row in rows89 if row.get("selector_name") == "exp84_four_model_fg_plus_noalert_repair")
        self.assertEqual(frozenset(origin89), paper89)
        self.assertEqual(paper89, frozenset({1}))

        weak_train = np.arange(100., dtype=float)
        weak_primary = {
            "rocket_exp40": FullFitScoreView("rocket_exp40", weak_train, np.arange(20., 0., -1.), .015),
            "exp55_best": FullFitScoreView("exp55_best", weak_train, np.r_[20., 19., 18., 17., 16., -100., np.arange(14., 0., -1.)], .015),
            "exp56_best": FullFitScoreView("exp56_best", weak_train, np.r_[20., 19., 18., 17., 16., -100., np.arange(14., 0., -1.)], .015),
        }
        weak_exp84 = FullFitScoreView("exp84", weak_train, np.zeros(20), .03)
        weak_blocks = ConfirmationBlocks(*(FullFitScoreView(name, weak_train, np.zeros(20), rate) for name, rate in (("a", .015), ("b", .015), ("b", .01), ("c", .01))))
        weak_evidence = LegacyRouteEvidence("weak-selector", "fixture", 100, 20, 3, weak_primary, weak_exp84, weak_blocks)
        paper93 = _replay_exp93_from_exp90(weak_evidence, frozenset({5}))
        exp93 = _origin_functions("run_experiment_93_nonpos_candidate_reranker.py", ("choose_rows_for_dataset", "parse_order", "bundle_order", "exp84_order", "rank_map", "candidate_pool", "score_candidates", "sorted_candidates", "top_non_base", "weak_base_indices", "passthrough_row"))["choose_rows_for_dataset"]
        exp93.__globals__.update({
            "CALIBRATION_PROFILES": {"relaxed_15pct": {}}, "MAX_TOP": 12, "EXPERIMENT_ID": "fixture",
            "parse_indices": parse, "as_float": lambda value, default=0.0: float(value) if value not in (None, "") else default,
            "top_score_order": lambda bundle, count: [int(index) for index in np.argsort(-bundle["test_scores"])[:count]],
            "load_candidate_predictions": lambda _name, threshold_rates: ({"train_series": [0] * 12}, np.zeros(20), {name: {"test_scores": view.test_score} for name, view in weak_primary.items()}),
            "row_with_metrics": lambda _dataset, _record, _test, _bundles, _base, selector, indices, *_args: {"selector_name": selector, "selected_indices": set(indices)},
        })
        exp87 = {("fixture", "family_guard_v1"): {"top_score_indices": "", "train_exceed_rate": "1"}, ("fixture", "count_cap_3pct"): {"top_score_indices": "", "train_exceed_rate": "1"}}
        rows93 = exp93(("fixture", {"selected_indices": "5", "train_normal_count": "12"}, {"selected_indices": "5"}, exp87))
        origin93 = next(row["selected_indices"] for row in rows93 if row.get("selector_name") == "nonpos_weak_alert_replace")
        self.assertEqual(frozenset(origin93), paper93)
        self.assertEqual(paper93, frozenset({0}))

    def test_executable_origin_exp134_and_exp135_cover_positive_and_negative_priority_membership(self) -> None:
        """Executable Block-B/C selectors agree with paper for eligible and rejected reviews."""
        from paper.equipment_anomaly_detection.hard_route_paper import replay_exp134
        parse = lambda value: {int(item) for item in str(value).split()} if value else set()
        train = np.arange(200., dtype=float)
        def blocks(*, block_b_support: bool) -> ConfirmationBlocks:
            a_test = np.array([0., 250., 0.]) if block_b_support else np.array([250., 0., 0.])
            c_test = np.array([0., 250., 0.]) if block_b_support else np.array([250., 0., 0.])
            return ConfirmationBlocks(
                FullFitScoreView("a", train, a_test, .015),
                FullFitScoreView("b", train, np.array([250., 250., 0.]), .015),
                FullFitScoreView("b", train, np.array([250., 250., 0.]), .01),
                FullFitScoreView("c", train, c_test, .01),
            )
        exp134 = _origin_functions("run_experiment_134_block_b_review_tail_guard.py", ("run_one",))["run_one"]
        exp134.__globals__.update({
            "load_dataset_data": lambda _name: (np.zeros((12, 3)), np.zeros((3, 3)), np.zeros(3)), "z_normalize": lambda values: values,
            "parse_indices": parse, "RATES": (.005, .01, .015), "np": np,
            "count_cap_threshold": lambda _scores, rate: (196.0 if rate == .015 else 197.0, rate, 0),
            "base": SimpleNamespace(make_row=lambda _d, _y, _hard, config, review, _reason, _diagnostics: {"config": config, "review": set(review)}),
        })
        exp135 = _origin_functions("run_experiment_135_block_c_review_confirmation.py", ("run_one",))["run_one"]
        exp135.__globals__.update({
            "load_dataset_data": lambda _name: (np.zeros((12, 3)), np.zeros((3, 3)), np.zeros(3)), "z_normalize": lambda values: values,
            "parse_indices": parse, "as_float": float,
            "base": SimpleNamespace(make_row=lambda _d, _y, _hard, config, review, _reason, _diagnostics: {"config": config, "review": set(review)}),
        })
        for support, expected in ((False, frozenset({0})), (True, frozenset())):
            current = blocks(block_b_support=support)
            exp93_index, review_index = (0, 1) if support else (2, 0)
            exp134.__globals__["base"].block_scores = lambda _train, _test, current=current: (train, current.block_a.test_score, train, current.block_b_01.test_score, 200, 0)
            exp135.__globals__["block_c_candidates"] = lambda _train, _test, current=current: (set(current.block_c.candidate), current.block_c.test_score, 200, 0, .01, 0)
            paper = replay_exp133_134_135_137(frozenset({exp93_index}), current)
            paper134 = replay_exp134(frozenset({exp93_index}), current)
            rows134 = exp134(("fixture", {"selected_indices": str(exp93_index)}, {}))
            origin134 = next(row["review"] for row in rows134 if row["config"] == "review_block_ba_tail_pct_1p0_top1")
            self.assertEqual(frozenset(origin134), paper134.review)
            self.assertEqual(paper134.review, frozenset({review_index}))
            rows135 = exp135(("fixture", {"selected_indices": str(exp93_index), "review_candidate_indices": " ".join(str(index) for index in sorted(origin134)), "train_normal_count": "200", "tiny_train": "0"}, {"high_confidence_count": "1" if support else "0", "standard_confidence_count": "0" if support else "1"}))
            origin135 = next(row["review"] for row in rows135 if row["config"] == "review_tail1pct_all_standard_and_block_c")
            self.assertEqual(frozenset(origin135), paper.priority)
            self.assertEqual(paper.priority, expected)

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

    def test_capable_ten_frozen_db_raw_smoke_freezes_then_compares_exp137_and_exp236_artifacts(self) -> None:
        """Ten DB rows freeze before separate original Exp137 and Exp236 artifact comparisons."""
        oracle = Path("/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python")
        if not oracle.is_file():
            self.skipTest("capable frozen oracle interpreter is unavailable")
        code = r'''
import csv, json, logging, tempfile
from pathlib import Path
logging.FileHandler=lambda *a, **k: logging.StreamHandler()
import numpy as np
import run_experiment_132_block_b_review_integration as exp132
import run_experiment_135_block_c_review_confirmation as exp135
from paper.equipment_anomaly_detection.config_paper import FROZEN_CONFIG
from paper.equipment_anomaly_detection.data_paper import load_population, load_dataset_values
from paper.equipment_anomaly_detection.hard_route_paper import freeze_canonical_smoke, minimum_source_expansion, replay_exp74d_89_90_93, replay_exp133_134_135_137, union_three_source_hard
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import score_hard_sources
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import build_legacy_route_evidence
from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import z_normalize_paper
names=load_population(FROZEN_CONFIG.database_path)[:10]
assert names == ('ACSF1_normal_0','ACSF1_normal_1','ACSF1_normal_2','ACSF1_normal_3','ACSF1_normal_4','ACSF1_normal_5','ACSF1_normal_6','ACSF1_normal_7','ACSF1_normal_8','ACSF1_normal_9')
rows={}; lane_rows={}; exp236_rows={}
for name in names:
  values=load_dataset_values(FROZEN_CONFIG.database_path,name); evidence=build_legacy_route_evidence(values)
  train,test=z_normalize_paper(np.asarray(values.train_rows)),z_normalize_paper(np.asarray(values.test_rows))
  a,at,b,bt,*_=exp132.block_scores(train,test); ci,cs,*_=exp135.block_c_candidates(train,test)
  np.testing.assert_allclose(a,evidence.confirmation_blocks.block_a.train_score,rtol=1e-6,atol=1e-6); np.testing.assert_allclose(at,evidence.confirmation_blocks.block_a.test_score,rtol=1e-6,atol=1e-6)
  np.testing.assert_allclose(b,evidence.confirmation_blocks.block_b_015.train_score,rtol=1e-6,atol=1e-6); np.testing.assert_allclose(bt,evidence.confirmation_blocks.block_b_015.test_score,rtol=1e-6,atol=1e-6)
  np.testing.assert_allclose(cs,evidence.confirmation_blocks.block_c.test_score,rtol=1e-6,atol=1e-6); assert ci == set(evidence.confirmation_blocks.block_c.candidate)
  _,_,_,exp93=replay_exp74d_89_90_93(evidence); lanes=replay_exp133_134_135_137(exp93,evidence.confirmation_blocks)
  lane_rows[name]=lanes; rows[name]=frozenset(lanes.hard|lanes.standard|lanes.priority)
  exp234=minimum_source_expansion(lanes,score_hard_sources(values),alpha=.01); exp236=union_three_source_hard(lanes,exp234)
  exp236_rows[name]=(exp236.hard,exp236.standard,exp236.priority)
with tempfile.TemporaryDirectory() as directory:
  frozen=freeze_canonical_smoke(rows,Path(directory)/'frozen-new-route.json')
  assert frozen.datasets == names and frozen.path.is_file()
  artifact=Path('results/exp137/experiment_137_operational_triage_results.csv')
  assert artifact.is_file(), artifact
  historical={row['dataset_name']: tuple(frozenset(int(x) for x in row[field].split() if x) for field in ('hard_alert_indices','standard_review_indices','priority_review_indices')) for row in csv.DictReader(artifact.open(encoding='utf-8')) if row['dataset_name'] in names}
  assert set(historical) == set(names)
  assert all((lanes.hard,lanes.standard,lanes.priority) == historical[name] for name, lanes in lane_rows.items())
  exp236_paths=(Path('outputs/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl'),Path('/Users/mac/Documents/Dataset/experiments/exp236_direct_run_antigravity/exp236_frozen_predictions.jsonl'))
  for exp236_path in exp236_paths:
    assert exp236_path.is_file(), exp236_path
    historical236={row['dataset']: tuple(frozenset(int(i) for i in row[field]) for field in ('hard_indices','standard_review_indices','priority_review_indices')) for row in (json.loads(line) for line in exp236_path.read_text(encoding='utf-8').splitlines() if line.strip()) if row['dataset'] in names}
    assert set(historical236) == set(names)
    assert all(exp236_rows[name] == historical236[name] for name in names)
print('frozen-db-10-exp137-exp236-artifact-parity',','.join(names))
'''
        completed = subprocess.run([str(oracle), "-c", code], cwd=Path.cwd(), check=True, capture_output=True, text=True, timeout=360)
        self.assertIn("frozen-db-10-exp137-exp236-artifact-parity", completed.stdout)

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
