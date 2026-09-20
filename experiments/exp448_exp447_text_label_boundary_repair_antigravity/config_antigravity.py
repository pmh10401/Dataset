"""Exact, evaluation-only configuration frozen for Exp448."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import NamedTuple


_FROZEN_VALUES_ANTIGRAVITY = {
    "experiment_id": "Exp448",
    "artifact_schema": "exp448",
    "run_identity": "exp448-exp447-text-label-boundary-repair-antigravity-20260812",
    "run_root": "/Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812",
    "dataset_count": 1117,
    "label_rows": 195900,
    "source_count": 30,
    "stable_count": 26,
    "unstable_count": 4,
    "bootstrap_seed": 44720260811,
    "bootstrap_samples": 20000,
    "bootstrap_draw_sha256": "f960dec37c48599a09d95f7a766d160d133eb9b085536974962bb5412dea993e",
    "baseline_confusion": (1915, 303, 2003, 191679),
    "exp442_source_30_confusion": (1930, 318, 1988, 191664),
    "exp442_evaluation_sha256": "f432ffdf530b2e32bc7a62fec6c7d07cff2eff9f3049489b5e03136b445a33cf",
    "exp442_evaluation_artifact_map_sha256": "e1155acf05e00935710243e875eaeb2d13047d6236d026ee93fe654fc33c670b",
    "exp404_hard_prediction_sha256": "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
    "exp446_prediction_sha256": "602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100",
    "exp446_manifest_file_sha256": "bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11",
    "exp446_artifact_map_file_sha256": "d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d",
    "scientific_success_criterion": None,
    "promotion_allowed": False,
    "review_only": True,
    "operational_alert_eligible": False,
    "scientific_winner_selected": False,
    "same_experiment_reselection_allowed": False,
    "test_informed_reselection_allowed": False,
}


@dataclass(frozen=True)
class Exp448FrozenConfigAntigravity:
    """Constants that must not vary across post-freeze evaluation replays."""

    experiment_id: str = "Exp448"
    artifact_schema: str = "exp448"
    run_identity: str = "exp448-exp447-text-label-boundary-repair-antigravity-20260812"
    run_root: str = "/Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812"
    dataset_count: int = 1117
    label_rows: int = 195900
    source_count: int = 30
    stable_count: int = 26
    unstable_count: int = 4
    bootstrap_seed: int = 44720260811
    bootstrap_samples: int = 20000
    bootstrap_draw_sha256: str = "f960dec37c48599a09d95f7a766d160d133eb9b085536974962bb5412dea993e"
    baseline_confusion: tuple[int, int, int, int] = (1915, 303, 2003, 191679)
    exp442_source_30_confusion: tuple[int, int, int, int] = (1930, 318, 1988, 191664)
    exp442_evaluation_sha256: str = "f432ffdf530b2e32bc7a62fec6c7d07cff2eff9f3049489b5e03136b445a33cf"
    exp442_evaluation_artifact_map_sha256: str = "e1155acf05e00935710243e875eaeb2d13047d6236d026ee93fe654fc33c670b"
    exp404_hard_prediction_sha256: str = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"
    exp446_prediction_sha256: str = "602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100"
    exp446_manifest_file_sha256: str = "bbaf9506e17045b90eb9f72c41fad19d0f8d5459084ead329c6441d8af303c11"
    exp446_artifact_map_file_sha256: str = "d28129dc6d6ca5097c31c26e44feadcabc422493fd9e2be7abbdaacc403f6d0d"
    scientific_success_criterion: None = None
    promotion_allowed: bool = False
    review_only: bool = True
    operational_alert_eligible: bool = False
    scientific_winner_selected: bool = False
    same_experiment_reselection_allowed: bool = False
    test_informed_reselection_allowed: bool = False

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"{name} is frozen")
        if (self.experiment_id, self.artifact_schema, self.run_identity) != (
            "Exp448", "exp448", "exp448-exp447-text-label-boundary-repair-antigravity-20260812",
        ):
            raise ValueError("Exp448 identity is frozen")
        if (self.dataset_count, self.label_rows, self.source_count, self.stable_count, self.unstable_count) != (1117, 195900, 30, 26, 4):
            raise ValueError("Exp448 population is frozen")
        if self.source_count != self.stable_count + self.unstable_count:
            raise ValueError("Exp448 lane population is frozen")
        if (self.bootstrap_seed, self.bootstrap_samples, self.baseline_confusion, self.exp442_source_30_confusion) != (44720260811, 20000, (1915, 303, 2003, 191679), (1930, 318, 1988, 191664)):
            raise ValueError("Exp448 evaluation statistics are frozen")
        if any(type(value) is not int or value < 0 for value in (*self.baseline_confusion, *self.exp442_source_30_confusion, self.dataset_count, self.label_rows, self.source_count, self.stable_count, self.unstable_count, self.bootstrap_seed, self.bootstrap_samples)):
            raise ValueError("Exp448 counts must be exact non-negative integers")
        hashes = (self.bootstrap_draw_sha256, self.exp404_hard_prediction_sha256, self.exp446_prediction_sha256, self.exp446_manifest_file_sha256, self.exp446_artifact_map_file_sha256, self.exp442_evaluation_sha256, self.exp442_evaluation_artifact_map_sha256)
        if any(type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in hashes):
            raise ValueError("Exp448 source hashes are invalid")
        if self.scientific_success_criterion is not None or self.promotion_allowed or not self.review_only:
            raise ValueError("Exp448 remains descriptive and review-only")
        if self.operational_alert_eligible or self.scientific_winner_selected or self.same_experiment_reselection_allowed or self.test_informed_reselection_allowed:
            raise ValueError("Exp448 promotion is prohibited")


FROZEN_CONFIG_ANTIGRAVITY = Exp448FrozenConfigAntigravity()


_IDENTITY_PROFILE_KEYS_ANTIGRAVITY = frozenset({
    "experiment_id", "artifact_schema", "run_identity", "run_root",
})


class PresentProfileComparisonAntigravity(NamedTuple):
    """A controlled-fork profile comparison for the Task2-present modules."""

    equal: bool
    changed_keys: frozenset[str]
    unexpected_keys: frozenset[str]


def compare_present_profile_antigravity(predecessor: object, successor: object) -> PresentProfileComparisonAntigravity:
    """Allow exactly the four Exp448 identity keys to differ from Exp447."""
    try:
        before = asdict(predecessor)
        after = asdict(successor)
    except TypeError as error:
        raise ValueError("controlled-fork profiles must be dataclass instances") from error
    if set(before) != set(after):
        raise ValueError("controlled-fork profile keys differ")
    changed = frozenset(name for name in before if before[name] != after[name])
    unexpected = changed - _IDENTITY_PROFILE_KEYS_ANTIGRAVITY
    return PresentProfileComparisonAntigravity(not unexpected, changed, unexpected)


class _ControlledForkAstNormalizerAntigravity(ast.NodeTransformer):
    """Erase only approved Exp447/Exp448 identity spelling from a module AST."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = node.id.replace("Exp447", "ExpExperiment").replace("Exp448", "ExpExperiment")
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = node.name.replace("Exp447", "ExpExperiment").replace("Exp448", "ExpExperiment")
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if type(node.value) is str:
            value = node.value
            value = value.replace("exp447-exp446-postfreeze-performance-evaluation-antigravity-20260811", "<IDENTITY>")
            value = value.replace("exp448-exp447-text-label-boundary-repair-antigravity-20260812", "<IDENTITY>")
            value = value.replace("/Users/mac/Documents/Dataset/exp447_exp446_postfreeze_performance_evaluation_antigravity_20260811", "<ROOT>")
            value = value.replace("/Users/mac/Documents/Dataset/exp448_exp447_text_label_boundary_repair_antigravity_20260812", "<ROOT>")
            value = value.replace("Exp447", "ExpExperiment").replace("Exp448", "ExpExperiment")
            value = value.replace("exp447", "expExperiment").replace("exp448", "expExperiment")
            node.value = value
        return node


_DIFFERENTIAL_HELPER_NAMES_ANTIGRAVITY = frozenset({
    "_IDENTITY_PROFILE_KEYS_ANTIGRAVITY", "_DIFFERENTIAL_HELPER_NAMES_ANTIGRAVITY",
    "PresentProfileComparisonAntigravity",
    "compare_present_profile_antigravity", "_ControlledForkAstNormalizerAntigravity",
    "_normalized_module_antigravity", "normalized_pair_equal_antigravity",
})


def _normalized_module_antigravity(source: str) -> ast.Module:
    tree = ast.parse(source)
    kept: list[ast.stmt] = []
    for node in tree.body:
        if getattr(node, "name", None) in _DIFFERENTIAL_HELPER_NAMES_ANTIGRAVITY:
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in _DIFFERENTIAL_HELPER_NAMES_ANTIGRAVITY
            for target in node.targets
        ):
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ) and isinstance(node.value, ast.List):
            node.value.elts = [
                value for value in node.value.elts
                if not (isinstance(value, ast.Constant) and value.value in _DIFFERENTIAL_HELPER_NAMES_ANTIGRAVITY)
            ]
        if isinstance(node, ast.Import):
            node.names = [alias for alias in node.names if alias.name != "ast"]
            if not node.names:
                continue
        if isinstance(node, ast.ImportFrom):
            if node.module == "dataclasses":
                node.names = [alias for alias in node.names if alias.name != "asdict"]
            if node.module == "typing":
                node.names = [alias for alias in node.names if alias.name != "NamedTuple"]
            if not node.names:
                continue
        kept.append(node)
    tree.body = kept
    return _ControlledForkAstNormalizerAntigravity().visit(tree)  # type: ignore[return-value]


def normalized_pair_equal_antigravity(predecessor_source: object, successor_source: object) -> bool:
    """Compare Task2 paired modules after narrow approved identity normalization."""
    if type(predecessor_source) is not str or type(successor_source) is not str:
        raise ValueError("controlled-fork source must be exact text")
    try:
        before = _normalized_module_antigravity(predecessor_source)
        after = _normalized_module_antigravity(successor_source)
    except SyntaxError as error:
        raise ValueError("controlled-fork source is not valid Python") from error
    return ast.dump(ast.fix_missing_locations(before), include_attributes=False) == ast.dump(ast.fix_missing_locations(after), include_attributes=False)


__all__ = [
    "Exp448FrozenConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY",
    "PresentProfileComparisonAntigravity", "compare_present_profile_antigravity",
    "normalized_pair_equal_antigravity",
]
