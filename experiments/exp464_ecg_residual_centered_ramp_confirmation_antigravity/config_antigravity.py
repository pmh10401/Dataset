"""Frozen Task 2 inputs for Exp464."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping


def _exact_antigravity(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _exact_antigravity(item, wanted)
            for item, wanted in zip(value, expected, strict=True)  # type: ignore[arg-type]
        )
    if isinstance(expected, MappingProxyType):
        if tuple(value) != tuple(expected):  # type: ignore[arg-type]
            return False
        return all(  # type: ignore[arg-type]
            _exact_antigravity(value[key], expected[key])  # type: ignore[index]
            for key in expected
        )
    return value == expected


def _validate_summary_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class HistoricalPeriodThresholdAntigravity:
    seed: int
    fold_index: int
    period_change_auc_min: float

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in (20260721, 20261721):
            raise ValueError("seed must be one of the frozen seeds")
        if type(self.fold_index) is not int or not 0 <= self.fold_index < 5:
            raise ValueError("fold_index must be one of the frozen blocked folds")
        _validate_summary_float_antigravity(
            self.period_change_auc_min,
            "period_change_auc_min",
        )


@dataclass(frozen=True, slots=True)
class Exp463MethodSummarySnapshotAntigravity:
    method: str
    fold_count: int
    passed_count: int
    failed_count: int
    min_gradual_drift_auc: float
    min_period_change_auc: float
    min_pooled_auc: float
    max_normal_wilson_upper: float
    all_scores_finite: bool

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in ("C0", "C1", "C2", "C7"):
            raise ValueError("method must be one of the frozen Exp463 control IDs")
        if type(self.fold_count) is not int or self.fold_count <= 0:
            raise ValueError("fold_count must be a positive integer")
        if type(self.passed_count) is not int or self.passed_count < 0:
            raise ValueError("passed_count must be a non-negative integer")
        if type(self.failed_count) is not int or self.failed_count < 0:
            raise ValueError("failed_count must be a non-negative integer")
        if self.fold_count != self.passed_count + self.failed_count:
            raise ValueError("fold_count must equal passed_count + failed_count")
        _validate_summary_float_antigravity(
            self.min_gradual_drift_auc,
            "min_gradual_drift_auc",
        )
        _validate_summary_float_antigravity(
            self.min_period_change_auc,
            "min_period_change_auc",
        )
        _validate_summary_float_antigravity(self.min_pooled_auc, "min_pooled_auc")
        _validate_summary_float_antigravity(
            self.max_normal_wilson_upper,
            "max_normal_wilson_upper",
        )
        if type(self.all_scores_finite) is not bool:
            raise ValueError("all_scores_finite must be bool")


_METHOD_DEFINITIONS_ANTIGRAVITY = MappingProxyType(
    {
        "C0": "Exp463 M0 exact replay control",
        "C1": "Exp463 M2 exact replay control",
        "C2": "Exp463 M4 exact replay control",
        "C3": "template residual centered ramp rank only without M0 max or cosine gate",
        "C4": "same C3 rank with unconditional max against C0 and no cosine gate",
        "C5": "centered ramp confirmation without template subtraction",
        "C6": "template residual centered ramp rank with cosine >= 0.95 confirmed max against C0",
        "C7": "Exp463 M6 exact replay control",
        "C8": "template residual ramp confirmation without centering",
    }
)

_CONTROL_SUMMARIES_ANTIGRAVITY = MappingProxyType(
    {
        "C0": Exp463MethodSummarySnapshotAntigravity(
            method="C0",
            fold_count=10,
            passed_count=9,
            failed_count=1,
            min_gradual_drift_auc=0.8402306222021025,
            min_period_change_auc=0.907392736783864,
            min_pooled_auc=0.9212266108344651,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        ),
        "C1": Exp463MethodSummarySnapshotAntigravity(
            method="C1",
            fold_count=10,
            passed_count=7,
            failed_count=3,
            min_gradual_drift_auc=0.83450279161008,
            min_period_change_auc=0.8984897640963734,
            min_pooled_auc=0.9128093405764297,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        ),
        "C2": Exp463MethodSummarySnapshotAntigravity(
            method="C2",
            fold_count=10,
            passed_count=9,
            failed_count=1,
            min_gradual_drift_auc=0.8399225391076908,
            min_period_change_auc=0.9044753785020874,
            min_pooled_auc=0.9193733967104271,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        ),
        "C7": Exp463MethodSummarySnapshotAntigravity(
            method="C7",
            fold_count=10,
            passed_count=3,
            failed_count=7,
            min_gradual_drift_auc=0.7953196519289775,
            min_period_change_auc=0.8960376741612595,
            min_pooled_auc=0.9010220185101353,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        ),
    }
)

# Provenance: these exact 40 control digests were reproduced under the fixed
# reviewed Exp463 runtime and accepted only because the canonical
# method_fold_metrics_antigravity.jsonl bytes rehashed to the frozen tracked
# predecessor digest 84f610e437c59cb6dce4bd73e2ae8ec09c056d7a41178c3f7b790c0f01803fa6.
_CONTROL_ROW_DIGESTS_ANTIGRAVITY = MappingProxyType(
    {
        "C0:20260721:0": "b73749fb4e799885055a93c753e780618ce21f50e919e185cc98866543f5e0cd",
        "C0:20260721:1": "a8460f37ef0027237b6676a8aee4451d4a361cc714ed9bdbf73cf20d7e434e8e",
        "C0:20260721:2": "2d08399b85e832f61414282051838cb63b58f63b9e01dfc58f52025b79577d4c",
        "C0:20260721:3": "342e295f711c761520d7c978cf75af38df5e32e912090f61a368a0f36891f6dd",
        "C0:20260721:4": "6b99d71d878531a330aab940c315d3e7dc6fce48eec380e8e1f4b8ef1a3f4ac8",
        "C0:20261721:0": "f22c43349022b03296bf66ab803f6f865e0b171b8905dce23e0ffef375668a5d",
        "C0:20261721:1": "ca895ef18f6873ba6ede95b037ac97b6042f53ccfbeef72e50d18c5c3b4868b1",
        "C0:20261721:2": "18f11840468dcb5facb5231ad980f5ffaee006b26d3d7fee713d54820c9da0d7",
        "C0:20261721:3": "f8a54afafd64d16ff365444a1f792ee17c534d93048c3d19ce3065b731e3f2bf",
        "C0:20261721:4": "0fd5d81fe06591aaea9883e73943beb1657b588beea91145dcd4bcbd51729624",
        "C1:20260721:0": "47bdc328963202322f1b728db83bc885f1e98fdcfdc59263563e9f30c8719733",
        "C1:20260721:1": "2f819ca50542eed6c208afe8b542d2b4266b445e4a9897db3dcd02cdfb096f11",
        "C1:20260721:2": "88fc5ea2919699b907d1d8603dfb51c6ede9d664c795274775f4307c58c2ae6f",
        "C1:20260721:3": "e7dcb6a3d9e6335d3f62b8d80b15c2f0095183852a4b65163ac94013aacbbe2b",
        "C1:20260721:4": "cd5999c8046d3a8eb93d88bec7a34884c5d33d929fb37986394606a244624af4",
        "C1:20261721:0": "9c3316f29ceace9aad9ca4cfd3ccefb0a632b10dba4d076b918adb4e62489405",
        "C1:20261721:1": "c0b8f9ae56e8cec7d3bd634e27bdb0b6e858410eae0cc65b9e061486417363d2",
        "C1:20261721:2": "f9ee1fff7a5a9177c1f09e1628a3fe2c27e8eefc108c8a895989e6a401b3b96e",
        "C1:20261721:3": "02475414d946064f9f08d1f47a8a0d57d4fadb45f0a59f5827215cd1603eaf4d",
        "C1:20261721:4": "9c52f294898ced7037021c2101f6b6f3175426b3207697964a492284de1fe487",
        "C2:20260721:0": "608ab51d3296359f8ee2b6c0e8b5c4d92321ff7f286134032d6c9be8aa3f0745",
        "C2:20260721:1": "37b5c51f6b6974228a4defde23cd8cb707af5801c7fa025166f407a13b8b6b2b",
        "C2:20260721:2": "a8869246d398ed7c8bf908f44776cedee930813b4da3da6ee361543a3831eb12",
        "C2:20260721:3": "f68d9a52a07c979dbe6e908f42d52656a2340841e4f6e62b6276b198a44d8309",
        "C2:20260721:4": "b0dd8a679d3b16c90b81b1f435d8944aee56a18301767f0eb92b0fd754fdb6a6",
        "C2:20261721:0": "98cc35d6d6f13326b929482ddd145b0e82ccb0732e94559a6b9efed68bc894e9",
        "C2:20261721:1": "467e99f1e4bf016d1a461e3997ca617b9c0c0cb3e4c7729ad543ce5f9116727f",
        "C2:20261721:2": "5228d499e751e12d5545ed6b4fdc62ad96e11a03e936227beca4ea4771f93d66",
        "C2:20261721:3": "5505e16d9ba875b76ade085f50e6a61a6a604421b68dde2473951e4467ea6030",
        "C2:20261721:4": "76271582290c6cbe9fbb75dd30daf939512d88810374dace13a9185dc088c6d8",
        "C7:20260721:0": "022e4ccd7c6a2d1834f8699d581f6dcf65404ef43b37a4e29d3e9612dcce93e2",
        "C7:20260721:1": "e830fe8fd9a691165a82662d3eac1020de6702f80013e7122b6bbdd5893d4902",
        "C7:20260721:2": "95e6f3a9edb841c79b09d6379cde321c5c0987edd081f95dfa1868bb46124359",
        "C7:20260721:3": "238d6fdcb61980d5731c7027790738db6323e146377fd9a7857e70360b8f51ac",
        "C7:20260721:4": "661f2bf62244a6499bbc840bfb63d53b503ad532398029545d07aba1d9391f09",
        "C7:20261721:0": "323e51da0c27f1b05cbadf99191d5cacf88c4cd91240246bd91669468b75292c",
        "C7:20261721:1": "c9157872b75212f55b7a7ae48b4cb1fd77c7c80ff3d264853116211cf9b00c11",
        "C7:20261721:2": "eb1ed107eab3d5fbb735fae31e18ecac733daee23afdba7b6ba9bd1847cfccf6",
        "C7:20261721:3": "a03ec96367339d0ee376eb91dff24663eca6583fd8f247ae2dc282857df14034",
        "C7:20261721:4": "5ff52cddd96bf3bd8ac0aa55f8efe60396261629b0839a11f57a77284dea4be2",
    }
)


@dataclass(frozen=True, slots=True)
class Exp464ConfigAntigravity:
    experiment: str
    kind: str
    run_identity: str
    official_result_root: str
    approved_design_sha256: str
    fixed_interpreter: str
    dataset: str
    data_split: str
    normal2_train_rows: int
    normal2_width: int
    hydra_seeds: tuple[int, int]
    folds: int
    onset: int
    cosine_min: float
    method_order: tuple[str, str, str, str, str, str, str, str, str]
    method_definitions: Mapping[str, str]
    primary_candidate: str
    per_type_auc_min: float
    pooled_auc_min: float
    normal_wilson_upper_max: float
    gradual_drift_auc_min: float
    gradual_drift_auc_strictly_above_exp463: float
    period_change_auc_global_min: float
    historical_period_fold_thresholds: tuple[
        HistoricalPeriodThresholdAntigravity,
        HistoricalPeriodThresholdAntigravity,
    ]
    predecessor_experiment: str
    predecessor_terminal_closeout_commit: str
    predecessor_task5_report_sha256: str
    predecessor_private_run_count: int
    predecessor_result_row_count: int
    predecessor_result_failed_row_count: int
    predecessor_result_summary_count: int
    predecessor_result_leaf_count: int
    predecessor_method_fold_metrics_sha256: str
    predecessor_artifact_map_sha256: str
    predecessor_control_summaries: Mapping[str, Exp463MethodSummarySnapshotAntigravity]
    predecessor_control_row_digests: Mapping[str, str]
    label_query_count_must_remain: int
    official_evaluate_count_must_remain: int
    official_inspect_count_must_remain: int

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp464 configuration")

    @property
    def train_shape(self) -> tuple[int, int]:
        return (self.normal2_train_rows, self.normal2_width)

    @property
    def blocked_fold_count(self) -> int:
        return self.folds

    @property
    def seed_pair(self) -> tuple[int, int]:
        return self.hydra_seeds


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp464",
        "kind": "ecg_residual_centered_ramp_confirmation_qualification",
        "run_identity": "exp464-ecg-residual-centered-ramp-confirmation-antigravity-20260821",
        "official_result_root": "/Users/mac/Documents/Dataset/exp464_ecg_residual_centered_ramp_confirmation_antigravity_20260821",
        "approved_design_sha256": "d0e36ae176598810289fbd5693a5e582ca6f0821a8b8fcd52edd1bcb96ec1f9f",
        "fixed_interpreter": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
        "dataset": "ECG5000_normal_2",
        "data_split": "TRAIN_only",
        "normal2_train_rows": 1413,
        "normal2_width": 140,
        "hydra_seeds": (20260721, 20261721),
        "folds": 5,
        "onset": 56,
        "cosine_min": 0.95,
        "method_order": ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
        "method_definitions": _METHOD_DEFINITIONS_ANTIGRAVITY,
        "primary_candidate": "template_residual_centered_ramp_confirmed_max",
        "per_type_auc_min": 0.85,
        "pooled_auc_min": 0.90,
        "normal_wilson_upper_max": 0.05,
        "gradual_drift_auc_min": 0.85,
        "gradual_drift_auc_strictly_above_exp463": 0.8402306222021025,
        "period_change_auc_global_min": 0.907392736783864,
        "historical_period_fold_thresholds": (
            HistoricalPeriodThresholdAntigravity(
                seed=20260721,
                fold_index=3,
                period_change_auc_min=0.907392736783864,
            ),
            HistoricalPeriodThresholdAntigravity(
                seed=20261721,
                fold_index=4,
                period_change_auc_min=0.9216148584075248,
            ),
        ),
        "predecessor_experiment": "Exp463",
        "predecessor_terminal_closeout_commit": "147225831e9c13e9df5043fa88757bb03e01b79a",
        "predecessor_task5_report_sha256": "e43abf66e43921941dece398514858be91e39d121c1173b26b352b67ba78e8e5",
        "predecessor_private_run_count": 2,
        "predecessor_result_row_count": 70,
        "predecessor_result_failed_row_count": 30,
        "predecessor_result_summary_count": 7,
        "predecessor_result_leaf_count": 7,
        "predecessor_method_fold_metrics_sha256": "84f610e437c59cb6dce4bd73e2ae8ec09c056d7a41178c3f7b790c0f01803fa6",
        "predecessor_artifact_map_sha256": "25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5",
        "predecessor_control_summaries": _CONTROL_SUMMARIES_ANTIGRAVITY,
        "predecessor_control_row_digests": _CONTROL_ROW_DIGESTS_ANTIGRAVITY,
        "label_query_count_must_remain": 0,
        "official_evaluate_count_must_remain": 0,
        "official_inspect_count_must_remain": 0,
    }
)


FROZEN_CONFIG_ANTIGRAVITY = Exp464ConfigAntigravity(**_FROZEN_VALUES_ANTIGRAVITY)  # type: ignore[arg-type]


__all__ = (
    "Exp463MethodSummarySnapshotAntigravity",
    "Exp464ConfigAntigravity",
    "FROZEN_CONFIG_ANTIGRAVITY",
    "HistoricalPeriodThresholdAntigravity",
)
