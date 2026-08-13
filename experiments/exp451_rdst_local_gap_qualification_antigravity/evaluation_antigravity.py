"""Single-session Exp451 TEST evaluation and immutable local publication."""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity

from .artifacts_antigravity import canonical_json_bytes_antigravity, write_create_only_antigravity
from .contracts_antigravity import FrozenPredictionAuthorityAntigravity
from .predecessor_antigravity import Exp448PredecessorAuthorityAntigravity, verify_exp448_predecessor_antigravity
from .prediction_antigravity import validate_frozen_prediction_authority_antigravity
from .sources_antigravity import LabelSnapshotAntigravity, open_ordered_label_session_antigravity
from .evaluation_codec_antigravity import DatasetEvaluationAntigravity, FrozenEvaluationAntigravity, baseline_authority_sha256_antigravity, dataset_payload_antigravity, derive_evaluation_antigravity, f1_antigravity, result_payload_antigravity

OFFICIAL_EVALUATION_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp451_rdst_local_gap_qualification_evaluation_antigravity_20260813")
BASELINE_GLOBAL_ANTIGRAVITY = ConfusionAntigravity(1930, 314, 1988, 191668)
EXP448_HASHES_ANTIGRAVITY=("ea9856c7de039697ed4f352864def0fde1c0d336714e68c9b0cd586d2c796f05","a58a74f69dcc13a64efb2a679ee7e93eca40ac8c5db87b29b12ed11ea068b21c","54ac58d89367ec5396dafdb180777f787384ffbd21ead1a96764bfe627bbddd5")
EXP448_BASELINE_AUTHORITY_SHA256_ANTIGRAVITY="19cbc7f6c52087c71a1637d8ea20d52cb9c332b2439f3338727c186c0f5d5160"


@dataclass(frozen=True, slots=True)
class EvaluationEvidenceRowAntigravity:
    sequence: int
    identity: tuple[int, str, str]
    target_length: int
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or type(self.identity) is not tuple or len(self.identity) != 3 or type(self.target_length) is not int or type(self.indices) is not tuple:
            raise ValueError("evaluation evidence row type differs")


@dataclass(frozen=True, slots=True)
class EvaluationDependenciesAntigravity:
    prediction_validator: Callable[[FrozenPredictionAuthorityAntigravity], FrozenPredictionAuthorityAntigravity]
    predecessor_loader: Callable[[], Exp448PredecessorAuthorityAntigravity]
    evidence_rows: tuple[EvaluationEvidenceRowAntigravity, ...]
    label_opener_factory: Callable[[], Callable[[FrozenPredictionAuthorityAntigravity], LabelSnapshotAntigravity]]
    evaluation_root: Path


def _f1_antigravity(value: ConfusionAntigravity) -> float:
    return f1_antigravity(value)


def qualifies_antigravity(confusion: object, target_fn_delta: object) -> bool:
    if type(confusion) is not ConfusionAntigravity or type(target_fn_delta) is not int:
        raise ValueError("qualification inputs must have exact types")
    return confusion.fn < 1988 and confusion.fp <= 314 and _f1_antigravity(confusion) >= 0.6264199935 and target_fn_delta < 0


def _confusion_antigravity(labels: np.ndarray, indices: tuple[int, ...]) -> ConfusionAntigravity:
    selected = np.zeros(len(labels), dtype=np.bool_); selected[list(indices)] = True
    positive = labels == 1
    return ConfusionAntigravity(int(np.count_nonzero(selected & positive)), int(np.count_nonzero(selected & ~positive)), int(np.count_nonzero(~selected & positive)), int(np.count_nonzero(~selected & ~positive)))


def _add_antigravity(values: tuple[ConfusionAntigravity, ...]) -> ConfusionAntigravity:
    return ConfusionAntigravity(*(sum(getattr(value, field) for value in values) for field in ("tp","fp","fn","tn")))


def _sub_add_antigravity(global_value: ConfusionAntigravity, old: ConfusionAntigravity, new: ConfusionAntigravity) -> ConfusionAntigravity:
    values = tuple(getattr(global_value, f)-getattr(old,f)+getattr(new,f) for f in ("tp","fp","fn","tn"))
    return ConfusionAntigravity(*values)


def _conf_payload_antigravity(value: ConfusionAntigravity) -> dict[str,int]: return dataclasses.asdict(value)


def _dataset_payload_antigravity(row: DatasetEvaluationAntigravity) -> dict[str,object]:
    return dataset_payload_antigravity(row)

def _baseline_authority_sha256_antigravity(predecessor: Exp448PredecessorAuthorityAntigravity) -> str:
    return baseline_authority_sha256_antigravity(predecessor)


def _result_payload_antigravity(result: FrozenEvaluationAntigravity) -> dict[str,object]:
    return result_payload_antigravity(result)


def _publish_antigravity(root: Path, result: FrozenEvaluationAntigravity, authority: FrozenPredictionAuthorityAntigravity, predecessor: Exp448PredecessorAuthorityAntigravity) -> None:
    if root.exists(): raise ValueError("evaluation root already exists")
    root.mkdir(mode=0o700)
    from .dashboard_export_antigravity import public_result_antigravity
    leaves={"context_antigravity.json":{"schema":"exp451.evaluation.context.v1","reviewed_commit":result.reviewed_commit,"prediction_authority":authority.to_payload_antigravity(),"predecessor_hashes":list(result.predecessor_hashes),"exp448_baseline_authority_sha256":_baseline_authority_sha256_antigravity(predecessor)},"datasets_antigravity.json":{"schema":"exp451.evaluation.datasets.v1","rows":[_dataset_payload_antigravity(x) for x in result.dataset_rows]},"result_antigravity.json":_result_payload_antigravity(result),"dashboard_antigravity.json":public_result_antigravity(result)}
    hashes={}
    for name,value in leaves.items():
        payload=canonical_json_bytes_antigravity(value); write_create_only_antigravity(root/name,payload); hashes[name]=hashlib.sha256(payload).hexdigest()
    artifact_map={"schema":"exp451.evaluation.artifact_map.v1","leaf_sha256":hashes}
    map_bytes=canonical_json_bytes_antigravity(artifact_map); write_create_only_antigravity(root/"artifact_map_antigravity.json",map_bytes)
    manifest={"schema":"exp451.evaluation.manifest.v1","inventory":[*leaves,"artifact_map_antigravity.json","manifest_antigravity.json"],"artifact_map_sha256":hashlib.sha256(map_bytes).hexdigest(),"result_sha256":hashes["result_antigravity.json"],"dashboard_sha256":hashes["dashboard_antigravity.json"]}
    write_create_only_antigravity(root/"manifest_antigravity.json",canonical_json_bytes_antigravity(manifest))


def _evaluate_with_dependencies_antigravity(authority: FrozenPredictionAuthorityAntigravity, reviewed_commit: str, dependencies: EvaluationDependenciesAntigravity) -> FrozenEvaluationAntigravity:
    if type(authority) is not FrozenPredictionAuthorityAntigravity or type(reviewed_commit) is not str or authority.reviewed_commit != reviewed_commit or type(dependencies) is not EvaluationDependenciesAntigravity:
        raise ValueError("evaluation authority differs")
    validated=dependencies.prediction_validator(authority)
    if type(validated) is not FrozenPredictionAuthorityAntigravity or validated != authority: raise ValueError("prediction validation differs")
    predecessor=dependencies.predecessor_loader()
    identities=tuple((row.dataset_order,row.family,name) for row,name in zip(predecessor.target_baseline_rows,predecessor.target_dataset_names,strict=True))
    exact=tuple(EvaluationEvidenceRowAntigravity(i,identities[i],predecessor.target_test_lengths[i],authority.candidate_indices[i]) for i in range(48))
    if predecessor.label_access_count != 0 or predecessor.population != 1117 or predecessor.target_dataset_count != 48 or predecessor.target_test_rows != 72300 or predecessor.baseline_confusion != (1930,314,1988,191668) or (predecessor.artifact_map_sha256,predecessor.evaluation_sha256,predecessor.decision_sha256)!=EXP448_HASHES_ANTIGRAVITY or authority.dataset_identities != identities or authority.target_lengths != predecessor.target_test_lengths or dependencies.evidence_rows != exact:
        raise ValueError("pre-label evidence gate differs")
    opener=dependencies.label_opener_factory(); snapshot=opener(authority)
    if type(snapshot) is not LabelSnapshotAntigravity or snapshot.dataset_order != tuple(x[2] for x in identities) or snapshot.lengths != authority.target_lengths or snapshot.row_count != 72300 or snapshot.session_count != 1 or len(snapshot.labels) != 48:
        raise ValueError("label snapshot differs")
    rows=[]
    for i,(identity,length,labels,indices,baseline) in enumerate(zip(identities,authority.target_lengths,snapshot.labels,authority.candidate_indices,predecessor.target_baseline_rows,strict=True)):
        if type(baseline) is not DatasetBaselineAntigravity or len(labels)!=length: raise ValueError("baseline or label row differs")
        candidate=_confusion_antigravity(labels,indices); stable=baseline.stable_candidates; chosen=frozenset(indices); positives=frozenset(int(x) for x in np.flatnonzero(labels))
        stable_sha=hashlib.sha256(canonical_json_bytes_antigravity(sorted(stable))).hexdigest()
        rows.append(DatasetEvaluationAntigravity(i,identity,length,baseline.confusion,candidate,len((chosen-stable)&positives),len((chosen-stable)-positives),len((stable-chosen)&positives),len((stable-chosen)-positives),len(stable),stable_sha))
    dataset_rows=tuple(rows)
    result=derive_evaluation_antigravity(dataset_rows,reviewed_commit,authority.canonical_sha256_antigravity,(predecessor.artifact_map_sha256,predecessor.evaluation_sha256,predecessor.decision_sha256))
    _publish_antigravity(dependencies.evaluation_root,result,authority,predecessor); return result


def evaluate_antigravity(prediction_authority: FrozenPredictionAuthorityAntigravity, reviewed_commit: str) -> FrozenEvaluationAntigravity:
    predecessor=verify_exp448_predecessor_antigravity(); evidence=tuple(EvaluationEvidenceRowAntigravity(i,identity,prediction_authority.target_lengths[i],prediction_authority.candidate_indices[i]) for i,identity in enumerate(prediction_authority.dataset_identities))
    if _baseline_authority_sha256_antigravity(predecessor)!=EXP448_BASELINE_AUTHORITY_SHA256_ANTIGRAVITY: raise ValueError("frozen Exp448 baseline authority differs")
    return _evaluate_with_dependencies_antigravity(prediction_authority,reviewed_commit,EvaluationDependenciesAntigravity(validate_frozen_prediction_authority_antigravity,lambda:predecessor,evidence,lambda:open_ordered_label_session_antigravity,OFFICIAL_EVALUATION_ROOT_ANTIGRAVITY))
