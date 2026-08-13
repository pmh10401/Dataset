"""Single-session Exp451 TEST evaluation and immutable local publication."""
from __future__ import annotations

import dataclasses
import ctypes
import errno
import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.exp449_residual_fn_method_qualification_antigravity.bootstrap_antigravity import paired_delta_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity

from .artifacts_antigravity import canonical_json_bytes_antigravity, write_create_only_antigravity
from .contracts_antigravity import FrozenPredictionAuthorityAntigravity
from .predecessor_antigravity import Exp448PredecessorAuthorityAntigravity, verify_exp448_predecessor_antigravity
from .prediction_antigravity import validate_frozen_prediction_authority_antigravity
from .sources_antigravity import LabelSnapshotAntigravity, open_ordered_label_session_antigravity

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
class DatasetEvaluationAntigravity:
    sequence: int; identity: tuple[int, str, str]; target_length: int
    baseline: ConfusionAntigravity; candidate: ConfusionAntigravity
    new_detections: int; new_false_alarms: int; detection_losses: int


@dataclass(frozen=True, slots=True)
class FrozenEvaluationAntigravity:
    identity: str; status: str; reviewed_commit: str; prediction_authority_sha256: str
    predecessor_hashes: tuple[str, str, str]; replaced_dataset_count: int; reused_dataset_count: int
    global_dataset_count: int; label_session_count: int; label_query_count: int; label_row_count: int
    bootstrap_dataset_pairs: int; target_order: tuple[tuple[int, str, str], ...]
    baseline_global: ConfusionAntigravity; candidate_global: ConfusionAntigravity
    baseline_f1: float; candidate_f1: float; target_baseline_fn: int; target_candidate_fn: int
    target_fn_delta: int; transition_totals: tuple[int, int, int]; transition_dataset_counts: tuple[int, int, int]
    family_fn_deltas: tuple[tuple[str, int], ...]; gates: tuple[tuple[str, bool], ...]; qualified: bool
    bootstrap: dict[str, object]; baseline_matrix: tuple[tuple[int, int, int, int], ...]
    candidate_matrix: tuple[tuple[int, int, int, int], ...]; dataset_rows: tuple[DatasetEvaluationAntigravity, ...]


@dataclass(frozen=True, slots=True)
class EvaluationDependenciesAntigravity:
    prediction_validator: Callable[[FrozenPredictionAuthorityAntigravity], FrozenPredictionAuthorityAntigravity]
    predecessor_loader: Callable[[], Exp448PredecessorAuthorityAntigravity]
    evidence_rows: tuple[EvaluationEvidenceRowAntigravity, ...]
    label_opener_factory: Callable[[], Callable[[FrozenPredictionAuthorityAntigravity], LabelSnapshotAntigravity]]
    evaluation_root: Path


def _f1_antigravity(value: ConfusionAntigravity) -> float:
    denominator = 2 * value.tp + value.fp + value.fn
    return 0.0 if denominator == 0 else 2.0 * value.tp / denominator


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
    return {"sequence":row.sequence,"identity":list(row.identity),"target_length":row.target_length,"baseline":_conf_payload_antigravity(row.baseline),"candidate":_conf_payload_antigravity(row.candidate),"new_detections":row.new_detections,"new_false_alarms":row.new_false_alarms,"detection_losses":row.detection_losses}


def _result_payload_antigravity(result: FrozenEvaluationAntigravity) -> dict[str,object]:
    return {"schema":"exp451.evaluation.result.v1","identity":result.identity,"status":result.status,"reviewed_commit":result.reviewed_commit,"prediction_authority_sha256":result.prediction_authority_sha256,"predecessor_hashes":list(result.predecessor_hashes),"replaced_dataset_count":48,"reused_dataset_count":1069,"global_dataset_count":1117,"label_session_count":1,"label_query_count":48,"label_row_count":72300,"bootstrap_dataset_pairs":48,"target_order":[list(x) for x in result.target_order],"baseline_global":_conf_payload_antigravity(result.baseline_global),"candidate_global":_conf_payload_antigravity(result.candidate_global),"baseline_f1":result.baseline_f1,"candidate_f1":result.candidate_f1,"target_baseline_fn":result.target_baseline_fn,"target_candidate_fn":result.target_candidate_fn,"target_fn_delta":result.target_fn_delta,"transition_totals":list(result.transition_totals),"transition_dataset_counts":list(result.transition_dataset_counts),"family_fn_deltas":[list(x) for x in result.family_fn_deltas],"gates":[list(x) for x in result.gates],"qualified":result.qualified,"bootstrap":result.bootstrap,"baseline_matrix":[list(x) for x in result.baseline_matrix],"candidate_matrix":[list(x) for x in result.candidate_matrix]}


def _rename_no_replace_antigravity(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink(): raise FileExistsError(destination)
    if __import__("sys").platform == "darwin":
        libc=ctypes.CDLL(None,use_errno=True); function=getattr(libc,"renamex_np",None)
        if function is None: raise OSError(errno.ENOTSUP,"renamex_np unavailable")
        function.argtypes=(ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint); function.restype=ctypes.c_int
        if function(os.fsencode(source),os.fsencode(destination),0x4):
            code=ctypes.get_errno(); raise OSError(code,os.strerror(code),destination)
    else: os.rename(source,destination)


def _remove_private_staging_antigravity(staging: Path) -> None:
    if not staging.exists() or staging.is_symlink(): return
    for name in tuple(os.listdir(staging)):
        path=staging/name
        if path.is_file() and not path.is_symlink(): path.unlink()
    staging.rmdir()

def _write_leaf_at_antigravity(directory_fd: int, name: str, payload: bytes) -> None:
    fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory_fd)
    try:
        offset=0
        while offset<len(payload):
            written=os.write(fd,payload[offset:])
            if written<=0: raise OSError(errno.EIO,"leaf write made no progress")
            offset+=written
        os.fsync(fd)
    finally: os.close(fd)


def _publish_antigravity(root: Path, result: FrozenEvaluationAntigravity, authority: FrozenPredictionAuthorityAntigravity) -> None:
    if root.exists() or root.is_symlink(): raise ValueError("evaluation root already exists")
    staging=root.parent/f".{root.name}.{os.getpid()}.staging_antigravity"
    if staging.exists() or staging.is_symlink(): raise ValueError("private staging collision")
    parent_fd=os.open(root.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    os.mkdir(staging.name,mode=0o700,dir_fd=parent_fd)
    staging_fd=os.open(staging.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd)
    from .dashboard_export_antigravity import public_result_antigravity
    leaves = {"context_antigravity.json":{"schema":"exp451.evaluation.context.v1","reviewed_commit":result.reviewed_commit,"prediction_authority":authority.to_payload_antigravity(),"predecessor_hashes":list(result.predecessor_hashes),"exp448_baseline_authority_sha256":EXP448_BASELINE_AUTHORITY_SHA256_ANTIGRAVITY},"datasets_antigravity.json":{"schema":"exp451.evaluation.datasets.v1","rows":[_dataset_payload_antigravity(x) for x in result.dataset_rows]},"result_antigravity.json":_result_payload_antigravity(result),"dashboard_antigravity.json":public_result_antigravity(result)}
    promoted=False
    try:
        hashes={}
        for name,value in leaves.items():
            payload=canonical_json_bytes_antigravity(value); _write_leaf_at_antigravity(staging_fd,name,payload); hashes[name]=hashlib.sha256(payload).hexdigest()
        artifact_map={"schema":"exp451.evaluation.artifact_map.v1","leaf_sha256":hashes}
        map_bytes=canonical_json_bytes_antigravity(artifact_map); _write_leaf_at_antigravity(staging_fd,"artifact_map_antigravity.json",map_bytes)
        inventory=[*leaves.keys(),"artifact_map_antigravity.json","manifest_antigravity.json"]
        manifest={"schema":"exp451.evaluation.manifest.v1","inventory":inventory,"artifact_map_sha256":hashlib.sha256(map_bytes).hexdigest(),"result_sha256":hashes["result_antigravity.json"],"dashboard_sha256":hashes["dashboard_antigravity.json"]}
        _write_leaf_at_antigravity(staging_fd,"manifest_antigravity.json",canonical_json_bytes_antigravity(manifest))
        os.fsync(staging_fd)
        from .replay_antigravity import inspect_replay_antigravity
        if inspect_replay_antigravity(staging,expected_prediction_authority=authority)!=result: raise ValueError("private replay differs")
        staging_identity=os.stat(staging,follow_symlinks=False)
        _rename_no_replace_antigravity(staging,root); promoted=True
        try:
            flush_parent_fd=os.open(root.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try: os.fsync(flush_parent_fd)
            finally: os.close(flush_parent_fd)
            if inspect_replay_antigravity(root,expected_prediction_authority=authority)!=result: raise ValueError("public replay differs")
        except BaseException as primary:
            try:
                current=os.stat(root,follow_symlinks=False)
                if (current.st_dev,current.st_ino)!=(staging_identity.st_dev,staging_identity.st_ino): raise ValueError("promoted namespace no longer names retained staging inode")
                forensic=None
                for _ in range(32):
                    candidate=root.parent/f".{root.name}.forensic.{secrets.token_hex(16)}_antigravity"
                    if not candidate.exists() and not candidate.is_symlink(): forensic=candidate; break
                if forensic is None: raise FileExistsError("forensic namespace allocation exhausted")
                _rename_no_replace_antigravity(root,forensic)
                moved=os.stat(forensic,follow_symlinks=False)
                if (moved.st_dev,moved.st_ino)!=(staging_identity.st_dev,staging_identity.st_ino) or root.exists() or root.is_symlink(): raise ValueError("forensic move identity differs")
                recovery_parent_fd=os.open(root.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                try: os.fsync(recovery_parent_fd)
                finally: os.close(recovery_parent_fd)
            except BaseException as recovery:
                primary.add_note(f"forensic_recovery_error={type(recovery).__name__}: {recovery}")
            raise primary
    except BaseException:
        if not promoted:
            held=os.fstat(staging_fd); named=os.stat(staging.name,dir_fd=parent_fd,follow_symlinks=False)
            if (held.st_dev,held.st_ino)!=(named.st_dev,named.st_ino): raise ValueError("private staging namespace changed")
            _remove_private_staging_antigravity(staging)
        raise
    finally:
        os.close(staging_fd); os.close(parent_fd)


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
        rows.append(DatasetEvaluationAntigravity(i,identity,length,baseline.confusion,candidate,len((chosen-stable)&positives),len((chosen-stable)-positives),len((stable-chosen)&positives)))
    dataset_rows=tuple(rows); old=_add_antigravity(tuple(x.baseline for x in dataset_rows)); new=_add_antigravity(tuple(x.candidate for x in dataset_rows)); global_candidate=_sub_add_antigravity(BASELINE_GLOBAL_ANTIGRAVITY,old,new)
    base_matrix=tuple(dataclasses.astuple(x.baseline) for x in dataset_rows); candidate_matrix=tuple(dataclasses.astuple(x.candidate) for x in dataset_rows); bootstrap=paired_delta_antigravity(base_matrix,candidate_matrix)
    delta=new.fn-old.fn; gates=(("global_fn",global_candidate.fn<1988),("global_fp",global_candidate.fp<=314),("global_f1",_f1_antigravity(global_candidate)>=0.6264199935),("target_fn_delta",delta<0))
    families=[]
    for family in dict.fromkeys(x[1] for x in identities): families.append((family,sum(x.candidate.fn-x.baseline.fn for x in dataset_rows if x.identity[1]==family)))
    totals=tuple(sum(getattr(x,f) for x in dataset_rows) for f in ("new_detections","new_false_alarms","detection_losses")); counts=tuple(sum(getattr(x,f)>0 for x in dataset_rows) for f in ("new_detections","new_false_alarms","detection_losses"))
    result=FrozenEvaluationAntigravity("Exp451","qualified" if all(x[1] for x in gates) else "not_qualified",reviewed_commit,authority.canonical_sha256_antigravity,(predecessor.artifact_map_sha256,predecessor.evaluation_sha256,predecessor.decision_sha256),48,1069,1117,1,48,72300,48,identities,BASELINE_GLOBAL_ANTIGRAVITY,global_candidate,_f1_antigravity(BASELINE_GLOBAL_ANTIGRAVITY),_f1_antigravity(global_candidate),old.fn,new.fn,delta,totals,counts,tuple(families),gates,qualifies_antigravity(global_candidate,delta),bootstrap,base_matrix,candidate_matrix,dataset_rows)
    _publish_antigravity(dependencies.evaluation_root,result,authority); return result


def evaluate_antigravity(prediction_authority: FrozenPredictionAuthorityAntigravity, reviewed_commit: str) -> FrozenEvaluationAntigravity:
    predecessor=verify_exp448_predecessor_antigravity(); evidence=tuple(EvaluationEvidenceRowAntigravity(i,identity,prediction_authority.target_lengths[i],prediction_authority.candidate_indices[i]) for i,identity in enumerate(prediction_authority.dataset_identities))
    return _evaluate_with_dependencies_antigravity(prediction_authority,reviewed_commit,EvaluationDependenciesAntigravity(validate_frozen_prediction_authority_antigravity,lambda:predecessor,evidence,lambda:open_ordered_label_session_antigravity,OFFICIAL_EVALUATION_ROOT_ANTIGRAVITY))
