"""Label-free Exp439 publication and frozen Hard source firewall for Exp440."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .config_antigravity import FROZEN_CONFIG


EXP439_PUBLICATION_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp439_scale_aware_reconstruction_repair_scheduler_antigravity_20260807/publication_antigravity")
EXP404_HARD_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl")


def _reject_forbidden_fields_antigravity(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized=str(key).lower()
            if ("label" in normalized or "groundtruth" in normalized or "anomaly_location" in normalized) and child not in (False,None,0):
                raise ValueError("prediction source contains a forbidden label field")
            _reject_forbidden_fields_antigravity(child)
    elif isinstance(value,(list,tuple)):
        for child in value:_reject_forbidden_fields_antigravity(child)


def validate_prediction_source_antigravity(path:Path,payload:Mapping[str,object],expected_sha256:str,observed_sha256:str,*,expected_population:tuple[str,...]|None=None)->tuple[str,...]:
    source=Path(path)
    if "evaluation_antigravity" in source.parts: raise ValueError("evaluation source is forbidden during prediction")
    if not isinstance(payload,Mapping): raise ValueError("prediction source payload is invalid")
    _reject_forbidden_fields_antigravity(payload)
    raw=source.read_bytes()
    actual=sha256(raw).hexdigest()
    if expected_sha256!=observed_sha256 or actual!=expected_sha256: raise ValueError("prediction source SHA256 mismatch")
    population=payload.get("datasets")
    if population is None: population=list(expected_population or ())
    if not isinstance(population,list) or any(not isinstance(x,str) or not x for x in population): raise ValueError("prediction population is invalid")
    result=tuple(population)
    if result!=tuple(sorted(set(result))): raise ValueError("prediction population must be sorted and unique")
    if expected_population is not None and result!=expected_population: raise ValueError("prediction population differs")
    return result


def parse_hard_coordinates_antigravity(raw:bytes,expected_sha256:str,*,expected_count:int|None=None)->frozenset[tuple[str,int]]:
    if sha256(raw).hexdigest()!=expected_sha256: raise ValueError("Hard SHA256 mismatch")
    coordinates:set[tuple[str,int]]=set()
    for line in raw.splitlines():
        if not line:continue
        try:value=json.loads(line)
        except (UnicodeDecodeError,json.JSONDecodeError) as error:raise ValueError("Hard source is invalid JSON") from error
        if not isinstance(value,Mapping) or value.get("contains_test_labels") is not False or value.get("contains_evaluation_metrics") is not False: raise ValueError("Hard source is not label-free")
        dataset=value.get("dataset");index=value.get("candidate_index")
        if not isinstance(dataset,str) or not dataset or type(index) is not int or index<0:raise ValueError("Hard coordinate is invalid")
        coordinate=(dataset,index)
        if coordinate in coordinates:raise ValueError("Hard coordinate is duplicated")
        coordinates.add(coordinate)
    if expected_count is not None and len(coordinates)!=expected_count:raise ValueError("Hard population is invalid")
    return frozenset(coordinates)


def load_frozen_hard_coordinates_antigravity(path:Path=EXP404_HARD_PATH_ANTIGRAVITY)->frozenset[tuple[str,int]]:
    source=Path(path)
    if source.resolve()!=EXP404_HARD_PATH_ANTIGRAVITY.resolve():raise ValueError("Hard source path is invalid")
    return parse_hard_coordinates_antigravity(source.read_bytes(),FROZEN_CONFIG.hard_prediction_sha256,expected_count=2218)


__all__=["EXP439_PUBLICATION_ANTIGRAVITY","EXP404_HARD_PATH_ANTIGRAVITY","validate_prediction_source_antigravity","parse_hard_coordinates_antigravity","load_frozen_hard_coordinates_antigravity"]
