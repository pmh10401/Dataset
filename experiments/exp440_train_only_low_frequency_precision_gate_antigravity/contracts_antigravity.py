"""Strict immutable evidence contracts for Exp440."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _exact_payload(payload: Mapping[str, object], keys: set[str]) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise ValueError("payload keys are invalid")
    return dict(payload)


@dataclass(frozen=True)
class DepthCalibrationAntigravity:
    depth: int
    oof_scores: tuple[float, ...]
    threshold: float
    delete_one_thresholds: tuple[float, ...]
    robust_scale: float
    stability_ratio: float
    stable: bool

    def __post_init__(self) -> None:
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("depth must be positive")
        oof = tuple(_finite(x, "OOF score", nonnegative=True) for x in self.oof_scores)
        delete = tuple(_finite(x, "delete-one threshold", nonnegative=True) for x in self.delete_one_thresholds)
        threshold = _finite(self.threshold, "threshold", nonnegative=True)
        scale = _finite(self.robust_scale, "robust scale", nonnegative=True)
        ratio = _finite(self.stability_ratio, "stability ratio", nonnegative=True)
        if len(oof) < 4 or len(delete) < 1:
            raise ValueError("calibration evidence is insufficient")
        expected_ratio = max(delete) / max(min(delete), 1e-12)
        if not math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("stability ratio is inconsistent")
        expected_stable = scale > 1e-12 and ratio <= 2.0
        if type(self.stable) is not bool or self.stable != expected_stable:
            raise ValueError("stable flag is inconsistent")
        object.__setattr__(self, "oof_scores", oof)
        object.__setattr__(self, "delete_one_thresholds", delete)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "robust_scale", scale)
        object.__setattr__(self, "stability_ratio", ratio)

    @property
    def threshold_denominator(self) -> float:
        return max(self.threshold, 1e-12)

    def to_payload(self) -> dict[str, object]:
        return {"depth":self.depth,"oof_scores":list(self.oof_scores),"threshold":self.threshold,"delete_one_thresholds":list(self.delete_one_thresholds),"robust_scale":self.robust_scale,"stability_ratio":self.stability_ratio,"stable":self.stable}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DepthCalibrationAntigravity":
        raw=_exact_payload(payload,{"depth","oof_scores","threshold","delete_one_thresholds","robust_scale","stability_ratio","stable"})
        if not isinstance(raw["oof_scores"], list) or not isinstance(raw["delete_one_thresholds"], list): raise ValueError("calibration arrays must be lists")
        return cls(raw["depth"],tuple(raw["oof_scores"]),raw["threshold"],tuple(raw["delete_one_thresholds"]),raw["robust_scale"],raw["stability_ratio"],raw["stable"])


@dataclass(frozen=True)
class GroupCalibrationAntigravity:
    raw_length: int
    preferred_depth: int
    depths: tuple[DepthCalibrationAntigravity, ...]
    eligible: bool

    def __post_init__(self) -> None:
        if type(self.raw_length) is not int or self.raw_length < 1 or type(self.preferred_depth) is not int or self.preferred_depth < 1: raise ValueError("group identity is invalid")
        values=tuple(self.depths); ids=tuple(x.depth for x in values)
        if ids != tuple(sorted(set(ids))): raise ValueError("depths must be sorted and unique")
        expected=sum(x.stable for x in values)>=2
        if type(self.eligible) is not bool or self.eligible != expected: raise ValueError("group eligibility is inconsistent")
        object.__setattr__(self,"depths",values)

    def to_payload(self) -> dict[str,object]: return {"raw_length":self.raw_length,"preferred_depth":self.preferred_depth,"depths":[x.to_payload() for x in self.depths],"eligible":self.eligible}
    @classmethod
    def from_payload(cls,payload:Mapping[str,object])->"GroupCalibrationAntigravity":
        raw=_exact_payload(payload,{"raw_length","preferred_depth","depths","eligible"}); items=raw["depths"]
        if not isinstance(items,list): raise ValueError("depths must be a list")
        return cls(raw["raw_length"],raw["preferred_depth"],tuple(DepthCalibrationAntigravity.from_payload(x) for x in items),raw["eligible"])


@dataclass(frozen=True)
class DepthScoreAntigravity:
    depth:int; score:float; threshold:float; exceeds:bool; excess_ratio:float; empirical_rank:float
    def __post_init__(self)->None:
        if type(self.depth) is not int or self.depth<1: raise ValueError("depth must be positive")
        score=_finite(self.score,"score",nonnegative=True); threshold=_finite(self.threshold,"threshold",nonnegative=True); ratio=_finite(self.excess_ratio,"excess ratio",nonnegative=True); rank=_finite(self.empirical_rank,"empirical rank",nonnegative=True)
        if rank<=0 or rank>1: raise ValueError("empirical rank is invalid")
        if type(self.exceeds) is not bool or self.exceeds != (score>threshold): raise ValueError("exceeds is inconsistent")
        if not math.isclose(ratio,score/max(threshold,1e-12),rel_tol=1e-12,abs_tol=1e-12): raise ValueError("excess ratio is inconsistent")
        object.__setattr__(self,"score",score);object.__setattr__(self,"threshold",threshold);object.__setattr__(self,"excess_ratio",ratio);object.__setattr__(self,"empirical_rank",rank)
    def to_payload(self)->dict[str,object]: return {k:getattr(self,k) for k in ("depth","score","threshold","exceeds","excess_ratio","empirical_rank")}
    @classmethod
    def from_payload(cls,payload:Mapping[str,object])->"DepthScoreAntigravity": return cls(**_exact_payload(payload,{"depth","score","threshold","exceeds","excess_ratio","empirical_rank"}))


@dataclass(frozen=True)
class ProvisionalCandidateAntigravity:
    dataset:str; instance_index:int; raw_length:int; depths:tuple[int,...]; agreement_count:int; second_excess_ratio:float; second_empirical_rank:float
    def __post_init__(self)->None:
        if not isinstance(self.dataset,str) or not self.dataset or type(self.instance_index) is not int or self.instance_index<0 or type(self.raw_length) is not int or self.raw_length<1: raise ValueError("candidate identity is invalid")
        depths=tuple(self.depths)
        if depths!=tuple(sorted(set(depths))) or len(depths)<2 or any(type(x) is not int or x<1 for x in depths): raise ValueError("candidate depths are invalid")
        if type(self.agreement_count) is not int or self.agreement_count!=len(depths): raise ValueError("agreement count is inconsistent")
        object.__setattr__(self,"depths",depths); object.__setattr__(self,"second_excess_ratio",_finite(self.second_excess_ratio,"second excess ratio",nonnegative=True)); rank=_finite(self.second_empirical_rank,"second empirical rank",nonnegative=True)
        if rank<=0 or rank>1: raise ValueError("second empirical rank is invalid")
        object.__setattr__(self,"second_empirical_rank",rank)
    @property
    def coordinate(self)->tuple[str,int]: return (self.dataset,self.instance_index)
    @property
    def ranking_key(self)->tuple[object,...]: return (-self.agreement_count,-self.second_excess_ratio,self.second_empirical_rank,self.dataset,self.instance_index)
    def to_payload(self)->dict[str,object]: return {"dataset":self.dataset,"instance_index":self.instance_index,"raw_length":self.raw_length,"depths":list(self.depths),"agreement_count":self.agreement_count,"second_excess_ratio":self.second_excess_ratio,"second_empirical_rank":self.second_empirical_rank}
    @classmethod
    def from_payload(cls,payload:Mapping[str,object])->"ProvisionalCandidateAntigravity":
        raw=_exact_payload(payload,{"dataset","instance_index","raw_length","depths","agreement_count","second_excess_ratio","second_empirical_rank"}); depths=raw["depths"]
        if not isinstance(depths,list): raise ValueError("depths must be a list")
        raw["depths"]=tuple(depths); return cls(**raw)


@dataclass(frozen=True)
class RetainedCandidateAntigravity(ProvisionalCandidateAntigravity):
    global_rank:int
    def __post_init__(self)->None:
        super().__post_init__()
        if type(self.global_rank) is not int or self.global_rank<1: raise ValueError("global rank must be positive")
    @classmethod
    def from_provisional(cls,item:ProvisionalCandidateAntigravity,rank:int)->"RetainedCandidateAntigravity": return cls(item.dataset,item.instance_index,item.raw_length,item.depths,item.agreement_count,item.second_excess_ratio,item.second_empirical_rank,rank)
    def to_payload(self)->dict[str,object]:
        raw=super().to_payload(); raw["global_rank"]=self.global_rank; return raw


def validate_retained_candidates_antigravity(items:tuple[RetainedCandidateAntigravity,...],hard_coordinates:frozenset[tuple[str,int]],*,budget:int=100)->tuple[RetainedCandidateAntigravity,...]:
    values=tuple(items)
    if budget!=100 or len(values)>budget: raise ValueError("outside-Hard budget exceeded")
    coords=[x.coordinate for x in values]
    if len(coords)!=len(set(coords)) or any(x in hard_coordinates for x in coords): raise ValueError("retained coordinates are invalid")
    if tuple(x.global_rank for x in values)!=tuple(range(1,len(values)+1)): raise ValueError("global ranks are invalid")
    return values


@dataclass(frozen=True)
class Exp440DatasetResultAntigravity:
    dataset:str; status:str; provisional_candidates:tuple[ProvisionalCandidateAntigravity,...]; abstention_indices:tuple[int,...]; excluded_train_rows:int; test_row_count:int
    def __post_init__(self)->None:
        if not self.dataset or self.status not in {"scored","abstained"}: raise ValueError("dataset result identity is invalid")
        if type(self.excluded_train_rows) is not int or self.excluded_train_rows<0 or type(self.test_row_count) is not int or self.test_row_count<0: raise ValueError("dataset counts are invalid")
        values=tuple(self.provisional_candidates); abst=tuple(self.abstention_indices)
        if any(x.dataset!=self.dataset for x in values) or abst!=tuple(sorted(set(abst))): raise ValueError("dataset result evidence is invalid")
        object.__setattr__(self,"provisional_candidates",values);object.__setattr__(self,"abstention_indices",abst)
    def to_payload(self)->dict[str,object]: return {"dataset":self.dataset,"status":self.status,"provisional_candidates":[x.to_payload() for x in self.provisional_candidates],"abstention_indices":list(self.abstention_indices),"excluded_train_rows":self.excluded_train_rows,"test_row_count":self.test_row_count}
    @classmethod
    def from_payload(cls,payload:Mapping[str,object])->"Exp440DatasetResultAntigravity":
        raw=_exact_payload(payload,{"dataset","status","provisional_candidates","abstention_indices","excluded_train_rows","test_row_count"})
        if not isinstance(raw["provisional_candidates"],list) or not isinstance(raw["abstention_indices"],list): raise ValueError("dataset arrays must be lists")
        raw["provisional_candidates"]=tuple(ProvisionalCandidateAntigravity.from_payload(x) for x in raw["provisional_candidates"]);raw["abstention_indices"]=tuple(raw["abstention_indices"]);return cls(**raw)


__all__=["DepthCalibrationAntigravity","GroupCalibrationAntigravity","DepthScoreAntigravity","ProvisionalCandidateAntigravity","RetainedCandidateAntigravity","Exp440DatasetResultAntigravity","validate_retained_candidates_antigravity"]
