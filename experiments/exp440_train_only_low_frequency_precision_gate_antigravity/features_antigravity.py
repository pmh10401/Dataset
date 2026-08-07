"""Fold-local adjacent-depth low-frequency calibration for Exp440."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable,Iterable
import numpy as np
from numpy.typing import NDArray
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.features_antigravity import fft_split_antigravity,lower_median_depth_antigravity,maximum_depth_antigravity,preferred_depth_antigravity
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.scoring_antigravity import score_channel_cpu_antigravity
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import DepthCalibrationAntigravity,GroupCalibrationAntigravity


@dataclass(frozen=True)
class DepthModelAntigravity:
    calibration:DepthCalibrationAntigravity;references:NDArray[np.float32];center:float;scale:float
    def __post_init__(self)->None:
        matrix=np.ascontiguousarray(self.references,dtype=np.float32).copy()
        if matrix.ndim!=2 or not np.isfinite(matrix).all() or not math.isfinite(self.center) or not math.isfinite(self.scale) or self.scale<=1e-12:raise ValueError("depth model is invalid")
        matrix.flags.writeable=False;object.__setattr__(self,"references",matrix)


@dataclass(frozen=True)
class FittedGroupAntigravity:
    calibration:GroupCalibrationAntigravity;models:tuple[DepthModelAntigravity,...]
    @property
    def depths(self):return self.calibration.depths
    @property
    def eligible(self):return self.calibration.eligible
    @property
    def raw_length(self):return self.calibration.raw_length


def adjacent_depths_antigravity(preferred_depth:int,maximum_depth:int)->tuple[int,...]:
    if type(preferred_depth) is not int or type(maximum_depth) is not int or preferred_depth<1 or maximum_depth<1 or preferred_depth>maximum_depth:raise ValueError("depth boundary is invalid")
    result=tuple(x for x in (preferred_depth-1,preferred_depth,preferred_depth+1) if 1<=x<=maximum_depth)
    return result if len(result)>=2 else ()


def _low_matrix_antigravity(matrix:NDArray[np.float32],depth:int)->NDArray[np.float32]:
    return np.ascontiguousarray(np.stack([fft_split_antigravity(row,depth)[0] for row in matrix]),dtype=np.float32)


def _fit_scalar_antigravity(matrix:NDArray[np.float32])->tuple[NDArray[np.float32],float,float]:
    values=matrix.astype(np.float64);center=float(np.median(values));scale=float(1.4826*np.median(np.abs(values-center)))
    if not math.isfinite(center) or not math.isfinite(scale) or scale<=1e-12:raise ValueError("zero robust scale")
    result=np.ascontiguousarray((values-center)/scale,dtype=np.float32);return result,center,scale


def fit_group_calibration_antigravity(matrix:object,*,distance:Callable[[object,object,int],float]=score_channel_cpu_antigravity,preferred_depths:Iterable[int]|None=None)->FittedGroupAntigravity:
    train=np.ascontiguousarray(matrix,dtype=np.float32)
    if train.ndim!=2 or train.shape[0]<4 or not np.isfinite(train).all():raise ValueError("finite TRAIN group with at least four rows is required")
    maximum=maximum_depth_antigravity(int(train.shape[1]));prefs=tuple(preferred_depths) if preferred_depths is not None else tuple(preferred_depth_antigravity(row) for row in train)
    if len(prefs)!=train.shape[0]:raise ValueError("preferred depth count differs")
    preferred=lower_median_depth_antigravity(prefs);available=adjacent_depths_antigravity(preferred,maximum)
    calibrations=[];models=[]
    for depth in available:
        low=_low_matrix_antigravity(train,depth);oof=[]
        for held in range(train.shape[0]):
            refs=np.delete(low,held,axis=0)
            standardized,center,scale=_fit_scalar_antigravity(refs)
            query=np.ascontiguousarray((low[held].astype(np.float64)-center)/scale,dtype=np.float32)
            oof.append(float(distance(query,standardized,FROZEN_CONFIG.nearest_k)))
        threshold=float(np.quantile(np.asarray(oof,dtype=np.float64),FROZEN_CONFIG.quantile,method="higher"))
        deletes=tuple(float(np.quantile(np.delete(np.asarray(oof,dtype=np.float64),i),FROZEN_CONFIG.quantile,method="higher")) for i in range(len(oof)))
        ratio=max(deletes)/max(min(deletes),1e-12)
        try:standardized,center,scale=_fit_scalar_antigravity(low)
        except ValueError:scale=0.0;center=0.0;standardized=None
        stable=bool(scale>1e-12 and math.isfinite(ratio) and ratio<=FROZEN_CONFIG.stability_ratio_max)
        calibration=DepthCalibrationAntigravity(depth,tuple(oof),threshold,deletes,scale,ratio,stable);calibrations.append(calibration)
        if stable and standardized is not None:models.append(DepthModelAntigravity(calibration,standardized,center,scale))
    group=GroupCalibrationAntigravity(int(train.shape[1]),preferred,tuple(calibrations),sum(x.stable for x in calibrations)>=2)
    return FittedGroupAntigravity(group,tuple(models))


__all__=["DepthModelAntigravity","FittedGroupAntigravity","adjacent_depths_antigravity","fit_group_calibration_antigravity"]
