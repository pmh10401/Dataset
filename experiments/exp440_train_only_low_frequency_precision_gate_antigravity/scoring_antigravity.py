"""Strict two-depth TEST consensus scoring for Exp440."""
from __future__ import annotations
import numpy as np
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.features_antigravity import fft_split_antigravity
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.scoring_antigravity import score_channel_cpu_antigravity
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import DepthScoreAntigravity,ProvisionalCandidateAntigravity
from .features_antigravity import FittedGroupAntigravity


def score_test_row_antigravity(dataset:str,instance_index:int,values:object,group:FittedGroupAntigravity)->tuple[tuple[DepthScoreAntigravity,...],ProvisionalCandidateAntigravity|None]:
    vector=np.ascontiguousarray(values,dtype=np.float32)
    if not dataset or type(instance_index) is not int or instance_index<0 or vector.ndim!=1 or vector.size!=group.raw_length or not np.isfinite(vector).all():raise ValueError("TEST row is invalid")
    scores=[]
    for model in group.models:
        low=fft_split_antigravity(vector,model.calibration.depth)[0]
        query=np.ascontiguousarray((low.astype(np.float64)-model.center)/model.scale,dtype=np.float32)
        score=score_channel_cpu_antigravity(query,model.references,FROZEN_CONFIG.nearest_k);threshold=model.calibration.threshold;rank=(1+sum(x>=score for x in model.calibration.oof_scores))/(len(model.calibration.oof_scores)+1)
        scores.append(DepthScoreAntigravity(model.calibration.depth,score,threshold,score>threshold,score/max(threshold,1e-12),rank))
    agreeing=tuple(x for x in scores if x.exceeds)
    if len(agreeing)<FROZEN_CONFIG.minimum_consensus_depths:return tuple(scores),None
    ratios=sorted((x.excess_ratio for x in agreeing),reverse=True);ranks=sorted(x.empirical_rank for x in agreeing);depths=tuple(sorted(x.depth for x in agreeing))
    return tuple(scores),ProvisionalCandidateAntigravity(dataset,instance_index,int(vector.size),depths,len(depths),ratios[1],ranks[1],tuple(scores))


__all__=["score_test_row_antigravity"]
