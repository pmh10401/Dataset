"""Exact-length, missing-safe, label-free input adapter for Exp440."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.contracts_antigravity import ExcludedTrainRowAudit,PreparedTrainGroup,RawValueSeries
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.inputs_antigravity import Exp439RawDataset,Exp439InputSession,open_exp439_input_session
from .config_antigravity import FROZEN_CONFIG


@dataclass(frozen=True)
class TrainPreparationAntigravity:
    dataset:str;groups:tuple[PreparedTrainGroup,...];exclusions:tuple[ExcludedTrainRowAudit,...];insufficient_lengths:tuple[int,...]
    def __post_init__(self)->None:
        if not self.dataset:raise ValueError("TRAIN dataset is invalid")
        if tuple(g.raw_length for g in self.groups)!=tuple(sorted(g.raw_length for g in self.groups)):raise ValueError("TRAIN groups are unordered")


def partition_train_rows_antigravity(train:Exp439RawDataset)->TrainPreparationAntigravity:
    if type(train) is not Exp439RawDataset or train.split!="TRAIN":raise ValueError("TRAIN dataset is required")
    finite:dict[int,list[RawValueSeries]]=defaultdict(list);excluded=[];lengths=set()
    for row in train.series:
        lengths.add(row.raw_length);positions=tuple(int(x) for x in np.flatnonzero(~np.isfinite(row.values)))
        if positions:excluded.append(ExcludedTrainRowAudit(row.dataset,row.instance_index,row.raw_length,positions,row.values_blob_sha256))
        else:finite[row.raw_length].append(row)
    groups=[];insufficient=[]
    for length in sorted(lengths):
        rows=tuple(finite.get(length,()))
        if len(rows)<FROZEN_CONFIG.minimum_train_rows or len(rows)>FROZEN_CONFIG.maximum_train_rows:insufficient.append(length)
        else:groups.append(PreparedTrainGroup(train.dataset,length,rows))
    return TrainPreparationAntigravity(train.dataset,tuple(groups),tuple(sorted(excluded,key=lambda x:x.instance_index)),tuple(insufficient))


def test_row_abstention_reason_antigravity(row:RawValueSeries)->str|None:
    if type(row) is not RawValueSeries or row.split!="TEST":raise ValueError("TEST row is required")
    return "nonfinite_test" if not np.isfinite(row.values).all() else None


__all__=["Exp439RawDataset","Exp439InputSession","TrainPreparationAntigravity","open_exp439_input_session","partition_train_rows_antigravity","test_row_abstention_reason_antigravity"]
