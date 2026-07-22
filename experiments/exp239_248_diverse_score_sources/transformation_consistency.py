from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.exp239_248_diverse_score_sources.config import SECOND_SEED
from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp239_248_diverse_score_sources.neural_common import CompactEncoder, prepare_rows, set_deterministic, tensor_rows
from experiments.exp239_248_diverse_score_sources.source_base import abstained_result


def _transforms(rows: np.ndarray, seed: int) -> list[np.ndarray]:
    values = np.asarray(rows, dtype=np.float32)
    length = values.shape[1]
    crop_width = max(8, int(length * 0.85))
    crop = values[:, :crop_width]
    old = np.linspace(0.0, 1.0, crop_width); new = np.linspace(0.0, 1.0, length)
    crop = np.vstack([np.interp(new, old, row) for row in crop]).astype(np.float32)
    masked = values.copy(); start = length // 3; masked[:, start : start + max(1, length // 4)] = 0.0
    jitter = values + np.random.default_rng(seed).normal(0.0, 0.03, values.shape).astype(np.float32)
    amplitude = values * np.float32(1.1)
    return [crop, masked, jitter, amplitude]


class _TransformModel(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.encoder = CompactEncoder(); self.classifier = nn.Linear(32, 4)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(values); return self.classifier(embedding), embedding


def _fit_model(rows: np.ndarray, seed: int, epochs: int, device: str) -> _TransformModel:
    set_deterministic(seed); model = _TransformModel().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    transformed = _transforms(rows, seed)
    x = np.concatenate(transformed); y = np.concatenate([np.full(len(rows), index, dtype=np.int64) for index in range(4)])
    inputs = tensor_rows(x, device); labels = torch.as_tensor(y, dtype=torch.long, device=device)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(); logits, _ = model(inputs); loss = nn.functional.cross_entropy(logits, labels); loss.backward(); optimizer.step()
    return model.eval()


def _score_model(model: _TransformModel, rows: np.ndarray, seed: int, device: str) -> np.ndarray:
    variants = [rows] + _transforms(rows, seed)
    with torch.no_grad():
        embeddings = [model.encoder(tensor_rows(value, device)).cpu().numpy() for value in variants]
    stack = np.stack(embeddings, axis=1); center = stack.mean(axis=1, keepdims=True)
    return np.mean(np.linalg.norm(stack - center, axis=2), axis=1)


class TransformationConsistency:
    name = "transformation_consistency"
    minimum_train = 20

    def __init__(self, epochs: int = 8, seeds: tuple[int, ...] | None = None, device: str = "cpu") -> None:
        self.epochs = int(epochs); self.seeds = seeds; self.device = device

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> ScoreResult:
        if len(bundle.train_normal) < self.minimum_train:
            return abstained_result(self.name, bundle, self.minimum_train)
        train = prepare_rows(bundle.train_normal); test = prepare_rows(bundle.test_values)
        seeds = self.seeds or (int(seed), SECOND_SEED); train_by_seed=[]; test_by_seed=[]
        for current_seed in seeds:
            oof=np.empty(len(train),dtype=float)
            for fold,(fit,query) in enumerate(crossfit_plan(len(train),seed=current_seed)):
                model=_fit_model(train[fit],current_seed+fold,self.epochs,self.device); oof[query]=_score_model(model,train[query],current_seed+fold,self.device)
            model=_fit_model(train,current_seed,self.epochs,self.device); train_by_seed.append(oof); test_by_seed.append(_score_model(model,test,current_seed,self.device))
        return ScoreResult(self.name,np.median(train_by_seed,axis=0),np.median(test_by_seed,axis=0),False,None,{"seeds":list(seeds),"epochs":self.epochs,"device":self.device,"cross_fitted":True,"self_reference_violations":0,"contains_test_labels":False},artifacts={"train_scores_by_seed":np.asarray(train_by_seed),"test_scores_by_seed":np.asarray(test_by_seed)})

