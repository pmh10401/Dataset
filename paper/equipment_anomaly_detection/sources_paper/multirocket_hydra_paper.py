"""Official frozen Exp84 aeon MultiRocket/Hydra source closure.

Provenance: run_model_hard_research_experiments.py SHA256
f9f06973009259590b4afd165dc275cff9b834a628ccb17e248d015d55680194 (aeon_collection,
aeon_fit_collection, aeon_multirocket_features, aeon_hydra_features,
numpy_feature_matrix, stable_tail_feature_prune, maybe_prune_features,
aeon_feature_pair), SHA256 bound in lineage_lock_paper.json.
"""
from __future__ import annotations

import numpy as np

from .knn_paper import density_knn_score_pair_paper, scale_feature_pair_paper
from .preprocessing_paper import DependencyUnavailablePaper


EXP84_CONFIG_PAPER = {"name": "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3", "kind": "aeon_multirocket_hydra", "num_kernels": 1024, "hydra_kernels": 4, "hydra_groups": 32, "neighbors": 3, "score_mode": "local_gap", "feature_prune": "stable_tail", "feature_keep": 1024, "random_state": 20260717, "normalise": False, "output_type": "tensor", "n_jobs": 1}
PROVENANCE_PAPER = {"run_model_hard_research_experiments.py": ("aeon_collection", "aeon_fit_collection", "aeon_multirocket_features", "aeon_hydra_features", "numpy_feature_matrix", "stable_tail_feature_prune", "maybe_prune_features", "aeon_feature_pair")}


def _collection_paper(matrix: np.ndarray) -> np.ndarray: return np.asarray(matrix, dtype=np.float32)[:, None, :]
def _fit_collection_paper(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    return _collection_paper(np.repeat(values, 2, axis=0) if len(values) == 1 else values)
def _matrix_paper(value) -> np.ndarray:
    if hasattr(value, "detach"): value = value.detach().cpu().numpy()
    elif hasattr(value, "to_numpy"): value = value.to_numpy()
    elif hasattr(value, "toarray"): value = value.toarray()
    values = np.asarray(value, dtype=np.float32)
    if values.ndim != 2: values = values.reshape(values.shape[0], -1)
    return np.nan_to_num(values, nan=0., posinf=1e6, neginf=-1e6).astype(np.float32)

def _stable_tail_paper(train: np.ndarray, test: np.ndarray, keep: int) -> tuple[np.ndarray, np.ndarray]:
    if train.shape[1] <= keep: return train, test
    q25, q50, q75 = np.percentile(train, 25, axis=0), np.percentile(train, 50, axis=0), np.percentile(train, 75, axis=0); usable = (q75 - q25) > 1e-8
    if not np.any(usable): return train[:, :keep], test[:, :keep]
    robust = np.abs((train[:, usable] - q50[usable]) / (q75[usable] - q25[usable] + 1e-6)); tail = np.percentile(robust, 95, axis=0) / (np.percentile(robust, 75, axis=0) + 1e-6); nonzero = np.mean(np.abs(train[:, usable]) > 1e-8, axis=0); selected = np.flatnonzero(usable)[np.argsort(tail + .25 * np.abs(nonzero - .5))[:min(keep, usable.sum())]]
    if len(selected) < keep: selected = np.concatenate((selected, np.setdiff1d(np.arange(train.shape[1]), selected)[:keep-len(selected)]))
    selected = np.sort(selected[:keep]); return train[:, selected], test[:, selected]

def exp84_score_pair_paper(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from aeon.transformations.collection.convolution_based import HydraTransformer, MultiRocket
    except ImportError as exc: raise DependencyUnavailablePaper("dependency_unavailable:aeon") from exc
    config = EXP84_CONFIG_PAPER; fit = _collection_paper(train); query = _collection_paper(test)
    mr = MultiRocket(n_kernels=config["num_kernels"], n_jobs=config["n_jobs"], random_state=config["random_state"], normalise=config["normalise"]); hydra = HydraTransformer(n_kernels=config["hydra_kernels"], n_groups=config["hydra_groups"], n_jobs=config["n_jobs"], random_state=config["random_state"], output_type=config["output_type"])
    mr.fit(_fit_collection_paper(train)); hydra.fit(_fit_collection_paper(train)); train_features = np.concatenate((_matrix_paper(mr.transform(fit)), _matrix_paper(hydra.transform(fit))), axis=1); test_features = np.concatenate((_matrix_paper(mr.transform(query)), _matrix_paper(hydra.transform(query))), axis=1)
    train_features, test_features = _stable_tail_paper(train_features, test_features, config["feature_keep"]); train_features, test_features = scale_feature_pair_paper(train_features, test_features)
    return density_knn_score_pair_paper(train_features, test_features, config["neighbors"], "local_gap")
