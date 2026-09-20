"""Learn feature attribution weights while keeping official representations intact."""
import numpy as np
from scipy.spatial.distance import cdist
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED
from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import perturbations, KINDS
from experiments.exp502_amplitude_envelope_antigravity.amplitude_antigravity import AmplitudeHeads, amplitude_views
from .optimizer_antigravity import fit_tail_weights


def distance_contributions(query, reference, exclude=None):
    q, r = matrix(query), matrix(reference)
    if q.shape[1] != r.shape[1] or len(r) < 2:
        raise ValueError('aligned features and at least two references required')
    d = cdist(q, r)
    if exclude is not None:
        e = np.asarray(exclude)
        if e.shape != (len(q),) or not np.issubdtype(e.dtype, np.integer) or np.any(e < 0) or np.any(e >= len(r)):
            raise ValueError('invalid excluded parent references')
        d[np.arange(len(q)), e] = np.inf
    k = min(3, len(r) - int(exclude is not None))
    neighbors = np.argsort(d, axis=1, kind='stable')[:, :k]
    result = np.zeros_like(q)
    for j in range(k):
        residual = q - r[neighbors[:, j]]
        radius = d[np.arange(len(q)), neighbors[:, j]]
        result += np.divide(residual ** 2, radius[:, None], out=np.zeros_like(residual), where=radius[:, None] > 0)
    result /= k
    if not np.isfinite(result).all(): raise ValueError('nonfinite distance contributions')
    return result


def unseen_perturbations(rows, seed):
    """Evaluation-only strengths and types; never supplied to fit_tail_weights."""
    x = matrix(rows); n, length = x.shape; rng = np.random.default_rng(seed)
    result = {key: x.copy() for key in ('weak_level', 'weak_scale', 'sparse_spike', 'time_warp')}
    width = max(1, length // 10); axis = np.arange(length, dtype=float)
    for i in range(n):
        start = int(rng.integers(0, length - width + 1)); region = slice(start, start + width)
        result['weak_level'][i, region] += 1.25 * rng.choice([-1., 1.])
        result['weak_scale'][i, region] *= rng.choice([.5, 1.75])
        index = rng.choice(length, size=max(1, length // 50), replace=False)
        result['sparse_spike'][i, index] += 4. * rng.choice([-1., 1.], size=len(index))
        if length > 1:
            exponent = rng.choice([.65, 1.5])
            result['time_warp'][i] = np.interp((axis / (length - 1)) ** exponent * (length - 1), axis, x[i])
    for key, value in result.items():
        scale = value.std(axis=1, keepdims=True)
        value = (value - value.mean(axis=1, keepdims=True)) / np.where(scale == 0, 1., scale)
        result[key] = value.astype(np.float32)
        if not np.isfinite(result[key]).all(): raise ValueError('nonfinite held-out probe')
    return result


class TailRankHeads:
    names = ('MiniRocket', 'oas_distance', 'square_joint', 'tail_rank')

    def __init__(self, rows):
        x = matrix(rows)
        if len(x) < 2: raise ValueError('at least two normal TRAIN rows required')
        self.length = x.shape[1]; self.base = AmplitudeHeads(x)
        self.scaler, self.geometry, self.reference = self.base.models['square_joint']
        self.weights = np.ones(self.reference.shape[1]); self.parents = np.array([], dtype=int)
        self.fallback = len(x) < 16; self.scale = 1.
        self.diagnostics = dict(fallback=self.fallback, n_reference=len(x), optimizer_trained=False)
        if self.fallback: return
        self.parents = np.sort(np.random.default_rng(SEED + 508).choice(len(x), min(128, len(x)), replace=False))
        d = cdist(self.reference, self.reference); np.fill_diagonal(d, np.inf)
        self.scale = float(np.median(np.sort(d, axis=1)[:, :min(3, len(x) - 1)].mean(axis=1)) + 1e-9)
        normal = distance_contributions(self.reference[self.parents], self.reference, exclude=self.parents) / self.scale
        synthetic = perturbations(x[self.parents], SEED + 1508)
        anomalies = np.vstack([distance_contributions(self._geometry(synthetic[k]), self.reference, exclude=self.parents) / self.scale for k in KINDS])
        self.weights, audit = fit_tail_weights(normal, anomalies)
        self.diagnostics.update(optimizer_trained=True, n_parents=len(self.parents), scale=self.scale, **audit)

    def _geometry(self, rows):
        x = matrix(rows)
        if x.shape[1] != self.length: raise ValueError('query length differs')
        raw = self.base.encoders[0].transform(x)
        square = self.base.encoders[1].transform(amplitude_views(x)[0])
        return self.geometry.transform(self.scaler.transform(np.column_stack((raw, square))))

    def score(self, rows):
        x = matrix(rows)
        if x.shape[1] != self.length: raise ValueError('query length differs')
        old = self.base.score(x); result = {key: old[key] for key in self.names[:3]}
        if self.fallback: result['tail_rank'] = old['square_joint'].copy()
        else:
            g = distance_contributions(self._geometry(x), self.reference) / self.scale
            result['tail_rank'] = g @ self.weights
        if not all(np.isfinite(value).all() for value in result.values()): raise ValueError('nonfinite score')
        return result
