"""Assess exact full-head acceleration without changing the official CPU transformer."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import platform
import numpy as np
import torch
from numba import njit
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity, cpu_worker_policy_antigravity
from . import optimizer_antigravity as optimizer
from .ranking_antigravity import TailRankHeads

torch.set_num_threads(1)
P = Path(__file__).resolve().parent


@njit
def objective_numba(w, normal, anomalous):
    z = (normal @ w) / .1
    maximum = np.max(z); ex = np.exp(z - maximum); total = np.sum(ex)
    tail = .1 * (maximum + np.log(total))
    delta = .25 + tail - anomalous @ w
    ep = np.exp(-np.abs(delta))
    probability = np.where(delta >= 0, 1. / (1. + ep), ep / (1. + ep))
    value = np.mean(np.maximum(delta, 0.) + np.log1p(ep)) + .1 * np.mean((w - 1.) ** 2)
    gradient = np.mean(probability) * ((ex / total) @ normal) - probability @ anomalous / len(anomalous)
    gradient += .2 * (w - 1.) / len(w)
    if not np.isfinite(value) or not np.isfinite(gradient).all(): raise ValueError('nonfinite accelerated objective')
    return value, gradient


def objective_torch(w, normal, anomalous):
    weights, n, a = [torch.as_tensor(x, dtype=torch.float64, device='cpu') for x in (w, normal, anomalous)]
    z = (n @ weights) / .1
    delta = .25 + .1 * torch.logsumexp(z, dim=0) - a @ weights
    probability = torch.sigmoid(delta)
    loss = torch.logaddexp(torch.zeros_like(delta), delta).mean() + .1 * ((weights - 1.) ** 2).mean()
    gradient = probability.mean() * (torch.softmax(z, dim=0) @ n) - probability @ a / len(a) + .2 * (weights - 1.) / len(w)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient).all()): raise ValueError('nonfinite accelerated objective')
    return float(loss), gradient.numpy()


def _stage(fit, query, objective):
    original = optimizer._objective
    optimizer._objective = objective
    try:
        start = perf_counter(); first_head = TailRankHeads(fit); first = first_head.score(query); cold = perf_counter() - start
        start = perf_counter(); warm_head = TailRankHeads(fit); warm = warm_head.score(query); seconds = perf_counter() - start
        return first_head, first, warm, cold, seconds
    finally:
        optimizer._objective = original


def _candidates(scores):
    return {name: {str(b): np.flatnonzero(v > np.sort(v[:1000])[::-1][b]).tolist() for b in (0, 1, 5, 10)} for name, v in scores.items()}


def _parity(actual, expected):
    finite = all(np.isfinite(v).all() for v in actual.values())
    close = all(np.allclose(actual[k], expected[k], rtol=1e-10, atol=1e-8) for k in expected)
    ranks = all(np.array_equal(np.argsort(actual[k], kind='stable'), np.argsort(expected[k], kind='stable')) for k in expected)
    candidates = _candidates(actual) == _candidates(expected)
    return dict(finite=bool(finite), allclose=bool(close), ranks_equal=bool(ranks), candidates_equal=bool(candidates),
                parity=bool(finite and close and ranks and candidates),
                max_abs_error=max(float(np.max(abs(actual[k] - expected[k]))) for k in expected))


def _gpu_probes():
    probes = {}
    try:
        import mlx.core as mx
        probes['mlx'] = dict(available=True, metal_available=mx.metal.is_available(), device=str(mx.default_device()), operation='float64 GPU matrix product')
        try:
            value = mx.array([[1., 2.], [3., 4.]], dtype=mx.float64)
            product = mx.matmul(value, value, stream=mx.gpu); mx.eval(product)
            if not np.isfinite(np.asarray(product)).all(): raise ValueError('nonfinite float64 GPU product')
            probes['mlx'].update(float64_supported=True)
        except Exception as error: probes['mlx'].update(float64_supported=False, fallback_reason=str(error))
    except ImportError as error: probes['mlx'] = dict(available=False, float64_supported=False, fallback_reason=str(error))
    probes['mps'] = dict(available=torch.backends.mps.is_available())
    try:
        torch.tensor([1., 2.], dtype=torch.float64, device='mps'); torch.mps.synchronize()
        probes['mps'].update(float64_supported=True)
    except Exception as error: probes['mps'].update(float64_supported=False, fallback_reason=str(error))
    return probes


def main():
    target = P / 'acceleration_results_antigravity.json'
    if target.exists(): raise RuntimeError('completed acceleration assessment must not be overwritten')
    rng = np.random.default_rng(508)
    fit = rng.normal(size=(64, 128)); fit[0] = 0.
    query = rng.normal(size=(1024, 128)); query[0] = 0.
    original = optimizer._objective; backends = []
    with threadpool_limits(limits=1):
        head, reference, warm, cold, warm_seconds = _stage(fit, query, original)
        repeated = _parity(warm, reference)
        parts = [head.score(query[:512]), head.score(query[512:])]
        batched = {name: np.concatenate([part[name] for part in parts]) for name in reference}
        batch = _parity(batched, reference)
        backends.append(dict(name='numpy_float64', first_seconds=cold, warm_seconds=warm_seconds, speedup=1., **_parity(reference, reference)))
        for name, objective in [('numba_float64', objective_numba), ('torch_cpu_float64', objective_torch)]:
            _, first, warm, first_s, warm_s = _stage(fit, query, objective)
            backends.append(dict(name=name, first_seconds=first_s, warm_seconds=warm_s, speedup=warm_seconds / warm_s,
                                 warm_parity=_parity(warm, reference), **_parity(first, reference)))
    versions = {}
    for dist in ('numpy', 'scipy', 'scikit-learn', 'aeon', 'numba', 'torch', 'mlx', 'mlx-metal'):
        try: versions[dist] = version(dist)
        except PackageNotFoundError: versions[dist] = None
    out = dict(state='complete_acceleration_assessment', seed=508, fit_shape=[64, 128], query_shape=[1024, 128],
               dtype='float64', success=bool(repeated['parity'] and batch['parity']), repeated_fit=repeated, batch_split=batch,
               backends=backends, gpu=_gpu_probes(), versions=versions, platform=platform.platform(), cpu_policy=cpu_worker_policy_antigravity(),
               selected_backend='numpy_float64', backend_promotion=None, tensor_eligible=['rank_objective', 'dense_OAS_geometry', 'distance_attribution'],
               official_transformer_rewritten=False, score_sha256={name: hashlib.sha256(v.tobytes()).hexdigest() for name, v in reference.items()},
               candidate_indices=_candidates(reference), benchmark_scope='Complete head fit+query; all 80 optimizer steps; first call includes JIT compilation and all tensor conversion.',
               fallback_reason='Canonical NumPy retained pending review of full-stage parity and speed. GPU float64 failure prevents preserving the frozen full pipeline.',
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json_atomic_antigravity(target, out)
    print(json.dumps({k: out[k] for k in ('success', 'backends', 'gpu', 'selected_backend')}), flush=True)


if __name__ == '__main__': main()
