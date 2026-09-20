"""Small numerical check of acceleration candidates before benchmarking."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
import importlib
import numpy as np


def test_accelerated_objective_matches_independent_numpy_gradient():
    assert (Path(__file__).parent / 'acceleration_antigravity.py').exists(), 'acceleration implementation missing'
    m = importlib.import_module('experiments.exp508_tail_rank_antigravity.acceleration_antigravity')
    from experiments.exp508_tail_rank_antigravity.optimizer_antigravity import loss_gradient
    rng = np.random.default_rng(508)
    n, a, w = rng.uniform(size=(5, 7)), rng.uniform(size=(20, 7)), rng.uniform(.5, 1.5, 7)
    loss, gradient = loss_gradient(w, n, a)
    for fn in (m.objective_numba, m.objective_torch):
        value, actual = fn(w, n, a)
        np.testing.assert_allclose(value, loss, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual, gradient, rtol=1e-12, atol=1e-12)


def test_mlx_probe_matches_actual_gpu_matrix_product():
    import pytest
    mx = pytest.importorskip('mlx.core')
    m = importlib.import_module('experiments.exp508_tail_rank_antigravity.acceleration_antigravity')
    try:
        a = mx.array([[1., 2.], [3., 4.]], dtype=mx.float64)
        mx.eval(mx.matmul(a, a, stream=mx.gpu))
        supported = True
    except Exception:
        supported = False
    assert m._gpu_probes()['mlx']['float64_supported'] == supported
