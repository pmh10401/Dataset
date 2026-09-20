"""Independent numeric and sample-isolation contracts for the Exp509 diagnostic."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
import importlib
import numpy as np
import pytest
from scipy.special import logsumexp


def module():
    assert Path(__file__).with_name('diagnostic_antigravity.py').exists(), 'diagnostic implementation missing'
    return importlib.import_module('experiments.exp509_rank_signal_diagnosis_antigravity.diagnostic_antigravity')


def test_bounded_solver_improves_and_satisfies_projected_gradient():
    m = module()
    from experiments.exp508_tail_rank_antigravity.optimizer_antigravity import loss_gradient
    n = np.tile([4., 0.], (8, 1)); a = np.tile([0., 2.], (32, 1))
    w, info = m.solve(n, a)
    loss, gradient = loss_gradient(w, n, a)
    assert loss < loss_gradient(np.ones(2), n, a)[0]
    assert np.all((w >= .25) & (w <= 4.))
    assert np.max(abs(w - np.clip(w-gradient, .25, 4.))) <= 1e-8
    assert info['projected_gradient_passed'] and info['objective'] == loss
    assert float(a[0] @ w) > float(n[0] @ w)
    # A constant feature problem has the exact solution w=1, not arbitrary bounds.
    wz, _ = m.solve(np.zeros((2, 4)), np.zeros((8, 4)))
    np.testing.assert_array_equal(wz, np.ones(4))
    for bad in (np.empty((0, 2)), np.array([[np.nan, 0.]]), np.array([[-1., 0.]])):
        with pytest.raises(ValueError): m.solve(bad, a)


def test_convex_certificate_has_correct_boundary_sign_and_exact_quadratic_gap():
    m = module()
    w = np.array([.25, .8, 2., 4.]); linear = np.array([.7, -.03, .08, -.3]); mu = .2/4
    gradient = mu*(w-1)+linear
    optimum = np.clip(1-linear/mu, .25, 4.)
    f = lambda z: .1*np.mean((z-1)**2) + linear@z
    cert = m.certificate(w, gradient)
    np.testing.assert_allclose(cert['objective_gap_upper_bound'], f(w)-f(optimum), atol=1e-14)
    assert m.certificate(optimum, mu*(optimum-1)+linear)['projected_gradient_inf'] < 1e-14
    with pytest.raises(ValueError): m.certificate(np.array([5.]), np.array([0.]))


def test_signal_statistics_agree_with_manual_loss_and_preserve_extremes():
    m = module(); ns = np.array([0., 2.]); az = np.array([3., 4., 5., 6.]); w = np.array([.5, 1.5])
    stats = m.signal_statistics(ns, az, w)
    expected = np.logaddexp(0., .25+.1*logsumexp(ns/.1)-az).mean()+.1*np.mean((w-1)**2)
    np.testing.assert_allclose(stats['objective'], expected, atol=1e-14)
    assert stats['normal_max'] == 2. and stats['above_normal_max_fraction'] == 1.
    extreme = m.signal_statistics(np.array([1e4]), np.array([0., 2., 4., 6.]), np.ones(3))
    assert np.isfinite(extreme['objective']) and extreme['active_001_fraction'] == 1.
    with pytest.raises(ValueError): m.signal_statistics(np.array([]), az, w)


def test_roles_are_equal_size_disjoint_deterministic_and_cover_fit():
    m = module()
    for size in (32, 33, 319):
        first = m.roles(size, 2); second = m.roles(size, 2)
        assert all(np.array_equal(first[k], second[k]) for k in first)
        r, s, b = (set(first[k]) for k in ('reference', 'seen', 'unseen'))
        assert len(s) == len(b) == size//4
        assert not (r&s or r&b or s&b) and r|s|b == set(range(size))
        assert set(first['representation_fit']) == r|s
    with pytest.raises(ValueError): m.roles(7, 0)


def test_capture_replays_original_optimizer_and_control_geometry_is_shared():
    m = module()
    from experiments.exp508_tail_rank_antigravity import ranking_antigravity as rank
    from experiments.exp508_tail_rank_antigravity.optimizer_antigravity import loss_gradient
    original = rank.fit_tail_weights
    rows = np.random.default_rng(509).normal(size=(32, 24))
    head, n, a = m.capture(rows)
    assert rank.fit_tail_weights is original
    np.testing.assert_allclose(loss_gradient(head.weights, n, a)[0], head.diagnostics['final_objective'], rtol=1e-12)
    evidence, meta = m.participation_probe(rows, 0)
    assert len(evidence['seen_normal']) == len(evidence['unseen_normal']) == 8
    assert meta['common_reference_count'] == 16 and meta['representation_fit_count'] == 24
    assert not set(meta['roles']['unseen']) & set(meta['roles']['representation_fit'])
    assert set(evidence) == {'seen_normal', 'seen_anomaly', 'unseen_normal', 'unseen_anomaly'}
    assert all(np.isfinite(x).all() for x in evidence.values())
    # Identity of the square-joint distance and the all-one attribution score.
    query = np.random.default_rng(27).normal(size=(3, 24))
    g = rank.distance_contributions(head._geometry(query), head.reference)/head.scale
    np.testing.assert_allclose(g.sum(axis=1)*head.scale, head.base.score(query)['square_joint'], rtol=1e-10, atol=1e-8)
