"""Independent contracts for extreme-normal ranking and distance attribution."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity, SEED
configure_inner_threads_antigravity()
import importlib
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.distance import cdist


def load(name):
    assert (Path(__file__).parent / f'{name}_antigravity.py').exists(), f'{name} implementation missing'
    return importlib.import_module(f'experiments.exp508_tail_rank_antigravity.{name}_antigravity')


def test_pairwise_gradient_and_extreme_normal_emphasis():
    m = load('optimizer')
    rng = np.random.default_rng(508)
    n, a = rng.uniform(size=(5, 7)), rng.uniform(size=(12, 7))
    w = rng.uniform(.5, 1.5, 7)
    value, gradient = m.loss_gradient(w, n, a)
    numerical = []
    for j in range(len(w)):
        delta = np.zeros_like(w); delta[j] = 1e-6
        numerical.append((m.loss_gradient(w + delta, n, a)[0] - m.loss_gradient(w - delta, n, a)[0]) / 2e-6)
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-6)
    assert np.isfinite(value)
    # The high normal, not the many easy normals, dominates the worst-normal term.
    _, g = m.loss_gradient(np.ones(2), np.array([[6., 0.], [0., 0.]]), np.array([[0., 6.]]))
    assert g[0] > 0 and g[1] < 0
    loss, g = m.loss_gradient(np.ones(2), np.array([[1e4, 0.]]), np.array([[0., 1.]]))
    assert np.isfinite(loss) and np.isfinite(g).all()


def test_optimizer_changes_order_is_deterministic_and_bounded():
    m = load('optimizer')
    n = np.tile([4., 0.], (8, 1)); a = np.tile([0., 2.], (32, 1))
    w, audit = m.fit_tail_weights(n, a)
    again, _ = m.fit_tail_weights(n, a)
    assert np.array_equal(w, again) and np.all((w >= .25) & (w <= 4.))
    assert float(a[0] @ w) > float(n[0] @ w)
    assert m.loss_gradient(w, n, a)[0] < m.loss_gradient(np.ones(2), n, a)[0]
    assert audit['steps'] == 80
    for bad in (np.array([[np.nan, 0.]]), np.array([[-1., 0.]]), np.empty((0, 2))):
        with pytest.raises(ValueError): m.fit_tail_weights(bad, a)
    with pytest.raises(ValueError): m.loss_gradient(np.ones(3), n, a)


def test_distance_attribution_reference_identity_and_parent_exclusion():
    m = load('ranking')
    rng = np.random.default_rng(508)
    reference = rng.normal(size=(7, 9)); query = rng.normal(size=(4, 9))
    g = m.distance_contributions(query, reference)
    expected = np.sort(cdist(query, reference), axis=1)[:, :3].mean(axis=1)
    np.testing.assert_allclose(g.sum(axis=1), expected, rtol=1e-12, atol=1e-12)
    parents = np.array([0, 2, 4, 6])
    g = m.distance_contributions(reference[parents], reference, exclude=parents)
    d = cdist(reference[parents], reference); d[np.arange(4), parents] = np.inf
    np.testing.assert_allclose(g.sum(axis=1), np.sort(d, axis=1)[:, :3].mean(axis=1), rtol=1e-12, atol=1e-12)
    assert np.array_equal(m.distance_contributions(np.zeros((3, 2)), np.zeros((2, 2))), np.zeros((3, 2)))
    with pytest.raises(ValueError): m.distance_contributions(query, reference, exclude=np.array([7, 0, 0, 0]))


def test_rank_head_baselines_fallback_and_batch_isolation():
    m = load('ranking')
    from experiments.exp502_amplitude_envelope_antigravity.amplitude_antigravity import AmplitudeHeads
    rng = np.random.default_rng(508)
    for count in (2, 16):
        x = rng.normal(size=(count, 24)); q = rng.normal(size=(7, 24))
        h = m.TailRankHeads(x); ref = AmplitudeHeads(x)
        got, old = h.score(q), ref.score(q)
        for name in ('MiniRocket', 'oas_distance', 'square_joint'):
            np.testing.assert_allclose(got[name], old[name], rtol=1e-10, atol=1e-8)
            assert np.array_equal(np.argsort(got[name], kind='stable'), np.argsort(old[name], kind='stable'))
        snapshot = h.weights.copy()
        parts = np.r_[h.score(q[:3])['tail_rank'], h.score(q[3:])['tail_rank']]
        np.testing.assert_allclose(parts, got['tail_rank'], rtol=1e-10, atol=1e-8)
        assert np.array_equal(h.weights, snapshot)
        if count < 16: assert np.array_equal(got['tail_rank'], got['square_joint'])
        else: assert len(h.parents) == count and len(np.unique(h.parents)) == count
        with pytest.raises(ValueError): h.score(np.zeros((2, 25)))
    h = m.TailRankHeads(np.zeros((16, 24)))
    assert np.isfinite(h.score(np.zeros((3, 24)))['tail_rank']).all()


def test_unseen_probes_are_deterministic_finite_and_not_training_kinds():
    m = load('ranking')
    from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import KINDS
    x = np.random.default_rng(12).normal(size=(5, 24))
    first, second = m.unseen_perturbations(x, SEED), m.unseen_perturbations(x, SEED)
    assert len(first) == 4 and not set(first).intersection(KINDS)
    for key, value in first.items():
        assert value.shape == x.shape and np.isfinite(value).all() and np.array_equal(value, second[key])


def test_pilot_gate_requires_margin_tail_and_unseen_generalization():
    m = load('pilot')
    from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import KINDS
    names = ('MiniRocket', 'oas_distance', 'square_joint', 'tail_rank')
    summary = {name: {kind: value for kind in KINDS} for name, value in zip(names, (.60, .65, .70, .73))}
    tails = dict.fromkeys(names, 1.)
    unseen = {name: dict.fromkeys(('weak_level', 'weak_scale', 'sparse_spike', 'time_warp'), .65) for name in names}
    result = m.select_candidate(summary, tails, unseen)
    assert result['selected'] == 'tail_rank' and result['best_anchor'] == 'square_joint'
    unseen['tail_rank']['weak_level'] = .60
    assert m.select_candidate(summary, tails, unseen)['selected'] == 'square_joint'
    unseen['tail_rank']['weak_level'] = .65
    tails['tail_rank'] = 3.01
    assert m.select_candidate(summary, tails, unseen)['selected'] == 'square_joint'
    tails['tail_rank'] = 1.
    summary['tail_rank'] = dict.fromkeys(KINDS, .71)
    assert m.select_candidate(summary, tails, unseen)['selected'] == 'square_joint'
