"""Contracts for role separation, frozen controls, inference and selection."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
import importlib
import numpy as np
import pytest


def module(name='separated'):
    assert Path(__file__).with_name(name+'_antigravity.py').exists(), name+' implementation missing'
    return importlib.import_module('experiments.exp510_separated_rank_antigravity.'+name+'_antigravity')


def test_split_is_fixed_disjoint_exhaustive_and_has_small_sample_fallback():
    m=module()
    for size in (2,31,32,33,1000):
        a,b=m.split_roles(size); aa,bb=m.split_roles(size)
        assert np.array_equal(a,aa) and np.array_equal(b,bb)
        assert len(np.intersect1d(a,b))==0 and sorted(np.r_[a,b].tolist())==list(range(size))
        assert np.array_equal(a,np.sort(a)) and np.array_equal(b,np.sort(b))
        assert len(b)==(size//4 if size>=32 else 0)
    for bad in (1,0,32.,True):
        with pytest.raises(ValueError): m.split_roles(bad)
    b=np.arange(250)
    parents=m.ranking_parents(b)
    assert len(parents)==128 and len(np.unique(parents))==128 and set(parents)<=set(b)
    np.testing.assert_array_equal(parents,m.ranking_parents(b))


def test_full_controls_and_same_reference_control_match_independent_models():
    m=module()
    from experiments.exp502_amplitude_envelope_antigravity.amplitude_antigravity import AmplitudeHeads
    x=np.random.default_rng(510).normal(size=(32,24));q=np.random.default_rng(1510).normal(size=(12,24))
    h=m.SeparatedRankHeads(x);full=AmplitudeHeads(x);subset=AmplitudeHeads(x[h.a_indices])
    got=h.score(q);old=full.score(q)
    for name in ('MiniRocket','oas_distance','square_joint'):
        np.testing.assert_allclose(got[name],old[name],rtol=1e-10,atol=1e-8)
        np.testing.assert_array_equal(np.argsort(got[name],kind='stable'),np.argsort(old[name],kind='stable'))
    np.testing.assert_allclose(got['a_unweighted']*h.scale,subset.score(q)['square_joint'],rtol=1e-10,atol=1e-8)
    g=h.contributions(q)
    np.testing.assert_allclose(g.sum(axis=1),got['a_unweighted'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(g@h.weights,got['separated_rank'],rtol=1e-12,atol=1e-12)
    assert len(h.parents)==8 and not set(h.parents)&set(h.a_indices)
    assert h.audit['normal_scores'].shape==(8,) and h.audit['anomaly_scores'].shape==(32,)


def test_changing_b_cannot_change_candidate_representation_or_reference():
    m=module();rng=np.random.default_rng(610)
    x=rng.normal(size=(32,24));a,b=m.split_roles(len(x));changed=x.copy()
    changed[b]=rng.normal(size=(len(b),24))*1.3
    first=m.SeparatedRankHeads(x);second=m.SeparatedRankHeads(changed)
    q=rng.normal(size=(9,24))
    # This fails if any B row enters candidate encoder/scaler/OAS/reference fitting.
    np.testing.assert_array_equal(first.reference,second.reference)
    np.testing.assert_array_equal(first.geometry.center,second.geometry.center)
    np.testing.assert_array_equal(first.scaler.center_,second.scaler.center_)
    np.testing.assert_array_equal(first.scaler.scale_,second.scaler.scale_)
    np.testing.assert_allclose(first.contributions(q),second.contributions(q),rtol=0,atol=0)
    assert first.scale==second.scale


def test_query_batch_order_and_fitted_state_are_independent():
    m=module();rng=np.random.default_rng(710);x=rng.normal(size=(32,17));q=rng.normal(size=(13,17))
    h=m.SeparatedRankHeads(x);weights=h.weights.copy();reference=h.reference.copy()
    scores=h.score(q);order=rng.permutation(len(q));permuted=h.score(q[order])
    batches=[h.score(q[:5]),h.score(q[5:])]
    for name in h.names:
        np.testing.assert_allclose(scores[name][order],permuted[name],rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(scores[name],np.r_[batches[0][name],batches[1][name]],rtol=1e-10,atol=1e-8)
    np.testing.assert_array_equal(weights,h.weights);np.testing.assert_array_equal(reference,h.reference)
    assert np.all((weights>=.25)&(weights<=4.))
    with pytest.raises(ValueError): h.score(np.zeros((2,18)))
    with pytest.raises(ValueError): h.score(np.full((2,17),np.nan))


def test_fallback_and_constant_short_rows_do_not_invent_a_trained_model():
    m=module()
    for count in (2,31,32):
        h=m.SeparatedRankHeads(np.zeros((count,9)));s=h.score(np.zeros((3,9)))
        assert all(np.isfinite(v).all() for v in s.values())
        assert h.diagnostics['fallback']==(count<32)
        if count<32:
            np.testing.assert_array_equal(s['separated_rank'],s['square_joint'])
            np.testing.assert_array_equal(s['a_unweighted'],s['square_joint'])
            assert not h.diagnostics['optimizer_trained'] and len(h.parents)==0
        else:
            assert h.diagnostics['optimizer_trained']
            np.testing.assert_array_equal(h.weights,np.ones_like(h.weights))


def test_gate_must_beat_full_and_same_reference_controls_without_tuning():
    m=module('pilot');names=m.NAMES
    standard={n:dict.fromkeys(m.KINDS,v) for n,v in zip(names,(.50,.55,.60,.58,.625))}
    unseen={n:dict.fromkeys(m.UNSEEN,.50) for n in names};tails=dict.fromkeys(names,1.)
    assert m.select_candidate(standard,unseen,tails)['selected']=='separated_rank'
    standard['a_unweighted']=dict.fromkeys(m.KINDS,.61)
    assert m.select_candidate(standard,unseen,tails)['selected']=='square_joint'
    standard['a_unweighted']=dict.fromkeys(m.KINDS,.58)
    standard['separated_rank']=dict.fromkeys(m.KINDS,np.nextafter(.60+.02,-np.inf))
    assert not m.select_candidate(standard,unseen,tails)['gates']['minimum_margin']
    standard['separated_rank']=dict.fromkeys(m.KINDS,.625)
    unseen['separated_rank'][m.UNSEEN[0]]=.49
    assert not m.select_candidate(standard,unseen,tails)['gates']['unseen_nonregression']
    unseen['separated_rank'][m.UNSEEN[0]]=.50;tails['separated_rank']=3.01
    assert not m.select_candidate(standard,unseen,tails)['gates']['normal_tail']
    tails['separated_rank']=1.;standard['square_joint'][m.KINDS[1]]=.90
    assert not m.select_candidate(standard,unseen,tails)['gates']['mean_nonregression']
