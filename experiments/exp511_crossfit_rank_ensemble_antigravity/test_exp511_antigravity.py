"""Role coverage, frozen first member, arithmetic pooling and held-query isolation."""
from pathlib import Path
import importlib
import numpy as np
import pytest


def module(name='ensemble'):
    assert Path(__file__).with_name(name+'_antigravity.py').exists(), name+' implementation missing'
    return importlib.import_module('experiments.exp511_crossfit_rank_ensemble_antigravity.'+name+'_antigravity')


def test_every_row_has_one_ranking_role_and_three_reference_roles():
    m=module()
    from experiments.exp510_separated_rank_antigravity.separated_antigravity import split_roles
    for n in (32,33,34,35,353,1000):
        roles=m.role_folds(n); assert len(roles)==4
        assert np.array_equal(roles[0][0],split_roles(n)[0])
        assert np.array_equal(roles[0][1],split_roles(n)[1])
        ac=np.zeros(n,int);bc=np.zeros(n,int)
        for a,b in roles:
            assert not set(a)&set(b) and set(a)|set(b)==set(range(n))
            assert np.array_equal(a,np.sort(a)) and np.array_equal(b,np.sort(b))
            assert len(b) in (n//4,(n+3)//4)
            ac[a]+=1;bc[b]+=1
        assert np.all(ac==3) and np.all(bc==1)
    for bad in (31,0,True,32.):
        with pytest.raises(ValueError):m.role_folds(bad)


def test_ensemble_replays_exp510_and_is_exact_mean_of_four_components():
    m=module()
    from experiments.exp510_separated_rank_antigravity.separated_antigravity import SeparatedRankHeads
    rng=np.random.default_rng(511);x=rng.normal(size=(33,24));q=rng.normal(size=(13,24))
    h=m.CrossfitRankHeads(x);prior=SeparatedRankHeads(x)
    scores,components=h.score_components(q)
    for name,old in prior.score(q).items():np.testing.assert_array_equal(scores[name],old)
    for name,kind in (('crossfit_unweighted','unweighted'),('crossfit_rank','rank')):
        values=[components[f'member{j}__{kind}'] for j in range(4)]
        np.testing.assert_array_equal(scores[name],np.mean(values,axis=0))
    for member in h.members:
        g=member.contributions(q)
        from experiments.exp502_amplitude_envelope_antigravity.amplitude_antigravity import AmplitudeHeads
        independent=AmplitudeHeads(x[member.a_indices])
        np.testing.assert_allclose(g.sum(axis=1)*member.scale,independent.score(q)['square_joint'],rtol=1e-10,atol=1e-8)
    order=rng.permutation(len(q));permuted=h.score(q[order]);left=h.score(q[:5]);right=h.score(q[5:])
    for name in h.names:
        np.testing.assert_allclose(permuted[name],scores[name][order],rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(np.r_[left[name],right[name]],scores[name],rtol=1e-10,atol=1e-8)


def test_member_geometry_is_unchanged_when_only_its_ranking_rows_change():
    m=module();rng=np.random.default_rng(1511);x=rng.normal(size=(32,17));a,b=m.role_folds(32)[2]
    changed=x.copy();changed[b]=rng.normal(size=(len(b),17))*1.3
    first=m.RankMember(x,a,b);second=m.RankMember(changed,a,b);q=rng.normal(size=(7,17))
    np.testing.assert_array_equal(first.reference,second.reference)
    np.testing.assert_array_equal(first.scaler.center_,second.scaler.center_)
    np.testing.assert_array_equal(first.scaler.scale_,second.scaler.scale_)
    np.testing.assert_array_equal(first.contributions(q),second.contributions(q))
    assert first.scale==second.scale
    with pytest.raises(ValueError):m.RankMember(x,a,np.r_[b,a[0]])


def test_small_sample_fallback_and_constant_inputs_remain_finite():
    m=module()
    for n in (2,31,32):
        h=m.CrossfitRankHeads(np.zeros((n,9)));scores=h.score(np.zeros((3,9)))
        assert all(np.isfinite(v).all() for v in scores.values())
        if n<32:
            assert h.fallback and not h.members
            np.testing.assert_array_equal(scores['crossfit_rank'],scores['square_joint'])
            np.testing.assert_array_equal(scores['crossfit_unweighted'],scores['square_joint'])
        else:
            assert len(h.members)==4
            for member in h.members:assert np.all((member.weights>=.25)&(member.weights<=4))
    with pytest.raises(ValueError):h.score(np.zeros((3,10)))
    with pytest.raises(ValueError):h.score(np.full((3,9),np.nan))


def test_promotion_requires_beating_all_six_controls_without_relaxation():
    m=module('pilot')
    standard={n:dict.fromkeys(m.KINDS,.6) for n in m.NAMES}
    extras={n:dict.fromkeys(m.UNSEEN,.5) for n in m.NAMES};tails=dict.fromkeys(m.NAMES,1.)
    standard['crossfit_rank']=dict.fromkeys(m.KINDS,.625)
    assert m.select_candidate(standard,extras,tails)['full_run_required']
    for control in m.NAMES[:-1]:
        standard[control]=dict.fromkeys(m.KINDS,.61)
        assert not m.select_candidate(standard,extras,tails)['gates']['minimum_margin']
        standard[control]=dict.fromkeys(m.KINDS,.6)
    standard['crossfit_rank']=dict.fromkeys(m.KINDS,np.nextafter(.6+.02,-np.inf))
    assert not m.select_candidate(standard,extras,tails)['gates']['minimum_margin']
    standard['crossfit_rank']=dict.fromkeys(m.KINDS,.625);extras['crossfit_rank'][m.UNSEEN[0]]=.49
    assert not m.select_candidate(standard,extras,tails)['gates']['unseen_nonregression']
    extras['crossfit_rank'][m.UNSEEN[0]]=.5;tails['crossfit_rank']=3.01
    assert not m.select_candidate(standard,extras,tails)['gates']['normal_tail']
    tails['crossfit_rank']=1.;standard['square_joint'][m.KINDS[0]]=.9
    assert not m.select_candidate(standard,extras,tails)['gates']['mean_nonregression']
