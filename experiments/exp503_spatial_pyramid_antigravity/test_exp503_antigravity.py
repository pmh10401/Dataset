"""Interpolation, sample-major shared encoder, baseline and OAS parity, isolation, fallback."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
import numpy as np
import pytest
from experiments.exp503_spatial_pyramid_antigravity.spatial_antigravity import quarter_segments,SpatialHeads,SEED
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import PairedHeads

def test_unequal_linear_interpolation_and_constants():
    row=np.array([[0.,10.,2.,1.,3.,5.,4.,8.,7.,9.],[6.,6.,6.,6.,6.,6.,6.,6.,6.,6.]])
    out=quarter_segments(row)
    assert out.shape==(8,9)
    target=np.linspace(0.0,1.0,9)
    np.testing.assert_allclose(out[0],np.interp(target,np.linspace(0.0,1.0,3),np.array([0.,10.,2.])))
    np.testing.assert_allclose(out[0],[0.,2.5,5.,7.5,10.,8.,6.,4.,2.])
    np.testing.assert_allclose(out[1],np.interp(target,np.linspace(0.0,1.0,3),np.array([1.,3.,5.])))
    np.testing.assert_allclose(out[2],np.interp(target,np.linspace(0.0,1.0,2),np.array([4.,8.])))
    np.testing.assert_allclose(out[3],np.interp(target,np.linspace(0.0,1.0,2),np.array([7.,9.])))
    np.testing.assert_allclose(out[4:8],6.)
    short=np.array([[1.,2.,3.,4.],[5.,5.,5.,5.]])
    resampled=quarter_segments(short)
    assert resampled.shape==(8,9)
    np.testing.assert_allclose(resampled[0],1.)
    np.testing.assert_allclose(resampled[1],2.)
    np.testing.assert_allclose(resampled[2],3.)
    np.testing.assert_allclose(resampled[3],4.)
    np.testing.assert_allclose(resampled[4:8],5.)

def test_sample_major_shared_encoder():
    rng=np.random.default_rng(503);x=rng.normal(size=(6,36));q=rng.normal(size=(4,36))
    stacked=quarter_segments(q)
    assert stacked.shape==(16,9)
    parts=np.array_split(q,4,axis=1)
    for i in range(4):
        for p in range(4):
            np.testing.assert_allclose(stacked[i*4+p],parts[p][i])
    head=SpatialHeads(x);order=(2,0,3,1)
    feat=head.quarter_encoder.transform(quarter_segments(q)).reshape(len(q),4,-1)
    permuted=np.concatenate([parts[i] for i in order],axis=1)
    swapped=head.quarter_encoder.transform(quarter_segments(permuted)).reshape(len(q),4,-1)
    np.testing.assert_allclose(swapped,feat[:,order,:],rtol=1e-10,atol=1e-8)
    assert head.diagnostics['fallback'] is False
    assert head.diagnostics['quarterlength']==9
    assert head.diagnostics['quarterfeaturedim']==1008

def _parity_and_isolation(fit,query):
    head=SpatialHeads(fit);scores=head.score(query)
    encoder=make_representation_antigravity('minirocket_local_gap',SEED)
    old=PairedHeads(encoder.fit_transform(fit)).score(encoder.transform(query))
    for name in ('MiniRocket','oas_distance'):np.testing.assert_allclose(scores[name],old[name],rtol=1e-10,atol=1e-8)
    before={name:r.copy() for name,(_,_,r) in head.models.items()};single=head.score(query[:1]);head.score(query*4)
    for name in head.names:np.testing.assert_allclose(single[name],scores[name][:1],rtol=1e-10,atol=1e-8)
    for name,(_,_,reference) in head.models.items():np.testing.assert_array_equal(before[name],reference)
    return head,scores

def test_baseline_raw_oas_isolation_and_n2():
    rng=np.random.default_rng(503);x=rng.normal(size=(12,64));q=rng.normal(size=(5,64))
    for n in (2,12):
        head,scores=_parity_and_isolation(x[:n],q)
        assert head.names==('MiniRocket','oas_distance','pyramid_distance','pyramid_joint')
        assert set(scores)==set(head.names)
        assert all(v.shape==(len(q),) and np.isfinite(v).all() for v in scores.values())
        assert head.diagnostics['fallback'] is False
        assert head.diagnostics['quarterlength']==max(9,64//4)
        assert head.diagnostics['quarterfeaturedim']==1008
    with pytest.raises(ValueError):SpatialHeads(x).score(q[:,:-1])

def test_short_length_fallback_matches_raw_oas():
    rng=np.random.default_rng(503)
    for length in (1,2,3):
        fit=rng.normal(size=(8,length));query=rng.normal(size=(5,length))
        head,scores=_parity_and_isolation(fit,query)
        assert head.diagnostics==dict(fallback=True,quarterlength=0,quarterfeaturedim=0)
        np.testing.assert_array_equal(scores['pyramid_distance'],scores['oas_distance'])
        np.testing.assert_array_equal(scores['pyramid_joint'],scores['oas_distance'])
        assert head.quarter_encoder is None
    with pytest.raises(ValueError):quarter_segments(np.ones((2,3)))
