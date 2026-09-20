"""Official QUANT parity, cap-512 fixture, baseline OAS, isolation, fallback."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
import numpy as np
import pytest
from aeon.transformations.collection.interval_based import QUANTTransformer
from experiments.exp504_quant_geometry_antigravity.quant_antigravity import cap_series,QuantHeads,SEED,QUANT_CAP,INTERVAL_DEPTH,QUANTILE_DIVISOR
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import PairedHeads

def test_cap_constants_and_official_quant_parity():
    assert SEED==20260721 and QUANT_CAP==512 and INTERVAL_DEPTH==6 and QUANTILE_DIVISOR==4
    assert QuantHeads.names==('MiniRocket','oas_distance','quant_distance','quant_joint')
    rng=np.random.default_rng(504)
    x=rng.normal(size=(6,128));collection=cap_series(x)
    assert collection.shape==(6,1,128) and collection.dtype==np.float32 and np.isfinite(collection).all()
    official=QUANTTransformer(interval_depth=INTERVAL_DEPTH,quantile_divisor=QUANTILE_DIVISOR)
    expected=np.asarray(official.fit_transform(collection),dtype=np.float64)
    assert expected.shape==(6,1160)
    head=QuantHeads(x)
    got=np.asarray(head.quant_encoder.transform(collection),dtype=np.float64)
    np.testing.assert_allclose(got,expected,rtol=1e-10,atol=1e-8)
    assert head.diagnostics==dict(fallback=False,quantlength=128,quantfeaturedim=1160)
    y=rng.normal(size=(4,777));capped=cap_series(y)
    assert capped.shape==(4,1,512) and capped.dtype==np.float32
    target=np.linspace(0.0,1.0,512);source=np.linspace(0.0,1.0,777)
    np.testing.assert_allclose(capped[0,0],np.interp(target,source,y[0]).astype(np.float32),rtol=1e-10,atol=1e-8)
    official512=QUANTTransformer(interval_depth=INTERVAL_DEPTH,quantile_divisor=QUANTILE_DIVISOR)
    expected512=np.asarray(official512.fit_transform(capped),dtype=np.float64)
    assert expected512.shape==(4,4500)
    head512=QuantHeads(y)
    got512=np.asarray(head512.quant_encoder.transform(capped),dtype=np.float64)
    np.testing.assert_allclose(got512,expected512,rtol=1e-10,atol=1e-8)
    assert head512.diagnostics==dict(fallback=False,quantlength=512,quantfeaturedim=4500)
    assert cap_series(rng.normal(size=(2,2700))).shape==(2,1,512)

def _parity_and_isolation(fit,query):
    head=QuantHeads(fit);scores=head.score(query)
    encoder=make_representation_antigravity('minirocket_local_gap',SEED)
    old=PairedHeads(encoder.fit_transform(fit)).score(encoder.transform(query))
    for name in ('MiniRocket','oas_distance'):np.testing.assert_allclose(scores[name],old[name],rtol=1e-10,atol=1e-8)
    before={name:r.copy() for name,(_,_,r) in head.models.items()};single=head.score(query[:1]);head.score(query*4)
    for name in head.names:np.testing.assert_allclose(single[name],scores[name][:1],rtol=1e-10,atol=1e-8)
    for name,(_,_,reference) in head.models.items():np.testing.assert_array_equal(before[name],reference)
    return head,scores

def test_baseline_raw_oas_isolation_and_n2():
    rng=np.random.default_rng(504);x=rng.normal(size=(12,64));q=rng.normal(size=(5,64))
    for n in (2,12):
        head,scores=_parity_and_isolation(x[:n],q)
        assert head.names==('MiniRocket','oas_distance','quant_distance','quant_joint')
        assert set(scores)==set(head.names)
        assert all(v.shape==(len(q),) and np.isfinite(v).all() for v in scores.values())
        assert head.diagnostics['fallback'] is False
        assert head.diagnostics['quantlength']==64
    with pytest.raises(ValueError):QuantHeads(x).score(q[:,:-1])
    with pytest.raises(ValueError):QuantHeads(np.ones((1,64)))
    with pytest.raises(ValueError):QuantHeads(np.array([[1.,2.,3.],[np.nan,0.,1.]]))
    bad=q.copy();bad[0,-1]=np.inf
    with pytest.raises(ValueError):QuantHeads(x).score(bad)

def test_short_length_fallback_and_l3():
    rng=np.random.default_rng(504)
    for length in (1,2):
        fit=rng.normal(size=(8,length));query=rng.normal(size=(5,length))
        head,scores=_parity_and_isolation(fit,query)
        assert head.diagnostics==dict(fallback=True,quantlength=0,quantfeaturedim=0)
        np.testing.assert_array_equal(scores['quant_distance'],scores['oas_distance'])
        np.testing.assert_array_equal(scores['quant_joint'],scores['oas_distance'])
        assert head.quant_encoder is None
    fit=rng.normal(size=(8,3));query=rng.normal(size=(5,3))
    head,scores=_parity_and_isolation(fit,query)
    assert head.diagnostics==dict(fallback=False,quantlength=3,quantfeaturedim=10)
    assert head.quant_encoder is not None
    assert all(v.shape==(5,) and np.isfinite(v).all() for v in scores.values())
    collection=cap_series(fit)
    official=QUANTTransformer(interval_depth=INTERVAL_DEPTH,quantile_divisor=QUANTILE_DIVISOR)
    expected=np.asarray(official.fit_transform(collection),dtype=np.float64)
    assert expected.shape==(8,10)
    np.testing.assert_allclose(np.asarray(head.quant_encoder.transform(collection),dtype=np.float64),expected,rtol=1e-10,atol=1e-8)

def test_constant_series_cap_and_score():
    out=cap_series(np.full((2,777),2.5))
    np.testing.assert_array_equal(out,np.float32(2.5))
    scores=QuantHeads(np.zeros((4,128))).score(np.zeros((3,128)))
    assert set(scores)==set(QuantHeads.names) and all(v.shape==(3,) and np.isfinite(v).all() for v in scores.values())
