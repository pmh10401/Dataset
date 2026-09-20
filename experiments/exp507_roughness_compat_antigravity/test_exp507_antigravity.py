"""Contracts for Exp507 roughness numerical compatibility."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity,SEED
configure_inner_threads_antigravity()
import numpy as np
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import PairedHeads
from experiments.exp507_roughness_compat_antigravity.roughness_antigravity import roughness_rows,RoughnessHeads,scale_roughness

def _raises(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('expected ValueError')

def test_roughness_rows_fixture_order_scale_sign():
    x=np.array([[0.,1.,4.,2.],[2.,0.,3.,7.]])
    r=roughness_rows(x)
    assert r.dtype==np.float32 and r.shape==x.shape
    assert np.allclose(r[0],np.array([1.,1.,3.,2.])-1.75)
    assert np.allclose(r[0],roughness_rows(x[:1])[0]) and np.allclose(r[1],roughness_rows(x[1:])[0])
    assert np.allclose(roughness_rows(5*x),5*r) and np.allclose(roughness_rows(-x),r)
    assert np.array_equal(roughness_rows([[3.],[8.]]),np.zeros((2,1),np.float32))

def test_official_minirocket_expected_rough_and_dims():
    rng=np.random.default_rng(506);x=rng.normal(size=(6,16))
    heads=RoughnessHeads(x);rough=roughness_rows(x)
    want=make_representation_antigravity('minirocket_local_gap',SEED).fit_transform(rough)
    got=heads.rough_encoder.transform(scale_roughness(rough))
    assert np.array_equal(got,want) and want.shape[1]==4032==heads.diagnostics['rough_featuredim']
    assert heads.diagnostics['scale_power']==20
    raw_ref=make_representation_antigravity('minirocket_local_gap',SEED).fit_transform(x)
    assert np.array_equal(heads.raw_encoder.transform(x),raw_ref)
    assert heads.raw_encoder.transform(x).shape[1]==4032 and heads.models['roughness_joint'][2].shape[1]==8064

def test_baseline_n2_constants_batch_isolation():
    rng=np.random.default_rng(7)
    for n,L in ((2,12),(8,12)):
        fit=rng.normal(size=(n,L));query=rng.normal(size=(n+3,L))
        enc=make_representation_antigravity('minirocket_local_gap',SEED)
        paired=PairedHeads(enc.fit_transform(fit))
        heads=RoughnessHeads(fit)
        got=heads.score(query);ref=paired.score(enc.transform(query))
        assert np.allclose(got['MiniRocket'],ref['MiniRocket'],rtol=1e-10,atol=1e-8)
        assert np.allclose(got['oas_distance'],ref['oas_distance'],rtol=1e-10,atol=1e-8)
        assert all(np.isfinite(got[k]).all() for k in RoughnessHeads.names)
        snap={name:np.array(model[2],copy=True) for name,model in heads.models.items()}
        heads.score(rng.normal(size=(5,L)))
        assert all(np.array_equal(snap[name],heads.models[name][2]) for name in snap)
    z=np.zeros((4,10));z[1]=1;h=RoughnessHeads(z);s=h.score(np.zeros((3,10)))
    assert h.diagnostics['rough_constant_row_fraction']==1.0 and all(np.isfinite(s[k]).all() for k in s)
    fit,query=rng.normal(size=(5,12)),rng.normal(size=(6,12))
    heads=RoughnessHeads(fit);full=heads.score(query)
    parts={k:np.concatenate((heads.score(query[:2])[k],heads.score(query[2:])[k])) for k in RoughnessHeads.names}
    again=heads.score(query)
    assert all(np.array_equal(full[k],parts[k]) and np.array_equal(full[k],again[k]) for k in RoughnessHeads.names)

def test_fallback_and_rejects():
    rng=np.random.default_rng(3)
    for L in (1,2):
        x=rng.normal(size=(4,L));h=RoughnessHeads(x);s=h.score(x)
        assert h.fallback and h.rough_encoder is None and h.diagnostics['fallback'] and h.diagnostics['rough_featuredim']==0
        assert np.array_equal(s['roughness_distance'],s['oas_distance']) and np.array_equal(s['roughness_joint'],s['oas_distance'])
    x=rng.normal(size=(4,3));h=RoughnessHeads(x);s=h.score(x)
    assert (not h.fallback) and h.rough_encoder is not None and h.diagnostics['rough_featuredim']==4032
    assert np.isfinite(s['roughness_distance']).all()
    h=RoughnessHeads(rng.normal(size=(3,8)))
    _raises(lambda:h.score(rng.normal(size=(2,7))))
    _raises(lambda:RoughnessHeads(rng.normal(size=(1,8))))
    _raises(lambda:RoughnessHeads([[1.,np.nan],[0.,1.]]))
    _raises(lambda:roughness_rows([[1.,np.inf],[0.,1.]]))
    _raises(lambda:roughness_rows([[0,1e40,4e40,2e40]]))
    _raises(lambda:h.score([[1.,np.nan]+[0.]*6,[0.]*8]))
    _raises(lambda:scale_roughness(np.array([[1e35,-1e35]],dtype=np.float32)))

def test_unscaled_vs_scale_ordinary_parity():
    rng=np.random.default_rng(507)
    for n,L in ((2,12),(8,16)):
        fit=rng.normal(size=(n,L));query=rng.normal(size=(n+4,L))
        ru,rq=roughness_rows(fit),roughness_rows(query)
        eu=make_representation_antigravity('minirocket_local_gap',SEED)
        es=make_representation_antigravity('minirocket_local_gap',SEED)
        assert np.array_equal(eu.fit_transform(ru),es.fit_transform(scale_roughness(ru)))
        assert np.array_equal(eu.transform(rq),es.transform(scale_roughness(rq)))

def test_nearlinear_unscaled_query_rejects_scaled_accepts():
    rng=np.random.default_rng(507)
    t=np.arange(128,dtype=np.float64)
    x=t+1e-6*rng.standard_normal((16,128))
    x=x-x.mean(axis=1,keepdims=True)
    x=np.asarray(x/x.std(axis=1,keepdims=True),dtype=np.float32)
    rough=roughness_rows(x)
    assert float(np.std(rough))>0
    unscaled_fit_rejected=False
    try:make_representation_antigravity('minirocket_local_gap',SEED).fit_transform(rough)
    except Exception:unscaled_fit_rejected=True
    assert unscaled_fit_rejected
    enc=make_representation_antigravity('minirocket_local_gap',SEED)
    scaled=scale_roughness(rough)
    assert np.isfinite(enc.fit_transform(scaled)).all()
    unscaled_query_rejected=False
    try:enc.transform(rough)
    except Exception:unscaled_query_rejected=True
    assert unscaled_query_rejected
    q=enc.transform(scaled)
    assert np.isfinite(q).all() and q.shape[1]==4032
    s=RoughnessHeads(x).score(x)
    assert all(np.isfinite(s[k]).all() for k in RoughnessHeads.names)
