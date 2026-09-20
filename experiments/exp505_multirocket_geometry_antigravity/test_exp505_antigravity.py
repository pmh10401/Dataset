"""Official MultiRocket features, scaler/geometry equality, baseline OAS, isolation, short L, rejects."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
import numpy as np
import pytest
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket
from sklearn.preprocessing import RobustScaler
from experiments.exp505_multirocket_geometry_antigravity.multirocket_antigravity import MultiRocketHeads,SEED,pad_multirocket_rows
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import PairedHeads,OASGeometry

def test_official_features_scaler_geometry_and_joint_dim():
    assert SEED==20260721
    assert MultiRocketHeads.names==('MiniRocket','oas_distance','multirocket_distance','multirocket_joint')
    rng=np.random.default_rng(505)
    x=rng.normal(size=(6,128))
    collection=np.asarray(x,dtype=np.float32)[:,np.newaxis,:]
    expected_raw=np.asarray(MiniRocket(n_kernels=4096,n_jobs=1,random_state=SEED).fit_transform(collection),dtype=float)
    expected_multi=np.asarray(MultiRocket(n_kernels=1024,max_dilations_per_kernel=32,n_features_per_kernel=4,normalise=False,n_jobs=1,random_state=SEED).fit_transform(collection),dtype=float)
    assert expected_raw.shape==(6,4032)
    head=MultiRocketHeads(x)
    np.testing.assert_allclose(head.raw_encoder.transform(x),expected_raw,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(head.multi_encoder.transform(x),expected_multi,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(make_representation_antigravity('minirocket_local_gap',SEED).fit_transform(x),expected_raw,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(make_representation_antigravity('multirocket_local_gap',SEED).fit_transform(x),expected_multi,rtol=1e-10,atol=1e-8)
    assert head.length==128
    assert set(head.diagnostics)=={'raw_featuredim','multi_featuredim','multi_zero_iqr_fraction'}
    assert head.diagnostics['raw_featuredim']==4032
    assert head.diagnostics['multi_featuredim']==int(expected_multi.shape[1])==8064
    assert 0<=head.diagnostics['multi_zero_iqr_fraction']<=1
    assert head.models['multirocket_joint'][2].shape==(6,4032+int(expected_multi.shape[1]))
    assert head.models['oas_distance'][0] is head.control.scaler
    control=LocalGapModelAntigravity(expected_raw)
    np.testing.assert_allclose(head.control.scaler.center_,control.scaler.center_,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(head.control.scaler.scale_,control.scaler.scale_,rtol=1e-10,atol=1e-8)
    z=control.scaler.transform(expected_raw)
    oas=OASGeometry(z);geo=head.models['oas_distance'][1]
    np.testing.assert_allclose(geo.center,oas.center,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(geo.basis,oas.basis,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(geo.factor,oas.factor,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose([geo.shrinkage,geo.tau],[oas.shrinkage,oas.tau],rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(head.models['oas_distance'][2],oas.transform(z),rtol=1e-10,atol=1e-8)
    ms=RobustScaler(quantile_range=(10,90)).fit(expected_multi)
    np.testing.assert_allclose(head.models['multirocket_distance'][0].center_,ms.center_,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(head.models['multirocket_distance'][0].scale_,ms.scale_,rtol=1e-10,atol=1e-8)
    mz=ms.transform(expected_multi);moas=OASGeometry(mz)
    np.testing.assert_allclose(head.models['multirocket_distance'][1].center,moas.center,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(head.models['multirocket_distance'][2],moas.transform(mz),rtol=1e-10,atol=1e-8)
    joint=np.column_stack((expected_raw,expected_multi))
    js=RobustScaler(quantile_range=(10,90)).fit(joint)
    np.testing.assert_allclose(head.models['multirocket_joint'][0].center_,js.center_,rtol=1e-10,atol=1e-8)
    np.testing.assert_allclose(head.models['multirocket_joint'][0].scale_,js.scale_,rtol=1e-10,atol=1e-8)
    jz=js.transform(joint);joas=OASGeometry(jz)
    np.testing.assert_allclose(head.models['multirocket_joint'][2],joas.transform(jz),rtol=1e-10,atol=1e-8)

def _parity_and_isolation(fit,query):
    head=MultiRocketHeads(fit);scores=head.score(query)
    encoder=make_representation_antigravity('minirocket_local_gap',SEED)
    old=PairedHeads(encoder.fit_transform(fit)).score(encoder.transform(query))
    for name in ('MiniRocket','oas_distance'):np.testing.assert_allclose(scores[name],old[name],rtol=1e-10,atol=1e-8)
    padded=pad_multirocket_rows(fit)
    assert padded.shape==(len(fit),max(10,int(fit.shape[1])))
    raw_fit=head.raw_encoder.transform(fit).copy();multi_fit=head.multi_encoder.transform(padded).copy()
    before={name:r.copy() for name,(_,_,r) in head.models.items()};single=head.score(query[:1]);head.score(np.concatenate((query,query)))
    for name in head.names:np.testing.assert_allclose(single[name],scores[name][:1],rtol=1e-10,atol=1e-8)
    for name,(_,_,reference) in head.models.items():np.testing.assert_array_equal(before[name],reference)
    np.testing.assert_array_equal(raw_fit,head.raw_encoder.transform(fit))
    np.testing.assert_array_equal(multi_fit,head.multi_encoder.transform(padded))
    return head,scores

def test_baseline_raw_oas_isolation_and_n2():
    rng=np.random.default_rng(505);x=rng.normal(size=(12,64));q=rng.normal(size=(5,64))
    for n in (2,12):
        head,scores=_parity_and_isolation(x[:n],q)
        assert head.names==('MiniRocket','oas_distance','multirocket_distance','multirocket_joint')
        assert set(scores)==set(head.names)
        assert all(v.shape==(len(q),) and np.isfinite(v).all() for v in scores.values())
        assert head.length==64
        assert head.diagnostics['raw_featuredim']+head.diagnostics['multi_featuredim']==head.models['multirocket_joint'][2].shape[1]
        assert head.diagnostics['multi_featuredim']==8064
        assert head.models['multirocket_joint'][2].shape[0]==n
    with pytest.raises(ValueError):MultiRocketHeads(x).score(q[:,:-1])
    with pytest.raises(ValueError):MultiRocketHeads(np.ones((1,64)))
    with pytest.raises(ValueError):MultiRocketHeads(np.array([[1.,2.,3.],[np.nan,0.,1.]]))
    bad=q.copy();bad[0,-1]=np.inf
    with pytest.raises(ValueError):MultiRocketHeads(x).score(bad)

def test_short_length_constants_and_l9():
    rng=np.random.default_rng(505)
    for length in (1,2,9):
        fit=rng.normal(size=(8,length));query=rng.normal(size=(5,length))
        head,scores=_parity_and_isolation(fit,query)
        assert head.length==length
        explicit=np.pad(np.asarray(fit,dtype=np.float64),((0,0),(0,10-length)))
        assert explicit.shape==(8,10)
        np.testing.assert_array_equal(pad_multirocket_rows(fit),explicit)
        collection=np.asarray(explicit,dtype=np.float32)[:,np.newaxis,:]
        official=MultiRocket(n_kernels=1024,max_dilations_per_kernel=32,n_features_per_kernel=4,normalise=False,n_jobs=1,random_state=SEED)
        expected=np.asarray(official.fit_transform(collection),dtype=float)
        assert expected.shape==(8,8064) and np.isfinite(expected).all()
        np.testing.assert_allclose(np.asarray(official.transform(collection),dtype=float),expected,rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(head.multi_encoder.transform(explicit),expected,rtol=1e-10,atol=1e-8)
        assert head.diagnostics['raw_featuredim']>=1 and head.diagnostics['multi_featuredim']==8064
        assert head.models['multirocket_joint'][2].shape==(8,head.diagnostics['raw_featuredim']+8064)
        assert all(v.shape==(5,) and np.isfinite(v).all() for v in scores.values())
        const_fit=np.full((4,length),2.5);const_query=np.full((3,length),2.5)
        chead,cscores=_parity_and_isolation(const_fit,const_query)
        cexplicit=np.pad(np.asarray(const_fit,dtype=np.float64),((0,0),(0,10-length)))
        ccollection=np.asarray(cexplicit,dtype=np.float32)[:,np.newaxis,:]
        cofficial=MultiRocket(n_kernels=1024,max_dilations_per_kernel=32,n_features_per_kernel=4,normalise=False,n_jobs=1,random_state=SEED)
        cexpected=np.asarray(cofficial.fit_transform(ccollection),dtype=float)
        assert cexpected.shape==(4,8064) and np.isfinite(cexpected).all()
        np.testing.assert_allclose(np.asarray(cofficial.transform(ccollection),dtype=float),cexpected,rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(chead.multi_encoder.transform(cexplicit),cexpected,rtol=1e-10,atol=1e-8)
        assert all(v.shape==(3,) and np.isfinite(v).all() for v in cscores.values())
    for n,length in ((2,10),(4,10),(12,128)):
        fit=rng.normal(size=(n,length));query=rng.normal(size=(3,length))
        head,scores=_parity_and_isolation(fit,query)
        assert head.length==length
        np.testing.assert_array_equal(pad_multirocket_rows(fit),np.asarray(fit,dtype=np.float64))
        collection=np.asarray(fit,dtype=np.float32)[:,np.newaxis,:]
        official=MultiRocket(n_kernels=1024,max_dilations_per_kernel=32,n_features_per_kernel=4,normalise=False,n_jobs=1,random_state=SEED)
        expected=np.asarray(official.fit_transform(collection),dtype=float)
        assert expected.shape==(n,8064) and np.isfinite(expected).all()
        np.testing.assert_allclose(np.asarray(official.transform(collection),dtype=float),expected,rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(head.multi_encoder.transform(fit),expected,rtol=1e-10,atol=1e-8)
        assert head.diagnostics['multi_featuredim']==8064
        assert all(v.shape==(3,) and np.isfinite(v).all() for v in scores.values())
    scores=MultiRocketHeads(np.zeros((4,128))).score(np.zeros((3,128)))
    assert set(scores)==set(MultiRocketHeads.names) and all(v.shape==(3,) and np.isfinite(v).all() for v in scores.values())
    const=np.ones((5,16));const[0]=0
    out=MultiRocketHeads(const).score(np.full((2,16),-3.0))
    assert all(v.shape==(2,) and np.isfinite(v).all() for v in out.values())
