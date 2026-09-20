"""Focused feature-update and held-family outcome isolation regressions."""
import numpy as np
from experiments.exp490_behavior_descriptors_antigravity.behavior_descriptors_antigravity import assemble_behavior_antigravity, BEHAVIOR_KEYS_ANTIGRAVITY
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import METHODS_ANTIGRAVITY
from .behavior_antigravity import update_descriptors_antigravity


def test_stable_descriptor_update_matches_full_assembly():
    a={m:np.array([0.,2.,1.,3.,5.,4.])+j for j,m in enumerate(METHODS_ANTIGRAVITY)}
    b={m:np.array([0.,1.,2.,3.,4.,5.])+j for j,m in enumerate(METHODS_ANTIGRAVITY)}
    stable_a={m:v.copy() for m,v in a.items()}; stable_b={m:v.copy() for m,v in b.items()}
    stable_a[METHODS_ANTIGRAVITY[2]]=np.array([0.,1.,2.,9.,4.,3.])
    stable_b[METHODS_ANTIGRAVITY[3]]=b[METHODS_ANTIGRAVITY[3]][::-1].copy()
    old,_=assemble_behavior_antigravity(a,b)
    expected,_=assemble_behavior_antigravity(stable_a,stable_b)
    actual=update_descriptors_antigravity(old,a,stable_a,b,stable_b)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(np.asarray(actual)[:6],np.asarray(old)[:6])


def test_family_isolation_and_frozen_routing():
    from .meta_antigravity import predict_antigravity, routes_antigravity
    rng=np.random.default_rng(11); x=rng.normal(size=(8,4)); y=rng.normal(size=(8,6))
    families=['held','held','a','a','b','c','d','e']; x[0,0]=np.nan
    pred,_=predict_antigravity(x,y,families)
    changed=y.copy(); changed[:2]=10000
    altered,_=predict_antigravity(x,changed,families)
    np.testing.assert_array_equal(pred[:2],altered[:2])
    for name,r in routes_antigravity(pred).items():
        np.testing.assert_array_equal(r[:2],routes_antigravity(altered)[name][:2])
    routes=routes_antigravity(np.array([[-1,-2,-3,0,0,0],[2,2,1,-2,3,3.]]))
    assert routes['auc'].tolist()==[0,1]
    assert routes['zf_sign'].tolist()==[0,2]


def test_reject_stale_descriptor_and_missing_dependency():
    import pytest
    a={m:np.arange(8,dtype=float)**(1+i*.2) for i,m in enumerate(METHODS_ANTIGRAVITY)}
    b={m:v[::-1].copy() for m,v in a.items()}; old,_=assemble_behavior_antigravity(a,b)
    stale=list(old);stale[6]+=1
    with pytest.raises(ValueError):update_descriptors_antigravity(stale,a,a,b,b)
    stable={m:v.copy() for m,v in a.items()};stable[METHODS_ANTIGRAVITY[2]][1]+=100
    aa={m:v for m,v in a.items() if m!=METHODS_ANTIGRAVITY[1]}
    ss={m:v for m,v in stable.items() if m!=METHODS_ANTIGRAVITY[1]}
    with pytest.raises(ValueError):update_descriptors_antigravity(old,aa,ss,b,b)
