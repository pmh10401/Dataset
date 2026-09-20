import numpy as np
import pytest
from .exposure_antigravity import bounded_exposure

def test_exposure_is_one_sided_bounded_and_degenerate_fit_adds_nothing():
    np.testing.assert_allclose(bounded_exposure([-2.,0.,2.],0.,2.),[0.,0.,.5],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(bounded_exposure([100.,-100.],0.,0.),[0.,0.])
    assert 0<=bounded_exposure([1e30],0.,1.)[0]<=1
    with pytest.raises(ValueError):bounded_exposure([np.nan],0.,1.)

def test_ridge_uses_equal_normal_synthetic_mass_and_correct_regularization():
    from .exposure_antigravity import ExposureHeads
    rng=np.random.default_rng(497);x=rng.normal(size=(24,10));syn=rng.normal(size=(96,10))+1.
    head=ExposureHeads(x,np.arange(24),syn);assert set(head.regressors)=={1,100}
    design=head.control.scaler.transform(np.vstack((x,syn)));target=np.r_[np.zeros(24),np.ones(96)];weight=np.r_[np.full(24,1/24),np.full(96,1/96)]
    for alpha,(model,_,_) in head.regressors.items():
        residual=design@model.coef_+model.intercept_-target
        np.testing.assert_allclose(design.T@(weight*residual)+alpha*model.coef_,0,atol=1e-10,rtol=0)
        assert abs(weight@residual)<1e-12
    scores=head.score(syn[:4]);assert set(scores)==set(head.names)
    for name in head.names[2:]:assert np.all(scores[name]>=scores['oas_distance']) and np.all(scores[name]-scores['oas_distance']<=.5+1e-12)

def test_tiny_train_uses_anchor_without_exposure():
    from .exposure_antigravity import ExposureHeads
    rng=np.random.default_rng(2);x=rng.normal(size=(2,10));q=rng.normal(size=(4,10))
    head=ExposureHeads(x,np.array([],int),np.empty((0,10)));scores=head.score(q)
    assert set(scores)==set(head.names) and not head.regressors
    for name in head.names[2:]:np.testing.assert_array_equal(scores[name],scores['oas_distance'])
