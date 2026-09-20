import numpy as np
import pytest
from .kernel_support_antigravity import log_support

def test_log_support_preserves_far_order_and_normalizes_positive_weights():
    scores=log_support(np.array([[0.,0.],[10000.,20000.],[40000.,80000.]]),1.,np.array([1.,3.]))
    np.testing.assert_allclose(scores,[0.,10000.-np.log(.25),40000.-np.log(.25)],rtol=0,atol=1e-10)
    assert scores[2]>scores[1] and np.isfinite(scores).all()
    with pytest.raises(ValueError):log_support([[1.,2.]],1.,[-1.,2.])

def test_kernel_score_matches_official_model_without_probability_underflow():
    from .kernel_support_antigravity import KernelHeads
    from scipy.spatial.distance import cdist
    from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import NeighborGeometry
    rng=np.random.default_rng(499);x=rng.normal(size=(32,12));q=rng.normal(size=(4,12));head=KernelHeads(x);scores=head.score(q)
    assert len(head.models)==4 and set(scores)==set(head.names)
    z=head.geometry.transform(head.control.scaler.transform(q));d=cdist(z,head.reference,'sqeuclidean')
    np.testing.assert_allclose(scores['oas_distance'],NeighborGeometry(head.reference).score(z)[1],rtol=1e-12,atol=1e-12)
    for name,(gamma,model) in head.models.items():
        expected=model.score_samples(np.exp(-gamma*d))/model.dual_coef_.sum()
        np.testing.assert_allclose(np.exp(-scores[name]),expected,rtol=1e-10,atol=1e-12)
    far=head.score(np.vstack((np.full(12,1e3),np.full(12,2e3))))
    for name in head.names:assert np.isfinite(far[name]).all() and far[name][1]>far[name][0]

def test_small_or_degenerate_training_has_fixed_kernel_fallback():
    from .kernel_support_antigravity import KernelHeads
    rng=np.random.default_rng(2)
    for x in (rng.normal(size=(2,10)),np.ones((20,10))):
        head=KernelHeads(x);scores=head.score(rng.normal(size=(4,10)));assert set(scores)==set(head.names) and not head.models
        for name in head.names[3:]:np.testing.assert_array_equal(scores[name],scores['oas_distance'])
