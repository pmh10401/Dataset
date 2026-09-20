"""Numerical regression checks at the two confirmed normalization boundaries."""
import numpy as np
import torch
from aeon.regression.convolution_based._hydra import _SparseScaler
from sklearn.preprocessing import RobustScaler
from .normalization_antigravity import StableSparseScalerAntigravity, StableRobustScalerAntigravity


def test_sparse_constant_retains_novelty_without_epsilon_amplification():
    reference = torch.tensor([[4., 0., 1.], [4., 0., 4.], [4., 0., 9.], [4., 0., 16.]])
    query = torch.tensor([[8., 1., 25.]])
    legacy = _SparseScaler()
    legacy.fit(reference)
    model = StableSparseScalerAntigravity()
    model.fit(reference)
    result = model.transform(query).numpy()
    assert 0.5 < result[0, 0] < 1.0  # actual novelty sqrt(8)-2; previously 82 million
    np.testing.assert_array_equal(result[:, 1:], legacy.transform(query).numpy()[:, 1:])
    before = model.sigma.clone()
    model.transform(query * 100)
    assert torch.equal(before, model.sigma)


def test_robust_tiny_float32_spread_without_losing_ordinary_columns():
    reference = np.column_stack((np.linspace(0, 1.8e-7, 20, dtype=np.float32), np.arange(20))).astype(float)
    query = np.array([[0.15, 24.]])
    legacy = RobustScaler(quantile_range=(10, 90)).fit(reference)
    model = StableRobustScalerAntigravity(quantile_range=(10, 90)).fit(reference)
    result = model.transform(query)
    assert 0.14 < result[0, 0] < 0.16  # previously greater than one million
    np.testing.assert_array_equal(result[:, 1:], legacy.transform(query)[:, 1:])
    before = model.scale_.copy()
    model.transform(query * 100)
    np.testing.assert_array_equal(before, model.scale_)


def test_guard_boundaries_and_large_scales_never_increase_amplification():
    eps=10*np.finfo(np.float32).eps
    unit=np.linspace(0,1,21)
    reference=np.column_stack((unit*(eps/.8),unit*(eps*1.01/.8),1e8+80*unit,unit*.01,np.zeros(21)))
    legacy=RobustScaler(quantile_range=(10,90)).fit(reference)
    model=StableRobustScalerAntigravity(quantile_range=(10,90)).fit(reference)
    assert model.scale_[0]==1.0
    np.testing.assert_array_equal(model.scale_[1:],legacy.scale_[1:])
    assert model.scale_[2]>1.0  # degeneracy tolerance may be >1 at large magnitudes
    result=model.transform(reference)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, np.vstack([model.transform(row[None,:]) for row in reference]),rtol=0,atol=0)


def test_nonfinite_features_are_rejected():
    import pytest
    with pytest.raises(ValueError):
        StableRobustScalerAntigravity(quantile_range=(10,90)).fit(np.array([[0.,1.],[np.nan,2.]]))
    with pytest.raises(ValueError):
        StableSparseScalerAntigravity().fit(torch.tensor([[0.,1.],[float('inf'),2.]]))


def test_checkpoints_reject_tampering_and_wrong_binding(tmp_path):
    import pytest
    from .run_antigravity import persist_antigravity,load_antigravity,paths_antigravity,METHODS
    pairs={m:{'old':{'train_oof':np.arange(6.),'test_scores':np.arange(3.)},
              'stable':{'train_oof':np.arange(6.),'test_scores':np.arange(3.)},
              'diagnostics':[], 'feature_dimension':3,'representation_fits':6} for m in METHODS}
    extra=dict(binding={'fixture':1},source_family='fixture',n_train=6,n_test=3,test_indices=[0,1,2])
    persist_antigravity(tmp_path,'fixture',pairs,extra)
    assert load_antigravity(tmp_path,'fixture',extra['binding']) is not None
    with pytest.raises(ValueError,match='binding'): load_antigravity(tmp_path,'fixture',{'fixture':2})
    npz,_,_=paths_antigravity(tmp_path,'fixture')
    npz.write_bytes(npz.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'): load_antigravity(tmp_path,'fixture',extra['binding'])


def test_checkpoint_order_must_match_value_only_preprocessing():
    import pytest
    from types import SimpleNamespace
    from .run_antigravity import validate_alignment_antigravity
    prepared=SimpleNamespace(train_rows=np.zeros((6,4)),test_rows=np.zeros((3,4)),target_length=4,test_indices=(2,4,6))
    record=dict(n_train=6,n_test=3,target_length=4,test_indices=[2,4,6])
    validate_alignment_antigravity(record,prepared)
    record['test_indices']=[6,4,2]
    with pytest.raises(ValueError,match='alignment'): validate_alignment_antigravity(record,prepared)
