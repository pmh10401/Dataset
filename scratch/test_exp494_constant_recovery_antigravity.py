"""Regression: an exact constant row must not invalidate the other rows."""
import numpy as np
from experiments.exp494_all_datasets_low_fpr_antigravity.constant_recovery_antigravity import normalize

def test_constant_rows_are_zero_without_changing_other_rows():
    rows=[np.array([2.,2.,2.]),np.array([-2.,0.,2.])]
    actual=normalize(rows,3)
    np.testing.assert_array_equal(actual[0],np.zeros(3))
    np.testing.assert_allclose(actual[1],[-np.sqrt(1.5),0,np.sqrt(1.5)],rtol=0,atol=1e-15)
    assert np.isfinite(actual).all()

if __name__=='__main__':test_constant_rows_are_zero_without_changing_other_rows()
