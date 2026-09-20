"""Energy algebra, unchanged baseline, tiny references and query isolation."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
import numpy as np
from experiments.exp501_energy_lift_antigravity.energy_antigravity import energy_rows,EnergyHeads,SEED
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import PairedHeads

def test_energy_contract():
    rng=np.random.default_rng(501);x=rng.normal(size=(12,64));q=rng.normal(size=(5,64))
    np.testing.assert_array_equal(energy_rows(x),energy_rows(-x))
    np.testing.assert_array_equal(energy_rows(np.ones((2,64))),np.zeros((2,64)))
    np.testing.assert_allclose(energy_rows(x).mean(axis=1),0,atol=1e-7)
    np.testing.assert_allclose(energy_rows(x).std(axis=1),1,atol=1e-7)
    for n in (2,12):
        head=EnergyHeads(x[:n]);scores=head.score(q)
        encoder=make_representation_antigravity('minirocket_local_gap',SEED);old=PairedHeads(encoder.fit_transform(x[:n])).score(encoder.transform(q))
        for name in ('MiniRocket','oas_distance'):np.testing.assert_allclose(scores[name],old[name],rtol=1e-10,atol=1e-8)
        before={n:r.copy() for n,(_,_,r) in head.models.items()};single=head.score(q[:1]);head.score(q*4)
        for name in head.names:np.testing.assert_allclose(single[name],scores[name][:1],rtol=1e-10,atol=1e-8)
        for name,(_,_,ref) in head.models.items():np.testing.assert_array_equal(before[name],ref)
