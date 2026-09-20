"""One frozen A/B split; A fits geometry, unseen B fits ranking weights."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
import numpy as np
from scipy.spatial.distance import cdist
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED
from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import perturbations, KINDS
from experiments.exp502_amplitude_envelope_antigravity.amplitude_antigravity import AmplitudeHeads, amplitude_views
from experiments.exp508_tail_rank_antigravity.ranking_antigravity import distance_contributions
from experiments.exp508_tail_rank_antigravity.optimizer_antigravity import loss_gradient
from experiments.exp509_rank_signal_diagnosis_antigravity.diagnostic_antigravity import solve, certificate, OPTIONS

NAMES=('MiniRocket','oas_distance','square_joint','a_unweighted','separated_rank')


def split_roles(size: int) -> tuple[np.ndarray,np.ndarray]:
    if type(size) is not int or size<2:
        raise ValueError('at least two normal reference rows required')
    if size<32:
        return np.arange(size),np.array([],dtype=int)
    order=np.random.default_rng(SEED+2510).permutation(size);cut=size//4
    return np.sort(order[cut:]),np.sort(order[:cut])


def ranking_parents(indices: np.ndarray) -> np.ndarray:
    b=np.asarray(indices)
    if b.ndim!=1 or not np.issubdtype(b.dtype,np.integer) or np.any(b<0) or not np.array_equal(b,np.unique(b)):
        raise ValueError('sorted unique nonnegative B indices required')
    if len(b)<=128:
        return b.copy()
    return np.sort(np.random.default_rng(SEED+1510).choice(b,128,replace=False))


class SeparatedRankHeads:
    names=NAMES

    def __init__(self,rows):
        x=matrix(rows);self.length=x.shape[1]
        self.a_indices,self.b_indices=split_roles(len(x));self.parents=ranking_parents(self.b_indices)
        # The full-F control never supplies fitted state to the separated candidate.
        self.full=AmplitudeHeads(x)
        self.fallback=len(x)<32
        self.base=self.full if self.fallback else AmplitudeHeads(x[self.a_indices])
        self.scaler,self.geometry,self.reference=self.base.models['square_joint']
        self.weights=np.ones(self.reference.shape[1]);self.scale=1.;self.audit={}
        self.diagnostics=dict(fallback=self.fallback,optimizer_trained=False,n_fit=len(x),
                              n_a=len(self.a_indices),n_b=len(self.b_indices),n_parents=len(self.parents))
        if self.fallback:
            return
        distances=cdist(self.reference,self.reference);np.fill_diagonal(distances,np.inf)
        self.scale=float(np.median(np.sort(distances,axis=1)[:,:min(3,len(self.reference)-1)].mean(axis=1))+1e-9)
        normal=self.contributions(x[self.parents])
        proxies=perturbations(x[self.parents],SEED+1508)
        anomalous=np.vstack([self.contributions(proxies[k]) for k in KINDS])
        self.weights,solver=solve(normal,anomalous)
        initial_loss,initial_gradient=loss_gradient(np.ones_like(self.weights),normal,anomalous)
        final_loss,gradient=loss_gradient(self.weights,normal,anomalous)
        self.audit=dict(weights=self.weights.copy(),gradient=gradient,initial_gradient=initial_gradient,
                        normal_scores=normal@self.weights,anomaly_scores=anomalous@self.weights,
                        initial_normal_scores=normal.sum(axis=1),initial_anomaly_scores=anomalous.sum(axis=1))
        self.diagnostics.update(optimizer_trained=True,scale=self.scale,initial_objective=initial_loss,
                                final_objective=final_loss,solver=solver,**certificate(self.weights,gradient))

    def contributions(self,rows):
        x=matrix(rows)
        if x.shape[1]!=self.length:
            raise ValueError('query length differs from fitted input')
        raw=self.base.encoders[0].transform(x)
        square=self.base.encoders[1].transform(amplitude_views(x)[0])
        z=self.geometry.transform(self.scaler.transform(np.column_stack((raw,square))))
        # B and future queries are absent from A, so no parent reference is removed.
        return distance_contributions(z,self.reference)/self.scale

    def score(self,rows):
        x=matrix(rows)
        if x.shape[1]!=self.length:
            raise ValueError('query length differs from fitted input')
        old=self.full.score(x);result={name:old[name] for name in NAMES[:3]}
        if self.fallback:
            result.update(a_unweighted=old['square_joint'].copy(),separated_rank=old['square_joint'].copy())
        else:
            g=self.contributions(x)
            result.update(a_unweighted=g.sum(axis=1),separated_rank=g@self.weights)
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('nonfinite role-separated score')
        return result
