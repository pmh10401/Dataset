"""Resolve warm-only speed ambiguity on the actual path preserving Exp510 controls."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
from statistics import median
from hashlib import sha256
import json
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity
from experiments.exp509_rank_signal_diagnosis_antigravity.diagnostic_antigravity import solve
from experiments.exp508_tail_rank_antigravity.optimizer_antigravity import _objective
from experiments.exp508_tail_rank_antigravity.acceleration_antigravity import objective_numba,objective_torch
from experiments.exp510_separated_rank_antigravity.acceleration_antigravity import parity
from experiments.exp511_crossfit_rank_ensemble_antigravity import ensemble_antigravity as model

P=Path(__file__).resolve().parents[1]/'experiments/exp511_crossfit_rank_ensemble_antigravity'


def main():
    target=P/'production_backend_assessment_antigravity.json'
    assert not target.exists(), 'Preserve completed qualification'
    rng=np.random.default_rng(511);fit=rng.normal(size=(64,128));query=rng.normal(size=(1024,128))
    fit[0]=0;query[0]=0
    functions=dict(numpy=_objective,numba=objective_numba,torch_cpu=objective_torch)
    original=model.solve;reference=None;records=[]
    # Orders fixed before measuring. First-round and median gains include whole-head cost.
    orders=(('numpy','numba','torch_cpu'),('torch_cpu','numba','numpy'),('numpy','torch_cpu','numba'))
    with threadpool_limits(limits=1):
        try:
            for repeat,order in enumerate(orders):
                for name in order:
                    fn=functions[name];model.solve=lambda n,a:solve(n,a,objective=fn)
                    start=perf_counter();head=model.CrossfitRankHeads(fit);scores=head.score(query);seconds=perf_counter()-start
                    if reference is None:reference=scores
                    check=parity(scores,reference)
                    check['frozen_five_exact']=all(np.array_equal(scores[k],reference[k]) for k in model.NAMES[:5])
                    records.append(dict(repeat=repeat,backend=name,seconds=seconds,**check))
                    print(json.dumps({k:records[-1][k] for k in ('repeat','backend','seconds','parity','frozen_five_exact')}),flush=True)
        finally:model.solve=original
    base=[r['seconds'] for r in records if r['backend']=='numpy'];summary={}
    for name in functions:
        rows=[r for r in records if r['backend']==name]
        summary[name]=dict(median_seconds=median(r['seconds'] for r in rows),median_speedup=median(base)/median(r['seconds'] for r in rows),
                           first_speedup=base[0]/rows[0]['seconds'],parity=all(r['parity'] and r['frozen_five_exact'] for r in rows))
    eligible=[name for name,s in summary.items() if name!='numpy' and s['parity'] and s['median_speedup']>=1.1 and s['first_speedup']>=1.1]
    out=dict(state='complete',success=summary['numpy']['parity'],scope='Actual production candidate path: exact original Exp510 member0 and five controls; accelerated objectives only in members1-3.',
             qualification_rule='All paired score/ranking/candidate checks and exact five controls; both first-round and median whole-stage speedup >=1.1.',
             records=records,summary=summary,eligible_alternatives=eligible,first_benchmark_preserved=True,
             seed=511,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),test_values_read=False,test_labels_read=False)
    write_json_atomic_antigravity(target,out);print(json.dumps(dict(summary=summary,eligible_alternatives=eligible)),flush=True)


if __name__=='__main__':main()
