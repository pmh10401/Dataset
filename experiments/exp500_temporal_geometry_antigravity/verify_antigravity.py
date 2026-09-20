"""Independent persisted-record audit; no raw TEST access."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    result=read(P/'pilot_results_antigravity.json');manifest=read(P/'pilot_manifest_antigravity.json')
    assert sha(P/'pilot_results_antigravity.json')==manifest['result_sha256']
    assert sha(P/'protocol_antigravity.html')==manifest['protocol_sha256']
    for path,h in manifest['code'].items():assert sha(ROOT/path)==h,path
    source=ROOT/'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json'
    assert sha(source)==manifest['source_population_sha256']
    assert read(source)['population']==manifest['population']
    prior=read(ROOT/'experiments/exp499_normal_support_antigravity/pilot_manifest_antigravity.json')
    assert prior['population']==manifest['population']
    assert prior['source_population_sha256']==manifest['source_population_sha256']
    rows=result['records'];names=('MiniRocket','oas_distance','dtw_distance','dtw_nearest','joint_geometric');kinds=('local_level','local_scale','permuted_quarters','phase_randomized')
    assert len(rows)==len({r['source_family'] for r in rows})==50
    assert {r['dataset'] for r in rows}=={r['dataset'] for r in manifest['population']}
    old={r['dataset']:r for r in read(ROOT/'experiments/exp495_normal_geometry_antigravity/pilot_results_antigravity.json')['records']}
    for row in rows:
        assert len(row['folds'])==len(row['normal_tail_by_fold'])==5
        for i,fold in enumerate(row['folds']):
            for kind in kinds:
                for name in names:assert np.isfinite(fold[kind][name]) and 0<=fold[kind][name]<=1
                for name in names[:2]:assert fold[kind][name]==old[row['dataset']]['folds'][i][kind][name]
    summary={n:{k:sum(sum(f[k][n] for f in row['folds'])/5 for row in rows)/50 for k in kinds} for n in names}
    tails={n:float(np.median([sum(t[n] for t in row['normal_tail_by_fold'])/5 for row in rows])) for n in names}
    for n in names:
        for k in kinds:np.testing.assert_allclose(summary[n][k],result['summary'][n][k],rtol=0,atol=1e-14)
        np.testing.assert_allclose(tails[n],result['normal_tail'][n],rtol=0,atol=1e-14)
    eligible=list(names[:2])+[n for n in names[2:] if tails[n]<=3*max(tails['MiniRocket'],1e-9)]
    criterion=lambda n:(min(summary[n].values()),sum(summary[n].values())/4,-names.index(n))
    anchor=max(names[:2],key=criterion);winner=max(eligible,key=criterion);selected=winner
    if winner not in names[:2] and (criterion(winner)[0]<criterion(anchor)[0]+.02 or criterion(winner)[1]<criterion(anchor)[1]):selected=anchor
    assert eligible==result['eligible'] and anchor==result['best_anchor'] and winner==result['unrestricted_winner'] and selected==result['selected']
    assert not result['test_values_read'] and not result['test_labels_read']
    assert not manifest['test_values_read'] and not manifest['test_labels_read']
    new_best=max((n for n in eligible if n not in names[:2]),key=criterion)
    out=dict(state='passed_train_screen',datasets=50,folds=250,selected=selected,unrestricted_winner=winner,best_new_candidate=new_best,new_candidate_worst_proxy=criterion(new_best)[0],anchor_worst_proxy=criterion(anchor)[0],new_candidate_margin=criterion(new_best)[0]-criterion(anchor)[0],required_proxy_margin=.02,eligible=eligible,normal_tail=tails,anchor_fold_recalls_exact=True,source_hashes_passed=True,pilot_sha256=sha(P/'pilot_results_antigravity.json'),full_run_skipped=selected in names[:2],test_labels_read=False)
    (P/'verification_antigravity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
