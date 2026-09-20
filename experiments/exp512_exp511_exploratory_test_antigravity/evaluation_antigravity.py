"""Retrospective evaluation, allowed only after every score and metadata hash freezes."""
import csv
import json
import numpy as np
from experiments.exp494_all_datasets_low_fpr_antigravity.evaluation_antigravity import metric,summarize
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity
from .scoring_antigravity import P,OUT,ROOT,BASE,EXP502,DB,SEED,NAMES,read,digest,canonical,check_hashes,checked,now,readonly_sqlite_antigravity,write_json_atomic_antigravity
from .scoring_antigravity import database_snapshot,assert_unbound_wal_antigravity

def write_family_csv(comparisons,target):
    if target.exists():raise RuntimeError('Preserve completed family comparison CSV')
    temporary=target.with_name(target.stem+'_temporary_antigravity.csv')
    with temporary.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['source_family','comparison','budget','delta_tpr','delta_percentage_points','status'])
        for control,item in comparisons.items():
            for family,values in item['family_deltas'].items():
                for budget,delta in zip(('FP0','FPR ≤0.1%','FPR ≤0.5%','FPR ≤1%'),values):
                    writer.writerow([family,control,budget,delta,delta*100,'improved' if delta>1e-12 else 'worsened' if delta < -1e-12 else 'unchanged'])
    temporary.replace(target)

def analyze(records):
    summaries={n:summarize([dict(source_family=r['source_family'],metrics=r['metrics'][n]) for r in records]) for n in NAMES}
    groups=sorted({r['source_family'] for r in records});family=np.asarray([r['source_family'] for r in records])
    dataset={n:np.array([r['metrics'][n]['tpr'] for r in records]) for n in NAMES}
    grouped={n:np.array([dataset[n][family==g].mean(axis=0) for g in groups]) for n in NAMES}
    draws=np.random.default_rng(SEED).integers(0,len(groups),size=(2000,len(groups)))
    cand=grouped['crossfit_rank'];cboot=cand[draws].mean(axis=1);comparisons={}
    for control in ('MiniRocket','square_joint','crossfit_unweighted'):
        base=grouped[control];bboot=base[draws].mean(axis=1)
        delta=cand-base;dd=dataset['crossfit_rank']-dataset[control]
        comparisons[control]=dict(absolute_delta=(cand.mean(axis=0)-base.mean(axis=0)).tolist(),
            relative_delta=(cand.mean(axis=0)/base.mean(axis=0)-1).tolist(),
            absolute_bootstrap95=np.quantile(cboot-bboot,[.025,.975],axis=0).T.tolist(),
            relative_bootstrap95=np.quantile(cboot/bboot-1,[.025,.975],axis=0).T.tolist(),
            family_improved=(delta>1e-12).sum(axis=0).tolist(),family_worsened=(delta < -1e-12).sum(axis=0).tolist(),
            family_unchanged=(np.abs(delta)<=1e-12).sum(axis=0).tolist(),
            dataset_improved=(dd>1e-12).sum(axis=0).tolist(),dataset_worsened=(dd < -1e-12).sum(axis=0).tolist(),
            dataset_unchanged=(np.abs(dd)<=1e-12).sum(axis=0).tolist(),
            family_deltas={g:delta[i].tolist() for i,g in enumerate(groups)})
    return summaries,comparisons

def main():
    assert_unbound_wal_antigravity(DB)
    target=P/'results_antigravity.json'
    if target.exists():raise RuntimeError('Completed results must be verified, not overwritten')
    manifest=read(P/'manifest_antigravity.json');binding=manifest['binding'];boundary=read(P/'score_freeze_antigravity.json')
    assert manifest['state']=='scores_frozen' and not boundary['test_labels_read']
    assert canonical(binding)==manifest['binding_sha256']==boundary['binding_sha256']
    assert digest(P/'score_freeze_antigravity.json')==manifest['boundary_sha256']
    assert digest(P/'preflight_antigravity.json')==binding['preflight_sha256']
    check_hashes({**binding['code'],**read(P/'preflight_antigravity.json')['artifacts']})
    assert digest(DB)==binding['database_sha256']
    rows=binding['population'];assert len(rows)==1117
    assert set(boundary['scores'])==set(boundary['metadata'])=={r['dataset'] for r in rows}
    metas={r['dataset']:checked(r['dataset'],manifest['binding_sha256']) for r in rows}
    for name,meta in metas.items():
        assert meta['score_sha256']==boundary['scores'][name]
        assert digest(OUT/f'{name}_meta_antigravity.json')==boundary['metadata'][name]
    # The next operation is the first TEST-label query; scores cannot change after this marker.
    write_json_atomic_antigravity(P/'labels_opened_antigravity.json',dict(utc=now(),boundary_sha256=manifest['boundary_sha256'],binding_sha256=manifest['binding_sha256']))
    manifest.update(state='evaluating',test_labels_read=True);write_json_atomic_antigravity(P/'manifest_antigravity.json',manifest)
    records=[]
    with database_snapshot(DB) as con:
        for row in rows:
            name=row['dataset'];labels=load_retained_test_labels_antigravity(con,name,tuple(metas[name]['test_indices']))
            with np.load(OUT/f'{name}_scores_antigravity.npz',allow_pickle=False) as z:
                values={n:metric(labels,z[n]) for n in NAMES}
            records.append(dict(dataset=name,source_family=row['source_family'],metrics=values))
    summaries,comparisons=analyze(records)
    old502=read(EXP502/'full_results_antigravity.json')
    assert digest(EXP502/'full_results_antigravity.json')==binding['source502_result_sha256']
    for name,key in (('MiniRocket','baseline'),('square_joint','candidate')):
        for field in ('family_macro_tpr','dataset_macro_tpr','micro_tpr','total_zf','family_macro_auc','dataset_macro_auc'):
            np.testing.assert_allclose(summaries[name][field],old502['summaries'][key][field],rtol=0,atol=1e-12)
    b=np.array(summaries['MiniRocket']['family_macro_tpr']);c=np.array(summaries['crossfit_rank']['family_macro_tpr'])
    result=dict(state='complete_exploratory_test',evaluation_model='unchanged_exp511',candidate='crossfit_rank',
                binding_sha256=manifest['binding_sha256'],boundary_sha256=manifest['boundary_sha256'],
                raw_datasets=1119,evaluated_datasets=1117,families=128,summaries=summaries,comparisons=comparisons,records=records,
                observed_target_met=bool(c[1]>=0.6004757235442457 and c[0]>=b[0]),
                train_gate_passed=False,model_promotion=False,test_labels_read=True,
                interpretation='User-authorized single fixed exploratory evaluation after TRAIN rejection; previously exposed development TEST, no retuning or operational FPR guarantee.')
    for name in metas:
        assert digest(OUT/f'{name}_scores_antigravity.npz')==boundary['scores'][name]
        assert digest(OUT/f'{name}_meta_antigravity.json')==boundary['metadata'][name]
    check_hashes(binding['code']);assert_unbound_wal_antigravity(DB);assert digest(DB)==binding['database_sha256']
    write_family_csv(comparisons,P/'family_comparison_antigravity.csv')
    result['family_csv_sha256']=digest(P/'family_comparison_antigravity.csv')
    write_json_atomic_antigravity(target,result)
    manifest.update(state=result['state'],result_sha256=digest(target),ended_utc=now())
    write_json_atomic_antigravity(P/'manifest_antigravity.json',manifest)
    print(json.dumps(dict(state=result['state'],summaries={n:summaries[n]['family_macro_tpr'] for n in ('MiniRocket','square_joint','crossfit_rank')},observed_target_met=result['observed_target_met'])),flush=True)

if __name__=='__main__':main()
