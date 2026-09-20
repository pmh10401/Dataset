"""Independent rank-sum AUC and strict-ZF audit of completed frozen scores."""
from pathlib import Path
import hashlib,json,sqlite3
from contextlib import closing
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'experiments/exp492_stable_normalization_antigravity'

def main():
    result=json.loads((PACKAGE/'results_antigravity.json').read_text())
    manifest=json.loads((PACKAGE/'run_manifest_antigravity.json').read_text())
    assert manifest['state']=='complete' and result['family_count']==125 and not manifest['errors']
    assert manifest['labels_opened'] and manifest['all_scores_verified_at']
    assert hashlib.sha256((PACKAGE/'results_antigravity.json').read_bytes()).hexdigest()==manifest['result_sha256']
    max_error=0.;checked=0;summary={};seen=set()
    for record in result['records']:
        name=record['dataset']; assert name not in seen;seen.add(name)
        prefix=PACKAGE/'score_checkpoints_antigravity'/name
        meta=json.loads(Path(str(prefix)+'_meta_antigravity.json').read_text())
        npz=Path(str(prefix)+'_scores_antigravity.npz')
        assert hashlib.sha256(npz.read_bytes()).hexdigest()==record['score_sha256']==meta['score_sha256']
        assert meta['binding']==manifest['binding'] and meta['dataset']==name
        uri=Path('/Users/mac/Documents/Dataset/univariate_ts.db').as_uri()+'?mode=ro'
        with closing(sqlite3.connect(uri,uri=True)) as connection:
            raw=dict(connection.execute("SELECT i.instance_index,i.label FROM instances i JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index",(name,)))
        labels=np.array([int(raw[i]) for i in meta['test_indices']]); assert set(labels)=={0,1}
        with np.load(npz,allow_pickle=False) as arrays:
            for method,branches in record['metrics'].items():
                for branch,expected in branches.items():
                    scores=arrays[f'{method}__{branch}__test_scores']; assert np.isfinite(scores).all() and len(scores)==len(labels)
                    n1=int(labels.sum());n0=len(labels)-n1
                    auc=float((rankdata(scores,method='average')[labels==1].sum()-n1*(n1+1)/2)/(n1*n0))
                    zf=int((scores[labels==1]>scores[labels==0].max()).sum())
                    error=abs(auc-expected['auc']);assert error<=2e-15 and zf==expected['zf']
                    max_error=max(max_error,error);checked+=1
                    summary.setdefault(method,{}).setdefault(branch,[]).append((auc,zf))
    assert len(seen)==125 and checked==500
    for method,branches in summary.items():
        for branch,metrics in branches.items():
            target=result['summary'][method][branch]
            assert abs(float(np.mean([a for a,z in metrics]))-target['mean_auc'])<2e-15
            assert sum(z for a,z in metrics)==target['total_zf']
    preservation=json.loads((PACKAGE/'preservation_antigravity.json').read_text())
    for path,sha in preservation.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    audit=dict(state='passed',datasets=len(seen),independent_score_evaluations=checked,max_auc_absolute_error=max_error,zf_exact=True,preserved_files=len(preservation),
               result_sha256=manifest['result_sha256'],method='Mann-Whitney rank-sum AUC with average ties; independent strict normal-max ZF')
    (PACKAGE/'verification_antigravity.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
