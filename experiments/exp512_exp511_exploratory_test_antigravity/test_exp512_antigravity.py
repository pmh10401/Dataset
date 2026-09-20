"""Tests for the new full-evaluation boundary and the unchanged scoring adapter."""
from pathlib import Path
import importlib
import json
import numpy as np
import pytest

def module(name):
    assert Path(__file__).with_name(name+'_antigravity.py').exists(),name+' implementation missing'
    return importlib.import_module('experiments.exp512_exp511_exploratory_test_antigravity.'+name+'_antigravity')

def test_adapter_preserves_all_scores_fallback_components_and_roles():
    m=module('scoring')
    from experiments.exp511_crossfit_rank_ensemble_antigravity.ensemble_antigravity import CrossfitRankHeads
    from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import _reference_indices_antigravity
    rng=np.random.default_rng(512)
    for n in (2,32):
        x=rng.normal(size=(n,17)).astype(np.float32);q=rng.normal(size=(9,17)).astype(np.float32)
        arrays,info=m.score_rows(x,q,batch_size=4)
        refs=_reference_indices_antigravity(np.arange(n),seed=20260721+7919+4096)
        h=CrossfitRankHeads(x[refs]);expected=h.score(q)
        for key,val in expected.items():np.testing.assert_allclose(arrays[key],val,rtol=1e-10,atol=1e-8)
        assert info['reference_indices']==refs.tolist() and info['fallback']==(n<32)
        if n<32:
            assert not info['members']
            np.testing.assert_array_equal(arrays['crossfit_rank'],arrays['square_joint'])
        else:
            assert len(info['members'])==4
            np.testing.assert_array_equal(arrays['crossfit_rank'],np.mean([arrays[f'member{j}__rank'] for j in range(4)],axis=0))
            assert all(arrays[f'member{j}__weights'].shape==(8064,) for j in range(4))
    with pytest.raises(ValueError):m.score_rows(x,q,batch_size=0)

def test_hash_boundary_refuses_tampered_artifact(tmp_path):
    m=module('scoring');p=tmp_path/'source_antigravity.json';p.write_text('{"a":1}')
    expected={p.name:m.digest(p)};m.check_hashes(expected,root=tmp_path)
    p.write_text('{"a":2}')
    with pytest.raises((AssertionError,ValueError)):m.check_hashes(expected,root=tmp_path)

def test_equal_family_weighting_bootstrap_and_regression_counts():
    m=module('evaluation');records=[]
    for family,b,c in [('a',.1,.2),('a',.1,.2),('b',.9,.8)]:
        metrics={n:{'tpr':[c if n=='crossfit_rank' else b]*4,'tp':[int((c if n=='crossfit_rank' else b)*10)]*4,'fp':[0]*4,'zf':int((c if n=='crossfit_rank' else b)*10),'n_normal':100,'n_anomaly':10,'auc':.8} for n in m.NAMES}
        records.append(dict(dataset=str(len(records)),source_family=family,metrics=metrics))
    summary,comparison=m.analyze(records)
    np.testing.assert_allclose(summary['crossfit_rank']['family_macro_tpr'],[.5]*4,atol=1e-14)
    item=comparison['square_joint']
    assert item['family_improved']==[1]*4 and item['family_worsened']==[1]*4
    assert item['dataset_improved']==[2]*4 and item['dataset_worsened']==[1]*4
    np.testing.assert_allclose(item['absolute_delta'],[0]*4,atol=1e-14)
    assert m.analyze(records)[1]==comparison

def test_retrospective_metric_preserves_ties_and_zero_fp():
    m=module('evaluation');y=np.array([0,0,1,1]);scores=np.array([1.,2.,2.,3.])
    r=m.metric(y,scores)
    assert r['fp']==[0]*4 and r['tp']==[1]*4 and r['tpr']==[.5]*4

def test_no_rescoring_after_label_access_even_if_freeze_missing(tmp_path):
    m=module('scoring');assert hasattr(m,'assert_scoring_allowed')
    m.assert_scoring_allowed(tmp_path)
    marker=tmp_path/'labels_opened_antigravity.json';marker.write_text('{}')
    with pytest.raises(RuntimeError):m.assert_scoring_allowed(tmp_path)
    marker.unlink()
    for state in ('evaluating','complete_exploratory_test'):
        (tmp_path/'manifest_antigravity.json').write_text(json.dumps({'state':state,'test_labels_read':True}))
        with pytest.raises(RuntimeError):m.assert_scoring_allowed(tmp_path)

def test_wal_guard_rejects_unbound_changes_before_opening_database(tmp_path):
    import sqlite3
    m=module('scoring');assert hasattr(m,'database_snapshot')
    db=tmp_path/'db_antigravity.sqlite3';con=sqlite3.connect(db);con.execute('create table probe (x int)');con.close()
    with m.database_snapshot(db) as reader:assert reader.execute('select count(*) from probe').fetchone()==(0,)
    Path(str(db)+'-wal').write_bytes(b'unbound')
    with pytest.raises(RuntimeError,match='WAL'):
        with m.database_snapshot(db):pass

def test_family_csv_includes_every_budget_and_unchanged_rows(tmp_path):
    import csv
    m=module('evaluation');assert hasattr(m,'write_family_csv')
    p=tmp_path/'family_antigravity.csv'
    m.write_family_csv({'square_joint':{'family_deltas':{'a':[.1,0.,-.2,.3]}}},p)
    with p.open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==4 and [r['status'] for r in rows]==['improved','unchanged','worsened','improved']
    assert [float(r['delta_tpr']) for r in rows]==[.1,0.,-.2,.3]
