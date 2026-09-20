import json
import numpy as np
import pytest
from . import full_run_antigravity as run

def test_resume_rejects_changed_baseline_even_with_matching_file_hash(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();base=tmp_path/'base';source=base/'score_checkpoints_antigravity';source.mkdir(parents=True)
    monkeypatch.setattr(run,'OUT',out);monkeypatch.setattr(run,'BASE',base)
    original=source/'tiny_scores_antigravity.npz';np.savez(original,test__0=np.array([0.,1.,2.]))
    source_meta=source/'tiny_meta_antigravity.json'
    source_meta.write_text(json.dumps(dict(test_indices=[0,1,2],n_train=2,n_test=3,source_family='tiny',score_sha256=run.file_sha256_antigravity(original))))
    altered=out/'tiny_scores_antigravity.npz';np.savez(altered,baseline=np.array([0.,2.,1.]),candidate=np.ones(3))
    (out/'tiny_meta_antigravity.json').write_text(json.dumps(dict(dataset='tiny',binding_sha256='bound',score_sha256=run.file_sha256_antigravity(altered),source_meta_sha256=run.file_sha256_antigravity(source_meta),test_indices=[0,1,2],n_train=2,n_test=3,source_family='tiny')))
    with pytest.raises(AssertionError):run.checked('tiny','bound')
