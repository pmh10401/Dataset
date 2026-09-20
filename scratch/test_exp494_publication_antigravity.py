"""Check authoritative publication and retry after a pre-manifest interruption."""
import tempfile
from pathlib import Path
from experiments.exp494_all_datasets_low_fpr_antigravity import finalize_recovered_antigravity as f

def test_atomic_annotated_publication_and_retry():
    original_p,original_s=f.P,f.SUPPLEMENT
    with tempfile.TemporaryDirectory(prefix='exp494_publication_antigravity_') as directory:
        try:
            f.P=Path(directory);f.SUPPLEMENT=f.P/'supplement_antigravity.json';f.SUPPLEMENT.write_text('{}')
            recovery={'binding':{'rule':'exact zero only'}}
            # A crash may leave the staged result, never a bare authoritative result.
            stage=f.P/'evaluation_staging_antigravity';stage.mkdir();(stage/'results_antigravity.json').write_text('{}')
            assert not (f.P/'results_antigravity.json').exists()
            f.publish({'state':'complete'},recovery,'test-code-hash')
            first=(f.P/'results_antigravity.json').read_bytes()
            assert f.read(f.P/'results_antigravity.json')['preprocessing_supplement']['constant_train_rows']==1
            f.publish({'state':'complete'},recovery,'test-code-hash')
            assert (f.P/'results_antigravity.json').read_bytes()==first
        finally:f.P,f.SUPPLEMENT=original_p,original_s

def test_resume_rejects_changed_code_before_reading_labels():
    try:f.validate_resume({'finalizer_sha256':'old-code'},'binding','new-code')
    except AssertionError:return
    raise AssertionError('changed code was accepted')

if __name__=='__main__':
    test_atomic_annotated_publication_and_retry()
    test_resume_rejects_changed_code_before_reading_labels()
