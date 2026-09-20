"""Reporting must distinguish verified artifacts from an unobserved process exit."""
from copy import deepcopy
import pytest
from scratch import report_exp512_antigravity as report


def evidence():
    names={str(i):'score-hash' for i in range(1117)}
    exits={'scoring':{'exit_code':None,'exit_code_observed':False,'boundary_sha256':'frozen'},
           'evaluation':{'exit_code':0},'verification':{'exit_code':0}}
    manifest={'boundary_sha256':'frozen','binding_sha256':'binding'}
    verification={'state':'passed','datasets':1117,'population_and_scores_frozen':True}
    boundary={'datasets':1117,'test_labels_read':False,'binding_sha256':'binding',
              'scores':names,'metadata':names.copy()}
    return exits,manifest,verification,boundary,'frozen'


def validator():
    assert hasattr(report,'validate_execution_evidence'), 'Reporting lacks artifact-backed completion validation'
    return report.validate_execution_evidence


def test_unobserved_scoring_exit_is_not_invented_after_verified_freeze():
    values=evidence();before=deepcopy(values)
    validator()(*values)
    assert values==before and values[0]['scoring']['exit_code'] is None


def test_observed_nonzero_scoring_exit_is_rejected():
    values=evidence();values[0]['scoring']['exit_code']=1
    with pytest.raises((ValueError,AssertionError)):validator()(*values)


def test_unverified_score_population_is_rejected():
    values=evidence();values[2]['population_and_scores_frozen']=False
    with pytest.raises((ValueError,AssertionError)):validator()(*values)


def test_changed_score_boundary_is_rejected():
    values=list(evidence());values[-1]='changed'
    with pytest.raises((ValueError,AssertionError)):validator()(*values)
