"""Publish the verified Exp510 TRAIN rejection without changing historical results."""
from pathlib import Path
from hashlib import sha256, file_digest
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
from statistics import median
from math import isclose
import json
import re
import sys

from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'experiments/exp510_separated_rank_antigravity'
DOC = ROOT / 'docs/EXP510_SEPARATED_RANK_REVIEW_antigravity.html'
STATUS = ROOT / 'docs/CURRENT_MODEL_IMPROVEMENT_STATUS_antigravity.html'
NAMES = ('MiniRocket', 'oas_distance', 'square_joint', 'a_unweighted', 'separated_rank')
LABELS = ('MiniRocket', '원시 OAS', 'Exp502 · 전체 정상 기준', 'A 전용 · 비가중', 'Exp510 · A/B 역할 분리 학습')
read = lambda path: json.loads(path.read_text())


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k == 'href')


def validate_links(path):
    parser = Links()
    parser.feed(path.read_text())
    for link in parser.links:
        if not link.startswith(('http:', 'https:', '#')):
            assert (path.parent / link.split('#')[0]).exists(), (str(path), link)
    return len(parser.links)


def main():
    assert not DOC.exists(), 'Do not overwrite a published experiment report'
    result = read(P / 'pilot_results_antigravity.json')
    verification = read(P / 'verification_antigravity.json')
    assert verification['state'] == 'passed'
    assert verification['result_sha256'] == sha256((P / 'pilot_results_antigravity.json').read_bytes()).hexdigest()
    assert result['gates'] == verification['gates']
    assert not result['full_run_required'] and result['selected'] == 'square_joint'
    assert not result['test_values_read'] and not result['test_labels_read']
    metrics = {}
    for name in NAMES:
        metrics[name] = (
            min(result['summary'][name].values()),
            sum(result['summary'][name].values()) / 4,
            sum(result['unseen_summary'][name].values()) / 4,
        )
        for key, value in zip(('worst_standard', 'mean_standard', 'mean_unseen'), metrics[name]):
            assert isclose(value, result[key][name], rel_tol=0, abs_tol=1e-14)
    candidate = metrics['separated_rank']
    baseline = metrics['square_joint']
    matched = metrics['a_unweighted']
    rows = ''.join(
        '<tr><th>' + escape(label) + '</th>'
        + ''.join(f'<td>{value:.4%}</td>' for value in metrics[name])
        + f'<td>{result["normal_tail"][name]:.4f}</td></tr>'
        for name, label in zip(NAMES, LABELS)
    )
    required = (
        max(metrics[n][0] for n in NAMES[:-1]) + .02,
        max(metrics[n][1] for n in NAMES[:-1]),
        max(metrics[n][2] for n in NAMES[:-1]),
        3 * max(result['normal_tail']['MiniRocket'], 1e-9),
    )
    actual = (*candidate, result['normal_tail']['separated_rank'])
    gate_names = ('기존 변형 최저 검출률 +2%p', '기존 4종 평균 비악화', '검증 전용 4종 평균 비악화', '정상 꼬리 상한')
    gate_keys = ('minimum_margin', 'mean_nonregression', 'unseen_nonregression', 'normal_tail')
    gate_rows = ''
    for i, (label, key) in enumerate(zip(gate_names, gate_keys)):
        passed = actual[i] >= required[i] if i < 3 else actual[i] <= required[i]
        assert passed == result['gates'][key]
        observed = f'{actual[i]:.4%}' if i < 3 else f'{actual[i]:.4f}'
        threshold = f'≥ {required[i]:.4%}' if i < 3 else f'≤ {required[i]:.4f}'
        gate_rows += f'<tr><th>{label}</th><td>{observed}</td><td>{threshold}</td><td>{"통과" if passed else "미달"}</td></tr>'
    folds = [fold for row in result['records'] for fold in row['folds']]
    assert len(folds) == 250 and len(result['records']) == 50
    gaps = [f['diagnostics']['objective_gap_upper_bound'] for f in folds]
    unresolved = len(folds) - verification['projected_gradient_passes']
    acceleration = read(P / 'acceleration_results_antigravity.json')
    preflight = read(P / 'preflight_antigravity.json')
    prefix = '../experiments/exp510_separated_rank_antigravity/'
    summary = (
        f'Exp510은 50계열·250분할 정상 TRAIN 평가에서 사전 승격 조건을 충족하지 못했습니다. '
        f'기존 변형 평균은 Exp502 {baseline[1]:.4%} → Exp510 {candidate[1]:.4%}, '
        f'검증 전용 변형 평균은 {baseline[2]:.4%} → {candidate[2]:.4%}입니다. '
        '기존 Exp502를 유지하며 Exp510의 새 TEST 평가나 모델 승격은 수행하지 않았습니다.'
    )
    header = f'<header><p class="meta">2026-09-18 · Exp510 TRAIN 평가·독립 검산 완료</p><h1>역할 분리 학습은 승격 조건을 충족하지 못했습니다</h1><p>{summary}</p><p class="note">합성 TRAIN 검증 결과입니다. 실제 이상 TEST에서의 성능 변화나 운영 오탐률을 의미하지 않습니다.</p></header>'
    style = 'body{font:16px/1.75 system-ui;max-width:1120px;margin:36px auto;padding:0 24px;color:#182b3a}h1,h2{line-height:1.4}section{margin:30px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:12px;border-bottom:1px solid #ced7df;text-align:right}th:first-child{text-align:left}.scroll{overflow-x:auto}.note{background:#fff3db;padding:16px}.meta{color:#526576}pre{white-space:pre-wrap;background:#eff3f6;padding:16px}a{color:#14558b}'
    page = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp510 역할 분리 학습 검증</title><style>{style}</style></head><body>{header}'
    page += '<section><h2>같은 보류 정상·같은 변형의 비교</h2><p>변형별 검출률은 5분할 평균 후 50계열 동일 가중 평균입니다. 최저 검출률은 기존 4종의 계열 평균 중 최솟값입니다. 정상 꼬리는 분할 평균 후 계열 중앙값이며, 정상 상단의 돌출 정도에 대한 상한 조건으로 사용합니다.</p><div class="scroll"><table><thead><tr><th>모델</th><th>기존 변형 최저</th><th>기존 4종 평균</th><th>검증 전용 4종 평균</th><th>정상 꼬리</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
    page += f'<p>Exp502 대비 차이는 최저 {(candidate[0]-baseline[0])*100:+.4f}%p, 기존 평균 {(candidate[1]-baseline[1])*100:+.4f}%p, 검증 전용 평균 {(candidate[2]-baseline[2])*100:+.4f}%p입니다. 같은 A를 쓰는 비가중 대조군 대비 차이는 각각 {(candidate[0]-matched[0])*100:+.4f}%p, {(candidate[1]-matched[1])*100:+.4f}%p, {(candidate[2]-matched[2])*100:+.4f}%p입니다. 이 비교는 참조 감소 효과와 가중치 학습 효과를 구분하는 데 사용합니다.</p></section>'
    page += '<section><h2>이번 결과가 보여 주는 것</h2><p>같은 A 참조를 사용하는 비가중 대조군과 비교하면 가중치 학습 후 세 검출 지표가 모두 높아졌습니다. 반면 전체 정상 F를 사용하는 기존 Exp502와 비교하면 기존 4종 평균의 이득은 작고, 학습에 넣지 않은 변형의 평균은 낮아졌습니다. 이 고정 구성에서 학습의 이점은 관측됐지만 기존 모델을 교체할 조건은 충족하지 못했습니다.</p></section>'
    page += '<section><h2>사전에 고정한 네 조건</h2><div class="scroll"><table><thead><tr><th>조건</th><th>후보 관측값</th><th>필요값</th><th>결과</th></tr></thead><tbody>' + gate_rows + '</tbody></table></div><p>최저 검출률과 두 평균은 네 대조군을 모두 기준으로 삼습니다. 조건을 완화하거나 다른 분할·최적화 설정을 재탐색하지 않았습니다. 전체 1,117개 데이터셋 평가는 실행하지 않았습니다.</p></section>'
    page += f'<section><h2>학습·검증·가속</h2><p>외부 적합 정상 F의 약 75%인 A로 표현·척도·OAS·참조를 만들고, 나머지 B로 가중치를 학습했습니다. 외부 보류 C와 그 변형은 검증에만 사용했습니다. 같은 F 전체를 쓰는 대조군은 별도 객체입니다.</p><p>회귀 테스트 {preflight["tests_passed"]}개, {verification["folds"]}분할의 {verification["arrays_verified"]:,}개 배열 검산, 기존 세 대조군의 분할별 검출률 재현을 확인했습니다. 최적화기 성공은 {verification["solver_successes"]}/250, 제약 반영 기울기 기준 통과는 {verification["projected_gradient_passes"]}/250입니다. 기울기 기준 미확정 {unresolved}분할도 사전 규칙대로 유한한 최종 가중치를 사용했습니다. 목적함수 오차 상한은 중앙값 {median(gaps):.6g}, 최댓값 {max(gaps):.6g}입니다.</p>'
    page += '<p>NumPy·Numba·Torch CPU 경로의 점수·순위·검출 대상은 일치했습니다. 준비 후 전체 단계의 속도 배수는 ' + ', '.join(f'{r["backend"]} {r["speedup"]:.4f}배' for r in acceleration['backends']) + '였습니다. 1.1배 기준을 충족한 대안이 없어 NumPy/SciPy float64를 유지했습니다. 실제 MLX GPU와 MPS의 float64 연산은 지원되지 않았습니다.</p></section>'
    page += '<section><h2>해석 범위</h2><p>이 결과는 합성 TRAIN 검증에서 이 고정 후보가 승격 기준을 충족하는지를 평가합니다. 역할 분리 일반의 무효나 실제 이상에서의 성능 하락을 입증하지 않습니다. 같은 내용을 가진 서로 다른 행은 A/B 양쪽에 존재할 수 있으며 새로운 파형에 대한 검증이라고 주장하지 않습니다. 기존 전처리의 목표 길이는 C를 포함한 전체 정상 TRAIN에서 정한 값을 상속했습니다.</p><p>독립 검산은 저장된 점수로 손실·검출률·꼬리·조건을, 저장된 기울기로 제약 반영 조건을 다시 계산했습니다. 원본 기여 행렬부터 전체 기울기와 표현을 독립 재생성한 검증은 아닙니다. 반복 노출된 개발 벤치마크이며 미사용 holdout이나 운영 FPR 보장으로 해석할 수 없습니다.</p></section>'
    page += f'<section><h2>근거와 재검산</h2><p><a href="{prefix}protocol_antigravity.html">고정 프로토콜</a> · <a href="{prefix}pilot_results_antigravity.json">전체 TRAIN 집계</a> · <a href="{prefix}verification_antigravity.json">독립 검산</a> · <a href="{prefix}preflight_antigravity.json">실행 전 동결</a> · <a href="{prefix}review_resolution_antigravity.json">독립 검토와 판단</a> · <a href="{prefix}code_review_antigravity.json">Grok 코드 검토</a> · <a href="{prefix}plan_antigravity.html">실행 계획</a></p><pre>NUMBA_CACHE_DIR=/private/tmp/exp510_numba_cache_antigravity .venv/bin/python -m experiments.exp510_separated_rank_antigravity.verify_antigravity</pre><p>완료된 파일럿을 덮어쓰지 않고 저장된 근거를 재검산합니다. <a href="EXP502_FULL_COMPARISON_antigravity.html">기존 Exp502 전체 평가</a> · <a href="CURRENT_MODEL_IMPROVEMENT_STATUS_antigravity.html">현재 연구 상태</a></p></section></body></html>'
    DOC.write_text(page + '\n')
    old = STATUS.read_text()
    assert 'id="exp510"' not in old
    changed, count = re.subn(r'<header>.*?</header>', header, old, count=1, flags=re.S)
    assert count == 1
    section = '<section id="exp510"><h2>Exp510: 정상 역할 분리 학습</h2><p>' + summary + '</p><p><a href="EXP510_SEPARATED_RANK_REVIEW_antigravity.html">다섯 모델·네 승격 조건·독립 검산 보고서</a></p></section>'
    STATUS.write_text(changed.replace('</header>', '</header>' + section, 1))
    counts = {str(path.relative_to(ROOT)): validate_links(path) for path in (DOC, STATUS)}
    print(json.dumps({'report': str(DOC), 'links_verified': counts}, ensure_ascii=False))


def close():
    """Seal existing verified results and report without rerunning any experiment."""
    target = P / 'closure_antigravity.json'
    assert not target.exists(), 'Do not overwrite an experiment closure'
    def digest(path):
        with path.open('rb') as stream:
            return file_digest(stream, 'sha256').hexdigest()
    result = read(P / 'pilot_results_antigravity.json')
    manifest = read(P / 'pilot_manifest_antigravity.json')
    verification = read(P / 'verification_antigravity.json')
    preflight = read(P / 'preflight_antigravity.json')
    exits = read(P / 'execution_exit_antigravity.json')
    assert manifest['state'] == result['state'] == 'complete_train_only_pilot'
    assert verification['state'] == preflight['state'] == 'passed'
    assert exits['pilot']['exit_code'] == exits['verification']['exit_code'] == 0
    assert not exits['full_test_evaluation_executed']
    assert result['gates'] == verification['gates'] and not result['full_run_required']
    assert result['selected'] == verification['selected'] == 'square_joint'
    assert digest(P / 'pilot_results_antigravity.json') == manifest['result_sha256'] == verification['result_sha256']
    assert digest(P / 'preflight_antigravity.json') == manifest['binding']['preflight_sha256']
    for relative, expected in {**manifest['binding']['code'], **preflight['artifacts']}.items():
        assert digest(ROOT / relative) == expected, relative
    for relative, expected in preflight['historical_artifacts'].items():
        assert digest(ROOT / relative) == expected, relative
    for doc in (result, manifest, verification):
        assert not doc['test_values_read'] and not doc['test_labels_read']
    assert not list(P.glob('full_*')), 'Unexpected full-evaluation artifact'
    links = {str(path.relative_to(ROOT)): validate_links(path) for path in (DOC, STATUS)}
    assert 'id="exp510"' in STATUS.read_text()
    plan = P / 'plan_antigravity.html'
    text = plan.read_text()
    for step in ('pilot', 'decision', 'report'):
        text = text.replace(f'<label id="{step}"><input type="checkbox" disabled>',
                            f'<label id="{step}"><input type="checkbox" checked disabled>')
    assert text.count('checked disabled') == 6
    plan.write_text(text)
    files = [P / name for name in ('pilot_results_antigravity.json', 'pilot_manifest_antigravity.json',
             'verification_antigravity.json', 'preflight_antigravity.json', 'execution_exit_antigravity.json',
             'plan_antigravity.html')]
    files += [DOC, STATUS, Path(__file__).resolve()]
    closure = dict(state='complete_verified_train_rejection', utc=datetime.now(timezone.utc).isoformat(),
                   datasets=50, families=50, folds=250, arrays_verified=verification['arrays_verified'],
                   tests_passed=preflight['tests_passed'], gates=result['gates'], selected='square_joint',
                   source_code_changed_after_run=False, historical_maps_unchanged=True,
                   full_test_evaluation_executed=False, test_values_read=False, test_labels_read=False,
                   model_promotion=False, execution_exit_codes=exits,
                   projected_gradient_passes=verification['projected_gradient_passes'],
                   solver_successes=verification['solver_successes'],
                   report=str(DOC.relative_to(ROOT)), report_links_verified=links,
                   browser_preview_verified=False,
                   browser_preview_limitation='Local file URL blocked by browser URL policy; no workaround attempted.',
                   artifacts={str(path.relative_to(ROOT)): digest(path) for path in files},
                   limitations='Synthetic normal-TRAIN screening; stored-vector audit does not regenerate full gradients or geometry.')
    write_json_atomic_antigravity(target, closure)
    print(json.dumps({k: closure[k] for k in ('state', 'tests_passed', 'arrays_verified', 'gates',
                                           'full_test_evaluation_executed', 'report')}, ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['--close']:
        close()
    else:
        assert not sys.argv[1:], 'Only --close is supported'
        main()
