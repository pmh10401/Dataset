"""Read completed Exp492 artifacts; summarize numerical changes without refitting."""
from pathlib import Path
import gzip, hashlib, html, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'experiments/exp492_stable_normalization_antigravity'
METHODS=('hydra_local_gap','hydra_multirocket_local_gap')
NAMES={'hydra_local_gap':'Hydra + LocalGap','hydra_multirocket_local_gap':'Hydra + MultiRocket + LocalGap'}

def main():
    result=json.loads((PACKAGE/'results_antigravity.json').read_text())
    manifest=json.loads((PACKAGE/'run_manifest_antigravity.json').read_text())
    assert result['state']=='complete_development_evaluation' and result['family_count']==125 and result['failures']==0
    assert manifest['state']=='complete' and manifest['labels_opened'] is True
    assert hashlib.sha256((PACKAGE/'results_antigravity.json').read_bytes()).hexdigest()==manifest['result_sha256']
    audit=json.loads((PACKAGE/'verification_antigravity.json').read_text())
    assert audit['state']=='passed' and audit['result_sha256']==manifest['result_sha256']
    numerical={m:dict(touched_families=[],sparse_changed_column_fits=0,robust_changed_column_fits=0,
                      old_feature_max_abs=0,stable_feature_max_abs=0,old_scaled_max_abs=0,stable_scaled_max_abs=0) for m in METHODS}
    for record in result['records']:
        p=PACKAGE/'score_checkpoints_antigravity'/f"{record['dataset']}_diagnostics_antigravity.json.gz"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==record['diagnostics_sha256']
        diagnostics=json.loads(gzip.decompress(p.read_bytes()))
        for method in METHODS:
            info=numerical[method]; folds=diagnostics[method]; assert len(folds)==6
            sparse=sum(sum(f.get('sparse_changed_mask_',[])) for f in folds)
            robust=sum(sum(f['robust_changed_mask_']) for f in folds)
            info['sparse_changed_column_fits']+=sparse; info['robust_changed_column_fits']+=robust
            if sparse+robust: info['touched_families'].append(record['source_family'])
            for key in ('old_feature_max_abs','stable_feature_max_abs','old_scaled_max_abs','stable_scaled_max_abs'):
                info[key]=max(info[key],max(f[key] for f in folds))
    for method in METHODS:
        info=numerical[method]
        for branch in ('old','stable'):
            summaries=[r['score_stats'][method][branch]['train_oof'] for r in result['records']]
            tails=[r['tail_q95'] for r in summaries if r['tail_q95'] is not None]
            info[branch]=dict(max_oof_abs=max(r['max_abs'] for r in summaries),median_family_oof_max=float(np.median([r['max_abs'] for r in summaries])),
                              median_tail_q95=float(np.median(tails)),max_tail_q95=max(tails),undefined_tail_count=len(summaries)-len(tails),
                              finite_count=sum(r['finite_count'] for r in summaries),score_count=sum(r['count'] for r in summaries))
        info['touched_family_count']=len(info['touched_families'])
    verified=dict(result_sha256=manifest['result_sha256'],numerical=numerical,performance=result['summary'],independent_verification=audit)
    (PACKAGE/'numerical_summary_antigravity.json').write_text(json.dumps(verified,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    rows=[]; stability=[]; distributions=[]
    for method in METHODS:
        s=result['summary'][method]; n=numerical[method]
        rows.append(f"<tr><th>{NAMES[method]}</th><td>{s['old']['mean_auc']:.6f}</td><td>{s['stable']['mean_auc']:.6f}</td><td>{s['auc_paired']['mean_delta']:+.6f}</td><td>{s['old']['total_zf']} → {s['stable']['total_zf']}</td><td>{s['zf_paired']['mean_delta']:+.3f}</td></tr>")
        stability.append(f"<tr><th>{NAMES[method]}</th><td>{n['touched_family_count']} / 125</td><td>{n['old_scaled_max_abs']:,.2f} → {n['stable_scaled_max_abs']:,.2f}</td><td>{n['old']['max_oof_abs']:,.2f} → {n['stable']['max_oof_abs']:,.2f}</td><td>{n['old']['median_tail_q95']:.3f} → {n['stable']['median_tail_q95']:.3f}</td></tr>")
        distributions.append(f"<li>{NAMES[method]} — AUC 개선 {s['auc_paired']['wins']}, 악화 {s['auc_paired']['losses']}, 동일 {s['auc_paired']['ties']}; ZF 개선 {s['zf_paired']['wins']}, 악화 {s['zf_paired']['losses']}, 동일 {s['zf_paired']['ties']}.</li>")
    case_rows=[]
    for r in result['records']:
        for m in METHODS:
            s=r['metrics'][m]; delta=s['stable']['auc']-s['old']['auc']
            case_rows.append(f"<tr><td>{html.escape(r['source_family'])}</td><td>{NAMES[m]}</td><td>{s['old']['auc']:.6f}</td><td>{s['stable']['auc']:.6f}</td><td>{delta:+.6f}</td><td>{s['old']['zf']} → {s['stable']['zf']}</td></tr>")
    all_auc=all(result['summary'][m]['auc_paired']['mean_delta']>0 for m in METHODS)
    conclusion='두 방법 모두 평균 AUC가 개선됐습니다.' if all_auc else '정규화 안정화가 두 방법 모두의 평균 AUC 개선으로 이어지지는 않았습니다.'
    smoke=json.loads((PACKAGE/'smoke_results_antigravity.json').read_text())
    specific=[]
    for dataset,method in (('DiatomSizeReduction_normal_1',METHODS[1]),('ProximalPhalanxOutlineCorrect_normal_0',METHODS[0])):
        s=smoke['cases'][dataset][method]['stats']
        specific.append(f"<li>{dataset}: 정상 TRAIN OOF 최대 절댓값 {s['old']['max_abs']:,.2f} → {s['stable']['max_abs']:,.2f}; q95 꼬리 비율 {s['old']['tail_q95']:,.2f} → {s['stable']['tail_q95']:,.2f}.</li>")
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp492 검증 결과</title><style>
body{font:16px/1.65 system-ui;max-width:1220px;margin:40px auto;padding:0 22px 80px;color:#192d38;background:#fbfcfd}h1{font-size:2rem}h2{margin-top:36px}.lead{padding:20px;background:#eaf1f5;border-left:5px solid #245775}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{border-bottom:1px solid #d6dfe5;padding:12px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#eaf1f5}input{font:inherit;padding:10px;width:min(90%,450px);margin:16px 0}code{overflow-wrap:anywhere}a{color:#17627d}</style>
<h1>Exp492 · 정규화 안정화와 평균 성능</h1>'''+f'<p class="lead"><strong>{conclusion}</strong><br>기존 125개 family 대표에서 동일 인코더·fold·seed로 기존/안정화 분기를 비교했습니다. 완료 125, 실패 0. 실제 평균 AUC와 ZF를 모두 평가했습니다.</p>'
    page+='<h2>평균 성능</h2><div class="scroll"><table><thead><tr><th>방법</th><th>기존 AUC</th><th>안정화 AUC</th><th>평균 ΔAUC</th><th>ZF 합계</th><th>평균 ΔZF</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><ul>'+''.join(distributions)+'</ul><p>AUC는 family 동등가중 평균입니다. ZF는 기존 정의인 TEST 정상 최대 점수를 엄격히 초과하는 이상 샘플 수이며 배포 임계값이 아닙니다. 평균 ΔZF는 합계 변화 ÷ 125입니다.</p>'
    page+='<h2>수치 증폭은 얼마나 줄었나</h2><div class="scroll"><table><thead><tr><th>방법</th><th>분모가 변경된 family</th><th>전체 정규화 특징 최대 절댓값</th><th>전체 TRAIN OOF 최대 절댓값</th><th>family q95 꼬리 비율 중앙값</th></tr></thead><tbody>'+''.join(stability)+'</tbody></table></div><p>특징 최대값은 5 OOF 적합과 full TRAIN 적합의 reference/query 전체를 포함합니다. 꼬리 비율은 (q95−q50)/(q75−q25)이며 IQR=0이면 미정의로 기록합니다. 큰 분모를 줄이지 않고, 지정된 작은 분모만 1로 바꾸었습니다. 모든 큰 점수를 제거하는 규칙은 아닙니다.</p><ul>'+''.join(specific)+'</ul>'
    page+='<h2>검증과 해석 범위</h2><ul><li>회귀 검사 6개 통과. 저장된 점수 500개 평가를 순위합 AUC 공식으로 독립 재계산해 일치 확인(AUC 최대 차이 2.22×10⁻¹⁶, ZF 완전 일치). 합성 데이터 및 두 실제 TRAIN 사례 × 두 방법에서 반복 점수 일치, 기존 Exp449 점수와 수치·후보 인덱스 일치.</li><li>125개 모두의 점수와 진단 파일을 저장·해시 검증한 후 TEST 라벨을 열었습니다. 재개 시 라벨 경계 이후 점수 재생성을 금지하고 원본 TEST 행 순서를 대조합니다.</li><li>CPU 작업자 11개, 내부 thread 1개. Numba는 신규 마스킹 단계 전체 비용에서 이득이 없었습니다. MLX float32 변환은 수치 parity 실패, 명시적인 float64 GPU 연산은 미지원이었습니다. 공식 aeon/기존 Numba/NumPy/sklearn CPU 경로를 유지했습니다.</li><li>기존 Exp489–491 및 원본 데이터는 보존했습니다. 진단에 사용한 개발 집단을 다시 평가했으므로 독립 검증은 아닙니다. Exp490 Ridge·실제 라우팅 성능은 이번 범위에 포함하지 않았고, 자동 승격하지 않았습니다.</li></ul>'
    page+='<p><a href="../experiments/exp492_stable_normalization_antigravity/protocol_antigravity.html">고정 실험 계획</a> · <a href="../experiments/exp492_stable_normalization_antigravity/results_antigravity.json">성능 원본 JSON</a> · <a href="../experiments/exp492_stable_normalization_antigravity/numerical_summary_antigravity.json">수치 검증 JSON</a> · <a href="../experiments/exp492_stable_normalization_antigravity/consultation_antigravity.html">Pro·Grok·독립 검토</a> · <a href="../experiments/exp492_stable_normalization_antigravity/verification_antigravity.json">독립 재계산 검증</a></p>'
    page+='<h2>계열별 비교</h2><label for="filter">계열 또는 방법 검색</label><br><input id="filter" placeholder="예: Diatom 또는 Hydra" oninput="document.querySelectorAll(\'#families tr\').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))"><div class="scroll"><table><thead><tr><th>family</th><th>방법</th><th>기존 AUC</th><th>안정화 AUC</th><th>ΔAUC</th><th>ZF</th></tr></thead><tbody id="families">'+''.join(case_rows)+'</tbody></table></div></html>'
    target=ROOT/'docs/EXP492_REVIEW_antigravity.html'; target.write_text(page)
    print(json.dumps(verified,ensure_ascii=False,indent=2))
    print(target)

if __name__=='__main__': main()
