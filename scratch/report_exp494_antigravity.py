"""Render the completed full-population comparison from verified artifacts."""
from pathlib import Path
import hashlib,html,json
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'experiments/exp494_all_datasets_low_fpr_antigravity'
r=json.loads((P/'results_antigravity.json').read_text());v=json.loads((P/'verification_antigravity.json').read_text())
assert v['state']=='passed' and v['result_sha256']==hashlib.sha256((P/'results_antigravity.json').read_bytes()).hexdigest()
fixed=['MiniRocket','MultiRocket','Hydra stable','Hydra+MultiRocket stable','MiniRocket+LOF']
current=[f'{kind}__stable / {head}' for kind in ['24D','old42D','stable42D'] for head in ['auc','zf_sign']]
labels={'MiniRocket':'MiniRocket','MultiRocket':'MultiRocket','Hydra stable':'Hydra 안정화','Hydra+MultiRocket stable':'Hydra + MultiRocket 안정화','MiniRocket+LOF':'MiniRocket + LOF','Hydra old':'Hydra 기존 정규화','Hydra+MultiRocket old':'Hydra + MultiRocket 기존 정규화'}
for kind,title in [('24D','24D'),('old42D','기존 42D 특징'),('stable42D','안정화 42D 특징')]:
 for target in ['old','stable']:
  for head,h in [('auc','AUC 선택'),('zf_sign','ZF 선택')]:labels[f'{kind}__{target} / {head}']=f'{title} · {h}'+(' (이전 학습 목표)' if target=='old' else '')
esc=html.escape

def table(group,metric,names):
    rows=r['summaries'][group];available=[n for n in names if n in rows];maxima=[max(rows[n][metric][j] for n in available) for j in range(4)]
    body=[]
    for name in available:
        s=rows[name];cells=''.join(f'<td class="{"best" if abs(x-maxima[j])<1e-12 else ""}">{x*100:.2f}%</td>' for j,x in enumerate(s[metric]))
        body.append(f'<tr><th>{esc(labels[name])}</th>{cells}<td>{s["total_zf"]:,}</td></tr>')
    return '<div class="scroll"><table><thead><tr><th>모델 / 선택 방식</th><th>관측 오탐 0건</th><th>오탐률 ≤0.1%</th><th>≤0.5%</th><th>≤1%</th><th>ZF 합계</th></tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'

def rankings(group):
    rows=r['summaries'][group];available=list(rows)
    best=max(available,key=lambda n:rows[n]['family_macro_tpr'][1]);base=rows['MiniRocket']['family_macro_tpr'][1]
    fixed_best=max(fixed,key=lambda n:rows[n]['family_macro_tpr'][1])
    value=rows[best]['family_macro_tpr'][1]
    router=max([n for n in rows if ' / ' in n],key=lambda n:rows[n]['family_macro_tpr'][1])
    return f'<p>비교 지점 0.1%에서 전체 조합의 최고 관측값은 <strong>{esc(labels[best])} {value*100:.2f}%</strong>입니다. 선택기 중 최고는 {esc(labels[router])} {rows[router]["family_macro_tpr"][1]*100:.2f}%입니다. 이 지점은 운영 기준으로 확정한 값이 아닙니다.</p>'
sections=[]
for group,title,description in [('all_evaluable','전체 평가 가능 데이터셋','1,117개 데이터셋 · 128개 원본 계열. 42D 선택기는 161개에서 정의되지 않아 이 표에 포함하지 않습니다.'),('common_42d','모든 모델의 공통 비교 범위','정상 학습 샘플이 5개 이상인 956개 데이터셋 · 125개 원본 계열. 모델 간 비교 모집단을 동일하게 맞췄습니다.')]:
    sections.append(f'<section><h2>{title}</h2><p>{description}</p>'+rankings(group)+table(group,'family_macro_tpr',fixed+current)+f'<details><summary>데이터셋별 동일 가중 평균</summary><p>원본 계열에서 파생된 데이터셋이 많을수록 해당 계열의 영향이 커지는 보조 집계입니다.</p>{table(group,"dataset_macro_tpr",fixed+current)}</details><details><summary>기존 정규화 및 이전 학습 목표까지 전체 조합</summary>{table(group,"family_macro_tpr",list(r["summaries"][group]))}</details></section>')
resolution=''.join(f'<tr><th>{a*100:g}%</th><td>{n:,}</td><td>{1117-n:,}</td></tr>' for a,n in zip(r['fpr_grid'],r['false_positive_resolution']['all_evaluable']))
exclusions=''.join(f'<li><code>{esc(x["dataset"])}</code>: '+('기존 벤치마크의 정상 클래스 매핑이 없습니다.' if x['dataset']=='CornellWhaleChallenge' else '정상 학습 샘플이 1개라 현재 점수 모델의 최소 2개 조건을 충족하지 못합니다.')+'</li>' for x in r['excluded'])
page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp494 전체 데이터셋 저오탐 비교</title><style>:root{color-scheme:light dark}body{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:36px auto;padding:0 24px;background:#f5f7fb;color:#172239}h1{font-size:32px;line-height:1.3}h2{font-size:23px}section,.notice{background:white;padding:24px;margin:24px 0;border:1px solid #d7ddeb;border-radius:12px}.notice{border-left:5px solid #83601c;background:#fffbef}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px;white-space:nowrap}td,th{border-bottom:1px solid #d7ddeb;padding:10px 12px;text-align:right}th:first-child{text-align:left}thead{background:#edf1f8}.best{font-weight:750;color:#145b3c;background:#edf9f2}details{margin-top:22px}summary{cursor:pointer;font-weight:650}code{overflow-wrap:anywhere}.meta{font-size:13px;color:#57627a}a{color:#174baf}@media(prefers-color-scheme:dark){body{background:#111823;color:#e4eaf6}section{background:#192332;border-color:#354258}.notice{background:#302b1a;color:#f4e8c5;border-color:#796835}thead{background:#223148}td,th{border-color:#354258}.best{background:#193f31;color:#a5efd0}.meta{color:#aebbd1}a{color:#a7c6ff}}</style></head><body>'''
page+='<p class="meta">Exp494 · 고정 모델의 평가 범위 확대 · 완료 및 독립 재계산 확인</p><h1>전체 데이터셋에서의 저오탐 검출률</h1><p>원본 1,119개 중 기존 평가 정의를 적용할 수 있는 1,117개를 측정했습니다. 표의 검출률은 원본 계열 안에서 데이터셋을 평균한 뒤 128개 또는 125개 계열을 동일하게 가중한 값입니다.</p><div class="notice"><strong>관측 성능의 비교입니다.</strong> TEST 라벨로 각 오탐 한도에서 가능한 최대 검출률을 계산했습니다. 동점을 나누거나 곡선을 보간하지 않았습니다. 별도 정상 자료로 교정한 운영 임계값의 성능이나 미래 오탐 보장을 뜻하지 않습니다. 데이터 계열과 기존 개발 결과를 반복 활용했으므로 통계적 최종 우위로 해석하지 않습니다.</div>'
page+=''.join(sections)
old=r['summaries']['previous_representatives'];common=r['summaries']['common_42d'];router='old42D__stable / auc'
old_delta=(old[router]['family_macro_tpr'][1]-old['MiniRocket']['family_macro_tpr'][1])*100
new_delta=(common[router]['family_macro_tpr'][1]-common['MiniRocket']['family_macro_tpr'][1])*100
page+=f'<section><h2>125개 대표 비교에서 달라진 결론</h2><p>기존 42D · AUC 선택기는 대표 125개에서 오탐률 0.1% 기준 MiniRocket보다 {old_delta:+.2f}%p 높았지만, 공통 956개에서는 {new_delta:+.2f}%p입니다. 기존의 근소한 우위가 전체 변형으로 확대됐을 때 유지되지 않았습니다.</p><p>MiniRocket은 전체 1,117개와 공통 956개 모두에서 계열 평균 및 데이터셋 평균의 네 비교 구간 전체 1위입니다. 다만 이는 반복 사용한 개발 자료에서의 관측 결과이며 최종 운영 검증이 아닙니다.</p></section>'
page+='<section><h2>전체 평가를 위한 전처리 예외 1건</h2><p><code>UWaveGestureLibraryZ_normal_8</code>의 정상 학습 448행 중 1행이 완전한 상수여서 기존 표준화가 중단됐습니다. TEST 평가 전에 보충 규칙을 고정하고, 정확히 표준편차가 0인 행의 분모만 1로 두어 0 벡터로 처리했습니다. 나머지 학습 행과 TEST 550행은 기존 전처리와 bitwise 일치하며 행을 제외하지 않았습니다.</p><p>이 데이터셋의 점수만 별도로 생성했습니다. 기존 성공 점수와 원래 코드 해시는 보존했습니다. 과거 전체 캐시의 해당 데이터셋을 재현했다고 주장하지 않으며, 이 결과에는 명시된 입력 처리 확장이 포함됩니다. <a href="../experiments/exp494_all_datasets_low_fpr_antigravity/constant_recovery_protocol_antigravity.html">보충 규칙</a> · <a href="../experiments/exp494_all_datasets_low_fpr_antigravity/constant_recovery_manifest_antigravity.json">복구 근거</a></p></section>'
page+='<section><h2>표본 해상도와 평가 범위</h2><p>정상 TEST 표본 수가 작으면 오탐률 한도가 달라도 허용 가능한 오탐 수가 동일합니다.</p><table><tr><th>오탐률 한도</th><th>오탐 1건 이상 허용 가능한 데이터셋</th><th>오탐 0건만 가능한 데이터셋</th></tr>'+resolution+'</table><p>42D 특징이 정의되지 않는 161개는 성능을 보고 제외한 것이 아니라, 고정된 5-fold 특징 계산의 최소 학습 샘플 조건을 충족하지 못한 경우입니다. 42D 선택기를 다른 모델로 임의 대체하지 않았습니다.</p><p>전체 평가에서 제외된 2개:</p><ul>'+exclusions+'</ul></section>'
page+='<section><h2>고정한 조건과 검증</h2><ul><li>125개 대표 데이터셋의 기존 점수·특징을 재사용하고 나머지 992개를 동일한 알고리즘과 seed로 계산했습니다.</li><li>선택기의 학습 모집단과 목표는 Exp493의 기존 125개로 고정했습니다. 같은 원본 계열은 학습에서 모두 제외했습니다. 확장된 TEST 결과로 선택기를 재학습하지 않았습니다.</li><li>전체 점수와 선택 결과를 저장한 뒤 새 평가 결과를 계산했습니다.</li><li>기존 125개에서 19개 조합의 저오탐 비교값을 재현했습니다.</li><li>'+f'{v["score_vectors"]:,}개 점수 벡터와 {v["aggregate_rows"]}개 집계 행을 독립 재계산했습니다. AUROC 최대 계산 차이는 {v["auc_max_abs_error"]:.3g}입니다.'+'</li></ul><p><a href="../experiments/exp494_all_datasets_low_fpr_antigravity/results_antigravity.json">전체 결과 JSON</a> · <a href="../experiments/exp494_all_datasets_low_fpr_antigravity/verification_antigravity.json">검증 결과</a> · <a href="../experiments/exp494_all_datasets_low_fpr_antigravity/protocol_antigravity.html">평가 정의</a></p><p class="meta">'+esc(v['result_sha256'])+'</p></section></body></html>'
out=ROOT/'docs/EXP494_ALL_DATASETS_LOW_FPR_antigravity.html';out.write_text(page)
assert '<script' not in page and len(r['records'])==1117
print(out)
items=[]
for group,title in [('all_evaluable','전체 1,117개 · 128개 계열 비교'),('common_42d','공통 956개 · 125개 계열 비교')]:
    columns=['모델','검출률 FP=0 (%)','검출률 FPR≤0.1% (%)','검출률 FPR≤0.5% (%)','검출률 FPR≤1% (%)','데이터셋 평균 FPR≤0.1% (%)','ZF 합계']
    rows=[]
    for name,s in r['summaries'][group].items():
        rows.append(dict(zip(columns,[labels[name],*[round(x*100,6) for x in s['family_macro_tpr']],round(s['dataset_macro_tpr'][1]*100,6),s['total_zf']])))
    source=dict(label='Exp494 검증된 전체 결과',files=[{'label':x} for x in ['results_antigravity.json','decisions_antigravity.json','verification_antigravity.json','constant_recovery_manifest_antigravity.json']],
        metricDefinitions=[dict(label='주 검출률',definition='원본 계열 안에서 데이터셋 검출률을 평균한 뒤 계열을 동일하게 가중한다. 각 데이터셋에서 floor(FPR×정상 표본 수) 이하 오탐으로 가능한 최대 검출률이다. 동점은 나누지 않으며 보간하지 않는다.')],
        filters=[title,'라우터 학습은 기존 125개 대표 데이터셋으로 고정하고 평가 대상과 같은 계열을 제외했다.'],
        caveats=['TEST ROC의 후향적 비교이며 독립 교정 임계값의 운영 성능이나 미래 오탐 보장이 아니다.','원본 1,119개 중 정상 클래스 매핑 미정 1개와 정상 TRAIN 1행인 1개를 제외했다.','정상 TRAIN 5행 미만 161개에는 42D 선택기가 정의되지 않는다.','상수 정상 TRAIN 행 1개의 분모를 1로 두는 사전 보충 규칙을 적용했다. 다른 행과 기존 성공 점수는 변경하지 않았다.'],
        evidenceFlow=[dict(kind='validation',title='독립 재계산 통과',description=f'{v["score_vectors"]:,}개 점수 벡터와 {v["aggregate_rows"]}개 집계 행, 원본 데이터 및 코드 해시, 복구 출처를 검증했다. 기존 125개에서 19개 모델 조합을 재현했다.')])
    items.append(dict(id=group.replace('_','-'),title=title,queries=[dict(id=group.replace('_','-')+'-metrics',source=source,columns=columns,rows=rows,preview=dict(kind='aggregate',note='표시된 동일 모집단 내 전체 모델 조합'),methods=[dict(language='calculation',code='budget=floor(alpha*n_normal); threshold=sort(normal_scores,descending=True)[budget]; TPR=mean(anomaly_scores>threshold); primary=mean_families(mean_datasets_within_family(TPR))')])]))
receipt=Path('/Users/mac/.codex/visualizations/2026/09/15/01a0a737-eba0-7db1-bd5d-b0f22f3f74fa/exp494_full_sources_antigravity.json')
receipt.write_text(json.dumps(dict(schemaVersion=1,items=items),ensure_ascii=False,indent=2)+'\n');print(receipt)
