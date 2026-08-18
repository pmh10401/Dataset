# Exp449 논문 작성 패키지

이 디렉터리는 **AI-Driven Equipment Intelligence: Equipment Anomaly Detection**의
논문 작성과 사내 심사에 필요한 최종 비교 근거를 한곳에 모은 패키지다. 원시
시계열 DB와 개별 TEST 라벨은 포함하지 않으며, 검증된 집계 결과·논문용 표·그림·
방법 설명·재현 정보만 공개한다.

## 가장 먼저 볼 자료

1. [공식 비교 보고서](report_antigravity/official_comparison_report_antigravity.html)
2. [논문 초안](MANUSCRIPT_DRAFT_KO_antigravity.md)
3. [결과와 허용 가능한 주장](RESULTS_AND_CLAIMS_antigravity.md)
4. [방법 및 지표 정의](METHODS_AND_METRICS_antigravity.md)
5. [그림·표 색인](FIGURE_TABLE_INDEX_antigravity.md)
6. [재현 절차](REPRODUCIBILITY_antigravity.md)
7. [데이터 공개 범위와 한계](DATA_AVAILABILITY_AND_LIMITATIONS_antigravity.md)

## 핵심 결과

- 평가 모집단: 1,117개 데이터셋, 195,900개 TEST 시계열
- 제안기술: Micro F1 `0.630610`, Macro F1 `0.622946`, AP `0.674086`,
  AUROC `0.961077`, 경보 2,276개
- 동일 경보 예산에서 가장 강한 공개 기준선은 OC-SVM이며, 제안기술의
  Micro F1 이득은 `+0.122054`, Macro F1 이득은 `+0.158084`다.
- 모든 7개 방법이 완주한 공통 922개 데이터셋에서는 제안기술 AP가
  `0.669675`, AUROC가 `0.964170`이다.
- 동일 경보 예산 Macro F1의 paired bootstrap에서 여섯 기준선의
  `기준선−제안기술` 95% 구간은 모두 0보다 작다.

## 디렉터리 구성

- `data_antigravity/`: 공식 요약, 데이터셋 단위 지표, bootstrap, ROC/PR 좌표
- `tables_antigravity/`: 논문에 바로 사용할 수 있는 CSV 표
- `figures_antigravity/`: 논문용 독립 SVG 그림 5개
- `report_antigravity/`: 브라우저에서 열 수 있는 공식 HTML 보고서
- `source_docs_antigravity/`: 사전등록·구현·설계 근거의 스냅샷

## 근거 우선순위

수치가 충돌할 경우 다음 순서를 따른다.

1. `data_antigravity/official_comparison_summary_antigravity.json`
2. `data_antigravity/official_common_support_roc_pr_curves_antigravity.json`
3. `data_antigravity/official_comparison_bootstrap_antigravity.json`
4. 공식 HTML 보고서와 논문용 파생 표
5. `source_docs_antigravity/`의 계획·사전등록 문서

초기 설계 문서에는 제안기술의 연속 점수가 없다고 가정한 문장이 남아 있다.
최종 공식 산출물은 연속 점수가 실제로 동결되었음을 검증했으므로, 그 초기
가정은 최종 산출물에 의해 대체된다.

## 공개 범위

이 패키지는 후향적 공개 벤치마크 연구 근거다. 실제 반도체 설비 현장에서의
고장 예측 성능, 인과 효과, 실시간 운영 안정성 또는 고장 확률 보정을 입증하지
않는다. 제안 연속 점수는 **순위용 이상 점수**이며 고장 확률이 아니다.
