# Exp449 공식 공개 기준선 비교 설계

## 목적

논문에 사용할 제안기술과 공개 이상 탐지 기법의 후향 비교 결과를 동일한
1,117개 데이터셋과 195,900개 TEST 행에서 산출한다. 본 비교는 저장된
benchmark에 대한 기술 비교이며 현장 일반화, 자동 champion 승격 또는 운영 적용을
주장하지 않는다.

## 비교 대상

- 제안기술: 동결된 최종 경보 정책
- kNN distance
- Local Outlier Factor
- RBF One-Class SVM
- Isolation Forest
- Deep SVDD 1D adapter
- DROCC 1D adapter (MPS 실행)

고전 기준선 4종은 기존 공식 동결·평가 결과를 그대로 사용한다. Deep SVDD는
혼합 실행 root에 게시된 1,117개 완료 shard를 byte 변경 없이 독립 동결 view로
구성한다. DROCC는 별도 MPS root의 1,117개 동결 score를 사용한다. CPU–MPS
동등성은 사용자가 수행하여 이상 없음을 확인한 사실로 기록하되, 별도 기계 판독
검증 artifact가 제공되지 않았다는 점도 함께 명시한다.

## 평가 규칙

1. 모든 score root와 shard hash를 TEST label 접근 전에 검증한다.
2. 주 비교는 데이터셋별 제안기술 경보 수만 맞춘 `matched_alert_budget`이다.
   공개 기준선은 제안기술 좌표를 복사하지 않고 각자의 anomaly score 상위 순위를
   사용한다.
3. 보조 비교는 conformal alpha `0.005`, `0.01`, `0.02`이다.
4. 모든 방법에 Precision, Recall, micro F1, macro F1, coverage와 alert 수를
   보고한다.
5. 연속 score가 있는 공개 기준선에는 AP와 AUROC를 보고한다. micro AP/AUROC는
   `1 - TRAIN-OOF upper-tail p-value`, macro AP/AUROC는 데이터셋 내부 원시
   anomaly-score 순위를 사용한다.
6. 제안기술의 동결 산출물에는 연속 score가 없으므로 AP/AUROC를 사후 생성하지
   않고 `—`로 표기한다.
7. 데이터셋 공유 bootstrap draw 2,000회로 공개 기준선과 제안기술 간 macro F1
   차이의 95% 구간을 계산한다.

## Deep SVDD 독립 동결 view

독립 view는 원본 Deep SVDD JSON shard 1,117개의 byte와 SHA-256을 그대로
보존한다. 새 run identity는 단일 방법 모집단만 표현하고, 각 metadata record는
새 run-config hash와 원본 shard hash를 결합한다. provenance에는 원본 root,
원본 run-config hash, 원본/독립 manifest hash와 전체 shard 집합 hash가 포함된다.
점수 재학습, 점수 변환 또는 좌표 변경은 금지한다.

## 결과 산출물

- 기술 요약 HTML 보고서
- 논문 표에 바로 사용할 전체 요약 CSV
- 데이터셋별 audit CSV
- paired-bootstrap JSON
- Deep SVDD 독립 view provenance
- 고전·Deep SVDD·DROCC 평가 artifact-map SHA-256과 최종 보고서 SHA-256

보고서의 주 표는 matched-alert-budget 결과를 사용하고, conformal 결과와
데이터셋별 지표는 보조 표로 둔다. 제안기술 AP/AUROC 부재와 MPS 동등성 증거의
출처를 표 주석에 명확히 적는다.

## 실패 처리

- 원본 shard 누락·변조·중복 또는 모집단 불일치: 비교 중단
- Deep SVDD derived view의 shard byte 불일치: 비교 중단
- 평가 root가 기존 다른 byte를 포함: 덮어쓰지 않고 새 root 사용
- 기준선 간 데이터셋 순서 또는 TEST 길이 불일치: 결합 중단
- 하나의 방법이라도 공식 평가가 완성되지 않음: 6개 방법 공식 완료로 표기 금지

## 검증

- 각 score manifest와 평가 artifact map을 scorer-free 방식으로 재검증한다.
- 요약 수치를 dataset-level confusion 합계에서 독립 재계산한다.
- HTML 보고서의 수치가 canonical summary와 일치하는지 확인한다.
- 제안기술 연속 score를 추정하거나 공개되지 않은 사내 버전을 기준선으로 넣지
  않는다.
