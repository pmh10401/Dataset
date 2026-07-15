# Exp151~166 연구 릴리스 인덱스

## 목적

이 문서는 Exp151부터 Exp166까지의 재현 코드, 사전등록 문서, 검증 자료와 핵심 결과 파일을 찾기 위한 인덱스다. GitHub 공개 범위를 명확히 하고, 연구 결과를 운영 성능으로 과장하지 않기 위한 해석 기준도 함께 기록한다.

## 포함 범위

- `experiments/exp151_152/`부터 `experiments/exp166/`까지의 실험 코드와 테스트
- 실험 계보 재현에 필요한 저장소 루트의 선행 실험 및 공통 유틸리티 코드
- `scratch/`의 구조, 불변성, 대시보드 및 큐 안전성 테스트
- `docs/EXP151_152_*.md`부터 `docs/EXP166_*.md`까지의 사전등록과 실행 문서
- `outputs/exp151_152_virtual_run_conformal_v2/`
- `outputs/exp158_159_small_train_similarity_pooling/`
- `outputs/exp160_candidate_zero_similarity_recovery/`
- `outputs/exp161_alternative_source_audit/`
- `outputs/exp162_full_coverage_alternative_score_cache/`
- `outputs/exp163_alternative_source_tail_agreement_review/`
- `outputs/exp164_train_normal_agreement_risk_audit/`
- `outputs/exp165_small_sample_calibration/`
- `outputs/exp166_small_sample_anomaly_coverage/`

각 output 디렉터리의 `MANIFEST.md`, 검증 보고서, summary 또는 result analysis를 먼저 읽으면 실험 계약과 결과 범위를 빠르게 확인할 수 있다.

## 제외 범위

다음 파일은 크기가 크거나 동일 명령으로 재생성 가능한 로컬 실행 산출물이므로 GitHub에 포함하지 않는다.

- `/Users/minho/Documents/Dataset/univariate_ts.db`와 기타 데이터베이스
- `.venv*` 가상환경
- score cache, target cache, `*.npz`
- 실행 체크포인트와 smoke 출력
- 런타임 로그, 압축파일, 발표자료

데이터베이스는 코드에 기록된 기존 절대 경로를 사용하며 이 릴리스에 복제하지 않는다.

## 실험 흐름

1. Exp151/152는 TEST label과 prediction을 분리하고, TRAIN-normal conformal 후보에 교차 확인과 보조 확인을 다시 적용했다.
2. Exp158/159는 작은 TRAIN 집합에 score-similarity pooling을 적용하되 autonomous Hard alert는 보존하고 review coverage만 진단했다.
3. Exp160~163은 candidate-zero 문제와 대안 점수원 coverage를 감사하고 review 후보를 확장했다.
4. Exp164는 TRAIN-normal pseudo-run에서 합의 규칙의 정상 오경보 위험을 감사했다.
5. Exp165는 `5 <= n_train < 20` 소표본에서 similarity pooling으로 threshold 해상도를 복구했다.
6. Exp166은 Exp165의 고정된 조합을 TEST label과 분리해 후향 평가했다.

## 핵심 해석 제한

- 모든 결과는 기존 benchmark를 사용한 retrospective 연구 결과다.
- Exp151~166만으로 실제 설비의 prospective 성능을 주장할 수 없다.
- Hard alert는 autonomous 지표이며 Standard/Priority review와 combined F1은 human-assisted 진단 지표다.
- TEST 결과를 이용해 alpha, threshold, donor, source pair 또는 agreement variant를 선택하지 않는다.
- Exp166에서는 similarity pooling이 TRAIN 정상 오경보와 threshold 해상도를 개선했지만 TEST TP/FP 균형과 Mean F1을 함께 개선하지 못했다.
- Exp165 pooling은 현재 운영 기본 정책이나 autonomous Hard alert로 채택하지 않는다.
- 다음 검증 과제는 cross-fit calibration과 full-TRAIN TEST scoring의 생성 조건을 맞추는 score-calibration parity다.

## 재현 시작점

실험별 실행 명령과 계약은 각 `docs/EXP*_RUNBOOK_KO.md` 및 output의 `MANIFEST.md`를 따른다. 데이터베이스와 입력 결과 파일의 존재 여부를 먼저 확인하고, 단위 테스트와 smoke test가 통과한 뒤 full run을 실행한다.
