# Exp414 결측 인지 Near-Repeat Qualification 사전등록

작성일: 2026-07-28
상태: 구현 전 동결
모드: `TRAIN-only missing-aware near-repeat qualification`

## 입력과 범위

Exp414는 Exp411 입력 계약 실패를 별도 실험 번호에서 다룬다. 입력 DB SHA256은
`8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`, 모집단
manifest SHA256은
`41a913415d22bb6cedb692a1c1cce85eafccae49f8a3fd108ed53e1bc3d2a7b8`다.
모집단은 1,117 dataset, 128 family다. TRAIN-only read-only inventory는 영향
dataset 19개, run 164개, non-finite value 983개, all-missing run 0개다.
정렬 영향 dataset 배열의 canonical JSON SHA256은
`ca376adad36fd68e65ea54e004fcb1e5288b3e7d243304b56def507ec17aa9b9`다.

SQL과 qualification은 TEST split, `labels_blob`, 실제 이상 위치, TEST 지표,
과거 family TEST 성능, source ranking을 읽지 않는다. DB와 blob은 수정하지
않는다.

## 복구와 identity 계약

각 정상 TRAIN `<f4` blob의 원본 bytes SHA256을 먼저 기록한다. NaN, +Inf,
-Inf를 결측으로 취급하고, 내부 결측은 원 시계열 좌표에서 선형 보간한다.
시작과 끝 결측은 가장 가까운 유한값으로 연장한다. 결과는 길이를 바꾸지 않은
little-endian float32다. 유한 blob은 재직렬화하지 않고 원본 bytes를 그대로
사용한다. tail replacement, 길이 보간, padding, truncation은 금지한다.
전부 결측인 run이 하나라도 있으면 dataset은 `missing_support_abstain`이다.

복구 후 bytes SHA256을 identity group key로 사용한다. 같은 복구 waveform은
하나의 group으로 묶어 fit/query 양쪽에 분리되지 않게 하며, `exact`도 이
identity를 기준으로 판정한다.

## 유지되는 Exp411 계약과 판정

Exp411 source commit은 `5c4268fb89ac806dac6bdccd31f5724ee520c63a`다. seed
411, `lower` quantile 0.01 / 0.05 / 0.10, identity-group fold 규칙, 동일 길이
비교, `raw_rms`/`diff_rms`/`spectrum_rms`, 일곱 TRAIN-only 합성 교란과 safety
gate를 변경하지 않는다. NumPy CPU를 정본으로 하고 Numba는 12 threads,
중첩 수치 라이브러리는 1 thread를 사용한다. 거리 허용오차는 `1e-6`이며
NumPy/Numba tag와 선택 quantile은 같아야 한다. MLX, PyTorch, MPS는 사용하지
않는다.

각 후보는 정상 near 인정률 >= 0.01, pooled synthetic false-normal <= 0.001,
종류별 최악 <= 0.005, family bootstrap 95% 상한 <= 0.005를 모두 만족해야
한다. 통과 후보 중 가장 넓은 quantile을 택하고, 없으면
`no_qualified_near_rule`이다. 전체 모집단 집계만 global 판정에 쓰며,
비결측·결측 복구·DodgerLoop·MelbournePedestrian·missing-support subgroup은
진단 전용이다. subgroup 결과로 quantile, 거리, 합의, threshold, 교란 또는
review budget을 변경하지 않는다.

## 제한적 Exp411 이전

Exp411 checkpoint는 읽기 전용이고 Exp414 checkpoint와 output은 별도다.
이전은 checkpoint root 및 record schema/hash, DB/manifest/raw TRAIN identity,
결측 복구 수 0, fold/distance/quantile/perturbation/gate contract, backend audit,
대표 표본의 tag/count/quantile/qualification 재계산 일치가 모두 검증될 때만
허용한다. 이전 행은 `migrated_exp411_verified`, 새 계산 행은
`exp414_computed`로 기록한다. 검증 불완전 또는 검증되지 않은 record 사용은
fail-closed다. stale progress 186이나 record 수 187은 migration coverage
성공 조건이 아니다.

## 산출물과 승인 gate

정본 산출물은 다음 9개다.

1. `exp414_frozen_config.json`
2. `exp414_missing_repair_audit.csv`
3. `exp414_dataset_qualification.csv`
4. `exp414_quantile_summary.csv`
5. `exp414_backend_assessment.json`
6. `exp414_selected_rule.json`
7. `exp414_summary.json`
8. `EXP414_RESULT_REPORT_KO.md`
9. `MANIFEST.json`

구현, 독립 검토, benchmark와 parity 검증이 끝난 뒤에도 1,117 dataset 전체
실행에는 사용자의 별도 명시적 승인이 필요하다. Exp412/413은 시작하지 않으며,
통과 뒤 label-blind prediction은 Exp415, 독립 감사는 Exp416으로 별도
사전등록한다. Exp414는 TEST 성능, 자동 정상 lane 승격 또는 운영 배포를
주장하지 않는다.
