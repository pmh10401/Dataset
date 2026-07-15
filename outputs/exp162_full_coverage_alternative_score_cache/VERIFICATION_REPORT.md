# Exp162 Full-Coverage Alternative Score Cache 검증 보고서

## 결론

Exp162는 frozen population 1,117개 데이터셋 전체에 대해 다음 세 TRAIN-normal 기반 점수원을 동일한 고정 설정으로 계산했습니다.

1. 미분·스펙트럼 통계 점수
2. 부분 형태 일치 점수
3. 구간 분포 점수

전체 cache 구축과 독립 전수 검증이 통과했습니다. 이 결과는 Exp163 후보 복구 실험의 입력으로 사용할 수 있지만, 자체적으로 이상 탐지 성능을 증명하지는 않습니다.

## 실행 결과

- 대상 데이터셋: `1,117`
- 성공 cache: `1,117`
- 누락: `0`
- 오류: `0`
- coverage: `100.00%`
- 새 계산: `1,117`
- checkpoint 재실행 재사용: `1,117/1,117`
- 전체 실행 시간: `284.28초` (약 4분 44초)
- workers: `7`

## 고정 해시

- Exp160 population file SHA256: `ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d`
- Dataset name list SHA256: `18aa5649e2b6a8b2240d6ba50995442771ebb9c0d351af8a1ceb7a70559ed8d1`
- Database SHA256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Source configuration SHA256: `889799817fe4984c7996717a88defea3417edd7679d9ed33c804a663fa8ae413`

## 독립 전수 검증

모든 1,117개 NPZ 파일을 runner와 별도의 read-only 집계로 다시 검사했습니다.

- manifest dataset 중복: `0`
- cache file 누락: `0`
- manifest SHA256 불일치: `0`
- 필수 score array 누락: `0`
- NaN/Inf array: `0`
- metadata dataset 불일치: `0`
- TRAIN/TEST row count 불일치: `0`
- 금지 evaluation field: `0`
- TEST label 포함 flag: 모두 `false`
- evaluation metric 포함 flag: `false`
- config hash 종류: `1`

각 cache에는 다음 여섯 배열만 저장됩니다.

- `spectral_train`, `spectral_test`
- `shapelet_train`, `shapelet_test`
- `interval_train`, `interval_test`

## 데이터 분포

- TRAIN count: 최소 `1`, 중앙값 `40`, 최대 `4,261`
- TEST count: 최소 `50`, 중앙값 `50`, 최대 `5,400`
- TRAIN-derived target length: 최소 `15`, 중앙값 `315`, 최대 `2,048`
- `n_train < 5`: `162`개
- 세 TRAIN score가 모두 0인 데이터셋: `Phoneme_normal_39` 1개 (`n_train=1`)
- 모든 TEST score가 0인 데이터셋: `0`개
- 전체 cache 용량: `5,275,880 bytes`

`n_train < 5` 데이터셋도 score cache는 생성했지만, 이후 calibration에서는 기존 원칙대로 abstention 대상입니다.

## Codex 검토에서 수정한 문제

Antigravity 1차 구현 후 full run 전에 다음을 수정했습니다.

1. variable-length row를 `np.vstack`하기 전에 TRAIN median 길이로 개별 정렬하도록 수정했습니다.
2. 부분 형태 일치 점수의 resampling 길이 상한 `512`를 실제 코드에 적용했습니다.
3. direct CLI가 `PYTHONPATH=.` 없이 실행되도록 repository path bootstrap을 추가했습니다.
4. cache validator가 여섯 배열의 TRAIN/TEST row count를 metadata와 대조하도록 강화했습니다.
5. NPZ 검증에서 `allow_pickle=False`를 사용하도록 수정했습니다.
6. forbidden metadata key 검사를 대소문자와 공백 표현에 독립적으로 강화했습니다.
7. checkpoint 재사용 시 config, DB, population-list, dataset, file hash를 모두 검증하도록 수정했습니다.
8. DB SHA256과 population-list SHA256을 contract와 summary에 기록했습니다.
9. Shapelet distance를 64-series batch로 계산해 대형 데이터셋의 메모리 폭증을 제한했습니다.

## 테스트와 smoke

```bash
python3 -m py_compile experiments/exp162/*.py
python3 -m unittest \
  experiments/exp162/test_exp162_full_coverage_score_cache.py \
  experiments/exp162/test_exp162_structure.py
```

- unit/structure tests: `14 passed`
- 10-dataset smoke: `10/10`, 오류 `0`
- smoke resume: `10/10 reused`
- 별도 2-dataset rebuild: NPZ byte SHA256 동일
- 대형 `StarLightCurves_normal_3` probe: 성공, `39.26초`, 최대 RSS 약 `772MB`

## 수치 안정성 제한

모든 score는 finite이지만 TRAIN feature의 median absolute deviation이 거의 0인 일부 데이터셋에서 매우 큰 robust-deviation 값이 발생했습니다.

- 미분·스펙트럼 통계 점수 최대 절댓값: 약 `2.736e12`
- 부분 형태 일치 점수 최대 절댓값: 약 `2.953e11`
- 구간 분포 점수 최대 절댓값: 약 `2.724e18`

이는 cache 실패나 TEST leakage가 아니지만, Exp163에서 서로 다른 원점수를 직접 더하면 안 된다는 근거입니다. 다음 단계에서는 사전등록된 TRAIN-rank 또는 conformal 변환으로 source scale을 통일해야 합니다. 이 관찰을 보고 Exp162 설정이나 score를 다시 변경하지 않았습니다.

## 해석 제한

- TEST label, anomaly position, TP, FP, FN, F1을 읽거나 계산하지 않았습니다.
- family와 과거 family TEST 성능을 source 설정에 사용하지 않았습니다.
- TEST 길이 기반 budget과 top-N을 사용하지 않았습니다.
- Exp160 candidate-zero 여부는 cache population이나 feature 설정을 제한하지 않았습니다.
- 이 결과는 retrospective infrastructure validation입니다.
- 이상 탐지 성능, 후보 복구 효과, false alarm 변화, prospective 설비 성능은 아직 확인되지 않았습니다.
- Exp163은 별도 사전등록과 frozen prediction/evaluation 분리 후 실행해야 합니다.
