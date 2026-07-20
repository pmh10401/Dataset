# Exp254 sktime Structural Features Design

## 목적

Exp254는 Exp252의 단순 통계 특징과 다른 구조 정보를 추출한다. 자동 경보를 즉시
변경하지 않고, RandomInterval, MatrixProfile, SAX, SFA가 기존 Exp236/Exp253
경보를 얼마나 정확히 재확인하고 남은 일반 검토 후보에서 유용한 추가 근거를
제공하는지 후향 평가한다.

## 쉬운 설명

- **RandomInterval**: 시계열의 여러 구간을 무작위로 잘라 구간별 모양을 본다.
- **MatrixProfile**: 시계열 안에서 반복되거나 유난히 다른 구간을 찾는다.
- **SAX**: 연속 값을 작은 기호 문장으로 바꾸어 전체적인 형태 변화를 본다.
- **SFA**: 짧은 구간의 주파수 모양을 기호 단어로 바꾸어 반복 패턴을 본다.

네 방식은 같은 시계열을 서로 다른 관점으로 본다. 결과가 나온 뒤 Exp252와 함께
자동 경보 신뢰도 등급, 검토 우선순위, 제한적 승격 가능성을 판단한다.

## 고정 구성

- 실험 ID: `experiment_254_sktime_structural_features`
- 전체 1,117개 데이터셋
- 정상 TRAIN만 feature fitting, scaling, score calibration에 사용
- TEST 라벨은 동결된 prediction 평가에만 사용
- seed: `20260723`
- 정상 TRAIN 최소 개수: 5
- working length: TRAIN에서 정한 정렬 길이를 사용하되 최대 256으로 고정 압축
- score: KNN-3 local-gap
- calibration: 5-fold TRAIN OOF empirical conformal
- alpha: 0.005, 0.01, 0.02, 0.05
- CPU 7 workers, MPS 미사용

## 특징 블록

1. `random_interval_224`
   - `RandomIntervals(n_intervals=32)`
   - 각 구간에서 mean, median, min, max, std, q25, q75
   - 224차원
2. `matrix_profile_16`
   - working length의 약 1/16 및 1/8 window
   - 각 profile에서 min, mean, std, q25, q50, q75, q90, max
   - 16차원
3. `sax_transition_72`
   - 32-symbol SAX, alphabet 8
   - symbol histogram 8개와 transition histogram 64개
   - 72차원
4. `sfa_hash_128`
   - TRAIN에만 fit한 공식 SFA transform
   - label-dependent ANOVA 미사용
   - dictionary words를 deterministic FeatureHasher 128차원으로 변환
5. `structural_combined_440`
   - 위 네 블록을 순서대로 결합

SFAFast는 현재 Python 3.14 환경에서 native unboxing 오류가 확인되어 사용하지 않는다.
공식 `SFA`와 scikit-learn `FeatureHasher`를 조합한다.

## 데이터 흐름

```text
DB -> 정상 TRAIN/TEST values 로드 -> TRAIN 길이 기준 정렬 -> 최대 256 압축
   -> 네 구조 특징 추출 -> TRAIN 기반 결측 보정 및 robust scaling
   -> KNN local-gap -> TRAIN OOF conformal candidate -> prediction hash 동결
   -> TEST label 사후 평가 -> Exp236/Exp253 lane 교차 분석
```

## 자동 경보 판단 기준

Exp254 자체는 자동 경보를 변경하지 않는다.

- 기존 Hard 재확인은 count 100 이상, precision 89.546% 이상을 연구 gate로 보고한다.
- Exp253 이후 남은 Standard 후보는 incremental precision 80% 이상인지 진단한다.
- lane 밖 후보는 자동 경보로 승격하지 않는다.
- 여러 특징의 합의는 별도로 보고하지만 TEST 결과를 보고 운영 규칙을 선택하지 않는다.

## 실패 처리

- 한 데이터셋이라도 source 생성에 실패하면 전체 성공으로 처리하지 않는다.
- SFA vocabulary가 비어도 128차원 0 벡터로 안전하게 변환한다.
- MatrixProfile window는 series length보다 작도록 검증한다.
- 비유한 특징 열은 TRAIN median으로 보정하고 전부 비유한 열은 0으로 비활성화한다.
- prediction에 TEST label, TP, FP, F1을 포함하지 않는다.

## 산출물과 해석 제한

새 output 폴더에 cache, frozen prediction, 평가 CSV, 독립 재집계, 한국어 보고서,
manifest를 저장하고 대시보드에 Exp254를 등록한다. 반복 관찰된 TEST를 사용하는
retrospective score-source qualification이며 prospective 또는 배포 검증이 아니다.
