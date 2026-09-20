# Exp409 전향적 준비 상태 사전등록

## 1. 상태와 목적

Exp409는 새 데이터가 현재 존재하지 않는다. 따라서 이 실험은 성능을
검증하거나 기존 Exp408 결과를 다시 최적화하는 실험이 아니다. 목적은 앞으로
독립적인 새 데이터가 도착했을 때, 정상 TRAIN 값만으로 후보를 생성하고
false alarm을 억제하는 고정 절차가 재현 가능하고 누수 없이 실행되는지를
미리 확인하는 것이다.

프로젝트의 최종 목표는 단일 mean F1 최대화가 아니다. 자동 경보를 보수적으로
유지하고, 충분히 독립적인 근거가 겹치는 후보만 사람에게 review-only로
제시하여 사용자 신뢰를 높이는 것이다.

## 2. 동결 구성

- experiment_id: `Exp409`
- variant: `two_new_plus_existing_independent`
- alpha: `0.005`
- 전체 publication review 예산: `100`
- seed: `20260727`
- TRAIN OOF independence threshold: `0.90`
- independence 확인 최소 OOF 행: `20`
- disposition: `review-only`
- performance_evaluation: `not_evaluated`
- prospective_validated: false
- deployment_eligible: false
- operational_policy_changed: false

이 값은 TEST 결과, 후보 수, 과거 family별 성능 또는 smoke 결과를 본 뒤 바꾸지
않는다. 변경이 필요하면 Exp409를 수정하지 않고 새 실험 번호와 별도
사전등록을 사용한다.

## 3. 입력 및 소스

입력은 값만 읽는 SQLite 경로와 해시 검증된 frozen lane으로 제한한다. 각
데이터셋에서 아래 13개 source status를 반드시 기록한다.

- 등록된 기존 FeatureBlock 9개
- fresh CPU tri-view 1개
- fresh synthetic anomaly 모델 3개

Exp403-408 checkpoint와 Exp408 evaluation CSV를 재사용하지 않는다. 기존
FeatureBlock은 등록 신원, source hash, TRAIN/TEST 길이와 유한성을 검증한 뒤
사용한다. tri-view와 synthetic 모델은 현재 정상 TRAIN 값에서 새로 fitting 및
OOF calibration한다.

TRAIN 정상 행이 50개 미만이면 learned source 4개는
`n_train_below_50`으로 명시적으로 abstain한다. 허용된 사유가 아닌 결측,
비정상 길이, NaN, source identity 불일치는 자동 대체하지 않고 실패한다.

## 4. 누수 방지 경계

- prediction에서 TEST label, 실제 이상 위치, TEST 평가 metric을 읽거나
  전달하지 않는다.
- TEST 결과로 alpha, threshold, source, review 예산을 재선택하지 않는다.
- 과거 family별 TEST 성능을 source 선택에 사용하지 않는다.
- TEST-derived ordering을 사용하지 않는다.
- tail replacement를 사용하지 않는다.
- mean_combined_f1은 human-assisted 지표로만 취급하며 Exp409 readiness
  publication에는 계산하지 않는다.

SQLite prediction loader는 `values_blob`, split, instance order만 사용하며 SQL
trace가 TEST label 또는 평가 접근을 탐지하면 publication을 차단한다. 정상
TRAIN 행 수를 세는 별도 selector 쿼리는 대표 smoke tier 선택에만 쓰고 TEST
순서나 TEST truth를 받지 않는다.

## 5. Lane 및 후보 정책

Exp291/Exp292에서 동결된 Hard / Standard / Priority membership은 입력과
출력에서 정확히 같아야 한다. Priority review와 Shadow 후보는 자동 경보로
승격하지 않는다. 후보는 frozen Hard 밖에 있어야 하며, 동결 variant에서
tri-view, synthetic, 기존 evidence가 independence collapse 후 요구되는
독립 지지를 만족할 때만 생성한다.

전체 데이터셋을 합친 후보가 100개를 넘으면 top-100을 고르지 않는다.
publication 전체를 차단하고 후보 artifact를 쓰지 않는다.

## 6. 이번 승인 범위와 판정

승인 범위는 TRAIN 정상 행 수의 네 eligibility tier를 포함하는 대표
데이터셋 8개의 smoke뿐이다. 1,117개 전체 파이프라인은 실행하지 않는다.

같은 frozen 입력으로 batch를 메모리에서 두 번 조립한다. 다음이 byte-level
canonical 표현에서 모두 같아야 readiness gate가 통과한다.

- candidate JSONL SHA256
- 데이터셋별 13개 source output SHA256
- Hard / Standard / Priority lane payload
- 데이터셋 TRAIN/TEST value SHA256
- canonical provenance 입력

통과는 실행 준비와 재현성을 뜻할 뿐 미래 성능, 전향 검증, 배포 적격 또는
운영 정책 변경을 뜻하지 않는다. 실제 새 데이터가 도착하면 같은 frozen
정책으로 별도의 prospective evaluation을 수행해야 한다.

## 7. Publication 원칙

Git worktree가 깨끗하지 않으면 실행을 차단한다. 모든 payload와 hash를 임시
디렉터리에서 검증한 뒤 디렉터리 단위로 한 번만 공개한다. 출력 경로가 이미
존재하면 덮어쓰지 않는다. 오류 시 publication 내부에는 일부 candidate를
남기지 않으며, 필요한 경우 바깥에 candidate-free failure JSON만 기록한다.

