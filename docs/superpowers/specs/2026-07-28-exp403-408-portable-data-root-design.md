# Exp403-408 이식 가능한 데이터 루트 설계

## 목적

Exp403-408을 다른 macOS 사용자 계정에서 복원할 때 기존 연구 계약과
체크포인트를 보존하면서 데이터베이스와 동결 artifact 경로를 명시적으로
재지정할 수 있게 한다.

## 범위

- 복원 데이터를 `/Users/mac/Documents/Dataset`에 배치한다.
- `TIMESERIES_DATA_ROOT` 환경변수로 Exp403-408 데이터 루트를 재정의한다.
- 환경변수가 없으면 기존 `/Users/minho/Documents/Dataset`를 사용한다.
- Exp403-408의 DB, Exp291/292 lane, feature source, 기본 출력 경로가 같은
  데이터 루트를 사용한다.
- 기존 prediction, consensus, DB 및 checkpoint 파일은 수정하지 않는다.
- 전체 Exp403-408 파이프라인은 실행하지 않는다.

## 경로 계약

하나의 경로 해석 함수를 Exp403-408 경로 설정의 기준으로 사용한다.

1. `TIMESERIES_DATA_ROOT`가 설정되어 있으면 해당 값을 사용자 홈과 환경변수
   확장 후 절대 경로로 해석한다.
2. 설정되어 있지 않으면 역사적 기본 경로
   `/Users/minho/Documents/Dataset`를 반환한다.
3. 데이터 루트에서 DB, Exp291/292 lane, feature source 및 기본 Exp403-408
   출력 경로를 파생한다.
4. 잘못된 경로는 import 시 조용히 대체하지 않고, 실제 파일을 읽는 기존
   검증 지점에서 명시적으로 실패한다.

## 호환성

- 기존 Mac에서는 환경변수 없이 현재 동작을 유지한다.
- 새 Mac에서는 다음 설정으로 복원 데이터를 사용한다.

  ```bash
  export TIMESERIES_DATA_ROOT=/Users/mac/Documents/Dataset
  ```

- checkpoint input hash와 publication hash 계산은 파일 내용에 기반하므로,
  동일 파일을 새 경로에서 읽는 것만으로 prediction 정책이나 연구 결과를
  변경하지 않는다.
- TEST label, 실제 이상 위치, 과거 family별 TEST 성능을 prediction 경로에
  추가하지 않는다.

## 테스트 전략

1. 환경변수 미설정 시 역사적 기본 경로가 유지되는 테스트를 추가한다.
2. 환경변수 설정 시 DB, lane, feature source, 기본 출력 경로가 모두 같은
   재정의 루트에서 파생되는 테스트를 먼저 추가하고 실패를 확인한다.
3. 최소 구현 후 새 경로 테스트를 통과시킨다.
4. 복원 데이터 경로를 설정하고 Exp403-408 전체 단위 테스트를 실행한다.
5. DB quick check, DB SHA256, prediction SHA256, consensus SHA256 및 stage별
   1,117 checkpoint 존재를 다시 검증한다.

## 사전등록 문서 누락 처리

`docs/EXP403_408_PREREGISTRATION_KO.md`는 현재 브랜치에 없다. Git 이력과
GitHub에서 동일 경로의 원본을 찾는다. 원본이 확인된 경우에만 그대로
복구한다. 원본이 없으면 추정 내용을 새 사전등록으로 만들지 않고 누락
상태를 보고한다.

## 완료 조건

- 데이터가 `/Users/mac/Documents/Dataset`에 존재한다.
- 환경변수 기반 경로 테스트와 기존 Exp403-408 단위 테스트가 통과한다.
- 복원된 DB와 핵심 publication hash가 인계 매니페스트와 일치한다.
- 기존 checkpoint가 경로 변경만으로 재계산되지 않음을 검증한다.
- 전체 장시간 파이프라인을 실행하지 않는다.
