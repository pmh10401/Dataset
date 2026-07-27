# Exp397-399 사전등록

## 목적

Exp292의 기존 일반 검토 후보 806건을 Exp393 Normal Process Grammar Score로 재정렬한다. 자동 경보, 기존 우선 검토, 일반 검토 후보 집합은 변경하지 않는다.

## 고정 조건

- alpha: 0.005, 0.01, 0.02, 0.05
- global workload K: 25, 50, 100, 200
- variant: Exp292 기준, 등급 보존 문법 정렬, 문법 근거 1단계 승격, 문법 단독 대조군
- TEST 라벨은 prediction SHA256 동결 후 평가에만 사용
- TEST 결과로 variant, alpha, K를 선택하지 않음
- retrospective review-ordering audit이며 실제 설비 prospective 검증이 아님

## 모집단

- 데이터셋 1,117개
- 일반 검토 806건: TP 365, FP 441
- 기존 우선 검토 31건은 순위 모집단에서 제외하고 그대로 유지
- 자동 경보 2,218건은 그대로 유지

