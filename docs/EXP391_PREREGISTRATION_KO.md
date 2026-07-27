# Exp391 FN/Zero-F1 회수 가능성 감사 사전등록

## 목적

동결된 Exp389/390 자동 경보의 FN 2,003건과 Zero-F1 356개 데이터셋을
원인별로 분리하고, 동결된 기존 근거가 놓친 이상 위치에 도달하는지 측정한다.

## 고정 조건

- Alpha: `0.005, 0.010, 0.020, 0.050, 0.100, 0.200`
- 독립 근거 합의 수: `1, 2, 3`
- 기존 Hard, Standard, Priority, exact-normal lane은 변경하지 않는다.
- Hard 및 exact-normal 인덱스는 회수 후보에서 제외한다.
- TEST 라벨은 모든 동결 입력의 SHA256 검증 이후 사후 평가에만 사용한다.
- TEST 결과로 Alpha, 합의 수, 점수원, family 또는 크기 규칙을 선택하지 않는다.

## 해석

Exp391은 retrospective counterfactual 진단이다. 결과가 좋아도 자동 경보에
적용하지 않으며, 별도 Exp392 사전등록과 신규 잠금 데이터 검증이 필요하다.
