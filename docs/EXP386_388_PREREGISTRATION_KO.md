# Exp386-388 기존 근거 Alpha Sweep 사전등록

- 원본 코드: Git commit `930b8ce8927d18a0101c647d26e21c2cd20be800`
- Alpha: `0.005`, `0.01`, `0.02`, `0.05`, `0.10`, `0.20`
- 모집단: 기존 Exp357과 같은 1,117개
- `n_train < 5`: 기권 유지
- 기존 Hard alert: 변경 금지

Exp386은 alpha 0.005 결과가 기존 Exp357과 bit-for-bit 같은지 먼저 확인한다.
불일치하면 Exp387/388은 진행하지 않는다.

Exp387은 나머지 alpha를 모두 실행한다. `0.10`과 `0.20`은 coverage 연구 및
검토용 sensitivity이며 자동 경보 기본값이 아니다.

Exp388은 결과를 비교하지만 TEST 성능으로 alpha를 선택하지 않는다. 모든
결과는 retrospective counterfactual이며 실제 설비 prospective 검증이 아니다.
