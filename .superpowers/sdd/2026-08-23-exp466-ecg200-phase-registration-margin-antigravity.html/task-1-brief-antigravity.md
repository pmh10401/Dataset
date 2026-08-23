## Task 1 — 권한·계약·문서 골격

Files: Create package config/contracts, two tests, preregistration/runbook/implementation report HTML and map builder. Modify three ledgers and dashboard.

Interfaces: Produce `FROZEN_CONFIG_ANTIGRAVITY`, `Exp466PhaseEvidenceAntigravity`, `Exp466FoldResultAntigravity`, `Exp466QualificationAntigravity`.

1. **Step 1 — RED config test.**

```python
def test_frozen_config_antigravity(self):
    c = FROZEN_CONFIG_ANTIGRAVITY
    self.assertEqual((c.experiment, c.method_order, c.primary_candidate),
                     ("Exp466", ("C0", "P9", "C9"), "C9"))
    self.assertEqual((c.warp_factor, c.shared_length), (1.25, 67))
    self.assertEqual((c.per_type_auc_min, c.mean_auc_min), (0.85, 0.90))
```

2. **Step 2 — RED contract test.** Mutable/reordered mapping, non-finite margin, wrong 69-count, wrong SHA와 0이 아닌 official/label count를 각각 거부한다.

```python
valid = valid_qualification_antigravity()
with self.assertRaisesRegex(ValueError, "normal confirmation"):
    dataclasses.replace(valid, normal_confirmations=MappingProxyType({20260721: 1, 20261721: 0}))
```

3. **Step 3 — RED 확인.**

```text
python3 -m unittest \
  experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_config_antigravity \
  experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_contracts_antigravity
```

Expected: import 또는 정의 부재로 FAIL.

4. **Step 4 — 최소 GREEN 구현.**

```python
_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType({
    "experiment": "Exp466", "dataset": "ECG200", "data_split": "TRAIN_only",
    "normal_rows": 69, "interpolated_width": 140,
    "seeds": (20260721, 20261721), "folds": 5,
    "onset": 56, "suffix_length": 84,
    "warp_factor": 1.25, "shared_length": 67,
    "method_order": ("C0", "P9", "C9"), "primary_candidate": "C9",
    "per_type_auc_min": 0.85, "mean_auc_min": 0.90,
    "approved_design_sha256": "eddb057724e0ef21402c07b9f362bc6b260fc917a9f6bfdcb130f118b30a5c66",
})
```

5. **Step 5 — HTML/ledger/map 골격.** 세 Exp466 truth HTML과 ledger에는 `planned_implementation_not_started`, official/result/hash 0 또는 null만 기록한다. map은 Task 1 caller/effect/test를 검증한다.

6. **Step 6 — GREEN 두 번·commit·fresh review.**

```text
python3 -m unittest experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_config_antigravity experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_contracts_antigravity
python3 -m unittest experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_config_antigravity experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_contracts_antigravity
git diff --check
git commit -m "feat: freeze Exp466 authority and contracts"
```

### Binding global constraints

- 모든 새 파일명은 `antigravity`를 포함한다.
- ECG200 TRAIN 정상 69개만 사용한다. TEST 파일·라벨 query·공식 명령 수는 항상 0이다.
- TRAIN MD5 `808875aeac1c358a90c630fe369327c4`, 입력 `100×96`, 정상 `69`, 보간 폭 `140`을 바꾸지 않는다.
- seed `(20260721, 20261721)`, 5 fold, onset `56`, suffix `84`, warp `1.25`, 공유 길이 `67`을 바꾸지 않는다.
- method order는 `(C0, P9, C9)`, 유일한 primary candidate는 `C9`다. P9는 설명용이다.
- C9 gate는 유형별 AUC `≥0.85`, 평균 `≥0.90`, period AUC가 매 fold C0보다 엄격히 높음, gradual AUC가 매 fold C0 이상, period confirmation 매 fold 1개 이상, seed별 정상 confirmation 정확히 0, finite, private 2회 exact equality다.
- warp grid, threshold/alpha 탐색, C8 임계값 변경, unconditional max, Exp461 FFT+ACF, TEST, routing, champion·pending·promotion 변경은 금지한다.
- full run 전에 CPU/MLX 동일 입력·seed parity와 단계 전체 속도를 측정한다. 후보 index가 같고 MLX가 `1.1×` 이상 빠를 때만 MLX를 사용한다.
- Task 1은 점수 계산이나 private/official 실행을 하지 않는다. 기존 Exp465 구현을 읽어 같은 검증 강도와 파일 구조를 따르되 Exp465 코드는 변경하지 않는다.
- 현재 dashboard의 승인 체크와 Task 1 진행 상태는 사용자의 승인 사실이므로 보존한다.
- Code-map baseline은 `docs/codemap/codemap_exp465_antigravity.{html,json,lock}`이다. 새 Exp466 map이 Task 1의 caller, effect, test를 답하게 한다.
