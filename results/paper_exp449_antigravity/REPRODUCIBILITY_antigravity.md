# 재현 및 무결성 가이드

## 고정 연구 경계

- 공식 실험: Exp449
- 데이터베이스 SHA-256:
  `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- 데이터셋: 1,117개
- TEST 행: 195,900개
- 기준선: 6개
- 주 프로토콜: matched alert budget
- conformal alpha: `0.005`, `0.01`, `0.02`
- bootstrap: 2,000회, seed `20260812`

## 원본 점수 manifest

- Classical:
  `c546c841c86c8f488e22d0b0b844a81bc17cdd36b1bfcec3b41869d04a774355`
- Deep SVDD:
  `b1c86dfab4287adb68dae4af05b17279897229138ad3137df8ef418d1ab2f06d`
- DROCC MPS:
  `999932ee7e41781fd8398bf8c0d386e05a95b29d9b51d10838046f507e256481`
- Proposal manifest:
  `0c179a84b73167b081249255ee3d6e7dc72a7099f9f5ccf211d84ad187a517af`

## 권위 산출물

`official_comparison_artifact_sha256_antigravity.json`은 최초 공식 비교의 핵심
네 파일만 검증한다. 이후 추가된 ROC/PR 좌표, HTML, 그림, 논문용 파생 표를
포함한 이 GitHub 패키지 전체의 검증에는 최상위
`CHECKSUMS_antigravity.sha256`을 사용한다.

## 검증 절차

패키지 루트에서 다음을 실행한다.

```bash
shasum -a 256 -c CHECKSUMS_antigravity.sha256
```

핵심 개수도 함께 확인한다.

```bash
python3 - <<'PY'
import csv, json
from pathlib import Path
p = Path('data_antigravity')
s = json.loads((p / 'official_comparison_summary_antigravity.json').read_text())
c = json.loads((p / 'official_common_support_roc_pr_curves_antigravity.json').read_text())
with (p / 'official_comparison_dataset_metrics_antigravity.csv').open() as f:
    rows = sum(1 for _ in csv.DictReader(f))
assert s['metadata']['dataset_count'] == 1117
assert s['metadata']['test_row_count'] == 195900
assert s['dataset_metric_row_count'] == rows == 26808
assert c['common_dataset_count'] == 922
assert c['test_row_count'] == 186150
assert len(c['methods']) == 7
print('verified')
PY
```

## 결과 재작성 순서

1. `official_comparison_summary_antigravity.json`에서 표 1, 5, 6을 생성한다.
2. summary와 common-support curve JSON을 결합해 표 2를 생성한다.
3. bootstrap JSON에서 `protocol=matched_alert_budget` 행만 선택해 표 3을 만든다.
4. summary의 `author_ablation`에서 표 4를 만든다.
5. ROC/PR는 common-support JSON의 exact curve rows를 사용한다. 표시용
   0.5%p grid는 그림 렌더링에만 쓰며 AP/AUROC는 exact 원본 값이다.

## 재현성 제한

- 원시 DB는 이 패키지에 포함하지 않는다.
- DROCC MPS 동등성 상태는 사용자 확인이며 기계 판독 가능한 parity 파일이 없다.
- 본 패키지는 결과 산출물을 재검증하고 논문 표를 재작성할 수 있게 하지만,
  원시 DB 없이 전체 모델 학습을 처음부터 재실행하는 패키지는 아니다.
