# Exp472 Makima 업샘플링 보조 실험 실행 안내

Exp472는 Exp471을 수정하지 않고, 조건 E(Makima)만 새로 계산합니다. 모든 명령은 고정된 경로와 고정된 Python을 사용하며 과학 파라미터를 받지 않습니다.

```bash
PYTHON=/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python
$PYTHON scratch/run_exp472_makima_upsampling_ablation_antigravity.py readiness
$PYTHON scratch/run_exp472_makima_upsampling_ablation_antigravity.py status
```

`readiness`와 `status`는 읽기 전용입니다. `status`의 `progress`는 `official_antigravity/progress_antigravity.jsonl`이며, 마지막 내구성 진행 레코드로 완료 수·실패 수·남은 수·ETA를 표시합니다.

값 전용 smoke가 사전에 승인된 뒤에만 다음을 실행합니다. 실행기는 동결된 설정과 공개 Exp471 권한을 다시 확인하여 23개 smoke용 타입 컨텍스트와 카탈로그를 자동으로 구성합니다.

```bash
$PYTHON scratch/run_exp472_makima_upsampling_ablation_antigravity.py smoke
```

공식 실행기는 같은 방식으로 1,117개 full 타입 컨텍스트를 자동 구성합니다. E shard가 모두 검증되고 실패가 0일 때만 종료 매니페스트를 만듭니다. MPS가 선택되면 한 번에 하나의 MPS-heavy 작업만 실행합니다. CPU-only 단계는 감지한 논리 CPU 수에서 하나를 뺀 worker 수를 사용합니다.

```bash
$PYTHON scratch/run_exp472_makima_upsampling_ablation_antigravity.py run
$PYTHON scratch/run_exp472_makima_upsampling_ablation_antigravity.py verify
$PYTHON scratch/run_exp472_makima_upsampling_ablation_antigravity.py evaluate
$PYTHON scratch/run_exp472_makima_upsampling_ablation_antigravity.py report
```

`verify`는 E 1,117개와 동결된 Exp471 권한을 읽기 전용으로 다시 확인합니다. `evaluate`는 두 권한이 모두 끝난 뒤에만 TEST label 경계를 엽니다. `report`는 완성된 평가만 소비하며 E 점수 생성을 시작하지 않습니다.

예상 최종 수는 E 1,117개 shard, Exp471 재사용 11,170개 shard(10,775 scored + 395 abstained + 0 failed), 공통 support 922개입니다. 보고서는 `report_antigravity`에 15개 리프, manifest, 완료 marker 순서로 생성됩니다.

설명할 수 없는 실패가 발생하면 재시작하지 마십시오. `progress_antigravity.jsonl`과 `failures_antigravity.jsonl` 및 실패 시점의 표준 출력·오류를 보존하고 원인을 먼저 검토합니다.

## actual-trace v3 확인

E shard는 이름만 같은 입력을 재사용할 수 없습니다. 각 shard의 sealed trace는 Exp450가 실제로 반환한 결과와 label-free raw 입력의 결정적 digest에 결속됩니다. trace가 계획된 phase/source 호출을 하나라도 빠뜨리거나, 점수·observer가 예외를 내거나, `<f4` C-order 출력 증거가 아니면 shard를 만들지 않습니다. v2/raw-replay trace는 검증 대상이 아닙니다.
