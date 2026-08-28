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

`verify`는 E 1,117개와 동결된 Exp471 권한을 읽기 전용으로 다시 확인합니다. `evaluate`는 공개 평가 파일과 별도로 같은 평가 루트에 `exp472_private_label_evidence_antigravity.json`을 0600 create-only 파일로 기록합니다. 이 파일은 TEST 라벨의 canonical 순서·binary vector와 Exp471/E/context 결속을 갖지만, 보고서의 15개 리프와 보고서 manifest에는 포함하지 않습니다. 재개 시에는 source label loader를 다시 열지 않고 이 비공개 증거와 검증된 Exp471/E terminal score를 이용해 전체 표·분모·ROC/PR·AP/AUROC를 재계산하여 공개 평가 파일과 정확히 비교합니다. 비공개 증거만 남은 중단은 같은 방식으로 복구합니다. `report`는 이 저장된 평가 파일과 E/Exp471 권한만 읽으며, 평가 함수·label loader·점수 생성을 호출하지 않습니다. 파일이 없거나 검증에 실패하면 15개 보고서 리프를 만들기 전에 중단합니다.

평가 파일은 경로가 존재한다고 완료로 간주하지 않습니다. `status`의
`evaluation_present`와 `report_present`는 단순 관측값이며, 실제 완료 판단은
`evaluate` 또는 `report`가 canonical bytes, digest, 1,117개 데이터셋과 11개
기법의 표, 부분집합·곡선·진단 요약을 다시 검증한 결과로만 합니다. 동시에
`evaluate`를 실행해도 OS 잠금 때문에 한 프로세스만 TEST label을 열 수
있습니다. 그 프로세스는 같은 디렉터리의 임시 파일을 fsync한 뒤 원자적
no-clobber 방식으로 최종 이름을 만들고, 기존 최종 파일을 교체하지 않습니다. 이 결속은 비공개 label evidence와 기존 Exp471/E terminal authority가 보존된다는 운영 경계에서 공개 평가 파일의 손상·일관된 재작성까지 탐지합니다. 모든 권한 파일을 삭제하거나 권한 있는 공격자가 비공개 evidence까지 교체하는 경우는 외부 서명 또는 WORM 저장소가 필요한 별도 위협 모델입니다.

예상 최종 수는 E 1,117개 shard, Exp471 재사용 11,170개 shard(10,775 scored + 395 abstained + 0 failed), 공통 support 922개입니다. 보고서는 `report_antigravity`에 15개 리프, manifest, 완료 marker 순서로 생성됩니다.

설명할 수 없는 실패가 발생하면 재시작하지 마십시오. `progress_antigravity.jsonl`과 `failures_antigravity.jsonl` 및 실패 시점의 표준 출력·오류를 보존하고 원인을 먼저 검토합니다.

## actual-trace v3 확인

E shard는 이름만 같은 입력을 재사용할 수 없습니다. 각 shard의 sealed trace는 Exp450가 실제로 반환한 결과와 label-free raw 입력의 결정적 digest에 결속됩니다. trace가 계획된 phase/source 호출을 하나라도 빠뜨리거나, 점수·observer가 예외를 내거나, `<f4` C-order 출력 증거가 아니면 shard를 만들지 않습니다. v2/raw-replay trace는 검증 대상이 아닙니다.

## v3 진단 표와 부분집합의 의미

`exp472_shape_diagnostics_antigravity.csv`의 한 행은 데이터셋 전체 요약이 아니라 Exp450가 실제로 수행한 한 번의 정렬 호출입니다. 따라서 각 행에는 `phase_tag`, `source_tag`, 그 호출의 `target_len`, 행 수, Makima 행 인덱스, 입력·출력 증거가 함께 기록됩니다. sealed trace에 호출이 0개이면 CSV에는 가짜 null 행을 만들지 않습니다. 그 데이터셋의 0값 요약은 평가 권한의 `diagnostic_summary`에 남습니다.

`upsampled_rows`는 검증된 trace에서 중복을 제거한 실제 `(TRAIN|TEST, row_index)` 슬롯 수입니다. 평가의 `upsampled_rows_gt_zero` 부분집합은 이 수가 0보다 큰 데이터셋으로만 만들며, 라벨이나 성능 지표로 대상을 고르지 않습니다. 반면 `makima_row_invocation_count`는 같은 슬롯이 서로 다른 경로에서 다시 Makima를 거친 경우도 모두 세는 호출 노출 횟수입니다. 두 수는 목적이 다르므로 논문 표와 해석에서 서로 바꾸어 쓰면 안 됩니다.
