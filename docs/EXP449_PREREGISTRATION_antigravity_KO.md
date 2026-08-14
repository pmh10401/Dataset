# Exp449 재현 가능한 재기준선 사전등록

```text
experiment = Exp449
approved_design_commit = 26830622
policy = hard_plus_stable (Candidate C)
dataset_count = 1117
test_row_count = 195900
implementation_status = implementation_verified_awaiting_smoke_authorization
```

## 고정된 과학적 정의

Exp449는 기존 paper 과학 primitive를 바꾸지 않는 label-free proposal freeze다.
Task4 전처리는 **TRAIN-only**이고, Task5 길이 선택은 **TRAIN+TEST 값만을
사용하는 label-free transductive** 동작이다. TEST label, historical acceptance,
historical oracle, parity replay는 Exp449의 입력ㆍ성공 조건ㆍrouting rule이 아니다.
특히 historical coordinate equality gate는 `false`이며, 과거 좌표는 진단용으로도
새 성공 기준을 만들지 않는다.

Task4/5의 dataset-local 결과가 frozen catalog와 정확한 bijection일 때만
population-global Task6, Task7, Candidate C (`hard_plus_stable`)와 typed shard/
outer manifest 단계로 진행한다. full mode에서는 정확히 1,117 dataset이어야 하며,
부분 coverage에서는 Task6/7을 호출하지 않는다. Task4/5 실패는 최대 두 wave로
회복하고, 끝까지 불완전하면 `proposal_score_incomplete`이며 proposal score를
만들지 않는다.

## 사전등록 smoke A/B

두 smoke는 다음의 정렬된 23 dataset 이름을 공유한다.

```text
ChlorineConcentration_normal_3
DodgerLoopDay_normal_1
DodgerLoopDay_normal_2
DodgerLoopDay_normal_4
DodgerLoopDay_normal_5
DodgerLoopDay_normal_6
DodgerLoopDay_normal_7
DodgerLoopGame_normal_1
DodgerLoopGame_normal_2
DodgerLoopWeekend_normal_1
DodgerLoopWeekend_normal_2
MelbournePedestrian_normal_1
MelbournePedestrian_normal_10
MelbournePedestrian_normal_2
MelbournePedestrian_normal_3
MelbournePedestrian_normal_4
MelbournePedestrian_normal_5
MelbournePedestrian_normal_6
MelbournePedestrian_normal_7
MelbournePedestrian_normal_8
MelbournePedestrian_normal_9
Phoneme_normal_39
PigAirwayPressure_normal_1
```

Smoke A와 B는 다음처럼 서로 독립된 scientific/operations root를 쓴다. checkpoint,
record, inode, manifest 또는 operations receipt를 공유할 수 없다.

```text
smoke_a_scientific = /Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_a_antigravity_20260814
smoke_a_operations = /Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_a_antigravity_20260814_ops_antigravity
smoke_b_scientific = /Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_b_antigravity_20260814
smoke_b_operations = /Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_b_antigravity_20260814_ops_antigravity
official_scientific = /Users/mac/Documents/Dataset/paper_proposal_exp449_official_antigravity_20260814
official_operations = /Users/mac/Documents/Dataset/paper_proposal_exp449_official_antigravity_20260814_ops_antigravity
```

동일 CPU에서 A/B가 모두 독립적으로 complete 검증을 통과한 뒤, 모든 policy의
coordinate는 **exact equality**여야 한다. continuous score의 all-policy 최대 절대
차이는 `<= 1e-12`여야 한다. 이 조건은 historical coordinate gate가 아니라 새 두
Exp449 실행의 재현성 기준이다.

## CPU와 가속 정책

새 공식 실행의 dataset 병렬 worker 수는 탐지한 logical CPU 수에서 1을 뺀 값,
`max(1, logical_cpu_count - 1)`이다. detected logical CPU와 실제 workers는
provenance에 기록한다. Task4/5가 outer CPU 병렬일 때 inner numerical thread는 1이다.

PyTorch, MLX, MPS를 각각 평가한다. 현재 canonical 경로는 CPU이며, MLX/MPS-heavy
job은 한 번에 하나만 실행한다. exact dense distance matrix, batched projection,
tensor feature fusion, neural embedding은 가속 후보일 수 있으나 frozen algorithm,
feature/score/threshold/routing을 바꾸어서는 안 된다. 후보를 canonical로 바꾸려면
동일 input/seed deterministic CPU 대비 smoke에서 finite output, preregistered alpha마다
동일 candidate index, score delta `<= 1e-5`, transferㆍcompilationㆍsynchronization을
포함한 end-to-end speedup `>= 1.1x`를 모두 입증해야 한다. MLX/mlx-metal version,
device, timing, parity, CPU fallback 이유를 manifest에 기록한다. 어느 하나라도
실패하면 CPU fallback을 유지한다.

## label firewall과 실행 권한

proposal freeze와 independent verification 동안 TEST label은 열 수 없다.
`test_labels_opened=false`, `official_label_access_count=0`이 계속 요구된다. 이 문서,
구현 완료, self-hash, local artifact, 또는 root 부재 자체는 실행 권한이 아니다.
Smoke A, Smoke B, official 실행은 각각 이후 별도 review와 명시적 사용자 승인이
필요하며, 순서는 A 승인ㆍ검증, B 승인ㆍ검증ㆍ비교, official 승인이다.
