# SDD ledger — plan: docs/superpowers/plans/2026-08-22-exp465-ecg200-c8-external-validation-antigravity.html

Preflight: existing isolated worktree verified; Exp464 codemap current; Task 1 RED test confirmed missing config module.
Task 1: complete (commits b2b8fb1..dbcbab7, Ox Alpha Ultra review clean)

## Task 2 preflight interface scan

<table><thead><tr><th>Producer</th><th>Consumer</th><th>Shared interface/file</th><th>Finding</th></tr></thead><tbody>
<tr><td>Task 1 frozen config</td><td>Task 2 source loader</td><td>ECG200 TRAIN MD5, 100×96, labels 69/31</td><td>일치. Task 2는 frozen 값을 읽고 변경하지 않는다.</td></tr>
<tr><td>Task 2 source loader</td><td>Task 2 interpolation</td><td>owned C-contiguous read-only float64 (69,96)</td><td>일치. 보간 입력 계약으로 그대로 사용한다.</td></tr>
<tr><td>Task 2 interpolation</td><td>Task 3 scoring</td><td>owned C-contiguous read-only float64 (69,140)</td><td>일치. smoothing, normalization, onset 재선택은 금지한다.</td></tr>
<tr><td>Task 2 modules/tests/docs</td><td>Exp465 codemap and governance</td><td>callers, effects, covering tests, HTML checkbox truth</td><td>새 노드와 edge를 같은 변경 경계에서 추가해야 한다.</td></tr>
<tr><td>Task 2 Step 5</td><td>Task 2 Step 6</td><td>official TRAIN-only smoke provenance</td><td>일치. TEST-named path는 loader가 거부하고 TEST는 다운로드·열기·조회하지 않는다.</td></tr>
</tbody></table>

Task 2: complete (base dbcbab78, functional bce0161f, map ccbc80d4, fresh Ox Alpha Ultra SPEC PASS / QUALITY APPROVED)

## Task 3 preflight interface scan

<table><thead><tr><th>Producer</th><th>Consumer</th><th>Shared interface/file</th><th>Finding</th></tr></thead><tbody>
<tr><td>Task 2 interpolation</td><td>Task 3 scorer</td><td>owned read-only float64 <code>(69,140)</code></td><td>일치. Task 3는 입력을 바꾸지 않는다.</td></tr>
<tr><td>Exp464 scorer</td><td>Exp465 projection</td><td>9-method <code>FoldMethodScoresAntigravity</code>의 13필드</td><td>Exp464 클래스는 9개 방법을 강제하므로 직접 재구성할 수 없다.</td></tr>
<tr><td>Exp465 frozen config</td><td>Exp465 projection result</td><td><code>("C0","C6","C8")</code></td><td>Exp465-local frozen/slots 13필드 결과로 고정한다.</td></tr>
<tr><td>Task 3 scorer</td><td>Task 4 qualification</td><td>세 방법의 normal·네 합성 score와 confirmation/evidence/provenance</td><td>Exp464 값을 exact projection하며 새 수학을 추가하지 않는다.</td></tr>
<tr><td>Task 3 docs/tests</td><td>Exp465 governance/map</td><td>체크박스, 원장, callers/effects/tests</td><td>fresh review 전 heading/Step 6은 unchecked로 둔다.</td></tr>
</tbody></table>

Ruling: Exp465-local immutable 13-field projection type을 정의한다 — Exp464 결과 클래스는 9-method invariant라 세 방법 결과를 표현할 수 없고, 별도 projection type이 명세의 exact 선택 경계다 — 잘못되면 Task 4 소비 계약을 다시 맞춰야 한다.
Ruling: Task 3 MLX는 조건 1 실패로 CPU fallback이다 — 공식 Exp464 CPU scorer를 다시 쓰지 않고는 exact stage를 MLX로 표현할 수 없다 — 잘못되면 별도 parity/speed benchmark를 추가해야 한다.

Task 3: complete (implementation 8eea61c8..64f08515, closeout 98cee2e8..43368dc7, fresh Ox Alpha Ultra review SPEC PASS / QUALITY APPROVED)

## Task 4 preflight interface scan

<table><thead><tr><th>Producer</th><th>Consumer</th><th>Shared interface/file</th><th>Finding</th></tr></thead><tbody>
<tr><td>Task 3 scorer</td><td>Task 4 qualification</td><td>C0/C6/C8 normal·네 변화 score, confirmation, evidence, provenance</td><td>일치. Task 4는 raw score에서 파생값을 다시 계산한다.</td></tr>
<tr><td>Task 4 qualification</td><td>evaluation/dashboard/artifacts</td><td>10 seed-fold × 3 method metric rows와 seed별 정상 confirmation</td><td>C8만 주 판정이며 C0/C6은 비교·설명용이다.</td></tr>
<tr><td>artifacts</td><td>semantic replay</td><td>canonical seven-leaf inventory</td><td>hash만 확인하지 않고 AUC·진단·요약·판정·대시보드를 재계산해야 한다.</td></tr>
<tr><td>Task 4 modules/tests</td><td>Exp465 code map/governance</td><td>callers, effects, covering tests, HTML checkbox truth</td><td>새 노드와 edge를 같은 변경 경계에 추가한다.</td></tr>
</tbody></table>

Ruling: Task 4는 CPU 집계 경로를 유지한다 — 새 score-source 수학이 아니라 69-row AUC·비교·canonical JSON의 작은 결정론적 집계이며 MLX용 재작성은 알고리즘 경계와 1.1배 속도 조건을 정당화하지 못한다 — 잘못되면 Task 5 전 동일 입력 parity/speed gate를 추가한다.

Task 4: complete (b76c04a3, 317050c0, dbea6a1f; final fresh review SPEC PASS / QUALITY APPROVED; exact 69 OOF, seed Wilson, C0 input replay, C6/C8 raw replay verified).
