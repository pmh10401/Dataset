# 논문 제출 체크리스트

## 내용

- [ ] 소속·저자·감사의 글 입력
- [ ] 사내 기밀 검토 완료
- [ ] “실제 설비 데이터”로 오해될 표현 제거
- [ ] full population과 common support를 명확히 분리
- [ ] Deep SVDD/DROCC coverage 82.54% 명시
- [ ] 제안 점수가 고장 확률이 아님을 명시
- [ ] paired bootstrap 부호가 `기준선−제안기술`임을 명시
- [ ] 수상 가능성 최종 판단 빈칸은 별도 심사 후 작성

## 수치

- [ ] 표 값은 파생 문서가 아니라 공식 summary JSON과 대조
- [ ] 제안기술 경보 수 2,276 확인
- [ ] 전체 모집단 1,117 / 195,900 확인
- [ ] 공통지원 922 / 186,150 확인
- [ ] ROC x축 FPR, PR x축 Recall 확인
- [ ] AP와 AUROC를 혼동하지 않았는지 확인

## 근거와 공개

- [ ] `shasum -a 256 -c CHECKSUMS_antigravity.sha256` 통과
- [ ] raw DB·개별 라벨·사내 데이터 미포함 확인
- [ ] DROCC MPS parity 근거 수준을 사용자 확인으로 표시
- [ ] 참고문헌 6개 원문 링크 확인
- [ ] 최종 PDF에 figure/table caption과 모집단 각주 추가
