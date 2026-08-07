# Exp423 실행 런북

실행 전에는 MPS가 사용 가능한지, Git HEAD가 깨끗한지, DB와 Exp291/292 입력 hash가
고정된 manifest와 일치하는지 확인한다. output과 checkpoint는 매 시도마다 새로운
경로를 사용하고 기존 부분 결과는 삭제하지 않는다.

모델 학습과 TEST 값 추론은 한 개의 MPS-heavy 프로세스로 순차 실행한다. 진행은
checkpoint root의 `progress.json`에서 dataset별로 확인할 수 있고, 로그에는 시작
상태와 종료 manifest가 남는다. 사용자가 중단을 요청하면 해당 프로세스만 중단하고
부분 checkpoint를 보존한다.

완료 후 `manifest.json`의 `state=review_only`, `test_labels_opened=false`,
`mps_requested=true`, `official_prediction=false`를 확인한다. 그 다음에만 별도
post-hoc 평가 작업에서 TEST label을 사용해 혼동행렬을 만들 수 있다.
