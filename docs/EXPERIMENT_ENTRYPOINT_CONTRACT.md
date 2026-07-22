# Experiment entrypoint contract

실험 진입점은 실행 위치와 관계없이 저장소의 `experiments` 패키지를 찾을 수 있어야 한다.

## Queue execution

`run_rank_experiments_sequential.py`는 모든 자식 실험에 저장소 루트를 `PYTHONPATH`의 첫 항목으로 전달한다. 따라서 순차 큐는 래퍼의 현재 작업 디렉터리에 의존하지 않는다.

## Direct execution

`scratch/run_*.py`가 `experiments` 패키지를 직접 import한다면 해당 import보다 먼저 아래 부트스트랩을 실행한다.

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

새 래퍼는 기존 검증된 래퍼를 복사해 만들거나 `python3 -m ...` 형태의 모듈 실행을 사용한다.

## Regression check

다음 테스트는 부트스트랩 누락, 큐 환경 누락, 저장소 밖에서의 직접 실행 실패를 검사한다.

```bash
python3 -m unittest scratch.test_experiment_entrypoint_contract -v
```
