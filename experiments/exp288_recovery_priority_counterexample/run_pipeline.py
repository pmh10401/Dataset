from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from .audit import run_audit


OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp288_recovery_priority_counterexample")
EXPERIMENT_ID = "experiment_288_recovery_priority_counterexample_audit"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args()
    root = args.output_root; root.mkdir(parents=True, exist_ok=True); (root / "logs").mkdir(exist_ok=True); started = time.time()
    def publish(status, done):
        payload = build_status(EXPERIMENT_ID, status, "288", done, 1117, None, errors=[], timestamps=[], stdout_log=root / "logs" / f"{EXPERIMENT_ID}.log", started_unix=started)
        payload.update({"autonomous_metric_scope": "Exp253 Hard unchanged", "review_metric_scope": "Retrospective FP counterexample diagnosis"}); publish_status(payload, output_root=root)
    publish("running", 0); run_audit(root); publish("completed", 1117)


if __name__ == "__main__": main()
