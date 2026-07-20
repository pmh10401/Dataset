#!/usr/bin/env python3
"""Validate, spool, and deliver prospective Exp265 shadow predictions."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from experiments.exp265_shadow_validation.config import SPOOL_PATH
from experiments.exp265_shadow_validation.contracts import validate_run_input
from experiments.exp265_shadow_validation.exporter import PermanentExportError, RetryableExportError, push_prediction
from experiments.exp265_shadow_validation.routing import route_shadow_run
from experiments.exp265_shadow_validation.spool import ShadowSpool


def enqueue_file(spool: ShadowSpool, path: Path) -> int:
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            prediction = route_shadow_run(validate_run_input(raw))
        except ValueError as exc:
            raise ValueError(f"invalid prospective input at line {line_number}: {exc}") from exc
        spool.enqueue(prediction)
        count += 1
    return count


def deliver_pending(spool: ShadowSpool, args: argparse.Namespace) -> tuple[int, int]:
    delivered = failed = 0
    for row in spool.pending(args.batch_size):
        try:
            receipt = push_prediction(
                args.target_url,
                args.ingest_token,
                args.supabase_access_token,
                row.payload,
                timeout=args.timeout,
            )
        except PermanentExportError as exc:
            spool.mark_failure(row.run_id, str(exc), permanent=True)
            failed += 1
        except RetryableExportError as exc:
            spool.mark_failure(row.run_id, str(exc), permanent=False)
            failed += 1
        else:
            spool.mark_delivered(row.run_id, receipt)
            delivered += 1
    return delivered, failed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prediction-jsonl", type=Path)
    result.add_argument("--spool", type=Path, default=SPOOL_PATH)
    result.add_argument("--target-url", default=os.environ.get("SHADOW_INGEST_URL"))
    result.add_argument("--ingest-token", default=os.environ.get("DASHBOARD_INGEST_TOKEN"))
    result.add_argument("--supabase-access-token", default=os.environ.get("DASHBOARD_SUPABASE_ACCESS_TOKEN"))
    result.add_argument("--batch-size", type=int, default=100)
    result.add_argument("--timeout", type=float, default=20)
    result.add_argument("--poll-seconds", type=float, default=10)
    result.add_argument("--once", action="store_true")
    result.add_argument("--enqueue-only", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    spool = ShadowSpool(args.spool)
    if args.prediction_jsonl:
        enqueued = enqueue_file(spool, args.prediction_jsonl)
        print(json.dumps({"enqueued": enqueued, "spool": str(args.spool)}, ensure_ascii=False), flush=True)
    if args.enqueue_only:
        return 0
    if not args.target_url or not args.ingest_token or not args.supabase_access_token:
        parser().error("delivery requires target URL, ingest token, and Supabase access token")
    while True:
        delivered, failed = deliver_pending(spool, args)
        print(json.dumps({"delivered": delivered, "failed": failed, "pending": len(spool.pending(1000))}), flush=True)
        if args.once:
            return 0 if failed == 0 else 1
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
