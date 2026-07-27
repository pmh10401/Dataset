#!/usr/bin/env python3
"""Spool and upload verified Exp400 retrospective Shadow ranking rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp265_shadow_validation.exporter import (
    PermanentExportError,
    RetryableExportError,
)
from experiments.exp265_shadow_validation.spool import ShadowSpool
from experiments.exp400_402.artifact_io import (
    MANIFEST_NAME,
    PREDICTION_NAME,
    load_verified_prediction,
)


DEFAULT_ROOT = Path(
    "/Users/minho/Documents/Dataset/exp400_402_exp397_shadow_validation"
)
DEFAULT_SPOOL = DEFAULT_ROOT / "exp400_shadow_ranking_spool.sqlite"
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024


def enqueue_prediction(
    spool: ShadowSpool,
    prediction_path: Path,
    manifest_path: Path,
) -> int:
    rows, _ = load_verified_prediction(prediction_path, manifest_path)
    for row in rows:
        spool.enqueue(row)
    return len(rows)


def push_prediction(
    url: str,
    ingest_token: str,
    access_token: str,
    payload: Mapping[str, object],
    *,
    timeout: float,
) -> dict[str, object]:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(body) > MAX_PAYLOAD_BYTES:
        raise PermanentExportError(
            f"ranking payload exceeds {MAX_PAYLOAD_BYTES} bytes"
        )
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {ingest_token}",
            "X-Supabase-Access-Token": access_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        if status in {400, 401, 403, 404, 409, 413, 422}:
            raise PermanentExportError(
                f"ranking ingest returned HTTP {status}"
            ) from exc
        raise RetryableExportError(
            f"ranking ingest returned HTTP {status}"
        ) from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RetryableExportError(f"ranking ingest unavailable: {exc}") from exc
    if type(result) is not dict or result.get("ok") is not True:
        raise RetryableExportError("ranking ingest returned an invalid receipt")
    return result


def deliver_pending(
    spool: ShadowSpool,
    *,
    target_url: str,
    ingest_token: str,
    access_token: str,
    batch_size: int,
    timeout: float,
) -> tuple[int, int]:
    delivered = failed = 0
    for row in spool.pending(batch_size):
        try:
            receipt = push_prediction(
                target_url,
                ingest_token,
                access_token,
                row.payload,
                timeout=timeout,
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
    result.add_argument(
        "--prediction-jsonl",
        type=Path,
        default=DEFAULT_ROOT / PREDICTION_NAME,
    )
    result.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_ROOT / MANIFEST_NAME,
    )
    result.add_argument("--spool", type=Path, default=DEFAULT_SPOOL)
    result.add_argument("--batch-size", type=int, default=100)
    result.add_argument("--timeout", type=float, default=20)
    result.add_argument("--poll-seconds", type=float, default=10)
    result.add_argument("--enqueue-only", action="store_true")
    result.add_argument("--once", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    spool = ShadowSpool(args.spool)
    enqueued = enqueue_prediction(
        spool,
        args.prediction_jsonl,
        args.manifest,
    )
    print(
        json.dumps(
            {"enqueued": enqueued, "spool": str(args.spool)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.enqueue_only:
        return 0

    target_url = os.environ.get("SHADOW_RANKING_INGEST_URL")
    ingest_token = os.environ.get("DASHBOARD_INGEST_TOKEN")
    access_token = os.environ.get("DASHBOARD_SUPABASE_ACCESS_TOKEN")
    if not target_url or not ingest_token or not access_token:
        parser().error(
            "delivery requires SHADOW_RANKING_INGEST_URL, "
            "DASHBOARD_INGEST_TOKEN, and DASHBOARD_SUPABASE_ACCESS_TOKEN"
        )

    while True:
        delivered, failed = deliver_pending(
            spool,
            target_url=target_url,
            ingest_token=ingest_token,
            access_token=access_token,
            batch_size=args.batch_size,
            timeout=args.timeout,
        )
        print(
            json.dumps(
                {
                    "delivered": delivered,
                    "failed": failed,
                    "pending": len(spool.pending(1000)),
                }
            ),
            flush=True,
        )
        if args.once:
            return 0 if failed == 0 else 1
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
