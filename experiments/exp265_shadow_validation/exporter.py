from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Mapping


MAX_PAYLOAD_BYTES = 512 * 1024


class ExportError(RuntimeError):
    """Base shadow export error."""


class PermanentExportError(ExportError):
    """The payload must not be retried without correction."""


class RetryableExportError(ExportError):
    """The payload remains pending for a later retry."""


def push_prediction(
    url: str,
    ingest_token: str,
    access_token: str,
    payload: Mapping[str, object],
    timeout: float = 20,
) -> dict[str, object]:
    if not url or not ingest_token or not access_token:
        raise ValueError("URL, ingest token, and access token are required")
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(body) > MAX_PAYLOAD_BYTES:
        raise PermanentExportError(f"shadow payload exceeds {MAX_PAYLOAD_BYTES} bytes")
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
        message = f"shadow ingest returned HTTP {exc.code}"
        exc.close()
        if exc.code in {400, 401, 403, 404, 409, 413, 422}:
            raise PermanentExportError(message) from exc
        raise RetryableExportError(message) from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RetryableExportError(f"shadow ingest unavailable: {exc}") from exc
    if type(result) is not dict or result.get("ok") is not True:
        raise RetryableExportError("shadow ingest returned an invalid receipt")
    return result
