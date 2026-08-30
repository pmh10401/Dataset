#!/usr/bin/env python3
"""Restore and run the byte-verified PR #27 review-fix applicator."""

from __future__ import annotations

import base64
import gzip
from hashlib import sha256
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
PAYLOAD_SHA256_ANTIGRAVITY = (
    "3cc3b45e5d083e2e474852ecca6cf993e9c54e7e3ea9ebb491cfdfe33341e4d7"
)
SOURCE_SHA256_ANTIGRAVITY = (
    "3f6c5f7c86381a638dc85366ab0b0cca8b2604a957aab8f46314d9e0d0c8a879"
)


def main_antigravity() -> None:
    payload_paths = tuple(
        ROOT_ANTIGRAVITY
        / ".github/workflows"
        / f"pr27-review-fixes-payload-{index:02d}-antigravity.txt"
        for index in range(5)
    )
    missing = tuple(path for path in payload_paths if not path.is_file())
    if missing:
        raise RuntimeError(f"PR27 applicator payload is incomplete: {missing}")
    payload = "".join(
        path.read_text(encoding="ascii") for path in payload_paths
    ).encode("ascii")
    if sha256(payload).hexdigest() != PAYLOAD_SHA256_ANTIGRAVITY:
        raise RuntimeError("PR27 applicator payload SHA256 mismatch")
    source = gzip.decompress(base64.b64decode(payload, validate=True))
    if sha256(source).hexdigest() != SOURCE_SHA256_ANTIGRAVITY:
        raise RuntimeError("PR27 applicator source SHA256 mismatch")
    decoded = (
        ROOT_ANTIGRAVITY
        / "scratch/apply_pr27_review_fixes_decoded_antigravity.py"
    )
    decoded.write_bytes(source)
    code = compile(source, str(decoded), "exec")
    namespace = {
        "__name__": "__main__",
        "__file__": str(decoded),
        "__package__": None,
    }
    exec(code, namespace)


if __name__ == "__main__":
    main_antigravity()
