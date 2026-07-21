#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp265_shadow_validation.replay import (
    DEFAULT_DB_PATH,
    DEFAULT_EXP264_PATH,
    build_replay_dataset,
    write_replay_dataset,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build label-free Exp265 retrospective dashboard replay rows.")
    result.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    result.add_argument("--predictions", type=Path, default=DEFAULT_EXP264_PATH)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--limit", type=int)
    return result


def main() -> int:
    args = parser().parse_args()
    rows = build_replay_dataset(db_path=args.db, prediction_path=args.predictions, limit=args.limit)
    manifest = write_replay_dataset(args.output, rows)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "output": str(args.output), "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
