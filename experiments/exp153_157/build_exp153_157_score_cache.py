"""Build the shared label-free score cache for Exp153--157.

The expensive feature extraction and OOF scoring is performed once.  Later
experiments consume only frozen NPZ cache files.  TEST labels are never loaded
by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import exp153_157_bridge as bridge
import exp153_157_core as core
import exp153_157_reporting as reporting

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/exp153_157_train_only_improvements"
EXPERIMENT_ID = "experiment_153_157_shared_label_free_score_cache"


def _build_one(args):
    dataset_name, cache_dir = args
    return bridge.build_dataset_cache(dataset_name, Path(cache_dir))


def build_cache(
    output_dir: Path,
    *,
    dataset_limit: int | None = None,
    workers: int = 7,
    reset: bool = False,
    skip_db_hash: bool = False,
) -> list[dict[str, object]]:
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "01_score_cache_manifest.csv"
    names = bridge.evaluation_dataset_names(dataset_limit)

    # Exp154--157 use Exp152 full-coverage source semantics.  Validate that the
    # frozen B2 contract is present before spending compute on the score cache.
    b2_path, _ = bridge.validate_b2_contract(names)

    existing = {}
    if manifest_path.exists() and not reset:
        existing = {
            row["dataset_name"]: row for row in reporting.read_rows(manifest_path)
        }
        for name, row in list(existing.items()):
            cache_file = cache_dir / row["cache_file"]
            if name not in names or not cache_file.exists():
                existing.pop(name, None)
                continue
            try:
                core.verify_frozen_file(cache_file, row["cache_sha256"])
            except ValueError:
                existing.pop(name, None)

    pending = [name for name in names if name not in existing]
    rows = list(existing.values())
    errors = []
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(_build_one, (name, str(cache_dir))): name
            for name in pending
        }
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # preserve every dataset error for audit
                errors.append({"dataset_name": name, "error": repr(exc)})
            if done % 10 == 0 or done == len(pending):
                print(
                    f"cache progress {len(existing)+done}/{len(names)} "
                    f"rows={len(rows)} errors={len(errors)}",
                    flush=True,
                )
    if errors:
        reporting.write_rows(output_dir / "CACHE_ERRORS.csv", errors)
        raise SystemExit(
            f"{len(errors)} cache errors; see {output_dir / 'CACHE_ERRORS.csv'}"
        )

    rows.sort(key=lambda row: row["dataset_name"])
    if len(rows) != len(names):
        raise ValueError(f"cache coverage mismatch {len(rows)} != {len(names)}")
    reporting.write_rows(manifest_path, rows)
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_count": len(names),
        "dataset_list_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
        "database_path": str(bridge.DB_PATH),
        "database_sha256": None if skip_db_hash else core.sha256_file(bridge.DB_PATH),
        "baseline_path": str(bridge.BASELINE_PATH),
        "baseline_sha256": core.sha256_file(bridge.BASELINE_PATH),
        "b2_manifest_path": str(b2_path),
        "b2_manifest_sha256": core.sha256_file(b2_path),
        "source_names": list(bridge.SOURCE_NAMES),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "seed": core.SEED,
    }
    (output_dir / "00_cache_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-db-hash", action="store_true")
    args = parser.parse_args()
    build_cache(
        Path(args.output_dir),
        dataset_limit=args.dataset_limit,
        workers=args.workers,
        reset=args.reset,
        skip_db_hash=args.skip_db_hash,
    )


if __name__ == "__main__":
    main()
