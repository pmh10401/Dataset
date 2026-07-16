from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

from experiments.exp224.artifacts import build_artifacts
from experiments.exp224.policy import freeze_exp224_policy
from experiments.exp224.replay import run_raw_replay
from experiments.exp224.scoring import predict_exp224_batch
from experiments.exp220_223.storage import ingest_values

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DEFAULT_ROOT = DATA_ROOT / "exp224_prospective"
DEFAULT_DB = DATA_ROOT / "univariate_ts.db"
DEFAULT_PROFILE_REGISTRY = (
    REPO_ROOT / "results/exp137/experiment_137_operational_triage_results.csv"
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _eligible_profile_ids(
    db_path: Path,
    *,
    registry_path: Path = DEFAULT_PROFILE_REGISTRY,
    expected_count: int = 639,
) -> list[str]:
    with Path(registry_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {
            "dataset_name",
            "train_normal_count",
        }.issubset(reader.fieldnames):
            raise ValueError("Exp137 profile registry columns are missing")
        registry_counts = {
            str(row["dataset_name"]): int(row["train_normal_count"])
            for row in reader
            if int(row["train_normal_count"]) >= 20
        }
    if len(registry_counts) != int(expected_count):
        raise ValueError(
            f"frozen profile registry count mismatch: {len(registry_counts)}/{expected_count}"
        )
    with sqlite3.connect(Path(db_path)) as connection:
        placeholders = ",".join("?" for _ in registry_counts)
        rows = connection.execute(
            f"SELECT name, train_normal_count FROM datasets WHERE name IN ({placeholders})",
            tuple(sorted(registry_counts)),
        ).fetchall()
    database_counts = {str(name): int(count) for name, count in rows}
    if database_counts != registry_counts:
        missing = sorted(set(registry_counts) - set(database_counts))
        changed = sorted(
            name
            for name in set(registry_counts) & set(database_counts)
            if registry_counts[name] != database_counts[name]
        )
        raise ValueError(
            f"profile registry/database mismatch: missing={missing} changed={changed}"
        )
    return sorted(registry_counts)


def _source_paths() -> dict[str, Path]:
    experiments = REPO_ROOT / "experiments"
    return {
        "exp224_core": Path(__file__).with_name("core.py"),
        "exp224_artifacts": Path(__file__).with_name("artifacts.py"),
        "exp224_policy": Path(__file__).with_name("policy.py"),
        "exp224_replay": Path(__file__).with_name("replay.py"),
        "exp224_scoring": Path(__file__).with_name("scoring.py"),
        "exp224_cli": Path(__file__),
        "exp137_profile_registry": DEFAULT_PROFILE_REGISTRY,
        "exp220_prediction": experiments / "exp220_223" / "prediction.py",
        "exp220_scoring": experiments / "exp220_223" / "scoring.py",
        "exp220_storage": experiments / "exp220_223" / "storage.py",
        "canonical_core": experiments / "exp177_185" / "core.py",
        "conformal_core": experiments / "exp186_200" / "core.py",
        "exp153_bridge": experiments / "exp153_157" / "exp153_157_bridge.py",
        "exp152_scorer": experiments
        / "exp151_152"
        / "run_experiment_151_152_virtual_run_conformal_policy_v2.py",
        "exp152_policy_core": experiments
        / "exp151_152"
        / "virtual_run_policy_core.py",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exp224 stable dual-source prospective shadow policy"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-policy")
    build.add_argument("--database", type=_path, default=DEFAULT_DB)
    build.add_argument("--output-root", type=_path, default=DEFAULT_ROOT)
    build.add_argument("--workers", type=int, default=7)

    replay = commands.add_parser("raw-replay-audit")
    replay.add_argument(
        "--policy", type=_path, default=DEFAULT_ROOT / "policy_bundle.json"
    )
    replay.add_argument("--database", type=_path, default=DEFAULT_DB)
    replay.add_argument(
        "--output", type=_path, default=DEFAULT_ROOT / "raw_replay_report.json"
    )
    replay.add_argument("--workers", type=int, default=7)
    replay.add_argument("--profile-limit", type=int)

    ingest = commands.add_parser("ingest-values")
    ingest.add_argument("--values-db", type=_path, required=True)
    ingest.add_argument("--source", type=_path, required=True)
    ingest.add_argument(
        "--policy", type=_path, default=DEFAULT_ROOT / "policy_bundle.json"
    )
    ingest.add_argument("--batch-id", required=True)
    ingest.add_argument("--collected-start", required=True)
    ingest.add_argument("--collected-end", required=True)

    predict = commands.add_parser("predict-shadow")
    predict.add_argument("--values-db", type=_path, required=True)
    predict.add_argument("--train-db", type=_path, default=DEFAULT_DB)
    predict.add_argument("--batch-id", required=True)
    predict.add_argument(
        "--policy", type=_path, default=DEFAULT_ROOT / "policy_bundle.json"
    )
    predict.add_argument(
        "--raw-report", type=_path, default=DEFAULT_ROOT / "raw_replay_report.json"
    )
    predict.add_argument("--exp137-predictions", type=_path, required=True)
    predict.add_argument("--exp137-manifest", type=_path, required=True)
    predict.add_argument("--output-dir", type=_path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-policy":
        profile_ids = _eligible_profile_ids(args.database)
        rows = build_artifacts(
            profile_ids,
            args.database,
            args.output_root,
            workers=args.workers,
        )
        policy_path = args.output_root / "policy_bundle.json"
        policy = freeze_exp224_policy(
            args.output_root / "artifact_manifest.csv",
            {"expected_profile_count": 639},
            {"db_path": args.database, "source_paths": _source_paths()},
            policy_path,
        )
        summary: dict[str, object] = {
            "policy_path": str(policy_path),
            "policy_hash": policy["policy_hash"],
            "profile_count": len(rows),
            "selected_alpha": policy["selected_alpha"],
            "shadow_execution_gate": policy["shadow_execution_gate"],
        }
    elif args.command == "raw-replay-audit":
        summary = run_raw_replay(
            args.policy,
            args.database,
            args.output,
            workers=args.workers,
            profile_limit=args.profile_limit,
        )
    elif args.command == "ingest-values":
        summary = ingest_values(
            db_path=args.values_db,
            source_path=args.source,
            policy_path=args.policy,
            batch_id=args.batch_id,
            collected_start=args.collected_start,
            collected_end=args.collected_end,
        )
    else:
        summary = predict_exp224_batch(
            values_db_path=args.values_db,
            train_db_path=args.train_db,
            batch_id=args.batch_id,
            policy_path=args.policy,
            raw_report_path=args.raw_report,
            exp137_predictions_path=args.exp137_predictions,
            exp137_manifest_path=args.exp137_manifest,
            output_dir=args.output_dir,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
