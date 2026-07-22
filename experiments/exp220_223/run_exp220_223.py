from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from experiments.exp220_223.evaluation import (
    evaluate_shadow_rows,
    import_outcomes,
    load_outcome_labels,
    write_evaluation_report,
)
from experiments.exp220_223.policy import freeze_default_policy, load_verified_policy
from experiments.exp220_223.prediction import predict_batch, verify_prediction_manifest
from experiments.exp220_223.replay import run_cached_replay_audit, run_raw_replay_audit
from experiments.exp220_223.scoring import RawEvidenceProvider
from experiments.exp220_223.storage import ingest_values


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DEFAULT_ROOT = DATA_ROOT / "exp220_223_shadow"


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exp220--223 prospective shadow validation"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-policy")
    freeze.add_argument("--repo-root", type=_path, default=REPO_ROOT)
    freeze.add_argument("--data-root", type=_path, default=DATA_ROOT)
    freeze.add_argument(
        "--database", type=_path, default=DATA_ROOT / "univariate_ts.db"
    )
    freeze.add_argument(
        "--output", type=_path, default=DEFAULT_ROOT / "policy_bundle.json"
    )

    replay = commands.add_parser("replay-audit")
    replay.add_argument("--repo-root", type=_path, default=REPO_ROOT)
    replay.add_argument("--data-root", type=_path, default=DATA_ROOT)
    replay.add_argument(
        "--policy", type=_path, default=DEFAULT_ROOT / "policy_bundle.json"
    )
    replay.add_argument(
        "--output", type=_path, default=DEFAULT_ROOT / "exp220_replay_audit.json"
    )
    replay.add_argument("--mode", choices=("cached", "raw"), default="raw")
    replay.add_argument(
        "--database", type=_path, default=DATA_ROOT / "univariate_ts.db"
    )
    replay.add_argument("--workers", type=int, default=1)
    replay.add_argument("--profile-limit", type=int)

    ingest = commands.add_parser("ingest-values")
    ingest.add_argument("--db", type=_path, required=True)
    ingest.add_argument("--source", type=_path, required=True)
    ingest.add_argument("--policy", type=_path, required=True)
    ingest.add_argument("--batch-id", required=True)
    ingest.add_argument("--collected-start", required=True)
    ingest.add_argument("--collected-end", required=True)

    predict = commands.add_parser("predict-shadow")
    predict.add_argument("--db", type=_path, required=True)
    predict.add_argument("--batch-id", required=True)
    predict.add_argument("--policy", type=_path, required=True)
    predict.add_argument("--audit-report", type=_path, required=True)
    predict.add_argument("--train-db", type=_path, default=DATA_ROOT / "univariate_ts.db")
    predict.add_argument("--exp137-predictions", type=_path, required=True)
    predict.add_argument("--exp137-manifest", type=_path, required=True)
    predict.add_argument("--output-dir", type=_path, required=True)
    predict.add_argument("--allow-blocked-synthetic-smoke", action="store_true")

    outcomes = commands.add_parser("import-outcomes")
    outcomes.add_argument("--outcomes-db", type=_path, required=True)
    outcomes.add_argument("--csv", type=_path, required=True)
    outcomes.add_argument("--predictions", type=_path, required=True)
    outcomes.add_argument("--manifest", type=_path, required=True)

    evaluate = commands.add_parser("evaluate-shadow")
    evaluate.add_argument("--values-db", type=_path, required=True)
    evaluate.add_argument("--outcomes-db", type=_path, required=True)
    evaluate.add_argument("--predictions-dir", type=_path, required=True)
    evaluate.add_argument("--output", type=_path, required=True)
    return parser


def _assert_shadow_gate(
    report_path: Path, *, batch_id: str, allow_blocked_synthetic_smoke: bool
) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    gate = report.get("shadow_execution_gate")
    if gate == "passed":
        return
    if allow_blocked_synthetic_smoke and batch_id.startswith("synthetic-"):
        return
    raise ValueError(
        f"shadow execution gate is not passed: {gate}; real batch prediction is blocked"
    )


def _load_frozen_directory(
    predictions_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    rows: list[dict[str, object]] = []
    policies: dict[str, str] = {}
    manifests = sorted(Path(predictions_dir).glob("*.prediction_manifest.json"))
    if not manifests:
        raise ValueError("no prediction manifests found")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        prediction_path = manifest_path.parent / manifest["prediction_file"]
        verified_manifest, verified_rows = verify_prediction_manifest(
            prediction_path, manifest_path
        )
        batch_id = str(verified_manifest["batch_id"])
        if batch_id in policies:
            raise ValueError(f"duplicate prediction batch: {batch_id}")
        policies[batch_id] = str(verified_manifest["policy_sha256"])
        rows.extend(verified_rows)
    if len(set(policies.values())) != 1:
        raise ValueError("prediction batches use different policy hashes")
    return rows, policies


def _batch_periods(values_db: Path, batch_ids: set[str]) -> dict[str, tuple[str, str]]:
    with sqlite3.connect(Path(values_db)) as connection:
        rows = connection.execute(
            """
            SELECT batch_id, collected_start_utc, collected_end_utc
            FROM shadow_batches
            """
        ).fetchall()
    periods = {
        str(batch): (str(start), str(end))
        for batch, start, end in rows
        if str(batch) in batch_ids
    }
    if set(periods) != batch_ids:
        raise ValueError("values DB does not contain every prediction batch")
    return periods


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-policy":
        result = freeze_default_policy(
            repo_root=args.repo_root,
            data_root=args.data_root,
            database_path=args.database,
            output_path=args.output,
        )
        summary = {
            "policy_path": str(args.output),
            "policy_sha256": result["policy_sha256"],
            "profiles": len(result["profiles"]),
        }
    elif args.command == "replay-audit":
        load_verified_policy(args.policy, verify_artifacts=True)
        if args.mode == "raw":
            summary = run_raw_replay_audit(
                repo_root=args.repo_root,
                data_root=args.data_root,
                database_path=args.database,
                policy_path=args.policy,
                output_path=args.output,
                workers=args.workers,
                profile_limit=args.profile_limit,
            )
        else:
            summary = run_cached_replay_audit(
                repo_root=args.repo_root,
                data_root=args.data_root,
                output_path=args.output,
            )
    elif args.command == "ingest-values":
        summary = ingest_values(
            db_path=args.db,
            source_path=args.source,
            policy_path=args.policy,
            batch_id=args.batch_id,
            collected_start=args.collected_start,
            collected_end=args.collected_end,
        )
    elif args.command == "predict-shadow":
        _assert_shadow_gate(
            args.audit_report,
            batch_id=args.batch_id,
            allow_blocked_synthetic_smoke=args.allow_blocked_synthetic_smoke,
        )
        provider = RawEvidenceProvider(
            train_db_path=args.train_db,
            exp137_predictions_path=args.exp137_predictions,
            exp137_manifest_path=args.exp137_manifest,
            batch_id=args.batch_id,
        )
        summary = predict_batch(
            db_path=args.db,
            batch_id=args.batch_id,
            policy_path=args.policy,
            output_dir=args.output_dir,
            evidence_provider=provider,
        )
    elif args.command == "import-outcomes":
        summary = import_outcomes(
            outcomes_db_path=args.outcomes_db,
            outcomes_csv_path=args.csv,
            prediction_path=args.predictions,
            manifest_path=args.manifest,
        )
    else:
        predictions, policies = _load_frozen_directory(args.predictions_dir)
        labels = load_outcome_labels(args.outcomes_db)
        periods = _batch_periods(args.values_db, set(policies))
        summary = evaluate_shadow_rows(predictions, labels, periods)
        write_evaluation_report(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
