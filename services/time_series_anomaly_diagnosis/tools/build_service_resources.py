from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil

EVIDENCE_SHA256 = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
MANIFEST_SHA256 = "0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0"


def find_file_by_sha256(
    repository_root: Path,
    expected: str,
    excluded_directory: Path,
    pattern: str,
) -> Path:
    excluded = excluded_directory.resolve()
    matches = [
        path
        for path in repository_root.rglob(pattern)
        if excluded not in path.resolve().parents
        and sha256(path.read_bytes()).hexdigest() == expected
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} file with sha256 {expected}, found {len(matches)}")
    return matches[0]


def build_review_resource(
    evidence_source: Path,
    manifest_source: Path,
    evidence_sha256: str,
) -> dict[str, object]:
    manifest = json.loads(manifest_source.read_text("utf-8"))
    if (
        manifest.get("prediction_sha256") != evidence_sha256
        or manifest.get("review_tier_policy_frozen_before_evaluation") is not True
        or manifest.get("review_only") is not True
        or manifest.get("contains_test_labels") is not False
        or manifest.get("contains_evaluation_metrics") is not False
    ):
        raise ValueError("review evidence manifest contract mismatch")

    datasets = []
    for line in evidence_source.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        if (
            source.get("review_only") is not True
            or source.get("contains_test_labels") is not False
            or source.get("contains_evaluation_metrics") is not False
        ):
            raise ValueError("review evidence row contract mismatch")
        datasets.append(
            {
                "dataset": source["dataset"],
                "priority_1": source["review_priority_1_indices"],
                "priority_2": source["review_priority_2_indices"],
                "priority_3": source["review_priority_3_indices"],
                "support": source["review_support_by_index"],
            }
        )

    return {"schema": "review_evidence_v1", "datasets": datasets}


def build_service_resources(
    repository_root: Path,
    output: Path,
    routing_sha256: str,
    recovery_sha256: str,
    review_sha256: str = EVIDENCE_SHA256,
    review_manifest_sha256: str = MANIFEST_SHA256,
) -> None:
    sources = {
        "routing_catalog.json": find_file_by_sha256(
            repository_root, routing_sha256, output, "*.json"
        ),
        "recovery_catalog.json": find_file_by_sha256(
            repository_root, recovery_sha256, output, "*.json"
        ),
    }
    review_source = find_file_by_sha256(
        repository_root, review_sha256, output, "*.jsonl"
    )
    review_manifest_source = find_file_by_sha256(
        repository_root, review_manifest_sha256, output, "*.json"
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copyfile(source, output / name)
    (output / "review_evidence.json").write_text(
        json.dumps(
            build_review_resource(review_source, review_manifest_source, review_sha256),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    sources["review_evidence.json"] = output / "review_evidence.json"
    files = {name: sha256((output / name).read_bytes()).hexdigest() for name in sources}
    (output / "model_manifest.json").write_text(
        json.dumps({"schema": "service_model_resources_v2", "files": files}, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routing-sha256", required=True)
    parser.add_argument("--recovery-sha256", required=True)
    parser.add_argument("--review-sha256", default=EVIDENCE_SHA256)
    parser.add_argument("--review-manifest-sha256", default=MANIFEST_SHA256)
    arguments = parser.parse_args()
    build_service_resources(
        arguments.repository_root,
        arguments.output,
        arguments.routing_sha256,
        arguments.recovery_sha256,
        arguments.review_sha256,
        arguments.review_manifest_sha256,
    )


if __name__ == "__main__":
    main()
