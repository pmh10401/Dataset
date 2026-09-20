"""Build the DB-backed Exp479 derived-dataset label report artifact."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
DATABASE_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
ARTIFACT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "scratch/exp479_class_count_report_artifact_antigravity.json"
)

SOURCE_SQL_ANTIGRAVITY = """SELECT
  d.name,
  d.train_normal_count,
  d.test_normal_count,
  d.total_normal_count,
  i.split,
  i.label,
  COUNT(i.id) AS instance_rows
FROM datasets AS d
LEFT JOIN instances AS i ON i.dataset_id = d.id
GROUP BY
  d.id,
  d.name,
  d.train_normal_count,
  d.test_normal_count,
  d.total_normal_count,
  i.split,
  i.label
ORDER BY d.name, i.split, i.label"""


def _label_key_antigravity(label: str) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def _counts_text_antigravity(
    rows: list[dict[str, int | str]], field: str
) -> str:
    return " · ".join(f"{row['label']}: {int(row[field]):,}" for row in rows)


def build_artifact_antigravity() -> dict[str, object]:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    connection = sqlite3.connect(
        f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
    )
    try:
        dataset_rows = connection.execute(
            "SELECT name, train_normal_count, test_normal_count, total_normal_count "
            "FROM datasets ORDER BY name"
        ).fetchall()
        standalone_rows = connection.execute(
            "SELECT d.name, i.split, i.label, COUNT(*) "
            "FROM datasets AS d JOIN instances AS i ON i.dataset_id=d.id "
            "WHERE d.name NOT LIKE '%_normal_%' "
            "GROUP BY d.name, i.split, i.label ORDER BY d.name, i.split, i.label"
        ).fetchall()
        derived_label_rows = connection.execute(
            "SELECT d.name, "
            "COUNT(DISTINCT CASE WHEN i.split='TRAIN' THEN i.label END), "
            "GROUP_CONCAT(DISTINCT CASE WHEN i.split='TRAIN' THEN i.label END), "
            "COUNT(DISTINCT i.label) "
            "FROM datasets AS d JOIN instances AS i ON i.dataset_id=d.id "
            "WHERE d.name LIKE '%_normal_%' "
            "GROUP BY d.id ORDER BY d.name"
        ).fetchall()
    finally:
        connection.close()

    mapped_by_family: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    standalone_names: list[str] = []
    for raw_name, raw_train, raw_test, raw_total in dataset_rows:
        name = str(raw_name)
        train = int(raw_train)
        test = int(raw_test)
        total = int(raw_total)
        assert train + test == total
        if "_normal_" not in name:
            standalone_names.append(name)
            continue
        family, label = name.rsplit("_normal_", 1)
        mapped_by_family[family].append(
            {
                "label": label,
                "train_normal": train,
                "test_normal": test,
                "total_normal": total,
            }
        )

    family_rows: list[dict[str, object]] = []
    candidate_count_distribution: Counter[int] = Counter()
    candidate_examples: dict[int, list[str]] = defaultdict(list)
    unique_candidate_families = 0
    candidate_variant_count = 0
    for family in sorted(mapped_by_family):
        variants = sorted(
            mapped_by_family[family],
            key=lambda row: _label_key_antigravity(str(row["label"])),
        )
        candidate_train_count = max(int(row["train_normal"]) for row in variants)
        source_label_candidates = [
            str(row["label"])
            for row in variants
            if int(row["train_normal"]) == candidate_train_count
        ]
        candidate_variant_count += len(source_label_candidates)
        candidate_count_distribution[len(source_label_candidates)] += 1
        candidate_examples[len(source_label_candidates)].append(family)
        if len(source_label_candidates) == 1:
            unique_candidate_families += 1
        family_rows.append(
            {
                "family": family,
                "source_label_variant_count": len(variants),
                "normal_class_count_per_variant": 1,
                "train_normal_counts_by_source_label": _counts_text_antigravity(
                    variants, "train_normal"
                ),
                "source_label_candidates": ", ".join(source_label_candidates),
                "candidate_train_normal_count": candidate_train_count,
                "candidate_status": (
                    "단일 후보" if len(source_label_candidates) == 1 else "동률 후보"
                ),
            }
        )

    candidate_distribution_rows = [
        {
            "candidate_count": count,
            "family_count": candidate_count_distribution[count],
            "families": ", ".join(candidate_examples[count]),
        }
        for count in sorted(candidate_count_distribution)
    ]

    standalone_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for raw_name, raw_split, raw_label, raw_count in standalone_rows:
        standalone_counts[(str(raw_name), str(raw_label))][str(raw_split)] = int(
            raw_count
        )
    standalone_detail = []
    for (name, label), split_counts in sorted(standalone_counts.items()):
        train = split_counts["TRAIN"]
        test = split_counts["TEST"]
        standalone_detail.append(
            {
                "dataset": name,
                "derived_label": label,
                "meaning": "정상" if label == "0" else "이상",
                "train_rows": train,
                "test_rows": test,
                "total_rows": train + test,
            }
        )

    mapped_variants = sum(
        int(row["source_label_variant_count"]) for row in family_rows
    )
    label_violations = [
        row for row in derived_label_rows if tuple(row[1:]) != (1, "0", 2)
    ]
    tied_candidate_families = len(family_rows) - unique_candidate_families
    database_hash = sha256(DATABASE_PATH_ANTIGRAVITY.read_bytes()).hexdigest()
    summary_row = {
        "database_entries": len(dataset_rows),
        "logical_datasets": len(family_rows) + len(standalone_names),
        "mapped_families": len(family_rows),
        "derived_dataset_count": mapped_variants,
        "source_label_variant_count": mapped_variants,
        "normal_class_count_per_derived_dataset": 1,
        "derived_dataset_label_violations": len(label_violations),
        "candidate_ranking_basis": "train_normal_count",
        "unique_source_label_candidate_families": unique_candidate_families,
        "tied_source_label_candidate_families": tied_candidate_families,
        "source_label_candidate_variants": candidate_variant_count,
    }
    assert summary_row == {
        "database_entries": 1119,
        "logical_datasets": 129,
        "mapped_families": 128,
        "derived_dataset_count": 1118,
        "source_label_variant_count": 1118,
        "normal_class_count_per_derived_dataset": 1,
        "derived_dataset_label_violations": 0,
        "candidate_ranking_basis": "train_normal_count",
        "unique_source_label_candidate_families": 69,
        "tied_source_label_candidate_families": 59,
        "source_label_candidate_variants": 674,
    }
    assert standalone_names == ["CornellWhaleChallenge"]
    assert len(derived_label_rows) == 1118 and not label_violations
    assert len(family_rows) == 128 and len(standalone_detail) == 2

    source = {
        "id": "derived_class_census_sql_antigravity",
        "label": "파생 시계열 데이터셋의 정상 라벨과 원본 라벨 변형 집계",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": SOURCE_SQL_ANTIGRAVITY,
            "description": (
                "datasets 메타데이터와 instances 라벨 행 수를 읽어 "
                "파생 데이터셋의 정상 클래스 구조와 원본 라벨별 변형을 검증합니다."
            ),
            "executed_at": generated_at,
            "tables_used": ["datasets", "instances"],
        },
    }

    artifact: dict[str, object] = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "파생 데이터셋의 정상 클래스와 원본 라벨 변형",
            "description": (
                "univariate_ts.db의 1,118개 one-class 파생 데이터셋이 각각 정상 "
                "클래스 하나를 갖는지 검증하고 원본 라벨 변형을 구분한 기술 보고서"
            ),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "database_entries_card_antigravity",
                    "description": "datasets 테이블에 등록된 전체 행 수입니다.",
                    "dataset": "summary_antigravity",
                    "sourceId": source["id"],
                    "metrics": [
                        {
                            "label": "DB 등록 데이터셋 행",
                            "field": "database_entries",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "logical_datasets_card_antigravity",
                    "description": "128개 변환 가족과 독립 데이터셋 1개를 합한 수입니다.",
                    "dataset": "summary_antigravity",
                    "sourceId": source["id"],
                    "metrics": [
                        {
                            "label": "논리 데이터셋",
                            "field": "logical_datasets",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "derived_datasets_card_antigravity",
                    "description": "원본 라벨을 하나씩 정상으로 선택해 만든 파생 데이터셋 수입니다.",
                    "dataset": "summary_antigravity",
                    "sourceId": source["id"],
                    "metrics": [
                        {
                            "label": "one-class 파생 데이터셋",
                            "field": "derived_dataset_count",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "normal_class_card_antigravity",
                    "description": "각 파생 데이터셋의 TRAIN에 포함된 정상 클래스 수입니다.",
                    "dataset": "summary_antigravity",
                    "sourceId": source["id"],
                    "metrics": [
                        {
                            "label": "데이터셋별 정상 클래스",
                            "field": "normal_class_count_per_derived_dataset",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "label_violations_card_antigravity",
                    "description": "TRAIN 정상 클래스가 하나가 아닌 파생 데이터셋 수입니다.",
                    "dataset": "summary_antigravity",
                    "sourceId": source["id"],
                    "metrics": [
                        {
                            "label": "라벨 구조 위반",
                            "field": "derived_dataset_label_violations",
                            "format": "number",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "source_label_candidate_distribution_chart_antigravity",
                    "title": "가족별 최다 원본 라벨 후보 수",
                    "subtitle": "TRAIN 정상 행 수가 가장 큰 원본 라벨 후보 개수별 가족 분포입니다.",
                    "type": "bar",
                    "dataset": "source_label_candidate_distribution_antigravity",
                    "sourceId": source["id"],
                    "valueFormat": "number",
                    "encodings": {
                        "x": {
                            "field": "candidate_count",
                            "type": "ordinal",
                            "label": "최다 원본 라벨 후보 수",
                        },
                        "y": {
                            "field": "family_count",
                            "type": "quantitative",
                            "label": "가족 수",
                            "format": "number",
                        },
                    },
                }
            ],
            "tables": [
                {
                    "id": "source_label_candidate_distribution_table_antigravity",
                    "title": "최다 원본 라벨 후보 수별 가족",
                    "subtitle": "TRAIN 정상 행 수가 같은 후보를 모두 남긴 128개 가족의 분포입니다.",
                    "dataset": "source_label_candidate_distribution_antigravity",
                    "sourceId": source["id"],
                    "defaultSort": {
                        "field": "candidate_count",
                        "direction": "asc",
                    },
                    "columns": [
                        {
                            "field": "candidate_count",
                            "label": "후보 수",
                            "format": "number",
                        },
                        {
                            "field": "family_count",
                            "label": "가족 수",
                            "format": "number",
                        },
                        {"field": "families", "label": "데이터셋 가족", "type": "text"},
                    ],
                },
                {
                    "id": "family_source_label_variants_table_antigravity",
                    "title": "128개 가족의 원본 라벨 파생 변형",
                    "subtitle": (
                        "각 _normal_ 변형은 정상 클래스 하나를 가지며, 후보는 TRAIN 정상 행 수로 판정합니다."
                    ),
                    "dataset": "family_source_label_variants_antigravity",
                    "sourceId": source["id"],
                    "defaultSort": {"field": "family", "direction": "asc"},
                    "columns": [
                        {"field": "family", "label": "데이터셋 가족", "type": "text"},
                        {
                            "field": "source_label_variant_count",
                            "label": "원본 라벨 파생 변형 수",
                            "format": "number",
                        },
                        {
                            "field": "normal_class_count_per_variant",
                            "label": "변형별 정상 클래스 수",
                            "format": "number",
                        },
                        {
                            "field": "train_normal_counts_by_source_label",
                            "label": "원본 라벨별 TRAIN 정상 행 수",
                            "type": "text",
                        },
                        {
                            "field": "source_label_candidates",
                            "label": "최다 원본 라벨 후보",
                            "type": "text",
                        },
                        {
                            "field": "candidate_train_normal_count",
                            "label": "후보 TRAIN 정상 행 수",
                            "format": "number",
                        },
                        {"field": "candidate_status", "label": "후보 판정", "type": "text"},
                    ],
                },
                {
                    "id": "standalone_class_counts_table_antigravity",
                    "title": "독립 데이터셋의 파생 0/1 라벨 수",
                    "subtitle": "_normal_ 변형이 없는 CornellWhaleChallenge의 DB 행 수입니다.",
                    "dataset": "standalone_class_counts_antigravity",
                    "sourceId": source["id"],
                    "defaultSort": {"field": "derived_label", "direction": "asc"},
                    "columns": [
                        {"field": "dataset", "label": "데이터셋", "type": "text"},
                        {"field": "derived_label", "label": "파생 라벨", "type": "text"},
                        {"field": "meaning", "label": "뜻", "type": "text"},
                        {"field": "train_rows", "label": "TRAIN 행", "format": "number"},
                        {"field": "test_rows", "label": "TEST 행", "format": "number"},
                        {"field": "total_rows", "label": "전체 행", "format": "number"},
                    ],
                },
            ],
            "sources": [
                {
                    "id": source["id"],
                    "label": source["label"],
                }
            ],
            "blocks": [
                {
                    "id": "title_antigravity",
                    "type": "markdown",
                    "body": "# 파생 데이터셋의 정상 클래스와 원본 라벨 변형",
                },
                {
                    "id": "technical_summary_antigravity",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## 1,118개 파생 데이터셋은 모두 정상 클래스가 하나입니다\n\n"
                        "각 `<가족>_normal_<원본라벨>`은 원본 라벨 하나를 정상으로 선택해 "
                        "별도로 만든 one-class 데이터셋입니다. DB의 **1,118개 파생 "
                        "데이터셋 모두 TRAIN 정상 클래스가 1개**이고 라벨은 `0`입니다. "
                        "이 규칙을 어긴 데이터셋은 **0개**입니다."
                    ),
                },
                {
                    "id": "metrics_antigravity",
                    "type": "metric-strip",
                    "cardIds": [
                        "database_entries_card_antigravity",
                        "logical_datasets_card_antigravity",
                        "derived_datasets_card_antigravity",
                        "normal_class_card_antigravity",
                        "label_violations_card_antigravity",
                    ],
                },
                {
                    "id": "key_findings_antigravity",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## 최다 원본 라벨 후보는 69개 가족에서 하나로 정해집니다\n\n"
                        "원본 라벨 빈도의 대용값으로 `train_normal_count`를 사용하면 "
                        "**69개 가족은 후보가 하나**, **59개 가족은 동률**입니다. 동률을 "
                        "모두 남기면 후보 변형은 **674개**입니다. 아래 차트는 가족별 후보 "
                        "개수의 분포이며, 정상 클래스 개수를 나타내는 차트가 아닙니다."
                    ),
                },
                {
                    "id": "source_label_candidate_distribution_visual_antigravity",
                    "type": "chart",
                    "chartId": "source_label_candidate_distribution_chart_antigravity",
                    "layout": "full",
                },
                {
                    "id": "source_label_candidate_distribution_antigravity",
                    "type": "table",
                    "tableId": "source_label_candidate_distribution_table_antigravity",
                    "layout": "full",
                },
                {
                    "id": "scope_definitions_antigravity",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## 정상 클래스와 원본 라벨 변형은 서로 다른 개념입니다\n\n"
                        "`원본 라벨 파생 변형 수`는 한 가족에서 만들어진 `_normal_*` "
                        "데이터셋의 개수입니다. 각 변형 내부의 정상 클래스는 항상 하나입니다. "
                        "TEST의 `0`과 `1`은 정상과 이상을 뜻하는 평가 라벨이며, 정상 클래스가 "
                        "두 개라는 뜻이 아닙니다."
                    ),
                },
                {
                    "id": "family_detail_antigravity",
                    "type": "table",
                    "tableId": "family_source_label_variants_table_antigravity",
                    "layout": "full",
                },
                {
                    "id": "standalone_note_antigravity",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## CornellWhaleChallenge는 별도 0/1 파생 데이터입니다\n\n"
                        "이 데이터셋은 `_normal_` 라벨 변형이 없어 원래 클래스별 비교에 "
                        "포함하지 않았습니다. 대신 DB에 저장된 파생 정상(0)·이상(1) 행 수를 "
                        "별도 표로 표시합니다."
                    ),
                },
                {
                    "id": "standalone_detail_antigravity",
                    "type": "table",
                    "tableId": "standalone_class_counts_table_antigravity",
                    "layout": "full",
                },
                {
                    "id": "methodology_antigravity",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## 최다 원본 라벨 후보는 TRAIN 정상 행 수로 판정합니다\n\n"
                        "데이터셋 이름을 마지막 `_normal_`에서 나눠 가족과 원본 라벨을 "
                        "구분했습니다. 생성 과정에서 원본 라벨 표본의 80%가 TRAIN 정상으로 "
                        "저장되므로, 가족 안에서 `train_normal_count`가 가장 큰 변형을 최다 "
                        "원본 라벨 후보로 표시했습니다. 같은 값이면 모두 후보로 남겼습니다. 정확한 "
                        f"DB 스냅샷 SHA-256은 `{database_hash}`입니다."
                    ),
                },
                {
                    "id": "limitations_antigravity",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## 동률 후보의 정확한 원본 표본 수는 DB만으로 복원할 수 없습니다\n\n"
                        "TRAIN 수는 원본 표본 수의 80%를 내림한 값이어서 서로 가까운 원본 "
                        "빈도가 같은 TRAIN 수가 될 수 있습니다. TEST 정상 수에는 2% 이상 비율을 "
                        "맞추기 위한 선택과 복원 추출이 반영되므로 `total_normal_count`는 후보 "
                        "판정에 사용하지 않았습니다. 동률 59개 가족은 원본 자료가 없으면 하나로 "
                        "확정할 수 없습니다."
                    ),
                },
                {
                    "id": "next_steps_antigravity",
                    "type": "markdown",
                    "body": (
                        "## Exp479에 적용할 다음 규칙\n\n"
                        "Exp479 모집단은 표의 `최다 원본 라벨 후보`를 기준으로 구성할 수 있습니다. "
                        "단일 후보 69개는 바로 선택할 수 있고, 동률 59개는 모두 포함하거나 원본 "
                        "자료로 다시 판정하는 규칙을 실험 계약에 고정해야 합니다."
                    ),
                },
                {
                    "id": "further_questions_antigravity",
                    "type": "markdown",
                    "body": (
                        "## 남은 결정\n\n"
                        "동률인 최다 원본 라벨 후보를 모두 포함할지 결정하면, 그 규칙으로 "
                        "Exp479 대상 목록과 성능 집계를 확정할 수 있습니다."
                    ),
                },
                {
                    "id": "verification_antigravity",
                    "type": "markdown",
                    "body": (
                        "## 보고서 검증 상태\n\n"
                        "- [x] DB 1,119개 등록 행 확인\n"
                        "- [x] 1,118개 파생 데이터셋의 TRAIN 정상 클래스 1개 확인\n"
                        "- [x] TRAIN 라벨 `0` 및 TEST 정상·이상 이진 라벨 확인\n"
                        "- [x] `train_normal_count` 기준 단일 후보 69개·동률 59개 검증\n"
                        "- [x] 원본 라벨 후보 변형 674개 검증\n"
                        "- [x] HTML 패키징 검증"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary_antigravity": [summary_row],
                "source_label_candidate_distribution_antigravity": candidate_distribution_rows,
                "family_source_label_variants_antigravity": family_rows,
                "standalone_class_counts_antigravity": standalone_detail,
            },
        },
        "sources": [source],
        "package_info": {},
    }
    return artifact


def main() -> None:
    artifact = build_artifact_antigravity()
    ARTIFACT_PATH_ANTIGRAVITY.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(ARTIFACT_PATH_ANTIGRAVITY)


if __name__ == "__main__":
    main()
