"""Typed Exp473 runtime over current verified values and official Exp472-E identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from paper.equipment_anomaly_detection.catalog_paper import build_catalog
from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.data_paper import (
    bind_preverified_database_read_only_paper,
    load_dataset_values,
)

from experiments.exp472_makima_upsampling_ablation_antigravity.artifacts_antigravity import (
    Exp472CatalogEntryAntigravity,
    catalog_sha256_exp472_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.config_antigravity import (
    EXP472_OFFICIAL_ROOT_ANTIGRAVITY,
    assess_exp472_backend_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
    canonical_json_bytes_exp472_antigravity,
)
from .artifacts_antigravity import (
    publish_run_context_exp473_antigravity,
    verify_manifest_exp473_antigravity,
    verify_run_context_exp473_antigravity,
    verify_score_shard_exp473_antigravity,
)
from .config_antigravity import (
    APPROVED_DESIGN_PATH_ANTIGRAVITY,
    EXP473_OFFICIAL_ROOT_ANTIGRAVITY,
    EXP473_SMOKE_ROOT_ANTIGRAVITY,
    FROZEN_EXP473_CONFIG_ANTIGRAVITY,
)
from .contracts_antigravity import (
    Exp473ScoreShardAntigravity,
    canonical_json_bytes_exp473_antigravity,
)
from .runner_antigravity import run_score_wave_exp473_antigravity
from .scoring_antigravity import (
    compose_final_score_exp473_antigravity,
    score_dataset_condition_d_prime_exp473_antigravity,
)


_CODE_CLOSURE_PATHS_ANTIGRAVITY = (
    "experiments/exp473_contemporary_pchip_control_antigravity/config_antigravity.py",
    "experiments/exp473_contemporary_pchip_control_antigravity/contracts_antigravity.py",
    "experiments/exp473_contemporary_pchip_control_antigravity/scoring_antigravity.py",
    "experiments/exp473_contemporary_pchip_control_antigravity/artifacts_antigravity.py",
    "experiments/exp473_contemporary_pchip_control_antigravity/runner_antigravity.py",
    "experiments/exp473_contemporary_pchip_control_antigravity/runtime_antigravity.py",
    "experiments/exp473_contemporary_pchip_control_antigravity/cli_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/contracts_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/scoring_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/transforms_antigravity.py",
    "experiments/exp450_resampling_image_ablation_antigravity/imaging_antigravity.py",
    "experiments/exp472_makima_upsampling_ablation_antigravity/artifacts_antigravity.py",
    "experiments/exp472_makima_upsampling_ablation_antigravity/contracts_antigravity.py",
    "experiments/exp472_makima_upsampling_ablation_antigravity/transforms_antigravity.py",
    "paper/equipment_anomaly_detection/catalog_paper.py",
    "paper/equipment_anomaly_detection/data_paper.py",
    "paper/equipment_anomaly_detection/contracts_paper.py",
    "paper/equipment_anomaly_detection/routing_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py",
    "paper/equipment_anomaly_detection/sources_paper/stability_paper.py",
    "docs/codemap/codemap_exp473_antigravity.lock",
)


def _repository_root_exp473_antigravity() -> Path:
    return Path(__file__).resolve().parents[2]


def _code_closure_exp473_antigravity() -> dict[str, str]:
    root = _repository_root_exp473_antigravity()
    return {
        relative: sha256((root / relative).read_bytes()).hexdigest()
        for relative in _CODE_CLOSURE_PATHS_ANTIGRAVITY
    }


def test_identity_from_catalog_exp473_antigravity(
    test_entry: Mapping[str, object],
) -> dict[str, object]:
    """Project current Task8 identity into the exact Exp472-E TEST identity."""
    entry = dict(test_entry)
    required = {
        "dataset", "dataset_id", "series_length", "split", "indices",
        "values_blob_sha256", "row_digest", "count",
    }
    if (
        set(entry) != required
        or type(entry["dataset"]) is not str
        or type(entry["series_length"]) is not int
        or entry["split"] != "TEST"
        or type(entry["count"]) is not int
        or type(entry["indices"]) is not list
        or entry["indices"] != list(range(entry["count"]))
        or type(entry["values_blob_sha256"]) is not list
        or len(entry["values_blob_sha256"]) != entry["count"]
    ):
        raise ValueError("Exp473 current TEST catalog identity differs")
    bare = {
        "dataset": entry["dataset"],
        "series_length": entry["series_length"],
        "split": "TEST",
        "indices": entry["indices"],
        "values_blob_sha256": entry["values_blob_sha256"],
    }
    return {
        **bare,
        "row_digest": sha256(
            canonical_json_bytes_exp472_antigravity(bare)
        ).hexdigest(),
        "count": entry["count"],
    }


def _canonical_exp472_record_antigravity(path: Path) -> dict[str, object]:
    details = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ValueError("Exp473 referenced Exp472 artifact is unsafe")
    raw = path.read_bytes()
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp473 referenced Exp472 artifact is invalid") from error
    if not isinstance(record, dict) or raw != canonical_json_bytes_exp472_antigravity(record):
        raise ValueError("Exp473 referenced Exp472 artifact is noncanonical")
    return record


def _official_e_input_authority_exp473_antigravity(
    catalog: tuple[Exp472CatalogEntryAntigravity, ...],
) -> dict[str, str]:
    """Bind D-prime coordinates directly to completed E shard value hashes."""
    manifest = _canonical_exp472_record_antigravity(
        EXP472_OFFICIAL_ROOT_ANTIGRAVITY / "exp472_score_manifest_antigravity.json"
    )
    manifest_bare = dict(manifest)
    manifest_digest = manifest_bare.pop("manifest_sha256", None)
    descriptors = manifest.get("descriptors")
    full_order = manifest.get("dataset_order")
    if (
        manifest.get("schema") != "exp472-score-manifest-antigravity.v1"
        or manifest.get("mode") != "full"
        or manifest.get("database_sha256")
        != FROZEN_EXP473_CONFIG_ANTIGRAVITY.database_sha256
        or type(full_order) is not list
        or len(full_order) != 1117
        or type(descriptors) is not list
        or len(descriptors) != 1117
        or manifest_digest
        != sha256(canonical_json_bytes_exp472_antigravity(manifest_bare)).hexdigest()
    ):
        raise ValueError("Exp473 Exp472-E manifest authority differs")
    by_descriptor = {
        item["dataset"]: item["shard_sha256"]
        for item in descriptors
        if isinstance(item, dict)
        and set(item) == {"dataset", "shard_sha256"}
        and type(item["dataset"]) is str
        and type(item["shard_sha256"]) is str
    }
    if len(by_descriptor) != 1117 or tuple(by_descriptor) != tuple(full_order):
        raise ValueError("Exp473 Exp472-E descriptor coverage differs")
    for entry in catalog:
        token = sha256(entry.dataset.encode("utf-8")).hexdigest()
        path = (
            EXP472_OFFICIAL_ROOT_ANTIGRAVITY / "score_shards_antigravity"
            / f"{token}_antigravity" / "score_shard_antigravity.json"
        )
        shard = _canonical_exp472_record_antigravity(path)
        shard_bare = dict(shard)
        shard_digest = shard_bare.pop("shard_sha256", None)
        if (
            entry.dataset not in by_descriptor
            or shard.get("schema") != "exp472-score-shard-antigravity.v1"
            or shard.get("context_sha256") != manifest.get("context_sha256")
            or shard.get("catalog_sha256") != manifest.get("catalog_sha256")
            or shard.get("database_sha256") != manifest.get("database_sha256")
            or shard.get("condition")
            != {"id": "E", "downsampling": "polyphase", "upsampling": "makima", "image_mode": "area"}
            or shard.get("dataset") != entry.dataset
            or shard.get("test_identity") != dict(entry.test_identity)
            or shard_digest != by_descriptor[entry.dataset]
            or shard_digest
            != sha256(canonical_json_bytes_exp472_antigravity(shard_bare)).hexdigest()
        ):
            raise ValueError(f"Exp473 Exp472-E input binding differs for {entry.dataset}")
    return {
        "manifest_sha256": str(manifest_digest),
        "context_sha256": str(manifest["context_sha256"]),
        "catalog_sha256": str(manifest["catalog_sha256"]),
    }


def _current_catalog_exp473_antigravity(
    mode: str,
) -> tuple[dict[str, object], tuple[Exp472CatalogEntryAntigravity, ...]]:
    smoke = FROZEN_EXP473_CONFIG_ANTIGRAVITY.smoke_datasets
    catalog = build_catalog(
        FROZEN_EXP473_CONFIG_ANTIGRAVITY.database_path,
        mode=mode,
        smoke_datasets=smoke if mode == "smoke" else (),
    )
    database = catalog.get("database")
    entries = catalog.get("datasets")
    expected = len(smoke) if mode == "smoke" else 1117
    if (
        not isinstance(database, dict)
        or database.get("sha256") != FROZEN_EXP473_CONFIG_ANTIGRAVITY.database_sha256
        or not isinstance(entries, list)
        or len(entries) != expected
    ):
        raise ValueError("Exp473 current values catalog differs")
    typed = tuple(
        Exp472CatalogEntryAntigravity(
            entry["dataset"], test_identity_from_catalog_exp473_antigravity(entry["test"])
        )
        for entry in entries
    )
    return catalog, typed


def _load_one_raw_exp473_antigravity(
    catalog_database: Mapping[str, object],
    dataset: str,
) -> RawDataset:
    binding = bind_preverified_database_read_only_paper(
        FROZEN_EXP473_CONFIG_ANTIGRAVITY.database_path,
        verified_catalog_database=catalog_database,
    )
    return load_dataset_values(
        FROZEN_EXP473_CONFIG_ANTIGRAVITY.database_path,
        dataset,
        preverified_database=binding,
    )


def _score_one_dataset_exp473_antigravity(
    catalog_database: Mapping[str, object],
    backend: Exp472BackendDecisionAntigravity,
    dataset: str,
) -> Exp473ScoreShardAntigravity:
    values = _load_one_raw_exp473_antigravity(catalog_database, dataset)
    result = score_dataset_condition_d_prime_exp473_antigravity(values, backend)
    return compose_final_score_exp473_antigravity(values, result)


@dataclass(frozen=True, slots=True)
class Exp473RuntimeAntigravity:
    mode: str
    root: Path
    catalog: tuple[Exp472CatalogEntryAntigravity, ...]
    catalog_database: Mapping[str, object]
    backend: Exp472BackendDecisionAntigravity
    context_bare: Mapping[str, object]
    context_sha256: str

    def __post_init__(self) -> None:
        expected_root = (
            EXP473_SMOKE_ROOT_ANTIGRAVITY if self.mode == "smoke"
            else EXP473_OFFICIAL_ROOT_ANTIGRAVITY if self.mode == "full" else None
        )
        expected_count = (
            len(FROZEN_EXP473_CONFIG_ANTIGRAVITY.smoke_datasets)
            if self.mode == "smoke" else 1117
        )
        entries = tuple(self.catalog)
        if (
            expected_root is None
            or Path(self.root) != expected_root
            or len(entries) != expected_count
            or not all(isinstance(entry, Exp472CatalogEntryAntigravity) for entry in entries)
            or not isinstance(self.backend, Exp472BackendDecisionAntigravity)
        ):
            raise ValueError("Exp473 runtime identity differs")
        bare = dict(self.context_bare)
        if self.context_sha256 != sha256(canonical_json_bytes_exp473_antigravity(bare)).hexdigest():
            raise ValueError("Exp473 runtime context digest differs")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "catalog", entries)
        object.__setattr__(self, "catalog_database", dict(self.catalog_database))
        object.__setattr__(self, "context_bare", bare)

    @property
    def catalog_antigravity(self) -> tuple[Exp472CatalogEntryAntigravity, ...]:
        return self.catalog

    @property
    def dataset_order_antigravity(self) -> tuple[str, ...]:
        return tuple(entry.dataset for entry in self.catalog)

    def _ensure_context_antigravity(self) -> dict[str, object]:
        try:
            record = verify_run_context_exp473_antigravity(self.root)
        except FileNotFoundError:
            record = publish_run_context_exp473_antigravity(self.root, dict(self.context_bare))
        if record.get("context_sha256") != self.context_sha256:
            raise ValueError("Exp473 persisted context differs")
        return record

    def run_exp473_antigravity(self) -> dict[str, object]:
        self._ensure_context_antigravity()
        score_one = partial(
            _score_one_dataset_exp473_antigravity,
            dict(self.catalog_database),
            self.backend,
        )
        return run_score_wave_exp473_antigravity(
            self.root, self.context_sha256, self.dataset_order_antigravity,
            score_one=score_one, cpu_workers=self.backend.cpu_workers,
            tensor_backend=self.backend.tensor_backend,
            expected_identities={
                entry.dataset: dict(entry.test_identity) for entry in self.catalog
            },
        )

    def verify_exp473_antigravity(self) -> dict[str, object]:
        context = verify_run_context_exp473_antigravity(self.root)
        if context.get("context_sha256") != self.context_sha256:
            raise ValueError("Exp473 persisted context differs")
        manifest = verify_manifest_exp473_antigravity(
            self.root, self.context_sha256, self.dataset_order_antigravity
        )
        for dataset in self.dataset_order_antigravity:
            verify_score_shard_exp473_antigravity(self.root, self.context_sha256, dataset)
        return {
            "context_sha256": self.context_sha256,
            "manifest_sha256": manifest["manifest_sha256"],
            "completed": len(self.dataset_order_antigravity),
            "failed": 0,
            "remaining": 0,
        }


def build_runtime_exp473_antigravity(mode: str) -> Exp473RuntimeAntigravity:
    if mode not in {"smoke", "full"}:
        raise ValueError("Exp473 runtime mode differs")
    numba_cache_dir = os.environ.setdefault(
        "NUMBA_CACHE_DIR", "/private/tmp/exp473_numba_cache_antigravity"
    )
    current_catalog, typed_catalog = _current_catalog_exp473_antigravity(mode)
    e_authority = _official_e_input_authority_exp473_antigravity(typed_catalog)
    backend = assess_exp472_backend_antigravity()
    root = EXP473_SMOKE_ROOT_ANTIGRAVITY if mode == "smoke" else EXP473_OFFICIAL_ROOT_ANTIGRAVITY
    design_path = _repository_root_exp473_antigravity() / APPROVED_DESIGN_PATH_ANTIGRAVITY
    bare: dict[str, object] = {
        "schema": "exp473-run-context-antigravity.v1",
        "experiment": "Exp473", "mode": mode, "root": str(root),
        "database_sha256": FROZEN_EXP473_CONFIG_ANTIGRAVITY.database_sha256,
        "current_catalog_sha256": current_catalog["catalog_sha256"],
        "exp473_catalog_sha256": catalog_sha256_exp472_antigravity(typed_catalog),
        "catalog": [entry.artifact_record() for entry in typed_catalog],
        "exp472_e_input_authority": e_authority,
        "condition": {"control_id": "D_prime", "base_condition_id": "D", "length_mode": "polyphase_pchip", "image_mode": "area"},
        "policy_id": "hard_plus_stable", "score_threshold": 0.5,
        "backend": backend.artifact_record(), "numba_cache_dir": numba_cache_dir,
        "code_closure": _code_closure_exp473_antigravity(),
        "approved_design_path": APPROVED_DESIGN_PATH_ANTIGRAVITY.as_posix(),
        "approved_design_sha256": sha256(design_path.read_bytes()).hexdigest(),
        "labels_opened": False,
    }
    digest = sha256(canonical_json_bytes_exp473_antigravity(bare)).hexdigest()
    return Exp473RuntimeAntigravity(
        mode, root, typed_catalog, current_catalog["database"], backend, bare, digest
    )


__all__ = [
    "Exp473RuntimeAntigravity",
    "build_runtime_exp473_antigravity",
    "test_identity_from_catalog_exp473_antigravity",
]
