"""Exp473 current values authority and Exp472-E input binding tests."""

from __future__ import annotations

from hashlib import sha256
import os
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset

from experiments.exp472_makima_upsampling_ablation_antigravity.artifacts_antigravity import (
    Exp472CatalogEntryAntigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
    canonical_json_bytes_exp472_antigravity,
)
from experiments.exp473_contemporary_pchip_control_antigravity.config_antigravity import (
    EXP473_SMOKE_ROOT_ANTIGRAVITY,
    P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY,
)
from experiments.exp473_contemporary_pchip_control_antigravity.contracts_antigravity import (
    Exp473ScoreShardAntigravity,
    canonical_json_bytes_exp473_antigravity,
)
import experiments.exp473_contemporary_pchip_control_antigravity.runtime_antigravity as runtime_mod
from experiments.exp473_contemporary_pchip_control_antigravity.runtime_antigravity import (
    Exp473RuntimeAntigravity,
    test_identity_from_catalog_exp473_antigravity,
)


class Exp473RuntimeTestsAntigravity(unittest.TestCase):
    def test_current_catalog_identity_drops_only_dataset_id_to_match_e(self) -> None:
        entry = {
            "dataset": "fixture",
            "dataset_id": 17,
            "series_length": 8,
            "split": "TEST",
            "indices": [0, 1],
            "values_blob_sha256": ["a" * 64, "b" * 64],
            "row_digest": "c" * 64,
            "count": 2,
        }
        bare = {
            "dataset": "fixture",
            "series_length": 8,
            "split": "TEST",
            "indices": [0, 1],
            "values_blob_sha256": ["a" * 64, "b" * 64],
        }
        expected = {
            **bare,
            "row_digest": sha256(
                canonical_json_bytes_exp472_antigravity(bare)
            ).hexdigest(),
            "count": 2,
        }
        self.assertEqual(
            test_identity_from_catalog_exp473_antigravity(entry), expected
        )

    def test_cpu_score_callback_pickle_excludes_raw_population_and_loads_one_dataset(
        self,
    ) -> None:
        id_markers = {
            P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY[0]: "EXP473_ID_A_"
            + os.urandom(24 * 1024).hex(),
            P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY[1]: "EXP473_ID_B_"
            + os.urandom(24 * 1024).hex(),
        }
        runtime = _smoke_runtime_antigravity(id_markers)
        first, second = runtime.dataset_order_antigravity[:2]
        markers = {
            first: b"EXP473_RAW_A_" + os.urandom(48 * 1024),
            second: b"EXP473_RAW_B_" + os.urandom(48 * 1024),
        }
        identities = {
            entry.dataset: dict(entry.test_identity) for entry in runtime.catalog
        }
        loaded: list[str] = []
        captured: dict[str, object] = {}

        def fake_load(_path: object, dataset: str, **_kwargs: object) -> RawDataset:
            loaded.append(dataset)
            marker = markers.get(dataset, b"EXP473_RAW_OTHER")
            train = np.zeros(8, dtype="<f4")
            test = np.zeros(8, dtype="<f4")
            return RawDataset(
                dataset,
                8,
                (0,),
                (train.tobytes(),),
                (train,),
                (0,),
                (marker,),
                (test,),
            )

        def fake_wave(
            _root: object,
            _context_sha256: str,
            catalog: tuple[str, ...],
            *,
            score_one: object,
            cpu_workers: int,
            tensor_backend: str,
            expected_identities: object = None,
        ) -> dict[str, object]:
            captured["score_one"] = score_one
            captured["payload"] = pickle.dumps(score_one)
            captured["tensor_backend"] = tensor_backend
            captured["expected_identities"] = expected_identities
            return {
                "waves": 2,
                "workers": cpu_workers,
                "completed": 0,
                "failed": 0,
                "remaining": len(catalog),
                "manifest_published": False,
            }

        shard = Exp473ScoreShardAntigravity(
            first,
            identities[first],
            np.asarray([0.25], dtype=np.float64),
            frozenset(),
        )
        with (
            patch.object(
                Exp473RuntimeAntigravity,
                "_ensure_context_antigravity",
                return_value={},
            ),
            patch.object(runtime_mod, "load_dataset_values", fake_load),
            patch.object(
                runtime_mod,
                "bind_preverified_database_read_only_paper",
                return_value=object(),
            ),
            patch.object(
                runtime_mod,
                "score_dataset_condition_d_prime_exp473_antigravity",
                return_value=object(),
            ),
            patch.object(
                runtime_mod,
                "compose_final_score_exp473_antigravity",
                return_value=shard,
            ),
            patch.object(
                runtime_mod, "run_score_wave_exp473_antigravity", fake_wave
            ),
        ):
            runtime.run_exp473_antigravity()
            payload = captured["payload"]
            score_one = captured["score_one"]
            self.assertEqual(captured["tensor_backend"], "cpu")
            self.assertEqual(loaded, [])
            self.assertIsInstance(payload, bytes)
            self.assertNotIn(markers[first], payload)
            self.assertNotIn(markers[second], payload)
            self.assertNotIn(id_markers[first].encode("ascii"), payload)
            self.assertNotIn(id_markers[second].encode("ascii"), payload)
            self.assertLess(len(payload), len(markers[first]))
            self.assertLess(len(payload), len(id_markers[first]))
            parent_identities = captured["expected_identities"]
            self.assertEqual(parent_identities[first]["marker"], id_markers[first])
            self.assertEqual(parent_identities[second]["marker"], id_markers[second])
            score_one(first)
        self.assertEqual(loaded, [first])


def _smoke_runtime_antigravity(
    id_markers: dict[str, str] | None = None,
) -> Exp473RuntimeAntigravity:
    markers = {} if id_markers is None else id_markers
    catalog = tuple(
        Exp472CatalogEntryAntigravity(
            dataset,
            {
                "dataset": dataset,
                "series_length": 8,
                "split": "TEST",
                "indices": [0],
                "values_blob_sha256": ["a" * 64],
                "row_digest": "b" * 64,
                "count": 1,
                **({"marker": markers[dataset]} if dataset in markers else {}),
            },
        )
        for dataset in P0_SMOKE_DATASETS_EXP473_ANTIGRAVITY
    )
    bare = {"schema": "exp473-run-context-antigravity.v1", "mode": "smoke"}
    return Exp473RuntimeAntigravity(
        "smoke",
        EXP473_SMOKE_ROOT_ANTIGRAVITY,
        catalog,
        {
            "unresolved_path": "fixture",
            "resolved_path": "/tmp/fixture.db",
            "size": 1,
            "mtime_ns": 1,
            "sha256": "c" * 64,
        },
        Exp472BackendDecisionAntigravity(
            pytorch_available=False,
            pytorch_version="unavailable",
            scipy_available=True,
            scipy_version="test",
            mlx_available=False,
            mlx_version="unavailable",
            mlx_metal_available=False,
            mlx_metal_version="unavailable",
            mps_available=False,
            tensor_backend="cpu",
            logical_cpu_count=1,
            cpu_workers=1,
        ),
        bare,
        sha256(canonical_json_bytes_exp473_antigravity(bare)).hexdigest(),
    )


if __name__ == "__main__":
    unittest.main()
