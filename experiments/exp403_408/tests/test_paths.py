from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPOSITORY = Path(__file__).resolve().parents[3]
PROBE = """
import json
from experiments.exp403_408 import feature_registry, foundation, pipeline
print(json.dumps({
    "data_root": str(foundation.DATA_ROOT),
    "db_path": str(foundation.DB_PATH),
    "exp291": str(foundation.EXP291_PREDICTION_PATH),
    "exp292": str(foundation.EXP292_PREDICTION_PATH),
    "feature_roots": sorted({str(source.root) for source in feature_registry.FEATURE_SOURCES}),
    "output_dir": str(pipeline.DEFAULT_OUTPUT_DIR),
}))
"""


class PortablePathTests(unittest.TestCase):
    def _probe(self, data_root: str | None) -> dict[str, object]:
        environment = dict(os.environ)
        if data_root is None:
            environment.pop("TIMESERIES_DATA_ROOT", None)
        else:
            environment["TIMESERIES_DATA_ROOT"] = data_root
        completed = subprocess.run(
            [sys.executable, "-c", PROBE],
            cwd=REPOSITORY,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(completed.stdout)

    def test_unset_environment_preserves_historical_default(self) -> None:
        result = self._probe(None)
        self.assertEqual(result["data_root"], "/Users/minho/Documents/Dataset")
        self.assertEqual(
            result["db_path"], "/Users/minho/Documents/Dataset/univariate_ts.db"
        )

    def test_override_rebinds_every_exp403_408_path(self) -> None:
        root = "/Users/mac/Documents/Dataset"
        result = self._probe(root)
        self.assertEqual(result["data_root"], root)
        self.assertEqual(result["db_path"], f"{root}/univariate_ts.db")
        self.assertEqual(
            result["exp291"],
            f"{root}/exp291_final_operational_routing_freeze/"
            "exp291_frozen_operational_routing.jsonl",
        )
        self.assertEqual(
            result["exp292"],
            f"{root}/exp292_standard_review_evidence_tiering/"
            "exp292_frozen_review_tiers.jsonl",
        )
        self.assertTrue(
            all(str(path).startswith(f"{root}/") for path in result["feature_roots"])
        )
        self.assertEqual(
            result["output_dir"],
            f"{root}/exp403_408_feature_ml_validation",
        )


if __name__ == "__main__":
    unittest.main()
