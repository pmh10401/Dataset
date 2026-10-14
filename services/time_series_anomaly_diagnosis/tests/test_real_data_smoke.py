import math
from pathlib import Path
import unittest

from services.time_series_anomaly_diagnosis.anomaly_detection import (
    calculate_evaluation_metrics,
    run_anomaly_detection,
)
from services.time_series_anomaly_diagnosis.dataset_repository import (
    get_dataset_catalog,
    load_dataset_values,
    load_test_labels,
    open_source_database,
)
from services.time_series_anomaly_diagnosis.model_resources import load_model_resources

SOURCE_DATABASE = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
RESOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "resources"

REPRESENTATIVES = (
    ("ACSF1_normal_0", "minirocket_local_gap", "standard"),
    ("EOGVerticalSignal_normal_12", "hydra_local_gap", "standard"),
    ("DistalPhalanxTW_normal_7", "multirocket_local_gap", "standard"),
    ("AllGestureWiimoteX_normal_1", "minirocket_local_gap", "recovery_numba"),
)

EXPECTED_AUC = {
    "ACSF1_normal_0": 1.0,
    "EOGVerticalSignal_normal_12": 1.0,
    "DistalPhalanxTW_normal_7": 0.7142857142857143,
    "AllGestureWiimoteX_normal_1": 0.9540816326530612,
}
EXPECTED_CANDIDATES = {
    "ACSF1_normal_0": ((), ()),
    "EOGVerticalSignal_normal_12": ((49,), (49,)),
    "DistalPhalanxTW_normal_7": ((), ()),
    "AllGestureWiimoteX_normal_1": ((), ()),
}


class RealDataSmokeTest(unittest.TestCase):
    def test_representative_routes_produce_finite_aligned_results(self) -> None:
        resources = load_model_resources(RESOURCE_DIRECTORY)
        with open_source_database(SOURCE_DATABASE) as connection:
            catalog = {
                row["dataset_name"]: row
                for row in get_dataset_catalog(connection, resources)
            }
            for dataset_name, method_name, availability in REPRESENTATIVES:
                self.assertEqual(catalog[dataset_name]["method_name"], method_name)
                self.assertEqual(catalog[dataset_name]["availability"], availability)
                prepared = load_dataset_values(connection, dataset_name, availability)
                detection = run_anomaly_detection(prepared, method_name)
                labels = load_test_labels(connection, dataset_name, prepared.test_indices)
                metrics = calculate_evaluation_metrics(detection.rows, labels)
                self.assertTrue(all(math.isfinite(row["score"]) for row in detection.rows))
                self.assertEqual(len(detection.rows), len(prepared.test_indices))
                self.assertTrue(math.isfinite(metrics["auc"]))
                self.assertAlmostEqual(metrics["auc"], EXPECTED_AUC[dataset_name], delta=1e-12)
                self.assertEqual(
                    (
                        detection.automatic_anomaly_indices,
                        detection.resolution_limited_warning_indices,
                    ),
                    EXPECTED_CANDIDATES[dataset_name],
                )


if __name__ == "__main__":
    unittest.main()
