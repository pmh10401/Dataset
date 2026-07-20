import unittest

from experiments.exp263_264_full_missing_aware_integration import config


class ConfigurationContractTest(unittest.TestCase):
    def test_policy_constants_are_preregistered(self) -> None:
        self.assertEqual(config.ALPHA, 0.005)
        self.assertEqual(config.MIN_TRAIN_FOR_PROMOTION, 100)
        self.assertEqual(config.REQUIRED_PROMOTION_AGREEMENT, 3)
        self.assertEqual(config.MASK_VARIANT, "mask_025__seed_intersection")

    def test_source_contract_has_exact_frozen_sources(self) -> None:
        self.assertEqual(
            set(config.SOURCE_CONTRACT),
            {
                "baseline",
                "statistical",
                "structural",
                "change_point",
                "reconstruction",
                "missing_override",
            },
        )
        self.assertEqual(
            {
                name: spec.variant
                for name, spec in config.SOURCE_CONTRACT.items()
            },
            {
                "baseline": "final_lanes",
                "statistical": "combined_40",
                "structural": "structural_combined_440",
                "change_point": "change_point_combined_33",
                "reconstruction": "mask_025__seed_intersection",
                "missing_override": "corrected_four_sources",
            },
        )
        self.assertEqual(
            {
                name: spec.alpha for name, spec in config.SOURCE_CONTRACT.items()
            },
            {
                "baseline": None,
                "statistical": 0.005,
                "structural": 0.005,
                "change_point": 0.005,
                "reconstruction": 0.005,
                "missing_override": 0.005,
            },
        )

    def test_database_output_and_source_paths_are_absolute(self) -> None:
        self.assertEqual(
            config.DB_PATH,
            config.Path("/Users/minho/Documents/Dataset/univariate_ts.db"),
        )
        self.assertEqual(
            config.OUTPUT_ROOT,
            config.Path(
                "/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration"
            ),
        )
        self.assertTrue(config.DB_PATH.is_absolute())
        self.assertTrue(config.OUTPUT_ROOT.is_absolute())
        expected_paths = {
            "baseline": "/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion/frozen_predictions.jsonl",
            "statistical": "/Users/minho/Documents/Dataset/exp252_sktime_statistical_features_safe_hurst/frozen_predictions.jsonl",
            "structural": "/Users/minho/Documents/Dataset/exp254_sktime_structural_features/frozen_predictions.jsonl",
            "change_point": "/Users/minho/Documents/Dataset/exp255_change_point_features/frozen_predictions.jsonl",
            "reconstruction": "/Users/minho/Documents/Dataset/exp257_mask_ratio_timeseries_sweep/frozen_predictions.jsonl",
            "missing_override": "/Users/minho/Documents/Dataset/exp260_262_missing_aware_alert/frozen_predictions.jsonl",
        }
        self.assertEqual(
            {
                name: str(spec.artifact_path)
                for name, spec in config.SOURCE_CONTRACT.items()
            },
            expected_paths,
        )
        for path in (config.DB_PATH, config.OUTPUT_ROOT):
            self.assertTrue(path.is_absolute())
        for spec in config.SOURCE_CONTRACT.values():
            self.assertTrue(spec.artifact_path.is_absolute())


if __name__ == "__main__":
    unittest.main()
