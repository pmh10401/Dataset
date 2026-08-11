import unittest

from experiments.exp449_residual_fn_method_qualification_antigravity.config_antigravity import (
    CANDIDATE_MATRIX_ANTIGRAVITY,
    FROZEN_CONFIG_ANTIGRAVITY,
)


class ConfigAntigravityTest(unittest.TestCase):
    def test_candidate_matrix_is_exact_antigravity(self):
        self.assertEqual(CANDIDATE_MATRIX_ANTIGRAVITY, (
            ("Crop", ("minirocket_local_gap", "multirocket_local_gap")),
            ("ElectricDevices", ("multirocket_local_gap", "hydra_multirocket_local_gap")),
            ("FordA", ("minirocket_local_gap", "hydra_local_gap")),
            ("FordB", ("multirocket_local_gap", "hydra_local_gap")),
            ("Yoga", ("multirocket_local_gap", "hydra_multirocket_local_gap")),
            ("ECG5000", ("multirocket_local_gap", "hydra_local_gap")),
            ("ChlorineConcentration", ("minirocket_local_gap", "multirocket_local_gap")),
            ("StarLightCurves", ("multirocket_local_gap", "hydra_local_gap")),
        ))

    def test_frozen_config_binds_environment_and_score_settings_antigravity(self):
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["python_version"], "3.14.6")
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["aeon_version"], "1.4.0")
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["numpy_version"], "2.3.5")
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["sklearn_version"], "1.7.2")
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["seeds"], (20260721, 20261721))
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["alphas"], (0.005,))


if __name__ == "__main__":
    unittest.main()
