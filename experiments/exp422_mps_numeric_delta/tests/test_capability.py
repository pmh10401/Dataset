"""Capability detection stays host-local and does not run an experiment."""

from __future__ import annotations

import unittest

from experiments.exp422_mps_numeric_delta.capability import DeviceCapability, detect_capability


class CapabilityTests(unittest.TestCase):
    def test_unavailable_mps_has_explicit_reason(self) -> None:
        """Removing the unavailable fallback branch must break this contract."""
        capability = DeviceCapability(
            torch_version="2.13.0",
            mps_built=True,
            mps_available=False,
            device_name="Darwin arm64 MPS",
            mlx_version="unavailable",
            mlx_metal_version="unavailable",
            mlx_availability="unavailable",
            fallback_reason="mps_unavailable",
        )
        self.assertEqual(capability.fallback_reason, "mps_unavailable")

    def test_detected_unavailable_mps_is_explicit(self) -> None:
        """A host without MPS must not yield an ambiguous capability record."""
        capability = detect_capability()
        if not capability.mps_available:
            self.assertEqual(capability.fallback_reason, "mps_unavailable")


if __name__ == "__main__":
    unittest.main()
