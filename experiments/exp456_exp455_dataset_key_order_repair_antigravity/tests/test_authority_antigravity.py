from __future__ import annotations

import unittest


class AuthorityAntigravityTest(unittest.TestCase):
    def test_authority_capture_surface_rejects_a_non_exp454_leaf_inventory_antigravity(self) -> None:
        """Break caught: authority capture accepts an incomplete predecessor artifact inventory."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity import authority_antigravity as authority

        with self.assertRaisesRegex(ValueError, "inventory"):
            authority._validate_leaves_antigravity(object(), {})
        self.assertEqual(authority.__all__, ["capture_routing_authority_antigravity"])
