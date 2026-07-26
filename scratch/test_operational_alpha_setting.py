import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from operational.alpha_setting import (
    ALLOWED_ALPHA_VALUES,
    apply_pending_alpha_setting,
    load_effective_alpha,
)


class FakeClient:
    def __init__(self, setting):
        self.setting = setting
        self.acknowledgements = []

    def fetch_setting(self, access_token, source_id):
        return self.setting

    def acknowledge(self, access_token, source_id, version, alpha, success, error_message=None):
        self.acknowledgements.append({
            "access_token": access_token,
            "source_id": source_id,
            "version": version,
            "alpha": alpha,
            "success": success,
            "error_message": error_message,
        })
        return {"status": "applied" if success else "failed"}


class OperationalAlphaSettingTests(unittest.TestCase):
    def test_only_frozen_alpha_values_are_allowed(self):
        self.assertEqual(ALLOWED_ALPHA_VALUES, (0.005, 0.01))

    def test_pending_setting_is_written_atomically_before_success_ack(self):
        client = FakeClient({
            "source_id": "macbook-air",
            "desired_alpha": 0.01,
            "effective_alpha": 0.005,
            "status": "pending",
            "version": 4,
        })
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "alpha.json"
            with mock.patch("operational.alpha_setting.os.replace", wraps=__import__("os").replace) as replace:
                result = apply_pending_alpha_setting(client, "token", "macbook-air", target)
            self.assertEqual(result["status"], "applied")
            self.assertEqual(load_effective_alpha(target), 0.01)
            self.assertTrue(json.loads(target.read_text())["automatic_hard_alerts_unchanged"])
            self.assertEqual(replace.call_count, 1)
        self.assertEqual(client.acknowledgements[0]["version"], 4)
        self.assertEqual(client.acknowledgements[0]["alpha"], 0.01)
        self.assertTrue(client.acknowledgements[0]["success"])

    def test_applied_setting_repairs_a_missing_local_file_without_new_ack(self):
        client = FakeClient({
            "source_id": "macbook-air",
            "desired_alpha": 0.005,
            "effective_alpha": 0.005,
            "status": "applied",
            "version": 2,
        })
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "alpha.json"
            result = apply_pending_alpha_setting(client, "token", "macbook-air", target)
            self.assertEqual(result["status"], "repaired")
            self.assertEqual(load_effective_alpha(target), 0.005)
        self.assertEqual(client.acknowledgements, [])

    def test_invalid_remote_alpha_is_rejected_without_local_application(self):
        client = FakeClient({
            "source_id": "macbook-air",
            "desired_alpha": 0.02,
            "status": "pending",
            "version": 3,
        })
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "alpha.json"
            result = apply_pending_alpha_setting(client, "token", "macbook-air", target)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(target.exists())
        self.assertEqual(client.acknowledgements, [])
        self.assertIn("unsupported alpha", result["error"])

    def test_loader_rejects_a_corrupt_or_mismatched_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "alpha.json"
            target.write_text('{"effective_alpha": 0.02}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_effective_alpha(target)


if __name__ == "__main__":
    unittest.main()
