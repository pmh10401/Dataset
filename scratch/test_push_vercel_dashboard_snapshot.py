import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("push_vercel_dashboard_snapshot.py")
SPEC = importlib.util.spec_from_file_location("dashboard_exporter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SnapshotSanitizerTests(unittest.TestCase):
    def test_removes_labels_logs_paths_and_process_ids(self):
        source = {
            "running": True,
            "active_experiment": "exp254",
            "active_label": "Exp254",
            "datasets_done": 2,
            "expected_datasets": 10,
            "progress_percent": 20.0,
            "elapsed_seconds": 10,
            "eta_seconds": 40,
            "current_dataset": "Demo_normal_1",
            "queue_items": [{
                "id": "exp254", "label": "Exp254", "status": "running",
                "datasets_done": 2, "expected_datasets": 10, "progress_percent": 20,
                "stdout_log": "/private/data.log", "detail_csv": "/private/result.csv",
                "pid": 1234, "contains_test_labels": True,
            }],
            "coverage": {"coverage_percent": 20.0},
            "throughput": {"recent_datasets_per_minute": 2.0},
            "runtime_health": {"server_pid": 1234, "heartbeat": {
                "age_seconds": 1, "pid": 1234, "stdout_log": "/private/nested.log",
                "contains_test_labels": True,
            }},
            "system": {"cpu": {"percent": 10}, "gpu": {"percent": 20}},
            "process": {"pid": 1234, "process_count": 2, "cpu": 88, "mem": 4.2,
                        "processes": [{"pid": 1234, "command": "secret"}]},
            "warning_summary": {"total_warnings": 0, "recent": ["/private/warn.log"]},
            "error_summary": {"total_errors": 0, "recent": ["secret"]},
            "last_completed_experiment": "exp253",
            "last_completed_label": "Exp253",
            "completed_results": [{
                "id": "exp253", "label": "Exp253", "mean_f1": 0.5,
                "summary_csv": "/private/summary.csv", "artifact_links": [{"href": "/api/file"}],
            }],
            "log_tail": "raw log",
            "dataset_context": {"test_labels": [0, 1]},
        }
        snapshot = MODULE.sanitize_status(source, source_instance_id="test-mac")
        encoded = MODULE.json.dumps(snapshot, ensure_ascii=False)
        self.assertNotIn("/private", encoded)
        self.assertNotIn("log_tail", encoded)
        self.assertNotIn("contains_test_labels", encoded)
        self.assertNotIn('"pid"', encoded)
        self.assertNotIn("test_labels", encoded)
        self.assertEqual(snapshot["source_instance_id"], "test-mac")
        self.assertEqual(snapshot["queue_items"][0]["id"], "exp254")
        self.assertEqual(snapshot["process_summary"]["process_count"], 2)
        self.assertEqual(snapshot["completed_results"][0]["mean_f1"], 0.5)

    def test_caps_collection_sizes(self):
        source = {
            "running": False, "datasets_done": 0, "expected_datasets": 0,
            "progress_percent": 0, "queue_items": [{"id": str(i)} for i in range(150)],
            "completed_results": [{"id": str(i)} for i in range(150)],
        }
        snapshot = MODULE.sanitize_status(source, source_instance_id="test-mac")
        self.assertLessEqual(len(snapshot["queue_items"]), 100)
        self.assertLessEqual(len(snapshot["completed_results"]), 50)
        self.assertEqual(snapshot["completed_results"][0]["id"], "149")


class SupabaseSessionTests(unittest.TestCase):
    def test_tracks_a_fresh_access_token_without_reauthentication(self):
        session = MODULE.SupabaseSession("https://example.supabase.co", "publishable", "a@example.com", "password")
        session.update_tokens({"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})
        self.assertEqual(session.current_access_token(), "access")


if __name__ == "__main__":
    unittest.main()
