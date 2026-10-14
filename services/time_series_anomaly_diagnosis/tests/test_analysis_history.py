from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from services.time_series_anomaly_diagnosis.analysis_history import (
    create_analysis_run,
    get_analysis_run,
    list_analysis_runs,
    mark_analysis_running,
    mark_interrupted_runs,
    open_history_database,
    save_completed_analysis,
    save_failed_analysis,
)


class AnalysisHistoryTest(unittest.TestCase):
    def test_completed_results_survive_reopen_and_running_becomes_interrupted(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            first = open_history_database(path)
            completed = create_analysis_run(first, "A", "minirocket_local_gap", "standard")
            mark_analysis_running(first, completed["run_id"])
            save_completed_analysis(first, completed["run_id"], {"rows": [{"series_index": 1}]})

            running = create_analysis_run(first, "B", "hydra_local_gap", "standard")
            mark_analysis_running(first, running["run_id"])

            queued = create_analysis_run(first, "C", "minirocket_local_gap", "standard")
            first.close()

            second = open_history_database(path)
            self.assertEqual(mark_interrupted_runs(second), 2)

            completed_record = get_analysis_run(second, completed["run_id"])
            self.assertIsNotNone(completed_record)
            self.assertEqual(completed_record["status"], "completed")
            self.assertEqual(completed_record["result"]["rows"][0]["series_index"], 1)
            self.assertIsNotNone(completed_record["started_at"])
            self.assertIsNotNone(completed_record["completed_at"])

            running_record = get_analysis_run(second, running["run_id"])
            self.assertIsNotNone(running_record)
            self.assertEqual(running_record["status"], "interrupted")
            self.assertIsNotNone(running_record["started_at"])

            queued_record = get_analysis_run(second, queued["run_id"])
            self.assertIsNotNone(queued_record)
            self.assertEqual(queued_record["status"], "interrupted")
            self.assertIsNone(queued_record["started_at"])

            second.close()

    def test_failed_analysis_saves_error_message_and_completed_at(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            connection = open_history_database(path)
            run = create_analysis_run(connection, "Dataset_fail", "method_x", "standard")
            mark_analysis_running(connection, run["run_id"])
            save_failed_analysis(connection, run["run_id"], "Internal calculation failed: division by zero")

            record = get_analysis_run(connection, run["run_id"])
            self.assertIsNotNone(record)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_message"], "Internal calculation failed: division by zero")
            self.assertIsNotNone(record["started_at"])
            self.assertIsNotNone(record["completed_at"])
            self.assertIsNone(record["result"])
            connection.close()

    def test_save_completed_analysis_rejects_non_finite_json(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            connection = open_history_database(path)
            run = create_analysis_run(connection, "Dataset_nan", "method_y", "standard")
            mark_analysis_running(connection, run["run_id"])

            with self.assertRaises(ValueError):
                save_completed_analysis(connection, run["run_id"], {"score": float("nan")})

            with self.assertRaises(ValueError):
                save_completed_analysis(connection, run["run_id"], {"score": float("inf")})

            with self.assertRaises(ValueError):
                save_completed_analysis(connection, run["run_id"], {"score": float("-inf")})

            connection.close()

    def test_list_analysis_runs_recent_order_limit_200_and_excludes_result(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            connection = open_history_database(path)

            for i in range(210):
                run = create_analysis_run(connection, f"Dataset_{i:03d}", "method_z", "standard")
                if i % 10 == 0:
                    save_completed_analysis(connection, run["run_id"], {"heavy_payload": [i] * 1000})

            runs = list_analysis_runs(connection)
            self.assertEqual(len(runs), 200)

            # Check descending order by requested_at
            requested_times = [item["requested_at"] for item in runs]
            self.assertEqual(requested_times, sorted(requested_times, reverse=True))

            # Check exclusion of result_json and result
            for item in runs:
                self.assertNotIn("result_json", item)
                self.assertNotIn("result", item)
                self.assertIn("run_id", item)
                self.assertIn("dataset_name", item)
                self.assertIn("status", item)

            connection.close()

    def test_get_unknown_run_returns_none(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            connection = open_history_database(path)
            self.assertIsNone(get_analysis_run(connection, "non_existent_run_id"))
            connection.close()
