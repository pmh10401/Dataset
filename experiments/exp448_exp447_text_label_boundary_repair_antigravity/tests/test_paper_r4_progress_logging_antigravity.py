"""Result-neutral paper r4 progress-callback contracts."""
from __future__ import annotations

from io import StringIO
import unittest


class ProgressLoggingAntigravityTests(unittest.TestCase):
    def test_rate_gate_emits_start_end_error_immediately_and_ticks_at_most_once_per_15s(self) -> None:
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            make_stderr_progress_sink_paper,
        )

        stamps = iter((0.0, 0.0, 1.0, 14.9, 15.0, 15.1, 20.0))
        stream = StringIO()
        sink = make_stderr_progress_sink_paper(clock=lambda: next(stamps), stream=stream)
        payload = {"stage": "task6", "completed": 0, "total": 3, "elapsed_s": 0.0}
        sink({**payload, "phase": "start", "completed": 0})
        sink({**payload, "phase": "tick", "completed": 0, "elapsed_s": 0.0})
        sink({**payload, "phase": "tick", "completed": 0, "elapsed_s": 1.0})
        sink({**payload, "phase": "tick", "completed": 1, "elapsed_s": 14.9})
        sink({**payload, "phase": "tick", "completed": 2, "elapsed_s": 15.0})
        sink({**payload, "phase": "end", "completed": 3, "elapsed_s": 15.1})
        sink({**payload, "phase": "error", "completed": 3, "elapsed_s": 20.0})
        lines = [line for line in stream.getvalue().splitlines() if line]
        self.assertEqual(sum(line.startswith("task6 0/3") and "0.0%" in line for line in lines[:2]), 2)
        self.assertTrue(any("task6 2/3" in line for line in lines))
        self.assertTrue(any("task6 3/3" in line and "100.0%" in line for line in lines))
        tick_lines = [line for line in lines if "task6 1/3" in line]
        self.assertEqual(tick_lines, [])
        self.assertGreaterEqual(len(lines), 4)
        self.assertLessEqual(len(lines), 5)

    def test_progress_text_firewall_rejects_label_and_score_fields(self) -> None:
        from paper.equipment_anomaly_detection.data_paper import emit_progress_paper
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            make_stderr_progress_sink_paper,
        )

        received: list[dict[str, object]] = []
        emit_progress_paper(
            received.append,
            {
                "stage": "task6",
                "phase": "tick",
                "completed": 1,
                "total": 2,
                "elapsed_s": 1.0,
                "score": 1.23,
                "pvalue": 0.01,
                "label": 1,
                "alerts": [0, 1],
                "detail": "score=1.23",
            },
        )
        self.assertEqual(len(received), 1)
        self.assertEqual(set(received[0]), {"stage", "phase", "completed", "total", "elapsed_s"})
        self.assertNotIn("detail", received[0])
        stream = StringIO()
        sink = make_stderr_progress_sink_paper(clock=lambda: 0.0, stream=stream)
        sink(received[0])
        text = stream.getvalue().lower()
        for token in ("label=", "p-value", "score=", "alerts=", "rank="):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
