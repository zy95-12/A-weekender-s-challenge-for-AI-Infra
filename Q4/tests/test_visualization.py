from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from split_serving_sim.visualization import render_gantt_html


class VisualizationTest(unittest.TestCase):
    def test_writes_standalone_gantt(self) -> None:
        trace = [
            {
                "batch_id": 0,
                "resource": "edge_gpu",
                "stage": "edge_front",
                "phases": ["prefill"],
                "request_ids": [0, 1],
                "start_time_ms": 0.0,
                "end_time_ms": 3.0,
                "duration_ms": 3.0,
                "total_tokens": 32,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "gantt.html"
            render_gantt_html(trace, target)
            rendered = target.read_text(encoding="utf-8")
        self.assertIn("Split-serving pipeline", rendered)
        self.assertIn("requests=[0, 1]", rendered)
        self.assertIn("<svg", rendered)


if __name__ == "__main__":
    unittest.main()
