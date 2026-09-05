from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from split_serving_sim.visualization import render_gantt_html, render_gantt_svg


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
                "input_shape": "B=2, tokens=[16, 16], hidden=64",
                "sub_operations": [
                    {
                        "name": "attention",
                        "category": "compute",
                        "start_time_ms": 0.0,
                        "end_time_ms": 2.0,
                        "duration_ms": 2.0,
                        "input_shape": "B=2, q=[16, 16], hidden=64",
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "gantt.html"
            render_gantt_html(trace, target)
            rendered = target.read_text(encoding="utf-8")
        self.assertIn("Split-serving pipeline", rendered)
        self.assertIn('"request_ids": [0, 1]', rendered)
        self.assertIn("<svg", rendered)
        self.assertIn("滚轮缩放", rendered)
        self.assertIn("attention", rendered)
        self.assertIn("ArrowLeft", rendered)
        self.assertIn("batchDependencies", rendered)
        self.assertIn("operatorGeometry", rendered)
        self.assertIn("selectedOperator", rendered)
        self.assertIn("计算 / Compute", rendered)

    def test_writes_script_free_svg_for_github_preview(self) -> None:
        trace = [
            {
                "batch_id": 0,
                "resource": "edge_gpu",
                "stage": "edge_front",
                "phases": ["prefill"],
                "request_ids": [0],
                "start_time_ms": 0.0,
                "end_time_ms": 3.0,
                "duration_ms": 3.0,
                "input_shape": "B=1, tokens=[16], hidden=64",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "gantt.svg"
            render_gantt_svg(trace, target)
            rendered = target.read_text(encoding="utf-8")
        self.assertIn("<svg", rendered)
        self.assertIn("edge_gpu", rendered)
        self.assertNotIn("<script", rendered)


if __name__ == "__main__":
    unittest.main()
