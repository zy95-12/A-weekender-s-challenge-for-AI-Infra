import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.summarize_baseline import main


class SummaryMetricTests(unittest.TestCase):
    def test_percentiles_repeats_and_completion_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"matrix_config.json").write_text(json.dumps({"pairs":"1:1","workloads":[[512,3]],
                "concurrency":"1","repeats":3,"skip_profile":True,"split":"1:3"}))
            for r in range(3):
                point = root/f"tp_1_1/isl_512_osl_3_c_1_r_{r}"
                (point/"qps_inf").mkdir(parents=True)
                (point/"summary.json").write_text(json.dumps({"points":[{"achieved_qps":1+r,
                    "mean_ttft_ms":150,"mean_tpot_ms":35}]}))
                (point/"qps_inf/benchmark.json").write_text(json.dumps({"ttfts":[.1,.2],
                    "itls":[[.02,.03],[.04,.05]],"output_lens":[3,3],"completed":2}))
            for status in ("RUNNING","COMPLETED"):
                (root/"completion.json").write_text(json.dumps({"result":status}))
                with patch.object(sys,"argv",["summary",str(root)]):
                    main()
                report = (root/"REPORT.md").read_text()
                self.assertIn("状态：矩阵完成" if status=="COMPLETED" else "状态：部分结果",report)
            with (root/"baseline_summary.csv").open() as file:
                row = next(csv.DictReader(file))
            self.assertEqual(int(row["requests"]),6)
            self.assertEqual(int(row["itl_samples"]),12)
            self.assertAlmostEqual(float(row["itl_p50_ms"]),35)
            self.assertAlmostEqual(float(row["itl_max_ms"]),50)
            self.assertAlmostEqual(float(row["qps_repeat_std"]),1)
            self.assertAlmostEqual(float(row["ttft_repeat_mean_std_ms"]),0)


if __name__ == "__main__":
    unittest.main()
