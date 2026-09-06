from __future__ import annotations

import json
import unittest
from pathlib import Path


class Issue6SweepConfigTest(unittest.TestCase):
    def test_all_entry_paths_and_concurrency_levels_are_declared(self) -> None:
        q4 = Path(__file__).parents[1]
        path = q4 / "configs" / "issue6_qps_sweep.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["concurrency"], [1, 2, 4, 8, 12, 16])
        for relative in data["variants"].values():
            self.assertTrue((path.parent / relative).is_file())
        self.assertTrue((path.parent / data["evidence"]).is_file())
        self.assertLessEqual(data["memory_limit_mib"], 768)


if __name__ == "__main__":
    unittest.main()
