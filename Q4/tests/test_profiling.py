from __future__ import annotations

import unittest

from split_serving_sim.config import PerformanceProfileConfig, ProfileSampleConfig
from split_serving_sim.profiling import ProfilingDatabase


class ProfilingDatabaseTest(unittest.TestCase):
    def test_exact_shape_overrides_and_type_average_corrects_other_shapes(self) -> None:
        database = ProfilingDatabase(
            PerformanceProfileConfig(
                enabled=True,
                samples=(
                    ProfileSampleConfig(
                        operator_type="mm",
                        signature="shape-a",
                        latency_ms=2.0,
                        roofline_ms=1.0,
                        match=(("tp_degree", "2"),),
                        source="unit-profile",
                    ),
                ),
            )
        )

        exact = database.correct(
            "mm", "shape-a", 0.001, {"tp_degree": 2}
        )
        averaged = database.correct(
            "mm", "shape-b", 0.003, {"tp_degree": 2}
        )
        fallback = database.correct(
            "flash_attention", "shape-a", 0.004, {"tp_degree": 2}
        )

        self.assertAlmostEqual(exact.duration_s, 0.002)
        self.assertTrue(exact.exact)
        self.assertAlmostEqual(averaged.duration_s, 0.006)
        self.assertEqual(averaged.source, "profile_type_average")
        self.assertAlmostEqual(fallback.duration_s, 0.004)
        self.assertEqual(fallback.source, "roofline")


if __name__ == "__main__":
    unittest.main()
