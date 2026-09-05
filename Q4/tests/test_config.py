from __future__ import annotations

import unittest

from split_serving_sim.config import ConfigError, parse_config

from tests.helpers import copied_toy_config


class ConfigTest(unittest.TestCase):
    def test_valid_config(self) -> None:
        config = parse_config(copied_toy_config())
        self.assertEqual(config.model.name, "toy")
        self.assertEqual([stage.num_layers for stage in config.stages], [1, 2, 1])

    def test_rejects_non_contiguous_split(self) -> None:
        data = copied_toy_config()
        data["topology"]["stages"][1]["layer_start"] = 2
        with self.assertRaisesRegex(ConfigError, "contiguous"):
            parse_config(data)

    def test_rejects_chunk_larger_than_token_budget(self) -> None:
        data = copied_toy_config()
        data["static_policy"]["prefill_chunk_size"] = 256
        with self.assertRaisesRegex(ConfigError, "prefill_chunk_size"):
            parse_config(data)

    def test_accepts_batch_size_input_name(self) -> None:
        data = copied_toy_config()
        data["static_policy"]["batch_size"] = data["static_policy"].pop(
            "max_batch_size"
        )
        config = parse_config(data)
        self.assertEqual(config.static_policy.max_batch_size, 4)


if __name__ == "__main__":
    unittest.main()
