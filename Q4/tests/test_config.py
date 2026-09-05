from __future__ import annotations

import unittest
from pathlib import Path

from split_serving_sim.config import ConfigError, load_config, parse_config

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

    def test_loads_checked_in_hugging_face_profile(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "example.json")
        self.assertEqual(config.model.architecture, "qwen3")
        self.assertEqual(config.model.num_layers, 64)
        self.assertEqual(config.model.hidden_size, 5120)
        self.assertEqual(config.model.intermediate_size, 25600)
        self.assertEqual(config.model.num_attention_heads, 64)
        self.assertEqual(config.model.num_key_value_heads, 8)
        self.assertEqual(config.model.dtype, "bfloat16")
        total_parameters = (
            config.model.parameters_per_layer * config.model.num_layers
            + 2 * config.model.vocab_size * config.model.hidden_size
        )
        self.assertEqual(total_parameters, 32_762_118_144)


if __name__ == "__main__":
    unittest.main()
