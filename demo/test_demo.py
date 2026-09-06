from __future__ import annotations

import importlib.util
import unittest
from html.parser import HTMLParser
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent


class StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "a" and values.get("href"):
            self.links.append(str(values["href"]))


class DemoTest(unittest.TestCase):
    def test_page_contains_all_six_sections_and_evidence_links(self) -> None:
        parser = StructureParser()
        page = (DEMO_DIR / "index.html").read_text(encoding="utf-8")
        parser.feed(page)
        self.assertTrue({"top", "insight", "security", "system", "optimization", "modeling"} <= parser.ids)
        joined = "\n".join(parser.links)
        self.assertIn("docs/01_tech_insight.md", joined)
        self.assertIn("embedding_attack.py", joined)
        self.assertIn("pull/8", joined)
        self.assertNotIn("研究主线", page)
        for technology in ("Confidential Computing", "数据最小化", "Split Inference", "全同态加密", "安全多方计算"):
            self.assertIn(technology, page)
        self.assertEqual(page.count('class="ratings"'), 5)
        self.assertIn("Hidden State 不安全", page)
        self.assertIn('id="attack-input"', page)
        self.assertIn('id="token-generate"', page)

    def test_backend_contract_rejects_reserved_and_bad_inputs(self) -> None:
        spec = importlib.util.spec_from_file_location("demo_server", DEMO_DIR / "server.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(NotImplementedError):
            module.validate_payload({"model": "DeepSeek-V3"})
        with self.assertRaises(ValueError):
            module.validate_payload({**module.SUPPORTED, "concurrency": 3, "input_tokens": 256, "output_tokens": 12})
        self.assertEqual(
            module.validate_payload({**module.SUPPORTED, "concurrency": 1, "input_tokens": 256, "output_tokens": 12}),
            (1, 256, 12),
        )


if __name__ == "__main__":
    unittest.main()
