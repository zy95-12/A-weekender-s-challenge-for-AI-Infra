import json
import unittest
from scripts.summarize_baseline import network_summary


class NetworkReportTests(unittest.TestCase):
    def test_structured_check(self):
        check = {"result":"PASS","measured":{"rtt_ms":10.012,"throughput_gbps":9.73}}
        self.assertEqual(network_summary(json.dumps(check)),("10.012","9.730"))
        with self.assertRaises(ValueError):
            network_summary(json.dumps({**check,"result":"FAIL"}))

    def test_legacy_and_missing(self):
        log = 'rtt min/avg/max/mdev = 10.0/10.012/10.2/0.01 ms\n'+json.dumps(
            {"end":{"sum_received":{"bits_per_second":9.73e9}}})
        self.assertEqual(network_summary(log),("10.012","9.730"))
        self.assertEqual(network_summary(""),("未采集","未采集"))


if __name__ == "__main__":
    unittest.main()
