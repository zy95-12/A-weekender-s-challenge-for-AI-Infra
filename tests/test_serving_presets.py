import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PresetTests(unittest.TestCase):
    def config(self, *flags):
        p = subprocess.run([sys.executable, str(ROOT/'scripts/manage.py'), 'up',
                            '--print-config', *flags], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout)

    def test_baseline_disables_optimization_paths(self):
        c = self.config('--preset', 'baseline')
        for key in ('pd', 'wire_fast', 'pd_chunk_transfer', 'pipeline_window',
                    'prefill_chunk_size', 'tcp_buffer_mib', 'pd_prefill_window'):
            self.assertFalse(c[key], key)
        self.assertEqual((c['tp'], c['ipc_mode'], c['scheduler_policy']), (2, 'pipe', 'legacy'))

    def test_optimized_matches_measured_configuration(self):
        c = self.config('--preset', 'optimized')
        self.assertTrue(c['pd'] and c['wire_fast'] and c['pd_control_channel'])
        self.assertEqual((c['prefill_replicas'], c['prefill_tp'], c['decode_tp']), (2, 1, 1))
        self.assertEqual((c['prefill_chunk_size'], c['pd_prefill_window'], c['pipeline_window']), (2048, 3, 2))
        self.assertFalse(c['pd_chunk_transfer'])

    def test_explicit_flags_override_preset_in_either_order(self):
        for flags in [('--preset','optimized','--no-wire-fast','--no-pd-control-channel'),
                      ('--no-wire-fast','--no-pd-control-channel','--preset','optimized')]:
            c = self.config(*flags)
            self.assertFalse(c['wire_fast'] or c['pd_control_channel'])
        c = self.config('--preset','optimized','--prefill-replicas','1','--prefill-tp','2')
        self.assertEqual((c['prefill_replicas'], c['prefill_tp']), (1,2))

    def test_invalid_dependency_is_rejected(self):
        p = subprocess.run([sys.executable, str(ROOT/'scripts/manage.py'), 'up',
                            '--preset','optimized','--no-pd','--print-config'], capture_output=True)
        self.assertNotEqual(p.returncode, 0)


if __name__ == '__main__':
    unittest.main()
