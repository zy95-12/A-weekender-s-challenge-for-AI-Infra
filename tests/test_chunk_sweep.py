import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import chunk_sweep


class SweepTests(unittest.TestCase):
    def test_correctness_precedes_benchmark_and_restore_disables_chunk(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"scan"
            argv = ["chunk_sweep","--output",str(out),"--sizes","0,512","--policies","legacy",
                    "--ipc-mode","shm","--wire-fast","--tcp-buffer-mib","16"]
            with patch.object(sys,"argv",argv), patch.object(chunk_sweep,"run") as run:
                chunk_sweep.main()
            names = [call.args[0] for call in run.call_args_list]
            self.assertEqual(names[:5],["up","correctness","isl_8192_osl_256_c_1_r_0",
                                       "isl_8192_osl_256_c_4_r_0","mixed"])
            self.assertEqual(names[-1],"restore_serial_unchunked")
            self.assertNotIn("--prefill-chunk-size",run.call_args_list[-1].args[1])
            self.assertIn("--wire-fast",run.call_args_list[-1].args[1])
            self.assertEqual(json.loads((out/"sweep.json").read_text())["result"],"PASS")

    def test_failed_correctness_stops_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"scan"
            def run(name,command,output):
                if name == "correctness":
                    raise RuntimeError("numerical mismatch")
            with patch.object(sys,"argv",["chunk_sweep","--output",str(out)]), \
                    patch.object(chunk_sweep,"run",side_effect=run) as calls:
                with self.assertRaisesRegex(RuntimeError,"numerical mismatch"):
                    chunk_sweep.main()
            self.assertEqual([call.args[0] for call in calls.call_args_list],["up","correctness"])
            status = json.loads((out/"sweep.json").read_text())
            self.assertEqual(status["result"],"FAIL")
            self.assertEqual(status["completed"],[])


if __name__ == "__main__":
    unittest.main()
