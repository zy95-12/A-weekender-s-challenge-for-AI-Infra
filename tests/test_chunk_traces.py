import copy
import unittest
from scripts.verify_baseline_traces import initial_state, check_step, check_complete


def row(phase, query, position, token, emits=True, explicit=True):
    value = {"phase": phase, "batch_size": 1, "context_len": position + query,
             "token_idx": token, "upload_bytes": 8192 * query + 300,
             "download_bytes": 8192 * query + 400}
    if explicit:
        value.update(query_len=query, position_start=position, emits_token=emits)
    return value


class ChunkTraceTests(unittest.TestCase):
    def test_old_and_chunked_traces(self):
        cases = [
            [row("prefill", 5, 0, 0, explicit=False), row("decode", 1, 5, 1, explicit=False)],
            [row("prefill", 2, 0, -1, False), row("prefill", 2, 2, -1, False),
             row("prefill", 1, 4, 0), row("decode", 1, 5, 1)]]
        for rows in cases:
            state = initial_state()
            for entry in rows:
                check_step(entry, state, 5)
            check_complete(state, 5, 2)

    def test_early_output_replay_and_wrong_bytes_rejected(self):
        valid = row("prefill", 2, 0, -1, False)
        for key, value in [("emits_token", True), ("position_start", 2),
                           ("token_idx", 0), ("upload_bytes", 8192 * 5 + 2**20)]:
            with self.assertRaises(AssertionError):
                check_step({**valid, key: value}, initial_state(), 5)
        state = initial_state()
        check_step(valid, state, 5)
        with self.assertRaises(AssertionError):
            check_step(valid, copy.deepcopy(state), 5)
        with self.assertRaises(AssertionError):
            check_complete(state, 5, 2)


if __name__ == "__main__":
    unittest.main()
