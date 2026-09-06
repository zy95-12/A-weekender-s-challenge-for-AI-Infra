"""Capture a warmed 4K request through the configured production scheduler."""
import argparse
import json
import shutil
from pathlib import Path

import httpx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--variant', choices=['baseline', 'stage123'], required=True)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    prompt = json.loads(a.prompt.read_text())
    ref = json.loads(a.reference.read_text())
    assert len(prompt['prompt_ids']) == 4096
    for source, name in [(a.prompt, 'prompt.json'), (a.reference, 'reference.json')]:
        if source.resolve() != (a.root / name).resolve():
            shutil.copyfile(source, a.root / name)
    root = Path(__file__).resolve().parents[1]
    live = Path((root / 'run/current_results').read_text().strip())
    (a.root / 'source_results.txt').write_text(str(live))
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=300, trust_env=False) as c:
        h = c.get('/health').json()
        assert h['tp'] == h['cloud_tp'] == 2
        opt = h['optimizations']
        expected = ({'ipc_mode': 'pipe', 'wire_fast': False, 'tcp_buffer_mib': 0,
                     'prefill_chunk_size': 0, 'pipeline_window': 0, 'scheduler_policy': 'legacy'}
                    if a.variant == 'baseline' else
                    {'ipc_mode': 'shm', 'wire_fast': True, 'tcp_buffer_mib': 16,
                     'prefill_chunk_size': 1024, 'pipeline_window': 2, 'scheduler_policy': 'decode-first', 'decode_quota': 1})
        assert all(opt[k] == v for k, v in expected.items()), opt
        (a.root / 'initial_health.json').write_text(json.dumps(h, indent=2))
        body = {'prompt_ids': prompt['prompt_ids'], 'steps': 9}
        warm = c.post('/debug/greedy', json=body)
        warm.raise_for_status()
        assert warm.json()['tokens'] == ref['greedy_ids'][:9]
        c.post('/start_profile').raise_for_status()
        try:
            response = c.post('/debug/greedy', json=body)
            response.raise_for_status()
            result = response.json()
            assert result['tokens'] == ref['greedy_ids'][:9]
            (a.root / 'captured_output.json').write_text(json.dumps(result, indent=2))
        finally:
            c.post('/stop_profile').raise_for_status()
        h = c.get('/health').json()
        assert h['active'] == h['waiting'] == h['kv_used_blocks'] == 0
        (a.root / 'final_health.json').write_text(json.dumps(h, indent=2))
    print(f'{a.variant}: prefill + 8 decode forwards; 9 token IDs match reference.')


if __name__ == '__main__':
    main()
