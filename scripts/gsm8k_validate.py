"""GSM8K API smoke and absolute-position-aligned prefill/decode logits validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_REV = '3101c7d5072418e28b9008a6636bde82a006892c'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def check_inputs(out, require_model=True):
    manifest = json.loads((out / 'manifest.json').read_text())
    if hashlib.sha256((out / 'prompts.json').read_bytes()).hexdigest() != manifest['prompts_sha256']:
        raise ValueError('Prompt manifest hash mismatch')
    if require_model and (ROOT / 'models/qwen/revision.txt').read_text().strip() != manifest['model_revision']:
        raise ValueError('Model revision mismatch')
    return manifest


def prepare(args):
    from transformers import AutoTokenizer
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    datasets = {}
    hashes = {}
    for split in ('train', 'test'):
        url = f'https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REV}/grade_school_math/data/{split}.jsonl'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        (out / f'{split}.jsonl').write_bytes(data)
        hashes[split] = hashlib.sha256(data).hexdigest()
        datasets[split] = [json.loads(line) for line in data.splitlines()]
    test = datasets['test']
    if args.limit < 0 or args.limit > len(test):
        raise ValueError('limit must be 0 (full test set) or <= test size')
    indices = sorted(random.Random(args.seed).sample(range(len(test)), args.limit)) if args.limit else list(range(len(test)))
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/qwen'), local_files_only=True)
    examples = []
    for row in datasets['train'][:args.few_shot]:
        examples += [{'role': 'user', 'content': row['question']}, {'role': 'assistant', 'content': row['answer']}]
    rows = []
    for index in indices:
        row = test[index]
        messages = [{'role': 'system', 'content': 'Solve the math problem step by step. End with #### followed by the final numeric answer.'}]
        messages += examples + [{'role': 'user', 'content': row['question']}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        rows.append({'index': index, 'question': row['question'], 'prompt_ids': ids})
    save(out / 'prompts.json', rows)
    save(out / 'manifest.json', {'dataset': 'openai/grade-school-math', 'revision': DATA_REV,
        'dataset_sha256': hashes, 'test_size': len(test), 'indices': indices, 'seed': args.seed,
        'few_shot_train_indices': list(range(args.few_shot)), 'max_tokens': args.max_tokens, 'steps': args.steps,
        'model_revision': (ROOT / 'models/qwen/revision.txt').read_text().strip(),
        'prompts_sha256': hashlib.sha256((out / 'prompts.json').read_bytes()).hexdigest(),
        'gate': 'Public API emits text; same-input absolute-position logits pass the fixed numerical gate, separately for prefill and decode. No answer agreement gate.'})


@contextmanager
def native_capture(engine):
    """Observe actual sampler-selected rows; return every original tensor unchanged."""
    runner = engine.llm_engine.model_executor.driver_worker.model_runner
    execute, compute = runner.execute_model, runner.model.compute_logits
    current = {}
    captured, rows = [], []
    def observe_execute(model_input, *args, **kwargs):
        current['input'] = model_input
        return execute(model_input, *args, **kwargs)
    def observe_logits(*args, **kwargs):
        logits = compute(*args, **kwargs)
        if logits is not None and len(logits):
            inp = current['input']
            selected = inp.sampling_metadata.selected_token_indices.cpu().tolist()
            positions = inp.input_positions.cpu().tolist()
            token_ids = inp.input_tokens.cpu().tolist()
            meta = inp.attn_metadata
            starts = meta.query_start_loc.cpu().tolist()
            values = logits.detach().float().cpu().numpy().copy()
            assert len(selected) == len(values)
            for row, hidden_row in enumerate(selected):
                request_row = next(i for i in range(len(starts)-1) if starts[i] <= hidden_row < starts[i+1])
                query_len = starts[request_row+1] - starts[request_row]
                rows.append({'token_id': token_ids[hidden_row], 'absolute_position': positions[hidden_row],
                    'num_computed_tokens': meta.seq_lens[request_row] - query_len, 'query_len': query_len,
                    'hidden_row_index': hidden_row, 'logits_row_index': row,
                    'top1_token': int(values[row].argmax()), 'top1_logit': float(values[row].max()),
                    'phase': 'prefill' if inp.is_prompt else 'decode'})
            captured.append(values)
        return logits
    runner.execute_model, runner.model.compute_logits = observe_execute, observe_logits
    try:
        yield captured, rows
    finally:
        runner.execute_model, runner.model.compute_logits = execute, compute


def native(args):
    if os.environ.get('VLLM_USE_V1') != '0':
        raise RuntimeError('Set VLLM_USE_V1=0 for the native reference')
    from vllm import LLM, SamplingParams
    import importlib.metadata
    out = Path(args.output); manifest = check_inputs(out)
    dest = out / ('native-' + args.variant); dest.mkdir(exist_ok=False)
    optimized = args.variant == 'optimized'
    # Default matches TP/attention; an explicit native TP also measures parallelism differences.
    config = dict(dtype='half', tensor_parallel_size=args.native_tp or (1 if optimized else 2),
        enforce_eager=True, max_model_len=8192, max_num_seqs=1,
        max_num_batched_tokens=2048 if optimized else 8192, gpu_memory_utilization=.65,
        enable_prefix_caching=False, enable_chunked_prefill=optimized,
        disable_custom_all_reduce=True, disable_async_output_proc=True, seed=0)
    save(dest/'config.json', {**config, 'model_revision': manifest['model_revision'],
        'vllm': importlib.metadata.version('vllm'), 'torch': importlib.metadata.version('torch'),
        'capture': 'read-only V0 execute_model and compute_logits wrappers; sampler row selection unchanged'})
    engine = LLM(model=str(ROOT/'models/qwen'), **config)
    for p in json.loads((out/'prompts.json').read_text()):
        with native_capture(engine) as (captured, rows):
            result = engine.generate([{'prompt_token_ids': p['prompt_ids']}], SamplingParams(
                temperature=0, max_tokens=manifest['steps'], ignore_eos=True), use_tqdm=False)[0]
        values = np.concatenate(captured)
        assert values.shape == (manifest['steps'], 151936)
        tokens = list(result.outputs[0].token_ids)
        np.savez_compressed(dest/f"{p['index']}.npz", logits=values, tokens=np.array(tokens),
            prompt_ids=np.array(p['prompt_ids']), rows_json=np.array(json.dumps(rows)))
        print(f"Native {args.variant} index={p['index']} rows={len(rows)}", flush=True)


def validate_alignment(prompt, forced, rows):
    if len(rows) != len(forced):
        raise ValueError('Missing logits rows')
    for step, row in enumerate(rows):
        expected_position = len(prompt) - 1 + step
        expected_token = prompt[-1] if step == 0 else forced[step-1]
        if (row['absolute_position'], row['token_id']) != (expected_position, expected_token):
            raise ValueError(f'Wrong logits input/absolute position at step {step}: {row}')
        if row['num_computed_tokens'] + row['query_len'] - 1 != expected_position:
            raise ValueError('Inconsistent query/context metadata')
        if step and (row['query_len'] != 1 or row['phase'] != 'decode'):
            raise ValueError('Expected a one-token decode row')
        if step == 0 and row['phase'] != 'prefill':
            raise ValueError('Expected the last prompt position')


def logit_metrics(a, b):
    # Same fixed gate as scripts/correctness.py; no answer/top1 agreement gate.
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Logits must be finite matching vocabulary vectors')
    d = a-b
    cosine = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    pa = np.exp(a-a.max()); pa /= pa.sum()
    pb = np.exp(b-b.max()); pb /= pb.sum()
    result = {'mae': float(np.abs(d).mean()), 'rmse': float(np.sqrt(np.mean(d*d))),
        'max_abs_error': float(np.abs(d).max()), 'cosine_similarity': cosine,
        'softmax_tv': float(np.abs(pa-pb).sum()/2), 'max_probability_delta': float(np.abs(pa-pb).max()),
        'top1_agreement_observation': bool(a.argmax() == b.argmax())}
    result['passed'] = result['mae'] <= .01 and result['rmse'] <= .02 and result['max_abs_error'] <= .1 and cosine >= .9999
    return result


def split(args):
    import httpx
    import io
    import time
    out = Path(args.output); manifest = check_inputs(out)
    dest = out / ('split-' + args.variant); dest.mkdir(exist_ok=False)
    prompts = json.loads((out/'prompts.json').read_text())
    with httpx.Client(base_url=args.url, timeout=600, trust_env=False) as client:
        response = client.get('/health'); response.raise_for_status(); state = response.json()
        assert state['revision'] == manifest['model_revision']
        opts = state['optimizations']
        if args.variant == 'baseline':
            assert not state['pd'] and state['tp'] == state['cloud_tp'] == 2
            assert opts['ipc_mode'] == 'pipe' and not any(opts[k] for k in ('wire_fast','tcp_buffer_mib','prefill_chunk_size','pipeline_window'))
        else:
            assert state['pd'] and state['prefill_replicas'] == 2 and state['tp'] == state['cloud_tp'] == 1
            assert opts['prefill_chunk_size'] == 2048 and opts['ipc_mode'] == 'shm' and opts['wire_fast']
        save(dest/'health.json', state)
        # Functional smoke uses the public serving API and checks generation only.
        def smoke(p):
            response = client.post('/v1/completions', json={'model':'split-qwen','prompt':p['prompt_ids'],
                'temperature':0, 'max_tokens':manifest['max_tokens']})
            response.raise_for_status(); body = response.json()
            assert body['usage']['prompt_tokens'] == len(p['prompt_ids'])
            assert body['usage']['completion_tokens'] > 0 and body['choices'][0]['text'].strip()
            return {'index':p['index'], 'response':body}
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            save(dest/'smoke.json', list(pool.map(smoke,prompts)))
        # Isolate each diagnostic request. Native uses the same one-request shape.
        for p in prompts:
            ref = np.load(out/('native-'+args.variant)/f"{p['index']}.npz")
            assert ref['prompt_ids'].tolist() == p['prompt_ids']
            forced = ref['tokens'].tolist()
            response = client.post('/debug/teacher_force', json={'prompt_ids':p['prompt_ids'], 'forced_tokens':forced})
            response.raise_for_status()
            result = np.load(io.BytesIO(response.content))
            rows = json.loads(str(result['rows_json']))
            validate_alignment(p['prompt_ids'], forced, rows)
            np.savez_compressed(dest/f"{p['index']}.npz", logits=result['logits'], tokens=result['tokens'],
                prompt_ids=ref['prompt_ids'], forced_tokens=ref['tokens'], rows_json=result['rows_json'])
            print(f"Split {args.variant} index={p['index']} aligned rows={len(rows)}", flush=True)
        for _ in range(100):
            state = client.get('/health').json()
            if state['active'] == state['waiting'] == state['kv_used_blocks'] == 0:
                break
            time.sleep(.1)
        assert state['active'] == state['waiting'] == state['kv_used_blocks'] == 0
        save(dest/'drained.json', state)
        save(dest/'launch.json', json.loads((ROOT/'run/launch.json').read_text()))


def compare(args):
    out = Path(args.output); check_inputs(out, require_model=False)
    prompts = json.loads((out/'prompts.json').read_text())
    report = {'passed':True, 'samples':len(prompts), 'variants':{},
        'scope':'Same input IDs and absolute positions; last prefill position plus forced decode. Native TP is recorded per variant; TP overrides include parallelism differences. No answer equality gate.'}
    for variant in args.compare_variants:
        rows = []
        native_config = json.loads((out/('native-'+variant)/'config.json').read_text())
        for p in prompts:
            name = f"{p['index']}.npz"
            a = np.load(out/('native-'+variant)/name); b = np.load(out/('split-'+variant)/name)
            assert a['prompt_ids'].tolist() == b['prompt_ids'].tolist() == p['prompt_ids']
            assert a['tokens'].tolist() == b['forced_tokens'].tolist()
            ar, br = json.loads(str(a['rows_json'])), json.loads(str(b['rows_json']))
            validate_alignment(p['prompt_ids'], a['tokens'].tolist(), ar)
            validate_alignment(p['prompt_ids'], a['tokens'].tolist(), br)
            assert a['logits'].shape == b['logits'].shape == (len(ar),151936)
            for i,(x,y) in enumerate(zip(a['logits'],b['logits'])):
                assert ar[i]['absolute_position'] == br[i]['absolute_position']
                rows.append({'index':p['index'], 'phase':'prefill' if i==0 else 'decode',
                    'absolute_position':ar[i]['absolute_position'], 'native_row':ar[i], 'split_row':br[i], **logit_metrics(x,y)})
        save(out/(variant+'-positions.json'), rows)
        summary = {}
        for phase in ('prefill','decode'):
            selected = [r for r in rows if r['phase']==phase]
            summary[phase] = {'rows':len(selected), 'passed':all(r['passed'] for r in selected),
                **{key:max(r[key] for r in selected) for key in ('mae','rmse','max_abs_error','softmax_tv','max_probability_delta')},
                'min_cosine_similarity':min(r['cosine_similarity'] for r in selected)}
        summary['native_config'] = native_config
        report['variants'][variant] = summary
        report['passed'] &= all(summary[p]['passed'] for p in ('prefill','decode'))
    save(out/'summary.json',report); print(json.dumps(report,indent=2),flush=True)
    if not report['passed']:
        raise SystemExit(1)


def all_phases(args):
    import httpx
    try:
        state = httpx.get(args.url+'/health',timeout=3,trust_env=False).json()
    except httpx.RequestError:
        state = None
    if state is not None and not (ROOT/'run/enterprise.pid.json').exists():
        raise RuntimeError('Another checkout owns the service; stop it using its own ./poc down')
    if state is not None and (state['active'] or state['waiting']):
        raise RuntimeError('Wait for active requests to drain')
    prepare(args)
    out = Path(args.output).resolve()
    env = {**os.environ,'VLLM_USE_V1':'0','VLLM_ATTENTION_BACKEND':'FLASH_ATTN',
        'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'4','NCCL_SOCKET_IFNAME':'lo',
        'GLOO_SOCKET_IFNAME':'lo','HF_HUB_OFFLINE':'1'}
    def run(label,command,environment=None):
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    run('initial-down',['./poc','down'])
    try:
        for variant in ('baseline','optimized'):
            run('native-'+variant,[sys.executable,str(Path(__file__).resolve()),'--phase','native',
                '--variant',variant,'--output',str(out)] + (['--native-tp',str(args.native_tp)] if args.native_tp else []),env)
        for variant in ('baseline','optimized'):
            print('Validating '+variant,flush=True)
            run(variant+'-up',['./poc','up','--preset',variant,'--wan'])
            args.variant = variant; split(args)
            if variant == 'baseline':
                run('baseline-down',['./poc','down'])
        compare(args)
    except BaseException:
        subprocess.run(['./poc','down'],cwd=ROOT,check=False)
        raise
    print('PASS; optimized service is running at '+args.url,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['all','prepare','native','split','compare'],default='all')
    parser.add_argument('--output',default='results/gsm8k-logits')
    parser.add_argument('--variant',choices=['baseline','optimized'],default='optimized')
    parser.add_argument('--native-tp',type=int,choices=[1,2],help='Override native full-model TP; default matches each split variant')
    parser.add_argument('--compare-variants',nargs='+',choices=['baseline','optimized'],default=['baseline','optimized'])
    parser.add_argument('--limit',type=int,default=8,help='0 selects the full test set')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--few-shot',type=int,choices=range(17),default=16)
    parser.add_argument('--steps',type=int,default=8,help='One prefill logit row plus steps-1 forced decode rows')
    parser.add_argument('--max-tokens',type=int,default=32,help='Functional smoke output limit; no answer scoring')
    parser.add_argument('--concurrency',type=int,default=8)
    parser.add_argument('--url',default='http://127.0.0.1:8000')
    args = parser.parse_args()
    if not 2 <= args.steps <= 257 or not 1 <= args.max_tokens <= 1024 or args.concurrency < 1:
        parser.error('Invalid step/token/concurrency limits')
    if args.phase=='all':
        all_phases(args)
    else:
        globals()[args.phase](args)


if __name__=='__main__':
    main()
