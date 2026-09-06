"""Join tensor annotations to GPU kernels through Nsight CUDA launch correlation."""
import argparse
import bisect
import collections
import csv
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/operator_baseline_4k'


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=OUT)
    OUT = parser.parse_args().root.resolve()
    records = {}
    for p in OUT.glob('operators_*.jsonl'):
        for line in p.open():
            r = json.loads(line)
            r.update(cpu_range_ns=0, gpu_kernel_ns=0, gpu_kernel_count=0)
            records[r['id']] = r
    kernels, unmatched = [], []
    for role in ('enterprise', 'cloud'):
        c = sqlite3.connect(f'file:{OUT / (role + ".sqlite")}?mode=ro', uri=True)
        c.row_factory = sqlite3.Row
        strings = dict(c.execute('select id,value from StringIds'))
        ranges = collections.defaultdict(list)
        for t in c.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null'):
            label = t['text'] or strings.get(t['textId'], '')
            if label in records:
                ranges[t['globalTid']].append((t['start'], t['end'], label))
                records[label]['cpu_range_ns'] = t['end'] - t['start']
        for v in ranges.values():
            v.sort()
        starts = {tid: [x[0] for x in v] for tid, v in ranges.items()}
        launches = {}
        for t in c.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
            tid = t['globalTid']
            v = ranges.get(tid, [])
            i = bisect.bisect_right(starts.get(tid, []), t['start']) - 1
            label = None
            # Nested ranges: select the innermost containing launch interval.
            while i >= 0:
                s, e, uid = v[i]
                if s <= t['start'] and t['end'] <= e:
                    label = uid
                    break
                i -= 1
            launches[(tid >> 24, t['correlationId'])] = label
        for k in c.execute('select start,end,globalPid,correlationId,deviceId,streamId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL'):
            uid = launches.get((k['globalPid'] >> 24, k['correlationId']))
            row = {'role': role, 'pid': k['globalPid'] >> 24,
                   'kernel': strings[k['demangledName']], 'gpu_start_ns': k['start'],
                   'gpu_end_ns': k['end'], 'gpu_duration_ns': k['end'] - k['start'],
                   'device_id': k['deviceId'], 'stream_id': k['streamId'], 'operator_id': uid}
            if uid is None:
                unmatched.append(row)
                continue
            r = records[uid]
            r['gpu_kernel_ns'] += row['gpu_duration_ns']
            r['gpu_kernel_count'] += 1
            row.update(operator=r['operator'], phase=r['phase'], rank=r['rank'])
            kernels.append(row)
        c.close()
    columns = ['role', 'rank', 'pid', 'phase', 'execution_op', 'batch_id', 'id', 'operator', 'items',
               'input_shapes', 'input_dtypes', 'inputs', 'cpu_range_ns', 'gpu_kernel_ns', 'gpu_kernel_count']
    prepared = []
    for r in records.values():
        row = dict(r)
        row.setdefault('execution_op', 'forward')
        row['input_shapes'] = json.dumps([t['shape'] for t in r['inputs']])
        row['input_dtypes'] = json.dumps([t['dtype'] for t in r['inputs']])
        row['inputs'] = json.dumps(r['inputs'])
        row['items'] = json.dumps(r['items'])
        prepared.append(row)
    for phase in ('prefill', 'decode'):
        with (OUT / f'{phase}_operators.csv').open('w') as f:
            w = csv.DictWriter(f, columns, extrasaction='ignore')
            w.writeheader()
            w.writerows(r for r in prepared if r['phase'] == phase)
    groups = collections.defaultdict(list)
    for r in prepared:
        key = tuple(r[k] for k in ('role', 'rank', 'phase', 'operator', 'input_shapes', 'input_dtypes'))
        groups[key].append(r)
    summary = []
    for key, values in groups.items():
        r = dict(zip(('role', 'rank', 'phase', 'operator', 'input_shapes', 'input_dtypes'), key))
        durations = sorted(v['gpu_kernel_ns'] for v in values)
        r.update(calls=len(values), gpu_kernel_count=sum(v['gpu_kernel_count'] for v in values),
                 gpu_total_us=sum(durations)/1000, gpu_mean_us=sum(durations)/len(values)/1000,
                 gpu_min_us=min(durations)/1000, gpu_max_us=max(durations)/1000)
        summary.append(r)
    with (OUT / 'operator_summary.csv').open('w') as f:
        w = csv.DictWriter(f, list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    if kernels:
        with (OUT / 'kernel_instances.csv').open('w') as f:
            w = csv.DictWriter(f, list(kernels[0]))
            w.writeheader()
            w.writerows(kernels)
    (OUT / 'unmatched_kernels.json').write_text(json.dumps(unmatched, indent=2))
    audit = {'operator_count': len(records), 'matched_kernels': len(kernels),
             'unmatched_kernels': len(unmatched),
             'matched_gpu_ns': sum(r['gpu_duration_ns'] for r in kernels),
             'unmatched_gpu_ns': sum(r['gpu_duration_ns'] for r in unmatched),
             'gpu_operator_types': sorted({r['operator'] for r in records.values() if r['gpu_kernel_count']}),
             'operators_by_role_rank_phase': dict(collections.Counter(
                 f"{r['role']}/rank{r['rank']}/{r['phase']}" for r in records.values()))}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
