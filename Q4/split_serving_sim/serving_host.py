"""Measured enterprise post-back work, outside the worker forward command.

A batch cost is charged across its per-request trace/event processing. This
advances the scheduler's CPU time while cloud work continues; no GPU kernel
cost or policy rule is changed.
"""
import io
import json
import math
from pathlib import Path

from .command_cost import command_scope


class ServingHostWork:
    def __init__(self, config, profile):
        if profile['scope'] != command_scope(config) or profile.get('operation') != 'scheduler_back_post':
            raise ValueError('host profile scope or operation mismatch')
        self.profile = profile
        self.samples = profile['samples']
        if not self.samples or any(s['batch_size'] <= 0 or s['sample_count'] <= 0
            or not math.isfinite(s['latency_ms']) or s['latency_ms'] < 0 for s in self.samples):
            raise ValueError('invalid serving host costs')
        self.coverage = {}
        self.total_ms = 0.0

    @classmethod
    def from_file(cls, config, path):
        return cls(config, json.loads(Path(path).read_text()))

    def per_row_ms(self, row):
        samples = [s for s in self.samples if s['phase'] == row['phase'] and s['emits_token'] == row['emits_token']]
        if not samples:
            raise ValueError('no serving host profile for phase/emission boundary')
        source = min(samples, key=lambda s: (abs(s['batch_size']-row['batch_size']), s['batch_size']))
        kind = 'exact_batch' if source['batch_size'] == row['batch_size'] else 'nearest_batch_per_request'
        self.coverage[kind] = self.coverage.get(kind, 0) + 1
        duration = source['latency_ms'] / source['batch_size']
        self.total_ms += duration
        return duration


class TimedServingTrace(io.StringIO):
    def __init__(self, runtime, host_work):
        super().__init__()
        self.runtime = runtime
        self.host_work = host_work

    def write(self, text):
        row = json.loads(text)
        duration = self.host_work.per_row_ms(row)/1000
        start = self.runtime.clock.now
        self.runtime.clock.sleep(duration)
        self.runtime.execution.append(dict(resource='enterprise_host',op='post_back',phase=row['phase'],
            start_time_ms=start*1000,end_time_ms=self.runtime.clock.now*1000,ids=[row['request_id']]))
        return super().write(text)
