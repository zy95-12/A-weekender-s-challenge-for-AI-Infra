"""Opt-in diagnostic NVTX annotations with real tensor metadata (no tensor reads)."""
import contextlib
import functools
import json
import os
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def tensors(value, path='args'):
    if isinstance(value, torch.Tensor):
        return [{'argument': path, 'shape': list(value.shape), 'dtype': str(value.dtype),
                 'device': str(value.device), 'stride': list(value.stride())}]
    if isinstance(value, (tuple, list)):
        return [t for i, v in enumerate(value) for t in tensors(v, f'{path}[{i}]')]
    if isinstance(value, dict):
        return [t for k, v in value.items() for t in tensors(v, f'{path}.{k}')]
    return []


class Capture(TorchDispatchMode):
    def __init__(self, role, rank, directory):
        super().__init__()
        self.role, self.rank = role, rank
        self.sequence = 0
        self.command = None
        self.active = False
        self.records = []
        self.path = Path(directory) / f'operators_{role}_rank{rank}_{os.getpid()}.jsonl'

    @contextlib.contextmanager
    def annotation(self, name, args, kwargs):
        self.sequence += 1
        uid = f'optrace:{os.getpid()}:{self.sequence}'
        c = self.command
        record = {'id': uid, 'pid': os.getpid(), 'role': self.role, 'rank': self.rank,
                  'operator': name, 'phase': c['phase'], 'execution_op': c['op'], 'batch_id': c['batch_id'],
                  'items': c['items'], 'inputs': tensors(args) + tensors(kwargs, 'kwargs')}
        self.records.append(record)
        torch.cuda.nvtx.range_push(uid)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        with self.annotation(str(func), args, kwargs):
            return func(*args, **kwargs)

    def wrap_communicator(self, communicator):
        # PyNCCL calls the shared library directly, outside TorchDispatch.
        for name in ('all_reduce', 'all_gather', 'reduce_scatter', 'broadcast', 'send', 'recv'):
            original = getattr(communicator, name, None)
            if original is None:
                continue
            def wrap(original=original, name=name):
                @functools.wraps(original)
                def call(*args, **kwargs):
                    if self.active and self.command is not None:
                        with self.annotation('pynccl.' + name, args, kwargs):
                            return original(*args, **kwargs)
                    return original(*args, **kwargs)
                return call
            setattr(communicator, name, wrap())

    def execute(self, runner, command, arrays):
        if command['op'] == 'profile_start':
            self.records = []
            result = runner.execute(command, arrays)
            self.active = True
            return result
        if command['op'] == 'profile_stop':
            self.active = False
            result = runner.execute(command, arrays)
            with self.path.open('w') as f:
                for r in self.records:
                    f.write(json.dumps(r) + '\n')
            return result
        if self.active and command.get('phase') in ('prefill', 'decode'):
            self.command = command
            try:
                with self:
                    return runner.execute(command, arrays)
            finally:
                self.command = None
        return runner.execute(command, arrays)
