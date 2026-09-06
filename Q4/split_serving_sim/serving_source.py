"""Load exact serving method bodies with injected execution dependencies.

Imports and application entrypoints are intentionally not executed. No method
body is rewritten. Each environment gets separate globals (no global patching).
"""
import ast
import hashlib
import json
from pathlib import Path

SOURCE = Path(__file__).parent / 'vendor' / 'serving'


def load_serving(bindings, source=SOURCE):
    manifest = json.loads((source/'manifest.json').read_text())
    namespace = dict(bindings, __name__='serving_virtual_runtime')
    selections = [('server.py', {'Job', 'Scheduler'}),
                  ('pipeline_state.py', {'KVAdmission'}),
                  ('scheduling.py', {'choose'}),
                  ('pipeline.py', {'PipelineScheduler'}),
                  ('pd_scheduler.py', {'PDScheduler'})]
    for name, names in selections:
        data = (source/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['files'][name]:
            raise ValueError(f'serving source hash mismatch: {name}')
        tree = ast.parse(data, filename=str(source/name))
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
        if {n.name for n in nodes} != names:
            raise ValueError(f'missing serving definitions: {name}')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source/name), 'exec'), namespace)
    return namespace, manifest
