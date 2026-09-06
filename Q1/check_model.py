import json
from pathlib import Path

REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'

def check_model(path):
 p=Path(path)
 if (p/'revision.txt').read_text().strip()!=REVISION:
  raise ValueError('Expected the pinned serving model revision')
 c=json.loads((p/'config.json').read_text())
 if (c['num_hidden_layers'],c['hidden_size'],c['vocab_size'])!=(36,2048,151936):
  raise ValueError('Expected Qwen2.5-3B-Instruct dimensions')
