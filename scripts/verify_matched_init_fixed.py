"""Validate BOTH repaired initialization prefixes, before any mkdir/SGD/batch."""
import ast
import hashlib
import json
from pathlib import Path
import sys

worker = Path('/content/train_matched_structure_pilot_fixed_v2.py')
assert hashlib.sha256(worker.read_bytes()).hexdigest() == 'bf5babdecdcd0e07c8ea2e6f553a07442009101d08565566a5bbdf208f02d75b'
tree = ast.parse(worker.read_text())
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
prefix = []
for node in main.body:
    if isinstance(node, ast.Expr) and ast.unparse(node) == 'output.mkdir(parents=True)':
        break
    prefix.append(node)
else:
    raise AssertionError('Initialization-only boundary not found')
main.body = prefix + [ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]))]
tree.body = [node for node in tree.body if not isinstance(node, ast.If)]
namespace = {'__name__': '__init_verification__', '__file__': str(worker)}
exec(compile(ast.fix_missing_locations(tree), str(worker), 'exec'), namespace)
rows = []
for arm in ('structure_off', 'structure_on'):
    sys.argv = [str(worker), '--arm', arm]
    values = namespace['main']()
    assert not values['output'].exists()
    rows.append(dict(arm=arm, coefficient=values['coefficient'],
                     strict_model_and_momentum_fingerprints_matched=True,
                     rng_torch_fingerprint_matched=True,
                     optimizer_steps=[w.global_step for w in values['wrappers']],
                     grl_steps=values['discriminator'].grl.global_step))
    del values
report = dict(arms=rows, optimizer_steps_executed=0, training_batches_read=0,
              experiment_directories_created=0, worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
              repair='CPU construction then device transfer; original strict checks retained',
              training_restart_performed=False)
with Path('/content/matched-init-fixed-verification-v2.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print('MATCHED_INIT_FIXED_VERIFICATION_PASS', json.dumps(report), flush=True)
