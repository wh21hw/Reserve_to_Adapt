"""Run only initialization prefix, compare CPU/GPU construction; no training."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch

worker = Path('/content/train_matched_structure_pilot.py')
assert hashlib.sha256(worker.read_bytes()).hexdigest() == '99fe702f81a0ced8e0e211f7ede4ac9c95295b89c3dd5970b4c73259ba4822ba'
tree = ast.parse(worker.read_text())
main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
prefix = []
for node in main.body:
    if isinstance(node, ast.Assert) and 'reference' in ast.unparse(node) and "['model']" in ast.unparse(node):
        break
    prefix.append(node)
else:
    raise AssertionError('Initialization boundary not found')
main.body = prefix + [ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]))]
tree.body = [node for node in tree.body if not isinstance(node, ast.If)]
namespace = {'__name__': '__diagnostic__', '__file__': str(worker)}
sys.argv = [str(worker), '--arm', 'structure_off']
exec(compile(ast.fix_missing_locations(tree), str(worker), 'exec'), namespace)
values = namespace['main']()
net = values['net']
reference = values['reference']['model']
differences = [name for name, tensor in net.state_dict().items() if namespace['tensor_sha'](tensor) != reference[name]]
known = values['checkpoint']['model']['1.fc.weight'][:10]
cpu_centers = values['centers'].cpu()
cpu_weights = torch.cat([known, torch.nn.functional.normalize(cpu_centers, dim=1) * known.norm(dim=1).mean()])
cpu_prior = namespace['dirichlet_log_prior'](torch.from_numpy(values['counts'][values['selected']]), concentration=1.)
weight_error = float((net[1].fc.weight.detach().cpu() - cpu_weights).abs().max())
prior_error = float((values['prior'].cpu() - cpu_prior).abs().max())
report = dict(different_keys=differences, candidate_head_max_abs_cpu_gpu_difference=weight_error,
              prior_max_abs_cpu_gpu_difference=prior_error,
              cpu_reconstruction_head_hash_matches_reference=namespace['tensor_sha'](cpu_weights) == reference['1.fc.weight'],
              cpu_reconstruction_prior_hash_matches_reference=namespace['tensor_sha'](cpu_prior) == reference['1.unknown_log_weights'],
              known_weights_exact=torch.equal(known, net[1].fc.weight[:10].detach().cpu()),
              optimizer_steps_executed=0, training_batches_read=0,
              training_directory_exists=values['output'].exists(),
              proposed_resolution='Construct candidate weights and prior once on CPU, then transfer; keep strict fingerprints. Requires separate validation before any training retry.')
with Path('/content/matched-init-parity-diagnostic-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print('MATCHED_INIT_PARITY_DIAGNOSTIC', json.dumps(report), flush=True)
