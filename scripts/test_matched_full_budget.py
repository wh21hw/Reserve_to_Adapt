"""Static policy/anchor test. No Torch imports, no training claim."""
import ast
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matched_full_budget import build

source = (Path(__file__).parent / 'train_matched_structure_pilot.py').read_bytes().decode()
full = build(source)
ast.parse(full)
assert 'for epoch in range(4, 70):' in full
assert 'if epoch <= 10:' in full and 'groups == unknown_group_id' in full
assert 'n_components=4 if epoch <= 30 else 2' in full
assert 'actual_new_optimizer_steps=924' in full
assert 'w.global_step == 980' in full and 'grl.global_step == 1960' in full
assert 'target_label_checkpoint_selection=False' in full
assert "coefficient = 0. if args.arm == 'structure_off' else .1" in full
assert full.index('if args.verify_init:') < full.index('output.mkdir(parents=True)')
assert "data_list_sha256=" in full
try:
    build(source + '\n')
except AssertionError:
    pass
else:
    raise AssertionError('Modified pilot must be rejected')
print('FULL_BUDGET_STATIC_POLICY_PASS; no SGD or runtime validation')
