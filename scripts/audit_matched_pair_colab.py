"""Audit the completed fixed pilot pair without training or selection."""
import json
import sys
from pathlib import Path
sys.path.insert(0, '/content')
from matched_pilot_audit import validate_pair

root = Path('/content/imp-runs/matched-structure-pilot-v1')
def load(arm):
    run = root / arm
    read = lambda name: json.loads((run / name).read_text())
    lines = lambda name: [json.loads(x) for x in (run / name).read_text().splitlines()]
    independent = read('independent-audit.json')
    assert independent['independent_checkpoint_evaluation_verified']
    return (read('config.json'), lines('history.jsonl'), lines('batches.jsonl'), read('summary.json'))

off, on = load('structure_off'), load('structure_on')
report = validate_pair(off, on)
report['checkpoint_evaluation_still_required'] = False
report['independent_checkpoint_evaluation_verified_both'] = True
report['final_delta_on_minus_off_pp'] = {
    key: 100 * (on[1][-1]['metrics'][key] - off[1][-1]['metrics'][key])
    for key in ('OS_star', 'UNK', 'HOS')}
report['scope'] = 'Seed1 warm4->6 only; no best-seed selection; no full-budget claim'
with Path('/content/matched-structure-pair-audit-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print('MATCHED_PAIR_PASS', json.dumps(report), flush=True)
