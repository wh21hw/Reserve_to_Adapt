"""Extend the saved complete one-shot comparison with the alternating arm."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'pipeline-results'
report = json.loads((root/'konly-seed3-comparison-v1.json').read_text())
summary = json.loads((root/'konly-alternating-v1-seed3/summary.json').read_text())
m = summary['metrics']
if m['epoch'] != 70 or m['seed'] != 3:
    raise ValueError('Expected full selected-seed alternating run')
keys = ('OS_star', 'unknown', 'HOS')
arm = dict(final={key: 100*m[key] for key in keys},
           best={key: 100*m['best'][key] for key in keys},
           best_completed_epoch=m['best']['epoch']+1,
           elapsed_seconds=m['elapsed_seconds'], K_path=summary['K_path'],
           best_K=summary['best_K'], final_K=summary['final_K'])
report['arms_percent']['source_prior_alternating_K'] = arm
report['alternating_minus_controls_pp'] = {
    name: {selection: {key: arm[selection][key]-previous[selection][key]
                      for key in keys} for selection in ('best', 'final')}
    for name, previous in report['arms_percent'].items()
    if name in ('source_prior_fixed_K2', 'source_prior_estimated_K')}
report['alternating_schedule'] = [20, 40, 60]
report['alternating_target_labels_used_for_K'] = False
with (root/'konly-seed3-comparison-alternating-v1.json').open('x') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps(report, indent=2, allow_nan=False))
