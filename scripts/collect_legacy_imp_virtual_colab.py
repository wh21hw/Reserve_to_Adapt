"""Save completed historical experiment metrics/logs without reevaluation."""
import json
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/legacy-imp-virtual-v1/seed3')
runs = list(root.glob('*_seed3/metrics.json'))
if len(runs) != 1:
    raise RuntimeError('Expected one historical run')
run = runs[0].parent
metrics = json.loads(runs[0].read_text())
history = [json.loads(row) for row in (run/'history.jsonl').read_text().splitlines() if row]
if metrics['epoch'] != 70 or metrics['seed'] != 3 or len(history) != 70:
    raise RuntimeError('Historical experiment incomplete')
baseline = dict(best=95.43614198180025, final=95.07257091620211)
summary = dict(metrics=metrics, classifier_unknown_slots=2,
    virtual_count_range=[min(row['virtual_prototypes'] for row in history),
                         max(row['virtual_prototypes'] for row in history)],
    baseline_HOS_percent=baseline,
    difference_HOS_pp=dict(best=100*metrics['best']['HOS']-baseline['best'],
                           final=100*metrics['HOS']-baseline['final']),
    comparison='Whole old recipe, including 5 source epochs; not a pure IMP effect',
    selection='Previously selected seed3, target-label oracle-best epoch')
(root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
output = Path('/content/legacy-imp-virtual-v1-seed3-results.zip')
with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.log'):
            archive.write(path, path.relative_to(root).as_posix())
print('LEGACY_IMP_RESULTS_SAVED', str(output), json.dumps(summary, allow_nan=False), flush=True)
