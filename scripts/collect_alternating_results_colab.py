"""Save completed alternating metrics/K trace without checkpoint reevaluation."""
import json
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/konly-alternating-v1/seed3')
run = root/'a2w_seed3'
metrics = json.loads((run/'metrics.json').read_text())
history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines() if line]
refreshes = [json.loads(line) for line in (run/'K-history.jsonl').read_text().splitlines() if line]
if metrics['epoch'] != 70 or metrics['seed'] != 3 or len(history) != 70:
    raise RuntimeError('Incomplete alternating arm')
if [row['completed_epochs'] for row in refreshes] != [20, 40, 60]:
    raise RuntimeError('Missing scheduled refresh')
best_K = history[metrics['best']['epoch']]['K']
summary = dict(metrics=metrics, best_K=best_K, final_K=history[-1]['K'],
               K_path=[history[0]['K']] + [row['K'] for row in refreshes],
               target_labels_used_for_K=False,
               best_uses_target_label_HOS=True, seed_selected_posthoc=True)
(root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
archive = Path('/content/konly-alternating-v1-seed3-results.zip')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.log', '.npz'):
            stream.write(path, path.relative_to(root).as_posix())
print('ALTERNATING_RESULTS_SAVED', str(archive), json.dumps(summary, allow_nan=False), flush=True)
