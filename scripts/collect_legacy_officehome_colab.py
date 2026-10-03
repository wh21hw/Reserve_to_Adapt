"""Collect ordinary completed OfficeHome artifacts, not model reevaluation."""
import json
import os
from pathlib import Path
import zipfile

seed = int(os.environ.get('OFFICEHOME_SEED', '3'))
root = Path('/content/imp-runs/legacy-officehome-v1')/('seed'+str(seed))
run = root/('officehome-pr2rw_seed'+str(seed))
metrics = json.loads((run/'metrics.json').read_text())
history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines() if line]
if metrics['epoch'] != 70 or metrics['seed'] != seed or len(history) != 70:
    raise RuntimeError('OfficeHome run is incomplete')
paper = dict(OS_star=82.1, unknown=77.2, HOS=79.5)
summary = dict(metrics=metrics, paper_percent=paper,
    difference_pp={selection: {key: 100*row[key]-paper[key] for key in paper}
                   for selection, row in [('best', metrics['best']), ('final', metrics)]},
    best_uses_target_label_HOS=True, protocol='Published-code transfer, not fully confirmed author config',
    virtual_clusters=29, K=4, epochs=70)
(root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
output = Path('/content/legacy-officehome-v1-seed'+str(seed)+'-results.zip')
with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.log'):
            stream.write(path, path.relative_to(root).as_posix())
print('OFFICEHOME_RESULTS_SAVED', str(output), json.dumps(summary, allow_nan=False), flush=True)
