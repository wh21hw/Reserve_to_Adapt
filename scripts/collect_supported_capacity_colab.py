"""Collect supported capacity without repeated checkpoint evaluation."""
import json
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/fusion-supported-capacity-v1')
folder = root/'rta/a2w_seed1'
history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
mechanism = [json.loads(line) for line in (folder/'mechanism-history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in history] != list(range(1,11)) or len(mechanism) != 10:
    raise RuntimeError('Incomplete supported capacity')
best = max(history,key=lambda row:row['HOS'])
def metric(row):
    return dict(epoch=row['epoch'],**{key:row[key]*100 for key in ('OS_star','unknown','HOS')})
report = dict(task='Office31 A->W',seed=1,epochs=10,
              inference=json.loads((root/'launch.json').read_text())['count_inference'],
              best=metric(best),final=metric(history[-1]),
              final_mechanism=mechanism[-1],selection='posthoc seed1; best target-label epoch')
(root/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
destination = Path('/content/fusion-supported-capacity-v1-results.zip')
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
    for file in root.rglob('*'):
        if file.is_file() and file.suffix in ('.json','.jsonl','.txt','.log'):
            archive.write(file,file.relative_to(root))
print(json.dumps(report,indent=2),flush=True)
print('SUPPORTED_CAPACITY_RESULTS',destination,flush=True)
