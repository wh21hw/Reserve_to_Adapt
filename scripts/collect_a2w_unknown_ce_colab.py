"""Compare both completed arms using ordinary logs, no model reevaluation."""
import json
import math
from pathlib import Path
import shutil
import zipfile

root = Path('/content/imp-runs/a2w-unknown-ce-10e-v1')
report = dict(task='Office31 A->W', C=10, K=2, Q=20, seed=3, epochs=10,
    changed_factor='Post-warmup unknown pseudo-label CE 1 versus 0', arms={},
    selection='Previously selected seed3; best uses target-label HOS; final/all epochs retained',
    caveat='Single-seed short-course mechanism experiment, not final method or full paper reproduction')
histories, launches = [], []
for arm in ('control', 'off'):
    folder = root / arm
    launches.append(json.loads((folder/'launch.json').read_text()))
    rows = [json.loads(line) for line in (folder/'office31-a2w_seed3/history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in rows] != list(range(1, 11)):
        raise ValueError('Expected exactly ten epochs: '+arm)
    for row in rows:
        if any(not math.isfinite(row[key]) for key in ('OS_star', 'unknown', 'HOS')):
            raise ValueError('Nonfinite result')
    def metrics(row):
        return dict(epoch=row['epoch'], OS_star=row['OS_star'], UNK=row['unknown'], HOS=row['HOS'])
    report['arms'][arm] = dict(best=metrics(max(rows, key=lambda r:r['HOS'])), final=metrics(rows[-1]))
    histories.append(rows)
for field in ('C', 'K', 'Q', 'seed', 'epochs', 'source_prior', 'encoder_bn', 'initialization'):
    if launches[0][field] != launches[1][field]:
        raise ValueError('Unmatched control: '+field)
if [launch['unknown_ce_weight'] for launch in launches] != [1, 0]:
    raise ValueError('Unexpected coefficient')
report['off_minus_control_pp'] = [dict(epoch=a['epoch'], **{
    key:100*(b[key]-a[key]) for key in ('OS_star', 'unknown', 'HOS')}) for a,b in zip(*histories)]
destination = root/'summary.json'
archive = Path('/content/a2w-unknown-ce-10e-v1-results.zip')
if destination.exists() or archive.exists():
    raise FileExistsError('Preserve existing collection')
destination.write_text(json.dumps(report, indent=2, allow_nan=False))
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.log'):
            bundle.write(path, str(path.relative_to(root)))
    bundle.write('/content/a2w-ce-source-console.log', 'source-console.log')
drive = Path('/content/drive/MyDrive/OSDA/runs/a2w-unknown-ce-10e-v1')
drive.mkdir(exist_ok=False)
shutil.copyfile(archive, drive/archive.name)
shutil.copyfile(root/'source/source-final.pt', drive/'source-final.pt')
for arm in ('control', 'off'):
    shutil.copyfile(root/arm/'office31-a2w_seed3/last.pt', drive/(arm+'-last.pt'))
print('A2W_CE_COMPLETE', json.dumps(report), flush=True)
