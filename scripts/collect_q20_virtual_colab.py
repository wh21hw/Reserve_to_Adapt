"""Collect the matched 10-epoch virtual-module control from ordinary logs."""
import json
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/fusion-q20-virtual-control-v1')
control = Path('/content/imp-runs/fusion-supported-capacity-v1')

def collect(run):
    folder = run / 'rta/a2w_seed1'
    history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in history] != list(range(1, 11)):
        raise RuntimeError('Expected exactly 10 completed epochs: ' + str(run))
    def metrics(row):
        return dict(epoch=row['epoch'], **{key: row[key]*100 for key in ('OS_star', 'unknown', 'HOS')})
    return dict(best=metrics(max(history, key=lambda row: row['HOS'])),
                final=metrics(history[-1]))

report = dict(task='Office31 A->W', seed=1, K=5, epochs=10,
              q20=collect(root), imp=collect(control),
              selection='posthoc seed1; best epoch uses target labels; exploratory single-seed comparison',
              scope='same frozen-feature extraction; only recurrent virtual update differs; not official baseline')
report['delta_HOS_pp'] = {key: report['q20'][key]['HOS']-report['imp'][key]['HOS']
                          for key in ('best', 'final')}
(root/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
destination = Path('/content/fusion-q20-virtual-control-v1-results.zip')
with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
    for file in root.rglob('*'):
        if file.is_file() and file.suffix in ('.json', '.jsonl', '.txt', '.log'):
            archive.write(file, file.relative_to(root))
print(json.dumps(report, indent=2), flush=True)
print('Q20_RESULTS', destination, flush=True)
