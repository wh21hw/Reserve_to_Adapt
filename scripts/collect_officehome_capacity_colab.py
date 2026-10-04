"""Summarize only the declared complete pair; no checkpoint reevaluation."""
import json
import math
from pathlib import Path
import zipfile

root=Path('/content/imp-runs/officehome-capacity-10e-v1')
archive=Path('/content/officehome-capacity-10e-v1-results.zip')
summary_path=root/'summary.json'
if summary_path.exists() or archive.exists():
    raise FileExistsError('Preserve previous collection')
summary=dict(task='OfficeHome Pr->Rw',seed=1,epochs=10,source_epochs=3,Q=29,
    selection='Best epoch uses target-label HOS; final is epoch10; not a three-seed result',
    caveat='Exploratory task-adapted release-code comparison, not verified paper OfficeHome settings',arms={})
files=[]
for arm,expected_k in [('fixed4',4),('estimated',1)]:
    directory=root/arm
    training=directory/'officehome-pr2rw_seed1'
    launch=json.loads((directory/'launch.json').read_text())
    history=[json.loads(line) for line in (training/'history.jsonl').read_text().splitlines()]
    if launch['K']!=expected_k or [row['epoch'] for row in history]!=list(range(1,11)):
        raise ValueError('Incomplete or wrong declared arm: '+arm)
    if not all(math.isfinite(row[key]) for row in history for key in ('OS_star','unknown','HOS')):
        raise ValueError('Nonfinite metrics: '+arm)
    def metrics(row):
        return dict(epoch=row['epoch'],OS_star=row['OS_star'],UNK=row['unknown'],HOS=row['HOS'])
    summary['arms'][arm]=dict(K=launch['K'],best=metrics(max(history,key=lambda row:row['HOS'])),
                             final=metrics(history[-1]))
    files.extend([directory/'launch.json',directory/'console.log',training/'history.jsonl',
                  training/'metrics.json',training/'config.json',training/'protocol.json'])
summary['estimated_minus_fixed_pp']={which:{key:100*(summary['arms']['estimated'][which][key]-
    summary['arms']['fixed4'][which][key]) for key in ('OS_star','UNK','HOS')} for which in ('best','final')}
summary_path.write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for file in files+[summary_path]:
        z.write(file,file.relative_to(root))
print('OFFICEHOME_CAPACITY_SUMMARY',json.dumps(summary),flush=True)
