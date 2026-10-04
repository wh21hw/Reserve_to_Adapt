"""Matched K1 baseline/intersection ten-epoch logs, no checkpoint inference."""
import json
import math
from pathlib import Path
import zipfile

root=Path('/content/imp-runs/officehome-imp-intersection-10e-v3')
baseline=Path('/content/imp-runs/officehome-capacity-10e-v1/estimated')
output=root/'comparison.json'
archive=Path('/content/officehome-imp-intersection-10e-v3-results.zip')
if output.exists() or archive.exists():
    raise FileExistsError('Preserve previous collection')
report=dict(task='OfficeHome Pr->Rw',K=1,seed=1,Q=29,epochs=10,source_epochs=3,
    selection='Target-label HOS selects best epoch; single seed exploratory, no significance claim',
    method='Only frozen initial IMP intersection for ce_ep; not K-only',arms={})
files=[]
launches=[]
for arm,folder in [('baseline',baseline),('intersection',root)]:
    launch=json.loads((folder/'launch.json').read_text());launches.append(launch)
    training=folder/'officehome-pr2rw_seed1'
    history=[json.loads(x) for x in (training/'history.jsonl').read_text().splitlines()]
    if [x['epoch'] for x in history]!=list(range(1,11)):
        raise ValueError('Incomplete arm: '+arm)
    if not all(math.isfinite(x[key]) for x in history for key in ('OS_star','unknown','HOS')):
        raise ValueError('Nonfinite metrics')
    def metric(row):
        return dict(epoch=row['epoch'],OS_star=row['OS_star'],UNK=row['unknown'],HOS=row['HOS'])
    report['arms'][arm]=dict(best=metric(max(history,key=lambda x:x['HOS'])),final=metric(history[-1]))
    files.extend((arm,p) for p in [folder/'launch.json',folder/'console.log',training/'history.jsonl',training/'metrics.json',training/'config.json',training/'protocol.json'])
for key in ('K','C','Q','seed','epochs','source_epochs','source_prior'):
    if launches[0][key]!=launches[1][key]:
        raise ValueError('Mismatched setting: '+key)
gate_path=root/'officehome-pr2rw_seed1/intersection-history.jsonl'
gates=[json.loads(x) for x in gate_path.read_text().splitlines()]
report['candidate_counts']=[]
for epoch in range(10):
    rows=[x for x in gates if x['epoch']==epoch]
    if not rows or any(not 0<=x['after']<=x['before'] for x in rows):
        raise ValueError('Missing/invalid gate records')
    before=sum(x['before'] for x in rows);after=sum(x['after'] for x in rows)
    report['candidate_counts'].append(dict(completed_epoch=epoch+1,batches=len(rows),before=before,after=after,
        retention=after/before if before else None,ce_ep_active=epoch>3))
report['intersection_minus_baseline_pp']={which:{key:100*(report['arms']['intersection'][which][key]-report['arms']['baseline'][which][key]) for key in ('OS_star','UNK','HOS')} for which in ('best','final')}
output.write_text(json.dumps(report,indent=2,allow_nan=False))
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for arm,path in files:
        folder=baseline if arm=='baseline' else root
        z.write(path,str(Path(arm)/path.relative_to(folder)))
    z.write(gate_path,'intersection/intersection-history.jsonl')
    z.write(output,'comparison.json')
print('INTERSECTION_COMPARISON',json.dumps(report),flush=True)
