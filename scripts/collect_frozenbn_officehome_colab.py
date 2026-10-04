"""Collect declared ten-epoch arms, without loading or reevaluating checkpoints."""
import json
import math
from pathlib import Path
import zipfile

root=Path('/content/imp-runs/officehome-frozenbn-capacity-10e-v1')
estimate=json.loads((root/'prior/capacity.json').read_text());k=int(estimate['K'])
if k<1 or not estimate['converged']:raise ValueError('No successful positive estimated capacity')
output=root/'summary.json';archive=Path('/content/officehome-frozenbn-capacity-10e-v1-results.zip')
if output.exists() or archive.exists():raise FileExistsError('Preserve prior collection')
summary=dict(task='OfficeHome Pr->Rw',seed=1,source_epochs=3,RTA_epochs=10,Q=29,
    estimated_K=k,encoder_bn_frozen_in_source_only=True,
    selection='Oracle-best target-label HOS epoch; final10; one exploratory seed, not mean or significance',
    caveat='OfficeHome Q29/K4 and author budget unresolved; not strict paper reproduction',arms={})
files=[root/'prior/capacity.json',root/'prior/launch.json',root/'prior/console.log',
    root/'prior/source/config.json',root/'prior/source/history.jsonl',root/'prior/source/summary.json']
for arm,expected in [('fixed4',4)]+([] if k==4 else [('estimated',k)]):
    directory=root/arm;training=directory/'officehome-pr2rw_seed1'
    launch=json.loads((directory/'launch.json').read_text())
    history=[json.loads(line) for line in (training/'history.jsonl').read_text().splitlines()]
    if launch['K']!=expected or [row['epoch'] for row in history]!=list(range(1,11)):
        raise ValueError('Wrong K or incomplete budget: '+arm)
    if not all(math.isfinite(row[key]) for row in history for key in ('OS_star','unknown','HOS')):
        raise ValueError('Nonfinite metrics')
    def metric(row):return dict(epoch=row['epoch'],OS_star=row['OS_star'],UNK=row['unknown'],HOS=row['HOS'])
    summary['arms'][arm]=dict(K=expected,best=metric(max(history,key=lambda row:row['HOS'])),final=metric(history[-1]))
    files.extend([directory/'launch.json',directory/'console.log',training/'history.jsonl',training/'metrics.json',training/'config.json',training/'protocol.json'])
if k==4:
    summary['equivalence']='Estimated K equals fixed K; only fixed4 was run, no independent estimated trial claimed'
else:
    summary['estimated_minus_fixed_pp']={which:{key:100*(summary['arms']['estimated'][which][key]-summary['arms']['fixed4'][which][key])
        for key in ('OS_star','UNK','HOS')} for which in ('best','final')}
output.write_text(json.dumps(summary,indent=2,allow_nan=False))
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for file in files+[output]:z.write(file,file.relative_to(root))
print('FROZENBN_RTA_SUMMARY',json.dumps(summary),flush=True)
