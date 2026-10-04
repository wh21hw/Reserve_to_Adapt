"""Read complete matched histories and BN buffers; no checkpoint inference."""
import json
import math
from pathlib import Path
import zipfile
import torch

root=Path('/content/imp-runs/officehome-rta-frozenbn-10e-v1')
launch=json.loads((root/'launch.json').read_text())
control=Path(launch['control'])
output=root/'summary.json';archive=Path('/content/officehome-rta-frozenbn-10e-v1-results.zip')
if output.exists() or archive.exists():raise FileExistsError('Preserve collected results')
report=dict(task='OfficeHome Pr->Rw',K=4,Q=29,seed=1,epochs=10,
    factor='Encoder BN running-statistics policy during RTA; same frozen-source prior',
    selection='Oracle-best target-label HOS epoch; one exploratory seed, not significance',
    caveat='Not K-only change or strict paper reproduction',arms={})
files=[];histories={}
for name,directory in [('update',control),('hold',root)]:
    configuration=json.loads((directory/'launch.json').read_text())
    if any(configuration[key]!=launch[key] for key in ('K','Q','seed','epochs','source_prior')):
        raise ValueError('Mismatched control settings')
    training=directory/'officehome-pr2rw_seed1'
    h=[json.loads(row) for row in (training/'history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in h]!=list(range(1,11)):raise ValueError('Incomplete ten-epoch arm')
    if not all(math.isfinite(row[key]) for row in h for key in ('OS_star','unknown','HOS')):
        raise ValueError('Nonfinite metrics')
    def metric(row):return dict(epoch=row['epoch'],OS_star=row['OS_star'],UNK=row['unknown'],HOS=row['HOS'])
    report['arms'][name]=dict(best=metric(max(h,key=lambda row:row['HOS'])),final=metric(h[-1]),
        observed_convergence_warning_messages=(directory/'console.log').read_text().count('ConvergenceWarning'))
    histories[name]=h
    files.extend((file,name+'/'+str(file.relative_to(directory))) for file in
        [directory/'launch.json',directory/'console.log',training/'history.jsonl',training/'config.json',training/'protocol.json',training/'metrics.json'])
# One read-only confirmation of the actual experimental factor, not model evaluation.
prior=torch.load(launch['source_prior'],map_location='cpu')['model']
final=torch.load(str(root/'officehome-pr2rw_seed1/last.pt'),map_location='cpu')['model']
keys=[key for key in prior if key.startswith('0.') and key.endswith(('running_mean','running_var','num_batches_tracked'))]
if not keys:raise ValueError('Missing encoder BN buffers')
report['encoder_bn_buffer_paths']=len(keys)
report['warning_count_caveat']='Observed console messages only; Python warning filtering can suppress repeated messages, so this is not the number of failed GMM fits.'
report['held_buffers_unchanged']=all(torch.equal(prior[key],final[key]) for key in keys)
if not report['held_buffers_unchanged']:raise RuntimeError('The hold-BN experimental factor was not maintained')
report['hold_minus_update_pp']={which:{key:100*(report['arms']['hold'][which][key]-report['arms']['update'][which][key])
    for key in ('OS_star','UNK','HOS')} for which in ('best','final')}
report['paired_epoch_deltas_pp']=[dict(epoch=left['epoch'],OS_star=100*(right['OS_star']-left['OS_star']),
    UNK=100*(right['unknown']-left['unknown']),HOS=100*(right['HOS']-left['HOS']))
    for left,right in zip(histories['update'],histories['hold'])]
output.write_text(json.dumps(report,indent=2,allow_nan=False))
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for file,name in files+[(output,'summary.json')]:z.write(file,name)
print('RTA_BN_ABLATION_SUMMARY',json.dumps(report),flush=True)
