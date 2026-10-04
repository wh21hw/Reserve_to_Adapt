"""Collect four declared ten-epoch arms, no inference or checkpoint reevaluation."""
import json
import math
from pathlib import Path
import zipfile
import torch

base=Path('/content/imp-runs')
pair=base/'officehome-frozenbn-capacity-10e-v1'
roots={'update_K4':pair/'fixed4','update_K2':pair/'estimated',
       'hold_K4':base/'officehome-rta-frozenbn-10e-v1',
       'hold_K2':base/'officehome-rta-frozenbn-k2-10e-v1'}
output=roots['hold_K2']/'cross-summary.json'
archive=Path('/content/officehome-bn-capacity-cross-10e-v1-results.zip')
if output.exists() or archive.exists():raise FileExistsError('Preserve results')
prior_path=str(pair/'prior/source/source-final.pt')
report=dict(task='OfficeHome Pr->Rw',seed=1,Q=29,C=25,epochs=10,source_epochs=3,estimated_K=2,
    selection='Oracle-best target-label HOS epoch; short-course exploratory module study, not significance',
    caveat='Fourth arm declared after observing earlier results; not strict paper reproduction or blind external validation',arms={})
histories={};files=[]
def metric(row):return dict(epoch=row['epoch'],OS_star=row['OS_star'],UNK=row['unknown'],HOS=row['HOS'])
for name,root in roots.items():
    launch=json.loads((root/'launch.json').read_text())
    expected=4 if name.endswith('K4') else 2
    if launch['K']!=expected or launch['source_prior']!=prior_path or any(launch[key]!=value
            for key,value in [('C',25),('Q',29),('seed',1),('epochs',10)]):
        raise ValueError('Arm mismatch: '+name)
    if bool(launch.get('RTA_FREEZE_ENCODER_BN',False))!=name.startswith('hold'):
        raise ValueError('BN policy mismatch: '+name)
    training=root/'officehome-pr2rw_seed1'
    h=[json.loads(line) for line in (training/'history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in h]!=list(range(1,11)):raise ValueError('Incomplete budget: '+name)
    if not all(math.isfinite(row[key]) for row in h for key in ('OS_star','unknown','HOS')):
        raise ValueError('Nonfinite metric: '+name)
    histories[name]=h
    report['arms'][name]=dict(K=expected,best=metric(max(h,key=lambda row:row['HOS'])),final=metric(h[-1]),
        observed_convergence_warning_messages=(root/'console.log').read_text().count('ConvergenceWarning'))
    files.extend((file,name+'/'+str(file.relative_to(root))) for file in
        [root/'launch.json',root/'console.log',training/'history.jsonl',training/'metrics.json',training/'config.json',training/'protocol.json'])
# Previous hold_K4 factor confirmed by its completed collector; check only the new arm.
if not json.loads((roots['hold_K4']/'summary.json').read_text())['held_buffers_unchanged']:
    raise ValueError('Previous BN factor not maintained')
prior=torch.load(prior_path,map_location='cpu')['model']
final=torch.load(str(roots['hold_K2']/'officehome-pr2rw_seed1/last.pt'),map_location='cpu')['model']
keys=[key for key in prior if key.startswith('0.') and key.endswith(('running_mean','running_var','num_batches_tracked'))]
if not keys or not all(torch.equal(prior[key],final[key]) for key in keys):
    raise RuntimeError('New held-BN factor not maintained')
report['new_held_bn_buffers_unchanged']=True
contrasts={'capacity_in_update_BN':('update_K2','update_K4'),
           'capacity_in_hold_BN':('hold_K2','hold_K4'),
           'BN_at_K4':('hold_K4','update_K4'),'BN_at_K2':('hold_K2','update_K2')}
report['contrasts_pp']={name:{which:{key:100*(report['arms'][left][which][key]-report['arms'][right][which][key])
    for key in ('OS_star','UNK','HOS')} for which in ('best','final')} for name,(left,right) in contrasts.items()}
report['final_capacity_BN_interaction_pp']={key:report['contrasts_pp']['capacity_in_hold_BN']['final'][key]
    -report['contrasts_pp']['capacity_in_update_BN']['final'][key] for key in ('OS_star','UNK','HOS')}
report['warning_count_caveat']='Observed console messages, not failed-fit counts; repeated warnings can be filtered.'
report['paired_capacity_epoch_deltas_pp']={policy:[dict(epoch=i+1,OS_star=100*(right['OS_star']-left['OS_star']),
    UNK=100*(right['unknown']-left['unknown']),HOS=100*(right['HOS']-left['HOS']))
    for i,(left,right) in enumerate(zip(histories[policy+'_K4'],histories[policy+'_K2']))] for policy in ('update','hold')}
output.write_text(json.dumps(report,indent=2,allow_nan=False))
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for file,name in files+[(pair/'prior/capacity.json','capacity.json'),(output,'cross-summary.json')]:z.write(file,name)
print('BN_CAPACITY_CROSS_SUMMARY',json.dumps(report),flush=True)
