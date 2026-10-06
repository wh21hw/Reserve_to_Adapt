"""Collect exact20 histories, never reevaluate a checkpoint or fit capacity."""
import json
import math
from pathlib import Path
import shutil
import zipfile

root = Path('/content/imp-runs/a2w-capacity-refresh-20e-v1')
report = dict(task='Office31 A->W',C=10,initial_K=8,Q=20,seed=3,epochs=20,arms={},
    change='Only applying current-representation inferred K after ten epochs',
    selection='Posthoc historical seed3; best epoch target-oracle; full final20 and trajectories retained',
    caveat='Single-seed exploratory pair; not semantic count, significant improvement or full70 proof')
histories,launches = [],[]


def metric(row):
    return dict(epoch=row['epoch'],K=row['K'],OS_star=row['OS_star'],UNK=row['unknown'],HOS=row['HOS'])


for arm in ('fixed8','refresh'):
    folder = root/arm
    launch = json.loads((folder/'launch.json').read_text())
    launches.append(launch)
    if (launch['C'],launch['initial_K'],launch['Q'],launch['seed'],launch['epochs']) != (10,8,20,3,20):
        raise ValueError('Unexpected declared settings')
    task = folder/'office31-a2w_seed3'
    rows = [json.loads(line) for line in (task/'history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in rows] != list(range(1,21)):
        raise ValueError('Incomplete twenty-epoch budget')
    if not all(math.isfinite(row[key]) for row in rows for key in ('OS_star','unknown','HOS')):
        raise ValueError('Nonfinite final metrics')
    boundary = json.loads((task/'capacity-after-010/estimate.json').read_text())
    if boundary['target_labels_used'] or boundary['completed_epochs'] != 10:
        raise ValueError('Unexpected inference protocol')
    expected_K = 8 if arm == 'fixed8' else boundary['inferred_K']
    if [row['K'] for row in rows] != [8]*10+[expected_K]*10:
        raise ValueError('Unexpected capacity timing/history')
    histories.append(rows)
    report['arms'][arm] = dict(best=metric(max(rows,key=lambda row:row['HOS'])),
        post_refresh_best=metric(max(rows[10:],key=lambda row:row['HOS'])),
        final=metric(rows[-1]),boundary=boundary)
for key in ('source_prior','initialization','runtime_environment','C','initial_K','Q','seed','epochs'):
    if launches[0][key] != launches[1][key]:
        raise ValueError('Shared factor differs: '+key)
base,candidate = (report['arms'][arm] for arm in ('fixed8','refresh'))
report['final_refresh_minus_fixed_pp'] = {key:100*(candidate['final'][key]-base['final'][key])
    for key in ('OS_star','UNK','HOS')}
report['best_refresh_minus_fixed_pp'] = {key:100*(candidate['best'][key]-base['best'][key])
    for key in ('OS_star','UNK','HOS')}
report['warmup_max_abs_delta_pp'] = max(100*abs(a[key]-b[key])
    for a,b in zip(histories[0][:10],histories[1][:10]) for key in ('OS_star','unknown','HOS'))
report['runtime_environment'] = launches[0]['runtime_environment']
report['manual_support_screen'] = dict(
    final_HOS_gain_at_least_1pp=report['final_refresh_minus_fixed_pp']['HOS']>=1,
    final_OS_star_loss_at_most_1pp=report['final_refresh_minus_fixed_pp']['OS_star']>=-1,
    final_UNK_loss_at_most_1pp=report['final_refresh_minus_fixed_pp']['UNK']>=-1,
    method_promoted=False,interpretation='Screen only; manual assessment, not target-label tuning or automatic keep')
summary,archive = root/'summary.json',Path('/content/a2w-capacity-refresh-20e-v1-results.zip')
if summary.exists() or archive.exists():
    raise FileExistsError('Preserve existing collection')
summary.write_text(json.dumps(report,indent=2,allow_nan=False))
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.jsonl','.log','.npz'):
            bundle.write(path,str(path.relative_to(root)))
saved = Path('/content/drive/MyDrive/OSDA/runs/a2w-capacity-refresh-20e-v1')
if not all((saved/arm/'office31-a2w_seed3/last.pt').is_file() for arm in ('fixed8','refresh')):
    raise RuntimeError('Both arm models must be durably saved before final collection')
shutil.copyfile(summary,saved/'summary.json')
shutil.copyfile(archive,saved/archive.name)
print('A2W_CAPACITY_REFRESH_COLLECTED',json.dumps(report),flush=True)
