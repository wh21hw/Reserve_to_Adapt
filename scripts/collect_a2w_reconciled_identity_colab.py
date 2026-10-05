"""Read the completed K8 pair, keeping capacity and identity effects separate."""
import json
import math
from pathlib import Path
import shutil
import zipfile

root = Path('/content/imp-runs/a2w-reconciled-identity-10e-v1')
report = dict(task='Office31 A->W', C=10, K=8, Q=20, seed=3, epochs=10, arms={},
    research_change='Only cluster identity transport versus original unknown argmax at the same K8',
    selection='Previously selected seed3, target-oracle HOS-best descriptive; final and full trajectories retained',
    caveat='Single seed, fixed offline clusters; no statistical or semantic class-count proof')
histories, launches = [], []
def metric(row):
    return dict(epoch=row['epoch'], OS_star=row['OS_star'], UNK=row['unknown'], HOS=row['HOS'])
for arm in ('argmax','identity'):
    folder = root/arm
    launch = json.loads((folder/'launch.json').read_text())
    launches.append(launch)
    if tuple(launch[k] for k in ('C','K','Q','seed','epochs','unknown_ce_weight')) != (10,8,20,3,10,1):
        raise ValueError('Unexpected declared arm settings')
    rows = [json.loads(line) for line in (folder/'office31-a2w_seed3/history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in rows] != list(range(1,11)):
        raise ValueError('Incomplete ten-epoch arm')
    if not all(math.isfinite(row[k]) for row in rows for k in ('OS_star','unknown','HOS')):
        raise ValueError('Nonfinite metrics')
    histories.append(rows)
    report['arms'][arm] = dict(best=metric(max(rows,key=lambda r:r['HOS'])), final=metric(rows[-1]))
base, candidate = launches
for key in ('source_prior','initialization','encoder_bn','cluster_artifact','capacity_source'):
    if base[key] != candidate[key]:
        raise ValueError('Unmatched shared factor: '+key)
expected = dict(base['environment'], RTA_CLUSTER_LABELS=base['cluster_artifact'])
if expected != candidate['environment']:
    raise ValueError('Unexpected environment changes beyond identity switch')
expected_command = list(base['command'])
expected_command[expected_command.index('--log_dir')+1] = str(root/'identity')
if expected_command != candidate['command']:
    raise ValueError('Unexpected command changes')
usage = [json.loads(line) for line in
    (root/'identity/office31-a2w_seed3/cluster-identity-history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in usage] != list(range(1,11)):
    raise ValueError('Incomplete identity usage')
for row in usage:
    if (len(row['pseudo_slot_counts']) != 8 or sum(row['pseudo_slot_counts']) != row['selected']
            or not 0 <= row['cluster_labels_used'] <= row['selected']):
        raise ValueError('Invalid slot-usage counts')
    if row['epoch'] >= 4 and sorted(row['cluster_to_slot']) != list(range(10,18)):
        raise ValueError('Missing bijective cluster/slot correspondence')
total = sum(row['selected'] for row in usage)
covered = sum(row['cluster_labels_used'] for row in usage)
report['identity_usage'] = dict(rows=usage, selected=total, overridden=covered,
    coverage=covered/total if total else None,
    slot_counts=[sum(row['pseudo_slot_counts'][i] for row in usage) for i in range(8)],
    limitation='Argmax arm has no slot-count instrumentation; cannot claim relative balancing')
report['identity_minus_argmax_pp'] = [dict(epoch=b['epoch'], **{
    key:100*(metric(b)[key]-metric(a)[key]) for key in ('OS_star','UNK','HOS')}) for a,b in zip(*histories)]
report['warmup_max_abs_delta_pp'] = max(abs(row[k]) for row in report['identity_minus_argmax_pp'][:4]
    for k in ('OS_star','UNK','HOS'))
historical = json.loads(Path('/content/imp-runs/a2w-unknown-ce-10e-v1/summary.json').read_text())
report['k2_context'] = dict(result=historical['arms']['control'],
    note='Compare K2 with K8 argmax for capacity context only. K8 identity versus K2 changes two factors.')
destination, archive = root/'summary.json', Path('/content/a2w-reconciled-identity-10e-v1-results.zip')
if destination.exists() or archive.exists():
    raise FileExistsError('Preserve existing result')
destination.write_text(json.dumps(report,indent=2,allow_nan=False))
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.jsonl','.log'):
            bundle.write(path,str(path.relative_to(root)))
drive = Path('/content/drive/MyDrive/OSDA/runs/a2w-reconciled-identity-10e-v1')
drive.mkdir(exist_ok=False)
shutil.copyfile(archive,drive/archive.name)
for arm in ('argmax','identity'):
    shutil.copyfile(root/arm/'office31-a2w_seed3/last.pt',drive/(arm+'-last.pt'))
print('A2W_K8_PAIR_COLLECTED',json.dumps(report),flush=True)
