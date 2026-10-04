"""Compare logged metrics and identity usage; no checkpoint forward or hash audit."""
import json
import math
from pathlib import Path
import zipfile

base = Path('/content/imp-runs')
root = base/'officehome-cluster-identity-10e-v1'
arms = dict(argmax=base/'officehome-rta-frozenbn-k2-10e-v1', cluster_identity=root/'rta')
output = root/'summary.json'
archive = Path('/content/officehome-cluster-identity-10e-v1-results.zip')
if output.exists() or archive.exists():
    raise FileExistsError('Preserve existing summary')
report = dict(task='OfficeHome Pr->Rw', C=25, K=2, Q=29, seed=1, epochs=10,
    research_change='Only unknown pseudo-label identity on covered RTA-selected candidates',
    selection='Target-label oracle best epoch; single-seed short-course exploration',
    caveat='No semantic class-count proof; fixed initial cluster identities can become stale', arms={})
files, histories = [], {}
prior = '/content/imp-runs/officehome-frozenbn-capacity-10e-v1/prior/source/source-final.pt'
def metric(row):
    return dict(epoch=row['epoch'], OS_star=row['OS_star'], UNK=row['unknown'], HOS=row['HOS'])
for name, path in arms.items():
    launch = json.loads((path/'launch.json').read_text())
    if any(launch[key] != value for key, value in
           [('K', 2), ('C', 25), ('Q', 29), ('seed', 1), ('epochs', 10), ('source_prior', prior)]):
        raise ValueError('Matched control factors differ')
    if not launch['RTA_FREEZE_ENCODER_BN']:
        raise ValueError('Expected shared held-BN policy')
    training = path/'officehome-pr2rw_seed1'
    history = [json.loads(line) for line in (training/'history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in history] != list(range(1, 11)):
        raise ValueError('Incomplete ten-epoch budget')
    if not all(math.isfinite(row[key]) for row in history for key in ('OS_star', 'unknown', 'HOS')):
        raise ValueError('Nonfinite metric')
    histories[name] = history
    report['arms'][name] = dict(best=metric(max(history, key=lambda row: row['HOS'])),
                               final=metric(history[-1]))
    for file in [path/'launch.json', path/'console.log', training/'history.jsonl',
                 training/'metrics.json', training/'config.json', training/'protocol.json']:
        files.append((file, name+'/'+str(file.relative_to(path))))
usage_path = root/'rta/officehome-pr2rw_seed1/cluster-identity-history.jsonl'
usage = [json.loads(line) for line in usage_path.read_text().splitlines()]
if [row['epoch'] for row in usage] != list(range(1, 11)):
    raise ValueError('Incomplete identity log')
for row in usage:
    if sum(row['pseudo_slot_counts']) != row['selected'] or not 0 <= row['cluster_labels_used'] <= row['selected']:
        raise ValueError('Identity usage counts conflict')
    if row['epoch'] >= 4 and sorted(row['cluster_to_slot']) != [25, 26]:
        raise ValueError('Missing fixed one-to-one slot mapping')
total = sum(row['selected'] for row in usage)
covered = sum(row['cluster_labels_used'] for row in usage)
counts = [sum(row['pseudo_slot_counts'][i] for row in usage) for i in range(2)]
report['identity_usage'] = dict(rows=usage, total_selected=total, cluster_labels_used=covered,
    coverage_fraction=covered/total if total else None, total_pseudo_slot_counts=counts,
    limitation='Historical argmax control lacks slot-usage logging; no claim of improved balance versus it')
report['delta_pp'] = {which: {key: 100*(report['arms']['cluster_identity'][which][key]
    -report['arms']['argmax'][which][key]) for key in ('OS_star', 'UNK', 'HOS')}
    for which in ('best', 'final')}
report['paired_epoch_deltas_pp'] = [dict(epoch=right['epoch'], **{
    key: 100*(metric(right)[key]-metric(left)[key]) for key in ('OS_star', 'UNK', 'HOS')})
    for left, right in zip(histories['argmax'], histories['cluster_identity'])]
report['warmup_max_absolute_metric_delta_pp'] = max(abs(row[key])
    for row in report['paired_epoch_deltas_pp'][:4] for key in ('OS_star', 'UNK', 'HOS'))
report['warmup_comparison_note'] = 'Identity labels begin after the original warm-end initialization; first four metric rows provide an observed control check, not a deterministic identity guarantee.'
output.write_text(json.dumps(report, indent=2, allow_nan=False))
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for file, name in files+[(usage_path, 'cluster-identity-history.jsonl'),
                             (root/'recovery.json', 'recovery.json'), (output, 'summary.json')]:
        z.write(file, name)
print('CLUSTER_IDENTITY_SUMMARY', json.dumps(report), flush=True)
