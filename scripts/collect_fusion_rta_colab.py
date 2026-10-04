"""Collect all three completed fusion runs without checkpoint reevaluation."""
import json
from pathlib import Path
import statistics
import zipfile

root = Path('/content/imp-runs/fusion-imp-rta-v1')
rows = []
for seed in (1, 2, 3):
    folder = root/('seed%d' % seed)/'rta'/('a2w_seed%d' % seed)
    history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    if len(history) != 70 or [r['epoch'] for r in history] != list(range(1, 71)):
        raise RuntimeError('Seed %d is not complete70' % seed)
    structure = [json.loads(line) for line in (folder/'fusion-history.jsonl').read_text().splitlines()]
    best = max(history, key=lambda r: r['HOS'])
    names = ('OS_star', 'unknown', 'HOS')
    rows.append(dict(seed=seed, K=structure[0]['K'],
        V_range=[min(r['V'] for r in structure), max(r['V'] for r in structure)],
        best=dict(epoch=best['epoch'], **{k:best[k]*100 for k in names}),
        final=dict(epoch=70, **{k:history[-1][k]*100 for k in names})))
aggregate = {stage:{name:dict(mean=statistics.mean(r[stage][name] for r in rows),
                            sample_sd=statistics.stdev(r[stage][name] for r in rows))
                    for name in ('OS_star', 'unknown', 'HOS')}
             for stage in ('best', 'final')}
report = dict(complete=True, task='Office31 A->W', gpu='L4', epochs=70, source_epochs=3,
              rows=rows, aggregate=aggregate,
              selection='Best epochs use target labels; all three seeds reported, not best-seed mean',
              method='Initial IMP K fixed; each-epoch IMP virtual directions; original warm-end K-means head init')
(root/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
destination = Path('/content/fusion-imp-rta-v1-results.zip')
with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
    for file in root.rglob('*'):
        if file.is_file() and file.suffix in ('.json', '.jsonl', '.txt', '.log'):
            archive.write(file, file.relative_to(root))
print(json.dumps(report, indent=2), flush=True)
print('FUSION_RESULTS_ZIP', destination, flush=True)
