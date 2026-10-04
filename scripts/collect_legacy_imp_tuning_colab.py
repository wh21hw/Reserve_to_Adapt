"""Compare five settings strictly within their first ten RTA epochs."""
import json
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/legacy-imp-tuning-v1')
original = list(Path('/content/imp-runs/legacy-imp-virtual-v3/seed3').glob('*_seed3/history.jsonl'))
if len(original) != 1:
    raise RuntimeError('Missing original .05/5 history')
rows = [json.loads(line) for line in original[0].read_text().splitlines() if line][:10]
if len(rows) != 10 or [row['epoch'] for row in rows] != list(range(1, 11)):
    raise RuntimeError('Missing reference ten-epoch window')
best = max(rows, key=lambda row: row['HOS'])
reference = dict(alpha=.05, source_epochs=5, rta_epochs=10,
    best={key: best[key] for key in ('epoch', 'OS_star', 'unknown', 'HOS')},
    final={key: rows[-1][key] for key in ('epoch', 'OS_star', 'unknown', 'HOS')},
    virtual_count_range=[min(row['virtual_prototypes'] for row in rows), max(row['virtual_prototypes'] for row in rows)],
    reused_first10=True)
arms = {'alpha0p05-ft5': reference}
for name in ('alpha0p01-ft5', 'alpha0p1-ft5', 'alpha0p05-ft3', 'alpha0p05-ft7'):
    summary = json.loads((root/name/'summary.json').read_text())
    metrics = summary['metrics']
    if metrics['epoch'] != 10:
        raise RuntimeError('Incomplete short arm')
    arms[name] = dict(alpha=summary['alpha'], source_epochs=summary['source_epochs'],
        rta_epochs=10, best=dict(epoch=metrics['best']['epoch']+1,
            **{key: metrics['best'][key] for key in ('OS_star', 'unknown', 'HOS')}),
        final={key: metrics[key] for key in ('epoch', 'OS_star', 'unknown', 'HOS')},
        virtual_count_range=summary['virtual_count_range'], reused_first10=False)
report = dict(task='Office31 A->W', seed=3, rta_epochs=10, arms=arms,
    oracle_best_setting=max(arms, key=lambda name: arms[name]['best']['HOS']),
    highest_final_setting=max(arms, key=lambda name: arms[name]['final']['HOS']),
    caveat='Short exploratory sweep, target-label epoch/config selection; not unbiased paper result',
    scheduler='Original max_iter10000 unchanged, allowing prefix comparison')
(root/'comparison-10e.json').write_text(json.dumps(report, indent=2, allow_nan=False))
output = Path('/content/legacy-imp-tuning-v1-10e-results.zip')
with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write(root/'comparison-10e.json', 'comparison-10e.json')
    for name in arms:
        if arms[name]['reused_first10']:
            continue
        for path in sorted((root/name).rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.txt', '.log'):
                archive.write(path, path.relative_to(root).as_posix())
print('TEN_EPOCH_SWEEP_RESULTS_SAVED', str(output), json.dumps(report, allow_nan=False), flush=True)
