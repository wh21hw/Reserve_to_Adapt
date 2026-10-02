"""Read saved metrics without altering experiments; emit full-precision comparison."""
import json
from pathlib import Path
import statistics

root = Path('/content/drive/MyDrive/OSDA/runs/rta-official-a2w-v2')
rows = []
for seed in (1, 2, 3):
    path = root/f'a2w_seed{seed}'/'metrics.json'
    if path.is_file():
        metrics = json.loads(path.read_text())
        rows.append(dict(seed=seed, completed=metrics['epoch'] == 70,
                         epoch=metrics['epoch'], final={key: metrics[key]*100 for key in
                            ('OS', 'OS_star', 'unknown', 'HOS')},
                         best={key: value*100 if key != 'epoch' else value for key,value in metrics['best'].items()}))
result = dict(paper=dict(OS_star=92.2, unknown=93.8, HOS=93.0), runs=rows,
              complete=len(rows) == 3 and all(row['completed'] for row in rows))
if result['complete']:
    result['best_mean'] = {key: statistics.mean(row['best'][key] for row in rows)
                           for key in ('OS_star','unknown','HOS')}
    result['best_sample_std'] = {key: statistics.stdev(row['best'][key] for row in rows)
                                 for key in ('OS_star','unknown','HOS')}
    result['difference_from_paper_pp'] = {key: result['best_mean'][key]-result['paper'][key]
                                         for key in result['paper']}
print(json.dumps(result, indent=2))
