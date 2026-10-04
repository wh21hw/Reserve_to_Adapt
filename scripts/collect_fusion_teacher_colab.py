"""Compare teacher and self-label K7 with identical ten-epoch budget."""
import json
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/fusion-unknown-teacher-v1')
report = dict(task='Office31 A->W',seed=1,K=7,epochs=10,rows=[],
              selection='posthoc seed1 and target-label best epochs; exploratory, no causal significance claim')
for name,folder in [('self-label',Path('/content/imp-runs/fusion-capacity-diagnostic-v1/K7/rta/a2w_seed1')),
                    ('prototype-teacher',root/'rta/a2w_seed1')]:
    history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in history] != list(range(1,11)):
        raise RuntimeError('Incomplete arm '+name)
    best = max(history,key=lambda row:row['HOS'])
    def metric(row):
        return dict(epoch=row['epoch'],**{key:row[key]*100 for key in ('OS_star','unknown','HOS')})
    report['rows'].append(dict(method=name,best=metric(best),final=metric(history[-1])))
teacher = [json.loads(line) for line in (root/'rta/a2w_seed1/teacher-history.jsonl').read_text().splitlines()]
if [row['epoch'] for row in teacher] != list(range(5,11)):
    raise RuntimeError('Teacher diagnostics incomplete')
counts = [sum(row['assignment_counts'][j] for row in teacher) for j in range(7)]
total = sum(row['selected_occurrences'] for row in teacher)
report['teacher'] = dict(selected_slot_counts=counts,empty_slots=sum(n==0 for n in counts),
                         weighted_logit_agreement=sum(row['logit_teacher_agreement']*row['selected_occurrences'] for row in teacher)/max(total,1),
                         final=teacher[-1])
report['best_HOS_delta'] = report['rows'][1]['best']['HOS']-report['rows'][0]['best']['HOS']
(root/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
destination = Path('/content/fusion-unknown-teacher-v1-results.zip')
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
    for file in root.rglob('*'):
        if file.is_file() and file.suffix in ('.json','.jsonl','.txt','.log'):
            archive.write(file,file.relative_to(root))
print(json.dumps(report,indent=2),flush=True)
print('UNKNOWN_TEACHER_RESULTS',destination,flush=True)
