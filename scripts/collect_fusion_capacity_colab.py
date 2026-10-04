"""Collect matched ten-epoch capacity arms; never reevaluate checkpoints."""
import json
from pathlib import Path
import zipfile

root = Path('/content/imp-runs/fusion-capacity-diagnostic-v1')
report = dict(task='Office31 A->W',seed=1,source_epochs=3,epochs=10,rows=[],
              selection='seed1 selected post hoc; best epoch uses target labels',
              diagnostic_scope='training augmented target occurrences, no target labels')
for k in (2,7):
    folder = root/('K%d' % k)/'rta/a2w_seed1'
    history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    mechanism = [json.loads(line) for line in (folder/'mechanism-history.jsonl').read_text().splitlines()]
    structure = [json.loads(line) for line in (folder/'fusion-history.jsonl').read_text().splitlines()]
    for rows in (history,mechanism):
        if [row['epoch'] for row in rows] != list(range(1,11)):
            raise RuntimeError('Incomplete matched arm K%d' % k)
    best = max(history,key=lambda row:row['HOS'])
    names = ('OS_star','unknown','HOS')
    def metric(row):
        return dict(epoch=row['epoch'],**{name:100*row[name] for name in names})
    def summarize(rows):
        selected = [sum(row['selected_unknown_slot_counts']) for row in rows]
        slot_counts = [sum(row['selected_unknown_slot_counts'][j] for row in rows) for j in range(k)]
        total = sum(row['target_occurrences'] for row in rows)
        return dict(unknown_selected_fraction=sum(selected)/total,
                    unknown_prediction_fraction=sum(sum(row['predicted_unknown_slot_counts']) for row in rows)/total,
                    selected_slot_counts=slot_counts,
                    selected_top_slot_share=max(slot_counts)/max(sum(slot_counts),1),
                    known_alignment_fraction=sum(row['known_alignment_fraction']*row['target_occurrences'] for row in rows)/total,
                    margin_mean=sum(row['unknown_minus_known_max_logit_mean']*row['target_occurrences'] for row in rows)/total,
                    split_fraction=sum(row['mass_exceeds_known_max_but_argmax_known_fraction']*row['target_occurrences'] for row in rows)/total)
    report['rows'].append(dict(K=k,best=metric(best),final=metric(history[-1]),
        V_range=[min(row['V'] for row in structure),max(row['V'] for row in structure)],
        postwarm=summarize(mechanism[4:]),final_mechanism=mechanism[-1]))
(root/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
destination = Path('/content/fusion-capacity-diagnostic-v1-results.zip')
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
    for file in root.rglob('*'):
        if file.is_file() and file.suffix in ('.json','.jsonl','.log','.txt'):
            archive.write(file,file.relative_to(root))
print(json.dumps(report,indent=2),flush=True)
print('CAPACITY_RESULTS',destination,flush=True)
