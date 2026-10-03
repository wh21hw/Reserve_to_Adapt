"""Audit completed pilot arms and archive metrics/code, without oracle selection."""
import hashlib
import json
from pathlib import Path
import zipfile

root=Path('/content/imp-runs/rta-capacity-pilot-v1')
rows=[]
for arm in ('original2','proto2','proto18'):
    run=root/arm/'a2w_seed1'
    history=[json.loads(row) for row in (run/'history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history]==[5,6]
    assert json.loads((root/arm/'process-status.json').read_text())['exit_code']==0
    handoff=json.loads((run/'handoff.json').read_text())
    rows.append(dict(arm=arm,final=history[-1],history=history,handoff=handoff,
        last_checkpoint_sha256=hashlib.sha256((run/'last.pt').read_bytes()).hexdigest()))
for key in ('known_weights_sha256','known_momentum_sha256','source_relation_bank_sha256','virtual_sha256','gmm_means','optimizer_steps','grl_steps'):
    assert all(row['handoff'][key]==rows[0]['handoff'][key] for row in rows)
report=dict(stage='Two adaptation epochs, Office31 A2W, one seed only',
    completed_full_baseline=False, semantic_unknown_count_inferred=False,
    target_label_selection=False,arms=rows)
report_path=root/'summary.json'
with report_path.open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
archive=Path('/content/rta-capacity-pilot-v1-results.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix != '.pt':
            bundle.write(path,str(path.relative_to(root)))
    for path in Path('/content/rta-capacity-pilot-v1').glob('*.py'):
        bundle.write(path,'code/'+path.name)
    for path in [Path('/content/capacity-pilot-handoff-v1.json'),Path('/content/run_capacity_pilot_colab.py')]:
        bundle.write(path,path.name)
print('PILOT_RESULTS',json.dumps(report),flush=True)
print('PILOT_RESULTS_ARCHIVE_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
