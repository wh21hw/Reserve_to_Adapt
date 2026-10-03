"""Same fixed final prior-control logits, TWO preregistered algebraic decisions.

Offline target-label evaluation only; no bias sweep, capacity/model selection,
training step, checkpoint change or reapplication of log(2/K).
"""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import torch

run=Path('/content/imp-runs/rta-capacity-prior-pilot-v1/uniform18')
logit_file=run/'final-target-logits.npz'
label_file=Path('/content/imp-runs/rta-space-warmup-l4-features-v1/evaluation-only.npz')
truth=np.load(label_file,allow_pickle=False)['target_labels']
logits=torch.from_numpy(np.load(logit_file,allow_pickle=False)['logits']).double()
assert logits.shape==(564,28) and truth.shape==(564,)
assert set(np.unique(truth))==set(range(10))|set(range(20,31))
slot_prediction=logits.argmax(1).numpy()
group_logits=torch.cat([logits[:,:10],logits[:,10:].logsumexp(1,keepdim=True)],1)
group_prediction=group_logits.argmax(1).numpy()
def metrics(prediction):
    known=np.mean([np.mean(prediction[truth==c]==c) for c in range(10)])
    unknown=np.mean([np.mean(prediction[truth==c]>=10) for c in range(20,31)])
    return dict(OS_star=float(known),unknown=float(unknown),HOS=float(2*known*unknown/(known+unknown)),
        OS=float((known*10+unknown)/11),unknown_predicted_samples=int((prediction>=10).sum()))
slot=metrics(slot_prediction); group=metrics(group_prediction)
previous=json.loads((run/'summary.json').read_text())['final']
assert all(np.isclose(slot[key],previous[key],atol=1e-12,rtol=0) for key in ('OS_star','unknown','HOS','OS'))
report=dict(stage='Fixed final decision-rule audit, no training or selection',
    checkpoint_sha256=json.loads((run/'summary.json').read_text())['checkpoint_sha256'],
    logits_sha256=hashlib.sha256(logit_file.read_bytes()).hexdigest(),
    evaluation_labels_sha256=hashlib.sha256(label_file.read_bytes()).hexdigest(),
    rules=dict(max_individual_slot_then_collapse=slot,marginalized_unknown_group=group),
    changed_decisions=int(((slot_prediction>=10)!=(group_prediction>=10)).sum()),
    correction_applied_twice=False,optimizer_steps=0,
    target_labels_used='Offline macro-per-class evaluation ONLY, not fitting or selecting a rule',
    limitation='Exploratory decision diagnostic, not an additional trained-model result')
path=Path('/content/unknown-decision-audit-v1.json')
with path.open('x') as stream:
    json.dump(report,stream,indent=2,allow_nan=False)
with zipfile.ZipFile('/content/unknown-decision-audit-v1.zip','x',zipfile.ZIP_DEFLATED) as bundle:
    bundle.write(path,path.name)
    bundle.write('/content/audit_unknown_decision_rule_v1.py','audit_unknown_decision_rule_v1.py')
print('UNKNOWN_DECISION_AUDIT',json.dumps(report),flush=True)
print('DECISION_AUDIT_ARCHIVE_SHA256',hashlib.sha256(Path('/content/unknown-decision-audit-v1.zip').read_bytes()).hexdigest(),flush=True)
