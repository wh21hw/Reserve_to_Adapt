"""One matched K2 identity-label arm; no new initialization, loss, or prediction."""
import json
import os
from pathlib import Path
import subprocess

control = Path('/content/imp-runs/officehome-rta-frozenbn-k2-10e-v1')
old = json.loads((control/'launch.json').read_text())
history = [json.loads(r) for r in
           (control/'officehome-pr2rw_seed1/history.jsonl').read_text().splitlines()]
if (old['K'], old['Q'], old['seed'], old['epochs']) != (2, 29, 1, 10):
    raise ValueError('Expected matched held-BN K2 control')
if [r['epoch'] for r in history] != list(range(1, 11)):
    raise ValueError('Missing complete matched control')
root = Path('/content/imp-runs/officehome-cluster-identity-10e-v1')
artifact = root/'clusters.npz'
if not artifact.is_file():
    raise FileNotFoundError('Recover and inspect identities before training')
output = root/'rta'
output.mkdir(exist_ok=False)
command = list(old['command'])
command[command.index('--log_dir')+1] = str(output)
command[command.index('/content/train_legacy_task_entry.py')] = '/content/cluster-identity-code-v1/train_cluster_identity_entry.py'
env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', RTA_SEED='1',
           RTA_EPOCHS='10', RTA_FREEZE_ENCODER_BN='1', KONLY_SOURCE_PRIOR=old['source_prior'],
           RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
           RTA_CLUSTER_LABELS=str(artifact), PYTHONPATH='/content/cluster-identity-code-v1')
manifest = dict(old, command=command, control=str(control), cluster_artifact=str(artifact),
    research_change='Only fixed offline cluster identities replace unknown argmax for covered selected candidates',
    head_initialization='Original warm-end K-means, unchanged',
    slot_alignment='Warm-end current target cluster means to initialized head cosine Hungarian; fixed permutation',
    fallback='Initial known/noise assignments retain original unknown-slot argmax',
    target_labels_used_for_structure=False, selection='Target-label oracle best epoch; single seed exploratory')
(output/'launch.json').write_text(json.dumps(manifest, indent=2))
with (output/'console.log').open('x') as log:
    p = subprocess.Popen(command, cwd='/content/rta-legacy-l4-bridge-v1', env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    if p.wait():
        raise RuntimeError('Identity arm failed; preserve evidence, no automatic retry')
rows = [json.loads(r) for r in
        (output/'officehome-pr2rw_seed1/history.jsonl').read_text().splitlines()]
if [r['epoch'] for r in rows] != list(range(1, 11)):
    raise RuntimeError('Incomplete ten-epoch identity arm')
print('CLUSTER_IDENTITY_RTA_COMPLETE', flush=True)
