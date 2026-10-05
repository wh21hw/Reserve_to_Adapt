"""Prepare a single causal ablation; no training unless --run is explicit."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--run', action='store_true', help='Use only after the new GPU budget is confirmed')
args = parser.parse_args()
control = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1/fixed2')
launch = json.loads((control/'launch.json').read_text())
rows = [json.loads(r) for r in (control/'visda-synthetic2real_seed1/history.jsonl').read_text().splitlines()]
if [r['epoch'] for r in rows] != list(range(1, 11)):
    raise ValueError('Require completed ten-epoch control')
if (launch['C'], launch['K'], launch['Q'], launch['seed'], launch['epochs']) != (6, 2, 8, 1, 10):
    raise ValueError('Unexpected matched control')
output = Path('/content/imp-runs/visda-unknown-ce-off-10e-v1')
command = list(launch['command'])
command[command.index('/content/train_visda_frozenbn_rta_entry.py')] = '/content/train_visda_unknown_ce_off_entry.py'
command[command.index('--log_dir')+1] = str(output)
manifest = dict(launch, arm='unknown-ce-off', command=command, control=str(control),
    research_change='Post-warmup unknown pseudo-label CE coefficient 1 -> 0 only',
    unknown_ce_control=1, unknown_ce_candidate=0,
    forward_policy='Retain unknown candidate selection and head forward/BN behavior',
    purpose='Causal failure-mechanism ablation, not final unknown modeling solution',
    initialization='Same shared source3 prior, seed1, original warm-end initialization',
    selection='Report all epochs and final; target-oracle best is descriptive, no target-driven early stop')
if not args.run:
    print('UNKNOWN_CE_ABLATION_PREPARED_NOT_STARTED', json.dumps(manifest), flush=True)
    raise SystemExit(0)
output.mkdir(exist_ok=False)
(output/'launch.json').write_text(json.dumps(manifest, indent=2))
env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', RTA_SEED='1',
    RTA_EPOCHS='10', RTA_FREEZE_ENCODER_BN='1', KONLY_SOURCE_PRIOR=launch['source_prior'],
    RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth')
env.pop('RTA_CLUSTER_LABELS', None)
env.pop('LEGACY_TASK_BUILD_ONLY', None)
with (output/'console.log').open('x') as log:
    process = subprocess.Popen(command, cwd='/content/rta-legacy-l4-bridge-v1', env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        log.write(line); log.flush()
    if process.wait():
        raise RuntimeError('Ablation failed; preserve evidence, no automatic retry')
history = [json.loads(r) for r in (output/'visda-synthetic2real_seed1/history.jsonl').read_text().splitlines()]
if [r['epoch'] for r in history] != list(range(1, 11)):
    raise RuntimeError('Incomplete declared budget')
print('UNKNOWN_CE_ABLATION_COMPLETE', flush=True)
