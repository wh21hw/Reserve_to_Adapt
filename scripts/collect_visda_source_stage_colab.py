"""Save the completed source stage's ordinary logs/config, no model reevaluation."""
import json
import math
from pathlib import Path
import zipfile
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--include-recovery', action='store_true',
    help='Include source checkpoint and frozen features; no model reevaluation')
args = parser.parse_args()

root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
source = root/'source'
summary = json.loads((source/'summary.json').read_text())
config = json.loads((source/'config.json').read_text())
history = [json.loads(line) for line in (source/'history.jsonl').read_text().splitlines()]
if not summary['complete'] or summary['source_shape'] != [79765, 256] or summary['target_shape'] != [55388, 256]:
    raise ValueError('Full-data source feature stage incomplete')
if [row['epoch'] for row in history] != [1, 2, 3] or not all(math.isfinite(row['loss']) for row in history):
    raise ValueError('Expected three complete finite-loss source epochs')
if config['known_classes'] != 6 or not config['freeze_backbone_bn'] or config['retention_weight']:
    raise ValueError('Source protocol conflicts with the declared candidate')
archive = Path('/content/visda-frozenbn-source3-v1-recovery.zip' if args.include_recovery
               else '/content/visda-frozenbn-source3-v1-results.zip')
paths = [root/'source-launch.json', root/'source-console.log', source/'config.json',
         source/'history.jsonl', source/'summary.json']
if args.include_recovery:
    paths += [source/'source-final.pt', source/'features.npz']
if any(not path.is_file() for path in paths):
    raise FileNotFoundError('Required source-stage material missing')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for path in paths:
        z.write(path, str(path.relative_to(root)))
print('VISDA_SOURCE_STAGE_SAVED', json.dumps(dict(archive=str(archive), epochs=3,
    final_source_loss=history[-1]['loss'], final_source_train_accuracy=history[-1]['accuracy'],
    source_shape=summary['source_shape'], target_shape=summary['target_shape'],
    target_metrics=None, K=None, recovery_model_features_included=args.include_recovery,
    caveat='Source-only stage, not domain-adaptation result or RTA optimizer-resume state')), flush=True)
