"""Save the completed source stage's ordinary logs/config, no model reevaluation."""
import json
import math
from pathlib import Path
import zipfile

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
archive = Path('/content/visda-frozenbn-source3-v1-results.zip')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for path in [root/'source-launch.json', root/'source-console.log', source/'config.json',
                 source/'history.jsonl', source/'summary.json']:
        z.write(path, str(path.relative_to(root)))
print('VISDA_SOURCE_STAGE_SAVED', json.dumps(dict(archive=str(archive), epochs=3,
    final_source_loss=history[-1]['loss'], final_source_train_accuracy=history[-1]['accuracy'],
    source_shape=summary['source_shape'], target_shape=summary['target_shape'],
    target_metrics=None, K=None, caveat='Source-only stage, not domain-adaptation result')), flush=True)
