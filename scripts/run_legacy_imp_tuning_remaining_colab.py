"""Run remaining declared arms serially after first complete arm; stop on failure."""
import json
import os
from pathlib import Path
import subprocess

root = Path('/content/imp-runs/legacy-imp-tuning-v1')
first = json.loads((root/'alpha0p01-ft5/summary.json').read_text())
if first['metrics']['epoch'] != 10:
    raise RuntimeError('First new configuration must finish before batch execution')
for alpha, epochs in ((.1, 5), (.05, 3), (.05, 7)):
    environment = dict(os.environ, LEGACY_IMP_ALPHA=str(alpha), LEGACY_SOURCE_EPOCHS=str(epochs))
    print('NEXT_DECLARED_ARM', alpha, epochs, flush=True)
    process = subprocess.Popen(['python3', '-u', '/content/run_legacy_imp_tuning_arm_colab.py'], env=environment)
    if process.wait():
        raise RuntimeError('Stop sweep after failed arm; no automatic setting changes')
print('DECLARED_SWEEP_COMPLETE', flush=True)
