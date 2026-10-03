"""Launch only the estimated-K arm after its fixed-K control finishes."""
import json
import os
from pathlib import Path

control = Path('/content/imp-runs/konly-rta-v1/fixed2/a2w_seed3/metrics.json')
metrics = json.loads(control.read_text())
if metrics['epoch'] != 70 or metrics['seed'] != 3:
    raise RuntimeError('Fixed-K control is not complete')
os.environ['KONLY_ARM'] = 'estimated'
exec(compile(Path('/content/run_konly_rta_seed3.py').read_text(),
             '/content/run_konly_rta_seed3.py', 'exec'),
     dict(__name__='__main__', __file__='/content/run_konly_rta_seed3.py'))
