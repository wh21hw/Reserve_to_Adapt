import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

run_dir = Path('/content/drive/MyDrive/OSDA/runs/a2w-pipeline-seed1')
metrics = json.loads((run_dir / 'a2w_smoke' / 'metrics.json').read_text())
assert metrics['epoch'] == 5 and metrics['steps'] == 25, metrics
assert all(math.isfinite(metrics[key]) for key in ('OS', 'OS_star', 'unknown', 'HOS')), metrics
assert (run_dir / 'a2w_smoke' / 'last.pt').is_file()
(run_dir / 'pip-freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
output = Path('/content/osda-pipeline-results.zip')
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for file in run_dir.rglob('*'):
        if file.is_file() and file.suffix in ('.json', '.txt', '.log'):
            archive.write(file, file.relative_to(run_dir))
print('RESULTS_READY', output, metrics, flush=True)
