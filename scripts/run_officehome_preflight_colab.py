"""Execute one real-batch worker; no model updates or full training."""
from pathlib import Path
import subprocess
import sys

assert Path('/content/officehome-input-verification-v1.json').is_file()
with Path('/content/officehome-preflight-console-v1.log').open('x') as stream:
    process = subprocess.Popen([sys.executable, '-u', '/content/preflight_officehome_colab.py'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError(f'OfficeHome preflight failed ({status}); no training launched')
