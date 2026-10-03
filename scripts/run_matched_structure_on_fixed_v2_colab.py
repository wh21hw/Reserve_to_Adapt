"""One structure-on arm, launched only after the off-arm audit."""
from pathlib import Path
import hashlib
import subprocess
import sys

worker = Path('/content/train_matched_structure_pilot_fixed_v2.py')
assert hashlib.sha256(worker.read_bytes()).hexdigest() == 'bf5babdecdcd0e07c8ea2e6f553a07442009101d08565566a5bbdf208f02d75b'
with Path('/content/matched-structure_on-console-v2.log').open('x') as stream:
    process = subprocess.Popen([sys.executable, '-u', str(worker), '--arm', 'structure_on'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError(f'Pilot failed ({status}); preserve evidence; no automatic retry')
