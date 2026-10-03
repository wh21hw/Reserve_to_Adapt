"""Run exactly one predeclared structure-off pilot in a fresh interpreter."""
from pathlib import Path
import subprocess
import sys

with Path('/content/matched-structure-off-console-v1.log').open('x') as stream:
    process = subprocess.Popen([sys.executable, '-u', '/content/train_matched_structure_pilot.py', '--arm', 'structure_off'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError(f'Pilot failed ({status}); preserve evidence, no automatic retuning/retry')
