"""Fresh-process audit only; invoke after seed2 training is confirmed done."""
from pathlib import Path
import subprocess
import sys

with Path('/content/rta-multitask-a2w-seed2-collect-console-v1.log').open('x') as stream:
    process = subprocess.Popen(
        [sys.executable, '-u', '/content/collect_multitask_a2w_seed1_colab.py', '--seed', '2'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError(f'Collector failed ({status}); preserve evidence, no training rerun')
