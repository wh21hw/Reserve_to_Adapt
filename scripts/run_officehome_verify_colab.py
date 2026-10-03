"""Validate direct-transfer data in a fresh worker; no training or Drive use."""
import os
from pathlib import Path
import subprocess
import sys

assert Path('/content/officehome-assembly-v1.json').is_file()
environment = dict(os.environ, OFFICEHOME_INPUTS='/content')
with Path('/content/officehome-verification-console-v1.log').open('x') as stream:
    process = subprocess.Popen([sys.executable, '-u', '/content/verify_officehome_colab.py'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
    for line in process.stdout:
        print(line, end='', flush=True)
        stream.write(line)
        stream.flush()
    status = process.wait()
if status:
    raise RuntimeError(f'Input verification failed ({status}); preserve evidence, do not train')
