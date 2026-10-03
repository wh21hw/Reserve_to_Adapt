from pathlib import Path
import subprocess
import sys

with Path('/content/hierarchical-collect-console-v1.log').open('x') as stream:
    process=subprocess.Popen([sys.executable,'-u','/content/collect_hierarchical_worker_v1.py'],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True); stream.write(line); stream.flush()
    status=process.wait()
if status:
    raise RuntimeError('Collector failed; preserve trained models, do not rerun training')
