from pathlib import Path
import subprocess
import sys

worker=Path('/content/preflight_hierarchical_worker_v1.py')
with Path('/content/hierarchical-preflight-console-v1.log').open('x') as stream:
    process=subprocess.Popen([sys.executable,'-u',str(worker)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:
        print(line,end='',flush=True); stream.write(line); stream.flush()
    status=process.wait()
if status:
    raise RuntimeError('Preflight failed; no training authorized by this worker')
