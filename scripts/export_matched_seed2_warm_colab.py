"""ONE seed2 warm feature export, no SGD and no chained candidate inference."""
import hashlib
from pathlib import Path
import subprocess
import sys
worker=Path('/content/extract_multitask_warm_seed.py')
assert hashlib.sha256(worker.read_bytes()).hexdigest()=='c33bd1be61fd422d4132bc4b0c4157f0c779820b55303965f6337f4ce138db64'
result=subprocess.run([sys.executable,'-u',str(worker),'--seed','2'],capture_output=True,text=True)
print(result.stdout,flush=True)
print(result.stderr,flush=True)
assert result.returncode==0
