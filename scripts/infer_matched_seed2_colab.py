"""ONE CPU-only own-seed candidate audit, no SGD and no chained experiment."""
import hashlib
from pathlib import Path
import subprocess
import sys
worker=Path('/content/infer_matched_candidates_seed.py')
assert hashlib.sha256(worker.read_bytes()).hexdigest()=='7866f0cc122d2cbf83eb582d3cd1c0ee6c5536cc84a1cf19241290b82da0edab'
result=subprocess.run([sys.executable,'-u',str(worker),'--seed','2','--feature-sha256',
                       '37972fed3fb89f8418e38186b050e3f525ab25f6515faf2a38a00c6172889f14'],capture_output=True,text=True)
print(result.stdout,flush=True)
print(result.stderr,flush=True)
assert result.returncode==0
