"""Fresh complete-budget off collector; no training."""
import subprocess
import sys
result = subprocess.run([sys.executable,'-u','/content/collect_matched_full_a2w_seed1.py','--arm','structure_off'],
                        capture_output=True,text=True)
print(result.stdout,flush=True)
print(result.stderr,flush=True)
assert result.returncode==0
