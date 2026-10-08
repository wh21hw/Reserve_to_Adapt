"""Single declared shorter-warmup arm after current parameter batch."""
import os
from pathlib import Path
import subprocess
import time

deadline=time.monotonic()+4500
while True:
    try:os.kill(8396,0)
    except ProcessLookupError:break
    if time.monotonic()>deadline:raise RuntimeError('Current parameter batch active; no overlappingGPU')
    time.sleep(5)
if not Path('/content/legacy-imp-param-screen-v2-results.zip').is_file():
    raise RuntimeError('Previous batch incomplete; do not launch next')
env=dict(os.environ,LEGACY_SCREEN_MODE='warmup')
env.pop('LEGACY_RUN_NAME',None)
with open('/content/legacy-warmup2-console.log','x') as log:
    worker=subprocess.Popen(['/content/rta-py38/bin/python','-u','/content/run_legacy_warmup_screen_colab.py'],
        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('LEGACY_WARMUP2_LAUNCHER',worker.pid,flush=True)
    code=worker.wait()
print('LEGACY_WARMUP2_DONE',code,flush=True)
if code:raise RuntimeError('Short warmup experiment failed, preserve evidence')
