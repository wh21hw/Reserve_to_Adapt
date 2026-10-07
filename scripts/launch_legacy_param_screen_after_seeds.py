"""Two declared parameter arms after current fresh-seed batch."""
import os
from pathlib import Path
import subprocess
import time

deadline=time.monotonic()+3600
while True:
    try:os.kill(234371,0)
    except ProcessLookupError:break
    if time.monotonic()>deadline:raise RuntimeError('Previous batch active, no concurrent GPU launch')
    time.sleep(5)
if not Path('/content/legacy-imp-seed-screen-v1-results.zip').is_file():
    raise RuntimeError('Previous batch did not complete; preserve evidence, no automatic following experiment')
env=dict(os.environ,LEGACY_SCREEN_MODE='parameters')
with open('/content/legacy-param-screen-console.log','x') as log:
    worker=subprocess.Popen(['/content/rta-py38/bin/python','-u','/content/run_legacy_seed_screen_colab.py'],
        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('LEGACY_PARAM_SCREEN_LAUNCHER',worker.pid,flush=True)
    code=worker.wait()
print('LEGACY_PARAM_SCREEN_DONE',code,flush=True)
if code:raise RuntimeError('Parameter screen failed, no automatic retry')
