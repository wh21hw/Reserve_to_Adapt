"""Fresh20-epoch declared two arms after interrupted Oct7 batch."""
import os
import subprocess

env=dict(os.environ,LEGACY_SCREEN_MODE='parameters',LEGACY_RUN_NAME='legacy-imp-param-screen-v2')
with open('/content/legacy-param-v2-console.log','x') as log:
    worker=subprocess.Popen(['/content/rta-py38/bin/python','-u','/content/run_legacy_param_screen_colab.py'],
        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('LEGACY_PARAM_V2_LAUNCHER',worker.pid,flush=True)
    code=worker.wait()
print('LEGACY_PARAM_V2_DONE',code,flush=True)
if code:raise RuntimeError('Batch failed, preserve evidence')
