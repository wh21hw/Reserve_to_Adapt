"""Independent shell launcher waits for current GPU worker without kernel queue."""
import os
from pathlib import Path
import subprocess
import time

deadline=time.monotonic()+900
while True:
    try:os.kill(230888,0)
    except ProcessLookupError:break
    if time.monotonic()>deadline:raise RuntimeError('Current worker still active; no concurrent GPU start')
    time.sleep(3)
required=['/content/legacy-user-oct7/main_user_snapshot.py',
    '/content/legacy-user-oct7/IMPClusterer_user_snapshot.py','/content/train_legacy_imp_virtual_entry.py',
    '/content/osda-office31-a2w-v1/amazon_0-9_train_all.txt',
    '/content/osda-office31-a2w-v1/webcam_0-9_20-30_test.txt',
    '/content/osda-datasets/resnet50-19c8e357.pth']
if not all(Path(item).is_file() for item in required):raise RuntimeError('Missing legacy runtime inputs')
with open('/content/legacy-seed-screen-console.log','x') as log:
    worker=subprocess.Popen(['/content/rta-py38/bin/python','-u','/content/run_legacy_seed_screen_colab.py'],
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('LEGACY_SEED_SCREEN_LAUNCHER',worker.pid,flush=True)
    code=worker.wait()
print('LEGACY_SEED_SCREEN_DONE',code,flush=True)
if code:raise RuntimeError('Legacy batch failed; retain evidence')
