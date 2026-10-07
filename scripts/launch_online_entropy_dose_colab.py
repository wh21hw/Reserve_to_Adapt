"""Three declared entropy working points, same shared warm state."""
import subprocess

with open('/content/online-entropy-dose-console.log','x') as log:
    worker=subprocess.Popen(['/content/rta-py38/bin/python','-u','/content/run_online_teacher_feature_colab.py',
        '--run','--preset','entropy-dose','--arm-seconds','3600','--total-seconds','10800'],
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('ONLINE_ENTROPY_DOSE_LAUNCHER',worker.pid,flush=True)
    code=worker.wait()
print('ONLINE_ENTROPY_DOSE_AND_COLLECTOR_DONE',code,flush=True)
if code:raise RuntimeError('Entropy working points failed; preserve evidence, no auto-retry')
