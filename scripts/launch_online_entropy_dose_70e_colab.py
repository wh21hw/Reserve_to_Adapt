"""Fixed half/zero working points at full budget, no new method changes."""
import subprocess

with open('/content/online-entropy-dose-70e-console.log','x') as log:
    worker=subprocess.Popen(['/content/rta-py38/bin/python','-u','/content/run_online_teacher_feature_colab.py',
        '--run','--preset','entropy-dose-validation','--run-name','online-imp-entropy-dose-70e-v1',
        '--epochs','70','--arm-seconds','5400','--total-seconds','10800'],
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('ONLINE_ENTROPY_DOSE_70E_LAUNCHER',worker.pid,flush=True)
    code=worker.wait()
print('ONLINE_ENTROPY_DOSE_70E_AND_COLLECTOR_DONE',code,flush=True)
if code:raise RuntimeError('Long working-point validation failed; preserve evidence, no auto-retry')
