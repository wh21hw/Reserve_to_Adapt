"""Launch the declared long validation; collector is part of the stable runner."""
import subprocess

command=['/content/rta-py38/bin/python','-u','/content/run_online_teacher_feature_colab.py',
    '--run','--preset','training-support','--run-name','online-imp-training-support-70e-v1',
    '--epochs','70','--arm-seconds','5400','--total-seconds','10800']
with open('/content/online-training-support-70e-console.log','x') as log:
    worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('ONLINE_TRAINING_SUPPORT_70E_LAUNCHER',worker.pid,flush=True)
    code=worker.wait()
print('ONLINE_TRAINING_SUPPORT_70E_AND_COLLECTOR_DONE',code,flush=True)
if code:
    raise RuntimeError('Long validation failed; preserve evidence, do not auto-retry')
