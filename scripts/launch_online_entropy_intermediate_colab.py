"""Two declared intermediate working points, stable entropy-weight interface."""
import subprocess

with open('/content/online-entropy-intermediate-10e-console.log','x') as log:
    worker=subprocess.Popen(['/content/rta-py38/bin/python','-u','/content/run_online_teacher_feature_colab.py',
        '--run','--preset','entropy-intermediate','--run-name','online-imp-entropy-intermediate-10e-v1',
        '--epochs','10','--arm-seconds','900','--total-seconds','1800'],
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('ONLINE_ENTROPY_INTERMEDIATE_LAUNCHER',worker.pid,flush=True)
    code=worker.wait()
print('ONLINE_ENTROPY_INTERMEDIATE_AND_COLLECTOR_DONE',code,flush=True)
if code:raise RuntimeError('Intermediate working-point experiment failed; preserve evidence')
