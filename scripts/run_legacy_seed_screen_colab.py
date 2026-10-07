"""User legacy recipe, fresh seeds1/2/3, bounded20-epoch screening."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import zipfile

root=Path('/content/imp-runs/legacy-imp-seed-screen-v1')
durable=Path('/content/drive/MyDrive/OSDA/runs/legacy-imp-seed-screen-v1')
if root.exists() or durable.exists():raise FileExistsError('Preserve previous results')
root.mkdir(parents=True);durable.mkdir()
started=time.monotonic();summary={}
for seed in (1,2,3):
    output=root/('seed%d'%seed);output.mkdir()
    data=Path('/content/osda-office31-a2w-v1')
    command=['/content/rta-py38/bin/python','-u','/content/train_legacy_imp_virtual_entry.py',
        '--source',str(data/'amazon_0-9_train_all.txt'),'--target',str(data/'webcam_0-9_20-30_test.txt'),
        '--data_dir',str(data)+'/', '--log_dir',str(output),'--shared_classes','10','--all_classes','12',
        '--batch_size','64','--learning_rate','0.00005','--lambda1','.01','--lambda2','1','--lambda3','.3',
        '--name','seed%d'%seed]
    settings=dict(RTA_SEED=str(seed),LEGACY_SOURCE_EPOCHS='5',LEGACY_RTA_EPOCHS='20',LEGACY_IMP_ALPHA='.05',
        LEGACY_SNAPSHOT_DIR='/content/legacy-user-oct7',RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    env=dict(os.environ,**settings)
    for key in ('KONLY_SOURCE_PRIOR','RTA_FREEZE_ENCODER_BN','ONLINE_FORK_INPUT'):
        env.pop(key,None)
    (output/'launch.json').write_text(json.dumps(dict(command=command,environment=settings,source_epochs=5,rta_epochs=20),indent=2))
    expired=threading.Event();budget=min(1500,4500-(time.monotonic()-started))
    if budget<=0:raise RuntimeError('Total budget exhausted')
    with (output/'console.log').open('x') as log:
        worker=subprocess.Popen(command,env=env,cwd='/content/rta-legacy-l4-bridge-v1',stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        def expire():
            expired.set()
            try:os.killpg(worker.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        timer=threading.Timer(budget,expire);timer.daemon=True;timer.start()
        print('LEGACY_SEED_START',seed,worker.pid,flush=True)
        try:code=worker.wait()
        finally:timer.cancel()
    (output/'process-status.json').write_text(json.dumps(dict(exit_code=code,timed_out=expired.is_set())))
    for item in output.rglob('*'):
        if item.is_file() and item.suffix!='.pt':
            dest=durable/item.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(item,dest)
    if code or expired.is_set():raise RuntimeError('Seed failed; preserve evidence, no automatic retry')
    histories=list(output.rglob('history.jsonl'))
    if len(histories)!=1:raise RuntimeError('Missing seed history')
    history=[json.loads(line) for line in histories[0].read_text().splitlines()]
    if [row['epoch'] for row in history]!=list(range(1,21)):raise RuntimeError('Incomplete20 epochs')
    summary['seed%d'%seed]=dict(best=max(history,key=lambda row:row['HOS']),final=history[-1],history=history,
        V_range=[min(row['virtual_prototypes'] for row in history),max(row['virtual_prototypes'] for row in history)])
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('LEGACY_SEED_COMPLETE',seed,summary['seed%d'%seed]['best']['HOS'],flush=True)
archive=Path('/content/legacy-imp-seed-screen-v1-results.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for item in root.rglob('*'):
        if item.is_file() and item.suffix!='.pt':bundle.write(item,item.relative_to(root))
shutil.copyfile(root/'summary.json',durable/'summary.json');shutil.copyfile(archive,durable/'results.zip')
print('LEGACY_SEED_SCREEN_COLLECTOR_DONE',0,flush=True)
