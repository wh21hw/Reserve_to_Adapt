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

mode=os.environ.get('LEGACY_SCREEN_MODE','seeds')
if mode not in ('seeds','parameters','warmup'):raise ValueError('Unknown declared screen')
run_name={'seeds':'legacy-imp-seed-screen-v1','parameters':'legacy-imp-param-screen-v1','warmup':'legacy-imp-warmup2-20e-v1'}[mode]
if os.environ.get('LEGACY_RUN_NAME'):
    if mode!='parameters' or os.environ['LEGACY_RUN_NAME']!='legacy-imp-param-screen-v2':
        raise ValueError('Undeclared resumed run name')
    run_name=os.environ['LEGACY_RUN_NAME']
arms=[('seed%d'%seed,seed,.05,5) for seed in (1,2,3)] if mode=='seeds' else [('alpha0p01-ft5',3,.01,5),('alpha0p05-ft3',3,.05,3)]
if mode=='warmup':arms=[('warm2-alpha0p05-ft5',3,.05,5)]
root=Path('/content/imp-runs')/run_name
durable=Path('/content/drive/MyDrive/OSDA/runs')/run_name
if root.exists() or durable.exists():raise FileExistsError('Preserve previous results')
root.mkdir(parents=True);durable.mkdir()
started=time.monotonic();summary={}
for arm,seed,alpha,source_epochs in arms:
    output=root/arm;output.mkdir()
    data=Path('/content/osda-office31-a2w-v1')
    entry='/content/train_legacy_imp_warmup_entry.py' if mode=='warmup' else '/content/train_legacy_imp_virtual_entry.py'
    command=['/content/rta-py38/bin/python','-u',entry,
        '--source',str(data/'amazon_0-9_train_all.txt'),'--target',str(data/'webcam_0-9_20-30_test.txt'),
        '--data_dir',str(data)+'/', '--log_dir',str(output),'--shared_classes','10','--all_classes','12',
        '--batch_size','64','--learning_rate','0.00005','--lambda1','.01','--lambda2','1','--lambda3','.3',
        '--name','seed%d'%seed]
    settings=dict(RTA_SEED=str(seed),LEGACY_SOURCE_EPOCHS=str(source_epochs),LEGACY_RTA_EPOCHS='20',LEGACY_IMP_ALPHA=str(alpha),
        LEGACY_RTA_WARMITER='1' if mode=='warmup' else '3',
        LEGACY_SNAPSHOT_DIR='/content/legacy-user-oct7',RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    env=dict(os.environ,**settings)
    for key in ('KONLY_SOURCE_PRIOR','RTA_FREEZE_ENCODER_BN','ONLINE_FORK_INPUT'):
        env.pop(key,None)
    (output/'launch.json').write_text(json.dumps(dict(command=command,environment=settings,source_epochs=source_epochs,rta_epochs=20,arm=arm),indent=2))
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
    summary[arm]=dict(best=max(history,key=lambda row:row['HOS']),final=history[-1],history=history,
        V_range=[min(row['virtual_prototypes'] for row in history),max(row['virtual_prototypes'] for row in history)])
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('LEGACY_SEED_COMPLETE',arm,summary[arm]['best']['HOS'],flush=True)
archive=Path('/content')/(run_name+'-results.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
    for item in root.rglob('*'):
        if item.is_file() and item.suffix!='.pt':bundle.write(item,item.relative_to(root))
shutil.copyfile(root/'summary.json',durable/'summary.json');shutil.copyfile(archive,durable/'results.zip')
print('LEGACY_SEED_SCREEN_COLLECTOR_DONE',0,flush=True)
