"""One warm4 + two resumed6-epoch arms; calibration support is the sole factor."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import zipfile


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def collect(root):
    shared=root/'warm/office31-a2w_seed3'
    warm=read_rows(shared/'history.jsonl')
    summary=dict(shared_warm_epochs=4,actual_new_training_epochs=16,
        pre4_max_abs_metric_difference_pp=0.,same_warm_state=True)
    for arm in ('confidence','structure_support'):
        run=root/arm/'office31-a2w_seed3'
        own=read_rows(run/'history.jsonl')
        if [row['epoch'] for row in own]!=list(range(5,11)):
            raise RuntimeError('Require exactly6 resumed epochs5–10')
        history=warm+own
        structures=read_rows(shared/'online-structure-history.jsonl')+read_rows(run/'online-structure-history.jsonl')
        support=read_rows(shared/'online-label-history.jsonl')+read_rows(run/'online-label-history.jsonl')
        summary[arm]=dict(best=max(history,key=lambda row:row['HOS']),
            post4_best=max(own,key=lambda row:row['HOS']),final=own[-1],
            history=history,structure=structures,support=support,
            K_trajectory=[row['K'] for row in history],V_trajectory=[row['V'] for row in history])
    summary['final_delta_pp']={key:100*(summary['structure_support']['final'][key]-summary['confidence']['final'][key])
        for key in ('OS_star','unknown','HOS')}
    summary['caveats']=['Posthocseed3','Oraclebest descriptive','Extra source3/frozenencoderBN',
        'Current source-only geometry is not independent semantic evidence',
        'Actual warmup trained once, not twice; no targettruth in support selection']
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    with zipfile.ZipFile('/content/online-imp-calibration-fork-v1-results.zip','x',zipfile.ZIP_DEFLATED) as bundle:
        for item in root.rglob('*'):
            if item.is_file() and item.suffix!='.pt':
                bundle.write(item,item.relative_to(root))
    durable=Path('/content/drive/MyDrive/OSDA/runs/online-imp-calibration-fork-v1')
    shutil.copyfile(root/'summary.json',durable/'summary.json')
    shutil.copyfile('/content/online-imp-calibration-fork-v1-results.zip',durable/'results.zip')
    print('ONLINE_CALIBRATION_FORK_COMPLETE',json.dumps(summary['final_delta_pp']),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run: raise ValueError('Explicit--run required')
    root=Path('/content/imp-runs/online-imp-calibration-fork-v1')
    durable=Path('/content/drive/MyDrive/OSDA/runs/online-imp-calibration-fork-v1')
    if root.exists() or durable.exists(): raise FileExistsError('Preserve existing output')
    if not durable.parent.is_dir(): raise RuntimeError('Drive mount required')
    root.mkdir(parents=True);durable.mkdir()
    started=time.monotonic()
    stages=[('warm',4,'confidence',None),
        ('restore_check',4,'confidence',root/'warm/office31-a2w_seed3/last.pt'),
        ('confidence',10,'confidence',root/'warm/office31-a2w_seed3/last.pt'),
        ('structure_support',10,'structure_supported',root/'warm/office31-a2w_seed3/last.pt')]
    for name,epochs,mode,fork in stages:
        output=root/name;output.mkdir()
        data=Path('/content/osda-office31-a2w-v1')
        command=['/content/rta-py38/bin/python','-u','/content/train_online_calibration_fork_entry.py',
            '--task','office31-a2w','--shared_classes','10','--all_classes','12','--virtual-clusters','20',
            '--source',str(data/'amazon_0-9_train_all.txt'),'--target',str(data/'webcam_0-9_20-30_test.txt'),
            '--data_dir',str(data),'--log_dir',str(output),'--name','seed3','--batch_size','64','--learning_rate','0.00005']
        overrides=dict(RTA_SEED='3',RTA_EPOCHS=str(epochs),RTA_FREEZE_ENCODER_BN='1',ONLINE_STRUCTURE_LABELS='1',
            ONLINE_CALIBRATION_MODE=mode,KONLY_SOURCE_PRIOR='/content/online-source/source-final.pt',
            RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',PYTHONPATH='/content',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if fork: overrides['ONLINE_FORK_INPUT']=str(fork)
        env=dict(os.environ,**overrides)
        for key in ('LEGACY_TASK_BUILD_ONLY','RTA_CLUSTER_LABELS','RTA_UNKNOWN_CE_WEIGHT'):
            env.pop(key,None)
        if not fork: env.pop('ONLINE_FORK_INPUT',None)
        (output/'launch.json').write_text(json.dumps(dict(stage=name,command=command,environment=overrides,
            effective_epoch_limit=epochs,actual_new_epochs=0 if name=='restore_check' else 4 if name=='warm' else 6,
            shared_checkpoint=None if fork is None else str(fork),model_save_policy='Full last/warm onruntime, arm best and smallrecords onDrive'),indent=2))
        budget=min(300 if name=='restore_check' else 3600,7200-(time.monotonic()-started))
        if budget<=0: raise RuntimeError('Declared total budget exhausted')
        expired=threading.Event()
        with (output/'console.log').open('x') as log:
            worker=subprocess.Popen(command,env=env,cwd='/content/rta-legacy-l4-bridge-v1',
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            def expire():
                expired.set()
                try: os.killpg(worker.pid,signal.SIGTERM)
                except ProcessLookupError: pass
            timer=threading.Timer(budget,expire);timer.daemon=True;timer.start()
            print('ONLINE_FORK_STAGE_STARTED',name,worker.pid,flush=True)
            try:
                for line in worker.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=worker.wait()
            finally: timer.cancel()
        (output/'process-status.json').write_text(json.dumps(dict(exit_code=code,timed_out=expired.is_set())))
        for item in output.rglob('*'):
            if item.is_file() and (item.suffix!='.pt' or name in ('confidence','structure_support') and item.name=='best.pt'):
                destination=durable/name/item.relative_to(output)
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(item,destination)
        if code or expired.is_set(): raise RuntimeError('Stage failed; preserve evidence, noautomaticretry')
        if name=='warm' and [row['epoch'] for row in read_rows(output/'office31-a2w_seed3/history.jsonl')]!=[1,2,3,4]:
            raise RuntimeError('Require4 complete commonwarm epochs')
        if name=='restore_check' and list(output.rglob('history.jsonl')):
            raise RuntimeError('Restorecheck must perform zero new training')
        print('ONLINE_FORK_STAGE_COMPLETE',name,flush=True)
    collect(root)


if __name__=='__main__': main()
