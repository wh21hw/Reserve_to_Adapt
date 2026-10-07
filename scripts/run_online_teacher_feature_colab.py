"""Reuse commonwarm4; compare IMP256 vs2048, no backbone architecture change."""
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


def rows(path): return [json.loads(line) for line in path.read_text().splitlines()]


def collect(root,shared):
    warm=rows(shared/'history.jsonl')
    summary=dict(shared_warm_epochs=4,reused_warm_checkpoint=str(shared/'last.pt'),
        actual_new_training_epochs=12,same_warm_state=True,pre4_max_abs_metric_difference_pp=0.)
    for arm in ('bottleneck','backbone'):
        run=root/arm/'office31-a2w_seed3'
        own=rows(run/'history.jsonl')
        if [row['epoch'] for row in own]!=list(range(5,11)):raise RuntimeError('Require6 resumed epochs5–10')
        all_rows=warm+own
        summary[arm]=dict(best=max(all_rows,key=lambda row:row['HOS']),post4_best=max(own,key=lambda row:row['HOS']),
            final=own[-1],history=all_rows,K_trajectory=[row['K'] for row in all_rows],V_trajectory=[row['V'] for row in all_rows],
            structure=rows(shared/'online-structure-history.jsonl')+rows(run/'online-structure-history.jsonl'),
            support=rows(shared/'online-label-history.jsonl')+rows(run/'online-label-history.jsonl'))
    summary['final_delta_pp']={key:100*(summary['backbone']['final'][key]-summary['bottleneck']['final'][key])
        for key in ('OS_star','unknown','HOS')}
    summary['caveats']=['Posthocseed3/oraclebest','Extra source3/frozenencoderBN',
        'Only IMP capacity/label input changes;virtual remains256',
        'Semantic ARI/NMI and saved exposure annotation are evaluation only; not calibration']
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    archive=Path('/content/online-imp-teacher-feature-v1-results.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for item in root.rglob('*'):
            if item.is_file() and item.suffix!='.pt':bundle.write(item,item.relative_to(root))
    durable=Path('/content/drive/MyDrive/OSDA/runs/online-imp-teacher-feature-v1')
    shutil.copyfile(root/'summary.json',durable/'summary.json');shutil.copyfile(archive,durable/'results.zip')
    print('ONLINE_TEACHER_FEATURE_PAIR_COMPLETE',json.dumps(summary['final_delta_pp']),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:raise ValueError('Explicit--run required')
    root=Path('/content/imp-runs/online-imp-teacher-feature-v1')
    durable=Path('/content/drive/MyDrive/OSDA/runs/online-imp-teacher-feature-v1')
    shared=Path('/content/imp-runs/online-imp-calibration-fork-v1/warm/office31-a2w_seed3')
    if root.exists() or durable.exists():raise FileExistsError('Preserve existing outputs')
    if not (shared/'last.pt').is_file() or [row['epoch'] for row in rows(shared/'history.jsonl')]!=[1,2,3,4]:
        raise RuntimeError('Require existing complete commonwarm4, never retrain it')
    root.mkdir(parents=True);durable.mkdir()
    started=time.monotonic()
    for arm in ('bottleneck','backbone'):
        output=root/arm;output.mkdir()
        data=Path('/content/osda-office31-a2w-v1')
        command=['/content/rta-py38/bin/python','-u','/content/train_online_teacher_feature_entry.py',
            '--task','office31-a2w','--shared_classes','10','--all_classes','12','--virtual-clusters','20',
            '--source',str(data/'amazon_0-9_train_all.txt'),'--target',str(data/'webcam_0-9_20-30_test.txt'),
            '--data_dir',str(data),'--log_dir',str(output),'--name','seed3','--batch_size','64','--learning_rate','0.00005']
        overrides=dict(RTA_SEED='3',RTA_EPOCHS='10',RTA_FREEZE_ENCODER_BN='1',ONLINE_STRUCTURE_LABELS='1',
            ONLINE_CALIBRATION_MODE='confidence',ONLINE_TEACHER_SPACE=arm,ONLINE_FORK_INPUT=str(shared/'last.pt'),
            KONLY_SOURCE_PRIOR='/content/online-source/source-final.pt',RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
            PYTHONPATH='/content',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        env=dict(os.environ,**overrides)
        for key in ('LEGACY_TASK_BUILD_ONLY','RTA_CLUSTER_LABELS','RTA_UNKNOWN_CE_WEIGHT','ONLINE_FEATURE_INTERFACE_ONLY'):env.pop(key,None)
        (output/'launch.json').write_text(json.dumps(dict(arm=arm,command=command,environment=overrides,
            actual_new_epochs=6,effective_final_epoch=10,backbone_architecture_changed=False,
            virtual_feature_space=256,teacher_feature_space=256 if arm=='bottleneck' else 2048,
            arm_seconds=3600,total_seconds=7200,model_save_policy='Full last local, best/smallrecords Drive'),indent=2))
        budget=min(3600,7200-(time.monotonic()-started));expired=threading.Event()
        if budget<=0:raise RuntimeError('Total budget exhausted')
        with (output/'console.log').open('x') as log:
            worker=subprocess.Popen(command,env=env,cwd='/content/rta-legacy-l4-bridge-v1',stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,start_new_session=True)
            def expire():
                expired.set()
                try:os.killpg(worker.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            timer=threading.Timer(budget,expire);timer.daemon=True;timer.start()
            print('ONLINE_TEACHER_ARM_STARTED',arm,worker.pid,flush=True)
            try:
                for line in worker.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                code=worker.wait()
            finally:timer.cancel()
        (output/'process-status.json').write_text(json.dumps(dict(exit_code=code,timed_out=expired.is_set())))
        for item in output.rglob('*'):
            if item.is_file() and (item.suffix!='.pt' or item.name=='best.pt'):
                target=durable/arm/item.relative_to(output);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(item,target)
        if code or expired.is_set():raise RuntimeError('Arm failed; preserve evidence, noalgorithmretry')
        print('ONLINE_TEACHER_ARM_COMPLETE',arm,flush=True)
    collect(root,shared)


if __name__=='__main__':main()
