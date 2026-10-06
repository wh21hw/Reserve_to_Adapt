"""A single declared fixed8/refresh20 pair, serial, no automatic retries."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    if args.build_only == args.run:
        raise ValueError('Choose exactly one build-only/run mode')
    root = Path('/content/imp-runs/a2w-capacity-refresh-20e-v1')
    prior = root/'source/source-final.pt'
    data = Path('/content/osda-office31-a2w-v1')
    if args.run:
        summary = json.loads((root/'source/summary.json').read_text())
        if not summary['complete'] or (summary['known_classes'],summary['epochs']) != (10,3):
            raise ValueError('Require completed shared C10 source3')
        if not prior.is_file() or not torch.cuda.is_available():
            raise RuntimeError('Missing shared prior or GPU; no CPU training fallback')
    started = time.monotonic()
    for arm,apply in (('fixed8','0'),('refresh','1')):
        output = root/arm
        command = ['/content/rta-py38/bin/python','-u','/content/train_a2w_capacity_refresh_entry.py',
            '--task','office31-a2w','--shared_classes','10','--all_classes','18',
            '--virtual-clusters','20','--source',str(data/'amazon_0-9_train_all.txt'),
            '--target',str(data/'webcam_0-9_20-30_test.txt'),'--data_dir',str(data),
            '--log_dir',str(output),'--name','seed3','--batch_size','64',
            '--learning_rate','0.00005']
        overrides = dict(RTA_SEED='3',RTA_EPOCHS='20',RTA_FREEZE_ENCODER_BN='1',
            RTA_APPLY_CAPACITY_REFRESH=apply,KONLY_SOURCE_PRIOR=str(prior),
            RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH='/content')
        env = dict(os.environ,**overrides)
        for key in ('RTA_CLUSTER_LABELS','RTA_UNKNOWN_CE_WEIGHT','LEGACY_TASK_BUILD_ONLY'):
            env.pop(key,None)
        if args.build_only:
            env['LEGACY_TASK_BUILD_ONLY'] = '1'
            subprocess.run(command,env=env,cwd='/content/rta-legacy-l4-bridge-v1',check=True)
            continue
        output.mkdir(exist_ok=False)
        launch = dict(task='Office31 A->W',arm=arm,C=10,initial_K=8,Q=20,seed=3,
            epochs=20,source_epochs=3,source_prior=str(prior),command=command,environment=overrides,
            runtime_environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),
            refresh_completed_epochs=[10],apply_capacity=apply=='1',
            inference='Frozen source-calibrated-birth-cost-v1 + known-head identity reconciliation',
            losses='Original released-code RTA; no new losses, gate, identity training labels or prediction rule',
            initialization='Shared source3 prior, initial C10+K8 and original warm-end K-means',
            head_resize='Known and matched unknown weights/SGD preserved; no scheduler/warmup reset',
            selection='Previously posthoc-selected seed3; target-oracle best descriptive, final20 also reported',
            budget_seconds=7200,caveat='Single-seed exploratory pair, not semantic count or full70 proof')
        (output/'launch.json').write_text(json.dumps(launch,indent=2))
        timeout_flag = threading.Event()
        remaining = min(7200,14400-(time.monotonic()-started))
        if remaining <= 0:
            raise RuntimeError('Declared total training time budget exhausted')
        with (output/'console.log').open('x') as log:
            process = subprocess.Popen(command,env=env,cwd='/content/rta-legacy-l4-bridge-v1',
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            def expire(worker=process,flag=timeout_flag):
                flag.set()
                try:
                    os.killpg(worker.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
            timer = threading.Timer(remaining,expire)
            timer.daemon = True
            timer.start()
            print('A2W_CAPACITY_ARM_STARTED',arm,'pid',process.pid,flush=True)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True)
                    log.write(line);log.flush()
                status = process.wait()
            finally:
                timer.cancel()
        (output/'process-status.json').write_text(json.dumps(dict(exit_code=status,
            timed_out=timeout_flag.is_set(),pid=process.pid)))
        if status or timeout_flag.is_set():
            raise RuntimeError('Arm failed/budget expired; preserve evidence, no retry')
        history = [json.loads(line) for line in (output/'office31-a2w_seed3/history.jsonl').read_text().splitlines()]
        if [row['epoch'] for row in history] != list(range(1,21)):
            raise RuntimeError('Incomplete declared twenty epochs')
        saved = Path('/content/drive/MyDrive/OSDA/runs/a2w-capacity-refresh-20e-v1')
        if not saved.parent.is_dir():
            raise RuntimeError('Drive unavailable; keep local result, no next arm')
        import shutil
        durable = saved/arm
        durable.mkdir(parents=True,exist_ok=False)
        shutil.copytree(output,durable,dirs_exist_ok=True)
        print('A2W_CAPACITY_ARM_COMPLETE_AND_SAVED',arm,flush=True)
    print('A2W_CAPACITY_PAIR_BUILD_COMPLETE' if args.build_only else 'A2W_CAPACITY_PAIR_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
