"""Two10-epoch arms, protected output paths, explicit subprocess deadlines."""
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


def collect(root):
    summary = {}
    for arm in ('self_label', 'structure_label'):
        run = root/arm/'office31-a2w_seed3'
        history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
        if [row['epoch'] for row in history] != list(range(1, 11)):
            raise RuntimeError('Require exactly10 complete epochs per arm')
        summary[arm] = dict(best=max(history, key=lambda row: row['HOS']), final=history[-1],
            K_trajectory=[row['K'] for row in history], V_trajectory=[row['V'] for row in history],
            structure=[json.loads(line) for line in (run/'online-structure-history.jsonl').read_text().splitlines()],
            support=[json.loads(line) for line in (run/'online-label-history.jsonl').read_text().splitlines()])
    summary['caveats'] = ['Posthoc seed3 selection', 'Target-oracle best descriptive only',
        'Extra source-only3/frozen encoder BN', 'Both arms share onlineK/V/threshold; only labels differ',
        'No certified semantic class discovery']
    summary['final_delta_pp'] = {key: 100*(summary['structure_label']['final'][key]-summary['self_label']['final'][key])
        for key in ('OS_star', 'unknown', 'HOS')}
    a = [json.loads(line) for line in (root/'self_label/office31-a2w_seed3/history.jsonl').read_text().splitlines()]
    b = [json.loads(line) for line in (root/'structure_label/office31-a2w_seed3/history.jsonl').read_text().splitlines()]
    summary['pre4_max_abs_metric_difference_pp'] = max(100*abs(x[key]-y[key])
        for x,y in zip(a[:4],b[:4]) for key in ('OS_star','unknown','HOS'))
    (root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    archive = Path('/content/online-imp-structure-v1-results.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for item in root.rglob('*'):
            if item.is_file() and item.suffix != '.pt':
                bundle.write(item, item.relative_to(root))
    shutil.copyfile(archive, Path('/content/drive/MyDrive/OSDA/runs')/archive.name)
    print('ONLINE_IMP_PAIR_COMPLETE', json.dumps(summary['final_delta_pp']), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        raise ValueError('Explicit --run required')
    root = Path('/content/imp-runs/online-imp-structure-v1')
    durable = Path('/content/drive/MyDrive/OSDA/runs/online-imp-structure-v1')
    if root.exists() or durable.exists():
        raise FileExistsError('Preserve existing run; no overwrite')
    if not durable.parent.is_dir():
        raise RuntimeError('Require mountedDrive before training')
    root.mkdir(parents=True)
    durable.mkdir()
    started = time.monotonic()
    for arm, use in (('self_label','0'), ('structure_label','1')):
        output = root/arm
        output.mkdir()
        data = Path('/content/osda-office31-a2w-v1')
        command = ['/content/rta-py38/bin/python','-u','/content/train_online_imp_structure_entry.py',
            '--task','office31-a2w','--shared_classes','10','--all_classes','12',
            '--virtual-clusters','20','--source',str(data/'amazon_0-9_train_all.txt'),
            '--target',str(data/'webcam_0-9_20-30_test.txt'),'--data_dir',str(data),
            '--log_dir',str(output),'--name','seed3','--batch_size','64','--learning_rate','0.00005']
        overrides = dict(RTA_SEED='3',RTA_EPOCHS='10',RTA_FREEZE_ENCODER_BN='1',
            ONLINE_STRUCTURE_LABELS=use,KONLY_SOURCE_PRIOR='/content/online-source/source-final.pt',
            RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH='/content')
        env = dict(os.environ, **overrides)
        for key in ('LEGACY_TASK_BUILD_ONLY','RTA_CLUSTER_LABELS','RTA_UNKNOWN_CE_WEIGHT'):
            env.pop(key, None)
        (output/'launch.json').write_text(json.dumps(dict(arm=arm, command=command,
            environment=overrides,epochs=10,arm_seconds=3600,total_seconds=7200), indent=2))
        budget = min(3600,7200-(time.monotonic()-started))
        if budget <= 0:
            raise RuntimeError('Declared total budget exhausted')
        expired = threading.Event()
        with (output/'console.log').open('x') as log:
            worker = subprocess.Popen(command,env=env,cwd='/content/rta-legacy-l4-bridge-v1',
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            def expire():
                expired.set()
                try:
                    os.killpg(worker.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
            timer = threading.Timer(budget, expire)
            timer.daemon=True
            timer.start()
            print('ONLINE_IMP_ARM_STARTED',arm,worker.pid,flush=True)
            try:
                for line in worker.stdout:
                    print(line,end='',flush=True); log.write(line); log.flush()
                status = worker.wait()
            finally:
                timer.cancel()
        (output/'process-status.json').write_text(json.dumps(dict(exit_code=status,timed_out=expired.is_set())))
        # Preserve logs even when worker failed. Model files ordinarycopy, nohash.
        shutil.copytree(output,durable/arm)
        if status or expired.is_set():
            raise RuntimeError('Worker failed; preserve evidence, no retry or next arm')
        print('ONLINE_IMP_ARM_SAVED',arm,flush=True)
    collect(root)


if __name__=='__main__':
    main()
