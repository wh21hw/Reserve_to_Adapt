"""Reuse commonwarm4; compare IMP256 vs2048, no backbone architecture change."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import zipfile


def rows(path): return [json.loads(line) for line in path.read_text().splitlines()]


def collect(root,shared,arms,run_name,epochs=10):
    warm=rows(shared/'history.jsonl')
    summary=dict(shared_warm_epochs=4,reused_warm_checkpoint=str(shared/'last.pt'),
        actual_new_training_epochs=len(arms)*(epochs-4),effective_epochs_per_arm=epochs,same_warm_state=True,pre4_max_abs_metric_difference_pp=0.)
    for arm in arms:
        run=root/arm/'office31-a2w_seed3'
        own=rows(run/'history.jsonl')
        if [row['epoch'] for row in own]!=list(range(5,epochs+1)):raise RuntimeError('Incomplete declared resumed epoch range')
        all_rows=warm+own
        summary[arm]=dict(best=max(all_rows,key=lambda row:row['HOS']),post4_best=max(own,key=lambda row:row['HOS']),
            final=own[-1],history=all_rows,K_trajectory=[row['K'] for row in all_rows],V_trajectory=[row['V'] for row in all_rows],
            structure=rows(shared/'online-structure-history.jsonl')+rows(run/'online-structure-history.jsonl'),
            support=rows(shared/'online-label-history.jsonl')+rows(run/'online-label-history.jsonl'))
    summary['final_delta_pp']={key:100*(summary[arms[1]]['final'][key]-summary[arms[0]]['final'][key])
        for key in ('OS_star','unknown','HOS')}
    summary['arms']=list(arms)
    summary['deltas_vs_control_pp']={arm:{key:100*(summary[arm]['final'][key]-summary[arms[0]]['final'][key])
        for key in ('OS_star','unknown','HOS')} for arm in arms[1:]}
    summary['caveats']=['Posthocseed3/oraclebest','Extra source3/frozenencoderBN',
        'Only the declared preset factor changes;virtual remains256',
        'Semantic ARI/NMI and saved exposure annotation are evaluation only; not calibration']
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    archive=Path('/content')/(run_name+'-results.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        for item in root.rglob('*'):
            if item.is_file() and item.suffix!='.pt':bundle.write(item,item.relative_to(root))
    durable=Path('/content/drive/MyDrive/OSDA/runs')/run_name
    shutil.copyfile(root/'summary.json',durable/'summary.json');shutil.copyfile(archive,durable/'results.zip')
    print('ONLINE_TEACHER_FEATURE_PAIR_COMPLETE',json.dumps(summary['final_delta_pp']),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true')
    parser.add_argument('--run-name',help='Separate declared confirmation output name; never overwrites prior runs')
    parser.add_argument('--epochs',type=int,choices=[10,70],default=10)
    parser.add_argument('--arm-seconds',type=int,default=3600)
    parser.add_argument('--total-seconds',type=int,default=7200)
    parser.add_argument('--preset',choices=['teacher-feature','label-coverage','known-veto','known-components','entropy-safety','head-bn','member-init','candidate-merge','training-support','entropy-dose','entropy-dose-validation'],default='teacher-feature')
    args=parser.parse_args()
    if not args.run:raise ValueError('Explicit--run required')
    if args.arm_seconds<=0 or args.total_seconds<=0:raise ValueError('Positive budgets required')
    if args.epochs!=10 and (args.preset not in ('training-support','entropy-dose-validation') or not args.run_name):
        raise ValueError('Long validation requires declared training-support preset and separate run-name')
    run_name={'teacher-feature':'online-imp-teacher-feature-v1','label-coverage':'online-imp-label-coverage-v1','known-veto':'online-imp-known-veto-v1','known-components':'online-imp-known-components-v1','entropy-safety':'online-imp-entropy-safety-v1','head-bn':'online-imp-head-bn-v1','member-init':'online-imp-member-init-v1','candidate-merge':'online-imp-candidate-merge-v1','training-support':'online-imp-training-support-v1','entropy-dose':'online-imp-entropy-dose-v1','entropy-dose-validation':'online-imp-entropy-dose-70e-v1'}[args.preset]
    if args.run_name:
        if not re.fullmatch(r'online-imp-[a-z0-9-]{1,64}',args.run_name):
            raise ValueError('Invalid managed run name')
        run_name=args.run_name
    arms={'teacher-feature':('bottleneck','backbone'),'label-coverage':('screened','all_candidates'),'known-veto':('rta_known','imp_veto'),'known-components':('entropy_veto','alignment_veto'),'entropy-safety':('raw_entropy','screened_entropy'),'head-bn':('batch_bn','fixed_bn'),'member-init':('source_only','current_members'),'candidate-merge':('no_merge','objective_merge'),'training-support':('rta_only','reliable_union'),'entropy-dose':('entropy1','entropy0p5','entropy0'),'entropy-dose-validation':('entropy0p5','entropy0')}[args.preset]
    root=Path('/content/imp-runs')/run_name
    durable=Path('/content/drive/MyDrive/OSDA/runs')/run_name
    shared=Path('/content/imp-runs/online-imp-calibration-fork-v1/warm/office31-a2w_seed3')
    if root.exists() or durable.exists():raise FileExistsError('Preserve existing outputs')
    if not (shared/'last.pt').is_file() or [row['epoch'] for row in rows(shared/'history.jsonl')]!=[1,2,3,4]:
        raise RuntimeError('Require existing complete commonwarm4, never retrain it')
    root.mkdir(parents=True);durable.mkdir()
    started=time.monotonic()
    for arm in arms:
        output=root/arm;output.mkdir()
        data=Path('/content/osda-office31-a2w-v1')
        command=['/content/rta-py38/bin/python','-u','/content/train_online_teacher_feature_entry.py',
            '--task','office31-a2w','--shared_classes','10','--all_classes','12','--virtual-clusters','20',
            '--source',str(data/'amazon_0-9_train_all.txt'),'--target',str(data/'webcam_0-9_20-30_test.txt'),
            '--data_dir',str(data),'--log_dir',str(output),'--name','seed3','--batch_size','64','--learning_rate','0.00005']
        teacher=arm if args.preset=='teacher-feature' else 'bottleneck'
        scope=arm if args.preset=='label-coverage' else 'screened'
        overrides=dict(RTA_SEED='3',RTA_EPOCHS=str(args.epochs),RTA_FREEZE_ENCODER_BN='1',ONLINE_STRUCTURE_LABELS='1',
            ONLINE_CALIBRATION_MODE='confidence',ONLINE_TEACHER_SPACE=teacher,ONLINE_LABEL_SCOPE=scope,
            ONLINE_KNOWN_VETO='1' if arm=='imp_veto' else '0',
            ONLINE_KNOWN_SCOPE={'imp_veto':'both','entropy_veto':'entropy','alignment_veto':'alignment','raw_entropy':'entropy','screened_entropy':'entropy'}.get(arm,'none'),
            ONLINE_VETO_ELIGIBILITY='screened' if arm=='screened_entropy' else 'raw',
            ONLINE_HEAD_BN_MODE='fixed' if arm=='fixed_bn' else 'batch',
            ONLINE_IMP_INITIALIZATION='current_members' if arm=='current_members' or args.preset=='candidate-merge' else 'source_only',
            ONLINE_IMP_MERGE='objective' if arm=='objective_merge' else 'none',
            ONLINE_UNKNOWN_SELECTION=arm if args.preset=='training-support' else 'rta_only',
            ONLINE_FORK_INPUT=str(shared/'last.pt'),
            KONLY_SOURCE_PRIOR='/content/online-source/source-final.pt',RTA_MODEL_PATH='/content/osda-datasets/resnet50-19c8e357.pth',
            PYTHONPATH='/content',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        env=dict(os.environ,**overrides)
        if args.preset in ('entropy-dose','entropy-dose-validation'):
            overrides.update(ONLINE_UNKNOWN_SELECTION='reliable_union',ONLINE_KNOWN_SCOPE='entropy',
                ONLINE_ENTROPY_CANDIDATE_SCALE={'entropy1':'1','entropy0p5':'0.5','entropy0':'0'}[arm])
            env=dict(os.environ,**overrides)
        for key in ('LEGACY_TASK_BUILD_ONLY','RTA_CLUSTER_LABELS','RTA_UNKNOWN_CE_WEIGHT','ONLINE_FEATURE_INTERFACE_ONLY'):env.pop(key,None)
        (output/'launch.json').write_text(json.dumps(dict(arm=arm,command=command,environment=overrides,
            actual_new_epochs=args.epochs-4,effective_final_epoch=args.epochs,backbone_architecture_changed=False,
            virtual_feature_space=256,teacher_feature_space=256 if teacher=='bottleneck' else 2048,
            label_scope=scope,preset=args.preset,head_initialization_isolates_sampling_rng=True,
            arm_seconds=args.arm_seconds,total_seconds=args.total_seconds,model_save_policy='Full last local, best/smallrecords Drive' if args.preset=='teacher-feature' else 'All modelsonruntime, smallrecords Drive'),indent=2))
        budget=min(args.arm_seconds,args.total_seconds-(time.monotonic()-started));expired=threading.Event()
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
            if item.is_file() and (item.suffix!='.pt' or args.preset=='teacher-feature' and item.name=='best.pt'):
                target=durable/arm/item.relative_to(output);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(item,target)
        if code or expired.is_set():raise RuntimeError('Arm failed; preserve evidence, noalgorithmretry')
        print('ONLINE_TEACHER_ARM_COMPLETE',arm,flush=True)
    collect(root,shared,arms,run_name,args.epochs)


if __name__=='__main__':main()
