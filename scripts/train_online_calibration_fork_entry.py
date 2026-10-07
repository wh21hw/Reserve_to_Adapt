"""Same4-epoch warmup checkpoint; alter only target calibration eligibility."""
import os
from pathlib import Path
from train_online_imp_structure_entry import build_training_source

source,namespace=build_training_source()
anchor="online_structure = OnlineStructure(args, os.environ['ONLINE_STRUCTURE_LABELS']=='1')"
if source.count(anchor)!=1:
    raise RuntimeError('Unexpected online structure initialization')
source=source.replace(anchor,anchor+'''
online_structure.calibration_mode = os.environ['ONLINE_CALIBRATION_MODE']
online_structure.teacher_space = os.environ.get('ONLINE_TEACHER_SPACE','bottleneck')
online_structure.label_scope = os.environ.get('ONLINE_LABEL_SCOPE','screened')
online_structure.known_veto = os.environ.get('ONLINE_KNOWN_VETO','0')=='1'
online_structure.known_scope = os.environ.get('ONLINE_KNOWN_SCOPE','both' if online_structure.known_veto else 'none')
online_structure.veto_eligibility = os.environ.get('ONLINE_VETO_ELIGIBILITY','raw')
online_structure.initialization_mode = os.environ.get('ONLINE_IMP_INITIALIZATION','source_only')
if os.environ.get('ONLINE_FORK_INPUT'):
    from online_imp_fork import restore_fork
    epoch,gmm,nomatch=restore_fork(os.environ['ONLINE_FORK_INPUT'],net,cls,discriminator,
        all_centroids,[optimizer_feature_extractor,optimizer_cls,optimizer_discriminator],
        online_structure,args)
''')
anchor='optimizer_discriminator=optimizer_discriminator.optimizer.state_dict()),'
replacement='''optimizer_discriminator=optimizer_discriminator.optimizer.state_dict(),
                    **extra_state(net,discriminator,all_centroids,
                        [optimizer_feature_extractor,optimizer_cls,optimizer_discriminator],
                        online_structure,gmm,nomatch)),'''
if source.count(anchor)!=1:
    raise RuntimeError('Unexpected saved last-checkpoint boundary')
source='from online_imp_fork import extra_state\n'+source.replace(anchor,replacement)
compiled=compile(source,'<online-calibration-fork>','exec')
if os.environ.get('LEGACY_TASK_BUILD_ONLY')=='1':
    print('ONLINE_CALIBRATION_FORK_BUILD_COMPLETE; no model or training')
else:
    exec(compiled,dict(__name__='__main__',__file__=str(namespace['root']/'main.py')))
