"""Wrap the stable legacy entry; alter capacity only after ten complete epochs."""
import os
from pathlib import Path

apply = os.environ['RTA_APPLY_CAPACITY_REFRESH']
if apply not in ('0','1') or os.environ.get('RTA_EPOCHS') != '20':
    raise ValueError('Require declared fixed/refresh20 arms')
if os.environ.get('RTA_CLUSTER_LABELS') or os.environ.get('RTA_UNKNOWN_CE_WEIGHT'):
    raise ValueError('This is capacity-only, not identity-label/CE ablation')
original = Path('/content/train_legacy_task_entry.py')
entry = original.read_text()
anchor = "compiled = compile(source, str(root/'main.py'), 'exec')"
if entry.count(anchor) != 1:
    raise ValueError('Unexpected stable legacy wrapper boundary')
patch = '''
replace_once('while epoch <20:', ''' + repr('''while epoch <20:
    if epoch == 10:
        from capacity_refresh_boundary import refresh_boundary
        refresh_boundary(net,cls,optimizer_cls,args,epoch,apply='''+str(apply=='1')+''')
''') + ''')
replace_once('epoch=epoch, HOS=float(hos))',
    'epoch=epoch, HOS=float(hos), K=args.all_classes-args.shared_classes)')
replace_once('elapsed_seconds=time.time()-started_at, seed=seed)',
    'elapsed_seconds=time.time()-started_at, seed=seed, K=args.all_classes-args.shared_classes)')
'''
entry = entry.replace(anchor,patch+'\n'+anchor)
exec(compile(entry,str(original),'exec'),dict(__name__='__main__',__file__=str(original)))
