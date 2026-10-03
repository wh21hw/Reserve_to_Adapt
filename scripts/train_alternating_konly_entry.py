"""Separate alternating arm; same initial K-only entry and original objectives."""
from pathlib import Path

base = Path('/content/train_konly_rta_entry.py').read_text()
boundary = 'exec(compile(source, '
if base.count(boundary) != 1:
    raise RuntimeError('Unexpected common initialization entry')
namespace = dict(__name__='__main__', __file__='/content/train_konly_rta_entry.py')
exec(compile(base.split(boundary)[0], '/content/train_konly_rta_entry.py', 'exec'), namespace)
source = namespace['source']
anchor = 'while epoch <70:'
if source.count(anchor) != 1:
    raise RuntimeError('Unexpected original RTA epoch loop')
source = 'from alternating_konly import refresh_current_network\n' + source.replace(anchor,
    anchor + '''
    # Fixed beforehand: refresh before epochs 21/41/61, total budget remains70.
    if epoch in (20, 40, 60):
        refresh_current_network(net, cls, optimizer_cls, args, epoch)
''')
anchor = 'epoch=epoch, HOS=float(hos))'
if source.count(anchor) != 1:
    raise RuntimeError('Unexpected best checkpoint writer')
source = source.replace(anchor, 'epoch=epoch, HOS=float(hos), K=args.all_classes-args.shared_classes)')
anchor = 'epoch=epoch, metrics=metrics,'
if source.count(anchor) != 1:
    raise RuntimeError('Unexpected final checkpoint writer')
source = source.replace(anchor, 'epoch=epoch, metrics=metrics, K=args.all_classes-args.shared_classes,')
anchor = 'elapsed_seconds=time.time()-started_at, seed=seed)'
if source.count(anchor) != 1:
    raise RuntimeError('Unexpected metrics writer')
source = source.replace(anchor, 'elapsed_seconds=time.time()-started_at, seed=seed, K=args.all_classes-args.shared_classes)')
compiled = compile(source, '/content/rta-legacy-l4-bridge-v1/main.py', 'exec')
if namespace['os'].environ.get('KONLY_BUILD_ONLY') == '1':
    print('ALTERNATING_SOURCE_BUILD_COMPLETE: no model or training executed', flush=True)
else:
    exec(compiled, namespace)
