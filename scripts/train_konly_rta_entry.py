"""Original RTA training with a shared source initialization and variable K.

Both fixed and estimated K arms use this exact entry. No extra objective,
unknown marginalization, relation gate change, or prototype head initializer.
"""
import os
from pathlib import Path
import sys

code_root = Path('/content/rta-legacy-l4-bridge-v1')
source_prior = os.environ['KONLY_SOURCE_PRIOR']
sys.path.insert(0, str(code_root))
source = (code_root/'main.py').read_text()
anchor = 'net = nn.Sequential(feature_extractor, cls).cuda()'
if source.count(anchor) != 1:
    raise RuntimeError('Unexpected RTA model construction')
initialization = '''
# Shared C-class source pretraining in BOTH arms. No target-selected checkpoint.
source_state = torch.load(os.environ['KONLY_SOURCE_PRIOR'], map_location='cpu')
if source_state['config']['known_classes'] != args.shared_classes:
    raise ValueError('Source prior class count differs from RTA')
with torch.no_grad():
    for parameter_name, value in net.state_dict().items():
        previous = source_state['model'][parameter_name]
        if previous.shape == value.shape:
            value.copy_(previous)
        elif parameter_name in ('1.fc.weight', '1.main.1.2.weight'):
            value[:args.shared_classes].copy_(previous)
        else:
            raise ValueError('Unexpected source initialization shape: ' + parameter_name)
del source_state
print('KONLY_RTA_START', dict(seed=seed, K=args.all_classes-args.shared_classes,
                            source_prior=os.environ['KONLY_SOURCE_PRIOR'],
                            loss='original published-code RTA'), flush=True)
'''
source = source.replace(anchor, anchor + initialization)
# Original initial virtual matching has only Q=20 rows. If C+K exceeds Q,
# retain unmatched random head rows rather than changing Q or truncating K.
# For the original K=2 this leaves the code's behavior unchanged.
anchor = 'param = torch.from_numpy(v.cpu().numpy()[t_match]).cuda().detach().clone()'
if source.count(anchor) != 1:
    raise RuntimeError('Unexpected RTA initial head permutation')
source = source.replace(anchor, '''if len(t_match) < args.all_classes:
                remaining = [j for j in range(args.all_classes) if j not in t_match]
                t_match = np.concatenate([t_match, np.asarray(remaining, dtype=np.int64)])
            ''' + anchor)
exec(compile(source, str(code_root/'main.py'), 'exec'),
     dict(__name__='__main__', __file__=str(code_root/'main.py')))
