"""Task adaptation around the selected legacy published-code RTA backbone/losses.

Office31 keeps Q20. Other tasks require an explicit virtual-cluster count;
do not claim undocumented author settings. This is not paper-equation RTA.
"""
import json
import os
from pathlib import Path
import sys

root = Path('/content/rta-legacy-l4-bridge-v1')
sys.path.insert(0, str(root))
source = (root/'main.py').read_text()


def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError('Unexpected legacy task anchor: ' + old[:80])
    source = source.replace(old, new)


source = '''from task_protocol import (OFFICE31_A2W, OFFICEHOME_PR2RW, visda_protocol,
                           audit_lists, macro_open_set_metrics)
''' + source
replace_once('    #domains', '''    parser.add_argument('--task', required=True,
        choices=['office31-a2w', 'officehome-pr2rw', 'visda-synthetic2real'])
    parser.add_argument('--virtual-clusters', required=True, type=int)
    parser.add_argument('--class-map')
    parser.add_argument('--accept-unresolved-visda-resnet', action='store_true')
    #domains''')
replace_once('args = get_args()', '''args = get_args()
if args.use_VGG:
    raise ValueError('This legacy entry only implements ResNet-50')
if args.task == 'visda-synthetic2real':
    if not args.class_map or not args.accept_unresolved_visda_resnet:
        raise ValueError('VisDA requires explicit map and unresolved-ResNet acceptance')
    with open(args.class_map) as mapping_file:
        protocol = visda_protocol({int(key): value for key, value in json.load(mapping_file).items()})
else:
    protocol = {'office31-a2w': OFFICE31_A2W, 'officehome-pr2rw': OFFICEHOME_PR2RW}[args.task]
if args.shared_classes != protocol.known_classes or args.all_classes <= args.shared_classes:
    raise ValueError('Classifier dimensions conflict with task protocol')
if args.virtual_clusters <= args.shared_classes:
    raise ValueError('Virtual clustering must leave unmatched directions')
protocol_manifest = audit_lists(protocol, args.source, args.target, args.data_dir)
args.data_dir = os.path.join(os.path.abspath(args.data_dir), '')
''')
replace_once("args.log_dir = args.log_dir + args.source.split('/')[-1][0]+'2'+args.target.split('/')[-1][0]+'_'+args.name",
'''args.log_dir = os.path.join(args.log_dir, args.task + '_' + args.name)
if os.path.exists(args.log_dir):
    raise FileExistsError('Refusing to overwrite an existing experiment')
''')
replace_once('    json.dump(dict(vars(args), seed=seed), config_file, indent=2)',
'''    json.dump(dict(vars(args), seed=seed), config_file, indent=2)
with open(os.path.join(args.log_dir, 'protocol.json'), 'w') as protocol_file:
    json.dump(protocol_manifest, protocol_file, indent=2)''')
replace_once('    label = one_hot(args.all_classes, label)',
             '    label = one_hot(args.all_classes, protocol.source_label(label))')
replace_once('''    if label in range(10):
        label = one_hot(11, label)
    else:
        label = one_hot(11,10)''',
'''    # Training does not see target semantics. lt is bookkeeping only.
    label = one_hot(args.shared_classes + 1, args.shared_classes)''')
replace_once('    label = one_hot(31,label)',
             '    label = one_hot(max(protocol.known_ids + protocol.unknown_ids) + 1, label)')
replace_once('    K_cluster =20# cluster target class centroids',
             '    K_cluster = args.virtual_clusters # Explicit task virtual count')
old_evaluation = '''    m = extended_confusion_matrix(y_true, y_pred, true_labels=(list(range(args.shared_classes))+list(range(20,31))), pred_labels=list(range(args.all_classes)))

    cm = m
    cm = cm.astype(float) / np.sum(cm, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(args.shared_classes)]) / args.shared_classes
    unkn = sum(sum([cm[i][args.shared_classes:] for i in range(10, 21)])) / 11  
    acc_os = (acc_os_star * args.shared_classes + unkn) / 11'''
replace_once(old_evaluation, '''    semantic = np.where(y_pred < args.shared_classes, y_pred, args.shared_classes)
    evaluation = macro_open_set_metrics(protocol, y_true, semantic)
    acc_os_star, unkn = evaluation['OS_star'], evaluation['UNK']
    acc_os = (acc_os_star * args.shared_classes + unkn) / (args.shared_classes + 1)''')
# Initial virtual matching may have fewer rows than C+K. Preserve remaining
# random rows; do not increase Q behind the user's back or truncate inferred K.
replace_once('param = torch.from_numpy(v.cpu().numpy()[t_match]).cuda().detach().clone()',
'''if len(t_match) < args.all_classes:
                remaining = [j for j in range(args.all_classes) if j not in t_match]
                t_match = np.concatenate([t_match, np.asarray(remaining, dtype=np.int64)])
            param = torch.from_numpy(v.cpu().numpy()[t_match]).cuda().detach().clone()''')
if os.environ.get('KONLY_SOURCE_PRIOR'):
    replace_once('net = nn.Sequential(feature_extractor, cls).cuda()', '''net = nn.Sequential(feature_extractor, cls).cuda()
source_state = torch.load(os.environ['KONLY_SOURCE_PRIOR'], map_location='cpu')
if source_state['config']['known_classes'] != args.shared_classes:
    raise ValueError('Shared source checkpoint has different C')
with torch.no_grad():
    for parameter_name, value in net.state_dict().items():
        previous = source_state['model'][parameter_name]
        if previous.shape == value.shape:
            value.copy_(previous)
        elif parameter_name in ('1.fc.weight', '1.main.1.2.weight'):
            value[:args.shared_classes].copy_(previous)
        else:
            raise ValueError('Unexpected shared source shape: '+parameter_name)
del source_state
print('TASK_KONLY_START', dict(C=args.shared_classes, K=args.all_classes-args.shared_classes,
    Q=args.virtual_clusters, source_prior=os.environ['KONLY_SOURCE_PRIOR']), flush=True)
''')
epochs = int(os.environ.get('RTA_EPOCHS', '70'))
bn_policy=os.environ.get('RTA_FREEZE_ENCODER_BN','0')
if bn_policy not in ('0','1'):
    raise ValueError('RTA_FREEZE_ENCODER_BN must be 0 or 1')
if bn_policy=='1':
    replace_once('net = nn.Sequential(feature_extractor, cls).cuda()', '''net = nn.Sequential(feature_extractor, cls).cuda()
from encoder_bn_policy import freeze_encoder_bn_on_forward
encoder_bn_hook, encoder_bn_count = freeze_encoder_bn_on_forward(feature_extractor)
print('RTA_ENCODER_BN_POLICY', dict(mode='frozen_running_stats', modules=encoder_bn_count,
    affine_trainable=True, head_bn_unchanged=True), flush=True)
''')
if epochs < 1:
    raise ValueError('RTA_EPOCHS must be positive')
replace_once('while epoch <70:', 'while epoch <'+str(epochs)+':')
replace_once('                loss.backward()', '''                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite RTA loss; preserve evidence')
                loss.backward()''')
compiled = compile(source, str(root/'main.py'), 'exec')
if os.environ.get('LEGACY_TASK_BUILD_ONLY') == '1':
    print('LEGACY_TASK_SOURCE_BUILD_COMPLETE: no model/training executed', flush=True)
else:
    exec(compiled, dict(__name__='__main__', __file__=str(root/'main.py')))
