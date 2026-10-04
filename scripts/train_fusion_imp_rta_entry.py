"""Independent fusion arm built on the existing published-code K-only entry.

Required: KONLY_SOURCE_PRIOR and FUSION_FEATURES (same prior's features.npz).
Optional: FUSION_EPOCHS (default 10), FUSION_FIXED_K (capacity control).
Does not launch anything on import.
"""
import os
from pathlib import Path


def build_source(source):
    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('Unexpected fusion anchor: ' + old[:80])
        source = source.replace(old, new)

    replace_once('args = get_args()', '''args = get_args()
fusion_result, fusion_settings = initial_structure(os.environ['FUSION_FEATURES'], args.shared_classes)
fusion_K = int(os.environ.get('FUSION_FIXED_K', fusion_result['candidate_count']))
if fusion_K < 1:
    raise ValueError('Fusion capacity must be positive')
args.all_classes = args.shared_classes + fusion_K
''')
    # Replace the original initial virtual clustering AND random head permutation.
    # Source known rows inherited from shared pretraining remain intact.
    start = source.index('customgenearator = DomainBus([source_train, target_train])')
    end = source.index('del(ProbRecorder)', start) + len('del(ProbRecorder)')
    source = source[:start] + '''nomatch = fusion_result['candidate_prototypes'].cuda()
import json
from pathlib import Path
fusion_config_path = Path(args.log_dir) / 'config.json'
fusion_config = json.loads(fusion_config_path.read_text()) if fusion_config_path.exists() else dict(vars(args))
fusion_config.update(design='fusion-v1: IMP capacity and refreshed virtual directions',
                    epochs=int(os.environ.get('FUSION_EPOCHS', '10')),
                    all_classes=args.all_classes, K=fusion_K,
                    K_policy='fixed control' if 'FUSION_FIXED_K' in os.environ else 'initial IMP estimate',
                    initial_V=fusion_result['candidate_count'],
                    frozen_features=os.environ['FUSION_FEATURES'],
                    source_prior=os.environ['KONLY_SOURCE_PRIOR'],
                    virtual_policy='each-epoch current frozen features and IMP',
                    target_labels_used_for_structure=False)
fusion_config_path.write_text(json.dumps(fusion_config, indent=2, allow_nan=False))
save_structure(fusion_result, fusion_settings, args.log_dir, 0,
               args.all_classes - args.shared_classes)
del fusion_result
''' + source[end:]
    # Retain source centroids used by the original warm-end unknown-head K-means.
    start = source.index('    faiss_kmeans = faiss.Kmeans(256, int(K_cluster)')
    end = source.index('    if epoch ==warmiter:', start)
    source = source[:start] + '''    nomatch = refresh_virtual(net, args, epoch + 1)

''' + source[end:]
    epochs = int(os.environ.get('FUSION_EPOCHS', '10'))
    if epochs < 1:
        raise ValueError('FUSION_EPOCHS must be positive')
    replace_once('while epoch <70:', 'while epoch <%d:' % epochs)
    replace_once('loss.backward()', "if not torch.isfinite(loss):\n                    raise RuntimeError('Nonfinite fusion loss; stop before update')\n                loss.backward()")
    # The engineering bridge writes these fields; official raw code does not.
    metric_anchor = 'elapsed_seconds=time.time()-started_at, seed=seed)'
    if metric_anchor in source:
        replace_once(metric_anchor, 'elapsed_seconds=time.time()-started_at, seed=seed, K=fusion_K, V=int(nomatch.size(0)))')
    if os.environ.get('FUSION_DIAGNOSTICS') == '1':
        replace_once('\nepoch = 0\n', '\nfusion_diagnostics = FusionDiagnostics()\nepoch = 0\n')
        replace_once('            ce = CrossEntropyLoss(label_source, nn.Softmax(-1)(fc_source))',
                     '            fusion_diagnostics.add(fc_target, args.shared_classes, r, weight)\n'
                     '            ce = CrossEntropyLoss(label_source, nn.Softmax(-1)(fc_source))')
        replace_once('    nomatch = refresh_virtual(net, args, epoch + 1)',
                     '    fusion_diagnostics.save(args.log_dir, epoch + 1)\n'
                     '    nomatch = refresh_virtual(net, args, epoch + 1)')
        source = 'from fusion_diagnostics import FusionDiagnostics\n' + source
    source = 'from fusion_imp_rta import initial_structure, save_structure, refresh_virtual\n' + source
    return source


def main():
    base = Path('/content/train_konly_rta_entry.py').read_text()
    boundary = 'exec(compile(source, '
    if base.count(boundary) != 1:
        raise RuntimeError('Unexpected K-only entry boundary')
    namespace = dict(__name__='__main__', __file__='/content/train_konly_rta_entry.py')
    exec(compile(base.split(boundary)[0], '/content/train_konly_rta_entry.py', 'exec'), namespace)
    source = build_source(namespace['source'])
    compiled = compile(source, '<fusion-rta>', 'exec')
    if os.environ.get('FUSION_BUILD_ONLY') == '1':
        print('FUSION_SOURCE_BUILD_COMPLETE; no training executed')
    else:
        exec(compiled, namespace)


if __name__ == '__main__':
    main()
