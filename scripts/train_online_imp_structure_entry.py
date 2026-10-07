"""Patch the task wrapper, leaving all user working-tree training files alone."""
import os
from pathlib import Path


def patch(source, epochs):
    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise RuntimeError('Unexpected online IMP entry anchor: '+old[:90])
        source = source.replace(old, new)
    replace('from domain_bus import DomainBus', 'from cluster_identity_bridge import IndexedDomainBus as DomainBus')
    replace('ds1 = CustomDataset(images,labels,img_transformer=transform,is_train=True)',
        'from cluster_identity_bridge import IndexedDataset\nds1 = IndexedDataset(CustomDataset(images,labels,img_transformer=transform,is_train=True))')
    # Source labels are constructed in workers with a fixed C width; pad in main
    # after each dynamic resize, rather than passing stale args into workers.
    replace('label = one_hot(args.all_classes, protocol.source_label(label))',
        'label = one_hot(args.shared_classes, protocol.source_label(label))')
    start = source.index('customgenearator = DomainBus([source_train, target_train])')
    end = source.index('del(ProbRecorder)', start)+len('del(ProbRecorder)')
    source = source[:start]+'nomatch = torch.empty(0,256,device="cuda")\n'+source[end:]
    loop = 'while epoch <'+str(epochs)+':'
    replace(loop, '''from online_imp_structure import OnlineStructure
online_structure = OnlineStructure(args, os.environ['ONLINE_STRUCTURE_LABELS']=='1')
'''+loop+'''
    nomatch = online_structure.refresh(net, cls, optimizer_cls, epoch,
        reset_correspondence=(epoch == warmiter+1))''')
    replace('(im_target, label_target)) in enumerate(customgenearator):',
        '(im_target, label_target, target_indices)) in enumerate(customgenearator):')
    replace('            label_source = label_source.cuda()',
        '            label_source = torch.nn.functional.pad(label_source.cuda(), (0,args.all_classes-args.shared_classes))')
    replace('                pseudo_index=pseudo_index + args.shared_classes',
        '''                pseudo_index=pseudo_index + args.shared_classes
                pseudo_index = online_structure.labels(
                    target_indices[r.view(-1).detach().cpu()].numpy(), pseudo_index)''')
    # Keep source centroids for original warm-end head initialization; no extra
    # Q-means virtual fit because IMP supplies the virtual directions each round.
    start = source.index('    faiss_kmeans = faiss.Kmeans(256, int(K_cluster)')
    end = source.index('    if epoch ==warmiter:', start)
    source = source[:start]+'    online_structure.finish_epoch(epoch)\n\n'+source[end:]
    replace('epoch=epoch, HOS=float(hos))', 'epoch=epoch, HOS=float(hos), K=args.all_classes-args.shared_classes)')
    replace('elapsed_seconds=time.time()-started_at, seed=seed)',
        'elapsed_seconds=time.time()-started_at, seed=seed, K=args.all_classes-args.shared_classes, V=len(nomatch), target_slot_counts=np.bincount(y_pred, minlength=args.all_classes).tolist())')
    return source


def build_training_source():
    original = Path('/content/train_legacy_task_entry.py')
    entry = original.read_text()
    # Local frozen baseline already has an env-defined loop, not the original
    # author's hard-coded70. The stable task adaptation is otherwise identical.
    entry = entry.replace("replace_once('while epoch <70:', 'while epoch <'+str(epochs)+':')",
        "replace_once('while epoch < run_epochs:', 'while epoch <'+str(epochs)+':')")
    boundary = "compiled = compile(source, str(root/'main.py'), 'exec')"
    if entry.count(boundary) != 1:
        raise RuntimeError('Unexpected stable task wrapper boundary')
    namespace = dict(__name__='online_builder', __file__=str(original))
    exec(compile(entry.split(boundary)[0], str(original), 'exec'), namespace)
    source = patch(namespace['source'], int(os.environ['RTA_EPOCHS']))
    return source, namespace


if __name__ == '__main__':
    source, namespace = build_training_source()
    compiled = compile(source, '<online-imp-rta>', 'exec')
    if os.environ.get('LEGACY_TASK_BUILD_ONLY') == '1':
        print('ONLINE_IMP_BUILD_COMPLETE; no training executed')
    else:
        exec(compiled, dict(__name__='__main__', __file__=str(namespace['root']/'main.py')))
