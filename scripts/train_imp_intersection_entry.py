"""Independent version: filter only ce_ep candidates by frozen initial IMP."""
import os
from pathlib import Path

entry=Path('/content/train_legacy_task_entry.py').read_text()
anchor="compiled = compile(source, str(root/'main.py'), 'exec')"
if entry.count(anchor)!=1:
    raise RuntimeError('Unexpected task entry compile anchor')
# Build the injected string explicitly so nested source literals stay readable.
dataset_code='''imp_assignment = np.load(os.environ['IMP_INITIAL_ASSIGNMENTS'])['assignments']
if args.task != 'officehome-pr2rw' or args.shared_classes != 25 or args.all_classes != 26:
    raise ValueError('Only declared OfficeHome K1 intersection arm is allowed')
if len(imp_assignment) != len(images):
    raise ValueError('IMP target row count mismatch')
if os.path.abspath(args.target) != '/content/osda-officehome-pr2rw-v1/real_world_0-64_test.txt':
    raise ValueError('IMP mapping requires the original ordered target list')
class IMPIndexedDataset(CustomDataset):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, (label, bool(imp_assignment[index] >= args.shared_classes))
ds1 = IMPIndexedDataset(images,labels,img_transformer=transform,is_train=True)
print('IMP_INTERSECTION_START', dict(K=1, geometry_unknown=int((imp_assignment>=25).sum()),
    noise=int((imp_assignment<0).sum()), policy='Filter ce_ep only'), flush=True)
'''
gate_code='''            original_candidates = int(r.numel())
            r = r.view(-1)
            geometry_keep = imp_geometry_unknown.to(device=ft1.device, dtype=torch.bool)
            r = r[geometry_keep[r]]
            with open(os.path.join(args.log_dir, 'intersection-history.jsonl'), 'a') as gate_log:
                gate_log.write(json.dumps(dict(epoch=int(epoch),batch=int(i),
                    before=original_candidates,after=int(r.numel())))+chr(10))
            feature_otherep = torch.index_select(ft1, 0, r.view(-1))'''
patch_code="replace_once('ds1 = CustomDataset(images,labels,img_transformer=transform,is_train=True)', "+repr(dataset_code)+")\n"
old='(im_target, label_target)) in enumerate(customgenearator):'
patch_code+="if source.count("+repr(old)+") != 2: raise RuntimeError('Expected two target loops')\n"
patch_code+="source=source.replace("+repr(old)+", '(im_target, (label_target, imp_geometry_unknown))) in enumerate(customgenearator):')\n"
patch_code+="replace_once('            feature_otherep = torch.index_select(ft1, 0, r.view(-1))', "+repr(gate_code)+")\n"
entry=entry.replace(anchor,patch_code+anchor)
exec(compile(entry,'/content/train_legacy_task_entry.py','exec'),dict(__name__='__main__'))
