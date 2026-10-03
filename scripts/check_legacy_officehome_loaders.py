"""One real-data CPU check of the new legacy task transforms; no model/SGD."""
import ast
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/content')
os.environ['LEGACY_TASK_BUILD_ONLY'] = '1'
builder = dict(__name__='__build__', __file__='/content/train_legacy_task_entry.py')
exec(compile(Path('/content/train_legacy_task_entry.py').read_text(),
             '/content/train_legacy_task_entry.py', 'exec'), builder)
source = builder['source'].replace('pin_memory=True', 'pin_memory=False')
tree = ast.parse(source)
prefix = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'all_centroids' for t in node.targets):
        break
    prefix.append(node)
else:
    raise RuntimeError('Cannot locate the pre-model boundary')
sys.argv = ['main.py', '--task', 'officehome-pr2rw', '--shared_classes', '25',
            '--all_classes', '29', '--virtual-clusters', '29',
            '--source', '/content/osda-officehome-pr2rw-v1/product_0-24_train_all.txt',
            '--target', '/content/osda-officehome-pr2rw-v1/real_world_0-64_test.txt',
            '--data_dir', '/content/osda-officehome-pr2rw-v1',
            '--log_dir', '/content/imp-runs/legacy-officehome-loader-preflight-v1', '--name', 'seed3']
namespace = dict(__name__='__loader_check__', __file__='/content/rta-legacy-l4-bridge-v1/main.py')
exec(compile(ast.Module(body=prefix, type_ignores=[]), namespace['__file__'], 'exec'), namespace)
import torch
torch.set_num_threads(2)
s_images, s_labels = next(iter(namespace['source_train']))
t_images, sentinel = next(iter(namespace['target_train']))
e_images, e_labels = next(iter(namespace['target_test']))
assert s_images.shape == t_images.shape == (64, 3, 224, 224)
assert s_labels.shape == (64, 29) and (s_labels.argmax(1) < 25).all()
assert sentinel.shape == (64, 26) and (sentinel.argmax(1) == 25).all()
assert e_labels.shape[1] == 65
report = dict(task='officehome-pr2rw', seed=3, source_samples=len(namespace['ds']),
              target_samples=len(namespace['ds1']), source_label_dimensions=29,
              target_training_labels='constant sentinel25, no semantic labels',
              evaluation_dimensions=65, optimizer_steps=0, model_constructed=False,
              virtual_clusters=29,
              virtual_count_status='Declared C+baselineK choice, not confirmed author configuration',
              caveat='CPU transforms/list interface only; no clustering, GPU forward or full training')
(Path(namespace['args'].log_dir)/'loader-check.json').write_text(json.dumps(report, indent=2))
print('LEGACY_OFFICEHOME_LOADERS_PASS', json.dumps(report), flush=True)
sys.stdout = namespace['orig_stdout']
namespace['f'].close()
