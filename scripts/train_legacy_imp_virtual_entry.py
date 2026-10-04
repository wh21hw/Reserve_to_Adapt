"""Run the user's historical virtual-prototype design with declared engineering edits."""
import os
from pathlib import Path
import sys

root = Path('/content/rta-legacy-l4-bridge-v1')
sys.path.insert(0, str(root))
sys.path.insert(0, '/content/legacy-imp-virtual-v1')
source = Path('/content/legacy-imp-virtual-v1/main_user_snapshot.py').read_text()


def replace_once(old, new):
    global source
    if source.count(old) != 1:
        raise RuntimeError('Unexpected historical anchor: '+old[:70])
    source = source.replace(old, new)


replace_once('from IMPClusterer import IMPClusterer',
             'from IMPClusterer_user_snapshot import IMPClusterer')
replace_once('args = get_args()', '''args = get_args()
import random, json, time
seed = int(os.environ.get('RTA_SEED', '3'))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
started = time.time()
''')
replace_once('os.makedirs(args.log_dir, exist_ok=True)',
             'os.makedirs(args.log_dir, exist_ok=False)')
replace_once('model_path=os.path.join(BASE_DIR, "预训练model", "resnet50-19c8e357.pth")',
             'model_path=os.environ["RTA_MODEL_PATH"]')
replace_once("print(f'Log file: {log_file}\\n')", """print(f'Log file: {log_file}\\n')
with open(os.path.join(args.log_dir, 'config.json'), 'w') as stream:
    json.dump(dict(vars(args), seed=seed, source_pretrain_epochs=5, epochs=70,
        alpha=.05, cluster_steps=5, source_centers_fixed=True,
        classifier_unknown_slots=2, design='Adaptive virtual prototypes, NOT adaptive output K',
        engineering_edits=['Paths, seed, no-overwrite, metrics/checkpoints, eval no_grad'],
        dependency_root='/content/rta-legacy-l4-bridge-v1',
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name()), stream, indent=2)
""")
replace_once("    with TrainingModeManager([feature_extractor, cls], train=False) as mgr, \\\n", "    with torch.no_grad(), TrainingModeManager([feature_extractor, cls], train=False) as mgr, \\\n")
replace_once('    epoch += 1', '''    record = dict(epoch=epoch+1, seed=seed, OS=acc_os,
        OS_star=acc_os_star, unknown=unkn, HOS=hos, Q_imp=K_cluster,
        virtual_prototypes=int(nomatch.size(0)), classifier_unknown_slots=2,
        elapsed_seconds=time.time()-started,
        best=dict(epoch=best_epoch, OS=best_os, OS_star=best_os_star,
                  unknown=best_unk, HOS=best_hos))
    with open(os.path.join(args.log_dir, 'history.jsonl'), 'a') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\\n')
    with open(os.path.join(args.log_dir, 'metrics.json'), 'w') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    checkpoint = dict(model=net.state_dict(), epoch=epoch+1, seed=seed, metrics=record)
    torch.save(checkpoint, os.path.join(args.log_dir, 'last.pt'))
    if best_epoch == epoch:
        torch.save(checkpoint, os.path.join(args.log_dir, 'best.pt'))
    epoch += 1''')
replace_once('    files.download(log_file)', '    pass # ordinary CLI collection, no browser download')
exec(compile(source, '/content/legacy-imp-virtual-v1/main_user_snapshot.py', 'exec'),
     dict(__name__='__main__', __file__='/content/legacy-imp-virtual-v1/main_user_snapshot.py'))
