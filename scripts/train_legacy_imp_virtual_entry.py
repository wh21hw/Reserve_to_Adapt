"""Run the user's historical virtual-prototype design with declared engineering edits."""
import os
from pathlib import Path
import sys

root = Path('/content/rta-legacy-l4-bridge-v1')
sys.path.insert(0, str(root))
snapshot = Path(os.environ.get('LEGACY_SNAPSHOT_DIR','/content/legacy-imp-virtual-v1'))
sys.path.insert(0, str(snapshot))
source = (snapshot/'main_user_snapshot.py').read_text()


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
imp_alpha = float(os.environ.get('LEGACY_IMP_ALPHA', '.05'))
source_epochs = int(os.environ.get('LEGACY_SOURCE_EPOCHS', '5'))
rta_epochs = int(os.environ.get('LEGACY_RTA_EPOCHS', '70'))
rta_warmiter = int(os.environ.get('LEGACY_RTA_WARMITER', '3'))
if rta_warmiter not in (1,3):raise ValueError('Declared warmup is2 or4 epochs')
if not 0 < imp_alpha < 1 or source_epochs < 1 or rta_epochs < 1:
    raise ValueError('Invalid historical tuning settings')
''')
replace_once('fine_tune_epochs = 5', 'fine_tune_epochs = source_epochs')
replace_once('imp = IMPClusterer(alpha = 0.05)', 'imp = IMPClusterer(alpha=imp_alpha)')
replace_once('while epoch < 70:', 'while epoch < rta_epochs:')
replace_once('warmiter = 3', 'warmiter = rta_warmiter')
replace_once('os.makedirs(args.log_dir, exist_ok=True)',
             'os.makedirs(args.log_dir, exist_ok=False)')
replace_once('model_path=os.path.join(BASE_DIR, "预训练model", "resnet50-19c8e357.pth")',
             'model_path=os.environ["RTA_MODEL_PATH"]')
replace_once("print(f'Log file: {log_file}\\n')", """print(f'Log file: {log_file}\\n')
with open(os.path.join(args.log_dir, 'config.json'), 'w') as stream:
    json.dump(dict(vars(args), seed=seed, source_pretrain_epochs=source_epochs, epochs=rta_epochs,
        alpha=imp_alpha, cluster_steps=5, source_centers_fixed=True,
        rta_warm_epochs=rta_warmiter+1,unknown_training_start_epoch=rta_warmiter+2,
        classifier_unknown_slots=2, design='Adaptive virtual prototypes, NOT adaptive output K',
        engineering_edits=['Paths, seed, no-overwrite, metrics/checkpoints, eval no_grad',
                           'Empty virtual collection uses shape(0,256); no forced births',
                           'If initial prototypes fewer than head rows, retain unused existing rows'],
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
empty_anchor = 'nomatch = np.stack(nomatch, axis=0)'
if source.count(empty_anchor) != 2:
    raise RuntimeError('Unexpected virtual-prototype stack anchors')
source = source.replace(empty_anchor,
    'nomatch = np.stack(nomatch, axis=0) if nomatch else np.empty((0, s_centroids.shape[1]), dtype=s_centroids.dtype)')
replace_once('param = torch.from_numpy(v.cpu().numpy()[t_match])', '''if len(t_match) < args.all_classes:
                remaining = [index for index in range(args.all_classes) if index not in t_match]
                t_match = np.concatenate([t_match, np.asarray(remaining, dtype=np.int64)])
            param = torch.from_numpy(v.cpu().numpy()[t_match])''')
compiled=compile(source, str(snapshot/'main_user_snapshot.py'), 'exec')
if os.environ.get('LEGACY_ENTRY_BUILD_ONLY')=='1':
    assert 'warmiter = rta_warmiter' in source and 'if epoch == warmiter:' in source
    print('LEGACY_WARMUP_INTERFACE_READY; loss activation and head initialization share warmiter')
else:
    exec(compiled,dict(__name__='__main__', __file__=str(snapshot/'main_user_snapshot.py')))
