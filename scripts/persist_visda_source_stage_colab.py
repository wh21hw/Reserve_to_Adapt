"""Persist a completed source prior before starting expensive later stages.

No hash audit, checkpoint evaluation or automatic training retry. Destination
must be a mounted Google Drive folder; existing outputs are never overwritten.
This preserves the source stage, not an optimizer-resumable RTA run.
"""
import argparse
import json
from pathlib import Path
import shutil


def persist(source_root, data_root, destination):
    source_root, data_root, destination = map(Path, (source_root, data_root, destination))
    drive = Path('/content/drive/MyDrive')
    if not drive.is_dir() or drive.resolve() not in destination.resolve().parents:
        raise ValueError('Destination must be inside an already mounted /content/drive/MyDrive')
    summary = json.loads((source_root/'source/summary.json').read_text())
    if not summary.get('complete'):
        raise ValueError('Only persist a completed source stage')
    files = [(source_root/relative, Path(relative)) for relative in (
        'source-launch.json', 'source-console.log', 'source/config.json',
        'source/history.jsonl', 'source/summary.json', 'source/source-final.pt',
        'source/features.npz')]
    files += [(data_root/name, Path('protocol')/name) for name in (
        'data-preparation.json', 'source-known-6.txt',
        'target-unlabeled-paths.txt', 'target-real-12.txt')]
    missing = [str(source) for source, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError('Required recovery material missing: '+', '.join(missing))
    destination.mkdir(parents=True, exist_ok=False)
    # A completion marker is written last. An interrupted copy is not a complete backup.
    for source, relative in files:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print('PERSISTED', relative, flush=True)
    manifest = dict(complete=True, source_root=str(source_root),
        files=[str(relative) for _, relative in files],
        target_labels_used_for_estimation=False,
        caveat='Source checkpoint and features only; not an RTA optimizer-resume checkpoint. Images and environment must be restored separately.')
    (destination/'recovery-complete.json').write_text(json.dumps(manifest, indent=2))
    print('VISDA_SOURCE_RECOVERY_SAVED', destination, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default='/content/imp-runs/visda-frozenbn-capacity-10e-v1')
    parser.add_argument('--data-root', default='/content/osda-visda-syn2real-v1')
    parser.add_argument('--destination', required=True, help='New folder inside mounted MyDrive')
    args = parser.parse_args()
    persist(args.source_root, args.data_root, args.destination)
