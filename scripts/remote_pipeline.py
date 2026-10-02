"""Execute on Colab with MurphyLo colab exec -f; environment variables configure paths."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def run(argv, **kwargs):
    print('+', ' '.join(map(str, argv)), flush=True)
    subprocess.run(argv, check=True, **kwargs)


def main():
    drive_root = Path(os.environ.get('OSDA_DRIVE_ROOT', '/content/drive/MyDrive/OSDA'))
    repo_url = os.environ.get('OSDA_REPO_URL', 'https://github.com/wh21hw/Reserve_to_Adapt.git')
    revision = os.environ.get('OSDA_REVISION', 'codex/colab-pipeline')
    workspace = Path('/content/osda-pipeline')
    run_dir = drive_root / 'runs' / os.environ.get('OSDA_RUN_NAME', 'a2w-pipeline-seed1')
    dataset_dir = Path('/content/osda-datasets')
    for folder in (dataset_dir, run_dir):
        folder.mkdir(parents=True, exist_ok=True)
    if not workspace.exists():
        run(['git', 'clone', '--depth', '1', '--branch', revision, repo_url, str(workspace)])
    actual_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip()
    run(['git', 'diff', '--exit-code'], cwd=workspace)
    run([sys.executable, '-m', 'pip', 'install', '-r', str(workspace / 'requirements-colab.txt')])
    manifest_path = drive_root / 'datasets' / 'data-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    local_files = {}
    for item in manifest:
        source = drive_root / ('pretrained' if item['file'].endswith('.pth') else 'datasets') / item['file']
        local = dataset_dir / item['file']
        shutil.copy2(source, local)
        digest = hashlib.sha256()
        with local.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != item['sha256'] or local.stat().st_size != item['bytes']:
            raise RuntimeError(f'Data integrity failure: {local}')
        local_files[item['file']] = local
    with tarfile.open(local_files['office31_images.tar']) as archive:
        # Locally generated archive; reject paths escaping the dataset directory.
        for member in archive.getmembers():
            destination = (dataset_dir / member.name).resolve()
            if not destination.is_relative_to(dataset_dir.resolve()) or member.issym() or member.islnk():
                raise RuntimeError(f'Unsafe archive entry: {member.name}')
        archive.extractall(dataset_dir)
    import torch
    import torchvision
    environment = dict(commit=actual_revision, python=sys.version, torch=torch.__version__,
                       torchvision=torchvision.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                       dataset_manifest=manifest)
    (run_dir / 'environment.json').write_text(json.dumps(environment, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError('Pipeline GPU test requires a GPU runtime')
    # Validate every image referenced by the original A->W lists before training.
    for name in ('amazon_0-9_train_all.txt', 'webcam_0-9_20-30_test.txt'):
        for row in (workspace / 'data' / name).read_text().splitlines():
            image, label = row.rsplit(' ', 1)
            if not (dataset_dir / image).is_file():
                raise FileNotFoundError(dataset_dir / image)
    command = [sys.executable, '-u', 'main.py', '--data_dir', str(dataset_dir),
               '--model_path', str(local_files['resnet50-19c8e357.pth']),
               '--log_dir', str(run_dir) + '/', '--name', 'smoke', '--epochs', '5',
               '--max_batches', '5', '--batch_size', '32', '--seed', '1',
               '--cluster_method', os.environ.get('OSDA_CLUSTER_METHOD', 'kmeans')]
    with (run_dir / 'console.log').open('w') as log:
        process = subprocess.Popen(command, cwd=workspace, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError(f'Training failed; inspect {run_dir / "console.log"}')
    outputs = list(run_dir.glob('*/metrics.json'))
    if len(outputs) != 1:
        raise RuntimeError('Expected exactly one metrics.json after training')
    metrics = json.loads(outputs[0].read_text())
    if metrics['epoch'] != 5 or metrics['steps'] != 25:
        raise RuntimeError(f'Unexpected training completion: {metrics}')
    checkpoint_path = outputs[0].with_name('last.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != 5:
        raise RuntimeError('Checkpoint verification failed')
    print('PIPELINE_SUCCESS', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
