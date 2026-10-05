"""Download one official archive only; inspect before extraction/next stage."""
import json
from pathlib import Path
import shutil
import subprocess

root = Path('/content/osda-visda-syn2real-v1')
root.mkdir(exist_ok=True)
archive = root/'train.tar'
expected_bytes = 7698031104  # Official server HEAD, 2026-10-03; no hash gate.
url = 'http://csr.bu.edu/ftp/visda17/clf/train.tar'
cached = Path('/content/drive/MyDrive/OSDA/datasets/visda-syn2real-v1/train.tar')
if shutil.disk_usage(root).free < 2 * expected_bytes + 5 * 1024**3:
    raise RuntimeError('Insufficient free space for archive and extraction')
launch = root/'download-train-launch.json'
if not launch.exists():
    launch.write_text(json.dumps(dict(url=url, expected_bytes=expected_bytes,
        stage='Download only, no extraction/training',
        terms='Noncommercial research/education; do not redistribute images',
        source='Official VisionLearningGroup/taskcv-2017-public classification README'), indent=2))
if not archive.exists() and cached.is_file():
    if cached.stat().st_size != expected_bytes:
        raise ValueError('Existing Drive archive incomplete; do not silently download a duplicate')
    shutil.copyfile(cached, archive)
    print('VISDA_TRAIN_RESTORED_FROM_DRIVE', str(cached), flush=True)
if not archive.exists() or archive.stat().st_size != expected_bytes:
    command = ['curl', '--location', '--fail', '--retry', '3', '--connect-timeout', '20',
               '--silent', '--show-error', '--continue-at', '-', '--output', str(archive),
               '--write-out', '\nVISDA_TRAIN_TRANSFER HTTP=%{http_code} bytes=%{size_download} seconds=%{time_total}\n', url]
    process = subprocess.Popen(command)
    print('VISDA_TRAIN_DOWNLOAD_PID', process.pid, 'url', url, 'destination', str(archive), flush=True)
    if process.wait():
        raise RuntimeError('Download failed; retain partial file for transport resume')
if archive.stat().st_size != expected_bytes:
    raise RuntimeError('Archive size differs from advertised download; inspect before extracting')
print('VISDA_TRAIN_ARCHIVE_DOWNLOADED', archive.stat().st_size,
      'Next stage must inspect archive/list; dataset not ready yet', flush=True)
