"""Download the official real validation domain only; no extraction or training."""
import json
from pathlib import Path
import shutil
import subprocess

root = Path('/content/osda-visda-syn2real-v1')
archive = root/'validation.tar'
expected_bytes = 1023758336  # Official HEAD, 2026-10-03.
url = 'http://csr.bu.edu/ftp/visda17/clf/validation.tar'
cached = Path('/content/drive/MyDrive/OSDA/datasets/visda-syn2real-v1/validation.tar')
if not (root/'train-archive-inspection.json').is_file():
    raise RuntimeError('Inspect the previous download before advancing')
if shutil.disk_usage(root).free < 3 * expected_bytes:
    raise RuntimeError('Insufficient disk space')
launch = root/'download-validation-launch.json'
if not launch.exists():
    launch.write_text(json.dumps(dict(url=url, expected_bytes=expected_bytes,
        stage='Download only, raw target labels for evaluation/preparation, not training',
        terms='Noncommercial research/education; do not redistribute images'), indent=2))
if not archive.exists() and cached.is_file():
    if cached.stat().st_size != expected_bytes:
        raise ValueError('Existing Drive archive incomplete; preserve it instead of downloading a duplicate')
    shutil.copyfile(cached, archive)
    print('VISDA_VALIDATION_RESTORED_FROM_DRIVE', str(cached), flush=True)
if not archive.exists() or archive.stat().st_size != expected_bytes:
    command = ['curl', '--location', '--fail', '--retry', '3', '--connect-timeout', '20',
               '--silent', '--show-error', '--continue-at', '-', '--output', str(archive),
               '--write-out', '\nVISDA_VALIDATION_TRANSFER HTTP=%{http_code} bytes=%{size_download} seconds=%{time_total}\n', url]
    process = subprocess.Popen(command)
    print('VISDA_VALIDATION_DOWNLOAD_PID', process.pid, flush=True)
    if process.wait():
        raise RuntimeError('Download failed; retain partial file for transport resume')
if archive.stat().st_size != expected_bytes:
    raise RuntimeError('Unexpected downloaded size; inspect before extracting')
print('VISDA_VALIDATION_ARCHIVE_DOWNLOADED', archive.stat().st_size,
      'Not extracted; no model/training', flush=True)
