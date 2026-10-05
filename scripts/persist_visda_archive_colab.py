"""Persist one completed official archive to Drive, no hashes or image re-audit."""
import argparse
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('domain', choices=['train', 'validation'])
args = parser.parse_args()
drive = Path('/content/drive/MyDrive')
if not drive.is_dir():
    raise RuntimeError('Mount Drive before persisting the dataset')
root = Path('/content/osda-visda-syn2real-v1')
expected = {'train': 7698031104, 'validation': 1023758336}[args.domain]
source = root/(args.domain+'.tar')
if not source.is_file() or source.stat().st_size != expected:
    raise ValueError('Download not complete; do not persist a partial archive')
destination = drive/'OSDA/datasets/visda-syn2real-v1'
destination.mkdir(parents=True, exist_ok=True)
final = destination/source.name
temporary = destination/(source.name+'.partial')
if final.exists():
    if final.stat().st_size != expected:
        raise FileExistsError('Existing archive differs in size; preserve it')
    print('VISDA_ARCHIVE_ALREADY_PERSISTED', str(final), flush=True)
else:
    if temporary.exists():
        raise FileExistsError('Previous partial copy exists; preserve evidence before retry')
    print('VISDA_ARCHIVE_COPY_START', str(source), 'to', str(temporary), flush=True)
    with source.open('rb') as reader, temporary.open('xb') as writer:
        shutil.copyfileobj(reader, writer, 8*1024**2)
    if temporary.stat().st_size != expected:
        raise RuntimeError('Drive copy incomplete')
    temporary.rename(final)
marker = destination/(args.domain+'-saved.json')
if not marker.exists():
    with marker.open('x') as stream:
        json.dump(dict(complete=True, archive=final.name, bytes=expected,
            source_url='http://csr.bu.edu/ftp/visda17/clf/'+final.name,
            terms='Noncommercial research/education; no redistribution'), stream, indent=2)
print('VISDA_ARCHIVE_PERSISTED', str(final), flush=True)
