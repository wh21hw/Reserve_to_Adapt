"""Local stdlib verification of downloaded archives and paired logs."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matched_pilot_audit import validate_pair

root = Path(__file__).resolve().parents[1] / 'pipeline-results'
arms = []
for arm in ('structure_off', 'structure_on'):
    with zipfile.ZipFile(root / f'matched-structure-pilot-v1-{arm}-results.zip') as archive:
        assert archive.testzip() is None
        read = lambda name: json.loads(archive.read(name))
        lines = lambda name: [json.loads(x) for x in archive.read(name).splitlines()]
        independent = read('independent-audit.json')
        summary = read('summary.json')
        assert independent['independent_checkpoint_evaluation_verified']
        assert independent['checkpoint_sha256'] == summary['checkpoint_sha256']
        assert independent['independent_final_metrics'] == summary['final']['metrics']
        arms.append((read('config.json'), lines('history.jsonl'), lines('batches.jsonl'), summary))
    checkpoint_archive = root / f'matched-structure-pilot-v1-{arm}-checkpoint.zip'
    if checkpoint_archive.exists():
        with zipfile.ZipFile(checkpoint_archive) as archive:
            assert archive.testzip() is None
            hasher = hashlib.sha256()
            with archive.open('last.pt') as stream:
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    hasher.update(chunk)
            assert hasher.hexdigest() == summary['checkpoint_sha256']
        print('CHECKPOINT_ARCHIVE_VERIFIED', arm)
    else:
        print('CHECKPOINT_ARCHIVE_NOT_DOWNLOADED', arm)
report = validate_pair(*arms)
print('LOCAL_PAIR_LOGS_AND_INDEPENDENT_EVALUATIONS_VERIFIED', report['shared_config_equal'], report['sample_batch_order_equal'])
