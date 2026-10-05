"""Summarize completed declared VisDA arms from logs, never reevaluate models."""
import json
import math
from pathlib import Path
import zipfile


def summarize_history(history):
    if [row['epoch'] for row in history] != list(range(1, 11)):
        raise ValueError('Expected exactly ten complete epochs')
    for row in history:
        for key in ('OS_star', 'unknown', 'HOS'):
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError('Invalid fractional metric: '+key)
    def metrics(row):
        return dict(epoch=row['epoch'], OS_star=row['OS_star'], UNK=row['unknown'], HOS=row['HOS'])
    return dict(best=metrics(max(history, key=lambda row: row['HOS'])), final=metrics(history[-1]))


def main():
    root = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1')
    destination = root/'summary.json'
    archive = Path('/content/visda-frozenbn-capacity-10e-v1-results.zip')
    if destination.exists() or archive.exists():
        raise FileExistsError('Preserve previous collection')
    capacity = json.loads((root/'capacity.json').read_text())
    source = json.loads((root/'source/summary.json').read_text())
    if (not capacity['converged'] or capacity['target_labels_used']
            or capacity['C'] != 6 or not isinstance(capacity['K'], int) or capacity['K'] < 0):
        raise ValueError('Expected converged nonnegative capacity without target labels')
    if not source['complete'] or source['epochs'] != 3:
        raise ValueError('Incomplete shared source stage')
    k = capacity['K']
    report = dict(task='VisDA Synthetic->Real', seed=1, source_epochs=3, RTA_epochs=10,
        backbone='ResNet50', C=6, Q=8, estimated_K=k, arms={},
        selection='Target-label oracle HOS-best epoch; final epoch10. Single predeclared seed, not a three-seed mean.',
        caveat='Exploratory release-code K-only comparison conditional on common source3/frozen encoder BN. Paper backbone/budget unresolved; Q held8 instead of coupling to inferred K.',
        capacity_semantics='Modeling capacity, not certified semantic unknown class count')
    files = [root/'capacity.json', root/'source-launch.json', root/'source-console.log',
             root/'source/config.json', root/'source/history.jsonl', root/'source/summary.json']
    names = [('fixed2', 2)] + ([('estimated', k)] if k > 0 and k != 2 else [])
    launches = []
    for name, expected_k in names:
        directory = root/name
        training = directory/'visda-synthetic2real_seed1'
        launch = json.loads((directory/'launch.json').read_text())
        if (launch['K'] != expected_k or launch['C'] != 6 or launch['Q'] != 8
                or launch['seed'] != 1 or launch['epochs'] != 10
                or launch['backbone'] != 'ResNet50' or not launch['RTA_FREEZE_ENCODER_BN']
                or launch['target_labels_for_structure']):
            raise ValueError('Unexpected declared arm: '+name)
        launches.append(launch)
        history = [json.loads(line) for line in (training/'history.jsonl').read_text().splitlines()]
        report['arms'][name] = dict(K=expected_k, **summarize_history(history))
        files.extend([directory/'launch.json', directory/'console.log', training/'history.jsonl',
                      training/'metrics.json', training/'config.json', training/'protocol.json'])
    if any(launch['source_prior'] != launches[0]['source_prior'] for launch in launches):
        raise ValueError('Different source prior paths')
    if k == 2:
        report['estimated_arm_status'] = 'Same K input as fixed2; reuse one equivalent run, not a second observation'
    elif k == 0:
        report['estimated_arm_status'] = 'Not run: no reserved unknown capacity; never force K1. Estimator remains unsuccessful for this adaptation workflow.'
    else:
        report['estimated_arm_status'] = 'Complete independent declared arm'
        report['estimated_minus_fixed_pp'] = {
            which: {key: 100*(report['arms']['estimated'][which][key]-report['arms']['fixed2'][which][key])
                for key in ('OS_star', 'UNK', 'HOS')} for which in ('best', 'final')}
    # Resolve ordinary files before creating an output archive; no hashes/checkpoints.
    if any(not file.is_file() for file in files):
        raise FileNotFoundError('Missing ordinary result artifact')
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        for file in files+[destination]:
            output.write(file, str(file.relative_to(root)))
    print('VISDA_CAPACITY_RTA_SUMMARY', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
