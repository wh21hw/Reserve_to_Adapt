"""Collect the declared coefficient ablation from logs only, not model eval."""
import json
from pathlib import Path
import zipfile

from collect_visda_capacity_colab import summarize_history


def main():
    control = Path('/content/imp-runs/visda-frozenbn-capacity-10e-v1/fixed2')
    candidate = Path('/content/imp-runs/visda-unknown-ce-off-10e-v1')
    destination = candidate / 'summary.json'
    archive = Path('/content/visda-unknown-ce-off-10e-v1-results.zip')
    if destination.exists() or archive.exists():
        raise FileExistsError('Preserve earlier collection')
    launches = [json.loads((p/'launch.json').read_text()) for p in (control, candidate)]
    base, new = launches
    fields = ('task', 'C', 'K', 'Q', 'seed', 'epochs', 'source_epochs', 'source_prior',
              'backbone', 'RTA_FREEZE_ENCODER_BN')
    if any(base[k] != new[k] for k in fields):
        raise ValueError('Control/candidate settings mismatch')
    if (new['C'], new['K'], new['Q'], new['seed'], new['epochs']) != (6, 2, 8, 1, 10):
        raise ValueError('Unexpected declared budget')
    if (new['unknown_ce_control'], new['unknown_ce_candidate']) != (1, 0):
        raise ValueError('Unexpected CE coefficient')
    expected = list(base['command'])
    expected[expected.index('/content/train_visda_frozenbn_rta_entry.py')] = '/content/train_visda_unknown_ce_off_entry.py'
    expected[expected.index('--log_dir')+1] = str(candidate)
    if expected != new['command']:
        raise ValueError('Command changes beyond declared entry/output')
    files, histories, summaries = [], [], {}
    for name, folder in [('control', control), ('unknown_ce_off', candidate)]:
        run = folder / 'visda-synthetic2real_seed1'
        history = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
        summaries[name] = summarize_history(history)
        histories.append(history)
        files.extend((folder / path, name + '/' + path) for path in (
            'launch.json', 'console.log', 'visda-synthetic2real_seed1/config.json',
            'visda-synthetic2real_seed1/history.jsonl', 'visda-synthetic2real_seed1/metrics.json',
            'visda-synthetic2real_seed1/protocol.json'))
    trajectory = []
    for a, b in zip(*histories):
        trajectory.append(dict(epoch=a['epoch'], candidate_minus_control_pp={
            label: 100*(b[key]-a[key]) for label, key in
            [('OS_star', 'OS_star'), ('UNK', 'unknown'), ('HOS', 'HOS')]}))
    report = dict(task=base['task'], C=6, K=2, Q=8, seed=1, epochs=10,
        changed_factor='Post-warmup unknown pseudo-label CE coefficient 1 -> 0',
        summaries=summaries, same_epoch_deltas=trajectory,
        final_delta=trajectory[-1],
        selection='All epochs/final; best uses target-label oracle HOS, not a search reward',
        caveat='Single-seed coefficient mechanism experiment; not final unknown modeling or full paper reproduction. No checkpoint re-evaluation.')
    if any(not path.is_file() for path, _ in files):
        raise FileNotFoundError('Missing ordinary result file')
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    with zipfile.ZipFile(str(archive), 'x', zipfile.ZIP_DEFLATED) as bundle:
        for path, name in files:
            bundle.write(str(path), name)
        bundle.write(str(destination), 'summary.json')
    print('UNKNOWN_CE_ABLATION_SUMMARY', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
