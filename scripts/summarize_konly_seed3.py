"""Compare completed selected-seed arms without checkpoint reevaluation."""
import argparse
import json
from pathlib import Path

METRICS = ('OS_star', 'unknown', 'HOS')


def load_completed(path):
    row = json.loads(Path(path).read_text(encoding='utf-8'))
    if row['epoch'] != 70 or row['seed'] != 3:
        raise ValueError('Expected completed 70-epoch selected seed3')
    return dict(final={key: 100 * row[key] for key in METRICS},
                best={key: 100 * row['best'][key] for key in METRICS},
                best_completed_epoch=row['best']['epoch'] + 1,
                elapsed_seconds=row['elapsed_seconds'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--estimated', required=True)
    parser.add_argument('--estimate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    estimate = json.loads(Path(args.estimate).read_text(encoding='utf-8'))
    if estimate['target_labels_used'] or estimate['seed'] != 3:
        raise ValueError('Do not use a target-label-tuned K estimate')
    arms = {name: load_completed(path) for name, path in
            [('imagenet_baseline_K2', args.baseline),
             ('source_prior_fixed_K2', args.control),
             ('source_prior_estimated_K', args.estimated)]}
    differences = {}
    for selection in ('best', 'final'):
        differences[selection] = {key:
            arms['source_prior_estimated_K'][selection][key]
            - arms['source_prior_fixed_K2'][selection][key] for key in METRICS}
    report = dict(task='Office31 A->W', seed=3, epochs=70, estimated_K=estimate['K'],
                  fixed_K=2, arms_percent=arms, estimated_minus_fixed_pp=differences,
                  seed_selection='Posthoc seed selection based on prior baseline results',
                  best_selection='Epoch selected with target-label HOS; oracle reporting',
                  final_selection='Fixed complete 70-epoch budget',
                  attribution='Pure K effect compares the two common-source-prior arms, not ImageNet baseline',
                  paper_percent=dict(OS_star=92.2, unknown=93.8, HOS=93.0),
                  K_interpretation='Inferred capacity, not proof of true semantic unknown count')
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
