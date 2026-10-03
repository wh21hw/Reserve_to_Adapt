"""Require all three independently audited L4 seeds; fixed final is primary."""
import argparse
import json
import math
from pathlib import Path
import statistics
import zipfile

def summarize(paths):
    reports, launches = [], []
    for seed, path in enumerate(paths, 1):
        with zipfile.ZipFile(path) as archive:
            report = json.loads(archive.read('audit-summary.json'))
            launch = json.loads(archive.read('launch.json'))
            history = [json.loads(line) for line in archive.read('history.jsonl').splitlines()]
        assert report['seed'] == launch['seed'] == seed
        assert report['epochs_verified'] == launch['epochs'] == 70
        assert [row['epoch'] for row in history] == list(range(1, 71))
        assert all(row['seed'] == seed for row in history)
        assert report['fixed_final'] == history[-1]
        assert report['oracle_best'] == history[max(range(70), key=lambda index: history[index]['HOS'])]
        assert report['gpu'] == 'NVIDIA L4'
        assert report['target_labels_used_for_evaluation_only']
        assert report['optimizer_steps_executed_by_collector'] == 0
        for key, independent in [('OS_star', 'OS_star'), ('unknown', 'UNK'), ('HOS', 'HOS')]:
            assert abs(report['fixed_final'][key] - report['independent_fixed_final_metrics'][independent]) < 1e-12
        reports.append(report)
        launches.append(launch)
    assert len(reports) == 3 and all(launch['code_sha256'] == launches[0]['code_sha256'] for launch in launches)
    result = dict(task='office31-a2w', seeds=[1, 2, 3], epochs_each=70, gpu='NVIDIA L4',
                  protocol=reports[0]['loss_protocol'], primary_selection='fixed_final',
                  oracle_best_uses_target_labels=True, sample_standard_deviation_ddof=1,
                  paper_reference_percent={'OS_star': 92.2, 'unknown': 93.8, 'HOS': 93.0}, selections={})
    for selection in ('fixed_final', 'oracle_best'):
        table = {}
        for key in ('OS_star', 'unknown', 'HOS'):
            values = [report[selection][key] * 100 for report in reports]
            assert all(math.isfinite(value) for value in values)
            table[key] = dict(per_seed_percent=values, mean_percent=statistics.mean(values),
                              sample_std_percent=statistics.stdev(values),
                              difference_from_paper_pp=statistics.mean(values) - result['paper_reference_percent'][key])
        result['selections'][selection] = table
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archives', nargs=3, type=Path, help='Seed1, seed2, seed3 results archives in order')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.archives)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
