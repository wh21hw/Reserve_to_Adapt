"""Stdlib log audit; does not replace independent checkpoint evaluation."""
import math
import statistics

LOSS_KEYS = ('total', 'structure', 'source_ce', 'virtual', 'entropy', 'unknown_ce', 'adversarial')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_logs(config, history, batches, summary):
    arm = config['arm']
    require(arm in ('structure_off', 'structure_on'), 'Unknown arm')
    coefficient = 0. if arm == 'structure_off' else .1
    require(config['coefficient'] == coefficient, 'Changed structure coefficient')
    require(config['warm_epoch'] == 4 and config['final_epoch'] == 6 and config['seed'] == 1, 'Changed pilot budget/seed')
    require(config['batch_size'] == 64 and config['support_min'] == 5, 'Changed batch/support rule')
    require(config['target_labels_used_for_training'] is False, 'Training label policy mismatch')
    count = config['capacity']
    require(count > 0 and len(config['selected_indices']) == count and len(set(config['selected_indices'])) == count, 'Invalid candidate mapping')
    require(len(config['prior']) == count and all(math.isfinite(x) for x in config['prior']), 'Invalid prior')
    require(math.isclose(sum(math.exp(x) for x in config['prior']), 1., abs_tol=1e-5), 'Unnormalized prior')
    require(config['variance'] >= 1e-8 and math.isfinite(config['variance']), 'Invalid variance floor')
    require([row['epoch'] for row in history] == [5, 6], 'Missing or duplicate epochs')
    require([(row['epoch'], row['batch']) for row in batches] == [(e, b) for e in (5, 6) for b in range(1, 15)], 'Missing or duplicate optimizer batches')
    for row in batches:
        require(all(math.isfinite(row[key]) and row[key] >= -1e-6 for key in LOSS_KEYS), 'Nonfinite/negative loss')
        expected = row['source_ce'] + .01 * row['virtual'] + .3 * row['adversarial'] + row['entropy'] + row['unknown_ce'] + coefficient * row['structure']
        require(math.isclose(row['total'], expected, rel_tol=1e-5, abs_tol=1e-5), 'Loss formula mismatch')
        require(0 <= row['unknown_selected'] <= 16, 'Unexpected early unknown selection')
    for row in history:
        require(row['arm'] == arm and row['coefficient'] == coefficient and row['batch_count'] == 14, 'Epoch identity mismatch')
        steps = row['epoch'] * 14
        require(row['optimizer_steps'] == [steps] * 3 and row['grl_steps'] == steps * 2, 'Scheduler/GRL counter mismatch')
        digest = row['batch_order_sha256']
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'Invalid batch order digest')
        metrics = row['metrics']
        require(all(math.isfinite(metrics[key]) and 0 <= metrics[key] <= 1 for key in ('OS_star', 'UNK', 'HOS')), 'Invalid metrics')
        expected_hos = 2 * metrics['OS_star'] * metrics['UNK'] / max(metrics['OS_star'] + metrics['UNK'], 1e-12)
        require(math.isclose(metrics['HOS'], expected_hos, abs_tol=1e-10), 'HOS formula mismatch')
        for key in LOSS_KEYS:
            expected_mean = statistics.mean(b[key] for b in batches if b['epoch'] == row['epoch'])
            require(math.isclose(row['loss_means'][key], expected_mean, abs_tol=1e-6, rel_tol=1e-6), 'Batch loss mean mismatch')
    require(summary['arm'] == arm and summary['config'] == config and summary['final'] == history[-1], 'Summary mismatch')
    require(summary['complete_epochs'] == [5, 6] and summary['actual_new_optimizer_steps'] == 28, 'Pilot update count mismatch')
    require(summary['target_label_checkpoint_selection'] is False and summary['complete_budget'] is False, 'Invalid pilot completion claim')
    return dict(arm=arm, log_epochs_verified=[5, 6], log_update_count_verified=28,
                final_metrics=history[-1]['metrics'], independent_checkpoint_evaluation_verified=False,
                mixture_converged_each=[row['mixture_converged'] for row in history])

def validate_pair(off, on):
    # Each tuple is config, history, batches, summary; order is explicit.
    require(off[0]['arm'] == 'structure_off' and on[0]['arm'] == 'structure_on', 'Arm pair order mismatch')
    audits = [validate_logs(*arm) for arm in (off, on)]
    shared = lambda config: {key: value for key, value in config.items() if key not in ('arm', 'coefficient')}
    require(shared(off[0]) == shared(on[0]), 'Changed shared configuration/input/code')
    require([r['batch_order_sha256'] for r in off[1]] == [r['batch_order_sha256'] for r in on[1]], 'Unmatched sample batch order')
    return dict(arms=audits, shared_config_equal=True, sample_batch_order_equal=True,
                augmented_image_equality_verified=False, complete_budget=False,
                checkpoint_evaluation_still_required=True)
