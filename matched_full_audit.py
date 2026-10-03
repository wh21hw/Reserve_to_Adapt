"""Complete-budget log validation, separate from checkpoint evaluation."""
import math
import statistics
from matched_pilot_audit import require, LOSS_KEYS

def validate_full_logs(config, history, batches, summary):
    arm = config['arm']
    require(arm in ('structure_off', 'structure_on'), 'Invalid arm')
    coefficient = 0. if arm == 'structure_off' else .1
    require(config['coefficient'] == coefficient, 'Changed coefficient')
    require(config['seed'] == 1 and config['warm_epoch'] == 4 and config['final_epoch'] == 70, 'Changed budget/seed')
    require(config['batch_size'] == 64 and config['support_min'] == 5, 'Changed batch/support')
    require(config['target_labels_used_for_training'] is False, 'Target training labels')
    require(config['evaluation_selection'] == 'fixed_final; all epoch metrics diagnostic only', 'Changed selection')
    require(config['unknown_selection_policy'] == '0-based epoch<=10: probability>.8 known, nonknown top16; later: min/max mixture groups', 'Changed selection schedule')
    require(config['mixture_components_policy'] == '4 through 0-based epoch30, then 2', 'Changed mixture schedule')
    count = config['capacity']
    require(count > 0 and len(config['selected_indices']) == len(set(config['selected_indices'])) == count, 'Candidate map')
    require(len(config['prior']) == count and all(math.isfinite(x) for x in config['prior']), 'Prior finite')
    require(math.isclose(sum(math.exp(x) for x in config['prior']), 1., abs_tol=1e-5), 'Prior normalized')
    require(math.isfinite(config['variance']) and config['variance'] >= 1e-8, 'Variance floor')
    epochs = list(range(5, 71))
    require([row['epoch'] for row in history] == epochs, 'Incomplete adaptation history')
    require([(r['epoch'], r['batch']) for r in batches] == [(e, b) for e in epochs for b in range(1, 15)], 'Incomplete updates')
    for row in batches:
        require(all(math.isfinite(row[k]) and row[k] >= -1e-6 for k in LOSS_KEYS), 'Nonfinite/negative loss')
        expected = row['source_ce'] + .01*row['virtual'] + .3*row['adversarial'] + row['entropy'] + row['unknown_ce'] + coefficient*row['structure']
        require(math.isclose(row['total'], expected, rel_tol=1e-5, abs_tol=1e-5), 'Loss formula')
        require(0 <= row['unknown_selected'] <= (16 if row['epoch'] <= 11 else 64), 'Selection capacity')
    for row in history:
        require(row['arm'] == arm and row['coefficient'] == coefficient and row['batch_count'] == 14, 'Epoch identity')
        require(row['optimizer_steps'] == [14*row['epoch']]*3 and row['grl_steps'] == 28*row['epoch'], 'Counter')
        digest = row['batch_order_sha256']
        require(len(digest) == 64 and all(x in '0123456789abcdef' for x in digest), 'Sample order hash')
        metrics = row['metrics']
        require(all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1 for k in ('OS_star', 'UNK', 'HOS')), 'Metrics finite')
        hos = 2*metrics['OS_star']*metrics['UNK']/max(metrics['OS_star']+metrics['UNK'], 1e-12)
        require(math.isclose(hos, metrics['HOS'], abs_tol=1e-10), 'HOS formula')
        epoch_batches = [b for b in batches if b['epoch'] == row['epoch']]
        for key in LOSS_KEYS:
            require(math.isclose(row['loss_means'][key], statistics.mean(b[key] for b in epoch_batches), rel_tol=1e-6, abs_tol=1e-6), 'Loss mean')
    require(summary['config'] == config and summary['arm'] == arm and summary['final'] == history[-1], 'Summary identity')
    require(summary['complete_epochs'] == epochs and summary['warm_epochs_from_baseline'] == [1,2,3,4], 'Budget evidence')
    require(summary['actual_new_optimizer_steps'] == 924 and summary['complete_budget'] is True, 'Update budget')
    require(summary['target_label_checkpoint_selection'] is False, 'Label-selected checkpoint')
    return dict(arm=arm, warm_epochs_from_baseline=[1,2,3,4], adaptation_epochs_verified=66,
                new_optimizer_updates_verified=924, final_metrics=history[-1]['metrics'],
                independent_checkpoint_evaluation_verified=False,
                mixture_converged_all=all(row['mixture_converged'] for row in history))
