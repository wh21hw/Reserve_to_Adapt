"""Require two complete, independently evaluated arms before reporting delta."""
from matched_full_audit import validate_full_logs
from matched_pilot_audit import require

def validate_full_pair(off, on, off_independent, on_independent):
    require(off[0]['arm']=='structure_off' and on[0]['arm']=='structure_on','Arm order')
    reports=[validate_full_logs(*arm) for arm in (off,on)]
    shared=lambda config:{key:value for key,value in config.items() if key not in ('arm','coefficient')}
    require(shared(off[0])==shared(on[0]),'Unmatched shared configuration/input/code')
    require([r['batch_order_sha256'] for r in off[1]]==[r['batch_order_sha256'] for r in on[1]],'Unmatched sample order')
    for values,independent,report in zip((off,on),(off_independent,on_independent),reports):
        require(independent['arm']==values[0]['arm'],'Independent arm mismatch')
        require(independent['independent_checkpoint_evaluation_verified'] is True,'Checkpoint evaluation required')
        require(independent['complete_budget'] is True and independent['final_epoch_verified']==70,'Independent budget')
        require(independent['collector_optimizer_steps']==0,'Collector trained')
        require(independent['independent_final_metrics']==values[3]['final']['metrics'],'Independent metrics mismatch')
        require(independent['checkpoint_sha256']==values[3]['checkpoint_sha256'],'Checkpoint identity mismatch')
        report['independent_checkpoint_evaluation_verified']=True
    return dict(arms=reports,shared_config_equal=True,sample_batch_order_equal=True,
                augmented_image_equality_verified=False,complete_budget=True,seed=1,
                checkpoint_evaluation_verified_both=True,seed_matrix_complete=False,
                final_delta_on_minus_off_pp={key:100*(on[1][-1]['metrics'][key]-off[1][-1]['metrics'][key])
                                            for key in ('OS_star','UNK','HOS')},
                scope='One seed, fixed final; not a three-seed significance or semantic class-count claim')
