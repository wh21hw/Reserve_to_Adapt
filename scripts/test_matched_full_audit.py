"""Synthetic full-log audit tests, never train or read target data."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matched_full_audit import validate_full_logs
from test_matched_pilot_audit import fixture

def full_fixture(arm):
    config, history, batches, summary = fixture(arm)
    config.update(final_epoch=70, evaluation_selection='fixed_final; all epoch metrics diagnostic only',
                  unknown_selection_policy='0-based epoch<=10: probability>.8 known, nonknown top16; later: min/max mixture groups',
                  mixture_components_policy='4 through 0-based epoch30, then 2')
    template_row, template_epoch = copy.deepcopy(batches[0]), copy.deepcopy(history[0])
    batches, history = [], []
    for epoch in range(5,71):
        for batch in range(1,15):
            row = dict(template_row, epoch=epoch, batch=batch, unknown_selected=16 if epoch<=11 else 48)
            batches.append(row)
        row = dict(template_epoch, epoch=epoch, optimizer_steps=[epoch*14]*3, grl_steps=epoch*28,
                   batch_order_sha256=hashlib.sha256(str(epoch).encode()).hexdigest())
        history.append(row)
    summary.update(config=config, final=history[-1], complete_epochs=list(range(5,71)),
                   warm_epochs_from_baseline=[1,2,3,4], actual_new_optimizer_steps=924, complete_budget=True)
    return config, history, batches, summary

class Tests(unittest.TestCase):
    def test_both_valid(self):
        for arm in ('structure_off','structure_on'):
            self.assertEqual(validate_full_logs(*full_fixture(arm))['new_optimizer_updates_verified'],924)
    def test_reject_corruption(self):
        for kind in ('missing_epoch','missing_batch','nan','formula','counter','early_selection','budget','target_selection','policy'):
            c,h,b,s = full_fixture('structure_on')
            if kind=='missing_epoch': h.pop(30)
            elif kind=='missing_batch': b.pop(30)
            elif kind=='nan': b[0]['total']=float('nan')
            elif kind=='formula': b[0]['total']+=.1
            elif kind=='counter': h[-1]['optimizer_steps']=[979]*3
            elif kind=='early_selection': b[0]['unknown_selected']=17
            elif kind=='budget': s['actual_new_optimizer_steps']=28
            elif kind=='target_selection': s['target_label_checkpoint_selection']=True
            else: c['mixture_components_policy']='always4'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_full_logs(c,h,b,s)

if __name__=='__main__': unittest.main()
