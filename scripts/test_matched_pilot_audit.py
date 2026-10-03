"""Synthetic log cases; no model, target data or optimizer execution."""
import copy
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matched_pilot_audit import LOSS_KEYS, validate_pair

def fixture(arm):
    coefficient = 0. if arm == 'structure_off' else .1
    config = dict(arm=arm, coefficient=coefficient, warm_epoch=4, final_epoch=6, seed=1,
                  batch_size=64, support_min=5, target_labels_used_for_training=False,
                  capacity=2, selected_indices=[3, 1], prior=[-math.log(2)] * 2,
                  variance=.001, input_sha256={'warm': 'a' * 64}, worker_sha256='b' * 64)
    batches = [dict(epoch=e, batch=b, total=1 + .01 + .3 + 1 + 1 + coefficient,
                    structure=1., source_ce=1., virtual=1., entropy=1., unknown_ce=1., adversarial=1., unknown_selected=16)
               for e in (5, 6) for b in range(1, 15)]
    history = [dict(epoch=e, arm=arm, coefficient=coefficient, batch_count=14,
                    optimizer_steps=[e * 14] * 3, grl_steps=e * 28,
                    batch_order_sha256=str(e) * 64, metrics=dict(OS_star=.8, UNK=.8, HOS=.8),
                    mixture_converged=True, loss_means={k: batches[(e - 5) * 14][k] for k in LOSS_KEYS})
               for e in (5, 6)]
    summary = dict(arm=arm, config=config, final=history[-1], complete_epochs=[5, 6],
                   actual_new_optimizer_steps=28, target_label_checkpoint_selection=False, complete_budget=False)
    return config, history, batches, summary

class AuditTests(unittest.TestCase):
    def test_valid_pair(self):
        report = validate_pair(fixture('structure_off'), fixture('structure_on'))
        self.assertTrue(report['sample_batch_order_equal'])
        self.assertFalse(report['augmented_image_equality_verified'])
        self.assertTrue(report['checkpoint_evaluation_still_required'])
    def test_reject_bad_batches(self):
        for kind in ('missing', 'nan', 'formula'):
            on = fixture('structure_on')
            if kind == 'missing':
                on[2].pop()
            elif kind == 'nan':
                on[2][0]['structure'] = float('nan')
            else:
                on[2][0]['total'] += .1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_pair(fixture('structure_off'), on)
    def test_reject_mismatched_pair(self):
        for kind in ('order', 'input', 'counter', 'target_selection'):
            on = fixture('structure_on')
            if kind == 'order':
                on[1][0]['batch_order_sha256'] = 'a' * 64
            elif kind == 'input':
                on[0]['input_sha256']['warm'] = 'c' * 64
            elif kind == 'counter':
                on[1][-1]['optimizer_steps'] = [83] * 3
            else:
                on[3]['target_label_checkpoint_selection'] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_pair(fixture('structure_off'), on)

if __name__ == '__main__':
    unittest.main()
