"""Synthetic full pair tests; no actual checkpoint evaluation performed."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matched_full_pair_audit import validate_full_pair
from test_matched_full_audit import full_fixture

def inputs():
    off,on=full_fixture('structure_off'),full_fixture('structure_on')
    independent=[]
    for arm in (off,on):
        arm[3]['checkpoint_sha256']=arm[0]['arm']+'_synthetic_only'
        independent.append(dict(arm=arm[0]['arm'],independent_checkpoint_evaluation_verified=True,
                                complete_budget=True,final_epoch_verified=70,collector_optimizer_steps=0,
                                independent_final_metrics=arm[3]['final']['metrics'],
                                checkpoint_sha256=arm[3]['checkpoint_sha256']))
    return off,on,*independent

class Tests(unittest.TestCase):
    def test_valid(self):
        report=validate_full_pair(*inputs())
        self.assertTrue(report['checkpoint_evaluation_verified_both'])
        self.assertFalse(report['seed_matrix_complete'])
        self.assertEqual(report['final_delta_on_minus_off_pp']['HOS'],0)
    def test_reject(self):
        for case in ('config','order','evaluation','metrics','checkpoint','collector'):
            off,on,io,inn=copy.deepcopy(inputs())
            if case=='config': on[0]['input_sha256']['warm']='different'
            elif case=='order': on[1][30]['batch_order_sha256']='a'*64
            elif case=='evaluation': inn['independent_checkpoint_evaluation_verified']=False
            elif case=='metrics': inn['independent_final_metrics']={'HOS':.1}
            elif case=='checkpoint': inn['checkpoint_sha256']='different'
            else: inn['collector_optimizer_steps']=1
            with self.subTest(case=case),self.assertRaises(ValueError):
                validate_full_pair(off,on,io,inn)

if __name__=='__main__': unittest.main()
