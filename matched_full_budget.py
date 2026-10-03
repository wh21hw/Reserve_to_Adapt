"""Pinned, inspectable extension of the verified pilot to the full A2W budget.

Seed1 only in this entry: other seeds need their own label-free warm proposals
and strict handoff fingerprints before using this same policy.
"""
import hashlib

PILOT_SHA256 = 'bf5babdecdcd0e07c8ea2e6f553a07442009101d08565566a5bbdf208f02d75b'

def build(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PILOT_SHA256
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, f'Expected exactly one anchor: {old}'
        source = source.replace(old, new)
    replace('"""ONE predeclared matched warm4->6 arm. No label-selected capacity or tuning."""',
            '"""ONE fixed seed1 warm4->70 arm; no target-label training or selection."""')
    replace("    args = parser.parse_args()", "    parser.add_argument('--verify-init', action='store_true')\n    args = parser.parse_args()")
    replace("Path('/content/imp-runs/matched-structure-pilot-v1')", "Path('/content/imp-runs/matched-structure-full-a2w-seed1-v1')")
    replace('    output.mkdir(parents=True)',
            "    if args.verify_init:\n        print('FULL_INIT_VERIFIED', args.arm, [w.global_step for w in wrappers], discriminator.grl.global_step, flush=True)\n        return\n    output.mkdir(parents=True)")
    replace('final_epoch=6, seed=1,', 'final_epoch=70, seed=1,')
    replace("scope='Two-epoch stability pilot, not complete-budget performance evidence'",
            "scope='Fixed warm4 plus 66 adaptation epochs; published-code budget, hierarchical method change',\n                  base_pilot_sha256=PILOT_SHA256, generated_training_sha256=GENERATED_SHA256,\n                  unknown_selection_policy='0-based epoch<=10: probability>.8 known, nonknown top16; later: min/max mixture groups',\n                  mixture_components_policy='4 through 0-based epoch30, then 2',\n                  evaluation_selection='fixed_final; all epoch metrics diagnostic only',\n                  data_list_sha256={str(p): sha(p) for p in (Path('/content/amazon_0-9_train_all.txt'), Path('/content/webcam_0-9_20-30_test.txt'))}")
    replace('    for epoch in (4, 5):', '    for epoch in range(4, 70):')
    replace("            weight = torch.as_tensor(prob > .8, device='cuda', dtype=sf.dtype)\n            chosen = torch.as_tensor(np.flatnonzero(groups != known_group), device='cuda')\n            if len(chosen) > 16:\n                chosen = score.argsort()[-16:]",
            "            if epoch <= 10:\n                weight = torch.as_tensor(prob > .8, device='cuda', dtype=sf.dtype)\n                chosen = torch.as_tensor(np.flatnonzero(groups != known_group), device='cuda')\n                if len(chosen) > 16:\n                    chosen = score.argsort()[-16:]\n            else:\n                weight = torch.as_tensor(groups == known_group, device='cuda', dtype=sf.dtype)\n                unknown_group_id = int(mixture.means_.argmax())\n                chosen = torch.as_tensor(np.flatnonzero(groups == unknown_group_id), device='cuda')")
    replace('BayesianGaussianMixture(n_components=4, max_iter=800).fit(values',
            'BayesianGaussianMixture(n_components=4 if epoch <= 30 else 2, max_iter=800).fit(values')
    replace("assert [r['epoch'] for r in history] == [5, 6]", "assert [r['epoch'] for r in history] == list(range(5, 71))")
    replace('assert wrappers[0].global_step == 84 and discriminator.grl.global_step == 168',
            'assert all(w.global_step == 980 for w in wrappers) and discriminator.grl.global_step == 1960')
    replace('complete_epochs=[5, 6],', 'complete_epochs=list(range(5, 71)), warm_epochs_from_baseline=[1, 2, 3, 4],')
    replace('actual_new_optimizer_steps=28,', 'actual_new_optimizer_steps=924,')
    replace('exact_resume=False, complete_budget=False)', 'exact_resume=False, complete_budget=True)')
    return source
