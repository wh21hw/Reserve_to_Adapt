"""Identical entry for both arms; change only post-warmup CE coefficient."""
import os
from pathlib import Path

original = Path('/content/train_legacy_task_entry.py')
entry = original.read_text()
anchor = "compiled = compile(source, str(root/'main.py'), 'exec')"
coefficient = os.environ['RTA_UNKNOWN_CE_WEIGHT']
if coefficient not in ('0', '1') or entry.count(anchor) != 1:
    raise ValueError('Unexpected ablation interface')
old = 'loss = ce + 0.01 * virtual_ce + 0.3 * adv_loss + 1 * entropy + 1 * ce_ep'
new = 'loss = ce + 0.01 * virtual_ce + 0.3 * adv_loss + 1 * entropy + ' + coefficient + ' * ce_ep'
patch = 'replace_once(' + repr(old) + ', ' + repr(new) + ')\n'
patch += "print('UNKNOWN_CE_WEIGHT', " + coefficient + ", 'warmup and head forward unchanged', flush=True)\n"
entry = entry.replace(anchor, patch + anchor)
exec(compile(entry, str(original), 'exec'), dict(__name__='__main__', __file__=str(original)))
