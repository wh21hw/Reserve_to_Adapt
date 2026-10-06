"""Plot one frozen proxy and its predeclared source guard, no model scoring."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--summary',required=True)
p.add_argument('--output',required=True)
p.add_argument('--style-root',required=True)
a=p.parse_args()
sys.path.insert(0,a.style_root)
from style_presets import rcparams
rcparams()
r=json.loads(Path(a.summary).read_text())
fig,ax=plt.subplots(figsize=(8,4.5))
ax.plot([0,1],[0,r['metric_new_rows_used']],color='#0072B2',label='New slots used (target)')
ax.scatter([0],[0],color='#0072B2')
reason = 'Rejected: source guard failed' if not r['source_guard_at_most_1pp'] else 'Rejected: no functional activation'
ax.scatter([1],[r['metric_new_rows_used']],facecolors='none',edgecolors='#D55E00',s=65,label=reason)
ax.plot([0,1],[0,0],ls=':',color='#999999',label='Best feasible metric')
ax.axhline(1,ls='--',color='#009E73',label='Functional activation threshold')
ax.set_ylim(-.4,4.5)
ax.set_xticks([0,1]);ax.set_xticklabels(['Original frozen head','Prototype directions'],fontsize=11)
ax.set_ylabel('New slots used\n(full-head winners)',fontsize=12)
ax.set_title('Frozen-state proxy: activation is not safe discovery',fontsize=13)
ax.text(.02,.95,'Source rejection increase: {:.2f}pp (guard ≤ 1pp)'.format(r['source_rejection_increase_pp']),transform=ax.transAxes,va='top',fontsize=11)
ax.legend(fontsize=9,loc='lower right')
fig.tight_layout()
fig.savefig(a.output)
plt.close(fig)
