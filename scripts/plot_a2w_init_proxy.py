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
fig,ax=plt.subplots(figsize=(7,4))
ax.plot([0,1],[0,r['metric_new_rows_used']],color='#0072B2',label='New slots used (target)')
ax.scatter([0],[0],color='#0072B2')
ax.scatter([1],[r['metric_new_rows_used']],facecolors='none',edgecolors='#D55E00',s=65,label='Rejected: source guard failed')
ax.plot([0,1],[0,0],ls=':',color='#999999',label='Best feasible metric')
ax.axhline(1,ls='--',color='#009E73',label='Functional activation threshold')
ax.set_ylim(-.4,4.5)
ax.set_xticks([0,1]);ax.set_xticklabels(['Original frozen head','Prototype directions'])
ax.set_ylabel('New slots with full-head winners')
ax.set_title('Frozen-state proxy: activation is not safe discovery')
ax.text(.02,.95,'Source rejection: 0 → 10.13% (guard ≤ 1pp)',transform=ax.transAxes,va='top',fontsize=11)
ax.legend(fontsize=9,loc='lower right')
fig.tight_layout()
fig.savefig(a.output)
plt.close(fig)
