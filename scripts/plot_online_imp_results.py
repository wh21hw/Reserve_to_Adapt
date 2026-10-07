"""Plot completed recordedmetrics only; never reload or reevaluate models."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--archive',required=True)
parser.add_argument('--summary',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--style-root',required=True)
args=parser.parse_args()
sys.path.insert(0,args.style_root)
from style_presets import rcparams
rcparams()
summary=json.loads(Path(args.summary).read_text())
output=Path(args.output)
output.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
colors={'self_label':'#0072B2','structure_label':'#D55E00'}
names={'self_label':'RTA self-label','structure_label':'Current IMP label'}
with zipfile.ZipFile(args.archive) as bundle:
    for arm in colors:
        rows=[json.loads(line) for line in bundle.read(arm+'/office31-a2w_seed3/history.jsonl').decode().splitlines()]
        x=[r['epoch'] for r in rows]
        for ax,key,title in zip(axes[0],('HOS','OS_star','unknown'),('HOS','Known accuracy (OS*)','Unknown recall (UNK)')):
            ax.plot(x,[100*r[key] for r in rows],marker='o',ms=4,color=colors[arm],label=names[arm])
            ax.set(title=title,ylabel='Percent',xlabel='Epoch',ylim=(0,100))
        for ax,key,title in zip(axes[1,:2],('K','V'),('Unknown output capacity K','Screened virtual directions V')):
            ax.step(x,[r[key] for r in rows],where='mid',color=colors[arm],label=names[arm])
            ax.set(title=title,ylabel='Count',xlabel='Epoch')
        support=summary[arm]['support']
        active=[int((np.asarray(r['pseudo_slot_counts'])>0).sum()) for r in support]
        axes[1,2].plot(x,active,marker='o',ms=4,color=colors[arm],label=names[arm])
        axes[1,2].set(title='Slots receiving pseudo-labels',ylabel='Count (not semantic classes)',xlabel='Epoch')
for ax in axes.flat:
    ax.axvspan(.5,4.5,color='#999999',alpha=.12,zorder=0)
    ax.set_xlim(.5,10.5)
axes[0,0].legend(fontsize=10,loc='lower right')
fig.suptitle('Office31 A→W | Shared online IMP backbone; only unknown labels differ',fontsize=17)
fig.savefig(output/'epoch-trajectories.png',dpi=180)
plt.close(fig)
scores=[100*summary[a]['final']['HOS'] for a in colors]
guard=all(summary['final_delta_pp'][key]>=-1 for key in ('OS_star','unknown'))
kept=guard and summary['final_delta_pp']['HOS']>=1
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
ax.scatter([0],[scores[0]],color=colors['self_label'],s=100,label='Shared-framework control')
ax.scatter([1],[scores[1]],facecolors=colors['structure_label'] if kept else 'none',
    edgecolors=colors['structure_label'],linewidths=2,s=100,label='Structure labels: '+('kept' if kept else 'not promoted'))
ax.plot([0,1],np.maximum.accumulate(scores),color='#444444',ls=':',label='Best observed (not significance)')
ax.axhline(scores[0]+1,color='#666666',ls='--',label='Exploratory +1pp screen')
ax.set(xticks=[0,1],xticklabels=['Self-label','IMP label'],ylabel='Final HOS (%)',title='One declared label-source comparison')
ax.legend(fontsize=10)
fig.savefig(output/'progress.png',dpi=180)
plt.close(fig)
print('ONLINE_RESULT_PLOTS_SAVED',output)
