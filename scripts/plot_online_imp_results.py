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
parser.add_argument('--arms',nargs=2,default=['self_label','structure_label'])
args=parser.parse_args()
sys.path.insert(0,args.style_root)
from style_presets import rcparams
rcparams()
summary=json.loads(Path(args.summary).read_text())
output=Path(args.output)
output.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
control,candidate=args.arms
colors={control:'#0072B2',candidate:'#D55E00'}
names={control:'RTA self-label' if control=='self_label' else 'Confidence calibration',
       candidate:'Current IMP label' if candidate=='structure_label' else 'Structure-supported calibration'}
if control=='bottleneck':
    names={control:'IMP: bottleneck256',candidate:'IMP: encoder2048'}
if control=='screened':
    names={control:'Screened IMP labels',candidate:'All candidate IMP labels'}
if control=='rta_known':
    names={control:'Original RTA known weights',candidate:'IMP veto of candidate known weights'}
if control=='entropy_veto':
    names={control:'Entropy-only veto',candidate:'Alignment-only veto'}
with zipfile.ZipFile(args.archive) as bundle:
    for arm in colors:
        rows=summary[arm].get('history') or [json.loads(line) for line in bundle.read(arm+'/office31-a2w_seed3/history.jsonl').decode().splitlines()]
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
factor='Only unknown labels differ' if control=='self_label' else 'Common warm state; only IMP feature layer differs' if control=='bottleneck' else 'Common warm state; only calibration support differs'
if control=='screened':factor='Common warm state; only structure-label coverage differs'
if control=='rta_known':factor='Common warm state; only target-known eligibility differs'
if control=='entropy_veto':factor='Common warm state; known entropy vs alignment veto'
fig.suptitle('Office31 A→W | '+factor,fontsize=17)
fig.savefig(output/'epoch-trajectories.png',dpi=180)
plt.close(fig)
scores=[100*summary[a]['final']['HOS'] for a in colors]
guard=all(summary['final_delta_pp'][key]>=-1 for key in ('OS_star','unknown'))
kept=guard and summary['final_delta_pp']['HOS']>=1
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
ax.scatter([0],[scores[0]],color=colors[control],s=100,label='Shared-framework control')
ax.scatter([1],[scores[1]],facecolors=colors[candidate] if kept else 'none',
    edgecolors=colors[candidate],linewidths=2,s=100,label='Candidate: '+('kept' if kept else 'not promoted'))
ax.plot([0,1],np.maximum.accumulate(scores),color='#444444',ls=':',label='Best observed (not significance)')
ax.axhline(scores[0]+1,color='#666666',ls='--',label='Exploratory +1pp screen')
ax.set(xticks=[0,1],xticklabels=[names[control],names[candidate]],ylabel='Final HOS (%)',title='One declared factor comparison')
ax.legend(fontsize=10)
fig.savefig(output/'progress.png',dpi=180)
plt.close(fig)
print('ONLINE_RESULT_PLOTS_SAVED',output)
