"""Display recorded working points; no model evaluation or guard-based rejection."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--summary',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--style-root',required=True)
parser.add_argument('--arms',nargs='+',default=['entropy1','entropy0p5','entropy0'])
args=parser.parse_args()
sys.path.insert(0,args.style_root)
from style_presets import rcparams
rcparams()
summary=json.loads(Path(args.summary).read_text())
arms=args.arms
label_map={'entropy1':'Candidate entropy 100%','entropy0p5':'Candidate entropy 50%','entropy0':'Candidate entropy 0%'}
color_map={'entropy1':'#0072B2','entropy0p5':'#009E73','entropy0':'#D55E00'}
labels=[label_map[a] for a in arms]
colors=[color_map[a] for a in arms]
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for arm,label,color in zip(arms,labels,colors):
    history=summary[arm]['history']
    for ax,key,title in zip(axes.flat,('HOS','OS_star','unknown'),('HOS','Known accuracy (OS*)','Unknown recall (UNK)')):
        ax.plot([r['epoch'] for r in history],[100*r[key] for r in history],
            color=color,marker='o',ms=3,label=label)
        ax.set(title=title,xlabel='Epoch',ylabel='Percent',ylim=(0,100))
        ax.axvspan(.5,4.5,color='#999999',alpha=.12)
    final=summary[arm]['final']
    axes[1,1].scatter(100*final['OS_star'],100*final['unknown'],s=90,color=color,label=label)
    axes[1,1].annotate('HOS %.2f'%(100*final['HOS']),
        (100*final['OS_star'],100*final['unknown']),xytext=(8,-18) if arm=='entropy0' else (-6,10),
        textcoords='offset points',ha='left' if arm=='entropy0' else 'right')
axes[0,0].legend(fontsize=9)
axes[1,1].set(title='Final OS*/UNK tradeoff',xlabel='Known accuracy (%)',ylabel='Unknown recall (%)')
axes[1,1].legend(fontsize=9,loc='best')
fig.suptitle('Office31 A→W | Only candidate-known entropy strength differs; reliable union fixed')
output=Path(args.output)
output.mkdir(parents=True,exist_ok=True)
fig.savefig(output/'tradeoff.png',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
scores=[100*summary[a]['final']['HOS'] for a in arms]
ax.bar(labels,scores,color=colors)
ax.axhline(scores[0]+1,ls='--',color='#555555',label='Exploratory +1pp HOS line; not a known guard')
ax.set(ylabel='Final HOS (%)',title='Declared working points; manual tradeoff review',ylim=(0,100))
ax.legend(fontsize=9)
fig.savefig(output/'progress.png',dpi=180)
plt.close(fig)
print('ENTROPY_TRADEOFF_PLOTS_SAVED',output)
