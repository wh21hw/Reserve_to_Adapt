"""Plot the one declared pair from saved metrics, without model evaluation."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary',required=True)
    parser.add_argument('--archive',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--style-root',required=True)
    args = parser.parse_args()
    sys.path.insert(0,args.style_root)
    from style_presets import rcparams
    rcparams()
    report = json.loads(Path(args.summary).read_text())
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    base = 100*report['arms']['fixed8']['final']['HOS']
    candidate = 100*report['arms']['refresh']['final']['HOS']
    screen = report['manual_support_screen']
    supported = all(screen[key] for key in ('final_HOS_gain_at_least_1pp',
        'final_OS_star_loss_at_most_1pp','final_UNK_loss_at_most_1pp'))
    fig,ax = plt.subplots(figsize=(6.2,4))
    ax.plot([0,1],[base,candidate],color='#0072B2',label='Final HOS: one declared pair')
    ax.scatter([0],[base],color='#0072B2',zorder=3)
    ax.scatter([1],[candidate],facecolors='#009E73' if supported else 'none',
               edgecolors='#009E73' if supported else '#D55E00',s=65,zorder=3)
    ax.plot([0,1],np.maximum.accumulate([base,candidate]),color='#999999',ls=':',label='Observed envelope')
    ax.axhline(base+1,color='#D55E00',ls='--',label='Exploratory +1pp screen')
    ax.set_xticks([0,1]);ax.set_xticklabels(['0: fixed K8','1: stage10 refresh'])
    ax.set_ylabel('Final HOS (%)');ax.set_title('A→W capacity refresh, seed3')
    ax.set_ylim(min(base,candidate)-3,max(base+1,candidate)+4)
    ax.legend(fontsize=10)
    fig.tight_layout();fig.savefig(output/'progress.png');plt.close(fig)
    with zipfile.ZipFile(args.archive) as bundle:
        histories = {}
        for arm in ('fixed8','refresh'):
            name=arm+'/office31-a2w_seed3/history.jsonl'
            histories[arm]=[json.loads(line) for line in bundle.read(name).decode().splitlines()]
    fig,axes = plt.subplots(1,3,figsize=(12,3.5),sharex=True)
    for ax,key,title in zip(axes,('OS_star','unknown','HOS'),('OS*','UNK','HOS')):
        for arm,color,label in (('fixed8','#0072B2','Fixed K8'),
                                ('refresh','#D55E00','K8→K4 after epoch10')):
            rows=histories[arm]
            ax.plot([r['epoch'] for r in rows],[100*r[key] for r in rows],color=color,label=label)
        ax.axvline(10.5,color='#888888',ls='--',lw=1)
        ax.set_title(title);ax.set_xlabel('Completed RTA epoch');ax.set_xlim(1,20)
    axes[0].set_ylabel('Macro accuracy (%)')
    axes[-1].legend(fontsize=9,loc='lower right')
    fig.tight_layout();fig.savefig(output/'epoch-trajectories.png');plt.close(fig)
    print('A2W_CAPACITY_RESULTS_PLOTTED',str(output))


if __name__ == '__main__':
    main()
